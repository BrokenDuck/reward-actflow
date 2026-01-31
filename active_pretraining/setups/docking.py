import os
import random
from abc import abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import yaml
from diffdock.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from diffdock.utils.download import download_and_extract
from diffdock.utils.geometry import matrix_to_axis_angle
from diffdock.utils.inference_utils import InferenceDataset, set_nones
from diffdock.utils.sampling import modify_conformer_batch
from diffdock.utils.torsion import modify_conformer_torsion_angles
from diffdock.utils.utils import get_model
from flowgym import BaseModel, Environment, FlowMixin, MemorylessNoiseSchedule, NoiseSchedule, Scheduler
from flowgym.types import BinaryOp, FlowTensor, UnaryOp
from flowgym.utils import append_dims
from matplotlib.pyplot import Figure
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import DataLoader
from torch_geometric.data.batch import Batch
from torch_geometric.data.hetero_data import HeteroData
from typing_extensions import Self

from active_pretraining.problem_setup import ProblemSetup

pose_els = {'tr', 'rot', 'tor'}

# TODO: check logic of DockResult data flow
# TODO: check feature_postprocess latent_postprocess
# TODO: check if we have right version of active exp (run.py line 237+)
# TODO: ideal data type is only pose and we run rotation at the end of every loop iter of sampling (change flowgym)


def init_graph(data: HeteroData,
               no_torsion=False,
               tr_sigma_max=1.,
               pocket_knowledge=False,
               pocket_cutoff=1.,
               no_random=False,
               choose_residue=False,
               initial_noise_std_proportion=-1.):
    """Initialize graph for DiffDock functionality."""
    complex_graph = data
    center_pocket = complex_graph['receptor'].pos.mean(dim=0)
    if pocket_knowledge:
        orig_pos = torch.from_numpy(complex_graph['ligand'].orig_pos[0]).float()
        d = torch.cdist(complex_graph['receptor'].pos, orig_pos - complex_graph.original_center)
        label = torch.any(d < pocket_cutoff, dim=1)

        if torch.any(label):
            center_pocket = complex_graph['receptor'].pos[label].mean(dim=0)
        else:
            print("No pocket residue below minimum distance ", pocket_cutoff, "taking closest at", torch.min(d))
            center_pocket = complex_graph['receptor'].pos[torch.argmin(torch.min(d, dim=1)[0])]

    if not no_torsion:
        # randomize torsion angles
        torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
        complex_graph['ligand'].pos = \
            modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                            complex_graph['ligand', 'ligand'].edge_index.T[
                                                complex_graph['ligand'].edge_mask],
                                            complex_graph['ligand'].mask_rotate, torsion_updates)
    else:
        torsion_updates = np.zeros((complex_graph['ligand'].edge_mask.sum(),))

    # randomize position
    pos: torch.Tensor = complex_graph['ligand'].pos
    molecule_center = torch.mean(pos, dim=0, keepdim=True)
    random_rotation = torch.from_numpy(R.random().as_matrix()).float()
    complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T + center_pocket

    if not no_random:  # note for now the torsion angles are still randomised
        if choose_residue:
            idx = random.randint(0, len(complex_graph['receptor'].pos)-1)
            tr_update = torch.normal(mean=complex_graph['receptor'].pos[idx:idx+1], std=0.01)
        elif initial_noise_std_proportion >= 0.0:
            std_rec = torch.sqrt(torch.mean(torch.sum(complex_graph['receptor'].pos ** 2, dim=1)))
            tr_update = torch.normal(mean=0, std=std_rec * initial_noise_std_proportion / 1.73, size=(1, 3))
        else:
            # if initial_noise_std_proportion < 0.0, we use the tr_sigma_max multiplied by -initial_noise_std_proportion
            tr_update = torch.normal(mean=0, std=-initial_noise_std_proportion * tr_sigma_max, size=(1, 3))
        complex_graph['ligand'].pos += tr_update

    else:
        tr_update = torch.zeros_like(complex_graph['receptor'].pos[0:1])

    return complex_graph, tr_update, random_rotation, torsion_updates


@dataclass(frozen=True)
class DockPose(FlowMixin): # TODO maybe just use dict-nodestorage functionality somehow
    tr: torch.Tensor
    rot: torch.Tensor
    tor: torch.Tensor


    def to(self, device: torch.device | str) -> "DockPose":
        return DockPose(self.tr.to(device), self.rot.to(device), self.tor.to(device))


    @property
    def base(self) -> bool:
        zero = torch.tensor(0.).double()
        return torch.allclose(self.tr.double(), zero) \
            and torch.allclose(self.rot.double(), zero) and torch.allclose(self.tor.double(), zero)


    def combine(self, other: Self | float | torch.Tensor, op: BinaryOp) -> "DockPose":
        if isinstance(other, DockPose):
            tr = op(self.tr, other.tr)
            rot = op(self.rot, other.rot)
            tor = op(self.tor, other.tor)
        else:
            tr = op(self.tr, other)
            rot = op(self.rot, other)
            tor = op(self.tor, other)

        return DockPose(tr, rot, tor)


    def apply(self, op: UnaryOp) -> "DockPose":
        tr = op(self.tr)
        rot = op(self.rot)
        tor = op(self.tor)

        return DockPose(tr, rot, tor)


    def __len__(self) -> int:
        return self.tr.shape[0]


    def __getitem__(self, idx: Union[int, slice]) -> "DockPose":
        return DockPose(self.tr[idx], self.rot[idx], self.tor[idx])


    @classmethod
    def collate(cls: type[Self], items: Sequence[Self]) -> "DockPose":
        if not items:
            raise ValueError("Cannot collate an empty sequence")
        tr = torch.vstack([i.tr for i in items])
        rot = torch.vstack([i.rot for i in items])
        tor = torch.vstack([i.tor for i in items])

        return DockPose(tr, rot, tor)


    def aggregate(self, reduction: str = "mean") -> torch.Tensor:
        raise NotImplementedError


class DockResult(FlowMixin): # TODO maybe change to subclass HeteroData???
    def __init__(
        self,
        complex_graph: Batch
    ):
        device = complex_graph['ligand'].pos.device
        complex_graph = complex_graph.clone().to(device)

        if 'ligand_pose' not in complex_graph.node_types:
            data_list = complex_graph.to_data_list()
            for graph in data_list:
                graph['ligand_pose'].rot = torch.zeros((1, 3)).to(device)
                graph['ligand_pose'].tr = torch.zeros((1, 3)).to(device)
                graph['ligand_pose'].tor = torch.zeros((1, 3)).to(device)
            complex_graph = Batch.from_data_list(data_list)

        self.graph = complex_graph


    def to(self, device: torch.device | str) -> "DockResult":
        return DockResult(self.graph.to(device))


    def clone(self) -> "DockResult":
        if isinstance(self.graph, Batch):
            data_list = [g.clone() for g in self.graph.to_data_list()]
            return DockResult(Batch.from_data_list(data_list))

        return DockResult(self.graph.clone())


    @property
    def pos(self) -> torch.Tensor:
        return self.graph['ligand'].pos


    @pos.setter
    def pos(self, val: torch.Tensor):
        self.graph['ligand'].pos = val


    @property
    def pose(self) -> "DockPose":
        lig_pose = self.graph['ligand_pose']
        return DockPose(lig_pose.tr, lig_pose.rot, lig_pose.tor)


    @pose.setter
    def pose(self, val: "DockPose"):
        pose_graph = Batch.from_data_list([HeteroData(ligand_pose={'tr': val.tr, 'rot': val.rot, 'tor': val.tor})])
        self.graph.update(pose_graph)


    def __repr__(self) -> str:
        rep  = f"""DockResult(
        {self.graph.__repr__()}
        )
        """
        return rep


    def __len__(self) -> int:
        return int(self.graph.num_graphs)


    def __getitem__(self, idx: Union[int, slice]) -> Self:
        if isinstance(idx, int):
            # Faster to use slice_batch if we only want one item
            if idx < 0:
                idx += len(self)

            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for batch size {len(self)}")
            return type(self)(Batch.from_data_list(self.graph.index_select(slice(idx, idx + 1))))

        return type(self)(Batch.from_data_list(self.graph.index_select(idx)))


    @classmethod
    def collate(cls, items: Sequence[Self]) -> Self:
        if not items:
            raise ValueError("Cannot collate an empty sequence")

        data_list = []
        for item in items:
            data_list = data_list + item.graph.to_data_list()

        graph_batch = Batch.from_data_list(data_list)
        result = cls(graph_batch)
        return result


    def apply(self, op: UnaryOp) -> "DockResult":
        res = self.clone()

        tr = op(res.pose.tr)
        rot = op(res.pose.rot)
        tor = op(res.pose.tor)
        res.pose = DockPose(tr, rot, tor)

        return res


    def apply_pose(self, pose: DockPose) -> torch.Tensor:
        tr_perturb, rot_perturb, tor_perturb = pose.tr.float(), pose.rot.float(), pose.tor.float()
        mask_rotate = torch.from_numpy(self.graph['ligand'].mask_rotate[0]).to(self.device)

        return modify_conformer_batch(self.graph['ligand'].pos, self.graph,
                                      tr_perturb, rot_perturb, tor_perturb, mask_rotate)


    def combine(self, other: Union[Self, float, torch.Tensor], op: BinaryOp) -> "DockResult":
        res = self.clone()
        if isinstance(other, DockResult):
            res.pose = self.pose.combine(other.pose, op)

            if self.pose.base and not other.pose.base: # other is transformation
                res.pos = self.apply_pose(res.pose)

            elif not self.pose.base and other.pose.base: # self is transformation
                res.pos = other.apply_pose(res.pose)

        else:
            res.pose = self.pose.combine(other, op)

        return res


    def aggregate(self, reduction: str = "mean") -> torch.Tensor:
        raise NotImplementedError


class DiffDockBaseModel(BaseModel[DockResult]):
    def __init__(self, scheduler_params: Namespace):
        self._scheduler = DiffDockScheduler(scheduler_params)
        # TODO fetch underlying model etc either AAModel or CGModel or old variations


    @property
    def scheduler(self) -> "DiffDockScheduler":
        return self._scheduler


    def sample_p0(self, n: int, **kwargs: Any) -> tuple[DockResult, dict[str, Any]]:

        results: Sequence[DockResult] = []
        for _ in range(n):
            complex_graph, tr_update, random_rotation, torsion_updates = init_graph(**kwargs)
            res = DockResult(Batch.from_data_list([complex_graph]))
            no_torsion, no_random, add_pose = kwargs['no_torsion'], kwargs['no_random'], kwargs['add_pose']

            if not no_torsion and not no_random and add_pose:
                rot_update = matrix_to_axis_angle(random_rotation).reshape((1, -1))
                tor_update = torch.from_numpy(torsion_updates).reshape((1, -1))
                res.pose = DockPose(tr_update, rot_update, tor_update)


            results.append(res)

        return DockResult.collate(results), kwargs


    def forward(self, x: DockResult, t: torch.Tensor, **kwargs: Any) -> DockResult:
        raise NotImplementedError


    def preprocess(self, x: DockResult, **kwargs: Any) -> tuple[DockResult, dict[str, Any]]:
        raise NotImplementedError

    def postprocess(self, x: DockResult) -> DockResult:
        raise NotImplementedError


class LogLinearScheduler(Scheduler[FlowTensor]):
    def __init__(self, sigma_max: torch.Tensor, sigma_min: torch.Tensor):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min


    def alpha(self, x: FlowTensor, t: torch.Tensor) -> FlowTensor:
        t = t.double().to(x.device)
        max_over_min = self.sigma_max / self.sigma_min
        min_over_max = self.sigma_min / self.sigma_max
        logr = 2 * torch.log(max_over_min).to(x.device)
        diff = (min_over_max).to(x.device)** (2 * t) - (min_over_max).to(x.device) **2
        result = torch.exp(-logr * self.sigma_max.to(x.device)**2 / (2 * logr) * diff)
        return FlowTensor(append_dims(result, x.data.ndim)).to(x.device)


    def alpha_dot(self, x: FlowTensor, t: torch.Tensor) -> FlowTensor:
        t = t.double().to(x.device)
        logr = 2 * torch.log(self.sigma_max / self.sigma_min).to(x.device)
        mul = 0.5 * logr * self.sigma_max.to(x.device)**2 * (self.sigma_min / self.sigma_max).to(x.device)**(2 * t)
        alpha = self.alpha(x, t)
        result = alpha * mul.reshape(alpha.data.shape)
        return result.to(x.device)


class DiffDockScheduler(Scheduler[DockResult]):
    def __init__(self, params: Namespace):
        self.schedulers = {
            'tr': LogLinearScheduler(torch.tensor(params.tr_sigma_max), torch.tensor(params.tr_sigma_min)),
            'rot': LogLinearScheduler(torch.tensor(params.rot_sigma_max), torch.tensor(params.rot_sigma_min)),
            'tor': LogLinearScheduler(torch.tensor(params.tor_sigma_max), torch.tensor(params.tor_sigma_min))
        }


    @property
    def noise_schedule(self) -> NoiseSchedule[DockResult]:
        """Get the current noise schedule."""
        if not hasattr(self, "_noise_schedule"):
            self._noise_schedule: NoiseSchedule[DockResult] = MemorylessNoiseSchedule(self)

        return self._noise_schedule

    @noise_schedule.setter
    def noise_schedule(self, schedule: NoiseSchedule[DockResult]) -> None:
        """Set the noise schedule. Defaults to the memoryless noise schedule."""
        self._noise_schedule = schedule


    def alpha(self, x: DockResult, t: torch.Tensor) -> DockResult:
        res = x.clone()
        pose = res.pose
        a_tr = self.schedulers['tr'].alpha(FlowTensor(pose.tr), t)
        a_rot = self.schedulers['rot'].alpha(FlowTensor(pose.rot), t)
        a_tor = self.schedulers['tor'].alpha(FlowTensor(pose.tor), t)
        res.pose = DockPose(a_tr.data, a_rot.data, a_tor.data)
        # TODO apply pose?? see if right to left or left to right, might not matter anyway

        return res


    def alpha_dot(self, x: DockResult, t: torch.Tensor) -> DockResult:
        res = x.clone()
        pose = res.pose
        a_tr = self.schedulers['tr'].alpha_dot(FlowTensor(pose.tr), t)
        a_rot = self.schedulers['rot'].alpha_dot(FlowTensor(pose.rot), t)
        a_tor = self.schedulers['tor'].alpha_dot(FlowTensor(pose.tor), t)
        res.pose = DockPose(a_tr.data, a_rot.data, a_tor.data)
        # TODO apply pose?? see if right to left or left to right, might not matter anyway

        return res

    def model_input(self, t: torch.Tensor) -> torch.Tensor:
        """Input to the model at time t.

        Defaults to t, but could be different if using a different time parameterization.
        """
        return 1. - t

REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")
REMOTE_URLS = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
               f"{REPOSITORY_URL}/releases/download/v1.1/diffdock_models.zip"]


class DockingProblemSetup(ProblemSetup[DockResult]):
    def __init__(self, dataset: str, args: dict, device: Optional[torch.device] = None):
        args_ns = Namespace(**args)
        # TODO for now dataset is a single complex
        self.dataset = dataset
        if args_ns.config:
            config_dict = yaml.load(args_ns.config, Loader=yaml.FullLoader)
            arg_dict = args_ns.__dict__
            for key, value in config_dict.items():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

        # Download models if they don't exist locally
        if not os.path.exists(args_ns.model_dir):
            remote_urls = REMOTE_URLS
            downloaded_successfully = False
            for remote_url in remote_urls:
                try:
                    files_downloaded = download_and_extract(remote_url, os.path.dirname(args_ns.model_dir))
                    if not files_downloaded:
                        continue
                    downloaded_successfully = True
                    # Once we have downloaded the models, we can break the loop
                    break
                except Exception as e:
                    pass

            if not downloaded_successfully:
                raise Exception(f"Models not found locally and failed to download them from {remote_urls}")

        os.makedirs(args_ns.out_dir, exist_ok=True)
        with open(f'{args_ns.model_dir}/model_parameters.yml') as f:
            score_model_args = Namespace(**yaml.full_load(f))
        if args_ns.confidence_model_dir is not None:
            with open(f'{args_ns.confidence_model_dir}/model_parameters.yml') as f:
                confidence_args = Namespace(**yaml.full_load(f))

        device = torch.device('cuda' if torch.cuda.is_available() and device is None else 'cpu')

        if args_ns.protein_ligand_csv is not None:
            df = pd.read_csv(args_ns.protein_ligand_csv)
            complex_name_list = set_nones(df['complex_name'].tolist())
            protein_path_list = set_nones(df['protein_path'].tolist())
            protein_sequence_list = set_nones(df['protein_sequence'].tolist())
            ligand_description_list = set_nones(df['ligand_description'].tolist())
        else:
            complex_name_list = [args_ns.complex_name if args_ns.complex_name else f"complex_0"]
            protein_path_list = [args_ns.protein_path]
            protein_sequence_list = [args_ns.protein_sequence]
            ligand_description_list = [args_ns.ligand_description]

        complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
        for name in complex_name_list:
            write_dir = f'{args_ns.out_dir}/{name}'
            os.makedirs(write_dir, exist_ok=True)

        # preprocessing of complexes into geometric graphs
        test_dataset = InferenceDataset(out_dir=args_ns.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                        ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                        lm_embeddings=True,
                                        receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                        all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                        atom_max_neighbors=score_model_args.atom_max_neighbors,
                                        knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        if args_ns.confidence_model_dir is not None and not confidence_args.use_original_model_cache:
            knn = False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph
            confidence_test_dataset = \
                InferenceDataset(out_dir=args_ns.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                lm_embeddings=True,
                                receptor_radius=confidence_args.receptor_radius, remove_hs=confidence_args.remove_hs,
                                c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                                all_atoms=confidence_args.all_atoms, atom_radius=confidence_args.atom_radius,
                                atom_max_neighbors=confidence_args.atom_max_neighbors,
                                precomputed_lm_embeddings=test_dataset.lm_embeddings,
                             knn_only_graph=knn)
        else:
            confidence_test_dataset = None

        t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

        model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args_ns.old_score_model)
        state_dict = torch.load(f'{args_ns.model_dir}/{args_ns.ckpt}', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)

        if args_ns.confidence_model_dir is not None:
            confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                        confidence_mode=True, old=args_ns.old_confidence_model)
            state_dict = torch.load(f'{args_ns.confidence_model_dir}/{args_ns.confidence_ckpt}',
                                    map_location=torch.device('cpu'))
            confidence_model.load_state_dict(state_dict, strict=True)
            confidence_model = confidence_model.to(device)
            confidence_model.eval()
        else:
            confidence_model = None
            confidence_args = None

        self.test_loader = test_loader
        self._base_model = model

    @property
    def base_model(self) -> BaseModel[DockResult]:
        raise NotImplementedError
        return self._base_model

    def validity(self, x: DockResult, kwargs: dict) -> torch.Tensor:
        raise NotImplementedError
        valids = torch.ones(len(x), dtype=torch.bool)

        x.graph.edata["ue_mask"] = x.ue_mask
        batched_graph = x.graph.cpu()

        for i, g in enumerate(dgl.unbatch(batched_graph)):
            # Check if it passes rdkit validity rules
            mol = SampledMolecule(g, self.base_model.model.atom_type_map).rdkit_mol
            mol = validate_mol(mol)

            if mol is None:
                valids[i] = False
                continue

            # Check if it has only a single connected component
            if len(Chem.GetMolFrags(mol)) != 1:
                valids[i] = False

        return valids

    @property
    def feature_layer(self) -> str:
        raise NotImplementedError
        return "model.vector_field.node_output_head.0"
