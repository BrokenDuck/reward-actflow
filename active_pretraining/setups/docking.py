import os
from abc import abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence, Union

import pandas as pd
import torch
import yaml
from diffdock.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from diffdock.utils.download import download_and_extract
from diffdock.utils.inference_utils import InferenceDataset, set_nones
from diffdock.utils.sampling import modify_conformer_batch
from diffdock.utils.utils import get_model
from flowgym import BaseModel, FlowMixin, MemorylessNoiseSchedule, NoiseSchedule, Scheduler
from flowgym.types import BinaryOp, FlowTensor, UnaryOp
from flowgym.utils import append_dims
from matplotlib.pyplot import Figure
from torch_geometric.data import DataLoader
from torch_geometric.data.batch import Batch
from torch_geometric.data.hetero_data import HeteroData
from typing_extensions import Self

from active_pretraining.problem_setup import ProblemSetup

pose_els = {'tr', 'rot', 'tor'}

# TODO: check logic of DockResult data flow
# TODO: check feature_postprocess latent_postprocess
# TODO: check if we have right version of active exp (run.py line 237+)


@dataclass(frozen=True)
class DockPose(object): # TODO maybe just use dict-nodestorage functionality somehow
    tr: torch.Tensor
    rot: torch.Tensor
    tor: torch.Tensor


    def to(self, device: torch.device) -> "DockPose":
        return DockPose(self.tr.to(device), self.rot.to(device), self.tor.to(device))


class DockResult(FlowMixin): # TODO maybe change to subclass HeteroData???
    def __init__(
        self,
        complex_graph: HeteroData | Batch
    ):
        device = complex_graph['ligand'].pos.device
        complex_graph = complex_graph.clone().to(device)

        if 'ligand_pose' not in complex_graph:
            if isinstance(complex_graph, Batch):
                data_list = complex_graph.to_data_list()
                for graph in data_list:
                    graph['ligand_pose'].rot = torch.zeros((1, 3)).to(device)
                    graph['ligand_pose'].tr = torch.zeros((1, 3)).to(device)
                    graph['ligand_pose'].tor = torch.zeros((1, 3)).to(device)
                complex_graph = Batch.from_data_list(data_list)

            else:
                complex_graph['ligand_pose'].rot = torch.zeros((1, 3)).to(device)
                complex_graph['ligand_pose'].tr = torch.zeros((1, 3)).to(device)
                complex_graph['ligand_pose'].tor = torch.zeros((1, 3)).to(device)

        self.graph = complex_graph


    def to(self, device: torch.device) -> "DockResult":
        return DockResult(self.graph.to(device))


    def clone(self) -> "DockResult":
        if isinstance(self.graph, Batch):
            data_list = [g.clone() for g in self.graph.to_data_list()]
            return DockResult(Batch.from_data_list(data_list))

        return DockResult(self.graph.clone())


    @property
    def pose(self) -> "DockPose":
        lig_pose = self.graph['ligand_pose']
        return DockPose(lig_pose.tr, lig_pose.rot, lig_pose.tor)


    @pose.setter
    def pose(self, val: "DockPose"):
        lig_pose = self.graph['ligand_pose']
        lig_pose.tr = val.tr
        lig_pose.rot = val.rot
        lig_pose.tor = val.tor


    def __repr__(self) -> str:
        rep  = f"""DockResult(
        {self.graph.__repr__()}
        )
        """
        return rep


    def __len__(self) -> int:
        return int(self.graph.num_graphs)


    def __getitem__(self, idx: Union[int, slice]) -> Self:
        if not isinstance(self.graph, Batch):
            raise ValueError("Trying to get an item of something that is not batched")

        if isinstance(idx, int):
            # Faster to use slice_batch if we only want one item
            if idx < 0:
                idx += len(self)

            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for batch size {len(self)}")
            return type(self)(self.graph.get_example(idx))

        return type(self)(Batch.from_data_list(self.graph.index_select(idx)))


    @classmethod
    def collate(cls, items: Sequence[Self]) -> Self:
        if not items:
            raise ValueError("Cannot collate an empty sequence")

        return cls(Batch.from_data_list([item.graph for item in items]))


    @classmethod
    def from_pose(cls, pose: DockPose) -> "DockResult":
        graph = HeteroData()
        graph['ligand_pose'].tor = pose.tor
        graph['ligand_pose'].rot = pose.rot
        graph['ligand_pose'].tr = pose.tr

        return cls(graph)


    def apply(self, op: UnaryOp) -> Self:
        res = self.clone()

        tr = op(res.pose.tr)
        rot = op(res.pose.rot)
        tor = op(res.pose.tor)
        res.pose = DockPose(tr, rot, tor)
        # TODO apply pose?

        return res


    def apply_pose(self, pose: DockPose) -> torch.Tensor:
        tr_perturb, rot_perturb, tor_perturb = pose.tr.float(), pose.rot.float(), pose.tor.float()
        mask_rotate = torch.from_numpy(self.graph['ligand'].mask_rotate[0]).to(self.device)

        return modify_conformer_batch(self.graph['ligand'].pos, self.graph,
                                      tr_perturb, rot_perturb, tor_perturb, mask_rotate)


    def combine(self, other: Union[Self, float, torch.Tensor], op: BinaryOp) -> Self:
        # TODO new idea: core of problem is that rotations are relative to a reference pose, 
        # so instead always "undo" current rotation and apply new one. problem: double computational costs...
        res = self.clone()
        if isinstance(other, DockResult):
            pose = other.pose
            tr = op(res.pose.tr, pose.tr)
            rot = op(res.pose.rot, pose.rot)
            tor = op(res.pose.tor, pose.tor)
            res.pose = DockPose(tr, rot, tor)

        else:
            tr = op(res.pose.tr, other)
            rot = op(res.pose.rot, other)
            tor = op(res.pose.tor, other)
            res.pose = DockPose(tr, rot, tor)

        res.graph['ligand'].pos = self.apply_pose(res.pose)
        return res


class DiffDockBaseModel(BaseModel[DockResult]):
    def __init__(self, scheduler_params):
        self._scheduler = DiffDockScheduler(scheduler_params)
        raise NotImplementedError


    @property
    def scheduler(self) -> "DiffDockScheduler":
        return self._scheduler


    @abstractmethod
    def sample_p0(self, n: int, **kwargs: Any) -> tuple[DockResult, dict[str, Any]]:
        """Sample n data points from the base distribution p0.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        samples : D
            Samples from the base distribution p0.

        kwargs : dict
            Additional keyword arguments.
        """

    @abstractmethod
    def forward(self, x: DockResult, t: torch.Tensor, **kwargs: Any) -> DockResult:
        raise NotImplementedError


    def preprocess(self, x: DockResult, **kwargs: Any) -> tuple[DockResult, dict[str, Any]]:
        """Preprocess data and keyword arguments for the base model.

        Parameters
        ----------
        x : D
            Input data to preprocess.

        **kwargs : dict
            Additional keyword arguments to preprocess.

        Returns
        -------
        output : D
            Preprocessed data.

        kwargs : dict
            Preprocessed keyword arguments.
        """
        return x, kwargs

    def postprocess(self, x: DockResult) -> DockResult:
        """Postprocess samples x_1 (e.g., decode with VAE).

        Parameters
        ----------
        x : D
            Input data to postprocess.

        Returns
        -------
        output : D
            Postprocessed output.
        """
        return x


class LogLinearScheduler(Scheduler[FlowTensor]):
    def __init__(self, sigma_max: torch.Tensor, sigma_min: torch.Tensor):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min


    def alpha(self, x: FlowTensor, t: torch.Tensor) -> FlowTensor:
        t = t.double().to(x.device)
        logr = 2 * torch.log(self.sigma_max / self.sigma_min).to(x.device)
        diff = (self.sigma_min / self.sigma_max).to(x.device)** (2 * t) - (self.sigma_min / self.sigma_max).to(x.device) **2
        result = torch.exp(-logr * self.sigma_max.to(x.device)**2 / (2 * logr) * diff)
        return FlowTensor(append_dims(result, x.data.ndim)).to(x.device)


    def alpha_dot(self, x: FlowTensor, t: torch.Tensor) -> FlowTensor:
        t = t.double().to(x.device)
        logr = 2 * torch.log(self.sigma_max / self.sigma_min).to(x.device)
        mul = 0.5 * logr * self.sigma_max.to(x.device)**2 * (self.sigma_min / self.sigma_max).to(x.device)**(2 * t)
        result = self.alpha(x, t) * mul
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

    def feature_postprocess(self, x: DockResult, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        # `feats` is a tensor of shape (num_nodes, feature_dim), which we want to take the mean over
        # for each graph in the batch. We have the indices that each node belongs to which graph in x.n_idx.
        device = x.device
        num_nodes, feature_dim = feats.shape
        num_graphs = int(x.n_idx.max().item()) + 1

        # Sum features per graph
        graph_sums = torch.zeros(num_graphs, feature_dim, device=device)
        graph_sums.index_add_(0, x.n_idx, feats)

        # Count nodes per graph
        graph_counts = torch.zeros(num_graphs, device=device)
        graph_counts.index_add_(0, x.n_idx, torch.ones(num_nodes, device=device))

        # Mean pooling
        return graph_sums / graph_counts.unsqueeze(1)

    def latent_postprocess(self, latents: DockResult, valids: torch.Tensor, kwargs: dict) -> DockResult:
        raise NotImplementedError

        graph = latents.graph.clone()

        def _discretize(x):
            # argmax finds the class index, one_hot creates the vector
            # type_as ensures we match the original float precision and device
            return F.one_hot(x.argmax(dim=-1), num_classes=x.shape[-1]).type_as(x)

        # Set categorical features to be one-hot vectors over dim -1
        graph.ndata["a_t"] = _discretize(graph.ndata["a_t"])
        graph.ndata["c_t"] = _discretize(graph.ndata["c_t"])
        graph.edata["e_t"] = _discretize(graph.edata["e_t"])

        # Relax the geometry, otherwise the structures will become increasingly distorted
        graph.edata["ue_mask"] = latents.ue_mask
        if self.geometry_opt != "none":
            graphs = []
            for i, g in enumerate(dgl.unbatch(graph)):
                if valids[i]:
                    g = relax_positions(g, self.base_model.model.atom_type_map, self.geometry_opt)

                graphs.append(g)

            graph = dgl.batch(graphs)

        # Remove center of mass
        init_coms = dgl.readout_nodes(graph, feat="x_t", op="mean")
        graph.ndata["x_t"] = graph.ndata["x_t"] - init_coms[latents.n_idx]

        return FlowGraph(graph, latents.ue_mask, latents.n_idx, latents.e_idx)


    @torch.no_grad()
    def visualize_sample(
        self,
        env: Environment[DockResult],
        samples: list[DockResult],
        valids: list[torch.Tensor],
    ) -> Figure:
        raise NotImplementedError
        mols = self._to_mols(samples[-1])

        fig = plt.figure(figsize=(12, 8))
        cols = 8
        rows = (len(mols) + cols - 1) // cols
        for i, mol in enumerate(mols):
            ax = fig.add_subplot(rows, cols, i + 1)
            if valids[-1][i]:
                ax.set_title("Valid", color="green", fontsize=8)
            else:
                ax.set_title("Invalid", color="red", fontsize=8)

            ax.axis("off")
            img = Draw.MolToImage(mol, size=(150, 150))
            ax.imshow(img)

        return fig

    def eval_sampling_kwargs(self, n: int) -> dict:
        raise NotImplementedError
        probs = self.base_model.model.n_atoms_dist.probs

        expected = n * probs
        counts = expected.round().long()

        # Correct for rounding errors
        remainder = n - counts.sum()
        if remainder > 0:
            fractional = expected - counts
            extra = torch.topk(fractional, remainder).indices
            counts[extra] += 1

        indices = torch.arange(len(counts), device=counts.device)
        result = torch.repeat_interleave(indices, counts)

        # Randomly order result
        perm = torch.randperm(result.numel(), device=result.device)
        result = result[perm]

        return {"n_atoms": result}

    def save_sample(self, sample: DockResult, kwargs: dict, filename: os.PathLike | str):
        raise NotImplementedError
        if len(sample) != 1:
            raise ValueError("Can only save a single sample at a time.")

        mol = SampledMolecule(sample.graph.cpu(), self.base_model.model.atom_type_map).rdkit_mol
        mol = validate_mol(mol)

        if mol is None:
            return

        Chem.MolToMolFile(mol, f"{filename}.mol")

    def compute_metrics(
        self,
        samples: list[DockResult],
        valids: list[torch.Tensor],
        kwargs: list[dict],
    ) -> dict[str, float]:
        raise NotImplementedError
        n_samples = 0
        mols = []
        for d, v in zip(samples, valids):
            for j in range(len(d)):
                n_samples += 1

                if not v[j]:
                    continue

                mol = SampledMolecule(d[j].graph.cpu(), self.base_model.model.atom_type_map).rdkit_mol
                mol = validate_mol(mol)

                if mol is not None:
                    mols.append(mol)

        # We want to make sure the number of samples we compute the diversity on is constant between iterations
        mols = mols[:n_samples // self.div_valid]
        K = molecule_utils.get_tanimoto_K(mols)
        vendi_score = vendi.score_K(K)

        return { "vendi": float(vendi_score) }

    def compute_sample_metrics(self, sample_files: list[SampleFile]) -> dict[str, dict[str, float]]:
        raise NotImplementedError
        # Load molecule samples
        valid_mols = []
        valid_files = []
        for sample_file in sample_files:
            if sample_file.is_valid:
                mol = Chem.MolFromMolFile(sample_file.file, sanitize=False, removeHs=False, strictParsing=False)
                if mol is not None:
                    valid_files.append(sample_file.file)
                    valid_mols.append(mol)

        # Compute quantum chemistry properties
        res = parallel_xtb(valid_mols)

        # Return in nice format
        metric_names = ["energy", "homo", "lumo", "homo_lumo_gap", "dipole_moment", "polarizability", "heat_capacity"]
        out = dict()
        for fn, xtb_res in zip(valid_files, res):
            if xtb_res is None:
                continue

            sample_name = os.path.basename(fn).replace(".mol", "")
            metrics = { name: getattr(xtb_res, name) for name in metric_names }
            out[sample_name] = metrics

        return out
