from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from diffdock.utils.inference_utils import InferenceDataset
from diffdock.utils.sampling import modify_conformer_batch
from flowgym import BaseModel, FlowMixin, MemorylessNoiseSchedule, NoiseSchedule, Scheduler
from flowgym.types import BinaryOp, UnaryOp, FlowTensor
from flowgym.utils import append_dims

from torch_geometric.data.batch import Batch
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.data.storage import NodeStorage
from typing_extensions import Self
from argparse import Namespace


@dataclass(frozen=True)
class DockPose(object):
    tr_pose: torch.Tensor
    rot_pose: torch.Tensor
    tor_pose: torch.Tensor


class DockResult(FlowMixin): # TODO maybe change to subclass HeteroData???
    def __init__(
        self,
        complex_graph: HeteroData | Batch
    ):
        if 'ligand_pose' not in complex_graph:
            if isinstance(complex_graph, Batch):
                data_list = complex_graph.to_data_list()
                for graph in data_list:
                    graph['ligand_pose'].rot = torch.zeros((1, 3))
                    graph['ligand_pose'].tr = torch.zeros((1, 3))
                    graph['ligand_pose'].tor = torch.zeros((1, 3))
                complex_graph = Batch.from_data_list(data_list)

            else:
                complex_graph['ligand_pose'].rot = torch.zeros((1, 3))
                complex_graph['ligand_pose'].tr = torch.zeros((1, 3))
                complex_graph['ligand_pose'].tor = torch.zeros((1, 3))

        self.graph = complex_graph


    def __repr__(self) -> str:
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError
        return getattr(self.graph, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise NotImplementedError
        if name == "graph":
            object.__setattr__(self, name, value)
        else:

            setattr(self.graph, name, value)

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
        graph['ligand_pose'].tor = pose.tor_pose
        graph['ligand_pose'].rot = pose.rot_pose
        graph['ligand_pose'].tr = pose.tr_pose

        return cls(graph)


    def apply(self, op: UnaryOp) -> Self:
        raise NotImplementedError
        res = self.graph.clone()

        for key, sub in self.graph.node_items():
            for nkey, ndata in sub.items():
                if isinstance(ndata, torch.Tensor):
                    res[key][nkey] = op(ndata)

        for key, sub in self.graph.edge_items():
            for ekey, edata in sub.items():
                if isinstance(edata, torch.Tensor):
                    res[key][ekey] = op(edata)

        return type(self)(res)


    def apply_pose(self, pose: DockPose) -> torch.Tensor:
        tr_perturb, rot_perturb, tor_perturb = pose.tr_pose, pose.rot_pose, pose.tor_pose
        mask_rotate = torch.from_numpy(self.graph['ligand'].mask_rotate[0]).to(self.device)

        return modify_conformer_batch(self.graph['ligand'].pos, self.graph,
                                      tr_perturb, rot_perturb, tor_perturb, mask_rotate)


    def combine(self, other: Union[Self, float, torch.Tensor], op: BinaryOp) -> Self:
        res = self.graph.clone()
        if isinstance(other, Self):
            pose = other['ligand_pose']
            rot, tr, tor = pose.rot, pose.tr, pose.tor
            res.pose.rot = op(self.graph.pose.rot, rot)
            res.pose.tr = op(self.graph.pose.tr, tr)
            res.pose.tor = op(self.graph.pose.tor, tor)

        else:
            res.pose.rot = op(self.graph.pose.rot, other)
            res.pose.tr = op(self.graph.pose.tr, other)
            res.pose.tor = op(self.graph.pose.tor, other)

        pose = DockPose(res.pose.tr, res.pose.rot, res.pose.tor)
        res['ligand'].pos = self.apply_pose(pose)
        return type(self)(res)


class DiffDockBaseModel(BaseModel[DockResult]):
    def __init__(self, scheduler_params):
        self._scheduler = None
        # TODO diffdock has drift = sigma_t^2 score, which corresponds to memoryless
        # i.e.: need eta_t = 0.5 sigma_t^2 (except where does the a * x come in?)
        raise NotImplementedError

    @property
    def scheduler(self) -> "DiffDockScheduler":
        """Base model-dependent scheduler used for sampling."""
        raise NotImplementedError

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
    def forward(self, x: D, t: torch.Tensor, **kwargs: Any) -> D:
        """Forward pass of the base model.

        Parameters
        ----------
        x : D
            Input data.

        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].

        Returns
        -------
        output : D
            Output of the model.
        """

    def preprocess(self, x: D, **kwargs: Any) -> tuple[D, dict[str, Any]]:
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

    def postprocess(self, x: D) -> D:
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


@dataclass(frozen=True)
class LogLinearScheduler(Scheduler[FlowTensor]):
    sigma_max: torch.Tensor
    sigma_min: torch.Tensor


    def alpha(self, x: FlowTensor, t: torch.Tensor) -> FlowTensor:
        t = t.double()
        logr = torch.tensor(2 * torch.log(self.sigma_max / self.sigma_min))
        diff = (self.sigma_min / self.sigma_max)** (2 * t) - (self.sigma_min / self.sigma_max) **2
        result = torch.exp(-logr * self.sigma_max**2 / (4 * torch.log(self.sigma_max / self.sigma_min)) * diff)
        return FlowTensor(append_dims(result, x.data.ndim))


    def alpha_dot(self, x: FlowTensor, t: torch.Tensor) -> FlowTensor:
        t = t.double()
        logr = torch.tensor(2 * torch.log(self.sigma_max / self.sigma_min))
        result = 0.5 * logr * self.sigma_max**2 * (self.sigma_min / self.sigma_max)**(2 * t) * self.alpha(x, t)
        return FlowTensor(append_dims(result, x.data.ndim))


class DiffDockScheduler(Scheduler[DockResult]):
    def __init__(self, params: Namespace):
        self.schedulers = {
            'tr': LogLinearScheduler(params.rot_sigma_max, params.rot_sigma_min),
            'rot': LogLinearScheduler(params.rot_sigma_max, params.rot_sigma_min),
            'tor': LogLinearScheduler(params.tor_sigma_max, params.tor_sigma_min)
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

    @abstractmethod
    def alpha(self, x: DockResult, t: torch.Tensor) -> DockResult:
        r""":math:`\alpha_t`.

        Can be overwritten if :math:`\alpha_t` is data-dependent.

        Parameters
        ----------
        x : D
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        alpha_t : D, same data shape as x
            Values of :math:`\alpha_t` at the given times.
        """
        ...

    @abstractmethod
    def alpha_dot(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\dot{\alpha}_t`.

        Can be overwritten if :math:`\dot{\alpha}_t` is data-dependent.

        Parameters
        ----------
        x : D
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        alpha_dot_t : D, same data shape as x
            Values of :math:`\dot{\alpha}_t` at the given times.
        """
        ...

    def model_input(self, t: torch.Tensor) -> torch.Tensor:
        """Input to the model at time t.

        Defaults to t, but could be different if using a different time parameterization.
        """
        return t

    def kappa(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\kappa_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u)."""
        return self.alpha_dot(x, t) / self.alpha(x, t)

    def eta(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\eta_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u)."""
        alpha = self.alpha(x, t)
        alpha_dot = self.alpha_dot(x, t)
        beta = self.beta(x, t)
        beta_dot = self.beta_dot(x, t)
        return beta * ((alpha_dot / alpha) * beta - beta_dot)