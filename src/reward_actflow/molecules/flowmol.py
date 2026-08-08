"""Pre-trained continuous FlowMol model, trained on GEOM-Drugs."""

import warnings
from typing import Any

import flowmol
import torch
import torch.nn.functional as F
from diffusiongym import BaseModel, ConstantNoiseSchedule, CosineScheduler, Scheduler
from diffusiongym.registry import base_model_registry
from diffusiongym.types import DDTensor
from flowmol.data_processing.utils import (
    build_edge_idxs,
    get_batch_idxs,
    get_upper_edge_mask,
)
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.nn import global_mean_pool

from reward_actflow.molecules.types import (
    DDGraph,
    _iter_edge_features,
    _iter_node_features,
)


class FlowMolBaseModel(BaseModel[DDGraph]):
    """Pre-trained FlowMol on GEOM-Drugs and QM9."""

    output_type = "endpoint"

    def __init__(
        self,
        model_name: str,
        scheduler_params: tuple[float, float, float, float],
        device: torch.device,
    ):
        super().__init__(device)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="To copy construct from a tensor, it is recommended to use sourceTensor.clone",
            )
            self.model = flowmol.load_pretrained(model_name).to(device)

        self._scheduler = FlowMolScheduler(scheduler_params)
        self._scheduler.noise_schedule = ConstantNoiseSchedule(0)

    @property
    def scheduler(self) -> "FlowMolScheduler":
        """Scheduler used for sampling."""
        return self._scheduler

    def sample_p0(self, n: int, **kwargs) -> tuple[DDGraph, dict[str, Any]]:
        """Sample n datapoints from the base distribution :math:`p_0`.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : DDGraph
            Samples from the base distribution :math:`p_0`.

        Notes
        -----
        The base distribution :math:`p_0` is a standard Gaussian distribution.
        """
        n_atoms = kwargs.get("n_atoms", None)
        if n_atoms is None:
            n_atoms = self.model.sample_n_atoms(n)

        if isinstance(n_atoms, int):
            n_atoms = torch.full((n,), n_atoms, dtype=torch.long, device=self.device)

        if len(n_atoms) != n:
            raise ValueError(f"n_atoms must be of length n ({len(n_atoms)} != {n}).")

        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        data_list: list[BaseData] = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            data_list.append(
                Data(edge_index=edge_idxs.to(self.device), num_nodes=int(n_atoms_i))
            )

        g = Batch.from_data_list(data_list)
        ue_mask = get_upper_edge_mask(g)
        n_idx, e_idx = get_batch_idxs(g)
        g = self.model.sample_prior(g, n_idx, ue_mask)

        # Rename _0 features to _t (FlowMol stores priors with _0 suffix)
        for key in list(g.keys()):
            if key.endswith("_0"):
                g[key[:-2] + "_t"] = g[key]
                del g[key]

        return DDGraph(g, ue_mask, n_idx, e_idx), kwargs

    def preprocess(self, x: DDGraph, **kwargs) -> tuple[DDGraph, dict[str, Any]]:
        if "n_atoms" not in kwargs:
            kwargs["n_atoms"] = x.graph.ptr.diff()

        return x, kwargs

    def postprocess(self, x: DDGraph) -> DDGraph:
        """Re-name features from x_t to x_1."""
        g = x.graph.clone()
        for key in list(g.keys()):
            if key.endswith("_t"):
                g[key[:-2] + "_1"] = g[key]
                del g[key]

        # SampledMolecule (PyG-native) expects upper_edge_mask as a direct attribute
        g.upper_edge_mask = x.ue_mask
        return DDGraph(g, x.ue_mask, x.n_idx, x.e_idx)

    def forward(self, x: DDGraph, t: torch.Tensor, **kwargs) -> DDGraph:
        r"""Compute the endpoint vector field :math:`\hat{x_1}(x, t)`."""
        output = self.model.vector_field(
            x.graph,
            t,
            x.n_idx,
            x.ue_mask,
            apply_softmax=kwargs.get("apply_softmax", True),
            remove_com=kwargs.get("remove_com", True),
        )

        out_graph = x._get_empty_graph()
        for key, _ in _iter_node_features(x.graph):
            out_graph[key] = output[key[:-2]]

        for key, data in _iter_edge_features(x.graph):
            # Output only contains upper edge data; expand to both upper and lower
            edge_data = torch.zeros_like(data)
            edge_data[x.ue_mask] = output[key[:-2]]
            edge_data[~x.ue_mask] = output[key[:-2]]
            out_graph[key] = edge_data

        return DDGraph(out_graph, x.ue_mask, x.n_idx, x.e_idx)

    def train_loss(
        self,
        x1: DDGraph,
        xt: DDGraph | None = None,
        t: torch.Tensor | None = None,
        pred: DDGraph | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss for a single batch training step.

        Parameters
        ----------
        x1 : DDGraph
            Target data points.
        xt : Optional[DDGraph], default=None
            Noisy data points at time t. If None, will be sampled.
        t : Optional[torch.Tensor], shape (len(x1),), default=None
            Time steps. If None, will be sampled.
        pred : Optional[DDGraph], default=None
            Model predictions. If None, will be computed by the model.
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        loss : torch.Tensor, shape (len(x1),)
            Computed loss for the training step.
        """
        if t is None:
            t = torch.rand(len(x1), device=x1.device)

        assert t.shape == (len(x1),)

        alpha = self.scheduler.alpha(x1, t)
        beta = self.scheduler.beta(x1, t)

        if xt is None:
            x0 = x1.randn_like()
            xt = alpha * x1 + beta * x0
        else:
            assert len(xt) == len(x1)
            x0 = (xt - alpha * x1) / beta

        if pred is None:
            pred = self.forward(xt, t, apply_softmax=False, remove_com=False)

        g = pred.graph

        # Compute per-node/edge losses without mutating pred.graph
        loss_x = F.mse_loss(g.x_t, x1.graph.x_t, reduction="none").mean(dim=-1)
        loss_a = F.cross_entropy(g.a_t, x1.graph.a_t.argmax(dim=-1), reduction="none")
        loss_c = F.cross_entropy(g.c_t, x1.graph.c_t.argmax(dim=-1), reduction="none")
        # todo: only over upper edge
        loss_e = F.cross_entropy(g.e_t, x1.graph.e_t.argmax(dim=-1), reduction="none")

        losses = {
            "x": global_mean_pool(loss_x.unsqueeze(-1), pred.n_idx).squeeze(-1),
            "a": global_mean_pool(loss_a.unsqueeze(-1), pred.n_idx).squeeze(-1),
            "c": global_mean_pool(loss_c.unsqueeze(-1), pred.n_idx).squeeze(-1),
            "e": global_mean_pool(loss_e.unsqueeze(-1), pred.e_idx).squeeze(-1),
        }

        total_loss = torch.zeros(len(x1), device=x1.device, requires_grad=True)
        loss_weights = self.model.total_loss_weights
        for feat, loss in losses.items():
            total_loss = total_loss + loss_weights[feat] * loss

        return total_loss


class FlowMolScheduler(Scheduler[DDGraph]):
    """Scheduler for the GEOM-Drugs dataset."""

    def __init__(self, cosine_params: tuple[float, float, float, float] = (1, 2, 2, 2)):
        self.schedulers = {
            "x_t": CosineScheduler(cosine_params[0]),
            "a_t": CosineScheduler(cosine_params[1]),
            "c_t": CosineScheduler(cosine_params[2]),
            "e_t": CosineScheduler(cosine_params[3]),
        }
        self.scheduler_order = ["x_t", "a_t", "c_t", "e_t"]

    def _alpha(self, x: DDGraph, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            t.shape[0], len(self.schedulers), device=t.device, dtype=t.dtype
        )
        for idx, key in enumerate(self.scheduler_order):
            val = self.schedulers[key].alpha(
                DDTensor(torch.zeros(x.graph.num_graphs, 1, device=x.device)), t
            )
            out[:, idx] = val.data.squeeze(-1)

        return out

    def alpha(self, x: DDGraph, t: torch.Tensor) -> DDGraph:
        r""":math:`\alpha_t`."""
        g = x.graph
        res = x._get_empty_graph()

        alphas = self._alpha(x, t)
        node_feat_keys = {k for k, _ in _iter_node_features(g)}
        edge_feat_keys = {k for k, _ in _iter_edge_features(g)}
        for idx, key in enumerate(self.scheduler_order):
            if key in node_feat_keys:
                t_alpha = alphas[:, idx].unsqueeze(-1)
                res[key] = t_alpha[x.n_idx]
            elif key in edge_feat_keys:
                t_alpha = alphas[:, idx].unsqueeze(-1)
                res[key] = t_alpha[x.e_idx]
            else:
                raise ValueError(f"Key {key} not found in graph data.")

        return DDGraph(res, x.ue_mask, x.n_idx, x.e_idx)

    def _alpha_dot(self, x: DDGraph, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            t.shape[0], len(self.schedulers), device=t.device, dtype=t.dtype
        )
        for idx, key in enumerate(self.scheduler_order):
            val = self.schedulers[key].alpha_dot(
                DDTensor(torch.zeros(x.graph.num_graphs, 1, device=x.device)), t
            )
            out[:, idx] = val.data.squeeze(-1)

        return out

    def alpha_dot(self, x: DDGraph, t: torch.Tensor) -> DDGraph:
        r""":math:`\dot{\alpha}_t`."""
        g = x.graph
        res = x._get_empty_graph()

        alpha_dots = self._alpha_dot(x, t)
        node_feat_keys = {k for k, _ in _iter_node_features(g)}
        edge_feat_keys = {k for k, _ in _iter_edge_features(g)}
        for idx, key in enumerate(self.scheduler_order):
            if key in node_feat_keys:
                t_alpha_dot = alpha_dots[:, idx].unsqueeze(-1)
                res[key] = t_alpha_dot[x.n_idx]
            elif key in edge_feat_keys:
                t_alpha_dot = alpha_dots[:, idx].unsqueeze(-1)
                res[key] = t_alpha_dot[x.e_idx]
            else:
                raise ValueError(f"Key {key} not found in graph data.")

        return DDGraph(res, x.ue_mask, x.n_idx, x.e_idx)


@base_model_registry.register("molecules/flowmol_geom")
class GEOMBaseModel(FlowMolBaseModel):
    """Pre-trained continuous flow matching model, trained on GEOM-Drugs."""

    def __init__(self, device: torch.device):
        super().__init__("flowmol3", (1, 2, 2, 2), device)


@base_model_registry.register("molecules/flowmol_qm9")
class QM9BaseModel(FlowMolBaseModel):
    """Pre-trained continuous flow matching model, trained on QM9."""

    def __init__(self, device: torch.device):
        super().__init__("qm9_gaussian", (1, 2, 2, 1.5), device)
