"""Types for molecular graphs."""

from typing import TYPE_CHECKING, Any, Self, Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from diffusiongym.types import BinaryOp, DDBatch, UnaryOp

if TYPE_CHECKING:
    import numpy as np
    from torch_geometric.data.data import BaseData
    from torch_geometric.data.dataset import IndexType

    class Batch(Data):
        batch_size: int
        num_graphs: int

        def __getitem__(self, idx: int | np.integer | str | IndexType) -> Any: ...  # ty: ignore[invalid-method-override]
        def __len__(self) -> int: ...
        def __reduce__(self) -> Any: ...
        @classmethod
        def from_data_list(
            cls,
            data_list: list[BaseData],
            follow_batch: list[str] | None = None,
            exclude_keys: list[str] | None = None,
        ) -> Self: ...
        def get_example(self, idx: int) -> BaseData: ...
        def index_select(self, idx: IndexType) -> list[BaseData]: ...

else:
    from torch_geometric.data import Batch

_RESERVED_KEYS: frozenset[str] = frozenset({"edge_index", "batch", "ptr", "num_nodes"})


def _iter_node_features(g: Batch) -> list[tuple[str, torch.Tensor]]:
    """Return (key, tensor) pairs for node-feature attributes on g."""
    result = []
    for key in g.keys():
        if key in _RESERVED_KEYS:
            continue
        val = g[key]
        if isinstance(val, torch.Tensor) and val.size(0) == g.num_nodes:
            result.append((key, val))
    return result


def _iter_edge_features(g: Batch) -> list[tuple[str, torch.Tensor]]:
    """Return (key, tensor) pairs for edge-feature attributes on g."""
    assert g.edge_index is not None, "Batch should contain graphs"
    num_edges = g.edge_index.size(1)
    result = []
    for key in g.keys():
        if key in _RESERVED_KEYS:
            continue
        val = g[key]
        if isinstance(val, torch.Tensor) and val.size(0) == num_edges:
            result.append((key, val))
    return result


def construct_ue_mask(g: Batch) -> torch.Tensor:
    """Construct a mask indicating upper edges in the graph."""
    assert g.edge_index is not None, "Batch should contain graphs"
    device = g.edge_index.device
    nodes_per_graph = g.ptr[1:] - g.ptr[:-1]
    edges_per_mol = nodes_per_graph * (nodes_per_graph - 1)
    ul_pattern = torch.tensor([1, 0], device=device).repeat(g.num_graphs)
    n_edges_pattern = (edges_per_mol // 2).repeat_interleave(2)
    return ul_pattern.repeat_interleave(n_edges_pattern).bool()


def construct_n_idx(g: Batch) -> torch.Tensor:
    """Construct a tensor which maps each node to its graph index in the batch."""
    assert g.batch is not None, "Batch should contain graphs"
    return g.batch


def construct_e_idx(g: Batch) -> torch.Tensor:
    """Construct a tensor which maps each edge to its graph index in the batch."""
    assert g.batch is not None, "Batch should contain graphs"
    assert g.edge_index is not None, "Batch should contain graphs"
    return g.batch[g.edge_index[0]]


class DDGraph(DDBatch):
    """A wrapper around a PyG Batch that supports required factory methods.

    Parameters
    ----------
    graph : Batch
        The batched graph to wrap.
    ue_mask : Optional[torch.Tensor], optional
        Mask indicating upper edges in the graph, by default None
    n_idx : Optional[torch.Tensor], optional
        Tensor mapping each node to its graph index in the batch, by default None
    e_idx : Optional[torch.Tensor], optional
        Tensor mapping each edge to its graph index in the batch, by default None
    """

    def __init__(
        self,
        graph: Batch,
        ue_mask: torch.Tensor | None = None,
        n_idx: torch.Tensor | None = None,
        e_idx: torch.Tensor | None = None,
    ):
        if ue_mask is None:
            ue_mask = construct_ue_mask(graph)
        if n_idx is None:
            n_idx = construct_n_idx(graph)
        if e_idx is None:
            e_idx = construct_e_idx(graph)

        self.graph = graph
        self.ue_mask = ue_mask
        self.n_idx = n_idx
        self.e_idx = e_idx

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_nodes={self.graph.num_nodes}, num_edges={self.graph.num_edges}, batch_size={len(self)})"

    @property
    def device(self) -> torch.device:
        assert self.graph.edge_index is not None, "Batch should contain graph"
        return self.graph.edge_index.device

    def to(self, device: torch.device | str) -> Self:
        return self.__class__(
            self.graph.to(device),
            self.ue_mask.to(device),
            self.n_idx.to(device),
            self.e_idx.to(device),
        )

    def __len__(self) -> int:
        return int(self.graph.num_graphs)

    def __getitem__(self, idx: int | slice) -> Self:
        if isinstance(idx, int):
            n = len(self)

            if idx < 0:
                idx += n

            if idx < 0 or idx >= n:
                raise IndexError(f"Index {idx} out of range for batch size {n}")

            return self.__class__(Batch.from_data_list([self.graph.get_example(idx)]))

        if isinstance(idx, slice):
            graphs = self.graph.to_data_list()
            selected_graphs = graphs[idx]

            if not selected_graphs:
                raise ValueError("The slice resulted in an empty graph sequence.")

            return self.__class__(Batch.from_data_list(selected_graphs))

        raise TypeError(f"Invalid index type: {type(idx)}")

    @classmethod
    def collate(cls, items: Sequence[Self]) -> Self:
        if not items:
            raise ValueError("Cannot collate an empty sequence")

        data_list: list[BaseData] = []
        for item in items:
            data_list.extend(item.graph.to_data_list())
        return cls(Batch.from_data_list(data_list))

    def _get_empty_graph(self) -> Batch:
        """Get a topology-only copy of self.graph (no feature attributes)."""
        empty = Batch()
        empty.edge_index = self.graph.edge_index
        empty.num_nodes = self.graph.num_nodes
        empty.batch = self.graph.batch
        empty.ptr = self.graph.ptr
        empty._num_graphs = self.graph.num_graphs
        return empty

    def apply(self, op: UnaryOp) -> Self:
        res = self._get_empty_graph()

        for key, val in _iter_node_features(self.graph):
            res[key] = op(val)

        for key, val in _iter_edge_features(self.graph):
            res[key] = op(val)

        return self.__class__(res, self.ue_mask, self.n_idx, self.e_idx)

    def combine(self, other: Self | float | torch.Tensor, op: BinaryOp) -> Self:  # ty: ignore[invalid-method-override]
        res = self._get_empty_graph()

        if isinstance(other, DDGraph):
            for key, val in _iter_node_features(self.graph):
                other_val = other.graph[key] if hasattr(other.graph, key) else None
                if other_val is not None and isinstance(other_val, torch.Tensor):
                    res[key] = op(val, other_val)
                else:
                    res[key] = val

            for key, val in _iter_edge_features(self.graph):
                other_val = other.graph[key] if hasattr(other.graph, key) else None
                if other_val is not None and isinstance(other_val, torch.Tensor):
                    res[key] = op(val, other_val)
                else:
                    res[key] = val
        else:
            for key, val in _iter_node_features(self.graph):
                res[key] = op(val, other)  # ty: ignore[invalid-argument-type]

            for key, val in _iter_edge_features(self.graph):
                res[key] = op(val, other)  # ty: ignore[invalid-argument-type]

        return self.__class__(res, self.ue_mask, self.n_idx, self.e_idx)

    def aggregate(self, reduction: str = "mean") -> torch.Tensor:
        batch_size = len(self)
        assert self.graph.edge_index is not None, "Batch should contain graph"
        device = self.graph.edge_index.device
        summed = torch.zeros(batch_size, device=device)

        counts = None
        if reduction == "mean":
            counts = torch.zeros(batch_size, device=device)

        for _, val in _iter_node_features(self.graph):
            aggregated = torch.zeros(
                batch_size, *val.shape[1:], device=val.device, dtype=val.dtype
            )
            aggregated.index_add_(0, self.n_idx, val)
            summed += aggregated.sum(dim=-1)

            if counts is not None:
                num_elements = val[0].numel()
                item_counts = torch.zeros(batch_size, device=val.device)
                ones = torch.ones(val.size(0), device=val.device)
                item_counts.index_add_(0, self.n_idx, ones)
                counts += item_counts * num_elements

        for _, val in _iter_edge_features(self.graph):
            aggregated = torch.zeros(
                batch_size, *val.shape[1:], device=val.device, dtype=val.dtype
            )
            aggregated.index_add_(0, self.e_idx, val)
            summed += aggregated.sum(dim=-1)

            if counts is not None:
                num_elements = val[0].numel()
                item_counts = torch.zeros(batch_size, device=val.device)
                ones = torch.ones(val.size(0), device=val.device)
                item_counts.index_add_(0, self.e_idx, ones)
                counts += item_counts * num_elements

        if counts is not None:
            return summed / counts.clamp(min=1)

        return summed

    def randn_like(self) -> Self:
        out = super().randn_like()

        # Remove COM
        init_coms = global_mean_pool(out.graph.x_t, out.n_idx)
        out.graph.x_t = out.graph.x_t - init_coms[out.n_idx]

        # Also make sure that both sides of edges are equivalent
        out.graph.e_t[~out.ue_mask] = out.graph.e_t[out.ue_mask]

        return out
