"""Tests for DDGraph (PyG-backed)."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from reward_actflow.molecules.types import (
    DDGraph,
    _iter_edge_features,
    _iter_node_features,
    construct_e_idx,
    construct_n_idx,
    construct_ue_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(n_nodes_list: list[int]) -> Batch:
    """Build a Batch of complete-graph molecules with given node counts."""
    from flowmol.data_processing.utils import build_edge_idxs  # noqa: PLC0415

    return Batch.from_data_list([Data(edge_index=build_edge_idxs(n), num_nodes=n) for n in n_nodes_list])


def _make_ddgraph(n_nodes_list: list[int], with_features: bool = False) -> DDGraph:
    g = _make_batch(n_nodes_list)
    ddg = DDGraph(g)
    if with_features:
        total_nodes = g.num_nodes
        total_edges = g.num_edges
        ddg.graph.x_t = torch.randn(total_nodes, 3)
        ddg.graph.a_t = torch.randn(total_nodes, 10)
        ddg.graph.c_t = torch.randn(total_nodes, 6)
        ddg.graph.e_t = torch.randn(total_edges, 4)
    return ddg


# ---------------------------------------------------------------------------
# construct_* helpers
# ---------------------------------------------------------------------------


class TestConstructHelpers:
    def test_construct_ue_mask_single_graph(self) -> None:
        """A 3-node complete graph has 6 edges: 3 upper, 3 lower."""
        g = _make_batch([3])
        mask = construct_ue_mask(g)
        assert mask.dtype == torch.bool
        assert mask.shape == (6,)
        assert mask.sum() == 3

    def test_construct_ue_mask_batched(self) -> None:
        """Batching two 3-node graphs gives 12 edges with 6 upper."""
        g = _make_batch([3, 3])
        mask = construct_ue_mask(g)
        assert mask.shape == (12,)
        assert mask.sum() == 6

    def test_construct_n_idx_maps_nodes_to_graphs(self) -> None:
        """3-node + 4-node: first 3 nodes map to 0, next 4 map to 1."""
        g = _make_batch([3, 4])
        n_idx = construct_n_idx(g)
        assert n_idx.tolist() == [0, 0, 0, 1, 1, 1, 1]

    def test_construct_e_idx_maps_edges_to_graphs(self) -> None:
        """All edges of graph 0 map to 0, all of graph 1 map to 1."""
        g = _make_batch([3, 3])
        e_idx = construct_e_idx(g)
        # 6 edges per 3-node complete graph
        assert (e_idx[:6] == 0).all()
        assert (e_idx[6:] == 1).all()


# ---------------------------------------------------------------------------
# DDGraph init / repr / len
# ---------------------------------------------------------------------------


class TestDDGraphBasics:
    def test_init_defaults(self) -> None:
        ddg = _make_ddgraph([3])
        assert ddg.ue_mask is not None
        assert ddg.n_idx is not None
        assert ddg.e_idx is not None

    def test_len(self) -> None:
        ddg = _make_ddgraph([3, 4, 5])
        assert len(ddg) == 3

    def test_repr_contains_counts(self) -> None:
        ddg = _make_ddgraph([3])
        r = repr(ddg)
        assert "DDGraph" in r
        assert "num_nodes=3" in r
        assert "batch_size=1" in r

    def test_device_is_cpu(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        assert ddg.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# iter helpers
# ---------------------------------------------------------------------------


class TestIterHelpers:
    def test_iter_node_features_finds_node_tensors(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        keys = {k for k, _ in _iter_node_features(ddg.graph)}
        assert "x_t" in keys
        assert "a_t" in keys
        assert "c_t" in keys
        assert "e_t" not in keys  # edge feature

    def test_iter_edge_features_finds_edge_tensors(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        keys = {k for k, _ in _iter_edge_features(ddg.graph)}
        assert "e_t" in keys
        assert "x_t" not in keys  # node feature

    def test_reserved_keys_excluded(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        all_keys = {k for k, _ in _iter_node_features(ddg.graph)} | {k for k, _ in _iter_edge_features(ddg.graph)}
        for reserved in ("edge_index", "batch", "ptr", "num_nodes"):
            assert reserved not in all_keys


# ---------------------------------------------------------------------------
# DDGraph.__getitem__
# ---------------------------------------------------------------------------


class TestDDGraphGetitem:
    def test_int_index(self) -> None:
        ddg = _make_ddgraph([3, 4], with_features=True)
        item = ddg[0]
        assert len(item) == 1
        assert item.graph.num_nodes == 3

    def test_negative_int_index(self) -> None:
        ddg = _make_ddgraph([3, 4])
        item = ddg[-1]
        assert item.graph.num_nodes == 4

    def test_out_of_range_raises(self) -> None:
        ddg = _make_ddgraph([3])
        with pytest.raises(IndexError):
            _ = ddg[5]

    def test_slice_index(self) -> None:
        ddg = _make_ddgraph([3, 4, 5])
        sliced = ddg[1:]
        assert len(sliced) == 2

    def test_empty_slice_raises(self) -> None:
        ddg = _make_ddgraph([3, 4])
        with pytest.raises(ValueError):
            _ = ddg[5:10]

    def test_invalid_type_raises(self) -> None:
        ddg = _make_ddgraph([3])
        with pytest.raises(TypeError):
            _ = ddg["bad"]


# ---------------------------------------------------------------------------
# DDGraph.collate
# ---------------------------------------------------------------------------


class TestDDGraphCollate:
    def test_collate_single(self) -> None:
        items = [_make_ddgraph([3], with_features=True)]
        collated = DDGraph.collate(items)
        assert len(collated) == 1

    def test_collate_multiple(self) -> None:
        items = [_make_ddgraph([3], with_features=True), _make_ddgraph([4], with_features=True)]
        collated = DDGraph.collate(items)
        assert len(collated) == 2
        assert collated.graph.num_nodes == 7

    def test_collate_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            DDGraph.collate([])


# ---------------------------------------------------------------------------
# DDGraph.to (device)
# ---------------------------------------------------------------------------


class TestDDGraphTo:
    def test_to_same_device_noop(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        moved = ddg.to("cpu")
        assert moved.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        moved = ddg.to("cuda:0")
        assert moved.device == torch.device("cuda:0")


# ---------------------------------------------------------------------------
# DDGraph.apply
# ---------------------------------------------------------------------------


class TestDDGraphApply:
    def test_apply_negate(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        negated = ddg.apply(torch.neg)
        torch.testing.assert_close(negated.graph.x_t, -ddg.graph.x_t)

    def test_apply_does_not_mutate_original(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        original_x_t = ddg.graph.x_t.clone()
        ddg.apply(lambda x: x * 99)
        torch.testing.assert_close(ddg.graph.x_t, original_x_t)

    def test_apply_preserves_batch_structure(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        result = ddg.apply(lambda x: x * 2)
        assert (result.ue_mask == ddg.ue_mask).all()
        assert result.graph.num_nodes == ddg.graph.num_nodes


# ---------------------------------------------------------------------------
# DDGraph.combine
# ---------------------------------------------------------------------------


class TestDDGraphCombine:
    def test_combine_scalar_multiply(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        result = ddg * 2.0
        torch.testing.assert_close(result.graph.x_t, ddg.graph.x_t * 2.0)

    def test_combine_two_ddgraphs_add(self) -> None:
        ddg1 = _make_ddgraph([3], with_features=True)
        # Build ddg2 with identical topology but different features
        g2 = _make_batch([3])
        g2.x_t = torch.randn(3, 3)
        g2.a_t = torch.randn(3, 10)
        g2.c_t = torch.randn(3, 6)
        g2.e_t = torch.randn(6, 4)
        ddg2 = DDGraph(g2)
        result = ddg1 + ddg2
        torch.testing.assert_close(result.graph.x_t, ddg1.graph.x_t + ddg2.graph.x_t)


# ---------------------------------------------------------------------------
# DDGraph.aggregate
# ---------------------------------------------------------------------------


class TestDDGraphAggregate:
    def test_aggregate_sum_known_value(self) -> None:
        """3-node graph, x_t all ones (shape 3x3): sum = 9 node contributions."""
        ddg = _make_ddgraph([3])
        ddg.graph.x_t = torch.ones(3, 3)
        # Remove other features to isolate
        result = ddg.aggregate(reduction="sum")
        assert result.shape == (1,)
        assert result[0].item() == pytest.approx(9.0)

    def test_aggregate_mean_shape(self) -> None:
        ddg = _make_ddgraph([3, 4], with_features=True)
        result = ddg.aggregate(reduction="mean")
        assert result.shape == (2,)

    def test_aggregate_mean_does_not_divide_by_zero(self) -> None:
        """Aggregate on a graph with zero features should return 0, not NaN."""
        ddg = _make_ddgraph([3])
        result = ddg.aggregate(reduction="mean")
        assert not result.isnan().any()


# ---------------------------------------------------------------------------
# DDGraph.randn_like
# ---------------------------------------------------------------------------


class TestDDGraphRandnLike:
    def test_randn_like_com_removed(self) -> None:
        """After randn_like, COM of x_t should be ~0 per graph."""
        ddg = _make_ddgraph([4], with_features=True)
        result = ddg.randn_like()
        com = result.graph.x_t.mean(dim=0)
        torch.testing.assert_close(com, torch.zeros(3), atol=1e-5, rtol=0)

    def test_randn_like_edge_symmetry(self) -> None:
        """Lower-triangle edges should equal upper-triangle after randn_like."""
        ddg = _make_ddgraph([4], with_features=True)
        result = ddg.randn_like()
        upper = result.graph.e_t[result.ue_mask]
        lower = result.graph.e_t[~result.ue_mask]
        torch.testing.assert_close(upper, lower)

    def test_randn_like_returns_same_type(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        result = ddg.randn_like()
        assert isinstance(result, DDGraph)
        assert len(result) == len(ddg)


# ---------------------------------------------------------------------------
# DDGraph._get_empty_graph
# ---------------------------------------------------------------------------


class TestDDGraphGetEmptyGraph:
    def test_empty_graph_has_topology(self) -> None:
        ddg = _make_ddgraph([3, 4])
        empty = ddg._get_empty_graph()
        assert (empty.edge_index == ddg.graph.edge_index).all()
        assert empty.num_nodes == ddg.graph.num_nodes
        assert (empty.batch == ddg.graph.batch).all()
        assert empty.num_graphs == ddg.graph.num_graphs

    def test_empty_graph_has_no_features(self) -> None:
        ddg = _make_ddgraph([3], with_features=True)
        empty = ddg._get_empty_graph()
        for key in ["x_t", "a_t", "c_t", "e_t"]:
            assert not hasattr(empty, key) or getattr(empty, key) is None

    def test_feature_can_be_set_on_empty(self) -> None:
        """Empty graph should accept feature assignment."""
        ddg = _make_ddgraph([3])
        empty = ddg._get_empty_graph()
        empty["x_t"] = torch.randn(3, 3)
        assert hasattr(empty, "x_t")
        assert empty.x_t.shape == (3, 3)
