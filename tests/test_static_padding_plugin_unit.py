"""
Production-grade unit + discovery tests for the static_padding plugin (Family H).

Ground-truth checks for each layout/alignment operator (counts, mask patterns, exact GCN
normalization weights, zero-padding), plus fail-closed prerequisites, determinism, input
immutability, attribute integrity, property-based invariants, boundaries, and discovery.
"""

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.plugins.static_padding.transforms import (
    EdgeAttrCanonicalPadding,
    LayerPreprocess,
    PowerOfTwoPadding,
    StaticBatchPadding,
    VirtualNodeMasking,
    _next_power_of_two,
)
from milia_pipeline.transformations.custom_transforms import TransformExecutionError


# --------------------------------------------------------------------------- helpers
def _data(edges, n, x=None, edge_attr=None, directed=False):
    if edges:
        if directed:
            src = [a for a, _ in edges]
            dst = [b for _, b in edges]
        else:
            src, dst = [], []
            for a, b in edges:
                src += [a, b]
                dst += [b, a]
        ei = torch.tensor([src, dst], dtype=torch.long)
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
    d = Data(edge_index=ei, num_nodes=n)
    if x is not None:
        d.x = x
    if edge_attr is not None:
        d.edge_attr = edge_attr
    return d


def _p3(with_x=True):
    x = torch.arange(6.0).view(3, 2) if with_x else None
    return _data([(0, 1), (1, 2)], 3, x=x)  # E = 4 directed columns


def _iter_edge_weights(out):
    return zip(out.edge_index.t().tolist(), out.edge_weight.tolist(), strict=True)


def _weight_map(out):
    return {tuple(e): w for e, w in _iter_edge_weights(out)}


# =========================================================================== _next_power_of_two
@pytest.mark.parametrize(
    "n, expected",
    [(0, 1), (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (7, 8), (8, 8), (9, 16), (16, 16)],
)
def test_next_power_of_two(n, expected):
    assert _next_power_of_two(n) == expected


# =========================================================================== H1 StaticBatchPadding
def test_static_batch_padding_nodes_and_edges():
    out = StaticBatchPadding(n_node=5, n_edge=6)(_p3())
    assert out.num_nodes == 5
    assert out.edge_index.size(1) == 6
    assert out.node_mask.tolist() == [True, True, True, False, False]
    assert out.edge_mask.tolist() == [True, True, True, True, False, False]
    assert out.x.shape == (5, 2)
    assert torch.equal(out.x[3:], torch.zeros((2, 2)))  # padding nodes zero features


def test_static_batch_padding_nodes_only():
    out = StaticBatchPadding(n_node=6)(_p3())
    assert out.num_nodes == 6
    assert out.node_mask.sum().item() == 3
    assert out.edge_mask.all()  # no edge padding -> all real
    assert out.edge_index.size(1) == 4  # unchanged


def test_static_batch_padding_masks_are_boolean():
    out = StaticBatchPadding(n_node=5, n_edge=6)(_p3())
    assert out.node_mask.dtype == torch.bool
    assert out.edge_mask.dtype == torch.bool


def test_static_batch_padding_n_node_too_small_fails():
    with pytest.raises(TransformExecutionError):
        StaticBatchPadding(n_node=2)(_p3())


def test_static_batch_padding_n_edge_too_small_fails():
    with pytest.raises(TransformExecutionError):
        StaticBatchPadding(n_node=5, n_edge=2)(_p3())


def test_static_batch_padding_edges_without_pad_node_fails():
    # n_node == N so there is no padding node to host padding edges
    with pytest.raises(TransformExecutionError):
        StaticBatchPadding(n_node=3, n_edge=8)(_p3())


@pytest.mark.parametrize("target", [3, 4, 7, 10])
def test_static_batch_padding_mask_preserves_real_count(target):
    out = StaticBatchPadding(n_node=target)(_p3())
    assert out.num_nodes == target
    assert out.node_mask.sum().item() == 3  # always the original node count


# =========================================================================== H2 VirtualNodeMasking
def test_virtual_node_masking_single():
    out = VirtualNodeMasking(num_virtual=1)(_p3())
    assert out.num_nodes == 4
    assert out.node_mask.tolist() == [True, True, True, False]
    # virtual node 3 connects to all 3 real nodes (bidirectional) -> +6 directed edges
    assert out.edge_index.size(1) == 4 + 6
    deg_virtual = (out.edge_index[0] == 3).sum().item()
    assert deg_virtual == 3


def test_virtual_node_masking_multiple():
    out = VirtualNodeMasking(num_virtual=2)(_p3())
    assert out.num_nodes == 5
    assert out.node_mask.sum().item() == 3
    # 2 virtual nodes x 3 reals x 2 directions = 12 new edges
    assert out.edge_index.size(1) == 4 + 12


def test_virtual_node_masking_features_zero_padded():
    out = VirtualNodeMasking(num_virtual=1)(_p3())
    assert out.x.shape == (4, 2)
    assert torch.equal(out.x[3], torch.zeros(2))


def test_virtual_node_masking_invalid_count_fails():
    with pytest.raises(TransformExecutionError):
        VirtualNodeMasking(num_virtual=0)(_p3())


# =========================================================================== H3 PowerOfTwoPadding
@pytest.mark.parametrize("n, target", [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (6, 8)])
def test_power_of_two_padding_node_counts(n, target):
    out = PowerOfTwoPadding()(_data([], n, x=torch.zeros((n, 2))))
    assert out.num_nodes == target
    assert out.node_mask.sum().item() == n
    assert out.node_mask.dtype == torch.bool


def test_power_of_two_padding_already_pow2_is_noop_on_count():
    out = PowerOfTwoPadding()(_data([], 4, x=torch.zeros((4, 2))))
    assert out.num_nodes == 4
    assert out.node_mask.all()


def test_power_of_two_padding_edges():
    # 3 nodes (pad to 4 -> a padding node exists), E=4 -> next pow2 4 already; use E=3 (directed)
    d = _data([(0, 1), (1, 2), (0, 2)], 3, directed=True)  # E=3 -> next pow2 4
    out = PowerOfTwoPadding(pad_edges=True)(d)
    assert out.num_nodes == 4
    assert out.edge_index.size(1) == 4  # padded to next power of two


def test_power_of_two_padding_edges_without_pad_node_fails():
    # 4 nodes already power-of-two (no padding node) but edges need padding
    d = _data([(0, 1), (1, 2), (2, 3)], 4, directed=True)  # E=3 -> wants 4, but no pad node
    with pytest.raises(TransformExecutionError):
        PowerOfTwoPadding(pad_edges=True)(d)


# =========================================================================== H4 LayerPreprocess
def test_layer_preprocess_gcn_weights_ground_truth():
    out = LayerPreprocess()(_p3())
    # augmented: original 4 edges (no self-loops) + 3 self-loops = 7
    assert out.edge_index.size(1) == 7
    assert out.edge_weight.shape[0] == 7
    wmap = _weight_map(out)
    # d~ = [2, 3, 2]; self-loop(1,1) = 1/3; edge(0,1) = 1/sqrt(6); self-loop(0,0) = 1/2
    assert wmap[(1, 1)] == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert wmap[(0, 1)] == pytest.approx(1.0 / math.sqrt(6), abs=1e-5)
    assert wmap[(0, 0)] == pytest.approx(0.5, abs=1e-5)


def test_layer_preprocess_symmetric_weights():
    out = LayerPreprocess()(_p3())
    wmap = _weight_map(out)
    assert wmap[(0, 1)] == pytest.approx(wmap[(1, 0)], abs=1e-6)


def test_layer_preprocess_dedups_existing_self_loops():
    # input already has a self-loop on node 0; result must still have exactly one per node
    d = _data([(0, 1), (1, 2), (0, 0)], 3, directed=True)
    out = LayerPreprocess()(d)
    self_loops = [(u, v) for u, v in out.edge_index.t().tolist() if u == v]
    assert sorted(self_loops) == [(0, 0), (1, 1), (2, 2)]


def test_layer_preprocess_requires_edge_index():
    d = Data(num_nodes=3)  # no edge_index
    with pytest.raises(TransformExecutionError):
        LayerPreprocess()(d)


# =========================================================== H5 EdgeAttrCanonicalPadding
def test_edge_attr_padding_widens():
    ea = torch.ones((4, 2))
    out = EdgeAttrCanonicalPadding(width=5)(_data([(0, 1), (1, 2)], 3, edge_attr=ea))
    assert out.edge_attr.shape == (4, 5)
    assert torch.equal(out.edge_attr[:, :2], torch.ones((4, 2)))
    assert torch.equal(out.edge_attr[:, 2:], torch.zeros((4, 3)))


def test_edge_attr_padding_equal_width_noop():
    ea = torch.ones((4, 3))
    out = EdgeAttrCanonicalPadding(width=3)(_data([(0, 1), (1, 2)], 3, edge_attr=ea))
    assert out.edge_attr.shape == (4, 3)


def test_edge_attr_padding_1d_input():
    ea = torch.ones(4)  # 1-D edge_attr -> treated as width 1
    out = EdgeAttrCanonicalPadding(width=3)(_data([(0, 1), (1, 2)], 3, edge_attr=ea))
    assert out.edge_attr.shape == (4, 3)


def test_edge_attr_padding_missing_attr_fails():
    with pytest.raises(TransformExecutionError):
        EdgeAttrCanonicalPadding(width=4)(_p3())


def test_edge_attr_padding_shrink_fails():
    ea = torch.ones((4, 5))
    with pytest.raises(TransformExecutionError):
        EdgeAttrCanonicalPadding(width=3)(_data([(0, 1), (1, 2)], 3, edge_attr=ea))


# =========================================================================== cross-cutting
def _edge_attr_data():
    return _data([(0, 1)], 2, edge_attr=torch.ones((2, 2)))


_CASES = [
    ("h1", lambda: StaticBatchPadding(n_node=6, n_edge=8), _p3),
    ("h2", lambda: VirtualNodeMasking(num_virtual=1), _p3),
    ("h3", lambda: PowerOfTwoPadding(), _p3),
    ("h4", lambda: LayerPreprocess(), _p3),
    ("h5", lambda: EdgeAttrCanonicalPadding(width=4), _edge_attr_data),
]
_CASE_IDS = [c[0] for c in _CASES]


@pytest.mark.parametrize("cid, make_t, make_d", _CASES, ids=_CASE_IDS)
def test_every_transform_runs_and_returns_data(cid, make_t, make_d):
    out = make_t()(make_d())
    assert isinstance(out, Data)
    assert out.edge_index.dim() == 2 and out.edge_index.size(0) == 2


@pytest.mark.parametrize("cid, make_t, make_d", _CASES, ids=_CASE_IDS)
def test_every_transform_is_deterministic(cid, make_t, make_d):
    t = make_t()
    a = t(make_d())
    b = t(make_d())
    assert torch.equal(a.edge_index, b.edge_index)


@pytest.mark.parametrize("cid, make_t, make_d", _CASES, ids=_CASE_IDS)
def test_every_transform_no_mutation(cid, make_t, make_d):
    d = make_d()
    ei_before = d.edge_index.clone()
    n_before = d.num_nodes
    make_t()(d)
    assert torch.equal(d.edge_index, ei_before)
    assert d.num_nodes == n_before


def test_all_transforms_declare_structural_metadata():
    for cls in (
        StaticBatchPadding,
        VirtualNodeMasking,
        PowerOfTwoPadding,
        LayerPreprocess,
        EdgeAttrCanonicalPadding,
    ):
        meta = cls.get_metadata()
        assert meta.category == "structural"
        assert meta.modifies_attributes


# =========================================================================== property-based
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8, 15, 16, 31])
def test_power_of_two_is_tight_bound(n):
    out = PowerOfTwoPadding()(_data([], n, x=torch.zeros((n, 2))))
    t = out.num_nodes
    assert t >= n
    assert (t & (t - 1)) == 0  # t is a power of two
    assert t < 2 * n or n == 1  # tight: no over-padding beyond the next power


@pytest.mark.parametrize("target", [3, 4, 8, 12])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_virtual_masking_edge_count_invariant(target, k):
    data = _p3()
    base_e = data.edge_index.size(1)
    out = VirtualNodeMasking(num_virtual=k)(data)
    assert out.num_nodes == 3 + k
    assert out.edge_index.size(1) == base_e + k * 3 * 2  # k virtuals x 3 reals x 2 dirs


# =========================================================================== discovery
def test_plugin_discovery_registers_all_five():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=True)
    assert "static_padding" in names

    info = PluginRegistry.get_plugin_info("static_padding")
    assert info is not None
    assert info["declared_count"] == 5
    declared = {
        "StaticBatchPadding",
        "VirtualNodeMasking",
        "PowerOfTwoPadding",
        "LayerPreprocess",
        "EdgeAttrCanonicalPadding",
    }
    assert declared.issubset(set(info["registered_transforms"]))
    assert info["missing_implementations"] == []


def test_plugin_discovery_is_idempotent():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    assert "static_padding" in names
    info = PluginRegistry.get_plugin_info("static_padding")
    registered = info["registered_transforms"]
    assert len(registered) == len(set(registered))


# =============================================================================
# Property-based invariants over seeded random graphs (no external dep).
# =============================================================================
_RANDOM = [(n, p, s) for n in (4, 6, 9) for p in (0.3, 0.6) for s in (0, 1)]


def _random_undirected(n, p, seed):
    rng = random.Random(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    x = torch.arange(n * 2, dtype=torch.float).view(n, 2)
    return edges, x, _data(edges, n, x=x)


def _gcn_dense_reference(edges, n):
    a = np.zeros((n, n), dtype=float)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    a_hat = a + np.eye(n)
    deg = a_hat.sum(axis=1)
    d_inv = 1.0 / np.sqrt(deg)
    return d_inv[:, None] * a_hat * d_inv[None, :]


@pytest.mark.parametrize("n, p, seed", _RANDOM)
def test_static_padding_preserves_reals_random(n, p, seed):
    edges, x, data = _random_undirected(n, p, seed)
    e = data.edge_index.size(1)
    out = StaticBatchPadding(n_node=n + 3, n_edge=e + 2)(data)
    assert out.num_nodes == n + 3
    assert torch.equal(out.x[:n], x)  # real features untouched
    assert torch.equal(out.x[n:], torch.zeros((3, 2)))  # padding zeroed
    assert out.node_mask.sum().item() == n
    assert out.edge_mask.sum().item() == e


@pytest.mark.parametrize("n, p, seed", _RANDOM)
def test_virtual_masking_preserves_reals_random(n, p, seed):
    edges, x, data = _random_undirected(n, p, seed)
    e = data.edge_index.size(1)
    out = VirtualNodeMasking(num_virtual=2)(data)
    assert out.num_nodes == n + 2
    assert torch.equal(out.x[:n], x)  # real features untouched
    assert torch.equal(out.x[n:], torch.zeros((2, 2)))  # virtual features zero
    assert torch.equal(out.edge_index[:, :e], data.edge_index)  # real edges preserved as prefix
    for virtual in (n, n + 1):
        assert (out.edge_index[0] == virtual).sum().item() == n  # connects to every real node


@pytest.mark.parametrize("n, p, seed", _RANDOM)
def test_pow2_preserves_reals_random(n, p, seed):
    edges, x, data = _random_undirected(n, p, seed)
    out = PowerOfTwoPadding()(data)
    t = out.num_nodes
    assert (t & (t - 1)) == 0 and t >= n
    assert torch.equal(out.x[:n], x)
    assert out.node_mask.sum().item() == n


@pytest.mark.parametrize("n, p, seed", _RANDOM)
def test_layer_preprocess_matches_dense_reference_random(n, p, seed):
    edges, _x, data = _random_undirected(n, p, seed)
    out = LayerPreprocess()(data)
    reference = _gcn_dense_reference(edges, n)
    wmap = {tuple(e): w for e, w in _iter_edge_weights(out)}
    # every emitted edge weight matches the closed-form symmetric normalization
    for (i, j), w in wmap.items():
        assert w == pytest.approx(reference[i, j], abs=1e-5)
    # every nonzero entry of the dense operator is represented in the output
    nnz = int((np.abs(reference) > 1e-12).sum())
    assert out.edge_index.size(1) == nnz


@pytest.mark.parametrize("n, p, seed", _RANDOM)
def test_layer_preprocess_has_one_self_loop_per_node_random(n, p, seed):
    _edges, _x, data = _random_undirected(n, p, seed)
    out = LayerPreprocess()(data)
    self_loops = [(u, v) for u, v in out.edge_index.t().tolist() if u == v]
    assert sorted(self_loops) == [(i, i) for i in range(n)]


@pytest.mark.parametrize("width", [2, 3, 5, 8])
def test_edge_attr_padding_preserves_and_zeros_random(width):
    ea = torch.randn(5, 2)
    out = EdgeAttrCanonicalPadding(width=width)(_data([(0, 1), (1, 2)], 3, edge_attr=ea))
    assert out.edge_attr.shape == (5, width)
    assert torch.equal(out.edge_attr[:, :2], ea)
    if width > 2:
        assert torch.equal(out.edge_attr[:, 2:], torch.zeros((5, width - 2)))


# =============================================================================
# Idempotence.
# =============================================================================
def test_pow2_padding_idempotent():
    once = PowerOfTwoPadding()(_data([], 5, x=torch.zeros((5, 2))))
    twice = PowerOfTwoPadding()(once)
    assert twice.num_nodes == once.num_nodes == 8


def _rounded_weights(out):
    return {tuple(e): round(w, 6) for e, w in _iter_edge_weights(out)}


def test_layer_preprocess_idempotent_weights():
    once = LayerPreprocess()(_p3())
    twice = LayerPreprocess()(once)
    assert _rounded_weights(once) == _rounded_weights(twice)


def test_edge_attr_padding_idempotent():
    d = _data([(0, 1)], 2, edge_attr=torch.ones((2, 2)))
    once = EdgeAttrCanonicalPadding(width=4)(d)
    twice = EdgeAttrCanonicalPadding(width=4)(once)
    assert torch.equal(once.edge_attr, twice.edge_attr)


# =============================================================================
# Boundary graphs.
# =============================================================================
def test_pow2_single_node():
    out = PowerOfTwoPadding()(_data([], 1, x=torch.zeros((1, 2))))
    assert out.num_nodes == 1
    assert out.node_mask.tolist() == [True]


def test_virtual_masking_without_features():
    out = VirtualNodeMasking(num_virtual=1)(_data([(0, 1)], 2))  # no x
    assert out.x is None
    assert out.num_nodes == 3
    assert out.node_mask.tolist() == [True, True, False]


def test_static_padding_without_features():
    out = StaticBatchPadding(n_node=4)(_data([(0, 1)], 2))  # no x
    assert out.x is None
    assert out.num_nodes == 4
    assert out.node_mask.sum().item() == 2


def test_layer_preprocess_edgeless_graph():
    out = LayerPreprocess()(_data([], 3, x=torch.zeros((3, 2))))
    # only the 3 self-loops remain; each weight = 1 (isolated node, d~=1)
    assert out.edge_index.size(1) == 3
    assert torch.allclose(out.edge_weight, torch.ones(3), atol=1e-6)


# =============================================================================
# dtype / tensor-type rigor.
# =============================================================================
def test_mask_and_weight_dtypes():
    h1 = StaticBatchPadding(n_node=5, n_edge=6)(_p3())
    assert h1.node_mask.dtype == torch.bool and h1.edge_mask.dtype == torch.bool
    assert h1.edge_index.dtype == torch.long

    h2 = VirtualNodeMasking(num_virtual=1)(_p3())
    assert h2.node_mask.dtype == torch.bool and h2.edge_index.dtype == torch.long

    h4 = LayerPreprocess()(_p3())
    assert h4.edge_weight.dtype == torch.float and h4.edge_index.dtype == torch.long


def test_static_padding_pad_nodes_have_no_real_edges():
    # padding nodes must be isolated except for designated self-loops on the first pad node
    out = StaticBatchPadding(n_node=6, n_edge=6)(_p3())  # 3 real nodes -> pad nodes 3,4,5
    real_incident = [
        (u, v) for u, v in out.edge_index.t().tolist() if (u >= 3 or v >= 3) and u != v
    ]
    assert real_incident == []  # no padding<->real cross edges
