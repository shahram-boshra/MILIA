"""
Production-grade unit + discovery tests for the graph_reduction plugin (Family F.a/F.b).

Ground-truth checks on named graphs (path/cycle/complete/star), edge-subset containment,
determinism + seed-sensitivity, coarsening node-count/aggregation, fail-closed validation,
input immutability, a cross-check of effective resistance against networkx, and plugin
discovery (missing==[]).
"""

from pathlib import Path

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.plugins.graph_reduction.transforms import (
    EffectiveResistanceSparsify,
    HeavyEdgeMatching,
    KCoreDecomposition,
    KNeighborSparsify,
    KronReduction,
    KTrussDecomposition,
    LocalDegreeSparsify,
    SpannerSparsify,
)
from milia_pipeline.transformations.custom_transforms import TransformExecutionError

SPARSIFIERS = [SpannerSparsify, KNeighborSparsify, LocalDegreeSparsify, KTrussDecomposition]
ALL_CLASSES = [
    EffectiveResistanceSparsify,
    SpannerSparsify,
    KNeighborSparsify,
    LocalDegreeSparsify,
    KronReduction,
    HeavyEdgeMatching,
    KCoreDecomposition,
    KTrussDecomposition,
]


# --------------------------------------------------------------------------- helpers
def _data(edges, n, x=None):
    if edges:
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
    return d


def _path(n):
    return _data([(i, i + 1) for i in range(n - 1)], n)


def _cycle(n):
    return _data([(i, (i + 1) % n) for i in range(n)], n)


def _complete(n):
    return _data([(i, j) for i in range(n) for j in range(i + 1, n)], n)


def _star(n):
    return _data([(0, i) for i in range(1, n)], n)


def _weighted(edges, weights, n, x=None):
    """Undirected graph carrying a scalar edge_weight aligned to both directions."""
    src, dst, ew = [], [], []
    for (a, b), w in zip(edges, weights, strict=True):
        src += [a, b]
        dst += [b, a]
        ew += [float(w), float(w)]
    d = Data(edge_index=torch.tensor([src, dst], dtype=torch.long), num_nodes=n)
    d.edge_weight = torch.tensor(ew, dtype=torch.float64)
    if x is not None:
        d.x = x
    return d


def _two_triangles():
    return _data([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], 6)


def _assert_valid_undirected(out):
    """A production graph output: long edge_index [2, E], symmetric, self-loop free,
    and edge_attr / edge_weight aligned to the edge count."""
    ei = out.edge_index
    assert ei.dtype == torch.long
    assert ei.dim() == 2 and ei.size(0) == 2
    pairs = ei.t().tolist()
    fwd = {(u, v) for u, v in pairs}
    bwd = {(v, u) for u, v in pairs}
    assert fwd == bwd, "edge_index is not symmetric (undirected invariant violated)"
    assert all(u != v for u, v in pairs), "self-loop present in output"
    ea = getattr(out, "edge_attr", None)
    if ea is not None:
        assert ea.size(0) == ei.size(1), "edge_attr rows != edge count"
    ew = getattr(out, "edge_weight", None)
    if ew is not None:
        assert ew.numel() == ei.size(1), "edge_weight length != edge count"


def _u_edges(out):
    return {tuple(sorted(e)) for e in out.edge_index.t().tolist() if e[0] != e[1]}


# =========================================================================== metadata
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_metadata_shape(cls):
    meta = cls.get_metadata()
    assert meta.name == cls.__name__
    assert meta.version == "1.0.0"
    assert meta.category == "structural"
    assert meta.author and meta.description
    assert meta.modifies_attributes


# =========================================================================== sparsifiers: subset
@pytest.mark.parametrize("cls", SPARSIFIERS)
def test_sparsifier_edges_are_subset_of_original(cls):
    src = _complete(6)
    original = _u_edges(src)
    out = cls()(src)
    assert _u_edges(out).issubset(original)


def test_effective_resistance_edges_subset_and_reweighted():
    src = _complete(6)
    out = EffectiveResistanceSparsify(ratio=0.5, seed=0)(src)
    assert _u_edges(out).issubset(_u_edges(src))
    assert getattr(out, "edge_weight", None) is not None
    assert out.edge_weight.numel() == out.edge_index.size(1)


# =========================================================================== F1 determinism
def test_effective_resistance_deterministic_and_seed_sensitive():
    a = EffectiveResistanceSparsify(ratio=0.5, seed=1)(_complete(8))
    b = EffectiveResistanceSparsify(ratio=0.5, seed=1)(_complete(8))
    c = EffectiveResistanceSparsify(ratio=0.5, seed=2)(_complete(8))
    assert _u_edges(a) == _u_edges(b)
    assert _u_edges(a) != _u_edges(c) or not _u_edges(a)


def test_effective_resistance_matches_networkx_on_cycle():
    # Cross-check R_e computed via pinv against nx.resistance_distance on a 5-cycle.
    data = _cycle(5)
    t = EffectiveResistanceSparsify()
    t._metadata = t.get_metadata()
    lap = t._laplacian(data, 5)
    lpinv = torch.linalg.pinv(lap)
    r_uv = float(lpinv[0, 0] + lpinv[1, 1] - 2 * lpinv[0, 1])
    g = nx.cycle_graph(5)
    r_nx = float(nx.resistance_distance(g, 0, 1))
    assert abs(r_uv - r_nx) < 1e-6


# =========================================================================== F3 spanner
def test_spanner_preserves_connectivity_and_is_subset():
    src = _complete(7)
    out = SpannerSparsify(stretch=3, seed=0)(src)
    assert _u_edges(out).issubset(_u_edges(src))
    g = nx.Graph()
    g.add_nodes_from(range(7))
    g.add_edges_from(_u_edges(out))
    assert nx.is_connected(g)


def test_spanner_deterministic_with_seed():
    a = SpannerSparsify(stretch=3, seed=5)(_complete(8))
    b = SpannerSparsify(stretch=3, seed=5)(_complete(8))
    assert _u_edges(a) == _u_edges(b)


def test_spanner_bad_stretch_fails_closed():
    with pytest.raises(TransformExecutionError):
        SpannerSparsify(stretch=0)(_complete(5))


# =========================================================================== F5 KNN
def test_kneighbor_caps_degree_on_star():
    # Star centre has degree n-1; leaves degree 1. k=1 keeps every leaf's single edge.
    out = KNeighborSparsify(k=1)(_star(6))
    assert _u_edges(out) == {(0, i) for i in range(1, 6)}


def test_kneighbor_k_ge_maxdeg_is_noop():
    src = _cycle(6)
    out = KNeighborSparsify(k=10)(src)
    assert _u_edges(out) == _u_edges(src)


def test_kneighbor_bad_k_fails_closed():
    with pytest.raises(TransformExecutionError):
        KNeighborSparsify(k=0)(_cycle(5))


# =========================================================================== F6 local degree
def test_local_degree_subset_and_hub_preserved_on_star():
    # Every leaf (deg 1) keeps its edge to the hub -> all star edges retained.
    out = LocalDegreeSparsify(alpha=0.5)(_star(7))
    assert _u_edges(out) == {(0, i) for i in range(1, 7)}


def test_local_degree_bad_alpha_fails_closed():
    with pytest.raises(TransformExecutionError):
        LocalDegreeSparsify(alpha=1.5)(_cycle(5))


# =========================================================================== F8 Kron
def test_kron_reduces_node_count():
    out = KronReduction(ratio=0.5)(_cycle(8))
    assert out.num_nodes == 4
    assert out.edge_index.size(0) == 2


def test_kron_reduced_is_valid_laplacian_structure():
    # Reduced edge weights must be non-negative (off-diagonal -L_red >= 0).
    out = KronReduction(ratio=0.5)(_complete(6))
    assert getattr(out, "edge_weight", None) is not None
    assert torch.all(out.edge_weight >= -1e-6)


def test_kron_preserves_effective_resistance_between_retained():
    # Kron reduction preserves effective resistance among retained nodes (Dörfler-Bullo).
    data = _cycle(6)
    t = KronReduction()
    t._metadata = t.get_metadata()
    full_lap = t._laplacian(data, 6)
    full_pinv = torch.linalg.pinv(full_lap)
    # retained set = top-degree (cycle is regular -> lowest indices 0,1,2)
    out = KronReduction(ratio=0.5)(data)
    red_lap = t._laplacian(out, out.num_nodes)
    red_pinv = torch.linalg.pinv(red_lap)
    r_full = float(full_pinv[0, 0] + full_pinv[1, 1] - 2 * full_pinv[0, 1])
    r_red = float(red_pinv[0, 0] + red_pinv[1, 1] - 2 * red_pinv[0, 1])
    assert abs(r_full - r_red) < 1e-4


def test_kron_bad_ratio_fails_closed():
    with pytest.raises(TransformExecutionError):
        KronReduction(ratio=1.0)(_cycle(6))


# =========================================================================== F11 heavy-edge matching
def test_heavy_edge_matching_halves_even_path_nodes():
    # Path of 6 nodes: perfect matching contracts to 3 super-nodes.
    out = HeavyEdgeMatching()(_path(6))
    assert out.num_nodes == 3


def test_heavy_edge_matching_sums_features():
    src = _path(4)
    src.x = torch.ones((4, 2))
    out = HeavyEdgeMatching()(src)
    assert out.x.size(0) == out.num_nodes
    assert float(out.x.sum()) == 8.0  # feature mass conserved


def test_heavy_edge_matching_deterministic():
    a = HeavyEdgeMatching()(_cycle(8))
    b = HeavyEdgeMatching()(_cycle(8))
    assert _u_edges(a) == _u_edges(b)
    assert a.num_nodes == b.num_nodes


# =========================================================================== F14 k-core
def test_kcore_removes_low_degree_nodes():
    # Path has all degrees <= 2; 2-core of a path is empty, of a cycle is itself.
    out_cycle = KCoreDecomposition(k=2)(_cycle(6))
    assert out_cycle.num_nodes == 6
    out_path = KCoreDecomposition(k=2)(_path(6))
    assert out_path.num_nodes == 0


def test_kcore_complete_graph_is_itself():
    out = KCoreDecomposition(k=3)(_complete(5))  # K5 is 4-core -> 3-core is all
    assert out.num_nodes == 5


def test_kcore_bad_k_fails_closed():
    with pytest.raises(TransformExecutionError):
        KCoreDecomposition(k=0)(_cycle(5))


# =========================================================================== F15 k-truss
def test_ktruss_keeps_triangles_drops_bridge():
    # Two triangles (0,1,2) and (3,4,5) joined by a bridge edge (2,3).
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    out = KTrussDecomposition(k=3)(_data(edges, 6))
    kept = _u_edges(out)
    assert (2, 3) not in kept  # bridge in 0 triangles -> removed
    assert (0, 1) in kept and (3, 4) in kept  # triangle edges retained


def test_ktruss_bad_k_fails_closed():
    with pytest.raises(TransformExecutionError):
        KTrussDecomposition(k=2)(_complete(5))


# =========================================================================== robustness
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_no_mutation_of_input(cls):
    src = _complete(6)
    ei = src.edge_index.clone()
    cls()(src)
    assert torch.equal(src.edge_index, ei)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_tiny_graph_does_not_crash(cls):
    out = cls()(_data([(0, 1)], 2))
    assert out.edge_index.size(0) == 2


# =========================================================================== output validity
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_output_is_valid_undirected_graph(cls):
    # Weighted complete graph exercises weight-aware paths and structural output invariants.
    src = _complete(6)
    src.edge_attr = torch.arange(src.edge_index.size(1) * 2, dtype=torch.float).reshape(-1, 2)
    out = cls()(src)
    _assert_valid_undirected(out)


DETERMINISTIC_CLASSES = [
    KNeighborSparsify,
    LocalDegreeSparsify,
    KronReduction,
    HeavyEdgeMatching,
    KCoreDecomposition,
    KTrussDecomposition,
]


@pytest.mark.parametrize("cls", DETERMINISTIC_CLASSES)
def test_deterministic_repeat(cls):
    # Parameter-free / unseeded-deterministic transforms: identical structure, twice.
    # (F1/F3 are stochastic; their determinism-under-seed is covered separately.)
    src = _complete(7)
    a = cls()(src.clone())
    b = cls()(src.clone())
    assert torch.equal(a.edge_index, b.edge_index)
    assert a.num_nodes == b.num_nodes


# =========================================================================== edge_attr carry
def test_sparsify_carries_edge_attr_aligned():
    # Realistic undirected graphs carry SYMMETRIC edge_attr (both directions of an edge share
    # its value, as produced by to_undirected). Verify each kept directed edge keeps the
    # attr row of its undirected edge.
    src = _cycle(6)
    ei = src.edge_index
    src.edge_attr = torch.tensor(
        [[min(u, v) * 10.0 + max(u, v)] for u, v in ei.t().tolist()], dtype=torch.float
    )
    lookup = {tuple(sorted((u, v))): src.edge_attr[i] for i, (u, v) in enumerate(ei.t().tolist())}
    out = KNeighborSparsify(k=1)(src)
    for i, (u, v) in enumerate(out.edge_index.t().tolist()):
        assert torch.equal(out.edge_attr[i], lookup[tuple(sorted((u, v)))])


def test_kcore_carries_edge_attr_via_subgraph():
    src = _cycle(6)
    src.edge_attr = torch.ones((src.edge_index.size(1), 3))
    out = KCoreDecomposition(k=2)(src)  # cycle is its own 2-core
    assert out.edge_attr.size(0) == out.edge_index.size(1)
    assert out.edge_attr.size(1) == 3


# =========================================================================== weight awareness
def test_kneighbor_prefers_heavier_edge():
    # Triangle with (0,2) heaviest; k=1 must keep (0,2) and drop the light (1,2).
    src = _weighted([(0, 1), (1, 2), (0, 2)], [1.0, 1.0, 5.0], 3)
    kept = _u_edges(KNeighborSparsify(k=1)(src))
    assert (0, 2) in kept
    assert (1, 2) not in kept


def test_heavy_edge_matching_contracts_heaviest_first():
    # Path 0-1-2-3 with heavy middle edge -> match (1,2) first -> 3 super-nodes
    # (an unweighted path4 would yield 2). Proves weight-driven matching.
    src = _weighted([(0, 1), (1, 2), (2, 3)], [1.0, 5.0, 1.0], 4)
    assert HeavyEdgeMatching()(src).num_nodes == 3
    assert HeavyEdgeMatching()(_path(4)).num_nodes == 2


# =========================================================================== disconnected graphs
def test_effective_resistance_on_disconnected_graph():
    # pinv of a disconnected Laplacian is well-defined; output must stay valid.
    out = EffectiveResistanceSparsify(ratio=0.6, seed=0)(_two_triangles())
    _assert_valid_undirected(out)
    assert _u_edges(out).issubset(_u_edges(_two_triangles()))


def test_kron_on_disconnected_graph_uses_pinv_fallback():
    # Retained {0,1,2} spans one component; L[beta,beta] is singular -> pinv branch.
    # With no cross-component edges the reduced graph is the intact first triangle.
    out = KronReduction(ratio=0.5)(_two_triangles())
    _assert_valid_undirected(out)
    assert out.num_nodes == 3
    assert _u_edges(out) == {(0, 1), (0, 2), (1, 2)}


def test_kcore_on_disconnected_graph():
    out = KCoreDecomposition(k=2)(_two_triangles())
    assert out.num_nodes == 6  # both triangles are 2-cores


# =========================================================================== property-based fuzz
@pytest.mark.parametrize("seed", list(range(6)))
def test_random_weighted_graphs_stay_valid_and_deterministic(seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 14))
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.35]
    if not edges:
        edges = [(0, 1)]
    weights = [float(rng.uniform(0.1, 5.0)) for _ in edges]
    base = _weighted(edges, weights, n, x=torch.tensor(rng.random((n, 4)), dtype=torch.float))
    base.edge_attr = torch.ones((base.edge_index.size(1), 2))

    for transform in (
        EffectiveResistanceSparsify(ratio=0.6, seed=seed),
        SpannerSparsify(stretch=3, seed=seed),
        KNeighborSparsify(k=3),
        LocalDegreeSparsify(alpha=0.6),
        KronReduction(ratio=0.5),
        HeavyEdgeMatching(),
        KCoreDecomposition(k=2),
        KTrussDecomposition(k=3),
    ):
        first = transform(base.clone())
        second = transform(base.clone())
        _assert_valid_undirected(first)
        assert torch.equal(first.edge_index, second.edge_index)


# =========================================================================== discovery
def test_plugin_discovery_registers_all_eight():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=True)
    assert "graph_reduction" in names

    info = PluginRegistry.get_plugin_info("graph_reduction")
    assert info is not None
    assert info["declared_count"] == 8
    declared = {
        "EffectiveResistanceSparsify",
        "SpannerSparsify",
        "KNeighborSparsify",
        "LocalDegreeSparsify",
        "KronReduction",
        "HeavyEdgeMatching",
        "KCoreDecomposition",
        "KTrussDecomposition",
    }
    assert declared.issubset(set(info["registered_transforms"]))
    assert info["missing_implementations"] == []


def test_plugin_discovery_is_idempotent():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    assert "graph_reduction" in names
    info = PluginRegistry.get_plugin_info("graph_reduction")
    registered = info["registered_transforms"]
    assert len(registered) == len(set(registered))
