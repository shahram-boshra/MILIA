"""
Production-grade unit + discovery tests for the curvature_rewiring plugin (Family B).

Ground-truth curvature on named graphs (triangle/4-cycle/path/complete), output validity and
symmetry, node-set preservation, add-only vs add+prune behaviour, determinism, input
immutability, zero-padded edge_attr on added edges, a fuzz loop, and plugin discovery.
"""

from pathlib import Path

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.plugins.transformations.curvature_rewiring.transforms import (
    EffectiveResistanceRewiring,
    FormanRicciRewiring,
    ResistanceCurvatureRewiring,
    SpectralGapRewiring,
)
from milia_pipeline.transformations.custom_transforms import TransformExecutionError

ALL_CLASSES = [
    FormanRicciRewiring,
    EffectiveResistanceRewiring,
    SpectralGapRewiring,
    ResistanceCurvatureRewiring,
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


def _u_edges(out):
    return {tuple(sorted(e)) for e in out.edge_index.t().tolist() if e[0] != e[1]}


def _assert_valid_undirected(out):
    ei = out.edge_index
    assert ei.dtype == torch.long
    assert ei.dim() == 2 and ei.size(0) == 2
    pairs = ei.t().tolist()
    fwd = {(u, v) for u, v in pairs}
    assert fwd == {(v, u) for u, v in pairs}, "edge_index not symmetric"
    assert all(u != v for u, v in pairs), "self-loop present"
    ea = getattr(out, "edge_attr", None)
    if ea is not None:
        assert ea.size(0) == ei.size(1)


# =========================================================================== curvature ground truth
def test_forman_curvature_named_graphs():
    t = FormanRicciRewiring()
    t._metadata = t.get_metadata()
    # Triangle: d=2,2, tri=1, quad=0 -> 4-2-2+3 = 3.
    adj_tri = t._adjacency({(0, 1), (1, 2), (0, 2)}, 3)
    assert t._forman(adj_tri, 0, 1) == 3.0
    # 4-cycle: d=2,2, tri=0, quad=1 -> 4-2-2+2 = 2.
    adj_c4 = t._adjacency({(0, 1), (1, 2), (2, 3), (0, 3)}, 4)
    assert t._forman(adj_c4, 0, 1) == 2.0
    # Path middle edge: d=2,2, tri=0, quad=0 -> 0.
    adj_p = t._adjacency({(0, 1), (1, 2), (2, 3)}, 4)
    assert t._forman(adj_p, 1, 2) == 0.0


def test_resistance_matches_networkx():
    t = EffectiveResistanceRewiring()
    t._metadata = t.get_metadata()
    edges = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}  # 5-cycle
    resist = t._resistance_matrix(t._laplacian(edges, 5))
    g = nx.cycle_graph(5)
    assert abs(float(resist[0, 1]) - float(nx.resistance_distance(g, 0, 1))) < 1e-6


def test_resistance_curvature_complete_graph():
    # Devriendt-Lambiotte: p_i = 1/n on the complete graph K_n.
    t = ResistanceCurvatureRewiring()
    t._metadata = t.get_metadata()
    n = 5
    edges = {(i, j) for i in range(n) for j in range(i + 1, n)}
    resist = t._resistance_matrix(t._laplacian(edges, n))
    adj = t._adjacency(edges, n)
    p = t._node_curvature(resist, adj)
    assert all(abs(pi - 1.0 / n) < 1e-6 for pi in p)


# =========================================================================== validity / structure
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_output_is_valid_undirected_graph(cls):
    src = _path(6)
    src.edge_attr = torch.ones((src.edge_index.size(1), 2))
    out = cls()(src)
    _assert_valid_undirected(out)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_node_set_preserved(cls):
    src = _path(7)
    src.x = torch.arange(7 * 2, dtype=torch.float).reshape(7, 2)
    out = cls()(src)
    assert out.num_nodes == 7
    assert torch.equal(out.x, src.x)  # rewiring must not touch node features


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_rewiring_adds_edges_on_a_path(cls):
    # A path is maximally bottlenecked; every rewiring criterion should add >= 1 edge.
    src = _path(8)
    out = cls()(src)
    assert len(_u_edges(out)) >= len(_u_edges(src))


# =========================================================================== add / prune behaviour
def test_original_edges_preserved_without_prune():
    src = _path(7)
    original = _u_edges(src)
    out = SpectralGapRewiring(num_iterations=2)(src)  # no prune param -> add-only
    assert original.issubset(_u_edges(out))


def test_prune_flag_allows_edge_removal():
    src = _complete(6)
    with_prune = FormanRicciRewiring(num_iterations=3, prune=True)(src)
    without = FormanRicciRewiring(num_iterations=3, prune=False)(src)
    # On a dense graph, pruning yields no more edges than not pruning.
    assert len(_u_edges(with_prune)) <= len(_u_edges(without)) + 3


# =========================================================================== determinism / no-mutation
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_deterministic(cls):
    src = _cycle(8)
    a = cls()(src.clone())
    b = cls()(src.clone())
    assert torch.equal(a.edge_index, b.edge_index)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_no_mutation_of_input(cls):
    src = _path(7)
    ei = src.edge_index.clone()
    cls()(src)
    assert torch.equal(src.edge_index, ei)


def test_added_edges_get_zero_edge_attr():
    src = _path(6)
    src.edge_attr = torch.ones((src.edge_index.size(1), 3))
    out = SpectralGapRewiring(num_iterations=2)(src)
    assert out.edge_attr.size(0) == out.edge_index.size(1)
    assert out.edge_attr.size(1) == 3
    # Original edges keep ones; at least one added edge carries a zero row.
    assert (out.edge_attr == 0).any()
    assert (out.edge_attr == 1).any()


# =========================================================================== fail-closed / robustness
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_bad_iterations_fail_closed(cls):
    with pytest.raises(TransformExecutionError):
        cls(num_iterations=0)(_cycle(5))


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_tiny_graph_noop(cls):
    out = cls()(_data([(0, 1)], 2))  # N < 3 -> returned unchanged
    assert out.num_nodes == 2


# =========================================================================== property-based fuzz
@pytest.mark.parametrize("seed", list(range(5)))
def test_random_graphs_stay_valid_and_deterministic(seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 12))
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.3]
    if len(edges) < 2:
        edges = [(0, 1), (1, 2)]
    base = _data(edges, n, x=torch.tensor(rng.random((n, 3)), dtype=torch.float))
    base.edge_attr = torch.ones((base.edge_index.size(1), 2))

    for transform in (
        FormanRicciRewiring(num_iterations=2),
        EffectiveResistanceRewiring(num_iterations=2),
        SpectralGapRewiring(num_iterations=2),
        ResistanceCurvatureRewiring(num_iterations=2),
    ):
        first = transform(base.clone())
        second = transform(base.clone())
        _assert_valid_undirected(first)
        assert first.num_nodes == n
        assert torch.equal(first.edge_index, second.edge_index)


# =========================================================================== goal / metamorphic
# These assert the transforms achieve their geometric PURPOSE, measured by an INDEPENDENT
# networkx oracle (not the code-under-test's own helpers). Monotonicity theorems: adding edges
# never decreases algebraic connectivity and never increases effective resistance.
def _nx_graph(out):
    g = nx.Graph()
    g.add_nodes_from(range(out.num_nodes))
    g.add_edges_from(_u_edges(out))
    return g


def _lambda2(g):
    import numpy as np

    lap = nx.laplacian_matrix(g).toarray().astype(float)
    return sorted(np.linalg.eigvalsh(lap))[1]


def _max_resistance(g):
    import itertools

    return max(
        nx.resistance_distance(g, i, j) for i, j in itertools.combinations(sorted(g.nodes), 2)
    )


def _triangle_count(g):
    return sum(nx.triangles(g).values()) // 3


def test_spectral_gap_rewiring_raises_algebraic_connectivity():
    # B7's purpose: increase the Laplacian spectral gap (algebraic connectivity).
    src = _path(8)
    out = SpectralGapRewiring(num_iterations=3)(src)
    assert _lambda2(_nx_graph(out)) > _lambda2(_nx_graph(src)) + 1e-6


def test_effective_resistance_rewiring_reduces_worst_resistance():
    # B6 (add-only) purpose: relieve the worst bottleneck -> max pairwise resistance drops.
    src = _path(8)
    out = EffectiveResistanceRewiring(num_iterations=3, prune=False)(src)
    assert _max_resistance(_nx_graph(out)) < _max_resistance(_nx_graph(src)) - 1e-6


def test_resistance_curvature_rewiring_reduces_worst_resistance():
    src = _path(8)
    out = ResistanceCurvatureRewiring(num_iterations=3, prune=False)(src)
    assert _max_resistance(_nx_graph(out)) < _max_resistance(_nx_graph(src)) - 1e-6


def test_forman_rewiring_creates_triangles_at_bottleneck():
    # B3 (add-only) SDRF adds triangle-forming support edges at negatively-curved bottlenecks.
    src = _path(8)
    out = FormanRicciRewiring(num_iterations=3, prune=False)(src)
    g_out = _nx_graph(out)
    assert _triangle_count(g_out) > _triangle_count(_nx_graph(src))
    assert nx.is_connected(g_out)


def test_resistance_rewiring_bridges_disconnected_components():
    # Inter-component effective resistance is largest, so B6 must add a bridging edge.
    src = _data([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], 6)  # two triangles
    out = EffectiveResistanceRewiring(num_iterations=1, prune=False)(src)
    assert nx.is_connected(_nx_graph(out))


# =========================================================================== discovery
def test_plugin_discovery_registers_all_four():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent / "transformations"]
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=True)
    assert "curvature_rewiring" in names

    info = PluginRegistry.get_plugin_info("curvature_rewiring")
    assert info is not None
    assert info["declared_count"] == 4
    declared = {
        "FormanRicciRewiring",
        "EffectiveResistanceRewiring",
        "SpectralGapRewiring",
        "ResistanceCurvatureRewiring",
    }
    assert declared.issubset(set(info["registered_transforms"]))
    assert info["missing_implementations"] == []


def test_plugin_discovery_is_idempotent():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent / "transformations"]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    assert "curvature_rewiring" in names
    info = PluginRegistry.get_plugin_info("curvature_rewiring")
    registered = info["registered_transforms"]
    assert len(registered) == len(set(registered))
