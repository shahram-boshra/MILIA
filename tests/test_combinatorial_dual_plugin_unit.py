"""
Production-grade unit + discovery tests for the combinatorial_dual plugin (Family D).

Every transform is checked against hand-computable ground truth (edge sets / counts on
small named graphs), plus fail-closed prerequisites, dependency-injection branches,
determinism, input immutability, parameter validation, and end-to-end discovery.
"""

from pathlib import Path

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.plugins.transformations.combinatorial_dual.transforms import (
    BipartiteProjection,
    ComplementGraph,
    CondensationTransform,
    GabrielGraph,
    GraphDual,
    KHopGraph,
    KNeighborGraphPower,
    MinimumSpanningTree,
    QuotientGraph,
    RelativeNeighborhoodGraph,
    ToLevi,
    TransitiveClosure,
    TransitiveReduction,
)
from milia_pipeline.transformations.custom_transforms import TransformExecutionError

ALL_UNDIRECTED = [
    ToLevi,
    QuotientGraph,
    BipartiteProjection,
    KHopGraph,
    ComplementGraph,
    KNeighborGraphPower,
    MinimumSpanningTree,
]


# --------------------------------------------------------------------------- helpers
def _data(edges, n, x=None, directed=False, pos=None):
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
    if pos is not None:
        d.pos = pos
    return d


def _path(n):
    return _data([(i, i + 1) for i in range(n - 1)], n)


def _cycle(n):
    return _data([(i, (i + 1) % n) for i in range(n)], n)


def _complete(n):
    return _data([(i, j) for i in range(n) for j in range(i + 1, n)], n)


def _u_edges(out):
    return out.edge_index.size(1) // 2


def _d_edges(out):
    return out.edge_index.size(1)


def _edge_set_undirected(out):
    return {tuple(sorted(e)) for e in out.edge_index.t().tolist()}


def _edge_set_directed(out):
    return {tuple(e) for e in out.edge_index.t().tolist()}


# =========================================================================== D9 Complement
def test_complement_of_path3():
    out = ComplementGraph()(_path(3))  # P3 edges {01,12} -> complement {02}
    assert out.num_nodes == 3
    assert _edge_set_undirected(out) == {(0, 2)}


def test_complement_of_c4():
    out = ComplementGraph()(_cycle(4))  # C4 -> two diagonals
    assert _edge_set_undirected(out) == {(0, 2), (1, 3)}


def test_complement_of_complete_is_empty():
    out = ComplementGraph()(_complete(4))
    assert _u_edges(out) == 0


def test_complement_carries_features():
    x = torch.arange(6.0).view(3, 2)
    out = ComplementGraph()(_data([(0, 1), (1, 2)], 3, x=x))
    assert out.x is not None and torch.equal(out.x, x)


# =========================================================================== D11 MST
def test_mst_of_cycle_is_tree():
    out = MinimumSpanningTree()(_cycle(4))
    assert _u_edges(out) == 3  # |V| - 1


def test_mst_of_tree_is_same_edge_count():
    out = MinimumSpanningTree()(_path(5))
    assert _u_edges(out) == 4


def test_mst_of_complete_is_spanning_tree():
    out = MinimumSpanningTree()(_complete(5))
    assert _u_edges(out) == 4


# =========================================================================== D8 KHop
def test_khop_p5_k2_ground_truth():
    out = KHopGraph(k=2)(_path(5))  # 0-1-2-3-4
    assert _edge_set_undirected(out) == {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # 1-hop
        (0, 2),
        (1, 3),
        (2, 4),  # 2-hop
    }


def test_khop_k1_is_original():
    out = KHopGraph(k=1)(_path(4))
    assert _edge_set_undirected(out) == {(0, 1), (1, 2), (2, 3)}


def test_khop_k3_on_p4_is_complete():
    out = KHopGraph(k=3)(_path(4))
    assert _u_edges(out) == 6  # all pairs reachable within 3 hops


def test_khop_invalid_k_fails_closed():
    with pytest.raises(TransformExecutionError):
        KHopGraph(k=0)(_path(4))


# =========================================================================== D10 A^k
def test_adjacency_power_p4_k2():
    out = KNeighborGraphPower(k=2)(_path(4))  # length-2 walks: (0,2),(1,3)
    assert _edge_set_undirected(out) == {(0, 2), (1, 3)}


def test_adjacency_power_c4_k2():
    out = KNeighborGraphPower(k=2)(_cycle(4))  # opposite corners reachable in 2
    assert _edge_set_undirected(out) == {(0, 2), (1, 3)}


def test_adjacency_power_invalid_k_fails_closed():
    with pytest.raises(TransformExecutionError):
        KNeighborGraphPower(k=0)(_path(4))


# =========================================================================== D1 ToLevi
def test_tolevi_counts_and_bipartite():
    out = ToLevi()(_path(3))  # 3 vertices + 2 edges = 5 nodes; 2*2 incidences = 4 edges
    assert out.num_nodes == 5
    assert _u_edges(out) == 4
    # every edge-node (indices 3,4) has degree 2
    deg = torch.bincount(out.edge_index[0], minlength=5).tolist()
    assert deg[3] == 2 and deg[4] == 2


def test_tolevi_feature_padding():
    x = torch.ones((3, 2))
    out = ToLevi()(_data([(0, 1), (1, 2)], 3, x=x))
    assert out.x.shape == (5, 2)
    assert torch.equal(out.x[:3], x)
    assert torch.equal(out.x[3:], torch.zeros((2, 2)))  # edge-nodes zero-padded


# =========================================================================== D5 TransitiveClosure
def test_transitive_closure_undirected_component_becomes_clique():
    out = TransitiveClosure()(_path(3))  # bidirectional P3 is strongly connected
    edges = _edge_set_directed(out)
    cross = {(u, v) for (u, v) in edges if u != v}
    # every ordered pair among the 3 mutually-reachable nodes
    assert cross == {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
    # nodes on a cycle are reachable from themselves -> self-loops in the true closure
    assert all((i, i) in edges for i in range(3))
    assert _d_edges(out) == 9  # 6 ordered pairs + 3 cycle self-loops


def test_transitive_closure_directed_path():
    # directed 0->1->2 (asymmetric edge_index)
    out = TransitiveClosure()(_data([(0, 1), (1, 2)], 3, directed=True))
    assert _edge_set_directed(out) == {(0, 1), (1, 2), (0, 2)}


# =========================================================================== D6 TransitiveReduction
def test_transitive_reduction_removes_implied_edge():
    # DAG 0->1, 1->2, 0->2  =>  reduction drops 0->2
    out = TransitiveReduction()(_data([(0, 1), (1, 2), (0, 2)], 3, directed=True))
    assert _edge_set_directed(out) == {(0, 1), (1, 2)}


def test_transitive_reduction_non_dag_fails_closed():
    # undirected input -> bidirectional -> cycles -> not a DAG
    with pytest.raises(TransformExecutionError):
        TransitiveReduction()(_path(3))


# =========================================================================== D4 Condensation
def test_condensation_of_directed_cycle_is_single_node():
    out = CondensationTransform()(_data([(0, 1), (1, 2), (2, 0)], 3, directed=True))
    assert out.num_nodes == 1
    assert _d_edges(out) == 0


def test_condensation_of_dag_preserves_components():
    out = CondensationTransform()(_data([(0, 1), (1, 2)], 3, directed=True))
    assert out.num_nodes == 3  # three singleton SCCs
    assert _d_edges(out) == 2


# =========================================================================== D3 QuotientGraph
def test_quotient_collapses_blocks():
    out = QuotientGraph(partition=[[0, 1], [2, 3]])(_path(4))  # edge 1-2 crosses blocks
    assert out.num_nodes == 2
    assert _u_edges(out) == 1


def test_quotient_singleton_blocks_identity():
    out = QuotientGraph(partition=[[0], [1], [2]])(_path(3))
    assert out.num_nodes == 3
    assert _u_edges(out) == 2


def test_quotient_partition_from_data_attribute():
    data = _path(4)
    data.partition = [[0, 1], [2, 3]]
    out = QuotientGraph()(data)
    assert out.num_nodes == 2


def test_quotient_missing_partition_fails_closed():
    with pytest.raises(TransformExecutionError):
        QuotientGraph()(_path(4))


# =========================================================================== D7 BipartiteProjection
def _k22():
    # complete bipartite K_{2,2}: A={0,1}, B={2,3}
    return _data([(0, 2), (0, 3), (1, 2), (1, 3)], 4)


def test_bipartite_projection_onto_set_a():
    out = BipartiteProjection(bipartite_nodes=[0, 1])(_k22())
    assert out.num_nodes == 2
    assert _u_edges(out) == 1  # 0,1 share neighbours -> connected


def test_bipartite_projection_from_data_attribute():
    data = _k22()
    data.bipartite_nodes = [2, 3]
    out = BipartiteProjection()(data)
    assert out.num_nodes == 2
    assert _u_edges(out) == 1


def test_bipartite_projection_non_bipartite_fails_closed():
    with pytest.raises(TransformExecutionError):
        BipartiteProjection(bipartite_nodes=[0])(_complete(3))  # triangle is not bipartite


def test_bipartite_projection_missing_nodes_fails_closed():
    with pytest.raises(TransformExecutionError):
        BipartiteProjection()(_k22())


# =========================================================================== D2 GraphDual
def test_graph_dual_of_c4_euler():
    out = GraphDual()(_cycle(4))  # planar; faces F = E - V + 2 = 2
    assert out.num_nodes == 2


def test_graph_dual_of_triangle():
    out = GraphDual()(_complete(3))  # C3 planar; F = 3 - 3 + 2 = 2
    assert out.num_nodes == 2


def test_graph_dual_non_planar_fails_closed():
    with pytest.raises(TransformExecutionError):
        GraphDual()(_complete(5))  # K5 is non-planar


# =========================================================================== D12 proximity
def test_gabriel_collinear_three_points():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = GabrielGraph()(_data([], 3, pos=pos))
    # midpoint blocks the 0-2 edge; chain remains
    assert _edge_set_undirected(out) == {(0, 1), (1, 2)}


def test_rng_collinear_three_points():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = RelativeNeighborhoodGraph()(_data([], 3, pos=pos))
    assert _edge_set_undirected(out) == {(0, 1), (1, 2)}


def test_gabriel_triangle_all_edges():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]])
    out = GabrielGraph()(_data([], 3, pos=pos))
    assert _u_edges(out) == 3


def test_proximity_requires_pos_fails_closed():
    with pytest.raises(TransformExecutionError):
        GabrielGraph()(_path(3))
    with pytest.raises(TransformExecutionError):
        RelativeNeighborhoodGraph()(_path(3))


def test_proximity_preserves_pos_and_features():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]])
    x = torch.arange(6.0).view(3, 2)
    out = GabrielGraph()(_data([], 3, x=x, pos=pos))
    assert out.x is not None and torch.equal(out.x, x)
    assert out.pos is not None and torch.equal(out.pos, pos)


# =========================================================================== cross-cutting
@pytest.mark.parametrize("transform_cls", ALL_UNDIRECTED)
def test_undirected_output_is_symmetric(transform_cls):
    kwargs = {}
    if transform_cls is QuotientGraph:
        kwargs = {"partition": [[0, 1], [2, 3]]}
        data = _path(4)
    elif transform_cls is BipartiteProjection:
        kwargs = {"bipartite_nodes": [0, 1]}
        data = _k22()
    else:
        data = _path(4)
    out = transform_cls(**kwargs)(data)
    edges = {tuple(e) for e in out.edge_index.t().tolist()}
    assert all((v, u) in edges for (u, v) in edges)


def _dpath():
    """Directed path 0->1->2 (a DAG, strongly connected only per edge)."""
    return _data([(0, 1), (1, 2)], 3, directed=True)


def _ddiamond():
    """DAG with a transitively-implied edge: 0->1, 1->2, 0->2."""
    return _data([(0, 1), (1, 2), (0, 2)], 3, directed=True)


def _tri_data():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]])
    return _data([], 3, pos=pos)


CASES = [
    ("tolevi", lambda: ToLevi(), lambda: _path(4)),
    ("dual", lambda: GraphDual(), lambda: _cycle(4)),
    ("quotient", lambda: QuotientGraph(partition=[[0, 1], [2, 3]]), lambda: _path(4)),
    ("condensation", lambda: CondensationTransform(), _dpath),
    ("closure", lambda: TransitiveClosure(), _dpath),
    ("reduction", lambda: TransitiveReduction(), _ddiamond),
    ("projection", lambda: BipartiteProjection(bipartite_nodes=[0, 1]), _k22),
    ("khop", lambda: KHopGraph(k=2), lambda: _path(5)),
    ("complement", lambda: ComplementGraph(), lambda: _cycle(4)),
    ("apow", lambda: KNeighborGraphPower(k=2), lambda: _path(4)),
    ("mst", lambda: MinimumSpanningTree(), lambda: _cycle(4)),
    ("gabriel", lambda: GabrielGraph(), lambda: _tri_data()),
    ("rng", lambda: RelativeNeighborhoodGraph(), lambda: _tri_data()),
]
_CASE_IDS = [c[0] for c in CASES]


@pytest.mark.parametrize("cid, make_t, make_d", CASES, ids=_CASE_IDS)
def test_every_transform_runs_and_returns_data(cid, make_t, make_d):
    out = make_t()(make_d())
    assert isinstance(out, Data)
    assert out.edge_index.dim() == 2 and out.edge_index.size(0) == 2


@pytest.mark.parametrize("cid, make_t, make_d", CASES, ids=_CASE_IDS)
def test_every_transform_is_deterministic(cid, make_t, make_d):
    t = make_t()
    a = t(make_d())
    b = t(make_d())
    assert torch.equal(a.edge_index, b.edge_index)


@pytest.mark.parametrize("cid, make_t, make_d", CASES, ids=_CASE_IDS)
def test_every_transform_no_mutation(cid, make_t, make_d):
    d = make_d()
    ei_before = d.edge_index.clone()
    n_before = d.num_nodes
    make_t()(d)
    assert torch.equal(d.edge_index, ei_before)
    assert d.num_nodes == n_before


def test_all_transforms_declare_structural_metadata():
    classes = ALL_UNDIRECTED + [
        GraphDual,
        CondensationTransform,
        TransitiveClosure,
        TransitiveReduction,
        GabrielGraph,
        RelativeNeighborhoodGraph,
    ]
    for cls in classes:
        meta = cls.get_metadata()
        assert meta.category == "structural"
        assert meta.modifies_attributes


# =========================================================================== discovery
def test_plugin_discovery_registers_all_thirteen():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent / "transformations"]
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=True)
    assert "combinatorial_dual" in names

    info = PluginRegistry.get_plugin_info("combinatorial_dual")
    assert info is not None
    assert info["declared_count"] == 13
    declared = {
        "ToLevi",
        "GraphDual",
        "QuotientGraph",
        "CondensationTransform",
        "TransitiveClosure",
        "TransitiveReduction",
        "BipartiteProjection",
        "KHopGraph",
        "ComplementGraph",
        "KNeighborGraphPower",
        "MinimumSpanningTree",
        "GabrielGraph",
        "RelativeNeighborhoodGraph",
    }
    assert declared.issubset(set(info["registered_transforms"]))
    assert info["missing_implementations"] == []


def test_plugin_discovery_is_idempotent():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent / "transformations"]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    assert "combinatorial_dual" in names
    info = PluginRegistry.get_plugin_info("combinatorial_dual")
    registered = info["registered_transforms"]
    assert len(registered) == len(set(registered))


# =============================================================================
# Property-based invariants over many seeded random graphs (no external dep).
# =============================================================================

_RANDOM_GNP = [(n, p, s) for n in (5, 8, 12) for p in (0.3, 0.6) for s in (0, 1)]


def _data_from_nx(graph):
    graph = nx.convert_node_labels_to_integers(graph)
    return _data(list(graph.edges()), graph.number_of_nodes())


def _is_symmetric_no_selfloops(out):
    edges = {tuple(e) for e in out.edge_index.t().tolist()}
    if any(u == v for u, v in edges):
        return False
    return all((v, u) in edges for (u, v) in edges)


@pytest.mark.parametrize("n, p, seed", _RANDOM_GNP)
def test_complement_involution_random(n, p, seed):
    g = nx.gnp_random_graph(n, p, seed=seed)
    data = _data_from_nx(g)
    comp = ComplementGraph()
    once = comp(data)
    twice = comp(once)
    assert _edge_set_undirected(twice) == _edge_set_undirected(data)


@pytest.mark.parametrize("n, p, seed", _RANDOM_GNP)
def test_complement_edge_complementarity_random(n, p, seed):
    g = nx.gnp_random_graph(n, p, seed=seed)
    data = _data_from_nx(g)
    out = ComplementGraph()(data)
    assert _u_edges(out) + g.number_of_edges() == n * (n - 1) // 2
    assert _is_symmetric_no_selfloops(out)


@pytest.mark.parametrize("n, p, seed", _RANDOM_GNP)
def test_mst_forest_invariants_random(n, p, seed):
    g = nx.gnp_random_graph(n, p, seed=seed)
    data = _data_from_nx(g)
    out = MinimumSpanningTree()(data)
    components = nx.number_connected_components(g)
    # A spanning forest: |E| = |V| - (#components), and a subset of the original edges.
    assert _u_edges(out) == n - components
    assert _edge_set_undirected(out).issubset(_edge_set_undirected(data))


@pytest.mark.parametrize("n, p, seed", _RANDOM_GNP)
def test_khop_k1_is_identity_random(n, p, seed):
    g = nx.gnp_random_graph(n, p, seed=seed)
    data = _data_from_nx(g)
    out = KHopGraph(k=1)(data)
    assert _edge_set_undirected(out) == _edge_set_undirected(data)


@pytest.mark.parametrize("n, p, seed", _RANDOM_GNP)
def test_khop_monotone_in_k_random(n, p, seed):
    g = nx.gnp_random_graph(n, p, seed=seed)
    data = _data_from_nx(g)
    e1 = _edge_set_undirected(KHopGraph(k=1)(data))
    e2 = _edge_set_undirected(KHopGraph(k=2)(data))
    assert e1.issubset(e2)  # more hops never remove edges


@pytest.mark.parametrize("n, p, seed", _RANDOM_GNP)
def test_adjacency_power_k1_is_identity_random(n, p, seed):
    g = nx.gnp_random_graph(n, p, seed=seed)
    data = _data_from_nx(g)
    out = KNeighborGraphPower(k=1)(data)
    assert _edge_set_undirected(out) == _edge_set_undirected(data)


@pytest.mark.parametrize("n, p, seed", _RANDOM_GNP)
def test_tolevi_invariants_random(n, p, seed):
    g = nx.gnp_random_graph(n, p, seed=seed)
    data = _data_from_nx(g)
    out = ToLevi()(data)
    assert out.num_nodes == g.number_of_nodes() + g.number_of_edges()
    assert _u_edges(out) == 2 * g.number_of_edges()  # each incidence


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_transitive_closure_idempotent_random_dag(seed):
    # random DAG via an upper-triangular edge set
    rng = torch.Generator().manual_seed(seed)
    n = 6
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if torch.rand(1, generator=rng).item() < 0.4
    ]
    data = _data(edges, n, directed=True)
    closure = TransitiveClosure()
    once = closure(data)
    twice = closure(once)
    assert _edge_set_directed(once) == _edge_set_directed(twice)


@pytest.mark.parametrize("n, p, seed", _RANDOM_GNP)
def test_khop_no_cross_component_edges_random(n, p, seed):
    # two disjoint random graphs -> k-hop must not connect them
    g1 = nx.gnp_random_graph(n, p, seed=seed)
    g2 = nx.gnp_random_graph(n, p, seed=seed + 100)
    union = nx.disjoint_union(g1, g2)
    data = _data(list(union.edges()), union.number_of_nodes())
    out = KHopGraph(k=3)(data)
    left = set(range(n))
    for u, v in _edge_set_undirected(out):
        assert (u in left) == (v in left)  # no edge bridges the two components


# =============================================================================
# Degenerate / boundary graphs.
# =============================================================================
def test_single_node_boundary():
    d = _data([], 1)
    assert _u_edges(ComplementGraph()(d)) == 0
    assert _u_edges(MinimumSpanningTree()(d)) == 0
    assert _u_edges(KHopGraph(k=2)(d)) == 0
    assert _u_edges(KNeighborGraphPower(k=2)(d)) == 0
    assert ToLevi()(d).num_nodes == 1  # 1 vertex, 0 edges


def test_empty_edge_graph_boundary():
    d = _data([], 4)  # 4 isolated nodes
    assert _u_edges(ComplementGraph()(d)) == 6  # -> K4
    assert _u_edges(MinimumSpanningTree()(d)) == 0  # forest of isolated nodes
    assert _u_edges(KHopGraph(k=2)(d)) == 0
    assert _u_edges(KNeighborGraphPower(k=2)(d)) == 0
    assert ToLevi()(d).num_nodes == 4


def test_disconnected_complement_and_mst():
    # two triangles (nodes 0-1-2 and 3-4-5)
    d = _data([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], 6)
    assert _u_edges(MinimumSpanningTree()(d)) == 4  # 6 nodes - 2 components
    # complement of 2xK3 = complete bipartite-ish; edges = C(6,2) - 6 = 9
    assert _u_edges(ComplementGraph()(d)) == 9


# =============================================================================
# Attribute integrity for identity-preserving transforms.
# =============================================================================
_IDENTITY_PRESERVING = [
    lambda: ComplementGraph(),
    lambda: KHopGraph(k=2),
    lambda: KNeighborGraphPower(k=2),
    lambda: MinimumSpanningTree(),
]


@pytest.mark.parametrize("make_t", _IDENTITY_PRESERVING)
def test_identity_transforms_preserve_features(make_t):
    x = torch.arange(8.0).view(4, 2)
    data = _data([(0, 1), (1, 2), (2, 3), (0, 3)], 4, x=x)  # C4 with features
    out = make_t()(data)
    assert out.x is not None
    assert out.x.shape == (4, 2)
    assert torch.equal(out.x, x)  # same node set/order -> features unchanged


@pytest.mark.parametrize("make_t", _IDENTITY_PRESERVING)
def test_identity_transforms_symmetric_output(make_t):
    data = _data([(0, 1), (1, 2), (2, 3), (0, 3)], 4)
    assert _is_symmetric_no_selfloops(make_t()(data))


def test_every_declared_transform_is_discoverable_by_name():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent / "transformations"]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    info = PluginRegistry.get_plugin_info("combinatorial_dual")
    registered = set(info["registered_transforms"])
    for name in (
        "ToLevi",
        "GraphDual",
        "QuotientGraph",
        "CondensationTransform",
        "TransitiveClosure",
        "TransitiveReduction",
        "BipartiteProjection",
        "KHopGraph",
        "ComplementGraph",
        "KNeighborGraphPower",
        "MinimumSpanningTree",
        "GabrielGraph",
        "RelativeNeighborhoodGraph",
    ):
        assert name in registered
