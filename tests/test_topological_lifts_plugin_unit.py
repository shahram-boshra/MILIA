"""
Production-grade unit + discovery tests for the topological_lifts plugin (Family A).

Ground-truth lifts on named graphs (clique expansion of a path/star, star-expansion node
count = 2N, H0 persistence on a known filtration), output validity/symmetry, node-set growth
for star expansion, determinism, input immutability, and plugin discovery.
"""

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.plugins.topological_lifts.transforms import (
    HypergraphCliqueExpansion,
    HypergraphStarExpansion,
    PersistentHomologyFeature,
)
from milia_pipeline.transformations.custom_transforms import TransformExecutionError

ALL_CLASSES = [
    HypergraphCliqueExpansion,
    HypergraphStarExpansion,
    PersistentHomologyFeature,
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


def _star(n):
    return _data([(0, i) for i in range(1, n)], n)


def _u_edges(out):
    return {tuple(sorted(e)) for e in out.edge_index.t().tolist() if e[0] != e[1]}


def _assert_valid_undirected(out):
    ei = out.edge_index
    assert ei.dtype == torch.long
    assert ei.dim() == 2 and ei.size(0) == 2
    pairs = ei.t().tolist()
    assert {(u, v) for u, v in pairs} == {(v, u) for u, v in pairs}, "not symmetric"
    assert all(u != v for u, v in pairs), "self-loop present"
    ew = getattr(out, "edge_weight", None)
    if ew is not None:
        assert ew.numel() == ei.size(1)


# =========================================================================== metadata
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_metadata_shape(cls):
    meta = cls.get_metadata()
    assert meta.name == cls.__name__
    assert meta.version == "1.0.0"
    assert meta.category == "structural"
    assert meta.author and meta.description
    assert meta.modifies_attributes


# =========================================================================== A11 clique expansion
def test_clique_expansion_of_star_connects_leaves():
    # Closed 1-hop hyperedge of the hub = all nodes; clique-expanding connects every pair,
    # so the star (only hub-leaf edges) becomes complete on 5 nodes.
    out = HypergraphCliqueExpansion(k=1)(_star(5))
    assert _u_edges(out) == {(i, j) for i in range(5) for j in range(i + 1, 5)}


def test_clique_expansion_weight_counts_shared_hyperedges():
    out = HypergraphCliqueExpansion(k=1)(_path(4))
    _assert_valid_undirected(out)
    assert getattr(out, "edge_weight", None) is not None
    assert out.edge_weight.numel() == out.edge_index.size(1)
    assert torch.all(out.edge_weight > 0)


def test_clique_expansion_superset_of_original_on_path():
    src = _path(6)
    out = HypergraphCliqueExpansion(k=1)(src)
    assert _u_edges(src).issubset(_u_edges(out))  # neighbours become mutually connected


def test_clique_expansion_bad_k_fails_closed():
    with pytest.raises(TransformExecutionError):
        HypergraphCliqueExpansion(k=0)(_path(4))


# =========================================================================== A10 star expansion
def test_star_expansion_adds_one_meta_node_per_hyperedge():
    # One hyperedge per node -> N meta-nodes -> total 2N.
    out = HypergraphStarExpansion(k=1)(_cycle(6))
    assert out.num_nodes == 12


def test_star_expansion_keeps_original_edges():
    src = _path(5)
    out = HypergraphStarExpansion(k=1)(src)
    assert _u_edges(src).issubset(_u_edges(out))


def test_star_expansion_meta_features_are_member_means():
    src = _path(3)
    src.x = torch.tensor([[0.0], [2.0], [4.0]])
    out = HypergraphStarExpansion(k=1)(src)
    # Meta-node for node 1's closed 1-hop {0,1,2} has feature mean (0+2+4)/3 = 2.
    assert out.x.size(0) == 6
    meta_for_node1 = out.x[3 + 1]
    assert torch.allclose(meta_for_node1, torch.tensor([2.0]))


def test_star_expansion_zero_pads_edge_attr():
    src = _cycle(5)
    src.edge_attr = torch.ones((src.edge_index.size(1), 2))
    out = HypergraphStarExpansion(k=1)(src)
    assert out.edge_attr.size(0) == out.edge_index.size(1)
    assert (out.edge_attr == 0).any() and (out.edge_attr == 1).any()


def test_star_expansion_bad_k_fails_closed():
    with pytest.raises(TransformExecutionError):
        HypergraphStarExpansion(k=0)(_cycle(5))


# =========================================================================== A16 H0 persistence
def test_persistence_feature_shape_and_finiteness():
    out = PersistentHomologyFeature()(_cycle(6))
    ph = out["persistence_h0"]
    assert ph.shape == (1, 5)
    assert torch.isfinite(ph).all()


def test_persistence_on_regular_graph_is_zero():
    # A cycle is degree-regular -> all vertices share one birth value -> merges have
    # death == birth -> zero-lifetime pairs excluded -> all-zero summary.
    ph = PersistentHomologyFeature()(_cycle(6))["persistence_h0"]
    assert torch.allclose(ph, torch.zeros_like(ph))


def test_persistence_on_star_is_nonzero():
    # Star: leaves have degree 1, hub degree n-1 -> non-trivial birth spread -> positive total.
    ph = PersistentHomologyFeature()(_star(6))["persistence_h0"]
    assert float(ph[0, 0]) > 0.0  # total persistence
    assert float(ph[0, 2]) >= 1.0  # at least one finite pair


def test_persistence_does_not_change_graph():
    src = _path(6)
    ei = src.edge_index.clone()
    out = PersistentHomologyFeature()(src)
    assert torch.equal(out.edge_index, ei)  # feature-only transform
    assert out.num_nodes == 6


# =========================================================================== hardening: k>1, semantics
def _edge_weight_map(out):
    ew = out.edge_weight
    return {
        tuple(sorted((u, v))): float(ew[i]) for i, (u, v) in enumerate(out.edge_index.t().tolist())
    }


def _neighbors(out, node):
    nb = set()
    for u, v in out.edge_index.t().tolist():
        if u == node:
            nb.add(v)
        elif v == node:
            nb.add(u)
    return nb


def test_clique_expansion_k2_exercises_multihop():
    # k=2 must expand further than k=1 (multi-hop BFS): path5 -> complete K5.
    k1 = _u_edges(HypergraphCliqueExpansion(k=1)(_path(5)))
    k2 = _u_edges(HypergraphCliqueExpansion(k=2)(_path(5)))
    assert k1 < k2  # strict superset
    assert k2 == {(i, j) for i in range(5) for j in range(i + 1, 5)}


def test_clique_expansion_weights_equal_cooccurrence_counts():
    # Weight = number of shared closed-1-hop hyperedges (A = HHᵀ − D_v).
    wmap = _edge_weight_map(HypergraphCliqueExpansion(k=1)(_path(4)))
    assert wmap[(0, 1)] == 2.0
    assert wmap[(1, 2)] == 2.0
    assert wmap[(0, 2)] == 1.0
    assert wmap[(2, 3)] == 2.0


def test_clique_expansion_drops_incompatible_edge_attr():
    # Edge set changes, so stale edge_attr must be dropped (not misaligned).
    src = _cycle(6)
    src.edge_attr = torch.ones((src.edge_index.size(1), 3))
    out = HypergraphCliqueExpansion(k=1)(src)
    assert getattr(out, "edge_attr", None) is None


def test_star_expansion_k2_meta_connects_two_hop_members():
    # Meta-node for node 0 (index N+0) must connect to node 0's closed 2-hop = {0,1,2}.
    out = HypergraphStarExpansion(k=2)(_path(5))
    assert _neighbors(out, 5) == {0, 1, 2}


# =========================================================================== hardening: exact H0
def test_persistence_star_exact_diagram():
    # Star6 degree filtration: 4 leaves (birth 1) merge into the hub-anchored elder at death 5.
    # Diagram = 4 x (1,5); summary = [total=16, entropy=log4, pairs=4, max=4, mean=4].
    import math

    ph = PersistentHomologyFeature()(_star(6))["persistence_h0"]
    expected = torch.tensor([[16.0, math.log(4), 4.0, 4.0, 4.0]])
    assert torch.allclose(ph, expected, atol=1e-3)


def test_persistence_on_disconnected_graph():
    # Triangle (regular, zero-lifetime) + path3 (one (1,2) pair). Components never merge;
    # infinite components are excluded -> exactly one finite pair, total persistence 1.
    src = _data([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5)], 6)
    ph = PersistentHomologyFeature()(src)["persistence_h0"]
    assert torch.isfinite(ph).all()
    assert abs(float(ph[0, 0]) - 1.0) < 1e-4  # total persistence
    assert float(ph[0, 2]) == 1.0  # exactly one finite pair


def test_persistence_custom_attr_name():
    out = PersistentHomologyFeature(attr_name="ph_custom")(_star(6))
    assert "ph_custom" in out
    assert out["ph_custom"].shape == (1, 5)


# =========================================================================== validity / determinism
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_output_edge_index_valid(cls):
    src = _cycle(6)
    src.x = torch.ones((6, 2))
    out = cls()(src)
    _assert_valid_undirected(out)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_deterministic(cls):
    src = _path(7)
    src.x = torch.arange(7, dtype=torch.float).reshape(7, 1)
    a = cls()(src.clone())
    b = cls()(src.clone())
    assert torch.equal(a.edge_index, b.edge_index)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_no_mutation_of_input(cls):
    src = _cycle(6)
    ei = src.edge_index.clone()
    cls()(src)
    assert torch.equal(src.edge_index, ei)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_tiny_graph_noop(cls):
    out = cls()(_data([], 1))
    assert out.num_nodes == 1


# =========================================================================== property-based fuzz
@pytest.mark.parametrize("seed", list(range(5)))
def test_random_graphs_stay_valid_and_deterministic(seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 11))
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.35]
    if not edges:
        edges = [(0, 1)]
    base = _data(edges, n, x=torch.tensor(rng.random((n, 3)), dtype=torch.float))

    for transform in (
        HypergraphCliqueExpansion(k=1),
        HypergraphStarExpansion(k=1),
        PersistentHomologyFeature(),
    ):
        first = transform(base.clone())
        second = transform(base.clone())
        _assert_valid_undirected(first)
        assert torch.equal(first.edge_index, second.edge_index)


# =========================================================================== discovery
def test_plugin_discovery_registers_all_three():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=True)
    assert "topological_lifts" in names

    info = PluginRegistry.get_plugin_info("topological_lifts")
    assert info is not None
    assert info["declared_count"] == 3
    declared = {
        "HypergraphCliqueExpansion",
        "HypergraphStarExpansion",
        "PersistentHomologyFeature",
    }
    assert declared.issubset(set(info["registered_transforms"]))
    assert info["missing_implementations"] == []


def test_plugin_discovery_is_idempotent():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    assert "topological_lifts" in names
    info = PluginRegistry.get_plugin_info("topological_lifts")
    registered = info["registered_transforms"]
    assert len(registered) == len(set(registered))
