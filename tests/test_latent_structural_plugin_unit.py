"""
Production-grade unit + discovery tests for the latent_structural plugin (Family J).

Ground-truth on named graphs (C6 ring -> 1 super-node; fused rings; ReFeX regular-graph
identical rows; anonymous-walk patterns of known walks; triangle vs path AWE differ), output
validity, determinism, input immutability, feature-carry, custom attrs, a fuzz loop, and
plugin discovery.
"""

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.plugins.latent_structural.transforms import (
    AnonymousWalkEmbedding,
    MotifCompression,
    StructuralRoleEncoding,
)
from milia_pipeline.transformations.custom_transforms import TransformExecutionError

ALL_CLASSES = [MotifCompression, StructuralRoleEncoding, AnonymousWalkEmbedding]


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


def _triangle():
    return _data([(0, 1), (1, 2), (0, 2)], 3)


def _u_edges(out):
    return {tuple(sorted(e)) for e in out.edge_index.t().tolist() if e[0] != e[1]}


def _assert_symmetric(out):
    pairs = out.edge_index.t().tolist()
    assert {(u, v) for u, v in pairs} == {(v, u) for u, v in pairs}
    assert all(u != v for u, v in pairs)
    assert out.edge_index.dtype == torch.long


# =========================================================================== metadata
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_metadata_shape(cls):
    meta = cls.get_metadata()
    assert meta.name == cls.__name__
    assert meta.version == "1.0.0"
    assert meta.category == "structural"
    assert meta.author and meta.description
    assert meta.modifies_attributes


# =========================================================================== J2 motif compression
def test_motif_compression_collapses_ring_to_one_node():
    # A benzene-like C6 ring collapses to a single super-node.
    out = MotifCompression()(_cycle(6))
    assert out.num_nodes == 1
    assert out.edge_index.size(1) == 0


def test_motif_compression_ring_with_pendant():
    # C6 ring (0..5) + pendant node 6 attached to 0 -> ring super-node + node 6, one edge.
    edges = [(i, (i + 1) % 6) for i in range(6)] + [(0, 6)]
    out = MotifCompression()(_data(edges, 7))
    assert out.num_nodes == 2
    assert out.edge_index.size(1) == 2  # one undirected edge


def test_motif_compression_no_ring_is_noop():
    src = _path(5)  # a tree has no cycles
    out = MotifCompression()(src)
    assert out.num_nodes == 5
    assert _u_edges(out) == _u_edges(src)


def test_motif_compression_averages_features():
    src = _cycle(6)
    src.x = torch.arange(6, dtype=torch.float).reshape(6, 1)
    out = MotifCompression()(src)
    assert out.x.size(0) == 1
    assert torch.allclose(out.x, torch.tensor([[2.5]]))  # mean(0..5)


def test_motif_compression_bad_ring_size_fails_closed():
    with pytest.raises(TransformExecutionError):
        MotifCompression(max_ring_size=2)(_cycle(6))


def test_motif_compression_symmetric_output():
    edges = [(i, (i + 1) % 5) for i in range(5)] + [(0, 5), (5, 6), (6, 0)]
    _assert_symmetric(MotifCompression()(_data(edges, 7)))


# =========================================================================== J3 ReFeX role encoding
def test_role_encoding_base_feature_dims_grow_with_iterations():
    # dim = 3 * 3**iterations (each iteration triples via [self, mean, sum]).
    r0 = StructuralRoleEncoding(iterations=0)(_cycle(6))["struct_roles"]
    r1 = StructuralRoleEncoding(iterations=1)(_cycle(6))["struct_roles"]
    r2 = StructuralRoleEncoding(iterations=2)(_cycle(6))["struct_roles"]
    assert r0.shape == (6, 3)
    assert r1.shape == (6, 9)
    assert r2.shape == (6, 27)


def test_role_encoding_regular_graph_has_identical_rows():
    # A cycle is vertex-transitive -> every node has the same structural role.
    roles = StructuralRoleEncoding(iterations=2)(_cycle(7))["struct_roles"]
    assert torch.allclose(roles, roles[0:1].expand_as(roles))


def test_role_encoding_base_features_ground_truth():
    # Path 0-1-2-3: node 1 has degree 2; egonet {0,1,2} has internal edges (0,1),(1,2)=2;
    # external edge (2,3) leaves the egonet -> external endpoint count 1.
    roles = StructuralRoleEncoding(iterations=0)(_path(4))["struct_roles"]
    assert roles[1, 0] == 2.0  # degree
    assert roles[1, 1] == 2.0  # egonet-internal edges


def test_role_encoding_distinguishes_hub_from_leaf():
    roles = StructuralRoleEncoding(iterations=1)(_data([(0, 1), (0, 2), (0, 3)], 4))["struct_roles"]
    assert not torch.allclose(roles[0], roles[1])  # hub != leaf


def test_role_encoding_custom_attr_name():
    out = StructuralRoleEncoding(attr_name="refex")(_cycle(6))
    assert "refex" in out


# =========================================================================== J4 anonymous walks
def test_anonymous_pattern_relabelling():
    t = AnonymousWalkEmbedding()
    assert t._anonymous([3, 7, 3, 2]) == (0, 1, 0, 2)
    assert t._anonymous([5, 5, 5]) == (0, 0, 0)


def test_awe_distribution_sums_to_one_on_nonempty_graph():
    out = AnonymousWalkEmbedding(walk_length=3)(_cycle(6))
    awe = out["awe"]
    assert awe.dim() == 2 and awe.size(0) == 1
    assert abs(float(awe.sum()) - 1.0) < 1e-5


def test_awe_triangle_differs_from_path():
    tri = AnonymousWalkEmbedding(walk_length=2)(_triangle())["awe"]
    pth = AnonymousWalkEmbedding(walk_length=2)(_path(3))["awe"]
    assert tri.shape == pth.shape
    assert not torch.allclose(tri, pth)  # a triangle admits the returning walk 0-1-0? vs closure


def test_awe_fixed_dimension_independent_of_graph():
    a = AnonymousWalkEmbedding(walk_length=3)(_cycle(6))["awe"]
    b = AnonymousWalkEmbedding(walk_length=3)(_path(5))["awe"]
    assert a.shape == b.shape  # canonical pattern space is graph-independent


def test_awe_bad_length_fails_closed():
    with pytest.raises(TransformExecutionError):
        AnonymousWalkEmbedding(walk_length=0)(_cycle(5))


# =========================================================================== hardening: J2 edge cases
def test_motif_compression_fuses_shared_ring_nodes():
    # Naphthalene: two 6-rings sharing edge (4,5) -> union-find merges -> ONE super-node.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4)]
    out = MotifCompression()(_data(edges, 10))
    assert out.num_nodes == 1


def test_motif_compression_leaves_oversized_ring():
    # A 10-ring exceeds max_ring_size=8 -> not compressed -> node set unchanged.
    out = MotifCompression(max_ring_size=8)(_cycle(10))
    assert out.num_nodes == 10


def test_motif_compression_disconnected_two_triangles():
    src = _data([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], 6)
    out = MotifCompression()(src)
    assert out.num_nodes == 2  # each triangle collapses independently


# =========================================================================== hardening: J3 exact recursion
def test_role_encoding_recursive_values_exact():
    # Path 0-1-2. base: n0=(1,1,1) n1=(2,2,0) n2=(1,1,1). After one mean+sum aggregation,
    # node 1 (neighbours {0,2}, both (1,1,1)) -> [self | mean | sum] = [2,2,0, 1,1,1, 2,2,2].
    roles = StructuralRoleEncoding(iterations=1)(_path(3))["struct_roles"]
    expected = torch.tensor([2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    assert torch.allclose(roles[1], expected, atol=1e-5)


# =========================================================================== hardening: J4 exact AWE
def test_awe_triangle_and_path_exact_distributions():
    # Pattern space (L=2) = [(0,1,0), (0,1,2)]. Triangle is symmetric -> [0.5, 0.5];
    # path 0-1-2 yields 4x(0,1,0) + 2x(0,1,2) over all starts -> [2/3, 1/3].
    tri = AnonymousWalkEmbedding(walk_length=2)(_triangle())["awe"]
    pth = AnonymousWalkEmbedding(walk_length=2)(_path(3))["awe"]
    assert torch.allclose(tri, torch.tensor([[0.5, 0.5]]), atol=1e-5)
    assert torch.allclose(pth, torch.tensor([[2 / 3, 1 / 3]]), atol=1e-5)


def test_awe_walk_cap_is_deterministic_and_valid():
    # Dense graph + tiny per-node cap exercises the truncation branch; must stay
    # deterministic and remain a valid (normalised) distribution.
    src = _data([(i, j) for i in range(5) for j in range(i + 1, 5)], 5)  # K5
    t = AnonymousWalkEmbedding(walk_length=3, max_walks_per_node=5)
    a = t(src.clone())["awe"]
    b = t(src.clone())["awe"]
    assert torch.equal(a, b)
    assert abs(float(a.sum()) - 1.0) < 1e-5


# =========================================================================== hardening: feature-only carry
def test_feature_only_transforms_preserve_graph_and_x():
    src = _cycle(6)
    src.x = torch.arange(12, dtype=torch.float).reshape(6, 2)
    ei = src.edge_index.clone()
    for transform in (StructuralRoleEncoding(iterations=2), AnonymousWalkEmbedding(walk_length=3)):
        out = transform(src.clone())
        assert torch.equal(out.edge_index, ei)  # topology untouched
        assert torch.equal(out.x, src.x)  # node features untouched
    roles = StructuralRoleEncoding(iterations=2)(src.clone())["struct_roles"]
    assert roles.size(0) == 6  # one role vector per node


# =========================================================================== determinism / no-mutation
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_deterministic(cls):
    src = _cycle(6)
    src.x = torch.arange(6, dtype=torch.float).reshape(6, 1)
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
    assert out.num_nodes in (0, 1)


# =========================================================================== property-based fuzz
@pytest.mark.parametrize("seed", list(range(5)))
def test_random_graphs_stay_valid_and_deterministic(seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 10))
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.4]
    if not edges:
        edges = [(0, 1)]
    base = _data(edges, n, x=torch.tensor(rng.random((n, 2)), dtype=torch.float))

    for transform in (
        MotifCompression(),
        StructuralRoleEncoding(iterations=2),
        AnonymousWalkEmbedding(walk_length=3),
    ):
        first = transform(base.clone())
        second = transform(base.clone())
        _assert_symmetric(first)
        assert torch.equal(first.edge_index, second.edge_index)


# =========================================================================== discovery
def test_plugin_discovery_registers_all_three():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=True)
    assert "latent_structural" in names

    info = PluginRegistry.get_plugin_info("latent_structural")
    assert info is not None
    assert info["declared_count"] == 3
    declared = {"MotifCompression", "StructuralRoleEncoding", "AnonymousWalkEmbedding"}
    assert declared.issubset(set(info["registered_transforms"]))
    assert info["missing_implementations"] == []


def test_plugin_discovery_is_idempotent():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    assert "latent_structural" in names
    info = PluginRegistry.get_plugin_info("latent_structural")
    registered = info["registered_transforms"]
    assert len(registered) == len(set(registered))
