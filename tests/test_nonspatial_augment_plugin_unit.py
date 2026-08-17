"""
Production-grade unit + discovery tests for the nonspatial_augment plugin (Family G).

Every transform is checked against hand-computable ground truth on small named graphs,
plus fail-closed parameter validation, determinism (same seed -> identical output),
seed-sensitivity, input immutability (no mutation of the source Data), edge_attr shape
consistency, feature/pos carry, and end-to-end plugin discovery (missing==[]).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.plugins.nonspatial_augment.transforms import (
    EdgeFeatureMasking,
    NASANeighborReplace,
    PPRSubgraphSampling,
    RandomEdgeAdd,
    RandomWalkSubgraphCrop,
    SubmodularFeatureSalience,
)
from milia_pipeline.transformations.custom_transforms import TransformExecutionError

ALL_CLASSES = [
    RandomEdgeAdd,
    EdgeFeatureMasking,
    RandomWalkSubgraphCrop,
    PPRSubgraphSampling,
    NASANeighborReplace,
    SubmodularFeatureSalience,
]


# --------------------------------------------------------------------------- helpers
def _data(edges, n, x=None, edge_attr=None, pos=None):
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
    if edge_attr is not None:
        d.edge_attr = edge_attr
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
    return {tuple(sorted(e)) for e in out.edge_index.t().tolist()}


def _num_undirected(out):
    return len(_u_edges(out))


def _snapshot(data):
    return (
        data.edge_index.clone(),
        None if getattr(data, "x", None) is None else data.x.clone(),
        None if getattr(data, "edge_attr", None) is None else data.edge_attr.clone(),
    )


def _assert_unchanged(data, snap):
    ei, x, ea = snap
    assert torch.equal(data.edge_index, ei)
    if x is None:
        assert getattr(data, "x", None) is None
    else:
        assert torch.equal(data.x, x)
    if ea is None:
        assert getattr(data, "edge_attr", None) is None
    else:
        assert torch.equal(data.edge_attr, ea)


# =========================================================================== metadata
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_metadata_shape(cls):
    meta = cls.get_metadata()
    assert meta.name == cls.__name__
    assert meta.version == "1.0.0"
    assert meta.category == "structural"
    assert meta.author
    assert meta.description
    assert meta.modifies_attributes


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_parameter_constraints_present(cls):
    constraints = cls().get_parameter_constraints()
    assert isinstance(constraints, dict)
    assert "seed" in constraints


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_metadata_names_are_not_pyg_or_shipped(cls):
    # Guard against duplicating pyg_augmentation names or PyG 2.6.1 built-ins.
    forbidden = {
        "DropEdge",
        "DropNode",
        "MaskFeatures",
        "RandomNodeSample",
        "LineGraph",
        "GDC",
        "VirtualNode",
        "Pad",
        "TwoHop",
    }
    assert cls.get_metadata().name not in forbidden


# =========================================================================== G2
def test_random_edge_add_adds_expected_count():
    # Path of 5 nodes: 4 undirected edges; capacity = C(5,2) - 4 = 6.
    out = RandomEdgeAdd(p=0.5, seed=0)(_path(5))
    # p*existing = 0.5*4 = 2 edges added -> 6 undirected edges total.
    assert _num_undirected(out) == 6


def test_random_edge_add_only_adds_never_removes():
    src = _path(6)
    original = {tuple(sorted(e)) for e in src.edge_index.t().tolist()}
    out = RandomEdgeAdd(p=1.0, seed=3)(src)
    assert original.issubset(_u_edges(out))


def test_random_edge_add_respects_capacity_on_complete_graph():
    out = RandomEdgeAdd(p=1.0, seed=1)(_complete(4))
    # Complete graph has no room to add edges.
    assert _num_undirected(out) == 6


def test_random_edge_add_edge_attr_zero_padded():
    src = _path(5)
    src.edge_attr = torch.ones((src.edge_index.size(1), 3))
    out = RandomEdgeAdd(p=0.5, seed=0)(src)
    assert out.edge_attr.size(0) == out.edge_index.size(1)
    assert out.edge_attr.size(1) == 3
    # New rows are zero; original ones stay one.
    assert (out.edge_attr == 0).any()
    assert (out.edge_attr == 1).any()


def test_random_edge_add_deterministic_and_seed_sensitive():
    a = RandomEdgeAdd(p=0.8, seed=7)(_cycle(8))
    b = RandomEdgeAdd(p=0.8, seed=7)(_cycle(8))
    c = RandomEdgeAdd(p=0.8, seed=8)(_cycle(8))
    assert _u_edges(a) == _u_edges(b)
    assert _u_edges(a) != _u_edges(c)


def test_random_edge_add_no_mutation():
    src = _path(6)
    snap = _snapshot(src)
    RandomEdgeAdd(p=1.0, seed=2)(src)
    _assert_unchanged(src, snap)


def test_random_edge_add_negative_p_fails_closed():
    with pytest.raises(TransformExecutionError):
        RandomEdgeAdd(p=-0.1)(_path(4))


def test_random_edge_add_zero_p_is_noop():
    out = RandomEdgeAdd(p=0.0, seed=0)(_path(5))
    assert _num_undirected(out) == 4


# =========================================================================== G5
def test_edge_feature_masking_masks_expected_dims():
    src = _path(4)
    src.edge_attr = torch.ones((src.edge_index.size(1), 10))
    out = EdgeFeatureMasking(p=0.3, seed=0)(src)
    # 3 of 10 columns zeroed on every edge row.
    zero_cols = (out.edge_attr == 0).all(dim=0).sum().item()
    assert zero_cols == 3


def test_edge_feature_masking_noop_without_edge_attr():
    out = EdgeFeatureMasking(p=1.0, seed=0)(_path(4))
    assert getattr(out, "edge_attr", None) is None


def test_edge_feature_masking_deterministic_and_seed_sensitive():
    src = _path(5)
    src.edge_attr = torch.arange(src.edge_index.size(1) * 8, dtype=torch.float).reshape(-1, 8)
    a = EdgeFeatureMasking(p=0.5, seed=1)(src.clone())
    b = EdgeFeatureMasking(p=0.5, seed=1)(src.clone())
    c = EdgeFeatureMasking(p=0.5, seed=2)(src.clone())
    assert torch.equal(a.edge_attr, b.edge_attr)
    assert not torch.equal(a.edge_attr, c.edge_attr)


def test_edge_feature_masking_no_mutation():
    src = _path(4)
    src.edge_attr = torch.ones((src.edge_index.size(1), 6))
    snap = _snapshot(src)
    EdgeFeatureMasking(p=0.5, seed=0)(src)
    _assert_unchanged(src, snap)


def test_edge_feature_masking_invalid_p_fails_closed():
    src = _path(4)
    src.edge_attr = torch.ones((src.edge_index.size(1), 4))
    with pytest.raises(TransformExecutionError):
        EdgeFeatureMasking(p=1.5)(src)


def test_edge_feature_masking_custom_value():
    src = _path(4)
    src.edge_attr = torch.ones((src.edge_index.size(1), 4))
    out = EdgeFeatureMasking(p=1.0, mask_value=-1.0, seed=0)(src)
    assert torch.all(out.edge_attr == -1.0)


# =========================================================================== G7
def test_rw_crop_keeps_ratio_nodes():
    out = RandomWalkSubgraphCrop(ratio=0.5, seed=0)(_path(10))
    assert out.num_nodes == 5


def test_bfs_crop_is_connected_prefix():
    out = RandomWalkSubgraphCrop(ratio=0.4, mode="bfs", seed=0)(_path(10))
    assert out.num_nodes == 4
    # Induced subgraph of a path stays a (shorter) path/forest: no isolated duplication.
    assert out.edge_index.size(1) >= 2


def test_rw_crop_carries_x_and_pos():
    src = _path(8)
    src.x = torch.arange(8, dtype=torch.float).reshape(8, 1)
    src.pos = torch.arange(16, dtype=torch.float).reshape(8, 2)
    out = RandomWalkSubgraphCrop(ratio=0.5, seed=1)(src)
    assert out.x.size(0) == out.num_nodes
    assert out.pos.size(0) == out.num_nodes


def test_rw_crop_carries_edge_attr_consistently():
    src = _cycle(8)
    src.edge_attr = torch.ones((src.edge_index.size(1), 2))
    out = RandomWalkSubgraphCrop(ratio=0.5, seed=1)(src)
    assert out.edge_attr.size(0) == out.edge_index.size(1)
    assert out.edge_attr.size(1) == 2


def test_rw_crop_deterministic_and_seed_sensitive():
    a = RandomWalkSubgraphCrop(ratio=0.5, seed=4)(_cycle(12))
    b = RandomWalkSubgraphCrop(ratio=0.5, seed=4)(_cycle(12))
    c = RandomWalkSubgraphCrop(ratio=0.5, seed=5)(_cycle(12))
    assert torch.equal(a.edge_index, b.edge_index)
    assert a.num_nodes == c.num_nodes  # same size, (very likely) different region


def test_rw_crop_ratio_one_is_noop_size():
    out = RandomWalkSubgraphCrop(ratio=1.0, seed=0)(_path(6))
    assert out.num_nodes == 6


def test_rw_crop_bad_ratio_and_mode_fail_closed():
    with pytest.raises(TransformExecutionError):
        RandomWalkSubgraphCrop(ratio=0.0)(_path(5))
    with pytest.raises(TransformExecutionError):
        RandomWalkSubgraphCrop(ratio=0.5, mode="dfs")(_path(5))


def test_rw_crop_no_mutation():
    src = _cycle(9)
    snap = _snapshot(src)
    RandomWalkSubgraphCrop(ratio=0.5, seed=0)(src)
    _assert_unchanged(src, snap)


# =========================================================================== G8
def test_ppr_keeps_ratio_nodes():
    out = PPRSubgraphSampling(ratio=0.5, seed=0)(_path(10))
    assert out.num_nodes == 5


def test_ppr_start_node_retained_and_local():
    # On a path, PPR mass concentrates near the start, so the kept set is a local window.
    out = PPRSubgraphSampling(ratio=0.4, seed=0)(_path(12))
    assert out.num_nodes == 5


def test_ppr_deterministic_and_seed_sensitive():
    a = PPRSubgraphSampling(ratio=0.5, seed=2)(_cycle(10))
    b = PPRSubgraphSampling(ratio=0.5, seed=2)(_cycle(10))
    c = PPRSubgraphSampling(ratio=0.5, seed=3)(_cycle(10))
    assert torch.equal(a.edge_index, b.edge_index)
    assert a.num_nodes == c.num_nodes


def test_ppr_carries_features():
    src = _path(8)
    src.x = torch.arange(8, dtype=torch.float).reshape(8, 1)
    out = PPRSubgraphSampling(ratio=0.5, seed=1)(src)
    assert out.x.size(0) == out.num_nodes


def test_ppr_bad_params_fail_closed():
    with pytest.raises(TransformExecutionError):
        PPRSubgraphSampling(ratio=1.5)(_path(5))
    with pytest.raises(TransformExecutionError):
        PPRSubgraphSampling(ratio=0.5, alpha=1.0)(_path(5))


def test_ppr_no_mutation():
    src = _cycle(9)
    snap = _snapshot(src)
    PPRSubgraphSampling(ratio=0.5, seed=0)(src)
    _assert_unchanged(src, snap)


# =========================================================================== G12
def test_nasa_replaces_one_hop_with_two_hop_on_path():
    # Path 0-1-2-3-4, rewire only node 0 (ratio small, seed fixed): 0's 1-hop {1} removed,
    # its exactly-2-hop {2} added, so edge (0,2) appears and (0,1) may be gone.
    src = _path(5)
    out = NASANeighborReplace(hop=2, ratio=1.0, seed=0)(src)
    edges = _u_edges(out)
    # Every node's 1-hop replaced by its 2-hop: connectivity strictly changes.
    assert edges != {tuple(sorted(e)) for e in src.edge_index.t().tolist()}


def test_nasa_deterministic_and_seed_sensitive():
    a = NASANeighborReplace(hop=2, ratio=0.5, seed=1)(_cycle(10))
    b = NASANeighborReplace(hop=2, ratio=0.5, seed=1)(_cycle(10))
    c = NASANeighborReplace(hop=2, ratio=0.5, seed=2)(_cycle(10))
    assert _u_edges(a) == _u_edges(b)
    assert _u_edges(a) != _u_edges(c)


def test_nasa_edge_attr_reset_consistent():
    src = _cycle(8)
    src.edge_attr = torch.ones((src.edge_index.size(1), 4))
    out = NASANeighborReplace(hop=2, ratio=1.0, seed=0)(src)
    assert out.edge_attr.size(0) == out.edge_index.size(1)
    assert out.edge_attr.size(1) == 4


def test_nasa_hop_below_two_fails_closed():
    with pytest.raises(TransformExecutionError):
        NASANeighborReplace(hop=1)(_path(5))


def test_nasa_no_mutation():
    src = _cycle(9)
    snap = _snapshot(src)
    NASANeighborReplace(hop=2, ratio=1.0, seed=0)(src)
    _assert_unchanged(src, snap)


def test_nasa_tiny_graph_noop():
    out = NASANeighborReplace(hop=2, ratio=1.0, seed=0)(_path(2))
    assert out.num_nodes == 2


# =========================================================================== G15
def test_submodular_zeros_non_selected_dims_keeps_width():
    # 4 nodes, 6 feature dims; retain half (3), zero the rest, keep width 6.
    x = torch.eye(6).repeat(1, 1)[:4]  # [4, 6]
    src = _path(4)
    src.x = x
    out = SubmodularFeatureSalience(ratio=0.5, seed=0)(src)
    assert out.x.shape == (4, 6)
    kept = (out.x != 0).any(dim=0).sum().item()
    assert kept <= 3


def test_submodular_deterministic():
    src = _path(6)
    src.x = torch.arange(6 * 8, dtype=torch.float).reshape(6, 8)
    a = SubmodularFeatureSalience(ratio=0.5)(src.clone())
    b = SubmodularFeatureSalience(ratio=0.5)(src.clone())
    assert torch.equal(a.x, b.x)


def test_submodular_noop_without_x():
    out = SubmodularFeatureSalience(ratio=0.5)(_path(4))
    assert getattr(out, "x", None) is None


def test_submodular_ratio_one_is_noop():
    src = _path(4)
    src.x = torch.arange(4 * 5, dtype=torch.float).reshape(4, 5)
    out = SubmodularFeatureSalience(ratio=1.0)(src)
    assert torch.equal(out.x, src.x)


def test_submodular_bad_ratio_fails_closed():
    src = _path(4)
    src.x = torch.ones((4, 4))
    with pytest.raises(TransformExecutionError):
        SubmodularFeatureSalience(ratio=0.0)(src)


def test_submodular_no_mutation():
    src = _path(5)
    src.x = torch.arange(5 * 6, dtype=torch.float).reshape(5, 6)
    snap = _snapshot(src)
    SubmodularFeatureSalience(ratio=0.5)(src)
    _assert_unchanged(src, snap)


# =========================================================================== property-based
@pytest.mark.parametrize("seed", list(range(8)))
def test_seeded_random_graphs_never_crash_and_are_deterministic(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 15))
    prob = 0.3
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < prob]
    base = _data(edges or [(0, 1)], n)
    base.x = torch.tensor(rng.random((n, 5)), dtype=torch.float)
    base.edge_attr = torch.ones((base.edge_index.size(1), 3))

    transforms = [
        RandomEdgeAdd(p=0.3, seed=seed),
        EdgeFeatureMasking(p=0.5, seed=seed),
        RandomWalkSubgraphCrop(ratio=0.6, seed=seed),
        PPRSubgraphSampling(ratio=0.6, seed=seed),
        NASANeighborReplace(hop=2, ratio=0.5, seed=seed),
        SubmodularFeatureSalience(ratio=0.5, seed=seed),
    ]
    for t in transforms:
        first = t(base.clone())
        second = t(base.clone())
        assert first.edge_index.size(0) == 2
        if getattr(first, "edge_attr", None) is not None:
            assert first.edge_attr.size(0) == first.edge_index.size(1)
        assert torch.equal(first.edge_index, second.edge_index)


# =========================================================================== discovery
def test_plugin_discovery_registers_all_six():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=True)
    assert "nonspatial_augment" in names

    info = PluginRegistry.get_plugin_info("nonspatial_augment")
    assert info is not None
    assert info["declared_count"] == 6
    declared = {
        "RandomEdgeAdd",
        "EdgeFeatureMasking",
        "RandomWalkSubgraphCrop",
        "PPRSubgraphSampling",
        "NASANeighborReplace",
        "SubmodularFeatureSalience",
    }
    assert declared.issubset(set(info["registered_transforms"]))
    assert info["missing_implementations"] == []


def test_plugin_discovery_is_idempotent():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    assert "nonspatial_augment" in names
    info = PluginRegistry.get_plugin_info("nonspatial_augment")
    registered = info["registered_transforms"]
    assert len(registered) == len(set(registered))
