"""
Unit + discovery tests for the graph_products plugin (catalogue Family C).

Covers each algebraic product transform's structure/determinism/feature-carry,
dependency-injected factor resolution (incl. fail-closed), the stochastic-Kronecker
generator, and end-to-end plugin discovery (Phase 4.2 gate: empty
``missing_implementations``).
"""

from pathlib import Path

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.plugins.transformations.graph_products.transforms import (
    CartesianProduct,
    CoronaProduct,
    KroneckerGraphGeneration,
    LexicographicProduct,
    RootedProduct,
    StrongProduct,
    TensorProduct,
)
from milia_pipeline.transformations.custom_transforms import TransformExecutionError

PATH2_SPEC = {"generator": "path_graph", "n": 2}


@pytest.fixture
def base_data():
    """Path graph P3 (3 nodes, edges 0-1, 1-2) with 2-dim node features."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.arange(6, dtype=torch.float).view(3, 2)
    return Data(edge_index=edge_index, x=x, num_nodes=3)


def _is_symmetric(edge_index: torch.Tensor) -> bool:
    edges = {tuple(e) for e in edge_index.t().tolist()}
    return all((v, u) in edges for (u, v) in edges)


@pytest.mark.parametrize(
    "transform_cls, expected_nodes",
    [
        (CartesianProduct, 6),
        (TensorProduct, 6),
        (StrongProduct, 6),
        (LexicographicProduct, 6),
        (RootedProduct, 6),
    ],
)
def test_tuple_product_structure(base_data, transform_cls, expected_nodes):
    """Tuple-product transforms give |V(G)|*|V(H)| nodes, carry x, stay symmetric."""
    transform = transform_cls(factor_graph=PATH2_SPEC)
    out = transform(base_data)

    assert isinstance(out, Data)
    assert out.num_nodes == expected_nodes
    assert out.x is not None and out.x.shape == (expected_nodes, 2)
    assert out.edge_index.dim() == 2 and out.edge_index.size(0) == 2
    assert _is_symmetric(out.edge_index)


def test_corona_product_structure(base_data):
    """Corona G o H: |V(G)| + |V(G)|*|V(H)| nodes = 3 + 3*2 = 9."""
    out = CoronaProduct(factor_graph=PATH2_SPEC)(base_data)
    assert isinstance(out, Data)
    assert out.num_nodes == 9
    assert _is_symmetric(out.edge_index)


def test_product_is_deterministic(base_data):
    """A non-stochastic product yields identical output across runs."""
    t = StrongProduct(factor_graph=PATH2_SPEC)
    a = t(base_data)
    b = t(base_data)
    assert torch.equal(a.edge_index, b.edge_index)
    assert torch.equal(a.x, b.x)


def test_factor_graph_accepts_nx_graph(base_data):
    """An explicit nx.Graph factor is accepted (not only a spec)."""
    out = CartesianProduct(factor_graph=nx.path_graph(2))(base_data)
    assert out.num_nodes == 6


def test_factor_resolution_from_data_attribute(base_data):
    """Per-sample fallback: H read from the named attribute on the Data."""
    t = CartesianProduct()  # no explicit factor
    base_data.product_factor_H = nx.path_graph(2)
    h = t._resolve_factor_graph(base_data)
    assert isinstance(h, nx.Graph) and h.number_of_nodes() == 2


def test_missing_factor_fails_closed(base_data):
    """No explicit factor and no data attribute -> fail closed."""
    t = CartesianProduct()
    with pytest.raises(TransformExecutionError):
        t._resolve_factor_graph(base_data)


def test_invalid_generator_spec_fails_closed(base_data):
    """An unknown generator name in a factor spec is rejected."""
    t = CartesianProduct(factor_graph={"generator": "not_a_real_generator"})
    with pytest.raises(TransformExecutionError):
        t(base_data)


def test_kronecker_complete_graph(base_data):
    """All-ones initiator, k=2 -> complete K4 (6 undirected edges), seeded/deterministic."""
    t = KroneckerGraphGeneration(initiator=[[1.0, 1.0], [1.0, 1.0]], k=2, seed=0)
    out = t(base_data)
    assert out.num_nodes == 4
    assert out.edge_index.size(1) == 12  # 6 undirected edges, both directions
    assert _is_symmetric(out.edge_index)
    assert torch.equal(out.edge_index, t(base_data).edge_index)


def test_kronecker_empty_graph(base_data):
    """All-zeros initiator -> no edges, but the node set is still b**k."""
    out = KroneckerGraphGeneration(initiator=[[0.0, 0.0], [0.0, 0.0]], k=2, seed=0)(base_data)
    assert out.num_nodes == 4
    assert out.edge_index.size(1) == 0


def test_kronecker_invalid_initiator_fails_closed(base_data):
    """A non-square initiator is rejected."""
    t = KroneckerGraphGeneration(initiator=[[0.5, 0.5, 0.5]], k=2)
    with pytest.raises(TransformExecutionError):
        t(base_data)


def test_metadata_contract():
    """Every transform declares category=structural, a product/Kronecker description,
    and a non-empty modifies_attributes (drives metadata-driven dependency inference)."""
    for cls in (
        CartesianProduct,
        TensorProduct,
        StrongProduct,
        LexicographicProduct,
        RootedProduct,
        CoronaProduct,
        KroneckerGraphGeneration,
    ):
        meta = cls.get_metadata()
        assert meta.category == "structural"
        assert meta.modifies_attributes
        blurb = f"{meta.name} {meta.description}".lower()
        assert "product" in blurb or "kronecker" in blurb


def test_plugin_discovery_registers_all_transforms():
    """End-to-end: discovery finds graph_products and registers all 7 declared transforms."""
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    plugins_parent = (
        Path(plugins_pkg.__file__).parent / "transformations"
    )  # discover_plugins globs */plugin.yaml

    names = PluginRegistry.discover_plugins(paths=[plugins_parent], auto_validate=True)
    assert "graph_products" in names

    info = PluginRegistry.get_plugin_info("graph_products")
    assert info is not None

    declared = {
        "CartesianProduct",
        "TensorProduct",
        "StrongProduct",
        "LexicographicProduct",
        "RootedProduct",
        "CoronaProduct",
        "KroneckerGraphGeneration",
    }
    assert declared.issubset(set(info["registered_transforms"]))
    assert info["missing_implementations"] == []


# =============================================================================
# Mathematical correctness — ground-truth product identities on P2 (□ P2).
#   Cartesian(P2,P2)=C4 (4 edges);  Tensor=2K2 (2);  Strong=K4 (6);
#   Lexicographic=K4 (6);  Rooted=tree (3);  Corona=6 nodes/7 edges.
# =============================================================================


def _p2_data():
    """Path graph P2 (nodes 0,1; edge 0-1) with distinct 2-dim features."""
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    return Data(edge_index=ei, x=x, num_nodes=2)


def _u_edges(out) -> int:
    """Undirected edge count from a symmetric edge_index."""
    return out.edge_index.size(1) // 2


def _degrees(out) -> list[int]:
    return torch.bincount(out.edge_index[0], minlength=out.num_nodes).tolist()


@pytest.mark.parametrize(
    "transform_cls, exp_nodes, exp_edges",
    [
        (CartesianProduct, 4, 4),  # C4
        (TensorProduct, 4, 2),  # 2K2
        (StrongProduct, 4, 6),  # K4
        (LexicographicProduct, 4, 6),  # K4
        (RootedProduct, 4, 3),  # spanning tree
        (CoronaProduct, 6, 7),
    ],
)
def test_product_ground_truth_counts(transform_cls, exp_nodes, exp_edges):
    out = transform_cls(factor_graph=nx.path_graph(2))(_p2_data())
    assert out.num_nodes == exp_nodes
    assert _u_edges(out) == exp_edges


def test_cartesian_p2_is_c4():
    out = CartesianProduct(factor_graph=nx.path_graph(2))(_p2_data())
    assert sorted(_degrees(out)) == [2, 2, 2, 2]  # 4-cycle


def test_strong_p2_is_k4():
    out = StrongProduct(factor_graph=nx.path_graph(2))(_p2_data())
    assert sorted(_degrees(out)) == [3, 3, 3, 3]  # complete K4


def test_lexicographic_p2_is_k4():
    out = LexicographicProduct(factor_graph=nx.path_graph(2))(_p2_data())
    assert sorted(_degrees(out)) == [3, 3, 3, 3]


def test_tensor_p2_is_two_k2():
    out = TensorProduct(factor_graph=nx.path_graph(2))(_p2_data())
    assert sorted(_degrees(out)) == [1, 1, 1, 1]  # two disjoint edges


# =============================================================================
# Feature carrying, immutability, determinism.
# =============================================================================


def test_feature_carry_tiles_base_rows():
    """Each product node (g, h) inherits G's row g; every base row repeats |V(H)| times."""
    out = CartesianProduct(factor_graph=nx.path_graph(2))(_p2_data())
    assert out.x is not None and out.x.shape == (4, 2)
    assert sorted(out.x.tolist()) == sorted(
        [[10.0, 11.0], [10.0, 11.0], [12.0, 13.0], [12.0, 13.0]]
    )


def test_feature_carry_none_when_input_x_none():
    data = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=2)
    out = CartesianProduct(factor_graph=nx.path_graph(2))(data)
    assert out.x is None
    assert out.num_nodes == 4


def test_transform_does_not_mutate_input():
    data = _p2_data()
    ei_before = data.edge_index.clone()
    x_before = data.x.clone()
    CartesianProduct(factor_graph=nx.path_graph(2))(data)
    assert torch.equal(data.edge_index, ei_before)
    assert torch.equal(data.x, x_before)
    assert data.num_nodes == 2


@pytest.mark.parametrize(
    "transform_cls",
    [CartesianProduct, TensorProduct, StrongProduct, LexicographicProduct, CoronaProduct],
)
def test_all_products_deterministic(transform_cls):
    t = transform_cls(factor_graph=nx.path_graph(2))
    a = t(_p2_data())
    b = t(_p2_data())
    assert torch.equal(a.edge_index, b.edge_index)


# =============================================================================
# Factor resolution — branch coverage of _coerce_to_graph / _build_factor_from_spec.
# =============================================================================


def test_coerce_accepts_data_factor():
    hdata = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=2)
    out = CartesianProduct(factor_graph=hdata)(_p2_data())
    assert out.num_nodes == 4


def test_coerce_rejects_unsupported_type():
    with pytest.raises(TransformExecutionError):
        CartesianProduct(factor_graph=42)(_p2_data())


def test_coerce_directed_factor_is_undirected():
    dig = nx.DiGraph()
    dig.add_edge(0, 1)
    out = CartesianProduct(factor_graph=dig)(_p2_data())
    assert out.num_nodes == 4
    assert _u_edges(out) == 4  # coerced to undirected P2 -> C4


def test_build_factor_complete_graph_spec():
    out = CartesianProduct(factor_graph={"generator": "complete_graph", "n": 3})(_p2_data())
    assert out.num_nodes == 6  # |V(G)| * |V(H)| = 2 * 3


def test_rooted_explicit_root():
    out = RootedProduct(factor_graph=nx.path_graph(2), root=1)(_p2_data())
    assert out.num_nodes == 4


def test_rooted_invalid_root_fails_closed():
    with pytest.raises(TransformExecutionError):
        RootedProduct(factor_graph=nx.path_graph(2), root=99)(_p2_data())


# =============================================================================
# Kronecker generator — reproducibility, edge cases, validation, guards.
# =============================================================================


def test_kronecker_reproducible_same_seed():
    a = KroneckerGraphGeneration(k=3, seed=7)(_p2_data())
    b = KroneckerGraphGeneration(k=3, seed=7)(_p2_data())
    assert torch.equal(a.edge_index, b.edge_index)


def test_kronecker_k1():
    out = KroneckerGraphGeneration(initiator=[[1.0, 1.0], [1.0, 1.0]], k=1, seed=0)(_p2_data())
    assert out.num_nodes == 2
    assert _u_edges(out) == 1


def test_kronecker_probability_below_zero_fails():
    with pytest.raises(TransformExecutionError):
        KroneckerGraphGeneration(initiator=[[-0.1, 0.5], [0.5, 0.5]], k=2)(_p2_data())


def test_kronecker_probability_above_one_fails():
    with pytest.raises(TransformExecutionError):
        KroneckerGraphGeneration(initiator=[[1.5, 0.5], [0.5, 0.5]], k=2)(_p2_data())


def test_kronecker_zero_power_fails():
    with pytest.raises(TransformExecutionError):
        KroneckerGraphGeneration(k=0)(_p2_data())


def test_kronecker_max_nodes_guard():
    with pytest.raises(TransformExecutionError):
        KroneckerGraphGeneration(k=20, max_nodes=1000)(_p2_data())  # 2**20 >> 1000


def test_kronecker_returns_no_features():
    out = KroneckerGraphGeneration(initiator=[[1.0, 1.0], [1.0, 1.0]], k=2, seed=0)(_p2_data())
    assert getattr(out, "x", None) is None


# =============================================================================
# Discovery robustness.
# =============================================================================


def test_discovery_is_idempotent_and_declares_seven():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent / "transformations"]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=False)  # second pass

    assert "graph_products" in names
    info = PluginRegistry.get_plugin_info("graph_products")
    assert info["declared_count"] == 7
    # Re-discovery must not duplicate registrations.
    registered = info["registered_transforms"]
    assert len(registered) == len(set(registered))
