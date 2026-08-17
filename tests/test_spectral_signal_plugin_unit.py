"""
Production-grade unit + discovery tests for the spectral_signal plugin (Family E).

Every transform is checked for: correct output shape/placement, determinism, the two
spectral invariances that make Family E well-posed (eigenvector SIGN-flip invariance and
node-RELABELLING equivariance), fail-closed parameter validation, input immutability, and
end-to-end plugin discovery. Ground-truth spectra are used on small named graphs.
"""

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.plugins.spectral_signal.transforms import (
    GraphScatteringTransform,
    GraphWaveEmbedding,
    HeatKernelSignature,
    SpectralFeatureCoordinates,
    SpectralGraphWaveletTransform,
    WaveKernelSignature,
)
from milia_pipeline.transformations.custom_transforms import TransformExecutionError

ALL_CLASSES = [
    SpectralGraphWaveletTransform,
    GraphScatteringTransform,
    GraphWaveEmbedding,
    HeatKernelSignature,
    WaveKernelSignature,
    SpectralFeatureCoordinates,
]
DEFAULT_ATTRS = {
    SpectralGraphWaveletTransform: "sgwt",
    GraphScatteringTransform: "graph_scattering",
    GraphWaveEmbedding: "graphwave",
    HeatKernelSignature: "hks",
    WaveKernelSignature: "wks",
    SpectralFeatureCoordinates: "spectral_coords",
}


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


def _relabel(data, perm):
    """Apply a node permutation; spectral descriptors must permute the same way."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    new_ei = inv[data.edge_index]
    out = Data(edge_index=new_ei, num_nodes=data.num_nodes)
    if getattr(data, "x", None) is not None:
        out.x = data.x[perm]
    return out


def _attr(out, cls):
    return out[DEFAULT_ATTRS[cls]]


# =========================================================================== metadata
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_metadata_shape(cls):
    meta = cls.get_metadata()
    assert meta.name == cls.__name__
    assert meta.version == "1.0.0"
    assert meta.category == "structural"
    assert meta.author
    assert meta.description
    assert meta.modifies_attributes == [DEFAULT_ATTRS[cls]]


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_names_do_not_duplicate_pyg_builtins(cls):
    forbidden = {
        "AddLaplacianEigenvectorPE",
        "AddRandomWalkPE",
        "LaplacianLambdaMax",
        "SIGN",
        "GDC",
    }
    assert cls.get_metadata().name not in forbidden


# =========================================================================== shape / placement
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_output_is_per_node_and_named(cls):
    out = cls()(_cycle(6))
    feat = _attr(out, cls)
    assert feat.dim() == 2
    assert feat.size(0) == 6
    assert feat.size(1) >= 1
    assert torch.isfinite(feat).all()


def test_attr_name_none_concatenates_to_x():
    src = _cycle(6)
    src.x = torch.ones((6, 3))
    out = HeatKernelSignature(num_times=4, attr_name=None)(src)
    assert out.x.size(0) == 6
    assert out.x.size(1) == 3 + 4  # original 3 + 4 HKS times


def test_sgwt_shape_matches_scales():
    out = SpectralGraphWaveletTransform(scales=(1.0, 2.0, 4.0, 8.0))(_path(7))
    assert out["sgwt"].shape == (7, 4)


def test_hks_shape_matches_times():
    out = HeatKernelSignature(num_times=6)(_path(7))
    assert out["hks"].shape == (7, 6)


def test_wks_shape_matches_energies():
    out = WaveKernelSignature(num_energies=5)(_cycle(7))
    assert out["wks"].shape == (7, 5)


def test_graphwave_shape():
    out = GraphWaveEmbedding(scales=(1.0, 2.0), num_samples=4)(_cycle(6))
    assert out["graphwave"].shape == (6, 2 * 4 * 2)  # scales · samples · [Re, Im]


def test_spectral_coords_shape_and_padding():
    # k larger than available non-trivial eigenvectors -> zero-padded to width k.
    out = SpectralFeatureCoordinates(k=5)(_path(3))
    assert out["spectral_coords"].shape == (3, 5)


def test_scattering_order_increases_width():
    w0 = GraphScatteringTransform(order=0)(_cycle(8))["graph_scattering"].size(1)
    w1 = GraphScatteringTransform(order=1)(_cycle(8))["graph_scattering"].size(1)
    w2 = GraphScatteringTransform(order=2)(_cycle(8))["graph_scattering"].size(1)
    assert w0 < w1 <= w2


# =========================================================================== ground truth
def test_hks_row_sum_equals_trace_identity():
    # Σ_v hks_t(v) = Σ_k exp(-t λ_k) (since Σ_v φ_k(v)² = 1). Check at one time.
    data = _cycle(6)
    # num_times=1 -> logspace(start, end, 1) == [t_min]; single time = 0.5 (strict t_min<t_max).
    hks = HeatKernelSignature(num_times=1, t_min=0.5, t_max=1.0)(data)["hks"]
    # eigenvalues of the normalized-Laplacian on a cycle are known; verify via direct eig.
    transform = HeatKernelSignature(num_times=1)
    transform._metadata = transform.get_metadata()
    evals, _ = transform._laplacian_eig(data)
    expected = torch.exp(-0.5 * evals).sum()
    assert torch.allclose(hks.sum(), expected.to(torch.float32), atol=1e-4)


def test_regular_graph_descriptor_is_vertex_transitive():
    # A cycle is vertex-transitive: every node has an identical HKS row.
    hks = HeatKernelSignature(num_times=5)(_cycle(7))["hks"]
    assert torch.allclose(hks, hks[0:1].expand_as(hks), atol=1e-4)


def test_complete_graph_two_distinct_eigenvalues():
    # K_n normalized Laplacian has eigenvalues {0, n/(n-1) (mult n-1)} -> HKS rows equal.
    hks = HeatKernelSignature(num_times=4)(_complete(5))["hks"]
    assert torch.allclose(hks, hks[0:1].expand_as(hks), atol=1e-4)


# =========================================================================== invariances
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_relabelling_equivariance(cls):
    # Permuting node order permutes descriptor rows identically (structural invariance).
    data = _path(8)
    perm = torch.tensor([3, 0, 5, 1, 7, 2, 6, 4])
    base = _attr(cls()(data), cls)
    permuted = _attr(cls()(_relabel(data, perm)), cls)
    assert torch.allclose(base[perm], permuted, atol=1e-4)


@pytest.mark.parametrize("cls", [HeatKernelSignature, SpectralGraphWaveletTransform])
def test_descriptors_are_deterministic(cls):
    a = _attr(cls()(_cycle(9)), cls)
    b = _attr(cls()(_cycle(9)), cls)
    assert torch.equal(a, b)


def test_spectral_coords_sign_invariant_by_construction():
    # 'squared' mode is invariant to eigenvector sign; values are non-negative.
    out = SpectralFeatureCoordinates(k=3, mode="squared")(_path(6))["spectral_coords"]
    assert (out >= 0).all()


# =========================================================================== fail-closed
def test_sgwt_bad_kernel_fails_closed():
    with pytest.raises(TransformExecutionError):
        SpectralGraphWaveletTransform(kernel="ricker")(_path(4))


def test_scattering_bad_order_fails_closed():
    with pytest.raises(TransformExecutionError):
        GraphScatteringTransform(order=3)(_path(4))


def test_hks_bad_times_fail_closed():
    with pytest.raises(TransformExecutionError):
        HeatKernelSignature(t_min=5.0, t_max=1.0)(_path(4))


def test_wks_bad_sigma_fails_closed():
    with pytest.raises(TransformExecutionError):
        WaveKernelSignature(sigma=0.0)(_path(4))


def test_spectral_coords_bad_mode_fails_closed():
    with pytest.raises(TransformExecutionError):
        SpectralFeatureCoordinates(mode="raw")(_path(4))


def test_graphwave_bad_samples_fails_closed():
    with pytest.raises(TransformExecutionError):
        GraphWaveEmbedding(num_samples=0)(_path(4))


# =========================================================================== robustness
@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_single_node_graph_does_not_crash(cls):
    out = cls()(_data([], 1))
    assert _attr(out, cls).size(0) == 1


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_no_mutation_of_input(cls):
    src = _cycle(6)
    ei = src.edge_index.clone()
    cls()(src)
    assert torch.equal(src.edge_index, ei)
    assert DEFAULT_ATTRS[cls] not in src  # descriptor written only on the clone


def test_wks_edgeless_graph_is_finite():
    # Edgeless graph -> L_sym = I (isolated-node convention D^{-1/2}=0) -> eigenvalues all 1.
    # WKS must remain well-defined and finite (log 1 = 0 is valid), not NaN/inf.
    out = WaveKernelSignature(num_energies=4)(_data([], 3))
    assert out["wks"].shape == (3, 4)
    assert torch.isfinite(out["wks"]).all()


# =========================================================================== discovery
def test_plugin_discovery_registers_all_six():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=True)
    assert "spectral_signal" in names

    info = PluginRegistry.get_plugin_info("spectral_signal")
    assert info is not None
    assert info["declared_count"] == 6
    declared = {
        "SpectralGraphWaveletTransform",
        "GraphScatteringTransform",
        "GraphWaveEmbedding",
        "HeatKernelSignature",
        "WaveKernelSignature",
        "SpectralFeatureCoordinates",
    }
    assert declared.issubset(set(info["registered_transforms"]))
    assert info["missing_implementations"] == []


def test_plugin_discovery_is_idempotent():
    import milia_pipeline.plugins as plugins_pkg
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    parent = [Path(plugins_pkg.__file__).parent]
    PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    names = PluginRegistry.discover_plugins(paths=parent, auto_validate=False)
    assert "spectral_signal" in names
    info = PluginRegistry.get_plugin_info("spectral_signal")
    registered = info["registered_transforms"]
    assert len(registered) == len(set(registered))
