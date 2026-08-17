"""
Spectral & signal-decomposition transforms (catalogue Family E, in-scope subset).

Six unary node-descriptor transforms (G -> G with new node attributes) satisfying the
``transform(data) -> data`` purity contract: single ``Data`` in/out, no model, no dataset,
no labels. MILIA-custom idiom (``CustomTransformBase`` + ``TransformMetadata``), category
``structural``.

Determinism guarantee. ``torch.linalg.eigh`` documents that eigenvectors are not unique
(sign flips, and arbitrary rotation within degenerate eigenspaces), so different platforms
may return different eigenvectors. Every transform here is therefore restricted to a form
that is invariant to that non-uniqueness: either a spectral-theorem operator
``Psi_s = U g(sΛ) Uᵀ = g(sL)`` (basis-independent by construction), or a signature of the
form ``Σ_k τ(λ_k) φ_k(v)²`` (the squared eigenvector coordinate is sign-invariant, and its
sum over a degenerate eigenspace equals a projector diagonal, hence rotation-invariant).

In scope (this file): E1 ``SpectralGraphWaveletTransform``, E2 ``GraphScatteringTransform``,
E5 ``GraphWaveEmbedding``, E6 ``HeatKernelSignature``, E7 ``WaveKernelSignature``,
E8 ``SpectralFeatureCoordinates``.

Deferred (Blueprint Family E design block): E9/E10 (raw-eigenbasis GFT — sign/degeneracy
non-deterministic), E3/E4 (diffusion wavelets / tight framelets — correctness risk). Not
duplicated (already bridged as PyG built-ins): ``AddLaplacianEigenvectorPE``,
``AddRandomWalkPE``, ``LaplacianLambdaMax``, ``SIGN``, ``GDC``.

Single-module layout with ``module_path: transforms`` matches the plugin loader.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from milia_pipeline.transformations.custom_transforms import (
    CustomTransformBase,
    TransformExecutionError,
    TransformMetadata,
)

_AUTHOR = "MILIA Team"
_EPS = 1e-12


def _add_node_attr(data: Data, value: torch.Tensor, attr_name: str | None) -> Data:
    """Store a per-node feature, mirroring PyG's convention.

    ``attr_name is None`` -> concatenate to ``data.x`` (creating it if absent);
    otherwise store under ``data[attr_name]``.
    """
    if attr_name is None:
        base = getattr(data, "x", None)
        if base is None:
            data.x = value
        else:
            base = base.view(-1, 1) if base.dim() == 1 else base
            data.x = torch.cat([base, value.to(base.dtype)], dim=-1)
    else:
        data[attr_name] = value
    return data


class SpectralSignalBase(CustomTransformBase):
    """Shared symmetric-normalised-Laplacian eigen-engine for Family E."""

    def _laplacian_eig(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (eigenvalues ascending in [0, 2], eigenvectors as columns) of L_sym.

        L_sym = I - D^{-1/2} A D^{-1/2}, built undirected and dense in float64 for
        numerical stability; tiny negative eigenvalues from round-off are clamped to 0.
        """
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 1:
            raise TransformExecutionError(
                f"{type(self).__name__} requires a graph with >= 1 node.",
                transform_name=self._metadata.name,
            )

        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
        if data.edge_index.numel() > 0:
            src = data.edge_index[0].to(torch.long)
            dst = data.edge_index[1].to(torch.long)
            adj[src, dst] = 1.0
            adj = torch.maximum(adj, adj.t())  # symmetrise (undirected)
            adj.fill_diagonal_(0.0)

        deg = adj.sum(dim=1)
        dinv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        lap = torch.eye(num_nodes, dtype=torch.float64) - (
            dinv_sqrt.view(-1, 1) * adj * dinv_sqrt.view(1, -1)
        )
        lap = 0.5 * (lap + lap.t())  # enforce exact symmetry for eigh

        eigenvalues, eigenvectors = torch.linalg.eigh(lap)
        eigenvalues = eigenvalues.clamp_min(0.0)
        return eigenvalues, eigenvectors

    def _wavelet_operator(
        self, evals: torch.Tensor, evecs: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """Heat wavelet operator Psi_s = U exp(-s Λ) Uᵀ (basis-independent)."""
        filt = torch.exp(-scale * evals)
        return (evecs * filt.view(1, -1)) @ evecs.t()

    def _signal(self, data: Data, num_nodes: int) -> torch.Tensor:
        """Signal for scattering: data.x (float64) if present, else node degree."""
        base = getattr(data, "x", None)
        if base is not None:
            return base.to(torch.float64).view(num_nodes, -1)
        if data.edge_index.numel() > 0:
            deg = torch.zeros(num_nodes, dtype=torch.float64)
            deg.index_add_(
                0,
                data.edge_index[0].to(torch.long),
                torch.ones(data.edge_index.size(1), dtype=torch.float64),
            )
            return deg.view(num_nodes, 1)
        return torch.zeros((num_nodes, 1), dtype=torch.float64)


# ---------------------------------------------------------------------------
# E1 — SpectralGraphWaveletTransform
# ---------------------------------------------------------------------------
class SpectralGraphWaveletTransform(SpectralSignalBase):
    """Per-node multiscale wavelet energy ``Σ_k g(sλ_k) φ_k(v)²`` (Hammond et al. 2011)."""

    def __init__(
        self,
        scales: tuple[float, ...] = (1.0, 2.0, 4.0),
        kernel: str = "heat",
        attr_name: str | None = "sgwt",
    ):
        super().__init__()
        self.scales = tuple(float(s) for s in scales)
        self.kernel = str(kernel)
        self.attr_name = attr_name

    def _kernel(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == "heat":  # low-pass exp(-x)
            return torch.exp(-x)
        if self.kernel == "mexican_hat":  # bandpass x·e^{1-x}, g(0)=0
            return x * torch.exp(1.0 - x)
        raise TransformExecutionError(
            f"SpectralGraphWaveletTransform kernel must be 'heat' or 'mexican_hat', got '{self.kernel}'.",
            transform_name=self._metadata.name,
        )

    def transform(self, data: Data) -> Data:
        if not self.scales:
            raise TransformExecutionError(
                "SpectralGraphWaveletTransform requires at least one scale.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        evals, evecs = self._laplacian_eig(data)
        evecs_sq = evecs.pow(2)  # [N, K], sign-invariant
        cols = []
        for scale in self.scales:
            filt = self._kernel(scale * evals)  # [K]
            cols.append((evecs_sq * filt.view(1, -1)).sum(dim=1, keepdim=True))
        feats = torch.cat(cols, dim=1).to(torch.float32)  # [N, |scales|]
        return _add_node_attr(data, feats, self.attr_name)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="SpectralGraphWaveletTransform",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="E1 spectral descriptor: per-node multiscale graph-wavelet energy.",
            paper_reference="Hammond, Vandergheynst, Gribonval, ACHA 2011",
            modifies_attributes=["sgwt"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "scales": {"type": tuple},
            "kernel": {"type": str, "choices": ["heat", "mexican_hat"]},
            "attr_name": {"type": str, "optional": True},
        }


# ---------------------------------------------------------------------------
# E2 — GraphScatteringTransform
# ---------------------------------------------------------------------------
class GraphScatteringTransform(SpectralSignalBase):
    """Fixed (non-trainable) wavelet-modulus scattering cascade (Gama/Gao 2019)."""

    def __init__(
        self,
        scales: tuple[float, ...] = (1.0, 2.0, 4.0),
        order: int = 2,
        lowpass_scale: float = 8.0,
        attr_name: str | None = "graph_scattering",
    ):
        super().__init__()
        self.scales = tuple(float(s) for s in scales)
        self.order = int(order)
        self.lowpass_scale = float(lowpass_scale)
        self.attr_name = attr_name

    def transform(self, data: Data) -> Data:
        if self.order not in (0, 1, 2):
            raise TransformExecutionError(
                f"GraphScatteringTransform supports order in {{0,1,2}}, got {self.order}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        evals, evecs = self._laplacian_eig(data)
        low = self._wavelet_operator(evals, evecs, self.lowpass_scale)  # Φ
        wavelets = [self._wavelet_operator(evals, evecs, s) for s in self.scales]
        signal = self._signal(data, num_nodes)  # [N, F]

        blocks = [low @ signal]  # S0 = Φ x
        if self.order >= 1:
            first = [(psi @ signal).abs() for psi in wavelets]  # |Ψ_s x|
            blocks.extend(low @ u for u in first)  # S1 = Φ|Ψ_s x|
            if self.order >= 2:
                for i, psi2 in enumerate(wavelets):
                    for u1 in first[:i]:  # s2 < s1 convention (strictly increasing path)
                        blocks.append(low @ (psi2 @ u1).abs())  # S2 = Φ|Ψ_{s2}|Ψ_{s1}x||
        feats = torch.cat(blocks, dim=1).to(torch.float32)  # [N, F·num_paths]
        return _add_node_attr(data, feats, self.attr_name)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="GraphScatteringTransform",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="E2 spectral descriptor: fixed wavelet-modulus scattering cascade.",
            paper_reference="Gama, Ribeiro, Bruna 2019; Gao, Wolf, Hirn 2019",
            modifies_attributes=["graph_scattering"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "scales": {"type": tuple},
            "order": {"type": int, "choices": [0, 1, 2]},
            "lowpass_scale": {"type": float},
            "attr_name": {"type": str, "optional": True},
        }


# ---------------------------------------------------------------------------
# E5 — GraphWaveEmbedding
# ---------------------------------------------------------------------------
class GraphWaveEmbedding(SpectralSignalBase):
    """Structural embedding from the empirical characteristic function of wavelet
    coefficient distributions (Donnat et al., GraphWave, KDD 2018)."""

    def __init__(
        self,
        scales: tuple[float, ...] = (1.0, 2.0),
        num_samples: int = 8,
        t_max: float = 1.0,
        attr_name: str | None = "graphwave",
    ):
        super().__init__()
        self.scales = tuple(float(s) for s in scales)
        self.num_samples = int(num_samples)
        self.t_max = float(t_max)
        self.attr_name = attr_name

    def transform(self, data: Data) -> Data:
        if self.num_samples < 1:
            raise TransformExecutionError(
                f"GraphWaveEmbedding requires num_samples >= 1, got {self.num_samples}.",
                transform_name=self._metadata.name,
            )
        if not self.scales:
            raise TransformExecutionError(
                "GraphWaveEmbedding requires at least one scale.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        evals, evecs = self._laplacian_eig(data)
        sample_t = torch.linspace(0.0, self.t_max, self.num_samples + 1, dtype=torch.float64)[1:]

        cols = []
        for scale in self.scales:
            psi = self._wavelet_operator(evals, evecs, scale)  # [N, N]; column a = Ψ_s δ_a
            # Empirical characteristic function χ_a(t) = mean_m exp(i t Ψ[m, a]).
            coeffs = psi.t()  # [N(node a), N(m)]
            for t in sample_t.tolist():
                phase = t * coeffs  # [N, N]
                cols.append(torch.cos(phase).mean(dim=1, keepdim=True))  # Re
                cols.append(torch.sin(phase).mean(dim=1, keepdim=True))  # Im
        feats = torch.cat(cols, dim=1).to(torch.float32)  # [N, |scales|·num_samples·2]
        return _add_node_attr(data, feats, self.attr_name)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="GraphWaveEmbedding",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="E5 spectral descriptor: GraphWave characteristic-function embedding.",
            paper_reference="Donnat, Zitnik, Hallac, Leskovec, KDD 2018",
            modifies_attributes=["graphwave"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "scales": {"type": tuple},
            "num_samples": {"type": int, "min": 1},
            "t_max": {"type": float},
            "attr_name": {"type": str, "optional": True},
        }


# ---------------------------------------------------------------------------
# E6 — HeatKernelSignature
# ---------------------------------------------------------------------------
class HeatKernelSignature(SpectralSignalBase):
    """Per-node HKS ``Σ_k exp(-t λ_k) φ_k(v)²`` at log-spaced times (Sun et al. 2009)."""

    def __init__(
        self,
        num_times: int = 10,
        t_min: float = 0.1,
        t_max: float = 10.0,
        attr_name: str | None = "hks",
    ):
        super().__init__()
        self.num_times = int(num_times)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.attr_name = attr_name

    def transform(self, data: Data) -> Data:
        if self.num_times < 1:
            raise TransformExecutionError(
                f"HeatKernelSignature requires num_times >= 1, got {self.num_times}.",
                transform_name=self._metadata.name,
            )
        if not 0.0 < self.t_min < self.t_max:
            raise TransformExecutionError(
                f"HeatKernelSignature requires 0 < t_min < t_max, got ({self.t_min}, {self.t_max}).",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        evals, evecs = self._laplacian_eig(data)
        evecs_sq = evecs.pow(2)  # [N, K]
        times = torch.logspace(
            torch.log10(torch.tensor(self.t_min)),
            torch.log10(torch.tensor(self.t_max)),
            self.num_times,
            dtype=torch.float64,
        )
        cols = []
        for t in times.tolist():
            weight = torch.exp(-t * evals)  # [K]
            cols.append((evecs_sq * weight.view(1, -1)).sum(dim=1, keepdim=True))
        feats = torch.cat(cols, dim=1).to(torch.float32)  # [N, num_times]
        return _add_node_attr(data, feats, self.attr_name)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="HeatKernelSignature",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="E6 spectral descriptor: multiscale heat-kernel-signature node features.",
            paper_reference="Sun, Ovsjanikov, Guibas 2009 (graph-adapted)",
            modifies_attributes=["hks"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "num_times": {"type": int, "min": 1},
            "t_min": {"type": float, "min": 0.0},
            "t_max": {"type": float, "min": 0.0},
            "attr_name": {"type": str, "optional": True},
        }


# ---------------------------------------------------------------------------
# E7 — WaveKernelSignature
# ---------------------------------------------------------------------------
class WaveKernelSignature(SpectralSignalBase):
    """Per-node WKS ``C_e^{-1} Σ_k exp(-(e-log λ_k)²/2σ²) φ_k(v)²`` (Aubry et al. 2011)."""

    def __init__(
        self,
        num_energies: int = 10,
        sigma: float = 1.0,
        attr_name: str | None = "wks",
    ):
        super().__init__()
        self.num_energies = int(num_energies)
        self.sigma = float(sigma)
        self.attr_name = attr_name

    def transform(self, data: Data) -> Data:
        if self.num_energies < 1:
            raise TransformExecutionError(
                f"WaveKernelSignature requires num_energies >= 1, got {self.num_energies}.",
                transform_name=self._metadata.name,
            )
        if self.sigma <= 0.0:
            raise TransformExecutionError(
                f"WaveKernelSignature requires sigma > 0, got {self.sigma}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        evals, evecs = self._laplacian_eig(data)
        positive = evals > _EPS  # log undefined at λ=0 (the constant mode)
        if positive.sum() == 0:
            feats = torch.zeros((data.num_nodes, self.num_energies), dtype=torch.float32)
            return _add_node_attr(data, feats, self.attr_name)

        log_evals = torch.log(evals[positive])  # [K']
        evecs_sq = evecs[:, positive].pow(2)  # [N, K']
        energies = torch.linspace(
            float(log_evals.min()), float(log_evals.max()), self.num_energies, dtype=torch.float64
        )
        cols = []
        for e in energies.tolist():
            weight = torch.exp(-((e - log_evals) ** 2) / (2.0 * self.sigma**2))  # [K']
            norm = weight.sum().clamp_min(_EPS)
            cols.append((evecs_sq * weight.view(1, -1)).sum(dim=1, keepdim=True) / norm)
        feats = torch.cat(cols, dim=1).to(torch.float32)  # [N, num_energies]
        return _add_node_attr(data, feats, self.attr_name)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="WaveKernelSignature",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="E7 spectral descriptor: multiscale wave-kernel-signature node features.",
            paper_reference="Aubry, Schlickewei, Cremers 2011 (graph-adapted)",
            modifies_attributes=["wks"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "num_energies": {"type": int, "min": 1},
            "sigma": {"type": float, "min": 0.0},
            "attr_name": {"type": str, "optional": True},
        }


# ---------------------------------------------------------------------------
# E8 — SpectralFeatureCoordinates
# ---------------------------------------------------------------------------
class SpectralFeatureCoordinates(SpectralSignalBase):
    """Sign-invariant low-frequency coordinates ``φ_k(v)²`` (or ``|φ_k(v)|``).

    Distinct from the already-bridged PyG ``AddLaplacianEigenvectorPE`` (raw, sign-ambiguous
    eigenvectors): squaring/abs removes the eigenvector sign ambiguity documented for
    ``torch.linalg.eigh``.
    """

    def __init__(
        self,
        k: int = 4,
        mode: str = "squared",
        attr_name: str | None = "spectral_coords",
    ):
        super().__init__()
        self.k = int(k)
        self.mode = str(mode)
        self.attr_name = attr_name

    def transform(self, data: Data) -> Data:
        if self.k < 1:
            raise TransformExecutionError(
                f"SpectralFeatureCoordinates requires k >= 1, got {self.k}.",
                transform_name=self._metadata.name,
            )
        if self.mode not in ("squared", "abs"):
            raise TransformExecutionError(
                f"SpectralFeatureCoordinates mode must be 'squared' or 'abs', got '{self.mode}'.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        evals, evecs = self._laplacian_eig(data)
        # Skip the trivial constant mode (smallest eigenvalue); take next k non-trivial.
        selected = evecs[:, 1 : 1 + self.k]  # [N, <=k]
        transformed = selected.pow(2) if self.mode == "squared" else selected.abs()
        # Zero-pad to a fixed width k for a stable feature contract.
        if transformed.size(1) < self.k:
            pad = torch.zeros((num_nodes, self.k - transformed.size(1)), dtype=transformed.dtype)
            transformed = torch.cat([transformed, pad], dim=1)
        feats = transformed.to(torch.float32)  # [N, k]
        return _add_node_attr(data, feats, self.attr_name)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="SpectralFeatureCoordinates",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="E8 spectral descriptor: sign-invariant low-frequency eigenvector coordinates.",
            paper_reference="Sign-invariant spectral features (cf. Lim et al. SignNet)",
            modifies_attributes=["spectral_coords"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "k": {"type": int, "min": 1},
            "mode": {"type": str, "choices": ["squared", "abs"]},
            "attr_name": {"type": str, "optional": True},
        }
