"""
Curvature & differential-geometric rewiring transforms (catalogue Family B, in-scope subset).

Four unary rewiring transforms (G -> G with edges added/pruned by a geometric criterion)
satisfying the ``transform(data) -> data`` purity contract: single ``Data`` in/out, no model,
no dataset, no labels. The node set is preserved (only edges change); added edges receive a
zero ``edge_attr`` slice (Step 6.3). MILIA-custom idiom (``CustomTransformBase`` +
``TransformMetadata``), category ``structural``. All transforms are deterministic (argmax with
index tie-breaks; no RNG).

In scope (this file): B3 ``FormanRicciRewiring``, B6 ``EffectiveResistanceRewiring``,
B7 ``SpectralGapRewiring``, B12 ``ResistanceCurvatureRewiring``.

Deferred (Blueprint Family B design block): B1 Ollivier (optimal-transport dependency + cost),
B2 Balanced-Forman (documented gamma_max/4-cycle counting discrepancy), B4/B5/B8/B9/B10/B11.
Zero new dependencies (torch only; ``L+`` via ``torch.linalg.pinv``). Single-module layout.
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
_EPS = 1e-9


class CurvatureRewiringBase(CustomTransformBase):
    """Shared geometry helpers: adjacency, motif counts, Laplacian pseudoinverse, rewiring."""

    # -- topology ------------------------------------------------------------
    @staticmethod
    def _undirected_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        for u, v in edge_index.t().tolist():
            if u != v:
                edges.add((u, v) if u < v else (v, u))
        return edges

    @staticmethod
    def _adjacency(edges: set[tuple[int, int]], num_nodes: int) -> list[set[int]]:
        adj: list[set[int]] = [set() for _ in range(num_nodes)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    @staticmethod
    def _triangles(adj: list[set[int]], u: int, v: int) -> int:
        return len(adj[u] & adj[v])

    @staticmethod
    def _quads(adj: list[set[int]], u: int, v: int) -> int:
        """4-cycles based at edge (u,v): neighbours of u and v (excluding shared/each other)
        that are themselves adjacent, not forming a triangle on (u,v)."""
        tri = adj[u] & adj[v]
        u_only = adj[u] - adj[v] - {v}
        v_only = adj[v] - adj[u] - {u}
        count = 0
        for a in u_only:
            for b in adj[a]:
                if b in v_only and b not in tri:
                    count += 1
        return count

    def _forman(self, adj: list[set[int]], u: int, v: int) -> float:
        """Augmented Forman-Ricci curvature 4 - d_u - d_v + 3*triangles + 2*quads."""
        return (
            4.0
            - len(adj[u])
            - len(adj[v])
            + 3.0 * self._triangles(adj, u, v)
            + 2.0 * self._quads(adj, u, v)
        )

    # -- spectral / resistance ----------------------------------------------
    def _laplacian(self, edges: set[tuple[int, int]], num_nodes: int) -> torch.Tensor:
        lap = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
        for u, v in edges:
            lap[u, v] -= 1.0
            lap[v, u] -= 1.0
            lap[u, u] += 1.0
            lap[v, v] += 1.0
        return lap

    def _resistance_matrix(self, lap: torch.Tensor) -> torch.Tensor:
        """R_ij = L+_ii + L+_jj - 2 L+_ij from the Laplacian pseudoinverse."""
        lpinv = torch.linalg.pinv(lap)
        diag = torch.diag(lpinv)
        return diag.view(-1, 1) + diag.view(1, -1) - 2.0 * lpinv

    def _algebraic_connectivity(self, lap: torch.Tensor) -> float:
        """Second-smallest Laplacian eigenvalue (spectral gap / Fiedler value)."""
        evals = torch.linalg.eigvalsh(lap)
        return float(evals[1]) if evals.numel() > 1 else 0.0

    # -- output construction -------------------------------------------------
    def _rebuild(self, data: Data, edges: set[tuple[int, int]], num_nodes: int) -> Data:
        """Rebuild edge_index from an undirected edge set; carry edge_attr for retained edges,
        zero-pad rows for newly added edges (Step 6.3). Node set is preserved."""
        out = data.clone()
        ea = getattr(data, "edge_attr", None)
        attr_lookup = None
        width = 0
        if ea is not None:
            attr_lookup = {}
            for i, (u, v) in enumerate(data.edge_index.t().tolist()):
                attr_lookup.setdefault((u, v) if u < v else (v, u), i)
            width = 1 if ea.dim() == 1 else ea.size(1)

        ordered = sorted(edges)
        src: list[int] = []
        dst: list[int] = []
        attr_rows: list[torch.Tensor] = []
        for u, v in ordered:
            src.extend((u, v))
            dst.extend((v, u))
            if ea is not None:
                idx = attr_lookup.get((u, v))
                if idx is None:  # newly added edge -> zero attr slice
                    zero = (
                        torch.zeros(width, dtype=ea.dtype)
                        if ea.dim() > 1
                        else torch.zeros((), dtype=ea.dtype)
                    )
                    attr_rows.extend((zero, zero))
                else:
                    attr_rows.extend((ea[idx], ea[idx]))

        out.edge_index = (
            torch.tensor([src, dst], dtype=data.edge_index.dtype)
            if src
            else torch.empty((2, 0), dtype=data.edge_index.dtype)
        )
        if ea is not None:
            out.edge_attr = torch.stack(attr_rows) if attr_rows else ea[:0]
        out.num_nodes = num_nodes
        return out

    def _validate_iterations(self, num_iterations: int) -> None:
        if num_iterations < 1:
            raise TransformExecutionError(
                f"{type(self).__name__} requires num_iterations >= 1, got {num_iterations}.",
                transform_name=self._metadata.name,
            )


# ===========================================================================
# B3 — FormanRicciRewiring
# ===========================================================================
class FormanRicciRewiring(CurvatureRewiringBase):
    """SDRF-style rewiring guided by augmented Forman-Ricci curvature (Topping/Fesser)."""

    def __init__(self, num_iterations: int = 3, prune: bool = True):
        super().__init__()
        self.num_iterations = int(num_iterations)
        self.prune = bool(prune)

    def transform(self, data: Data) -> Data:
        self._validate_iterations(self.num_iterations)
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 3:
            return data
        edges = self._undirected_edges(data.edge_index)

        for _ in range(self.num_iterations):
            if not edges:
                break
            adj = self._adjacency(edges, num_nodes)
            # Bottleneck = the most negatively curved edge (min curvature, index tie-break).
            u, v = min(edges, key=lambda e: (self._forman(adj, *e), e))
            # Candidate support edges around the bottleneck; add the one most improving F(u,v).
            best_gain = 0.0
            best_edge = None
            for x in sorted(adj[u] | {u}):
                for y in sorted(adj[v] | {v}):
                    if x == y:
                        continue
                    cand = (x, y) if x < y else (y, x)
                    if cand in edges:
                        continue
                    trial = adj[u] | ({x} if x != u else set())
                    shared_after = trial & (adj[v] | ({y} if y != v else set()))
                    gain = float(len(shared_after) - self._triangles(adj, u, v))
                    if gain > best_gain or (best_edge is None and gain >= best_gain):
                        best_gain = gain
                        best_edge = cand
            if best_edge is not None:
                edges.add(best_edge)
            if self.prune and len(edges) > num_nodes - 1:
                adj = self._adjacency(edges, num_nodes)
                pe = max(edges, key=lambda e: (self._forman(adj, *e), e))
                if self._forman(adj, *pe) > 0:
                    edges.discard(pe)

        return self._rebuild(data, edges, num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="FormanRicciRewiring",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="B3 rewiring: augmented Forman-Ricci curvature SDRF add/prune.",
            paper_reference="Topping et al. ICLR 2022; Fesser & Weber LoG 2023",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"num_iterations": {"type": int, "min": 1}, "prune": {"type": bool}}


# ===========================================================================
# B6 — EffectiveResistanceRewiring
# ===========================================================================
class EffectiveResistanceRewiring(CurvatureRewiringBase):
    """GTR-style rewiring: add high-effective-resistance pairs, prune low-resistance edges."""

    def __init__(self, num_iterations: int = 3, prune: bool = True):
        super().__init__()
        self.num_iterations = int(num_iterations)
        self.prune = bool(prune)

    def transform(self, data: Data) -> Data:
        self._validate_iterations(self.num_iterations)
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 3:
            return data
        edges = self._undirected_edges(data.edge_index)

        for _ in range(self.num_iterations):
            resist = self._resistance_matrix(self._laplacian(edges, num_nodes))
            # Add the highest-resistance non-adjacent pair (worst bottleneck).
            best_r = -1.0
            best_pair = None
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if (i, j) in edges:
                        continue
                    r = float(resist[i, j])
                    if r > best_r:
                        best_r = r
                        best_pair = (i, j)
            if best_pair is not None:
                edges.add(best_pair)
            if self.prune and len(edges) > num_nodes - 1:
                resist = self._resistance_matrix(self._laplacian(edges, num_nodes))
                pe = min(edges, key=lambda e: (float(resist[e[0], e[1]]), e))
                edges.discard(pe)

        return self._rebuild(data, edges, num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="EffectiveResistanceRewiring",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="B6 rewiring: add high effective-resistance pairs, prune low ones (GTR).",
            paper_reference="Black et al. GTR, ICML 2023",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"num_iterations": {"type": int, "min": 1}, "prune": {"type": bool}}


# ===========================================================================
# B7 — SpectralGapRewiring
# ===========================================================================
class SpectralGapRewiring(CurvatureRewiringBase):
    """FoSR-style rewiring: greedily add the edge maximising the Laplacian spectral gap."""

    def __init__(self, num_iterations: int = 3):
        super().__init__()
        self.num_iterations = int(num_iterations)

    def transform(self, data: Data) -> Data:
        self._validate_iterations(self.num_iterations)
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 3:
            return data
        edges = self._undirected_edges(data.edge_index)

        for _ in range(self.num_iterations):
            best_gap = -1.0
            best_pair = None
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if (i, j) in edges:
                        continue
                    trial = edges | {(i, j)}
                    gap = self._algebraic_connectivity(self._laplacian(trial, num_nodes))
                    if gap > best_gap + _EPS:
                        best_gap = gap
                        best_pair = (i, j)
            if best_pair is None:
                break
            edges.add(best_pair)

        return self._rebuild(data, edges, num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="SpectralGapRewiring",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="B7 rewiring: add edges maximising the Laplacian spectral gap (FoSR).",
            paper_reference="Karhadkar et al. FoSR, ICLR 2023",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"num_iterations": {"type": int, "min": 1}}


# ===========================================================================
# B12 — ResistanceCurvatureRewiring
# ===========================================================================
class ResistanceCurvatureRewiring(CurvatureRewiringBase):
    """Rewiring by Devriendt-Lambiotte resistance curvature: add min-curvature pairs, prune max."""

    def __init__(self, num_iterations: int = 3, prune: bool = True):
        super().__init__()
        self.num_iterations = int(num_iterations)
        self.prune = bool(prune)

    def _node_curvature(self, resist: torch.Tensor, adj: list[set[int]]) -> list[float]:
        # p_i = 1 - (1/2) sum_{j ~ i} R_ij  (unit-weight edges).
        return [1.0 - 0.5 * sum(float(resist[i, j]) for j in adj[i]) for i in range(len(adj))]

    def _edge_curvature(self, resist: torch.Tensor, p: list[float], u: int, v: int) -> float:
        # kappa_uv = 2 (p_u + p_v) / R_uv.
        r = float(resist[u, v])
        return 2.0 * (p[u] + p[v]) / r if r > _EPS else 0.0

    def transform(self, data: Data) -> Data:
        self._validate_iterations(self.num_iterations)
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 3:
            return data
        edges = self._undirected_edges(data.edge_index)

        for _ in range(self.num_iterations):
            resist = self._resistance_matrix(self._laplacian(edges, num_nodes))
            adj = self._adjacency(edges, num_nodes)
            p = self._node_curvature(resist, adj)
            # Add the lowest-curvature non-adjacent pair (bottleneck).
            best_kappa = float("inf")
            best_pair = None
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if (i, j) in edges:
                        continue
                    kappa = self._edge_curvature(resist, p, i, j)
                    if kappa < best_kappa:
                        best_kappa = kappa
                        best_pair = (i, j)
            if best_pair is not None:
                edges.add(best_pair)
            if self.prune and len(edges) > num_nodes - 1:
                resist = self._resistance_matrix(self._laplacian(edges, num_nodes))
                adj = self._adjacency(edges, num_nodes)
                p = self._node_curvature(resist, adj)
                pe = max(edges, key=lambda e: (self._edge_curvature(resist, p, *e), e))
                edges.discard(pe)

        return self._rebuild(data, edges, num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="ResistanceCurvatureRewiring",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="B12 rewiring: Devriendt-Lambiotte resistance-curvature add/prune.",
            paper_reference="Devriendt & Lambiotte, J. Phys. Complex. 2022",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"num_iterations": {"type": int, "min": 1}, "prune": {"type": bool}}
