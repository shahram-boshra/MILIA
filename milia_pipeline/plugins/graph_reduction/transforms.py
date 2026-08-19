"""
Sparsification & coarsening transforms (catalogue Family F.a/F.b, in-scope subset).

Eight unary graph-reduction transforms (G -> smaller/sparser G) satisfying the
``transform(data) -> data`` purity contract: single ``Data`` in/out, no model, no dataset,
no labels. MILIA-custom idiom (``CustomTransformBase`` + ``TransformMetadata``), category
``structural``.

In scope (this file):
  F.a sparsification : F1 ``EffectiveResistanceSparsify``, F3 ``SpannerSparsify``,
                       F5 ``KNeighborSparsify``, F6 ``LocalDegreeSparsify``.
  F.b coarsen/decomp : F8 ``KronReduction``, F11 ``HeavyEdgeMatching``,
                       F14 ``KCoreDecomposition``, F15 ``KTrussDecomposition``.

Deferred (Blueprint Family F design block): F2/F4/F7/F9/F10/F12/F13 (barrier-function /
edge-strength / iterative-affinity / clustering — correctness or dependency risk). Out of
scope: F.c F16–F20 (condensation — needs a GNN training loop). Zero new dependencies
(``networkx`` + ``torch`` only). Single-module layout (``module_path: transforms``).
"""

from __future__ import annotations

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx

from milia_pipeline.transformations.custom_transforms import (
    CustomTransformBase,
    TransformExecutionError,
    TransformMetadata,
)

_AUTHOR = "MILIA Team"
_EPS = 1e-12


class GraphReductionBase(CustomTransformBase):
    """Shared helpers: seeded RNG, weights, edge sets, Laplacian, subset induction."""

    # -- reproducibility -----------------------------------------------------
    def _rng(self):
        import numpy as np

        return np.random.default_rng(getattr(self, "seed", None))

    # -- weights / edges -----------------------------------------------------
    @staticmethod
    def _edge_weight(data: Data) -> torch.Tensor:
        """Per-edge scalar weight: from edge_weight, else |edge_attr|, else ones."""
        ew = getattr(data, "edge_weight", None)
        if ew is not None:
            return ew.to(torch.float64).view(-1)
        ea = getattr(data, "edge_attr", None)
        if ea is not None:
            ea = ea.to(torch.float64)
            return ea.abs().view(-1) if ea.dim() == 1 else ea.norm(dim=1)
        return torch.ones(data.edge_index.size(1), dtype=torch.float64)

    @staticmethod
    def _undirected_weight_map(
        edge_index: torch.Tensor, weight: torch.Tensor
    ) -> dict[tuple[int, int], float]:
        """Canonical (min,max) -> max incident weight, self-loops excluded."""
        wmap: dict[tuple[int, int], float] = {}
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        wl = weight.tolist()
        for u, v, w in zip(src, dst, wl, strict=True):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            wmap[key] = max(wmap.get(key, float("-inf")), float(w))
        return wmap

    def _adjacency(self, edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
        adj: list[set[int]] = [set() for _ in range(num_nodes)]
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst, strict=True):
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
        return [sorted(nbrs) for nbrs in adj]

    def _laplacian(self, data: Data, num_nodes: int) -> torch.Tensor:
        """Weighted combinatorial Laplacian L = D - W (float64, undirected)."""
        wmat = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
        weight = self._edge_weight(data)
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        for u, v, w in zip(src, dst, weight.tolist(), strict=True):
            if u != v:
                wmat[u, v] = w
                wmat[v, u] = w
        deg = wmat.sum(dim=1)
        return torch.diag(deg) - wmat

    # -- construction from an undirected edge set ----------------------------
    def _rebuild_from_edges(self, data: Data, edges: set[tuple[int, int]], num_nodes: int) -> Data:
        """Keep the same node set; rebuild edge_index from a kept undirected edge set,
        carrying edge_attr for retained (undirected) edges."""
        out = data.clone()
        ea = getattr(data, "edge_attr", None)
        attr_lookup = None
        if ea is not None:
            attr_lookup = {}
            src = data.edge_index[0].tolist()
            dst = data.edge_index[1].tolist()
            for i, (u, v) in enumerate(zip(src, dst, strict=True)):
                key = (u, v) if u < v else (v, u)
                attr_lookup.setdefault(key, i)

        ordered = sorted(edges)
        if not ordered:
            out.edge_index = torch.empty((2, 0), dtype=data.edge_index.dtype)
            if ea is not None:
                shape = (0,) if ea.dim() == 1 else (0, ea.size(1))
                out.edge_attr = torch.zeros(shape, dtype=ea.dtype)
            if getattr(out, "edge_weight", None) is not None:
                out.edge_weight = torch.empty(0, dtype=out.edge_weight.dtype)
            return out

        src: list[int] = []
        dst: list[int] = []
        attr_rows: list[int] = []
        for u, v in ordered:
            src.extend((u, v))
            dst.extend((v, u))
            if attr_lookup is not None:
                idx = attr_lookup.get((u, v), attr_lookup.get((v, u)))
                attr_rows.extend((idx, idx))
        out.edge_index = torch.tensor([src, dst], dtype=data.edge_index.dtype)
        if ea is not None:
            out.edge_attr = ea[torch.tensor(attr_rows, dtype=torch.long)]
        if getattr(out, "edge_weight", None) is not None:
            out.edge_weight = None
        out.num_nodes = num_nodes
        return out


# ===========================================================================
# F1 — EffectiveResistanceSparsify
# ===========================================================================
class EffectiveResistanceSparsify(GraphReductionBase):
    """Spectral sparsifier: sample edges ∝ w_e·R_e and reweight (Spielman–Srivastava)."""

    def __init__(self, ratio: float = 0.5, seed: int | None = None):
        super().__init__()
        self.ratio = float(ratio)
        self.seed = None if seed is None else int(seed)

    def transform(self, data: Data) -> Data:
        if not 0.0 < self.ratio <= 1.0:
            raise TransformExecutionError(
                f"EffectiveResistanceSparsify requires 0 < ratio <= 1, got {self.ratio}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        wmap = self._undirected_weight_map(data.edge_index, self._edge_weight(data))
        edges = sorted(wmap)
        if num_nodes is None or num_nodes < 3 or len(edges) < 2:
            return data

        lap = self._laplacian(data, num_nodes)
        lpinv = torch.linalg.pinv(lap)  # Moore–Penrose pseudoinverse
        resist = []
        for u, v in edges:
            r = lpinv[u, u] + lpinv[v, v] - 2.0 * lpinv[u, v]
            resist.append(max(float(r), 0.0))
        weights = torch.tensor([wmap[e] for e in edges], dtype=torch.float64)
        scores = weights * torch.tensor(resist, dtype=torch.float64)
        total = float(scores.sum())
        if total <= _EPS:
            return data
        probs = (scores / total).clamp_min(0.0)

        q = max(1, int(round(self.ratio * len(edges))))
        rng = self._rng()
        picks = rng.choice(len(edges), size=q, replace=True, p=probs.numpy())
        acc: dict[tuple[int, int], float] = {}
        counts: dict[tuple[int, int], int] = {}
        for idx in picks.tolist():
            counts[edges[idx]] = counts.get(edges[idx], 0) + 1
        for e, c in counts.items():
            p_e = float(probs[edges.index(e)])
            acc[e] = c * float(wmap[e]) / (q * p_e) if p_e > 0 else float(wmap[e])

        kept = set(acc)
        out = self._rebuild_from_edges(data, kept, num_nodes)
        # Record the spectral reweight as scalar edge_weight (symmetric, per directed edge).
        ei = out.edge_index
        ew = []
        for a, b in zip(ei[0].tolist(), ei[1].tolist(), strict=True):
            key = (a, b) if a < b else (b, a)
            ew.append(acc.get(key, 0.0))
        out.edge_weight = torch.tensor(ew, dtype=torch.float32)
        return out

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="EffectiveResistanceSparsify",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="F1 sparsification: effective-resistance spectral edge sampling.",
            paper_reference="Spielman & Srivastava, STOC 2008 / SICOMP 2011",
            modifies_attributes=["edge_index", "edge_attr", "edge_weight"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "ratio": {"type": float, "min": 0.0, "max": 1.0},
            "seed": {"type": int, "optional": True},
        }


# ===========================================================================
# F3 — SpannerSparsify
# ===========================================================================
class SpannerSparsify(GraphReductionBase):
    """t-spanner preserving pairwise distances within the stretch factor (Baswana–Sen)."""

    def __init__(self, stretch: int = 3, seed: int | None = None):
        super().__init__()
        self.stretch = int(stretch)
        self.seed = None if seed is None else int(seed)

    def transform(self, data: Data) -> Data:
        if self.stretch < 1:
            raise TransformExecutionError(
                f"SpannerSparsify requires stretch >= 1, got {self.stretch}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 3 or data.edge_index.size(1) == 0:
            return data
        graph = to_networkx(data, to_undirected=True)
        graph.add_nodes_from(range(num_nodes))
        try:
            spanner = nx.spanner(graph, self.stretch, seed=self.seed)
        except (nx.NetworkXError, ValueError) as exc:
            raise TransformExecutionError(
                "SpannerSparsify: spanner construction failed.",
                transform_name=self._metadata.name,
                original_error=exc,
            ) from exc
        kept = {(u, v) if u < v else (v, u) for u, v in spanner.edges() if u != v}
        return self._rebuild_from_edges(data, kept, num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="SpannerSparsify",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="F3 sparsification: t-spanner (distance-preserving) edge subset.",
            paper_reference="Althöfer et al. 1993; Baswana & Sen 2007 (networkx.spanner)",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"stretch": {"type": int, "min": 1}, "seed": {"type": int, "optional": True}}


# ===========================================================================
# F5 — KNeighborSparsify
# ===========================================================================
class KNeighborSparsify(GraphReductionBase):
    """Keep the top-k highest-weight edges per node (union over endpoints)."""

    def __init__(self, k: int = 4):
        super().__init__()
        self.k = int(k)

    def transform(self, data: Data) -> Data:
        if self.k < 1:
            raise TransformExecutionError(
                f"KNeighborSparsify requires k >= 1, got {self.k}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or data.edge_index.size(1) == 0:
            return data
        weight = self._edge_weight(data)
        incident: list[list[tuple[float, int, int]]] = [[] for _ in range(num_nodes)]
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        for u, v, w in zip(src, dst, weight.tolist(), strict=True):
            if u != v:
                incident[u].append((float(w), v, u))
        kept: set[tuple[int, int]] = set()
        for node in range(num_nodes):
            ranked = sorted(incident[node], key=lambda t: (-t[0], t[1]))[: self.k]
            for _, v, u in ranked:
                kept.add((u, v) if u < v else (v, u))
        return self._rebuild_from_edges(data, kept, num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="KNeighborSparsify",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="F5 sparsification: keep top-k highest-weight edges per node.",
            paper_reference="Standard local top-k sparsification",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"k": {"type": int, "min": 1}}


# ===========================================================================
# F6 — LocalDegreeSparsify
# ===========================================================================
class LocalDegreeSparsify(GraphReductionBase):
    """Keep edges to each node's ``deg(v)^alpha`` highest-degree neighbours (Lindner et al.)."""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)

    def transform(self, data: Data) -> Data:
        if not 0.0 <= self.alpha <= 1.0:
            raise TransformExecutionError(
                f"LocalDegreeSparsify requires 0 <= alpha <= 1, got {self.alpha}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or data.edge_index.size(1) == 0:
            return data
        adj = self._adjacency(data.edge_index, num_nodes)
        degree = [len(a) for a in adj]
        kept: set[tuple[int, int]] = set()
        for node in range(num_nodes):
            if degree[node] == 0:
                continue
            keep_count = max(1, int(degree[node] ** self.alpha))
            ranked = sorted(adj[node], key=lambda nbr: (-degree[nbr], nbr))[:keep_count]
            for nbr in ranked:
                kept.add((node, nbr) if node < nbr else (nbr, node))
        return self._rebuild_from_edges(data, kept, num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="LocalDegreeSparsify",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="F6 sparsification: keep edges to local hub (high-degree) neighbours.",
            paper_reference="Lindner, Staudt, Hamann, Meyerhenke, Wagner 2015",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"alpha": {"type": float, "min": 0.0, "max": 1.0}}


# ===========================================================================
# F8 — KronReduction
# ===========================================================================
class KronReduction(GraphReductionBase):
    """Schur-complement Laplacian reduction onto a retained node subset (Dörfler–Bullo)."""

    def __init__(self, ratio: float = 0.5):
        super().__init__()
        self.ratio = float(ratio)

    def transform(self, data: Data) -> Data:
        if not 0.0 < self.ratio < 1.0:
            raise TransformExecutionError(
                f"KronReduction requires 0 < ratio < 1, got {self.ratio}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 3:
            return data
        adj = self._adjacency(data.edge_index, num_nodes)
        degree = [len(a) for a in adj]
        keep_count = max(2, int(round(self.ratio * num_nodes)))
        if keep_count >= num_nodes:
            return data
        alpha = sorted(range(num_nodes), key=lambda n: (-degree[n], n))[:keep_count]
        alpha = sorted(alpha)
        beta = [n for n in range(num_nodes) if n not in set(alpha)]

        lap = self._laplacian(data, num_nodes)
        a_idx = torch.tensor(alpha, dtype=torch.long)
        b_idx = torch.tensor(beta, dtype=torch.long)
        l_aa = lap[a_idx][:, a_idx]
        l_ab = lap[a_idx][:, b_idx]
        l_bb = lap[b_idx][:, b_idx]
        try:
            l_bb_inv = torch.linalg.inv(l_bb)
        except RuntimeError:
            l_bb_inv = torch.linalg.pinv(l_bb)
        l_red = l_aa - l_ab @ l_bb_inv @ l_ab.t()

        # Reduced graph: off-diagonal -L_red[i,j] are the (nonnegative) edge weights.
        k = len(alpha)
        src: list[int] = []
        dst: list[int] = []
        w: list[float] = []
        for i in range(k):
            for j in range(i + 1, k):
                weight = -float(l_red[i, j])
                if weight > _EPS:
                    src.extend((i, j))
                    dst.extend((j, i))
                    w.extend((weight, weight))
        out = Data(num_nodes=k)
        out.edge_index = (
            torch.tensor([src, dst], dtype=data.edge_index.dtype)
            if src
            else torch.empty((2, 0), dtype=data.edge_index.dtype)
        )
        out.edge_weight = torch.tensor(w, dtype=torch.float32)
        base_x = getattr(data, "x", None)
        if base_x is not None:
            out.x = base_x[a_idx]
        return out

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="KronReduction",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="F8 coarsening: Schur-complement (Kron) Laplacian reduction onto a subset.",
            paper_reference="Dörfler & Bullo, IEEE TCAS-I 2013",
            modifies_attributes=["edge_index", "edge_weight", "x", "num_nodes"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"ratio": {"type": float, "min": 0.0, "max": 1.0}}


# ===========================================================================
# F11 — HeavyEdgeMatching
# ===========================================================================
class HeavyEdgeMatching(GraphReductionBase):
    """Contract a heavy-edge maximal matching into super-nodes (Graclus coarsening)."""

    def __init__(self):
        super().__init__()

    def transform(self, data: Data) -> Data:
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 2 or data.edge_index.size(1) == 0:
            return data
        wmap = self._undirected_weight_map(data.edge_index, self._edge_weight(data))
        # Greedy heaviest-first maximal matching (deterministic tie-break by node ids).
        ordered = sorted(wmap.items(), key=lambda kv: (-kv[1], kv[0]))
        matched: set[int] = set()
        cluster = list(range(num_nodes))
        for (u, v), _w in ordered:
            if u not in matched and v not in matched:
                matched.add(u)
                matched.add(v)
                cluster[max(u, v)] = min(u, v)

        # Relabel clusters to a contiguous 0..K-1 range (deterministic).
        roots = sorted(set(cluster))
        remap = {r: i for i, r in enumerate(roots)}
        assign = [remap[cluster[n]] for n in range(num_nodes)]
        num_super = len(roots)

        new_edges: dict[tuple[int, int], float] = {}
        for (u, v), w in wmap.items():
            a, b = assign[u], assign[v]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            new_edges[key] = new_edges.get(key, 0.0) + w

        out = Data(num_nodes=num_super)
        if new_edges:
            src: list[int] = []
            dst: list[int] = []
            w: list[float] = []
            for (a, b), weight in sorted(new_edges.items()):
                src.extend((a, b))
                dst.extend((b, a))
                w.extend((weight, weight))
            out.edge_index = torch.tensor([src, dst], dtype=data.edge_index.dtype)
            out.edge_weight = torch.tensor(w, dtype=torch.float32)
        else:
            out.edge_index = torch.empty((2, 0), dtype=data.edge_index.dtype)
            out.edge_weight = torch.empty(0, dtype=torch.float32)

        base_x = getattr(data, "x", None)
        if base_x is not None:  # sum features over merged nodes
            agg = torch.zeros((num_super, base_x.size(-1)), dtype=base_x.dtype)
            agg.index_add_(0, torch.tensor(assign, dtype=torch.long), base_x)
            out.x = agg
        return out

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="HeavyEdgeMatching",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="F11 coarsening: contract a heavy-edge maximal matching into super-nodes.",
            paper_reference="Dhillon, Guan, Kulis 2007 (Graclus)",
            modifies_attributes=["edge_index", "edge_weight", "x", "num_nodes"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {}


# ===========================================================================
# F14 — KCoreDecomposition
# ===========================================================================
class KCoreDecomposition(GraphReductionBase):
    """Induce the k-core: recursively remove nodes of degree < k (Batagelj–Zaveršnik)."""

    def __init__(self, k: int = 2):
        super().__init__()
        self.k = int(k)

    def _induce(self, data: Data, keep_nodes: list[int], num_nodes: int) -> Data:
        if not keep_nodes:
            out = Data(num_nodes=0)
            out.edge_index = torch.empty((2, 0), dtype=data.edge_index.dtype)
            return out
        subset = torch.tensor(sorted(keep_nodes), dtype=torch.long)
        ea = getattr(data, "edge_attr", None)
        new_ei, new_ea = subgraph(
            subset, data.edge_index, edge_attr=ea, relabel_nodes=True, num_nodes=num_nodes
        )
        out = Data(edge_index=new_ei, num_nodes=int(subset.numel()))
        if new_ea is not None:
            out.edge_attr = new_ea
        base_x = getattr(data, "x", None)
        if base_x is not None:
            out.x = base_x[subset]
        return out

    def transform(self, data: Data) -> Data:
        if self.k < 1:
            raise TransformExecutionError(
                f"KCoreDecomposition requires k >= 1, got {self.k}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes == 0:
            return data
        graph = to_networkx(data, to_undirected=True)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        core = nx.k_core(graph, k=self.k)
        return self._induce(data, list(core.nodes()), num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="KCoreDecomposition",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="F14 decomposition: induce the k-core subgraph.",
            paper_reference="Batagelj & Zaveršnik 2003 (networkx.k_core)",
            modifies_attributes=["edge_index", "edge_attr", "x", "num_nodes"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"k": {"type": int, "min": 1}}


# ===========================================================================
# F15 — KTrussDecomposition
# ===========================================================================
class KTrussDecomposition(GraphReductionBase):
    """Retain edges contained in at least (k-2) triangles: the k-truss (Cohen)."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = int(k)

    def transform(self, data: Data) -> Data:
        if self.k < 3:
            raise TransformExecutionError(
                f"KTrussDecomposition requires k >= 3, got {self.k}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes == 0:
            return data
        graph = to_networkx(data, to_undirected=True)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        truss = nx.k_truss(graph, k=self.k)
        kept = {(u, v) if u < v else (v, u) for u, v in truss.edges() if u != v}
        return self._rebuild_from_edges(data, kept, num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="KTrussDecomposition",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="F15 decomposition: retain edges in at least (k-2) triangles (k-truss).",
            paper_reference="Cohen 2008 (networkx.k_truss)",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"k": {"type": int, "min": 3}}
