"""
Topological & higher-order lifts (catalogue Family A, zero-dependency Data->Data subset).

Three unary transforms that stay inside the ``transform(data) -> data`` purity contract
(single PyG ``Data`` in/out, no model, no dataset/labels) AND require no new dependency.
Most of Family A (A1/A2/A6/A7/A14/A15) produces simplicial/cell/combinatorial complexes that
are not PyG ``Data`` graphs, and the TDA lifts (A3/A4/A5, H1 persistence) need ``gudhi`` +
node coordinates; per the catalogue's own dependency guidance ("isolate in an extra") those are
deferred. The three shipped here realise hypergraph lifts as ordinary graphs (clique / star
expansion of the k-hop hypergraph) plus zero-dimensional persistent-homology features.

In scope: A11 ``HypergraphCliqueExpansion``, A10 ``HypergraphStarExpansion``,
A16 ``PersistentHomologyFeature`` (H0 only, union-find).

MILIA-custom idiom (``CustomTransformBase`` + ``TransformMetadata``), category ``structural``,
deterministic (no RNG). Single-module layout (``module_path: transforms``).
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


class TopologicalLiftBase(CustomTransformBase):
    """Shared helpers: adjacency, closed-k-hop hyperedges, incidence, union-find."""

    @staticmethod
    def _adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
        adj: list[set[int]] = [set() for _ in range(num_nodes)]
        for u, v in edge_index.t().tolist():
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
        return adj

    def _closed_khop(self, adj: list[set[int]], num_nodes: int, k: int) -> list[set[int]]:
        """Closed k-hop neighbourhood of each node: {v} plus all nodes within k hops."""
        hyperedges: list[set[int]] = []
        for v in range(num_nodes):
            frontier = {v}
            seen = {v}
            for _ in range(k):
                nxt: set[int] = set()
                for node in frontier:
                    nxt |= adj[node]
                nxt -= seen
                if not nxt:
                    break
                seen |= nxt
                frontier = nxt
            hyperedges.append(seen)
        return hyperedges

    @staticmethod
    def _incidence(hyperedges: list[set[int]], num_nodes: int) -> torch.Tensor:
        """H[i, e] = 1 if node i is a member of hyperedge e (float64)."""
        inc = torch.zeros((num_nodes, len(hyperedges)), dtype=torch.float64)
        for e, members in enumerate(hyperedges):
            for i in members:
                inc[i, e] = 1.0
        return inc


class _UnionFind:
    """Union-find with birth-time tracking for elder-rule H0 persistence."""

    def __init__(self, birth: list[float]):
        self.parent = list(range(len(birth)))
        self.birth = list(birth)  # birth time of each component root

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, death: float) -> tuple[float, float] | None:
        """Merge components; the younger root dies. Returns (birth, death) or None."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        # Elder rule: the component born LATER dies; the elder survives.
        if self.birth[ra] <= self.birth[rb]:
            elder, younger = ra, rb
        else:
            elder, younger = rb, ra
        pair = (self.birth[younger], death)
        self.parent[younger] = elder
        return pair


# ===========================================================================
# A11 — HypergraphCliqueExpansion
# ===========================================================================
class HypergraphCliqueExpansion(TopologicalLiftBase):
    """Lift to the k-hop hypergraph, then clique-expand to a weighted graph (A = HHᵀ − D_v)."""

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = int(k)

    def transform(self, data: Data) -> Data:
        if self.k < 1:
            raise TransformExecutionError(
                f"HypergraphCliqueExpansion requires k >= 1, got {self.k}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 2:
            return data

        adj = self._adjacency(data.edge_index, num_nodes)
        hyperedges = self._closed_khop(adj, num_nodes, self.k)
        inc = self._incidence(hyperedges, num_nodes)
        co = inc @ inc.t()  # co-occurrence counts; co[i,j] = #hyperedges containing both
        co.fill_diagonal_(0.0)

        src: list[int] = []
        dst: list[int] = []
        weight: list[float] = []
        for i in range(num_nodes):
            row = co[i]
            for j in range(i + 1, num_nodes):
                w = float(row[j])
                if w > 0.0:
                    src.extend((i, j))
                    dst.extend((j, i))
                    weight.extend((w, w))

        data.edge_index = (
            torch.tensor([src, dst], dtype=data.edge_index.dtype)
            if src
            else torch.empty((2, 0), dtype=data.edge_index.dtype)
        )
        data.edge_weight = torch.tensor(weight, dtype=torch.float32)
        # edge_attr semantics no longer align with the new edge set; drop it.
        if getattr(data, "edge_attr", None) is not None:
            data.edge_attr = None
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="HypergraphCliqueExpansion",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="A11 lift: k-hop hypergraph clique expansion to a weighted graph.",
            paper_reference="Standard hypergraph clique expansion (A = HHᵀ − D_v)",
            modifies_attributes=["edge_index", "edge_weight", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"k": {"type": int, "min": 1}}


# ===========================================================================
# A10 — HypergraphStarExpansion
# ===========================================================================
class HypergraphStarExpansion(TopologicalLiftBase):
    """Lift to the k-hop hypergraph, then star-expand: one meta-node per hyperedge."""

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = int(k)

    def transform(self, data: Data) -> Data:
        if self.k < 1:
            raise TransformExecutionError(
                f"HypergraphStarExpansion requires k >= 1, got {self.k}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 2:
            return data

        adj = self._adjacency(data.edge_index, num_nodes)
        hyperedges = self._closed_khop(adj, num_nodes, self.k)

        # Keep original edges; append one meta-node per hyperedge (id = num_nodes + e).
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        for e, members in enumerate(hyperedges):
            meta = num_nodes + e
            for i in sorted(members):
                src.extend((i, meta))
                dst.extend((meta, i))

        total_nodes = num_nodes + len(hyperedges)
        data.edge_index = torch.tensor([src, dst], dtype=data.edge_index.dtype)

        # Meta-node features = mean of member features; zero-pad edge_attr for star edges.
        base_x = getattr(data, "x", None)
        if base_x is not None:
            meta_x = torch.stack(
                [base_x[sorted(m)].to(base_x.dtype).mean(dim=0) for m in hyperedges]
            )
            data.x = torch.cat([base_x, meta_x], dim=0)
        ea = getattr(data, "edge_attr", None)
        if ea is not None:
            num_star = data.edge_index.size(1) - ea.size(0)
            pad_shape = (num_star,) if ea.dim() == 1 else (num_star, ea.size(1))
            data.edge_attr = torch.cat([ea, torch.zeros(pad_shape, dtype=ea.dtype)], dim=0)
        data.num_nodes = total_nodes
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="HypergraphStarExpansion",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="A10 lift: k-hop hypergraph star expansion with per-hyperedge meta-nodes.",
            paper_reference="Standard hypergraph star (bipartite incidence) expansion",
            modifies_attributes=["edge_index", "edge_attr", "x", "num_nodes"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"k": {"type": int, "min": 1}}


# ===========================================================================
# A16 — PersistentHomologyFeature (H0)
# ===========================================================================
class PersistentHomologyFeature(TopologicalLiftBase):
    """Attach 0-dimensional persistent-homology summary features (union-find, elder rule).

    Filtration = node degree (ascending); a vertex is born at its degree, an edge appears at
    ``max(deg(u), deg(v))``; on a merge the younger component dies (elder rule). The graph-level
    summary ``[total_persistence, persistence_entropy, num_finite_pairs, max_lifetime,
    mean_lifetime]`` is stored under ``data.persistence_h0``.
    """

    _NUM_FEATURES = 5

    def __init__(self, attr_name: str = "persistence_h0"):
        super().__init__()
        self.attr_name = str(attr_name)

    def transform(self, data: Data) -> Data:
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 1:
            data[self.attr_name] = torch.zeros((1, self._NUM_FEATURES), dtype=torch.float32)
            return data

        adj = self._adjacency(data.edge_index, num_nodes)
        filtration = [float(len(adj[v])) for v in range(num_nodes)]  # node-degree filtration

        # Edges sorted ascending by appearance value = max(f(u), f(v)); tie-break by endpoints.
        undirected = {
            (u, v) if u < v else (v, u) for u, v in data.edge_index.t().tolist() if u != v
        }
        edges_sorted = sorted(
            undirected, key=lambda e: (max(filtration[e[0]], filtration[e[1]]), e)
        )

        uf = _UnionFind(filtration)
        pairs: list[tuple[float, float]] = []
        for u, v in edges_sorted:
            death = max(filtration[u], filtration[v])
            pair = uf.union(u, v, death)
            if pair is not None and pair[1] > pair[0]:
                pairs.append(pair)

        feats = self._summarise(pairs)
        data[self.attr_name] = feats
        return data

    def _summarise(self, pairs: list[tuple[float, float]]) -> torch.Tensor:
        if not pairs:
            return torch.zeros((1, self._NUM_FEATURES), dtype=torch.float32)
        lifetimes = torch.tensor([d - b for b, d in pairs], dtype=torch.float64)
        total = float(lifetimes.sum())
        probs = lifetimes / lifetimes.sum().clamp_min(1e-12)
        entropy = float(-(probs * (probs.clamp_min(1e-12)).log()).sum())
        summary = [
            total,
            entropy,
            float(len(pairs)),
            float(lifetimes.max()),
            float(lifetimes.mean()),
        ]
        return torch.tensor([summary], dtype=torch.float32)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="PersistentHomologyFeature",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="A16 (H0): zero-dimensional persistent-homology summary features.",
            paper_reference="Persistent homology H0 via union-find (elder rule)",
            modifies_attributes=["persistence_h0"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"attr_name": {"type": str, "optional": True}}
