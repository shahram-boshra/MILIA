"""
Latent-space & self-supervised structural transforms (catalogue Family J, deterministic subset).

Three unary transforms with pure, zero-dependency, deterministic cores that satisfy the
``transform(data) -> data`` purity contract (single PyG ``Data`` in/out, no model, no dataset,
no labels). J1 (latent-space post-augmentation) is out of scope — it needs a trained model.
The learned variants of J3/J4 (struc2vec / RolX-NMF, word2vec-AWE) need gensim/sklearn; the
explicit feature cores shipped here are dependency-free.

In scope: J2 ``MotifCompression``, J3 ``StructuralRoleEncoding`` (ReFeX core),
J4 ``AnonymousWalkEmbedding``.

MILIA-custom idiom (``CustomTransformBase`` + ``TransformMetadata``), category ``structural``,
deterministic (no RNG). Single-module layout (``module_path: transforms``).
"""

from __future__ import annotations

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from milia_pipeline.transformations.custom_transforms import (
    CustomTransformBase,
    TransformExecutionError,
    TransformMetadata,
)

_AUTHOR = "MILIA Team"


class LatentStructuralBase(CustomTransformBase):
    """Shared helpers: adjacency, egonet features, anonymous-walk relabelling."""

    @staticmethod
    def _undirected_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
        return {(u, v) if u < v else (v, u) for u, v in edge_index.t().tolist() if u != v}

    @staticmethod
    def _adjacency(edges: set[tuple[int, int]], num_nodes: int) -> list[set[int]]:
        adj: list[set[int]] = [set() for _ in range(num_nodes)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    @staticmethod
    def _anonymous(walk: list[int]) -> tuple[int, ...]:
        """Relabel a walk by first-occurrence order: (a,b,a,c) -> (0,1,0,2)."""
        mapping: dict[int, int] = {}
        out: list[int] = []
        for node in walk:
            if node not in mapping:
                mapping[node] = len(mapping)
            out.append(mapping[node])
        return tuple(out)


# ===========================================================================
# J2 — MotifCompression
# ===========================================================================
class MotifCompression(LatentStructuralBase):
    """Contract each ring (cycle-basis motif) into a virtual super-node (ring rewiring)."""

    def __init__(self, max_ring_size: int = 8):
        super().__init__()
        self.max_ring_size = int(max_ring_size)

    def transform(self, data: Data) -> Data:
        if self.max_ring_size < 3:
            raise TransformExecutionError(
                f"MotifCompression requires max_ring_size >= 3, got {self.max_ring_size}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 3 or data.edge_index.size(1) == 0:
            return data

        graph = to_networkx(data, to_undirected=True)
        graph.add_nodes_from(range(num_nodes))
        rings = [c for c in nx.cycle_basis(graph) if 3 <= len(c) <= self.max_ring_size]

        # Union-find: nodes in the same (or fused) rings share a super-node.
        parent = list(range(num_nodes))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for ring in rings:
            anchor = ring[0]
            for node in ring[1:]:
                parent[find(node)] = find(anchor)

        roots = sorted({find(i) for i in range(num_nodes)})
        relabel = {r: i for i, r in enumerate(roots)}
        assign = [relabel[find(i)] for i in range(num_nodes)]
        num_super = len(roots)
        if num_super == num_nodes:  # no rings compressed
            return data

        new_edges: set[tuple[int, int]] = set()
        for u, v in self._undirected_edges(data.edge_index):
            a, b = assign[u], assign[v]
            if a != b:
                new_edges.add((a, b) if a < b else (b, a))

        src: list[int] = []
        dst: list[int] = []
        for u, v in sorted(new_edges):
            src.extend((u, v))
            dst.extend((v, u))
        out = Data(num_nodes=num_super)
        out.edge_index = (
            torch.tensor([src, dst], dtype=data.edge_index.dtype)
            if src
            else torch.empty((2, 0), dtype=data.edge_index.dtype)
        )
        base_x = getattr(data, "x", None)
        if base_x is not None:  # mean of member features
            agg = torch.zeros((num_super, base_x.size(-1)), dtype=base_x.dtype)
            counts = torch.zeros(num_super, dtype=base_x.dtype)
            idx = torch.tensor(assign, dtype=torch.long)
            agg.index_add_(0, idx, base_x)
            counts.index_add_(0, idx, torch.ones(num_nodes, dtype=base_x.dtype))
            out.x = agg / counts.clamp_min(1).view(-1, 1)
        return out

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="MotifCompression",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="J2 latent-structural: contract ring motifs into virtual super-nodes.",
            paper_reference="Motif-based coarsening (ring rewiring)",
            modifies_attributes=["edge_index", "x", "num_nodes"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"max_ring_size": {"type": int, "min": 3}}


# ===========================================================================
# J3 — StructuralRoleEncoding (ReFeX core)
# ===========================================================================
class StructuralRoleEncoding(LatentStructuralBase):
    """Recursive structural (ReFeX) role features: local + egonet, aggregated over neighbours."""

    def __init__(self, iterations: int = 2, attr_name: str = "struct_roles"):
        super().__init__()
        self.iterations = int(iterations)
        self.attr_name = str(attr_name)

    def _base_features(self, adj: list[set[int]], num_nodes: int) -> torch.Tensor:
        feats = torch.zeros((num_nodes, 3), dtype=torch.float64)
        for v in range(num_nodes):
            egonet = adj[v] | {v}
            internal = 0
            external = 0
            for u in egonet:
                for w in adj[u]:
                    if w in egonet:
                        internal += 1  # counts each internal edge twice
                    else:
                        external += 1
            feats[v, 0] = float(len(adj[v]))  # degree
            feats[v, 1] = internal / 2.0  # egonet-internal edges
            feats[v, 2] = float(external)  # egonet-external edge endpoints
        return feats

    def transform(self, data: Data) -> Data:
        if self.iterations < 0:
            raise TransformExecutionError(
                f"StructuralRoleEncoding requires iterations >= 0, got {self.iterations}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 1:
            data[self.attr_name] = torch.zeros((0, 3), dtype=torch.float32)
            return data

        edges = self._undirected_edges(data.edge_index)
        adj = self._adjacency(edges, num_nodes)
        feats = self._base_features(adj, num_nodes)

        for _ in range(self.iterations):
            mean_agg = torch.zeros_like(feats)
            sum_agg = torch.zeros_like(feats)
            for v in range(num_nodes):
                if adj[v]:
                    neigh = torch.stack([feats[u] for u in sorted(adj[v])])
                    mean_agg[v] = neigh.mean(dim=0)
                    sum_agg[v] = neigh.sum(dim=0)
            feats = torch.cat([feats, mean_agg, sum_agg], dim=1)

        data[self.attr_name] = feats.to(torch.float32)
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="StructuralRoleEncoding",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="J3 latent-structural: recursive (ReFeX) structural-role node features.",
            paper_reference="Henderson et al. 2011 (ReFeX); RolX 2012",
            modifies_attributes=["struct_roles"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "iterations": {"type": int, "min": 0},
            "attr_name": {"type": str, "optional": True},
        }


# ===========================================================================
# J4 — AnonymousWalkEmbedding
# ===========================================================================
class AnonymousWalkEmbedding(LatentStructuralBase):
    """Graph feature = distribution over anonymous walks of a fixed length (Ivanov & Burnaev)."""

    def __init__(
        self, walk_length: int = 3, max_walks_per_node: int = 2000, attr_name: str = "awe"
    ):
        super().__init__()
        self.walk_length = int(walk_length)
        self.max_walks_per_node = int(max_walks_per_node)
        self.attr_name = str(attr_name)

    def _patterns(self, length: int) -> list[tuple[int, ...]]:
        """All valid anonymous-walk patterns of `length` steps (length+1 nodes): restricted-
        growth strings with consecutive elements distinct (no self-loops), canonically sorted."""
        results: list[tuple[int, ...]] = []

        def extend(seq: list[int], max_label: int) -> None:
            if len(seq) == length + 1:
                results.append(tuple(seq))
                return
            for nxt in range(max_label + 2):
                if nxt == seq[-1]:  # consecutive nodes of a walk are distinct
                    continue
                extend([*seq, nxt], max(max_label, nxt))

        extend([0], 0)
        return sorted(results)

    def transform(self, data: Data) -> Data:
        if self.walk_length < 1:
            raise TransformExecutionError(
                f"AnonymousWalkEmbedding requires walk_length >= 1, got {self.walk_length}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        patterns = self._patterns(self.walk_length)
        pattern_index = {p: i for i, p in enumerate(patterns)}
        counts = torch.zeros(len(patterns), dtype=torch.float64)

        if num_nodes is not None and num_nodes >= 1 and data.edge_index.size(1) > 0:
            adj = self._adjacency(self._undirected_edges(data.edge_index), num_nodes)
            for start in range(num_nodes):
                walks_from_start = 0
                stack: list[list[int]] = [[start]]
                while stack:
                    walk = stack.pop()
                    if len(walk) == self.walk_length + 1:
                        counts[pattern_index[self._anonymous(walk)]] += 1.0
                        walks_from_start += 1
                        if walks_from_start >= self.max_walks_per_node:
                            break
                        continue
                    for nbr in sorted(adj[walk[-1]], reverse=True):
                        stack.append([*walk, nbr])

        total = counts.sum()
        dist = (counts / total) if float(total) > 0 else counts
        data[self.attr_name] = dist.to(torch.float32).view(1, -1)
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="AnonymousWalkEmbedding",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="J4 latent-structural: anonymous-walk distribution graph embedding.",
            paper_reference="Ivanov & Burnaev, ICML 2018",
            modifies_attributes=["awe"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "walk_length": {"type": int, "min": 1},
            "max_walks_per_node": {"type": int, "min": 1},
            "attr_name": {"type": str, "optional": True},
        }
