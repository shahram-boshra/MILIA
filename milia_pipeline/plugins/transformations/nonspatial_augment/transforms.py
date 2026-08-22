"""
Strict non-spatial augmentations (catalogue Family G, in-scope subset).

Six unary, structure/feature-perturbing transforms (G -> G') that satisfy the
``transform(data) -> data`` purity contract: single ``Data`` in/out, no model, no
dataset, no labels. MILIA-custom idiom (``CustomTransformBase`` + ``TransformMetadata``),
category ``structural``. Every stochastic transform takes an explicit ``seed`` and is
reproducible via a local ``numpy`` generator (no global RNG state) per Phase 0.4.

In scope (this file): G2 ``RandomEdgeAdd``, G5 ``EdgeFeatureMasking``,
G7 ``RandomWalkSubgraphCrop``, G8 ``PPRSubgraphSampling``, G12 ``NASANeighborReplace``,
G15 ``SubmodularFeatureSalience``.

Deferred (needs-model / needs-dataset, see Blueprint Family G design block): G6
DropMessage, G9 ifMixup, G10 G-Mixup, G11 GraphTransplant, G13 FLAG, G14 SimGCL.
Already shipped in ``pyg_augmentation`` (do not duplicate): G1/G3/G4 + RandomNodeSample.

Single-module layout with ``module_path: transforms`` matches the plugin loader
(``_load_transform_class`` execs this file as a standalone module: absolute imports only).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx

from milia_pipeline.transformations.custom_transforms import (
    CustomTransformBase,
    TransformExecutionError,
    TransformMetadata,
)

_AUTHOR = "MILIA Team"


class NonSpatialAugmentBase(CustomTransformBase):
    """Shared helpers for Family G: seeded RNG, edge-set ops, subset induction."""

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.seed = None if seed is None else int(seed)

    # -- reproducibility -----------------------------------------------------
    def _rng(self) -> np.random.Generator:
        """Local, seeded NumPy generator (never touches global RNG state)."""
        return np.random.default_rng(self.seed)

    # -- edge helpers --------------------------------------------------------
    @staticmethod
    def _undirected_edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
        """Canonical (min, max) undirected edge set, self-loops excluded."""
        edges: set[tuple[int, int]] = set()
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst, strict=True):
            if u != v:
                edges.add((u, v) if u < v else (v, u))
        return edges

    @staticmethod
    def _edge_attr_width(edge_attr: torch.Tensor | None) -> int:
        """Feature width of edge_attr (0 if absent; 1 if 1-D)."""
        if edge_attr is None:
            return 0
        return 1 if edge_attr.dim() == 1 else edge_attr.size(1)

    @staticmethod
    def _zero_edge_rows(edge_attr: torch.Tensor, count: int) -> torch.Tensor:
        """Zero edge_attr rows matching the dtype/rank of an existing edge_attr."""
        if edge_attr.dim() == 1:
            return torch.zeros(count, dtype=edge_attr.dtype)
        return torch.zeros((count, edge_attr.size(1)), dtype=edge_attr.dtype)

    def _adjacency(self, edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
        """Undirected neighbour lists (deterministic, sorted)."""
        adj: list[set[int]] = [set() for _ in range(num_nodes)]
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst, strict=True):
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
        return [sorted(nbrs) for nbrs in adj]

    # -- subset induction ----------------------------------------------------
    def _induce_subset(self, data: Data, keep: list[int]) -> Data:
        """Induce the subgraph on ``keep`` (relabelled 0..K-1), carrying x/pos/edge_attr."""
        num_nodes = data.num_nodes
        subset = torch.tensor(sorted(set(keep)), dtype=torch.long)
        edge_attr = getattr(data, "edge_attr", None)
        new_edge_index, new_edge_attr = subgraph(
            subset,
            data.edge_index,
            edge_attr=edge_attr,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        out = Data(edge_index=new_edge_index, num_nodes=int(subset.numel()))
        if new_edge_attr is not None:
            out.edge_attr = new_edge_attr
        base_x = getattr(data, "x", None)
        if base_x is not None:
            out.x = base_x[subset]
        base_pos = getattr(data, "pos", None)
        if base_pos is not None:
            out.pos = base_pos[subset]
        return out


# ---------------------------------------------------------------------------
# G2 — RandomEdgeAdd
# ---------------------------------------------------------------------------
class RandomEdgeAdd(NonSpatialAugmentBase):
    """Add a seeded random set of non-existent (undirected) edges (GraphCL perturbation)."""

    def __init__(self, p: float = 0.1, seed: int | None = None):
        super().__init__(seed=seed)
        self.p = float(p)

    def transform(self, data: Data) -> Data:
        if self.p < 0.0:
            raise TransformExecutionError(
                f"RandomEdgeAdd requires p >= 0, got {self.p}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes < 2 or self.p == 0.0:
            return data

        existing = self._undirected_edge_set(data.edge_index)
        max_possible = num_nodes * (num_nodes - 1) // 2
        capacity = max_possible - len(existing)
        num_add = min(capacity, int(round(self.p * max(len(existing), 1))))
        if num_add <= 0:
            return data

        rng = self._rng()
        added: set[tuple[int, int]] = set()
        # Rejection sampling — scalable on sparse molecular graphs (avoids O(N^2) enumeration).
        max_tries = 20 * num_add + 100
        tries = 0
        while len(added) < num_add and tries < max_tries:
            tries += 1
            i = int(rng.integers(0, num_nodes))
            j = int(rng.integers(0, num_nodes))
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            if pair in existing or pair in added:
                continue
            added.add(pair)

        if not added:
            return data

        new_pairs = sorted(added)
        src: list[int] = []
        dst: list[int] = []
        for i, j in new_pairs:
            src.extend((i, j))
            dst.extend((j, i))
        extra = torch.tensor([src, dst], dtype=data.edge_index.dtype)
        data.edge_index = torch.cat([data.edge_index, extra], dim=1)

        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            pad = self._zero_edge_rows(edge_attr, extra.size(1))
            data.edge_attr = torch.cat([edge_attr, pad], dim=0)
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="RandomEdgeAdd",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="G2 non-spatial augmentation: add seeded random non-existent edges.",
            paper_reference="GraphCL edge perturbation (You et al., NeurIPS 2020)",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "p": {"type": float, "min": 0.0},
            "seed": {"type": int, "optional": True},
        }


# ---------------------------------------------------------------------------
# G5 — EdgeFeatureMasking
# ---------------------------------------------------------------------------
class EdgeFeatureMasking(NonSpatialAugmentBase):
    """Mask a seeded random subset of edge-feature dimensions (GCA family)."""

    def __init__(self, p: float = 0.1, mask_value: float = 0.0, seed: int | None = None):
        super().__init__(seed=seed)
        self.p = float(p)
        self.mask_value = float(mask_value)

    def transform(self, data: Data) -> Data:
        if not 0.0 <= self.p <= 1.0:
            raise TransformExecutionError(
                f"EdgeFeatureMasking requires 0 <= p <= 1, got {self.p}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        edge_attr = getattr(data, "edge_attr", None)
        # Fail-open no-op when there is no multi-dimensional edge feature to mask.
        if edge_attr is None or edge_attr.dim() < 2 or edge_attr.size(1) == 0:
            return data

        num_dims = edge_attr.size(1)
        num_mask = int(round(self.p * num_dims))
        if num_mask <= 0:
            return data

        rng = self._rng()
        chosen = rng.choice(num_dims, size=num_mask, replace=False)
        cols = torch.tensor(np.sort(chosen), dtype=torch.long)
        masked = edge_attr.clone()
        masked[:, cols] = torch.as_tensor(self.mask_value, dtype=masked.dtype)
        data.edge_attr = masked
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="EdgeFeatureMasking",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="G5 non-spatial augmentation: mask random edge-feature dimensions.",
            paper_reference="GCA / GraphCL feature masking",
            required_edge_features=["edge_attr"],
            modifies_attributes=["edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "p": {"type": float, "min": 0.0, "max": 1.0},
            "mask_value": {"type": float},
            "seed": {"type": int, "optional": True},
        }


# ---------------------------------------------------------------------------
# G7 — RandomWalkSubgraphCrop
# ---------------------------------------------------------------------------
class RandomWalkSubgraphCrop(NonSpatialAugmentBase):
    """Crop a connected subgraph via a seeded random walk or BFS (GraphCL / GCC).

    Distinct from ``pyg_augmentation.RandomNodeSample`` (independent node sampling): this
    grows a *connected* region, so ``allow_override_builtin: false`` is respected.
    """

    def __init__(self, ratio: float = 0.5, mode: str = "rw", seed: int | None = None):
        super().__init__(seed=seed)
        self.ratio = float(ratio)
        self.mode = str(mode)

    def transform(self, data: Data) -> Data:
        if not 0.0 < self.ratio <= 1.0:
            raise TransformExecutionError(
                f"RandomWalkSubgraphCrop requires 0 < ratio <= 1, got {self.ratio}.",
                transform_name=self._metadata.name,
            )
        if self.mode not in ("rw", "bfs"):
            raise TransformExecutionError(
                f"RandomWalkSubgraphCrop mode must be 'rw' or 'bfs', got '{self.mode}'.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes <= 1:
            return data

        target = max(1, int(round(self.ratio * num_nodes)))
        if target >= num_nodes:
            return data

        rng = self._rng()
        adj = self._adjacency(data.edge_index, num_nodes)
        start = int(rng.integers(0, num_nodes))
        keep: set[int] = {start}

        if self.mode == "bfs":
            frontier = [start]
            while frontier and len(keep) < target:
                node = frontier.pop(0)
                for nbr in adj[node]:
                    if nbr not in keep:
                        keep.add(nbr)
                        frontier.append(nbr)
                        if len(keep) >= target:
                            break
        else:  # random walk with restart on dead ends
            current = start
            stall = 0
            while len(keep) < target and stall < 10 * target:
                nbrs = adj[current]
                if nbrs:
                    current = int(nbrs[int(rng.integers(0, len(nbrs)))])
                    keep.add(current)
                    stall += 1
                else:
                    current = int(rng.integers(0, num_nodes))
                    stall += 1

        # Top up deterministically if the component is smaller than the target.
        if len(keep) < target:
            for node in range(num_nodes):
                if node not in keep:
                    keep.add(node)
                    if len(keep) >= target:
                        break

        return self._induce_subset(data, sorted(keep))

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="RandomWalkSubgraphCrop",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="G7 non-spatial augmentation: seeded RW/BFS connected-subgraph crop.",
            paper_reference="GraphCL subgraph (You et al., 2020); GCC (Qiu et al., 2020)",
            modifies_attributes=["edge_index", "edge_attr", "x", "pos", "num_nodes"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "ratio": {"type": float, "min": 0.0, "max": 1.0},
            "mode": {"type": str, "choices": ["rw", "bfs"]},
            "seed": {"type": int, "optional": True},
        }


# ---------------------------------------------------------------------------
# G8 — PPRSubgraphSampling
# ---------------------------------------------------------------------------
class PPRSubgraphSampling(NonSpatialAugmentBase):
    """Keep the top-ratio nodes by personalized-PageRank around a seeded start (SUBG-CON)."""

    def __init__(self, ratio: float = 0.5, alpha: float = 0.85, seed: int | None = None):
        super().__init__(seed=seed)
        self.ratio = float(ratio)
        self.alpha = float(alpha)

    def transform(self, data: Data) -> Data:
        if not 0.0 < self.ratio <= 1.0:
            raise TransformExecutionError(
                f"PPRSubgraphSampling requires 0 < ratio <= 1, got {self.ratio}.",
                transform_name=self._metadata.name,
            )
        if not 0.0 < self.alpha < 1.0:
            raise TransformExecutionError(
                f"PPRSubgraphSampling requires 0 < alpha < 1, got {self.alpha}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes <= 1:
            return data

        target = max(1, int(round(self.ratio * num_nodes)))
        if target >= num_nodes:
            return data

        import networkx as nx

        rng = self._rng()
        start = int(rng.integers(0, num_nodes))
        graph = to_networkx(data, to_undirected=True)
        graph.add_nodes_from(range(num_nodes))  # ensure isolated nodes are present

        try:
            scores = nx.pagerank(graph, alpha=self.alpha, personalization={start: 1.0})
        except nx.PowerIterationFailedConvergence as exc:
            raise TransformExecutionError(
                "PPRSubgraphSampling: PageRank failed to converge.",
                transform_name=self._metadata.name,
                original_error=exc,
            ) from exc

        # Deterministic ordering: score desc, then node id asc; always retain the start.
        ranked = sorted(range(num_nodes), key=lambda n: (-scores.get(n, 0.0), n))
        keep = set(ranked[:target])
        keep.add(start)
        return self._induce_subset(data, sorted(keep))

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="PPRSubgraphSampling",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="G8 non-spatial augmentation: personalized-PageRank top-k subgraph.",
            paper_reference="SUBG-CON (Jiao et al., ICDM 2020); networkx.pagerank",
            modifies_attributes=["edge_index", "edge_attr", "x", "pos", "num_nodes"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "ratio": {"type": float, "min": 0.0, "max": 1.0},
            "alpha": {"type": float, "min": 0.0, "max": 1.0},
            "seed": {"type": int, "optional": True},
        }


# ---------------------------------------------------------------------------
# G12 — NASANeighborReplace
# ---------------------------------------------------------------------------
class NASANeighborReplace(NonSpatialAugmentBase):
    """Replace a sampled node's 1-hop neighbours with its exactly-``hop``-distance nodes (NASA)."""

    def __init__(self, hop: int = 2, ratio: float = 0.5, seed: int | None = None):
        super().__init__(seed=seed)
        self.hop = int(hop)
        self.ratio = float(ratio)

    def _exact_hop_nodes(self, adj: list[list[int]], source: int) -> list[int]:
        """BFS shortest-path layers; return nodes at distance exactly ``self.hop``."""
        dist = {source: 0}
        frontier = [source]
        while frontier:
            nxt: list[int] = []
            for node in frontier:
                for nbr in adj[node]:
                    if nbr not in dist:
                        dist[nbr] = dist[node] + 1
                        nxt.append(nbr)
            frontier = nxt
        return sorted(n for n, d in dist.items() if d == self.hop)

    def transform(self, data: Data) -> Data:
        if self.hop < 2:
            raise TransformExecutionError(
                f"NASANeighborReplace requires hop >= 2, got {self.hop}.",
                transform_name=self._metadata.name,
            )
        if not 0.0 <= self.ratio <= 1.0:
            raise TransformExecutionError(
                f"NASANeighborReplace requires 0 <= ratio <= 1, got {self.ratio}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes <= 2:
            return data

        adj = self._adjacency(data.edge_index, num_nodes)
        edges = self._undirected_edge_set(data.edge_index)

        rng = self._rng()
        num_pick = int(round(self.ratio * num_nodes))
        if num_pick <= 0:
            return data
        chosen = rng.choice(num_nodes, size=min(num_pick, num_nodes), replace=False)

        for source in sorted(int(c) for c in chosen):
            replacements = self._exact_hop_nodes(adj, source)
            if not replacements:
                continue
            for nbr in list(adj[source]):  # drop current 1-hop incident edges
                edges.discard((source, nbr) if source < nbr else (nbr, source))
            for tgt in replacements:  # add hop-distance edges
                if tgt != source:
                    edges.add((source, tgt) if source < tgt else (tgt, source))

        ordered = sorted(edges)
        if ordered:
            src: list[int] = []
            dst: list[int] = []
            for u, v in ordered:
                src.extend((u, v))
                dst.extend((v, u))
            data.edge_index = torch.tensor([src, dst], dtype=data.edge_index.dtype)
        else:
            data.edge_index = torch.empty((2, 0), dtype=data.edge_index.dtype)

        # Structure changed unpredictably per edge: reset edge_attr to a consistent zero block.
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            data.edge_attr = self._zero_edge_rows(edge_attr, data.edge_index.size(1))
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="NASANeighborReplace",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="G12 non-spatial augmentation: replace 1-hop neighbours with k-hop nodes.",
            paper_reference="NASA neighbour-replacement augmentation",
            modifies_attributes=["edge_index", "edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "hop": {"type": int, "min": 2},
            "ratio": {"type": float, "min": 0.0, "max": 1.0},
            "seed": {"type": int, "optional": True},
        }


# ---------------------------------------------------------------------------
# G15 — SubmodularFeatureSalience
# ---------------------------------------------------------------------------
class SubmodularFeatureSalience(NonSpatialAugmentBase):
    """Retain the top-k node-feature dimensions by greedy facility-location coverage.

    Deterministic (greedy submodular maximisation with index tie-breaks). Non-selected
    feature columns are zeroed, preserving the feature width for downstream stability.
    """

    def __init__(self, ratio: float = 0.5, seed: int | None = None):
        super().__init__(seed=seed)
        self.ratio = float(ratio)

    def transform(self, data: Data) -> Data:
        if not 0.0 < self.ratio <= 1.0:
            raise TransformExecutionError(
                f"SubmodularFeatureSalience requires 0 < ratio <= 1, got {self.ratio}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        x = getattr(data, "x", None)
        if x is None or x.dim() < 2 or x.size(1) <= 1:
            return data

        num_dims = x.size(1)
        k = max(1, int(round(self.ratio * num_dims)))
        if k >= num_dims:
            return data

        # Column-column cosine similarity (facility-location ground set = feature dims).
        feats = x.to(torch.float32).t()  # [F, N]
        norms = feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
        unit = feats / norms
        sim = (unit @ unit.t()).clamp(-1.0, 1.0)  # [F, F]

        selected: list[int] = []
        best_cover = torch.full((num_dims,), float("-inf"))
        remaining = set(range(num_dims))
        # Greedy: each step add the column maximising sum_f max(best_cover, sim[:, c]).
        while len(selected) < k:
            best_gain = None
            best_col = None
            for col in sorted(remaining):
                cover = torch.maximum(best_cover, sim[:, col])
                gain = float(cover.sum())
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_col = col
            selected.append(best_col)
            remaining.discard(best_col)
            best_cover = torch.maximum(best_cover, sim[:, best_col])

        keep = torch.tensor(sorted(selected), dtype=torch.long)
        mask = torch.zeros(num_dims, dtype=torch.bool)
        mask[keep] = True
        masked = x.clone()
        masked[:, ~mask] = torch.zeros_like(masked[:, ~mask])
        data.x = masked
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="SubmodularFeatureSalience",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="G15 non-spatial augmentation: greedy submodular feature-dim selection.",
            paper_reference="Submodular (facility-location) feature selection",
            required_node_features=["x"],
            modifies_attributes=["x"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {
            "ratio": {"type": float, "min": 0.0, "max": 1.0},
            "seed": {"type": int, "optional": True},
        }
