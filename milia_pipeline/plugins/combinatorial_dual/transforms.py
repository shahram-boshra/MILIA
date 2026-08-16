"""
Combinatorial & dual graph operators (catalogue Family D).

Thirteen unary transforms (G -> G') built on NetworkX (+ NumPy/SciPy), MILIA-custom
idiom (``CustomTransformBase`` + ``TransformMetadata``), category ``structural``.
All clone-first, convert structure-only to PyG (features carried deterministically
only where node identity is preserved), and **fail closed** with
``TransformExecutionError`` when a prerequisite (planarity, DAG, bipartiteness,
positions, a partition) is not met.

Single-module layout with ``module_path: transforms`` matches the plugin loader.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from milia_pipeline.transformations.custom_transforms import (
    CustomTransformBase,
    TransformExecutionError,
    TransformMetadata,
)

_AUTHOR = "MILIA Team"


def _pairs_to_data(
    pairs: list[tuple[int, int]],
    num_nodes: int,
    x: torch.Tensor | None = None,
    directed: bool = False,
    pos: torch.Tensor | None = None,
) -> Data:
    """Build a Data from integer node pairs. Undirected pairs are symmetrized."""
    if pairs:
        src: list[int] = []
        dst: list[int] = []
        for i, j in pairs:
            if directed:
                src.append(i)
                dst.append(j)
            else:
                src.extend((i, j))
                dst.extend((j, i))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    out = Data(edge_index=edge_index, num_nodes=num_nodes)
    if x is not None:
        out.x = x
    if pos is not None:
        out.pos = pos
    return out


class CombinatorialDualBase(CustomTransformBase):
    """Base for Family D operators: NetworkX conversion + deterministic feature carry."""

    def _operand(self, data: Data, directed: bool = False) -> nx.Graph:
        """Convert the input Data to a NetworkX graph (integer nodes 0..N-1)."""
        return to_networkx(data, to_undirected=not directed)

    @staticmethod
    def _carry_identity(nodes: list[Any], source: Data) -> torch.Tensor | None:
        """Carry x for a node set that is a subset of the originals (integer labels)."""
        base_x = getattr(source, "x", None)
        if base_x is None:
            return None
        rows: list[int] = []
        for node in nodes:
            if not isinstance(node, int) or not (0 <= node < base_x.size(0)):
                return None
            rows.append(node)
        return base_x[torch.tensor(rows, dtype=torch.long)]

    def _convert(
        self, graph: nx.Graph, source: Data, directed: bool = False, carry: bool = True
    ) -> Data:
        """Structure-only conversion; carry x by identity when node labels are original ints."""
        nodes = list(graph.nodes())
        mapping = {node: idx for idx, node in enumerate(nodes)}
        pairs = [(mapping[u], mapping[v]) for u, v in graph.edges()]
        x = self._carry_identity(nodes, source) if carry else None
        return _pairs_to_data(pairs, len(nodes), x=x, directed=directed)

    def _resolve_injected(self, data: Data, value: Any, key: str, label: str) -> Any:
        """Dependency injection: explicit value, else data attribute, else fail closed."""
        if value is not None:
            return value
        candidate = getattr(data, key, None)
        if candidate is not None:
            return candidate
        raise TransformExecutionError(
            f"{type(self).__name__} requires {label}: pass it to the constructor or attach "
            f"data.{key}.",
            transform_name=self._metadata.name,
        )


# ---------------------------------------------------------------------------
# D1 — ToLevi (incidence / Levi graph)
# ---------------------------------------------------------------------------
class ToLevi(CombinatorialDualBase):
    """Levi (incidence) graph: a node per original vertex and per edge; incidences become edges."""

    def transform(self, data: Data) -> Data:
        data = data.clone()
        g = self._operand(data)
        vertices = list(g.nodes())
        edges = list(g.edges())
        v_index = {v: i for i, v in enumerate(vertices)}
        n_v = len(vertices)
        num_nodes = n_v + len(edges)

        pairs: list[tuple[int, int]] = []
        for e_pos, (u, v) in enumerate(edges):
            e_node = n_v + e_pos
            pairs.append((v_index[u], e_node))
            pairs.append((v_index[v], e_node))

        x = None
        base_x = getattr(data, "x", None)
        if base_x is not None:
            feat = base_x.size(1) if base_x.dim() > 1 else 1
            edge_feats = torch.zeros((len(edges), feat), dtype=base_x.dtype)
            x = torch.cat([base_x, edge_feats], dim=0)

        return _pairs_to_data(pairs, num_nodes, x=x)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="ToLevi",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Levi/incidence graph: a node per vertex and per edge (dual operator).",
            paper_reference="Levi graph / incidence graph",
            modifies_attributes=["edge_index", "x", "num_nodes"],
        )


# ---------------------------------------------------------------------------
# D2 — GraphDual (planar face dual)
# ---------------------------------------------------------------------------
class GraphDual(CombinatorialDualBase):
    """Planar face dual: faces -> nodes; two faces sharing a primal edge -> a dual edge."""

    def transform(self, data: Data) -> Data:
        data = data.clone()
        g = self._operand(data)
        is_planar, embedding = nx.check_planarity(g)
        if not is_planar:
            raise TransformExecutionError(
                "GraphDual requires a planar graph; input is non-planar.",
                transform_name=self._metadata.name,
            )

        # Enumerate faces via half-edge traversal.
        seen: set[tuple[Any, Any]] = set()
        faces: list[frozenset[frozenset]] = []
        for u, v in list(embedding.edges()):
            if (u, v) in seen:
                continue
            boundary = embedding.traverse_face(u, v, mark_half_edges=seen)
            edge_set = frozenset(
                frozenset((boundary[i], boundary[(i + 1) % len(boundary)]))
                for i in range(len(boundary))
            )
            faces.append(edge_set)

        # Dual edge for each primal edge shared by two faces.
        primal_to_faces: dict[frozenset, list[int]] = {}
        for f_idx, edge_set in enumerate(faces):
            for pe in edge_set:
                primal_to_faces.setdefault(pe, []).append(f_idx)

        pairs: set[tuple[int, int]] = set()
        for face_list in primal_to_faces.values():
            for a in range(len(face_list)):
                for b in range(a + 1, len(face_list)):
                    i, j = face_list[a], face_list[b]
                    if i != j:
                        pairs.add((min(i, j), max(i, j)))

        return _pairs_to_data(sorted(pairs), len(faces))

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="GraphDual",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Planar face dual: faces become nodes; shared edges become dual edges.",
            paper_reference="Planar graph duality",
            modifies_attributes=["edge_index", "x", "num_nodes"],
        )


# ---------------------------------------------------------------------------
# D3 — QuotientGraph
# ---------------------------------------------------------------------------
class QuotientGraph(CombinatorialDualBase):
    """Collapse each partition block to a super-node (networkx.quotient_graph)."""

    def __init__(self, partition: list[list[int]] | None = None, partition_key: str = "partition"):
        super().__init__()
        self.partition = partition
        self.partition_key = partition_key

    def transform(self, data: Data) -> Data:
        data = data.clone()
        g = self._operand(data)
        partition = self._resolve_injected(data, self.partition, self.partition_key, "a partition")
        try:
            blocks = [set(block) for block in partition]
            quotient = nx.quotient_graph(g, blocks, relabel=True)
        except Exception as exc:
            raise TransformExecutionError(
                f"QuotientGraph failed for the given partition: {exc}",
                transform_name=self._metadata.name,
            ) from exc
        return self._convert(quotient, data, carry=False)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="QuotientGraph",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Quotient graph: collapse each partition block to a super-node.",
            paper_reference="networkx.quotient_graph",
            modifies_attributes=["edge_index", "x", "num_nodes"],
        )


# ---------------------------------------------------------------------------
# D4 — CondensationTransform (SCC condensation -> DAG)
# ---------------------------------------------------------------------------
class CondensationTransform(CombinatorialDualBase):
    """Strongly-connected-component condensation of the directed view -> a DAG of components."""

    def transform(self, data: Data) -> Data:
        data = data.clone()
        digraph = self._operand(data, directed=True)
        if not isinstance(digraph, nx.DiGraph):
            digraph = nx.DiGraph(digraph)
        condensed = nx.condensation(digraph)
        # condensation nodes are 0..k-1 (not original ids) -> no identity carry.
        pairs = [(u, v) for u, v in condensed.edges()]
        return _pairs_to_data(pairs, condensed.number_of_nodes(), directed=True)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="CondensationTransform",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="SCC condensation of the directed graph into a DAG of components.",
            paper_reference="networkx.condensation",
            modifies_attributes=["edge_index", "x", "num_nodes"],
        )


# ---------------------------------------------------------------------------
# D5 — TransitiveClosure
# ---------------------------------------------------------------------------
class TransitiveClosure(CombinatorialDualBase):
    """Add edge (u, v) whenever v is reachable from u (networkx.transitive_closure)."""

    def transform(self, data: Data) -> Data:
        data = data.clone()
        digraph = self._operand(data, directed=True)
        closure = nx.transitive_closure(digraph, reflexive=False)
        return self._convert(closure, data, directed=True, carry=True)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="TransitiveClosure",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Transitive closure: edge (u,v) whenever v is reachable from u.",
            paper_reference="networkx.transitive_closure",
            modifies_attributes=["edge_index"],
        )


# ---------------------------------------------------------------------------
# D6 — TransitiveReduction (DAG only)
# ---------------------------------------------------------------------------
class TransitiveReduction(CombinatorialDualBase):
    """Minimal edge set preserving reachability; requires a DAG (fail closed otherwise)."""

    def transform(self, data: Data) -> Data:
        data = data.clone()
        digraph = self._operand(data, directed=True)
        if not nx.is_directed_acyclic_graph(digraph):
            raise TransformExecutionError(
                "TransitiveReduction requires a directed acyclic graph (input has cycles).",
                transform_name=self._metadata.name,
            )
        reduced = nx.transitive_reduction(digraph)
        return self._convert(reduced, data, directed=True, carry=True)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="TransitiveReduction",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Transitive reduction: minimal reachability-preserving edge set (DAGs).",
            paper_reference="networkx.transitive_reduction",
            modifies_attributes=["edge_index"],
        )


# ---------------------------------------------------------------------------
# D7 — BipartiteProjection
# ---------------------------------------------------------------------------
class BipartiteProjection(CombinatorialDualBase):
    """Project a bipartite graph onto one node set (co-membership graph)."""

    def __init__(
        self, bipartite_nodes: list[int] | None = None, nodes_key: str = "bipartite_nodes"
    ):
        super().__init__()
        self.bipartite_nodes = bipartite_nodes
        self.nodes_key = nodes_key

    def transform(self, data: Data) -> Data:
        data = data.clone()
        g = self._operand(data)
        if not nx.is_bipartite(g):
            raise TransformExecutionError(
                "BipartiteProjection requires a bipartite graph.",
                transform_name=self._metadata.name,
            )
        nodes = self._resolve_injected(data, self.bipartite_nodes, self.nodes_key, "a node set")
        try:
            projected = nx.bipartite.projected_graph(g, list(nodes))
        except Exception as exc:
            raise TransformExecutionError(
                f"BipartiteProjection failed: {exc}",
                transform_name=self._metadata.name,
            ) from exc
        return self._convert(projected, data, carry=True)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="BipartiteProjection",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Bipartite projection onto one node set (co-membership graph).",
            paper_reference="networkx.algorithms.bipartite.projection",
            modifies_attributes=["edge_index", "x", "num_nodes"],
        )


# ---------------------------------------------------------------------------
# D8 — KHopGraph (K >= 1)
# ---------------------------------------------------------------------------
class KHopGraph(CombinatorialDualBase):
    """Add an edge between any pair of nodes within k hops (generalises PyG TwoHop)."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = int(k)

    def transform(self, data: Data) -> Data:
        if self.k < 1:
            raise TransformExecutionError(
                f"KHopGraph requires k >= 1, got {self.k}.", transform_name=self._metadata.name
            )
        data = data.clone()
        g = self._operand(data)
        n = g.number_of_nodes()
        pairs: set[tuple[int, int]] = set()
        for source_node in g.nodes():
            lengths = nx.single_source_shortest_path_length(g, source_node, cutoff=self.k)
            for target, dist in lengths.items():
                if 1 <= dist <= self.k and source_node < target:
                    pairs.add((source_node, target))
        x = self._carry_identity(list(range(n)), data)
        return _pairs_to_data(sorted(pairs), n, x=x)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="KHopGraph",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="K-hop densification: connect all node pairs within k hops.",
            paper_reference="Generalises torch_geometric.transforms.TwoHop",
            modifies_attributes=["edge_index"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"k": {"type": int}}


# ---------------------------------------------------------------------------
# D9 — ComplementGraph
# ---------------------------------------------------------------------------
class ComplementGraph(CombinatorialDualBase):
    """Complement: edge iff the pair is non-adjacent in the original (networkx.complement)."""

    def transform(self, data: Data) -> Data:
        data = data.clone()
        g = self._operand(data)
        complement = nx.complement(g)
        return self._convert(complement, data, carry=True)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="ComplementGraph",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Complement graph: edge iff non-adjacent in the original.",
            paper_reference="networkx.complement",
            modifies_attributes=["edge_index"],
        )


# ---------------------------------------------------------------------------
# D10 — KNeighborGraphPower (A^k)
# ---------------------------------------------------------------------------
class KNeighborGraphPower(CombinatorialDualBase):
    """k-th adjacency power: edge iff (A^k)_{ij} > 0 for i != j (path-count densification)."""

    def __init__(self, k: int = 2):
        super().__init__()
        self.k = int(k)

    def transform(self, data: Data) -> Data:
        if self.k < 1:
            raise TransformExecutionError(
                f"KNeighborGraphPower requires k >= 1, got {self.k}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        g = self._operand(data)
        n = g.number_of_nodes()
        adjacency = nx.to_numpy_array(g, nodelist=list(range(n)))
        power = np.linalg.matrix_power(adjacency, self.k)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if power[i, j] > 0]
        x = self._carry_identity(list(range(n)), data)
        return _pairs_to_data(pairs, n, x=x)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="KNeighborGraphPower",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="k-th adjacency power A^k as a new edge set (path-count densification).",
            paper_reference="Adjacency matrix power",
            modifies_attributes=["edge_index"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"k": {"type": int}}


# ---------------------------------------------------------------------------
# D11 — MinimumSpanningTree
# ---------------------------------------------------------------------------
class MinimumSpanningTree(CombinatorialDualBase):
    """Replace the graph by its minimum spanning tree (weighted skeleton)."""

    def transform(self, data: Data) -> Data:
        data = data.clone()
        g = self._operand(data)
        # Use edge weights if present; otherwise unit weights (any MST, deterministic per nx).
        mst = nx.minimum_spanning_tree(g, weight="weight")
        return self._convert(mst, data, carry=True)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="MinimumSpanningTree",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Minimum spanning tree skeleton of the graph.",
            paper_reference="networkx.minimum_spanning_tree",
            modifies_attributes=["edge_index"],
        )


# ---------------------------------------------------------------------------
# D12a — GabrielGraph  /  D12b — RelativeNeighborhoodGraph (proximity graphs)
# ---------------------------------------------------------------------------
class _ProximityGraphBase(CombinatorialDualBase):
    """Shared logic for position-based proximity graphs (requires data.pos)."""

    max_nodes = 2000

    def _positions(self, data: Data) -> np.ndarray:
        pos = getattr(data, "pos", None)
        if pos is None:
            raise TransformExecutionError(
                f"{type(self).__name__} requires node positions (data.pos).",
                transform_name=self._metadata.name,
            )
        arr = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos)
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise TransformExecutionError(
                "data.pos must be a 2D [num_nodes, dim] array.",
                transform_name=self._metadata.name,
            )
        if arr.shape[0] > self.max_nodes:
            raise TransformExecutionError(
                f"{type(self).__name__} is O(n^3); {arr.shape[0]} nodes exceeds max_nodes="
                f"{self.max_nodes}.",
                transform_name=self._metadata.name,
            )
        return arr

    def _emit(self, data: Data, pairs: list[tuple[int, int]], n: int) -> Data:
        x = self._carry_identity(list(range(n)), data)
        pos = getattr(data, "pos", None)
        return _pairs_to_data(pairs, n, x=x, pos=pos)


class GabrielGraph(_ProximityGraphBase):
    """Gabriel graph: edge (i,j) iff no other point lies in the closed disk with diameter ij."""

    def transform(self, data: Data) -> Data:
        data = data.clone()
        p = self._positions(data)
        n = p.shape[0]
        d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2)
        pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                # No k inside the disk with diameter ij: d(i,k)^2 + d(j,k)^2 >= d(i,j)^2.
                ok = True
                for k in range(n):
                    if k in (i, j):
                        continue
                    if d2[i, k] + d2[j, k] < d2[i, j] - 1e-12:
                        ok = False
                        break
                if ok:
                    pairs.append((i, j))
        return self._emit(data, pairs, n)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="GabrielGraph",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Gabriel proximity graph from node positions (subgraph of Delaunay).",
            paper_reference="Gabriel & Sokal (1969)",
            required_graph_attributes=["pos"],
            modifies_attributes=["edge_index"],
        )


class RelativeNeighborhoodGraph(_ProximityGraphBase):
    """RNG: edge (i,j) iff no point k is closer to both i and j than they are to each other."""

    def transform(self, data: Data) -> Data:
        data = data.clone()
        p = self._positions(data)
        n = p.shape[0]
        d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2)
        pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                # Edge iff no k with d(i,k) < d(i,j) AND d(j,k) < d(i,j).
                ok = True
                for k in range(n):
                    if k in (i, j):
                        continue
                    if d2[i, k] < d2[i, j] - 1e-12 and d2[j, k] < d2[i, j] - 1e-12:
                        ok = False
                        break
                if ok:
                    pairs.append((i, j))
        return self._emit(data, pairs, n)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="RelativeNeighborhoodGraph",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Relative neighborhood proximity graph from node positions.",
            paper_reference="Toussaint (1980)",
            required_graph_attributes=["pos"],
            modifies_attributes=["edge_index"],
        )
