"""
Algebraic graph-product transforms (catalogue Family C).

NetworkX-native two-graph operators plus a stochastic-Kronecker generator. These
depend only on NetworkX (already a MILIA dependency) and PyG, and follow the
MILIA-custom idiom (``CustomTransformBase`` + ``TransformMetadata``).

Design (evidence-verified against NetworkX and PyG docs):

* The second operand ``H`` is supplied by dependency injection — an ``nx.Graph``
  or a declarative generator spec ``{"generator": <nx-generator>, ...}`` — with a
  per-sample ``data`` attribute fallback; missing ``H`` fails closed. No global
  state.
* NetworkX products relabel nodes to 2-tuples and collapse node attributes into
  2-tuples / ``None``, which PyG's ``from_networkx`` rejects. We therefore convert
  *topology only* and re-attach node features deterministically from the base
  operand ``G`` by the first tuple coordinate (preserving feature dimensionality);
  edge attributes are undefined for algebraic products and are dropped.
* Category is ``structural`` (the registry auto-derives ``research_applicability``
  from it); the fine "product" taxonomy is carried in each transform's name and
  description. Every transform self-declares ``modifies_attributes`` so the
  registry's metadata-driven dependency inference is populated.

Single-module layout with ``module_path: transforms`` matches the plugin loader
(``_load_transform_class`` resolves ``plugin_dir / f"{module_path}.py"`` and execs
each declared class as a standalone module — no intra-package imports).
"""

from __future__ import annotations

from typing import Any

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


class GraphProductBase(CustomTransformBase):
    """Abstract base for two-graph algebraic product transforms.

    Subclasses implement :meth:`_compute_product` (the NetworkX operation on two
    undirected operand graphs) and :meth:`get_metadata`.
    """

    def __init__(
        self,
        factor_graph: nx.Graph | dict[str, Any] | None = None,
        factor_graph_key: str = "product_factor_H",
        seed: int | None = None,
    ):
        super().__init__()  # sets self._metadata via get_metadata()
        self.factor_graph = factor_graph
        self.factor_graph_key = factor_graph_key
        self.seed = seed

    # -- factor resolution (dependency injection, fail-closed) ----------------

    def _resolve_factor_graph(self, data: Data) -> nx.Graph:
        if self.factor_graph is not None:
            return self._coerce_to_graph(self.factor_graph, source="factor_graph")

        candidate = getattr(data, self.factor_graph_key, None)
        if candidate is not None:
            return self._coerce_to_graph(candidate, source=f"data.{self.factor_graph_key}")

        raise TransformExecutionError(
            f"{type(self).__name__} requires a second operand graph H: pass "
            f"factor_graph=<nx.Graph|spec> or attach data.{self.factor_graph_key}.",
            transform_name=self._metadata.name,
        )

    def _coerce_to_graph(self, value: Any, source: str) -> nx.Graph:
        if isinstance(value, nx.Graph):
            graph = value
        elif isinstance(value, Data):
            graph = to_networkx(value, to_undirected=True)
        elif isinstance(value, dict) and "generator" in value:
            graph = self._build_factor_from_spec(value)
        else:
            raise TransformExecutionError(
                f"Unsupported factor graph from {source}: expected nx.Graph, Data, or a "
                f"{{'generator': ...}} spec, got {type(value).__name__}.",
                transform_name=self._metadata.name,
            )
        return graph.to_undirected() if graph.is_directed() else graph

    def _build_factor_from_spec(self, spec: dict[str, Any]) -> nx.Graph:
        spec = dict(spec)
        gen_name = spec.pop("generator")
        generator = getattr(nx, gen_name, None)
        if not callable(generator):
            raise TransformExecutionError(
                f"Unknown NetworkX graph generator '{gen_name}' in factor spec.",
                transform_name=self._metadata.name,
            )
        try:
            graph = generator(**spec)
        except Exception as exc:
            raise TransformExecutionError(
                f"Failed to build factor graph via nx.{gen_name}({spec}): {exc}",
                transform_name=self._metadata.name,
            ) from exc
        if not isinstance(graph, nx.Graph):
            raise TransformExecutionError(
                f"nx.{gen_name} did not return a graph.",
                transform_name=self._metadata.name,
            )
        return graph

    # -- product hook + transform --------------------------------------------

    def _compute_product(self, g: nx.Graph, h: nx.Graph) -> nx.Graph:
        raise NotImplementedError

    def transform(self, data: Data) -> Data:
        data = data.clone()
        g = to_networkx(data, to_undirected=True)
        h = self._resolve_factor_graph(data)
        product = self._compute_product(g, h)
        return self._product_to_data(product, data)

    # -- structure-only conversion + deterministic feature carry --------------

    def _product_to_data(self, product: nx.Graph, source: Data) -> Data:
        nodes = list(product.nodes())
        mapping = {node: idx for idx, node in enumerate(nodes)}
        num_nodes = len(nodes)

        if product.number_of_edges() > 0:
            src: list[int] = []
            dst: list[int] = []
            for u, v in product.edges():
                iu, iv = mapping[u], mapping[v]
                src.extend((iu, iv))
                dst.extend((iv, iu))
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        out = Data(edge_index=edge_index, num_nodes=num_nodes)

        if getattr(source, "x", None) is not None:
            base_x = source.x
            ng = base_x.size(0)
            rows: list[int] = []
            ok = True
            for node in nodes:
                g_idx = node[0] if isinstance(node, tuple) else node
                if not isinstance(g_idx, int) or not (0 <= g_idx < ng):
                    ok = False
                    break
                rows.append(g_idx)
            if ok:
                out.x = base_x[torch.tensor(rows, dtype=torch.long)]

        return out


class CartesianProduct(GraphProductBase):
    """Cartesian graph product G □ H (a structural product transform)."""

    def _compute_product(self, g: nx.Graph, h: nx.Graph) -> nx.Graph:
        return nx.cartesian_product(g, h)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="CartesianProduct",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Cartesian graph product G box H (structural product transform).",
            paper_reference="networkx.algorithms.operators.product.cartesian_product",
            modifies_attributes=["edge_index", "x"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"seed": {"type": (int, type(None))}}


class TensorProduct(GraphProductBase):
    """Tensor / categorical graph product G × H (a structural product transform)."""

    def _compute_product(self, g: nx.Graph, h: nx.Graph) -> nx.Graph:
        return nx.tensor_product(g, h)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="TensorProduct",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Tensor (categorical/Kronecker) graph product G x H product transform.",
            paper_reference="networkx.algorithms.operators.product.tensor_product",
            modifies_attributes=["edge_index", "x"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"seed": {"type": (int, type(None))}}


class StrongProduct(GraphProductBase):
    """Strong graph product G ⊠ H (Cartesian ∪ tensor edges); a product transform."""

    def _compute_product(self, g: nx.Graph, h: nx.Graph) -> nx.Graph:
        return nx.strong_product(g, h)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="StrongProduct",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Strong graph product G box-times H (structural product transform).",
            paper_reference="networkx.algorithms.operators.product.strong_product",
            modifies_attributes=["edge_index", "x"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"seed": {"type": (int, type(None))}}


class LexicographicProduct(GraphProductBase):
    """Lexicographic graph product G · H (a structural product transform)."""

    def _compute_product(self, g: nx.Graph, h: nx.Graph) -> nx.Graph:
        return nx.lexicographic_product(g, h)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="LexicographicProduct",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Lexicographic graph product G . H (structural product transform).",
            paper_reference="networkx.algorithms.operators.product.lexicographic_product",
            modifies_attributes=["edge_index", "x"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"seed": {"type": (int, type(None))}}


class RootedProduct(GraphProductBase):
    """Rooted graph product of G and H rooted at ``root`` in H (product transform)."""

    def __init__(
        self,
        factor_graph: nx.Graph | dict[str, Any] | None = None,
        factor_graph_key: str = "product_factor_H",
        seed: int | None = None,
        root: Any = None,
    ):
        super().__init__(factor_graph=factor_graph, factor_graph_key=factor_graph_key, seed=seed)
        self.root = root

    def _compute_product(self, g: nx.Graph, h: nx.Graph) -> nx.Graph:
        root = self.root if self.root is not None else next(iter(h.nodes()), None)
        if root is None or root not in h:
            raise TransformExecutionError(
                f"RootedProduct requires a valid root in H; got {root!r}.",
                transform_name=self._metadata.name,
            )
        return nx.rooted_product(g, h, root)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="RootedProduct",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Rooted graph product of G and H (structural product transform).",
            paper_reference="networkx.algorithms.operators.product.rooted_product",
            modifies_attributes=["edge_index", "x"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"seed": {"type": (int, type(None))}}


class CoronaProduct(GraphProductBase):
    """Corona graph product G ∘ H (a structural product transform).

    G plus |V(G)| copies of H, the i-th copy joined to the i-th vertex of G.
    """

    def _compute_product(self, g: nx.Graph, h: nx.Graph) -> nx.Graph:
        return nx.corona_product(g, h)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="CoronaProduct",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Corona graph product G o H (structural product transform).",
            paper_reference="networkx.algorithms.operators.product.corona_product",
            modifies_attributes=["edge_index", "x"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"seed": {"type": (int, type(None))}}


class KroneckerGraphGeneration(CustomTransformBase):
    """Stochastic Kronecker graph generation (Leskovec et al., KronGen).

    Generates a synthetic undirected graph as the ``k``-th Kronecker power of a
    small stochastic initiator matrix, sampling each edge independently. This is a
    generator: it ignores the input graph's topology and returns a fresh graph.
    Randomness is drawn from a local, seeded generator for reproducibility.

    Args:
        initiator: Square 2D matrix of edge probabilities in [0, 1]. Default is
            the classic 2x2 ``[[0.9, 0.5], [0.5, 0.1]]``.
        k: Kronecker power (number of nodes = ``b ** k`` for a ``b x b`` initiator).
        seed: Seed for the local RNG.
        max_nodes: Safety cap on generated node count (edge sampling is O(n^2)).
    """

    def __init__(
        self,
        initiator: Any = None,
        k: int = 3,
        seed: int | None = None,
        max_nodes: int = 4096,
    ):
        super().__init__()
        self.initiator = initiator if initiator is not None else [[0.9, 0.5], [0.5, 0.1]]
        self.k = int(k)
        self.seed = seed
        self.max_nodes = int(max_nodes)

    def transform(self, data: Data) -> Data:
        import numpy as np

        init = np.asarray(self.initiator, dtype=float)
        if init.ndim != 2 or init.shape[0] != init.shape[1] or init.shape[0] < 1:
            raise TransformExecutionError(
                "Kronecker initiator must be a non-empty square 2D matrix.",
                transform_name=self._metadata.name,
            )
        if not (np.all(init >= 0.0) and np.all(init <= 1.0)):
            raise TransformExecutionError(
                "Kronecker initiator entries must be probabilities in [0, 1].",
                transform_name=self._metadata.name,
            )
        if self.k < 1:
            raise TransformExecutionError(
                f"Kronecker power k must be >= 1, got {self.k}.",
                transform_name=self._metadata.name,
            )

        b = init.shape[0]
        num_nodes = b**self.k
        if num_nodes > self.max_nodes:
            raise TransformExecutionError(
                f"Kronecker generation would produce {num_nodes} nodes (> max_nodes="
                f"{self.max_nodes}); reduce k or raise max_nodes.",
                transform_name=self._metadata.name,
            )

        # k-th Kronecker power gives the full n x n edge-probability matrix.
        prob = init.copy()
        for _ in range(self.k - 1):
            prob = np.kron(prob, init)

        rng = np.random.default_rng(self.seed)
        # Undirected simple graph: sample the strict upper triangle only.
        iu, iv = np.triu_indices(num_nodes, k=1)
        draws = rng.random(iu.shape[0])
        keep = draws < prob[iu, iv]
        rows = iu[keep]
        cols = iv[keep]

        if rows.size > 0:
            src = np.concatenate([rows, cols])
            dst = np.concatenate([cols, rows])
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(edge_index=edge_index, num_nodes=num_nodes)

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="KroneckerGraphGeneration",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Stochastic Kronecker-product synthetic graph generation (KronGen).",
            paper_reference="Leskovec et al., Kronecker Graphs (JMLR 2010)",
            modifies_attributes=["edge_index"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"k": {"type": int}, "seed": {"type": (int, type(None))}}
