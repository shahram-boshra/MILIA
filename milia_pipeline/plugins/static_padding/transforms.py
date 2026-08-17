"""
Array layout & alignment transforms for static compilation (catalogue Family H).

Five unary tensor-layout operators (padding / masking / operator-precompute) that make
graph tensors statically shaped or alignment-friendly for XLA / ``torch.compile``.
Pure PyG ``Data`` tensor manipulation (torch/numpy only, no new dependency), MILIA-custom
idiom, category ``structural``, clone-first, fail-closed on unmet prerequisites.

Padding follows jraph ``pad_with_graphs`` conventions: padding nodes/edges are appended and
a boolean mask marks real (True) vs padding (False) entries; ``LayerPreprocess`` follows
Spektral ``GCNConv.preprocess`` (Kipf–Welling renormalization ``D̃^{-1/2}(A+I)D̃^{-1/2}``).

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


def _next_power_of_two(n: int) -> int:
    """Smallest power of two >= n (>=1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _num_edges(data: Data) -> int:
    return data.edge_index.size(1) if getattr(data, "edge_index", None) is not None else 0


def _edge_index(data: Data) -> torch.Tensor:
    ei = getattr(data, "edge_index", None)
    return ei if ei is not None else torch.empty((2, 0), dtype=torch.long)


class StaticBatchPadding(CustomTransformBase):
    """Pad a graph to fixed total ``(n_node, n_edge)`` budgets with masks (jraph-style).

    Padding nodes are isolated with zero features; padding edges are self-loops on the first
    padding node (computation-preserving). Adds boolean ``node_mask`` / ``edge_mask`` where
    True marks real entries and False marks padding.
    """

    def __init__(self, n_node: int, n_edge: int | None = None):
        super().__init__()
        self.n_node = int(n_node)
        self.n_edge = None if n_edge is None else int(n_edge)

    def transform(self, data: Data) -> Data:
        data = data.clone()
        n = data.num_nodes
        e = _num_edges(data)
        target_e = e if self.n_edge is None else self.n_edge

        if self.n_node < n:
            raise TransformExecutionError(
                f"StaticBatchPadding: n_node={self.n_node} < current {n}.",
                transform_name=self._metadata.name,
            )
        if target_e < e:
            raise TransformExecutionError(
                f"StaticBatchPadding: n_edge={target_e} < current {e}.",
                transform_name=self._metadata.name,
            )

        pad_nodes = self.n_node - n
        pad_edges = target_e - e
        if pad_edges > 0 and pad_nodes < 1:
            raise TransformExecutionError(
                "StaticBatchPadding: edge padding needs >=1 padding node to host self-loops "
                "(increase n_node).",
                transform_name=self._metadata.name,
            )

        if getattr(data, "x", None) is not None and pad_nodes > 0:
            feat = data.x.size(1) if data.x.dim() > 1 else 1
            data.x = torch.cat([data.x, torch.zeros((pad_nodes, feat), dtype=data.x.dtype)], dim=0)

        ei = _edge_index(data)
        if pad_edges > 0:
            pad_node = n  # first padding node index
            loops = torch.full((2, pad_edges), pad_node, dtype=torch.long)
            ei = torch.cat([ei, loops], dim=1)
        data.edge_index = ei
        data.num_nodes = self.n_node

        node_mask = torch.zeros(self.n_node, dtype=torch.bool)
        node_mask[:n] = True
        data.node_mask = node_mask
        edge_mask = torch.zeros(target_e, dtype=torch.bool)
        edge_mask[:e] = True
        data.edge_mask = edge_mask
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="StaticBatchPadding",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Pad graph to fixed (n_node, n_edge) budgets with real/pad masks (jraph).",
            paper_reference="jraph.pad_with_graphs",
            modifies_attributes=["x", "edge_index", "num_nodes", "node_mask", "edge_mask"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"n_node": {"type": int}}


class VirtualNodeMasking(CustomTransformBase):
    """Add masked virtual node(s) connected to all real nodes (static-shape friendly).

    Unlike PyG ``VirtualNode``, this also emits a boolean ``node_mask`` (True for real nodes)
    so the virtual nodes can be excluded from loss / message passing.
    """

    def __init__(self, num_virtual: int = 1):
        super().__init__()
        self.num_virtual = int(num_virtual)

    def transform(self, data: Data) -> Data:
        if self.num_virtual < 1:
            raise TransformExecutionError(
                f"VirtualNodeMasking: num_virtual must be >= 1, got {self.num_virtual}.",
                transform_name=self._metadata.name,
            )
        data = data.clone()
        n = data.num_nodes
        k = self.num_virtual

        src: list[int] = []
        dst: list[int] = []
        for virtual in range(n, n + k):
            for real in range(n):
                src.extend((virtual, real))
                dst.extend((real, virtual))
        new_edges = (
            torch.tensor([src, dst], dtype=torch.long)
            if src
            else torch.empty((2, 0), dtype=torch.long)
        )
        ei = _edge_index(data)
        data.edge_index = torch.cat([ei, new_edges], dim=1) if ei.numel() else new_edges

        if getattr(data, "x", None) is not None:
            feat = data.x.size(1) if data.x.dim() > 1 else 1
            data.x = torch.cat([data.x, torch.zeros((k, feat), dtype=data.x.dtype)], dim=0)
        data.num_nodes = n + k

        node_mask = torch.zeros(n + k, dtype=torch.bool)
        node_mask[:n] = True
        data.node_mask = node_mask
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="VirtualNodeMasking",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Add masked virtual node(s) connected to all real nodes (compilation).",
            paper_reference="Compilation-oriented virtual node (distinct from PyG VirtualNode)",
            modifies_attributes=["x", "edge_index", "num_nodes", "node_mask"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"num_virtual": {"type": int}}


class PowerOfTwoPadding(CustomTransformBase):
    """Pad the node count (optionally the edge count) up to the next power of two.

    Adds isolated padding nodes with zero features and a boolean ``node_mask`` (True = real).
    With ``pad_edges=True`` also pads edges to the next power of two via self-loops on the
    first padding node (requires at least one padding node).
    """

    def __init__(self, pad_edges: bool = False):
        super().__init__()
        self.pad_edges = bool(pad_edges)

    def transform(self, data: Data) -> Data:
        data = data.clone()
        n = data.num_nodes
        target = _next_power_of_two(n)
        pad_nodes = target - n

        if getattr(data, "x", None) is not None and pad_nodes > 0:
            feat = data.x.size(1) if data.x.dim() > 1 else 1
            data.x = torch.cat([data.x, torch.zeros((pad_nodes, feat), dtype=data.x.dtype)], dim=0)

        ei = _edge_index(data)
        if self.pad_edges:
            e = ei.size(1)
            target_e = _next_power_of_two(e)
            pad_edges = target_e - e
            if pad_edges > 0:
                if pad_nodes < 1:
                    raise TransformExecutionError(
                        "PowerOfTwoPadding: edge padding needs >=1 padding node.",
                        transform_name=self._metadata.name,
                    )
                loops = torch.full((2, pad_edges), n, dtype=torch.long)
                ei = torch.cat([ei, loops], dim=1)
                edge_mask = torch.zeros(target_e, dtype=torch.bool)
                edge_mask[:e] = True
                data.edge_mask = edge_mask
        data.edge_index = ei
        data.num_nodes = target

        node_mask = torch.zeros(target, dtype=torch.bool)
        node_mask[:n] = True
        data.node_mask = node_mask
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="PowerOfTwoPadding",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Pad node/edge counts up to the next power of two (kernel alignment).",
            paper_reference="Hardware kernel-alignment practice",
            modifies_attributes=["x", "edge_index", "num_nodes", "node_mask"],
        )


class LayerPreprocess(CustomTransformBase):
    """Precompute and cache the GCN propagation operator ``D̃^{-1/2}(A+I)D̃^{-1/2}``.

    Adds self-loops and stores the symmetric-normalized values as ``edge_weight`` aligned to
    the augmented ``edge_index`` (Spektral ``GCNConv.preprocess`` / Kipf–Welling).
    """

    def transform(self, data: Data) -> Data:
        data = data.clone()
        if getattr(data, "edge_index", None) is None:
            raise TransformExecutionError(
                "LayerPreprocess requires edge_index.", transform_name=self._metadata.name
            )
        n = data.num_nodes
        ei = data.edge_index
        # Drop any existing self-loops, then add a fresh self-loop per node (A -> A + I).
        if ei.numel() > 0:
            keep = ei[0] != ei[1]
            ei = ei[:, keep]
        loops = torch.arange(n, dtype=torch.long)
        augmented = torch.cat([ei, torch.stack([loops, loops])], dim=1)

        deg = torch.bincount(augmented[0], minlength=n).to(torch.float)
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        row, col = augmented[0], augmented[1]
        weight = d_inv_sqrt[row] * d_inv_sqrt[col]

        data.edge_index = augmented
        data.edge_weight = weight
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="LayerPreprocess",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Precompute cached GCN operator D^-1/2 (A+I) D^-1/2 as edge_weight.",
            paper_reference="Spektral LayerPreprocess / Kipf & Welling (2017)",
            required_graph_attributes=["edge_index"],
            modifies_attributes=["edge_index", "edge_weight"],
        )


class EdgeAttrCanonicalPadding(CustomTransformBase):
    """Zero-pad ``edge_attr`` to a canonical feature width across a dataset."""

    def __init__(self, width: int):
        super().__init__()
        self.width = int(width)

    def transform(self, data: Data) -> Data:
        data = data.clone()
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            raise TransformExecutionError(
                "EdgeAttrCanonicalPadding requires edge_attr.",
                transform_name=self._metadata.name,
            )
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        current = edge_attr.size(1)
        if self.width < current:
            raise TransformExecutionError(
                f"EdgeAttrCanonicalPadding: width={self.width} < current width {current}.",
                transform_name=self._metadata.name,
            )
        if self.width > current:
            pad = torch.zeros((edge_attr.size(0), self.width - current), dtype=edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, pad], dim=1)
        data.edge_attr = edge_attr
        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="EdgeAttrCanonicalPadding",
            version="1.0.0",
            author=_AUTHOR,
            category="structural",
            description="Zero-pad edge_attr to a canonical width across a dataset.",
            paper_reference="MILIA edge-attr-aware system alignment",
            required_edge_features=["edge_attr"],
            modifies_attributes=["edge_attr"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict:
        return {"width": {"type": int}}
