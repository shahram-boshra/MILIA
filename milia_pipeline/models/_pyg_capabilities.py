"""Capabilities of PyG models whose outputs MILIA cannot read from a signature.

Each entry follows the class's own documentation and source (PyG 2.8.1); ``check_model_capabilities``
verifies them against the real outputs in ``tests/test_model_capabilities.py``. Classes not listed
here get ``default_capabilities`` (one feature row per node), which is correct for PyG's node-level
GNNs (``BasicGNN`` and its subclasses, ``PMLP``, ``Polynormer``, ...).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from milia_pipeline.models.capabilities import ModelCapabilities, OutputSpec, register_capabilities

_PYG = "torch_geometric.nn.models"
_GRAPH_PROPERTY = OutputSpec("property", "graph", "property")
_FORCES = OutputSpec("forces", "node", "vector", width=3)
_NODE_EMBEDDING = OutputSpec("embedding", "node", "embedding")


@register_capabilities(f"{_PYG}.basic_gnn.GAT", f"{_PYG}.basic_gnn.PNA")
def _edge_dim_gnn(hyperparameters: Mapping[str, Any]) -> ModelCapabilities:
    # BasicGNN passes edge_attr to every conv when the class sets supports_edge_attr (GAT, PNA).
    # The convs use it only when built with edge_dim: GATConv creates lin_edge and PNAConv
    # edge_encoder only then; without it GATConv ignores edge_attr and PNAConv fails on it.
    return ModelCapabilities(
        outputs=(OutputSpec("features", "node", "features"),),
        accepts_edge_attr=hyperparameters.get("edge_dim") is not None,
    )


@register_capabilities(f"{_PYG}.attentive_fp.AttentiveFP")
def _attentive_fp(hyperparameters: Mapping[str, Any]) -> ModelCapabilities:
    # forward(x, edge_index, edge_attr, batch): GRU readout over the molecule -> [num_graphs, out]
    return ModelCapabilities(outputs=(_GRAPH_PROPERTY,), accepts_edge_attr=True)


@register_capabilities(f"{_PYG}.neural_fingerprint.NeuralFingerprint")
def _neural_fingerprint(hyperparameters: Mapping[str, Any]) -> ModelCapabilities:
    # forward(x, edge_index, batch, batch_size): sum of per-layer softmax readouts -> [num_graphs, out]
    return ModelCapabilities(outputs=(_GRAPH_PROPERTY,))


@register_capabilities(f"{_PYG}.schnet.SchNet", f"{_PYG}.dimenet.DimeNet")
def _energy_model(hyperparameters: Mapping[str, Any]) -> ModelCapabilities:
    # forward(z, pos, batch): interaction graph built internally (radius graph), atomwise outputs
    # summed per molecule -> [num_graphs, out]; DimeNetPlusPlus subclasses DimeNet.
    return ModelCapabilities(
        outputs=(_GRAPH_PROPERTY,), requires_positions=True, uses_graph_structure=False
    )


@register_capabilities(f"{_PYG}.visnet.ViSNet")
def _visnet(hyperparameters: Mapping[str, Any]) -> ModelCapabilities:
    # forward(z, pos, batch) -> (y, -dy) with derivative=True, else (y, None): "y: The energies or
    # properties for each molecule", "dy: The negative derivative of energies" (per atom, 3D).
    # derivative=True: forces = -dE/dpos via torch.autograd.grad inside forward.
    return ModelCapabilities(
        outputs=(OutputSpec("energy", "graph", "property"), _FORCES),
        requires_positions=True,
        uses_graph_structure=False,
        differentiates_inputs=bool(hyperparameters.get("derivative", False)),
    )


@register_capabilities(f"{_PYG}.gnnff.GNNFF")
def _gnnff(hyperparameters: Mapping[str, Any]) -> ModelCapabilities:
    # "directly predicts atomic forces"; forward returns scatter(force, i, dim=0) -> [num_atoms, 3]
    return ModelCapabilities(
        outputs=(_FORCES,), requires_positions=True, uses_graph_structure=False
    )


@register_capabilities(
    f"{_PYG}.autoencoder.GAE", f"{_PYG}.node2vec.Node2Vec", f"{_PYG}.metapath2vec.MetaPath2Vec"
)
def _embedding_model(hyperparameters: Mapping[str, Any]) -> ModelCapabilities:
    # Graph autoencoders (GAE, VGAE, ARGA, ARGVA subclass GAE) and random-walk embeddings learn one
    # embedding per node for link prediction; they are not property predictors.
    return ModelCapabilities(outputs=(_NODE_EMBEDDING,))
