"""Edge-level output widths follow the edge targets the trainer uses."""

from __future__ import annotations

import logging

import pytest
import torch
from torch_geometric.data import Batch, Data

from milia_pipeline.models.factory.model_factory import (
    ModelFactory,
    _edge_regression_target_width,
)

NUM_NODES = 5
EDGE_INDEX = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
NUM_EDGES = EDGE_INDEX.size(1)


def _data(**edge_fields) -> Data:
    return Data(
        x=torch.randn(NUM_NODES, 4),
        edge_index=EDGE_INDEX,
        edge_attr=torch.randn(NUM_EDGES, 3),
        **edge_fields,
    )


@pytest.mark.parametrize(
    ("fields", "width"),
    [
        ({"edge_value": torch.randn(NUM_EDGES, 2), "edge_y": torch.randn(NUM_EDGES, 1)}, 2),
        ({"edge_y": torch.randn(NUM_EDGES, 1)}, 1),
        ({"edge_y": torch.randn(NUM_EDGES)}, 1),
        ({}, 3),  # not prepared for the task: edge_attr
    ],
    ids=["edge_value-first", "edge_y", "edge_y-1d", "edge_attr-fallback"],
)
def test_edge_regression_width_follows_the_trainer_target(fields, width):
    assert _edge_regression_target_width(_data(**fields)) == width


def test_edge_regression_width_without_an_edge_target():
    assert _edge_regression_target_width(None) is None
    assert _edge_regression_target_width(Data(x=torch.randn(3, 4))) is None


def _ensemble() -> dict:
    return {
        "ensemble_config": {
            "name": "EdgeEnsemble",
            "strategy": "parallel",
            "fusion": {"method": "mean"},
            "models": [
                {"name": "GCN", "hyperparameters": {"hidden_channels": 16}},
                {"name": "GraphSAGE", "hyperparameters": {"hidden_channels": 16}},
            ],
        }
    }


@pytest.mark.parametrize("mode", ["single", "ensemble"])
@pytest.mark.parametrize("target_width", [1, 2])
def test_edge_regression_outputs_one_column_per_target(mode, target_width):
    """Three edge features, ``target_width`` selected targets: the model outputs the targets."""
    sample = _data(edge_y=torch.randn(NUM_EDGES, target_width))
    factory = ModelFactory()
    if mode == "single":
        model = factory.create_model("GCN", {"hidden_channels": 16}, "edge_regression", sample)
    else:
        model = factory.create_model("ensemble", _ensemble(), "edge_regression", sample)

    batch = Batch.from_data_list([sample])
    out = model(batch.x, batch.edge_index, batch=batch.batch)
    assert out.size(0) == NUM_EDGES
    assert (out.size(-1) if out.dim() > 1 else 1) == target_width


def test_edge_classification_without_num_classes_uses_the_default(caplog):
    """No ``UnboundLocalError``: the documented default of 10 classes, with a warning."""
    sample = _data(edge_y=torch.randint(0, 3, (NUM_EDGES,)))
    with caplog.at_level(logging.WARNING):
        model = ModelFactory().create_model(
            "GCN", {"hidden_channels": 16}, "edge_classification", sample
        )
    out = model(sample.x, sample.edge_index)
    assert out.shape == (NUM_EDGES, 10)
    assert "num_classes not specified, defaulting to 10" in caplog.text


def test_edge_classification_with_num_classes_override():
    sample = _data(edge_y=torch.randint(0, 4, (NUM_EDGES,)))
    model, _ = ModelFactory().create_model_with_info(
        "GCN", {"hidden_channels": 16}, "edge_classification", sample, num_classes_override=4
    )
    assert model(sample.x, sample.edge_index).shape == (NUM_EDGES, 4)
