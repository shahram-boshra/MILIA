"""The custom model-selection mode and ``builder_type`` are deprecated (removal in 2.0.0)."""

from __future__ import annotations

import logging
import warnings

import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.cli_manager import CLIManager
from milia_pipeline.models._deprecations import (
    BUILDER_TYPE_DEPRECATION,
    CUSTOM_ARCHITECTURE_DEPRECATION,
    REMOVAL_VERSION,
)
from milia_pipeline.models.factory.model_factory import ModelFactory

LAYERS = [
    {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
    {"type": "ReLU", "params": {}},
    {"type": "Linear", "params": {"in_features": 8, "out_features": 1}},
]


def _sample() -> Data:
    return Data(x=torch.randn(5, 4), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.randn(5, 1))


def test_messages_name_the_removal_and_the_migration():
    assert REMOVAL_VERSION == "2.0.0"
    assert "removed in MILIA 2.0.0" in CUSTOM_ARCHITECTURE_DEPRECATION
    assert "model plugin" in CUSTOM_ARCHITECTURE_DEPRECATION
    assert "'single' and 'ensemble'" in CUSTOM_ARCHITECTURE_DEPRECATION
    assert "has no effect" in BUILDER_TYPE_DEPRECATION


def test_custom_model_still_builds_with_a_future_warning(caplog):
    """Deprecated, not removed: the model is built, users see a FutureWarning and the log."""
    hyperparameters = {"architecture_config": {"name": "Deprecated", "layers": LAYERS}}
    with caplog.at_level(logging.WARNING), pytest.warns(FutureWarning) as record:
        model = ModelFactory().create_model("custom", hyperparameters, "node_regression", _sample())

    assert isinstance(model, torch.nn.Module)
    messages = [str(w.message) for w in record if issubclass(w.category, FutureWarning)]
    assert messages == [CUSTOM_ARCHITECTURE_DEPRECATION]
    assert record[0].filename == __file__  # attributed to the caller
    assert CUSTOM_ARCHITECTURE_DEPRECATION in caplog.text


def test_builder_type_in_the_configuration_is_warned_about():
    config = {"name": "Deprecated", "builder_type": "parallel", "layers": LAYERS}
    with pytest.warns(FutureWarning) as record:
        ModelFactory().create_model(
            "custom", {"architecture_config": config}, "node_regression", _sample()
        )
    messages = {str(w.message) for w in record if issubclass(w.category, FutureWarning)}
    assert messages == {CUSTOM_ARCHITECTURE_DEPRECATION, BUILDER_TYPE_DEPRECATION}


def test_single_and_ensemble_modes_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ModelFactory().create_model("ExampleGNN", {}, "node_regression", _sample())


@pytest.mark.parametrize(("argv", "warned"), [(["--builder-type", "parallel"], True), ([], False)])
def test_cli_builder_type_flag_is_warned_about_only_when_given(argv, warned):
    cli = CLIManager()
    args = cli.parser.parse_args(argv)
    args.train = True
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cli.handle_training_operations(args)
    messages = [str(w.message) for w in record if issubclass(w.category, FutureWarning)]
    assert (BUILDER_TYPE_DEPRECATION in messages) is warned
