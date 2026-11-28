"""The graph-level readout (models/readout.py): configured once, applied, recorded, rebuilt.

``models.selection.pooling_method`` reaches every model that pools per-atom outputs (PyG's
message-passing models, read from the model registry; MILIA's ensembles), in standard training and
HPO alike; the factory records the readout it applied (``model_info["readout"]``) and a model
rebuilt from its checkpoint pools the same way. An unknown readout is refused. A checkpoint from
before the readout was recorded is rebuilt as it was trained (mean).

The behaviour checked is the one a readout exists for: on two molecules made of identical atoms,
one with twice as many (rings of 3 and 6), ``add`` doubles the prediction, ``mean`` and ``max`` do not.
"""

from __future__ import annotations

import argparse
import copy
import logging

import pytest
import torch

# The shipped example plugin, discovered afresh and removed after the test (as in its e2e test)
from test_model_plugin_e2e import example_models, fresh  # noqa: F401
from torch_geometric.data import Batch, Data
from torch_geometric.nn.models.basic_gnn import BasicGNN

from milia_pipeline.exceptions import HPOConfigurationError, ModelValidationError
from milia_pipeline.models import list_models
from milia_pipeline.models.factory.model_factory import ModelFactory, applied_readout
from milia_pipeline.models.readout import (
    DEFAULT_READOUT,
    READOUTS,
    configured_readout,
    global_pool,
    validate_readout,
    with_readout,
)
from milia_pipeline.models.registry.model_registry import ModelRegistry

FEATURES = 4
BASE = {"in_channels": FEATURES, "hidden_channels": 8, "out_channels": 1, "num_layers": 2}


def _ring(n: int) -> Data:
    """A ring of ``n`` identical atoms: every atom has the same neighbourhood, so the same output."""
    ring = torch.arange(n)
    edge_index = torch.cat(
        [torch.stack([ring, (ring + 1) % n]), torch.stack([(ring + 1) % n, ring])], 1
    )
    return Data(x=torch.ones(n, FEATURES), edge_index=edge_index)


# The factory's sample (bonded, as the data is: PNA's degree histogram is computed from it)
SAMPLE = Data(x=_ring(4).x, edge_index=_ring(4).edge_index, y=torch.zeros(1, 1))


def _molecules(*sizes: int) -> Batch:
    """Rings of identical atoms (sizes ≥ 3)."""
    return Batch.from_data_list([_ring(n) for n in sizes])


def _predict(model, batch: Batch) -> list[float]:
    model.eval()
    with torch.no_grad():
        return model(batch.x, batch.edge_index, batch=batch.batch).view(-1).tolist()


def _message_passing_models() -> list[str]:
    """The registry's PyG message-passing models (``BasicGNN``): node outputs MILIA pools."""
    list_models()
    registry = ModelRegistry.get_instance()
    names = []
    for name in sorted(list_models()):
        cls = registry.get_model(name)
        if isinstance(cls, type) and issubclass(cls, BasicGNN):
            names.append(name)
    return names


MODELS = _message_passing_models()


def _build(name: str, readout: str, **extra):
    torch.manual_seed(0)
    return ModelFactory().create_model_with_info(
        name=name,
        hyperparameters={**BASE, **extra, "pooling_method": readout},
        task_type="graph_regression",
        sample_data=SAMPLE,
    )


def _ensemble_config(strategy: str = "parallel") -> dict:
    from milia_pipeline.config.config_loader import load_config

    ensemble = copy.deepcopy(load_config("configs")["models"]["ensemble"])
    for member in ensemble["models"]:
        member.setdefault("hyperparameters", {}).update(in_channels=FEATURES)
    ensemble["strategy"] = strategy
    return ensemble


def _expected(readout: str, small: float, large: float) -> bool:
    """Rings of 3 and 6 identical atoms: ``add`` doubles, ``mean``/``max`` do not."""
    if readout == "add":
        return large == pytest.approx(2 * small, rel=1e-5)
    return large == pytest.approx(small, rel=1e-5)


# --------------------------------------------------------------------------- the module


@pytest.mark.parametrize("readout", READOUTS)
def test_a_readout_pools_as_named(readout):
    x = torch.tensor([[1.0], [3.0], [10.0]])
    batch = torch.tensor([0, 0, 1])
    want = {"mean": [[2.0], [10.0]], "add": [[4.0], [10.0]], "max": [[3.0], [10.0]]}[readout]
    assert global_pool(x, batch, readout, size=2).tolist() == want
    assert global_pool(x[:2], None, readout).tolist() == [want[0]]  # one graph


@pytest.mark.parametrize("name", ["sum", "Mean", "", None, 1])
def test_an_unknown_readout_is_refused_naming_the_options(name):
    with pytest.raises(ValueError, match="not a readout; use one of mean, add, max"):
        validate_readout(name)


def test_the_configured_readout_defaults_to_mean_and_is_validated():
    assert configured_readout({}) == configured_readout(None) == DEFAULT_READOUT
    assert configured_readout({"selection": {"pooling_method": "add"}}) == "add"
    with pytest.raises(ValueError, match="models.selection.pooling_method"):
        configured_readout({"selection": {"pooling_method": "sum"}})


def test_with_readout_places_it_where_the_factory_reads_it_and_keeps_an_explicit_one():
    assert with_readout("GCN", {"hidden_channels": 8}, "add") == {
        "hidden_channels": 8,
        "pooling_method": "add",
    }
    assert with_readout("GCN", {"pooling_method": "max"}, "add")["pooling_method"] == "max"
    ensemble = with_readout("ensemble", {"ensemble_config": {"models": []}}, "add")
    assert ensemble["ensemble_config"]["pooling_method"] == "add"
    kept = with_readout("Ensemble", {"ensemble_config": {"pooling_method": "max"}}, "add")
    assert kept["ensemble_config"]["pooling_method"] == "max"
    with pytest.raises(ValueError, match="ensemble_config.pooling_method"):
        with_readout("ensemble", {"ensemble_config": {"pooling_method": "sum"}}, "add")


# --------------------------------------------------------------------------- every model kind


def test_the_registry_has_message_passing_models():
    assert {"GCN", "GAT", "GraphSAGE", "GIN"} <= set(MODELS)


@pytest.mark.parametrize("readout", READOUTS)
@pytest.mark.parametrize("name", MODELS)
def test_every_message_passing_model_pools_with_the_configured_readout(name, readout):
    try:
        model, info = _build(name, readout)
    except Exception as exc:  # a model needing parameters beyond the basic ones (PNA: deg)
        pytest.skip(f"{name} is not built from {sorted(BASE)} alone: {type(exc).__name__}")
    assert model.pooling_method == readout and info["readout"] == readout
    assert info["hyperparameters_values"]["pooling_method"] == readout  # recorded = applied
    small, large = _predict(model, _molecules(3, 6))
    assert _expected(readout, small, large)


@pytest.mark.usefixtures("example_models")
def test_a_plugin_model_pools_with_the_configured_readout():
    model, info = _build("ExampleGNN", "add")
    assert info["readout"] == "add"
    small, large = _predict(model, _molecules(3, 6))
    assert _expected("add", small, large)


@pytest.mark.parametrize("readout", READOUTS)
def test_an_ensemble_pools_with_its_readout(readout):
    config = {**_ensemble_config("parallel"), "pooling_method": readout}
    torch.manual_seed(0)
    model, info = ModelFactory().create_model_with_info(
        name="ensemble",
        hyperparameters={"ensemble_config": config, "in_channels": FEATURES, "out_channels": 1},
        task_type="graph_regression",
        sample_data=SAMPLE,
    )
    assert info["readout"] == applied_readout(model) == readout
    small, large = _predict(model, _molecules(3, 6))
    assert _expected(readout, small, large)


def test_a_hierarchical_ensemble_reads_out_with_its_readout_at_its_last_level():
    from milia_pipeline.models.builders.model_composer import ModelComposer, ModelSpec

    composer = ModelComposer(task_type="graph_regression").set_readout("add")
    composer.set_strategy("hierarchical")
    for level in (0, 1):
        composer.models.append(
            ModelSpec(model=torch.nn.Linear(FEATURES, FEATURES), level=level, weight=1.0)
        )
    composed = composer.build()
    assert composed.pooling_method == "add"
    assert [e.pooling_method for e in composed.level_ensembles] == ["add", "add"]


def test_an_unknown_readout_is_refused_by_the_factory():
    with pytest.raises(ModelValidationError, match="not a readout"):
        _build("GCN", "sum")
    config = {**_ensemble_config(), "pooling_method": "sum"}
    with pytest.raises(Exception, match="not a readout"):
        ModelFactory().create_model_with_info(
            name="ensemble",
            hyperparameters={"ensemble_config": config, "in_channels": FEATURES},
            task_type="graph_regression",
            sample_data=SAMPLE,
        )


def test_a_node_level_task_has_no_readout():
    torch.manual_seed(0)
    _, info = ModelFactory().create_model_with_info(
        name="GCN", hyperparameters=dict(BASE), task_type="node_regression", sample_data=SAMPLE
    )
    assert info["readout"] is None


# --------------------------------------------------------------------------- rebuilt as trained


def _checkpoint(tmp_path, model, info, name="GCN"):
    from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
        CheckpointManager,
    )

    return CheckpointManager(working_root_dir=tmp_path).save(
        tmp_path / "best.pt",
        model,
        hyper_parameters={
            "model_name": name,
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )


def _rebuild(path):
    from milia_pipeline.models.post_training.inference.model_loader import ModelLoader

    model, _ = ModelLoader.load_from_checkpoint(
        path, working_root_dir=path.parent, device=torch.device("cpu")
    )
    return model


@pytest.mark.parametrize("readout", READOUTS)
def test_a_rebuilt_model_pools_as_it_was_trained(tmp_path, readout):
    model, info = _build("GCN", readout)
    rebuilt = _rebuild(_checkpoint(tmp_path, model, info))
    assert applied_readout(rebuilt) == readout
    batch = _molecules(3, 6, 4)
    assert _predict(rebuilt, batch) == pytest.approx(_predict(model, batch))


def test_a_checkpoint_from_before_the_recorded_readout_is_rebuilt_with_the_mean(tmp_path, caplog):
    """Before models.readout, a recorded pooling_method was filtered out: the model pooled the mean."""
    model, info = _build("GCN", "mean")
    legacy = {k: v for k, v in info.items() if k != "readout"}
    legacy["hyperparameters_values"] = {**info["hyperparameters_values"], "pooling_method": "add"}
    with caplog.at_level(logging.WARNING):
        rebuilt = _rebuild(_checkpoint(tmp_path, model, legacy))
    assert applied_readout(rebuilt) == "mean"
    assert "predates models.readout" in caplog.text
    batch = _molecules(3, 6)
    assert _predict(rebuilt, batch) == pytest.approx(_predict(model, batch))


# --------------------------------------------------------------------------- training and HPO


def test_hpo_refuses_an_unknown_readout_before_any_trial():
    import main

    config = {"models": {"selection": {"pooling_method": "sum"}, "hpo": {}}}
    with pytest.raises(HPOConfigurationError, match="models.selection.pooling_method"):
        main._build_hpo_config(argparse.Namespace(), logging.getLogger("test"), config)


def test_hpo_no_longer_hoists_the_readout_into_ensembles_itself():
    """The readout is applied once for every model (with_readout), not by the ensemble builder."""
    from milia_pipeline.models.hpo.hpo_manager import HPOManager

    manager = HPOManager.__new__(HPOManager)
    built = manager._build_composite_hyperparameters(
        "ensemble", {}, {"selection": {"pooling_method": "add"}, "ensemble": _ensemble_config()}
    )
    assert "pooling_method" not in built["ensemble_config"]
    assert with_readout("ensemble", built, "add")["ensemble_config"]["pooling_method"] == "add"
