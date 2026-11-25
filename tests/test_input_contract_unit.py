"""The input contract: which input kinds a model accepts, derived from its checkpoint records.

Rows of the matrix: every record a checkpoint can lack (fail closed), QM inputs from the
dataset, positional models (by declared capabilities, by forward signature, inside a composed
model), and the refusal at every place a model meets raw inputs: ``Predictor.prepare_inputs``,
packaging, serving start-up, the serving image and ONNX export.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from milia_pipeline.exceptions import ConfigurationError, DeploymentError
from milia_pipeline.models.capabilities import ModelCapabilities, OutputSpec, attach_capabilities
from milia_pipeline.models.input_contract import (
    DATASET,
    IDENTIFIER,
    IDENTIFIER_RECORDS,
    INPUT_KINDS,
    STRUCTURE,
    input_contract,
    model_requires_positions,
)
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.inference.predictor import Predictor

RECORDED = {
    "explicit_hydrogens": False,
    "qm_input_features": [],
    "input_transforms": None,
    "input_filters": {"max_atoms": None, "min_atoms": None, "heavy_atom_filter": None},
}


def test_a_fully_recorded_identifier_model_accepts_identifiers_and_datasets():
    contract = input_contract(RECORDED, requires_positions=False)
    assert contract.accepted == [IDENTIFIER, DATASET]
    assert set(contract.refused) == {STRUCTURE}
    contract.require(IDENTIFIER)
    contract.require(DATASET)
    json.dumps(contract.to_dict())


@pytest.mark.parametrize("record", IDENTIFIER_RECORDS)
def test_a_missing_record_refuses_identifiers(record):
    """Not recorded is unknown, never "no": the checkpoint cannot show inputs match training."""
    data_info = {k: v for k, v in RECORDED.items() if k != record}
    contract = input_contract(data_info, requires_positions=False)
    assert contract.accepted == [DATASET]
    assert f"does not record {record}" in contract.refused[IDENTIFIER]


def test_a_checkpoint_without_data_info_accepts_datasets_only():
    assert input_contract(None, requires_positions=False).accepted == [DATASET]


def test_qm_inputs_from_the_dataset_refuse_identifiers_naming_them():
    data_info = {**RECORDED, "qm_input_features": ["atom.atomic_charge", "x.Vesp"]}
    contract = input_contract(data_info, requires_positions=False)
    assert contract.accepted == [DATASET]
    assert "atom.atomic_charge, x.Vesp from its dataset's QM data" in contract.refused[IDENTIFIER]


def test_inputs_from_the_given_coordinates_point_identifiers_to_structures():
    """A geometric input (bond lengths of the given coordinates) is not dataset QM data: a
    structure carries it, an identifier does not."""
    data_info = {
        **RECORDED,
        "qm_input_features": ["bond.bond_length"],
        "qm_input_sources": {"bond.bond_length": "coordinates"},
    }
    reason = input_contract(data_info, requires_positions=False).refused[IDENTIFIER]
    assert "bond.bond_length from the given coordinates" in reason
    assert "give the molecule as a structure" in reason and "QM data" not in reason


def test_a_positional_model_refuses_identifiers():
    contract = input_contract(RECORDED, requires_positions=True)
    assert contract.accepted == [DATASET]
    assert "reads atomic positions" in contract.refused[IDENTIFIER]


def test_require_names_the_reason_and_the_accepted_kinds():
    contract = input_contract({}, requires_positions=False)
    with pytest.raises(ConfigurationError, match=r"Accepted: dataset\.") as info:
        contract.require(IDENTIFIER)
    assert "does not accept identifier inputs" in str(info.value)
    with pytest.raises(ConfigurationError, match="Unknown input kind"):
        contract.require("image")
    assert set(INPUT_KINDS) == {IDENTIFIER, STRUCTURE, DATASET}


class _Positional(nn.Module):
    def forward(self, z, pos, batch=None):
        return pos.sum(-1)


class _Feature(nn.Module):
    def forward(self, x, edge_index, batch=None):
        return x


def _declared(requires_positions):
    module = _Feature()
    attach_capabilities(
        module,
        ModelCapabilities(
            outputs=(OutputSpec("features", "node", "features"),),
            requires_positions=requires_positions,
        ),
    )
    return module


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_Feature(), False),
        (_Positional(), True),  # by forward signature
        (_declared(True), True),  # by attached capabilities
        (_declared(False), False),
        (nn.ModuleList([_Feature(), _Positional()]), True),  # composed: one member reads pos
        (nn.Sequential(_declared(False)), False),
    ],
)
def test_positions_are_read_from_every_network_inside_the_model(model, expected):
    assert model_requires_positions(model) is expected


def test_a_factory_model_reports_its_capabilities():
    from milia_pipeline.models.factory.model_factory import ModelFactory

    gcn, _ = ModelFactory().create_model_with_info(
        name="GCN",
        hyperparameters={
            "in_channels": 4,
            "hidden_channels": 8,
            "out_channels": 1,
            "num_layers": 2,
        },
        task_type="graph_regression",
    )
    assert model_requires_positions(gcn) is False


def test_schnet_requires_positions():
    pytest.importorskip("pyg_lib", reason="SchNet's radius_graph needs pyg-lib")
    from milia_pipeline.models.factory.model_factory import ModelFactory

    schnet, _ = ModelFactory().create_model_with_info(
        name="SchNet",
        hyperparameters={
            "hidden_channels": 16,
            "num_filters": 16,
            "num_interactions": 1,
            "num_gaussians": 10,
            "cutoff": 5.0,
            "out_channels": 1,
        },
        task_type="graph_regression",
    )
    assert model_requires_positions(schnet) is True


def _predictor(model, data_info):
    return Predictor(
        model=model,
        working_root_dir=".",
        device=torch.device("cpu"),
        model_info={"data_info": data_info},
    )


def test_a_positional_predictor_refuses_featurized_identifiers():
    predictor = _predictor(_Positional(), RECORDED)
    with pytest.raises(ConfigurationError, match="reads atomic positions"):
        predictor.prepare_inputs([convert_to_pyg("CCO", format="smiles")])


def test_an_identifier_predictor_prepares_featurized_identifiers():
    predictor = _predictor(nn.Identity(), RECORDED)
    (graph,) = predictor.prepare_inputs([convert_to_pyg("CCO", format="smiles")])
    assert graph.x is not None


def _checkpoint(tmp_path, data_info):
    from milia_pipeline.models.factory.model_factory import ModelFactory
    from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
        CheckpointManager,
    )

    torch.manual_seed(0)
    in_channels = convert_to_pyg("CCO", format="smiles").x.shape[1]
    model, info = ModelFactory().create_model_with_info(
        name="GCN",
        hyperparameters={
            "in_channels": in_channels,
            "hidden_channels": 8,
            "out_channels": 1,
            "num_layers": 2,
        },
        task_type="graph_regression",
    )
    return CheckpointManager(working_root_dir=tmp_path).save(
        tmp_path / "checkpoints" / "best.pt",
        model,
        data_info=data_info,
        hyper_parameters={
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )


def test_an_unrecorded_checkpoint_packages_for_datasets_but_is_not_served(tmp_path):
    """F95: a checkpoint written before the records packages (checksums verified), records
    its contract, and is neither served, given a serving image, nor exported to ONNX."""
    from milia_pipeline.models.deployment.deploy import _export_onnx
    from milia_pipeline.models.deployment.package import PACKAGE_MANIFEST, package_checkpoint
    from milia_pipeline.models.deployment.settings import load_deployment_settings

    package = package_checkpoint(_checkpoint(tmp_path, {}), tmp_path / "package")
    kinds = json.loads((package / PACKAGE_MANIFEST).read_text())["signature"]["inputs"]["kinds"]
    assert kinds["accepted"] == [DATASET]
    assert "does not record" in kinds["refused"][IDENTIFIER]
    with pytest.raises(DeploymentError, match="does not accept identifier inputs"):
        _export_onnx(package)

    from milia_pipeline.models.deployment.serving import write_serving_dockerfile

    with pytest.raises(DeploymentError, match="'local' target"):
        write_serving_dockerfile(package)
    assert not (package / "Dockerfile").exists()

    from milia_pipeline.models.deployment.deploy import deploy

    out = tmp_path / "out"
    with pytest.raises(DeploymentError, match="Cannot serve"):
        deploy(
            tmp_path / "checkpoints" / "best.pt",
            load_deployment_settings({"target": "container", "output_dir": str(out)}),
            tmp_path,
        )
    assert not any(p.name.isdigit() for p in out.rglob("*"))  # nothing published
    local = deploy(
        tmp_path / "checkpoints" / "best.pt",
        load_deployment_settings({"target": "local", "output_dir": str(out)}),
        tmp_path,
    )
    assert local.manifest["signature"]["inputs"]["kinds"]["accepted"] == [DATASET]

    pytest.importorskip("fastapi", reason="the `serve` extra is not installed")
    from milia_pipeline.models.deployment.serving import PackagedPredictor

    with pytest.raises(DeploymentError, match="does not record"):
        PackagedPredictor.from_package(package)
