"""Serving structure inputs, structure examples at packaging, and the training filters on inputs.

* The training filters (``data_info.input_filters``: atom counts, heavy-atom set) refuse inputs
  outside them, identifiers and structures alike, each named with its reason, by the training
  filter code; a record that a run did not compute (``None``) is unrecorded, never "nothing".
* Training records a structure example (``run_info.structure_example``); packaging verifies it;
  the server serves structure instances only when it is recorded.
* The server takes SMILES strings and structure objects in one request, in order; a refused
  instance fails the request (422) with every refused index and reason.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time

import numpy as np
import pytest
import torch
import torch.nn as nn

from milia_pipeline.exceptions import DeploymentError, InputRefusedError, StructureInputError
from milia_pipeline.models.input_contract import DATASET, IDENTIFIER, STRUCTURE, input_contract
from milia_pipeline.molecules.featurization_record import (
    input_filters_record,
    molecule_creation_record,
)
from milia_pipeline.molecules.molecule_filters import input_filter_refusals
from milia_pipeline.molecules.structure_inputs import StructureBuilder, StructureInput

FEATURES = {"atom": ["degree", "hybridization", "is_aromatic"], "bond": ["bond_type"]}
NO_FILTERS = {"max_atoms": None, "min_atoms": None, "heavy_atom_filter": None}
WATER = StructureInput.create(
    [8, 1, 1], [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]], charge=0
)
AMMONIA = StructureInput.create(
    [7, 1, 1, 1],
    [[0.0, 0.0, 0.1], [0.94, 0.0, -0.27], [-0.47, 0.81, -0.27], [-0.47, -0.81, -0.27]],
    charge=0,
)


def _graph(atomic_numbers):
    from torch_geometric.data import Data

    return Data(z=torch.tensor(atomic_numbers), num_nodes=len(atomic_numbers))


# ----------------------------------------------------------------------------- filters


def test_the_record_holds_the_configured_structural_filters():
    from milia_pipeline.config.config_containers import FilterConfig

    record = input_filters_record(
        FilterConfig(max_atoms=50, heavy_atom_filter={"mode": "include", "atoms": ["C", "O"]})
    )
    assert record == {
        "max_atoms": 50,
        "min_atoms": None,
        "heavy_atom_filter": {"mode": "include", "atoms": ["C", "O"]},
    }
    assert input_filters_record(FilterConfig()) == NO_FILTERS
    json.dumps(input_filters_record())  # the global configuration's, JSON for the checkpoint


@pytest.mark.parametrize(
    ("record", "atoms", "refused_with"),
    [
        ({**NO_FILTERS, "max_atoms": 3}, [6, 1, 1, 1, 1], "exceeds max_atoms=3"),
        ({**NO_FILTERS, "min_atoms": 4}, [8, 1, 1], "below min_atoms=4"),
        (
            {**NO_FILTERS, "heavy_atom_filter": {"mode": "include", "atoms": ["C", "O"]}},
            [7, 1, 1, 1],
            "unallowed heavy atoms [7]",
        ),
        (
            {**NO_FILTERS, "heavy_atom_filter": {"mode": "exclude", "atoms": ["Cl"]}},
            [6, 17, 1, 1, 1],
            "excluded heavy atom(s) [17]",
        ),
    ],
)
def test_an_input_outside_a_training_filter_is_refused_with_its_reason(record, atoms, refused_with):
    """The training filter code decides (``apply_atom_count_filters``, ``apply_heavy_atom_filters``)."""
    refusals = input_filter_refusals([_graph([8, 1, 1]), _graph(atoms)], record, indices=[4, 7])
    if "min_atoms" in refused_with:
        assert [index for index, _ in refusals] == [4, 7]
    else:
        assert [index for index, _ in refusals] == [7]
    assert refused_with in refusals[-1][1] and "outside the training filters" in refusals[-1][1]
    assert input_filter_refusals([_graph([8, 1, 1])], NO_FILTERS) == []


@pytest.mark.parametrize(
    "record", ["qm_input_features", "qm_input_sources", "molecule_creation", "input_filters"]
)
def test_a_record_the_run_did_not_compute_is_unrecorded(record):
    """A checkpoint writer stores ``None`` for a record its training path did not compute (the
    HPO final model's ``data_info``): unknown, never "no QM input" or "no filter"."""
    creation = molecule_creation_record("QDPi")
    creation["charges"] = [0]
    data_info = {
        "explicit_hydrogens": False,
        "qm_input_features": [],
        "qm_input_sources": {},
        "input_transforms": None,
        "structural_features_config": FEATURES,
        "molecule_creation": creation,
        "input_filters": NO_FILTERS,
    }
    assert input_contract(data_info, requires_positions=False).accepted == [
        IDENTIFIER,
        STRUCTURE,
        DATASET,
    ]
    data_info[record] = None
    contract = input_contract(data_info, requires_positions=False)
    # identifiers read qm_input_features and input_filters; structures every record but
    # qm_input_features (their inputs' sources are qm_input_sources)
    if record in ("qm_input_features", "input_filters"):
        assert f"does not record {record}" in contract.refused[IDENTIFIER]
    if record != "qm_input_features":
        assert f"does not record {record}" in contract.refused[STRUCTURE]
    else:
        assert contract.accepts(STRUCTURE)


def test_every_record_unset_accepts_datasets_only():
    every = (
        "explicit_hydrogens",
        "qm_input_features",
        "qm_input_sources",
        "input_transforms",
        "structural_features_config",
        "molecule_creation",
        "input_filters",
    )
    assert input_contract(dict.fromkeys(every), requires_positions=False).accepted == [DATASET]


def _data_info(dataset="QDPi", filters=NO_FILTERS, sources=None):
    creation = molecule_creation_record(dataset)
    creation["charges"] = [0]
    sources = sources or {}
    return {
        "explicit_hydrogens": True,
        "qm_input_features": list(sources),
        "qm_input_sources": sources,
        "input_transforms": None,
        "structural_features_config": FEATURES,
        "molecule_creation": creation,
        "input_filters": filters,
    }


def _predictor(data_info, model=None):
    from milia_pipeline.models.post_training.inference.predictor import Predictor

    return Predictor(
        model=model or nn.Identity(),
        working_root_dir=".",
        device=torch.device("cpu"),
        model_info={"data_info": data_info},
    )


def test_identifier_inputs_outside_the_filters_are_refused_by_index():
    from milia_pipeline.models.post_training import convert_to_pyg

    predictor = _predictor(_data_info(filters={**NO_FILTERS, "max_atoms": 5}))
    graphs = [
        convert_to_pyg(s, format="smiles", add_hydrogens=True, structural_features_config=FEATURES)
        for s in ("O", "CCO", "N")
    ]
    with pytest.raises(InputRefusedError, match="exceeds max_atoms=5") as info:
        predictor.prepare_inputs(graphs, indices=[0, 2, 5])
    assert [index for index, _ in info.value.refusals] == [2]  # CCO: 9 atoms with hydrogens
    assert len(predictor.prepare_inputs([graphs[0], graphs[2]])) == 2


def test_structure_refusals_of_every_kind_are_reported_together_in_order():
    predictor = _predictor(_data_info(filters={**NO_FILTERS, "max_atoms": 3}))
    no_charge = StructureInput.create(WATER.atomic_numbers, WATER.coordinates)
    with pytest.raises(StructureInputError) as info:
        predictor.prepare_structures([AMMONIA, WATER, no_charge], indices=[3, 5, 9])
    refusals = dict(info.value.refusals)
    assert sorted(refusals) == [3, 9]
    assert "exceeds max_atoms=3" in refusals[3] and "needs the molecular charge" in refusals[9]
    assert len(predictor.prepare_structures([WATER])) == 1


# ----------------------------------------------------------------------------- run_info


def _training_graph(dataset, structure, inchi=None):
    graph = StructureBuilder(_data_info(dataset)["molecule_creation"], FEATURES).build_one(
        structure
    )
    if inchi is not None:
        graph.inchi = inchi
    return graph


def test_training_records_structure_examples_that_rebuild_their_graphs():
    from milia_pipeline.models.training.run_info import example_structures

    graphs = [_training_graph("QDPi", WATER), _training_graph("QDPi", AMMONIA)]
    examples = example_structures(iter(graphs), "QDPi", limit=1)
    assert len(examples) == 1 and examples[0]["identifier"] is None  # coordinate-based
    json.dumps(examples)
    rebuilt = StructureBuilder(_data_info("QDPi")["molecule_creation"], FEATURES).build_one(
        StructureInput.from_dict(examples[0])
    )
    for name in ("x", "edge_index", "edge_attr", "z", "molecular_charge"):
        assert torch.equal(rebuilt[name], graphs[0][name]), name
    assert torch.allclose(rebuilt.pos, graphs[0].pos)


def test_an_identifier_dataset_example_carries_its_identifier():
    from milia_pipeline.models.training.run_info import example_structures

    water = StructureInput.create(
        WATER.atomic_numbers,
        WATER.coordinates,
        identifier="InChI=1S/H2O/h1H2",
        identifier_type="inchi",
    )
    (example,) = example_structures([_training_graph("DFT", water)], "DFT")
    assert (example["identifier"], example["identifier_type"]) == ("InChI=1S/H2O/h1H2", "inchi")
    assert example_structures([_graph([8, 1, 1])], "DFT") == []  # no pos, no charge: skipped
    assert example_structures([_training_graph("DFT", water)], "NoSuchDataset") == []


def test_run_info_reads_the_training_molecules_once_for_both_examples():
    from milia_pipeline.models.training.run_info import collect_run_info

    graphs = [_training_graph("QDPi", WATER)]
    graphs[0].smiles = "O"
    info = collect_run_info(
        {"dataset_type": "QDPi"},
        task_type="graph_regression",
        out_channels=1,
        target_names=None,
        data_split=None,
        train_graphs=(g for g in graphs),  # a generator, as main.py passes it
    )
    assert info["input_example"] == ["O"] and len(info["structure_example"]) == 1


def test_structure_input_dict_round_trip_and_unknown_fields():
    assert StructureInput.from_dict(AMMONIA.to_dict()) == AMMONIA
    with pytest.raises(ValueError, match="unknown structure fields"):
        StructureInput.from_dict({**WATER.to_dict(), "spin": 1})
    with pytest.raises(ValueError, match="needs atomic_numbers and coordinates"):
        StructureInput.from_dict({"charge": 0})


# ----------------------------------------------------------------------------- packaging, serving


def _width(dataset="QDPi"):
    return _training_graph(dataset, WATER).x.shape[1]


def _checkpoint(tmp_path, data_info, structure_example):
    from milia_pipeline.models.factory.model_factory import ModelFactory
    from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
        CheckpointManager,
    )

    torch.manual_seed(0)
    model, info = ModelFactory().create_model_with_info(
        name="GCN",
        hyperparameters={
            "in_channels": _width(),
            "hidden_channels": 8,
            "out_channels": 2,
            "num_layers": 2,
        },
        task_type="graph_regression",
    )
    path = tmp_path / "checkpoints" / "best.pt"
    CheckpointManager(working_root_dir=tmp_path).save(
        path,
        model,
        data_info=data_info,
        hyper_parameters={
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": info["hyperparameters_values"],
            "model_info": info,
        },
    )
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["run_info"] = {
        "input_example": ["O"],
        "structure_example": structure_example,
        "targets": {"names": ["a", "b"], "count": 2},
    }
    torch.save(checkpoint, path)
    return path


STRUCTURE_ONLY = {"bond.bond_type": "coordinates"}  # an input from the given geometry


@pytest.fixture
def structure_package(tmp_path):
    from milia_pipeline.models.deployment.package import package_checkpoint

    examples = [WATER.to_dict(), AMMONIA.to_dict()]
    path = _checkpoint(tmp_path, _data_info(sources=STRUCTURE_ONLY), examples)
    return path, package_checkpoint(path, tmp_path / "package")


def _manifest(package):
    from milia_pipeline.models.deployment.package import PACKAGE_MANIFEST

    return json.loads((package / PACKAGE_MANIFEST).read_text())


def test_packaging_records_and_verifies_the_structure_example(structure_package):
    from milia_pipeline.models.deployment.package import MODEL_CARD

    _, package = structure_package
    manifest = _manifest(package)
    assert manifest["format"] == "milia-deployment/4"
    assert [s["atomic_numbers"] for s in manifest["structure_example"]] == [[8, 1, 1], [7, 1, 1, 1]]
    assert manifest["signature"]["inputs"]["kinds"]["accepted"] == [STRUCTURE, DATASET]
    card = (package / MODEL_CARD).read_text()
    assert "2 training molecules, verified at packaging" in card


def test_a_structure_example_the_package_does_not_reproduce_removes_it(tmp_path, monkeypatch):
    from milia_pipeline.models.deployment import package as package_module

    path = _checkpoint(tmp_path, _data_info(sources=STRUCTURE_ONLY), [WATER.to_dict()])
    calls = []

    def predictions(checkpoint, example):
        calls.append(checkpoint)
        return torch.zeros(len(example), 2) + len(calls)  # the package's differs from the source's

    monkeypatch.setattr(package_module, "_structure_predictions", predictions)
    with pytest.raises(DeploymentError, match="structure example"):
        package_module.package_checkpoint(path, tmp_path / "package")
    assert not (tmp_path / "package" / package_module.PACKAGE_MANIFEST).exists()


def test_a_structure_model_without_an_example_packages_but_is_not_served(tmp_path):
    from milia_pipeline.models.deployment.package import MODEL_CARD, package_checkpoint
    from milia_pipeline.models.deployment.serving import PackagedPredictor, require_servable

    path = _checkpoint(tmp_path, _data_info(sources=STRUCTURE_ONLY), [])
    package = package_checkpoint(path, tmp_path / "package")
    assert _manifest(package)["structure_example"] == []
    assert "structures are not served" in (package / MODEL_CARD).read_text()
    with pytest.raises(DeploymentError, match="records no structure example"):
        PackagedPredictor.from_package(package)
    with pytest.raises(DeploymentError, match="'local' target"):
        require_servable(package)


def test_a_structure_only_model_is_served_structures(structure_package):
    from milia_pipeline.models.deployment.serving import PackagedPredictor
    from milia_pipeline.models.post_training.inference.predictor import Predictor

    source, package = structure_package
    served = PackagedPredictor.from_package(package)
    assert served.serves_structures and not served.serves_identifiers
    rows = served.predict_instances([AMMONIA.to_dict(), WATER.to_dict()])
    direct = Predictor.from_checkpoint(source, working_root_dir=source.parent)
    expected = direct.predict_batch(direct.prepare_structures([AMMONIA, WATER]), return_numpy=True)
    np.testing.assert_allclose(rows, expected, rtol=1e-6)
    with pytest.raises(InputRefusedError, match="does not accept SMILES") as info:
        served.predict_instances([WATER.to_dict(), "CCO"])
    assert [index for index, _ in info.value.refusals] == [1]


def test_a_model_taking_both_kinds_serves_a_mixed_request_in_order(tmp_path):
    from milia_pipeline.models.deployment.package import package_checkpoint
    from milia_pipeline.models.deployment.serving import PackagedPredictor

    path = _checkpoint(tmp_path, _data_info(), [WATER.to_dict()])
    served = PackagedPredictor.from_package(package_checkpoint(path, tmp_path / "package"))
    assert served.serves_identifiers and served.serves_structures
    mixed = served.predict_instances(["O", AMMONIA.to_dict(), "N"])
    alone = served.predict_instances([AMMONIA.to_dict()]) + served.predict_instances(["O", "N"])
    np.testing.assert_allclose(mixed, [alone[1], alone[0], alone[2]], rtol=1e-6)


@contextlib.contextmanager
def _serving(package):
    requests = pytest.importorskip("requests")
    uvicorn = pytest.importorskip("uvicorn")
    from milia_pipeline.models.deployment.serving import create_app

    server = uvicorn.Server(
        uvicorn.Config(create_app(package), host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield lambda body: requests.post(f"http://127.0.0.1:{port}/predict", json=body, timeout=60)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_the_predict_route_takes_structure_objects_and_names_refusals(tmp_path):
    pytest.importorskip("fastapi", reason="the `serve` extra is not installed")
    from milia_pipeline.models.deployment.package import package_checkpoint

    path = _checkpoint(
        tmp_path,
        _data_info(sources=STRUCTURE_ONLY, filters={**NO_FILTERS, "max_atoms": 3}),
        [WATER.to_dict()],
    )
    package = package_checkpoint(path, tmp_path / "package")
    with _serving(package) as post:
        ok = post({"instances": [WATER.to_dict()]})
        refused = post({"instances": [WATER.to_dict(), AMMONIA.to_dict(), "CCO"]})
        unknown = post({"instances": [{**WATER.to_dict(), "spin": 1}]})
    assert ok.status_code == 200 and len(ok.json()["predictions"][0]) == 2
    assert ok.json()["output_names"] == ["a", "b"]
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert [entry["index"] for entry in detail["refused"]] == [2]  # the kind is checked first
    assert "does not accept SMILES" in detail["refused"][0]["reason"]
    assert unknown.status_code == 422  # unknown fields are rejected


def test_the_predict_route_refuses_a_structure_outside_the_filters(tmp_path):
    pytest.importorskip("fastapi", reason="the `serve` extra is not installed")
    from milia_pipeline.models.deployment.package import package_checkpoint

    path = _checkpoint(
        tmp_path,
        _data_info(sources=STRUCTURE_ONLY, filters={**NO_FILTERS, "max_atoms": 3}),
        [WATER.to_dict()],
    )
    with _serving(package_checkpoint(path, tmp_path / "package")) as post:
        response = post({"instances": [WATER.to_dict(), AMMONIA.to_dict()]})
    assert response.status_code == 422
    (entry,) = response.json()["detail"]["refused"]
    assert entry["index"] == 1 and "exceeds max_atoms=3" in entry["reason"]
