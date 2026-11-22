"""Explicit hydrogens: recorded at training, reproduced at inference (F87-1)."""

from __future__ import annotations

import argparse
import json
import logging

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.post_training import convert_sdf_to_pyg_list, convert_to_pyg
from milia_pipeline.models.post_training.inference.predictor import Predictor
from milia_pipeline.molecules.featurization_record import graphs_have_explicit_hydrogens

CONFIG = {"atom": ["degree", "total_degree", "hybridization"], "bond": ["bond_type"]}
ETHANOL_INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"


class TestRecord:
    def test_hydrogens_are_detected_from_atomic_numbers(self):
        with_h = Data(z=torch.tensor([6, 1, 1, 1, 8, 1]))
        heavy = Data(z=torch.tensor([6, 6, 8]))
        assert graphs_have_explicit_hydrogens([heavy, with_h]) is True
        assert graphs_have_explicit_hydrogens([heavy, heavy]) is False
        assert graphs_have_explicit_hydrogens([Data(x=torch.zeros(2, 1))]) is None
        assert graphs_have_explicit_hydrogens([]) is None


class TestFeaturization:
    @pytest.mark.parametrize("text,fmt", [("CCO", "smiles"), (ETHANOL_INCHI, "inchi")])
    def test_add_hydrogens_survives_the_structural_features(self, text, fmt):
        heavy = convert_to_pyg(text, format=fmt, structural_features_config=CONFIG)
        explicit = convert_to_pyg(
            text, format=fmt, structural_features_config=CONFIG, add_hydrogens=True
        )
        assert heavy.x.size(0) == 3
        assert explicit.x.size(0) == 9  # C2H6O: 3 heavy atoms + 6 hydrogens
        assert explicit.x.size(1) == heavy.x.size(1)
        assert explicit.edge_index.size(1) == 16  # 8 bonds, both directions

    def test_coordinate_formats_ignore_the_option(self, tmp_path):
        pytest.importorskip("ase", reason="XYZ conversion needs ASE")
        xyz = tmp_path / "water.xyz"
        xyz.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n")
        graph = convert_to_pyg(str(xyz), format="xyz", add_hydrogens=True)
        assert graph.z.tolist() == [8, 1, 1]  # every atom of the file, positions as given
        assert torch.allclose(graph.pos[1], torch.tensor([0.96, 0.0, 0.0]))

    def test_sdf_molecules_get_hydrogens(self, tmp_path):
        from rdkit import Chem

        sdf = tmp_path / "ethanol.sdf"
        writer = Chem.SDWriter(str(sdf))
        writer.write(Chem.MolFromSmiles("CCO"))
        writer.close()
        (heavy,) = convert_sdf_to_pyg_list(sdf, structural_features_config=CONFIG)
        (explicit,) = convert_sdf_to_pyg_list(
            sdf, structural_features_config=CONFIG, add_hydrogens=True
        )
        assert (heavy.x.size(0), explicit.x.size(0)) == (3, 9)


def _predictor(explicit):
    return Predictor(
        model=nn.Identity(),
        working_root_dir=".",
        device=torch.device("cpu"),
        model_info={"data_info": {"explicit_hydrogens": explicit}},
    )


def test_predictor_featurization_kwargs():
    assert _predictor(True).featurization_kwargs == {"add_hydrogens": True}
    assert _predictor(False).featurization_kwargs == {}
    assert _predictor(None).featurization_kwargs == {}


def _dataset_with_hydrogens():
    from rdkit import Chem

    class _Dataset(list):
        structural_features_config = CONFIG

    graphs = _Dataset()
    for i, smiles in enumerate(["CCO", "CCN", "CCC", "CO", "CC=O"] * 4):
        graph = convert_to_pyg(
            smiles, format="smiles", structural_features_config=CONFIG, add_hydrogens=True
        )
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        graph.z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        graph.y = torch.tensor([[float(graph.x.size(0)) + 0.1 * i]])
        graphs.append(graph)
    return graphs


def test_training_records_and_serving_reproduces_the_hydrogens(tmp_path):
    import main
    from milia_pipeline.models.deployment.package import (
        MODEL_CARD,
        PACKAGE_MANIFEST,
        package_checkpoint,
    )

    config = {
        "global_paths": {"working_root_dir": str(tmp_path / "run")},
        "models": {
            "selection": {"mode": "single", "model_name": "GCN", "task_type": "graph_regression"},
            "training": {"epochs": 2, "batch_size": 4},
            "evaluation": {"visualization": {"enabled": False}},
            "GCN": {"hidden_channels": 8, "num_layers": 2},
        },
    }
    args = argparse.Namespace(
        checkpoint=None,
        mode=None,
        task_type=None,
        model_name=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        evaluate_only=False,
    )
    torch.manual_seed(0)
    dataset = _dataset_with_hydrogens()
    assert main._run_standard_training(args, logging.getLogger("test"), dataset, config) == 0
    best = tmp_path / "run" / "checkpoints" / "best.pt"
    assert load_checkpoint_file(best, map_location="cpu")["data_info"]["explicit_hydrogens"] is True

    package = package_checkpoint(best, tmp_path / "package")
    inputs = json.loads((package / PACKAGE_MANIFEST).read_text())["signature"]["inputs"]
    assert inputs["explicit_hydrogens"] is True
    assert "Explicit hydrogens: yes" in (package / MODEL_CARD).read_text()

    pytest.importorskip("fastapi", reason="the `serve` extra is not installed")
    from milia_pipeline.models.deployment.serving import PackagedPredictor

    served = PackagedPredictor.from_package(package)
    (graph,) = served.featurize(["CCO"])
    assert graph.x.size(0) == 9  # as the training graphs: hydrogens are atoms
    model = served.predictor
    expected = model.predict_batch([dataset[0]], return_numpy=True)
    assert torch.allclose(
        torch.tensor(served.predict_smiles(["CCO"])), torch.as_tensor(expected), atol=1e-5
    )
