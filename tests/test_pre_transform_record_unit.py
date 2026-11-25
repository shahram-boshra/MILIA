"""Training pre-transforms recorded in the checkpoint and applied at inference (F83)."""

from __future__ import annotations

import argparse
import logging
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose, NormalizeFeatures

from milia_pipeline.exceptions import ConfigurationError
from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.inference.predictor import Predictor
from milia_pipeline.transformations.graph_transforms import register_all_custom_transforms
from milia_pipeline.transformations.pre_transform_record import (
    RECORD_VERSION,
    instantiate_transforms,
    load_input_transform,
    pipeline_repr,
    record_pre_transforms,
)

DEFAULT_SPECS = [
    {"name": "NormalizeFeatures", "kwargs": {"attrs": ["x"]}},
    {"name": "StandardizeTargets", "kwargs": {"attrs": ["y"], "eps": 1e-8}},
]
DEFAULT_REPR = "Compose([\n  NormalizeFeatures(),\n  StandardizeTargets()\n])"


def _record(specs=DEFAULT_SPECS, description=DEFAULT_REPR):
    return {"version": RECORD_VERSION, "transforms": specs, "repr": description}


class _Processed(InMemoryDataset):
    """A dataset PyG processes with ``pre_transform`` (writes ``processed/pre_transform.pt``)."""

    def __init__(self, root, pre_transform, specs=None):
        self.pre_transform_config = specs
        super().__init__(str(root), pre_transform=pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        graphs = []
        for i, smiles in enumerate(["CCO", "CCN", "c1ccccc1", "CC(=O)O"] * 5):
            graph = convert_to_pyg(smiles, format="smiles")
            graph.y = torch.tensor([[float(graph.x.size(0)), float(i % 3)]])
            graphs.append(self.pre_transform(graph) if self.pre_transform else graph)
        self.save(graphs, self.processed_paths[0])


class TestRecord:
    def test_pipeline_repr_is_pygs_description(self):
        assert pipeline_repr(None) == "None"
        assert pipeline_repr(Compose([NormalizeFeatures()])) == (
            "Compose([\n  NormalizeFeatures()\n])"
        )

    def test_a_processed_dataset_is_recorded_with_its_configuration(self, tmp_path):
        specs = [{"name": "NormalizeFeatures", "kwargs": {"attrs": ["x"]}}]
        dataset = _Processed(tmp_path, instantiate_transforms(specs), specs)
        assert record_pre_transforms(dataset) == {
            "version": RECORD_VERSION,
            "transforms": specs,
            "repr": "Compose([\n  NormalizeFeatures()\n])",
        }

    def test_a_dataset_processed_with_another_pipeline_is_refused(self, tmp_path):
        _Processed(tmp_path, None)  # processed without a pre-transform
        with pytest.warns(UserWarning, match="pre_transform"):
            reopened = _Processed(tmp_path, Compose([NormalizeFeatures()]), DEFAULT_SPECS[:1])
        with pytest.raises(ConfigurationError, match="was processed with the pre-transform 'None'"):
            record_pre_transforms(reopened)

    def test_no_pre_transform_records_nothing(self, tmp_path):
        assert record_pre_transforms(_Processed(tmp_path, None)) is None

    def test_a_pipeline_without_configuration_is_recorded_as_not_rebuildable(self, tmp_path):
        record = record_pre_transforms(_Processed(tmp_path, Compose([NormalizeFeatures()])))
        assert record["transforms"] is None
        with pytest.raises(ConfigurationError, match="does not record as a configuration"):
            load_input_transform(record)


class TestLoad:
    def test_the_recorded_pipeline_is_rebuilt_and_applied(self):
        transform = load_input_transform(_record())
        graph = convert_to_pyg("CCO", format="smiles")
        assert not torch.allclose(graph.x.sum(-1), torch.ones(graph.num_nodes))
        assert torch.allclose(transform(graph).x.sum(-1), torch.ones(graph.num_nodes))

    def test_no_record_means_no_transform(self):
        assert load_input_transform(None) is None

    def test_a_pipeline_that_rebuilds_differently_is_refused(self):
        with pytest.raises(ConfigurationError, match="differ from the recorded ones"):
            load_input_transform(_record(description="Compose([\n  NormalizeFeatures()\n])"))

    def test_an_unknown_transform_or_version_is_refused(self):
        with pytest.raises(ConfigurationError, match="Cannot rebuild"):
            load_input_transform(_record(specs=[{"name": "NoSuchTransform", "kwargs": {}}]))
        with pytest.raises(ConfigurationError, match="record version 2"):
            load_input_transform({**_record(), "version": 2})


class TestMiliaDataset:
    def _dataset(self, root):
        from milia_pipeline.config.config_containers import (
            DatasetConfig,
            FilterConfig,
            ProcessingConfig,
        )
        from milia_pipeline.datasets.milia_dataset import miliaDataset

        with (
            patch.object(miliaDataset, "_download", return_value=None),
            patch.object(miliaDataset, "_process", return_value=None),
        ):
            return miliaDataset(
                root=str(root),
                dataset_config=DatasetConfig(dataset_type="DFT"),
                filter_config=FilterConfig(),
                processing_config=ProcessingConfig(scalar_graph_targets=[]),
            )

    def test_the_configured_pipeline_is_recorded_and_rebuilds(self, tmp_path):
        register_all_custom_transforms()  # as the CLI does at startup
        dataset = self._dataset(tmp_path)
        record = record_pre_transforms(dataset)
        assert record["transforms"] == [
            {"name": t["name"], "kwargs": t["kwargs"]} for t in dataset.pre_transform_config
        ]
        assert pipeline_repr(load_input_transform(record)) == pipeline_repr(dataset.pre_transform)

    def test_a_pipeline_built_differently_from_its_configuration_is_not_recorded(self, tmp_path):
        from torch_geometric.transforms import AddSelfLoops

        from milia_pipeline.datasets.milia_dataset import miliaDataset

        with patch.object(
            miliaDataset,
            "_create_transforms_with_phase2_system",
            return_value=Compose([AddSelfLoops()]),  # not the configured pipeline
        ):
            dataset = self._dataset(tmp_path)
        assert dataset.pre_transform_config is None


def _predictor(data_info, **records):
    """A predictor whose checkpoint records ``data_info`` and, by default, the other input
    contract records of a heavy-atom run without QM inputs."""
    records = {
        "explicit_hydrogens": False,
        "qm_input_features": [],
        "input_filters": {"max_atoms": None, "min_atoms": None, "heavy_atom_filter": None},
        **records,
    }
    return Predictor(
        model=nn.Identity(),
        working_root_dir=".",
        device=torch.device("cpu"),
        model_info={"data_info": {**records, **data_info}},
    )


class TestPredictor:
    def test_raw_inputs_get_the_recorded_transforms(self):
        predictor = _predictor({"input_transforms": _record()})
        (graph,) = predictor.prepare_inputs([convert_to_pyg("CCO", format="smiles")])
        assert torch.allclose(graph.x.sum(-1), torch.ones(graph.num_nodes))

    def test_a_dataset_without_pre_transform_leaves_inputs_unchanged(self):
        raw = convert_to_pyg("CCO", format="smiles")
        (graph,) = _predictor({"input_transforms": None}).prepare_inputs([raw])
        assert torch.equal(graph.x, raw.x)

    def test_a_checkpoint_without_the_record_refuses_identifier_inputs(self):
        """Not recorded is unknown, not "none": featurized inputs could differ from training."""
        from milia_pipeline.exceptions import ConfigurationError

        predictor = _predictor({})
        assert not predictor.records_input_transforms
        with pytest.raises(ConfigurationError, match="does not record input_transforms"):
            predictor.prepare_inputs([convert_to_pyg("CCO", format="smiles")])


def _train(tmp_path, dataset):
    import main

    config = {
        "global_paths": {"working_root_dir": str(tmp_path)},
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
    assert main._run_standard_training(args, logging.getLogger("test"), dataset, config) == 0
    return tmp_path / "checkpoints" / "best.pt"


class TestTrainingToServing:
    def test_training_records_and_the_package_serves_with_the_transforms(self, tmp_path):
        pytest.importorskip("fastapi", reason="the `serve` extra is not installed")
        from milia_pipeline.models.deployment.package import (
            MODEL_CARD,
            PACKAGE_MANIFEST,
            package_checkpoint,
        )
        from milia_pipeline.models.deployment.serving import PackagedPredictor

        specs = [{"name": "NormalizeFeatures", "kwargs": {"attrs": ["x"]}}]
        dataset = _Processed(tmp_path / "data", instantiate_transforms(specs), specs)
        dataset.structural_features_config = None
        best = _train(tmp_path / "run", dataset)
        data_info = load_checkpoint_file(best, map_location="cpu")["data_info"]
        record = data_info["input_transforms"]
        assert record == record_pre_transforms(dataset)
        # the input contract's records: no QM input, no registered dataset type in this config
        assert data_info["qm_input_sources"] == {} and data_info["molecule_creation"] is None

        package = package_checkpoint(best, tmp_path / "package")
        import json

        inputs = json.loads((package / PACKAGE_MANIFEST).read_text())["signature"]["inputs"]
        assert inputs["transforms"] == record["repr"] and inputs["transforms_recorded"] is True
        assert "`NormalizeFeatures`, the training data's pre-transforms" in (
            (package / MODEL_CARD).read_text()
        )

        served = PackagedPredictor.from_package(package)
        rows = served.predict_smiles(["CCO", "c1ccccc1"])
        graphs = [convert_to_pyg(s, format="smiles") for s in ("CCO", "c1ccccc1")]
        transformed = served.predictor.predict_batch(
            [dataset.pre_transform(g) for g in graphs], return_numpy=True
        )
        raw = served.predictor.predict_batch(graphs, return_numpy=True)
        assert torch.allclose(torch.tensor(rows), torch.as_tensor(transformed), atol=1e-6)
        assert not torch.allclose(torch.tensor(rows), torch.as_tensor(raw), atol=1e-4)

    def test_predict_mode_transforms_molecule_files_but_not_processed_graphs(self, tmp_path):
        import main

        specs = [{"name": "NormalizeFeatures", "kwargs": {"attrs": ["x"]}}]
        dataset = _Processed(tmp_path / "data", instantiate_transforms(specs), specs)
        dataset.structural_features_config = None
        best = _train(tmp_path / "run", dataset)
        molecules = tmp_path / "molecules.csv"
        molecules.write_text("smiles\nCCO\nc1ccccc1\n")
        seen = []
        original = Predictor.predict_batch

        def spy(self, data, *args, **kwargs):
            seen.append([g.x.sum(-1) for g in data])
            return original(self, data, *args, **kwargs)

        config = {"global_paths": {"working_root_dir": str(tmp_path / "run")}}
        with patch.object(Predictor, "predict_batch", spy):
            for test_path in (molecules, dataset.processed_paths[0]):
                args = argparse.Namespace(
                    model_path=str(best),
                    test_path=str(test_path),
                    preds_path=str(tmp_path / "preds.csv"),
                )
                assert main.handle_predict_mode(args, logging.getLogger("test"), config) == 0
        for rows in seen:  # molecule file: transformed here; processed file: already transformed
            assert all(torch.allclose(s, torch.ones_like(s)) for s in rows)
        assert len(seen) == 2

    def test_a_package_without_the_record_says_so(self, tmp_path):
        from milia_pipeline.models.deployment.package import MODEL_CARD, package_checkpoint

        best = tmp_path / "old.pt"
        model_dataset = _Processed(tmp_path / "data", None)
        model_dataset.structural_features_config = None
        trained = load_checkpoint_file(_train(tmp_path / "run", model_dataset), map_location="cpu")
        del trained["data_info"]["input_transforms"]
        torch.save(trained, best)
        card = (package_checkpoint(best, tmp_path / "package") / MODEL_CARD).read_text()
        assert "Input transforms: not recorded" in card


def test_trainer_save_checkpoint_writes_the_record(tmp_path):
    from milia_pipeline.models.training.trainer import Trainer

    model = nn.Linear(2, 1)
    trainer = Trainer(
        model=model,
        train_loader=[Data(x=torch.zeros(1, 2))],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        model_info={"task_type": "graph_regression", "input_transforms": _record()},
    )
    path = tmp_path / "manual.pt"
    trainer.save_checkpoint(path)
    assert load_checkpoint_file(path, map_location="cpu")["data_info"]["input_transforms"] == (
        _record()
    )
