"""Regression targets scaled with training-split statistics and unscaled at prediction (F83-2)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.post_training.conformal import conformal_quantiles
from milia_pipeline.models.post_training.inference.predictor import Predictor
from milia_pipeline.models.training.callbacks import ModelCheckpoint
from milia_pipeline.models.training.target_scaling import (
    fit_target_scaler,
    scale_targets,
    unscale_outputs,
    unscale_spread,
)
from milia_pipeline.models.training.trainer import Trainer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
OFFSET = torch.tensor([-400.0, 3.0])  # an energy-like target next to a small one
SCALE = torch.tensor([50.0, 0.5])


class TestScaler:
    def test_per_target_mean_and_population_std(self):
        targets = torch.tensor([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
        scaler = fit_target_scaler(targets)
        assert scaler["method"] == "standard" and scaler["n"] == 3
        assert scaler["mean"] == pytest.approx([3.0, 10.0])
        # population std (numpy/scikit-learn); zero spread keeps std 1
        assert scaler["std"] == pytest.approx([(8 / 3) ** 0.5, 1.0])

    def test_missing_values_are_ignored_and_one_target_may_be_1d(self):
        scaler = fit_target_scaler(torch.tensor([[1.0], [float("nan")], [3.0]]))
        assert (scaler["mean"], scaler["std"]) == ([2.0], [1.0])
        assert fit_target_scaler(torch.tensor([2.0, 4.0]))["mean"] == [3.0]

    def test_scale_and_unscale_are_inverse_and_check_the_width(self):
        targets = torch.randn(6, 2) * SCALE + OFFSET
        scaler = fit_target_scaler(targets)
        scaled = scale_targets(targets, scaler)
        assert torch.allclose(scaled.mean(0), torch.zeros(2), atol=1e-5)
        assert torch.allclose(unscale_outputs(scaled, scaler), targets, atol=1e-4)
        spread = torch.ones(6, 2)
        assert torch.allclose(unscale_spread(spread, scaler), torch.tensor(scaler["std"]).float())
        with pytest.raises(ValueError, match="2 targets; the tensor has 3 columns"):
            scale_targets(torch.zeros(4, 3), scaler)

    def test_unknown_method_and_no_scaler(self):
        with pytest.raises(ValueError, match="Unknown target scaling"):
            fit_target_scaler(torch.zeros(2, 1), method="minmax")
        x = torch.randn(3, 2)
        assert scale_targets(x, None) is x and unscale_outputs(x, None) is x


class _Pooled(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x, edge_index, batch=None):
        return self.lin(global_mean_pool(x, batch))


def _graphs(n, seed, offset=OFFSET):
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(3, 4, generator=generator)
        y = (x.mean(0)[:2] * SCALE + offset).reshape(1, 2)
        out.append(Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), y=y))
    return out


def _trainer(tmp_path, scaling="standard", model=None, epochs=3, **kwargs):
    model = model or _Pooled()
    return Trainer(
        model=model,
        train_loader=DataLoader(_graphs(24, 1), batch_size=8),
        # validation molecules from another distribution: the scaler must not see them
        val_loader=DataLoader(_graphs(19, 2, offset=OFFSET + 100), batch_size=8),
        test_loader=DataLoader(_graphs(8, 3), batch_size=8),
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        device=torch.device("cpu"),
        callbacks=[ModelCheckpoint(dirpath=tmp_path / "checkpoints", verbose=False)],
        model_info={"task_type": "graph_regression"},
        max_epochs=epochs,
        target_scaling=scaling,
        **kwargs,
    )


class TestTrainer:
    def test_the_scaler_is_fit_on_the_training_split_and_stored(self, tmp_path):
        torch.manual_seed(0)
        trainer = _trainer(tmp_path)
        trainer.fit()
        train_y = torch.cat([g.y for g in _graphs(24, 1)])
        assert trainer.target_scaler == fit_target_scaler(train_y)
        best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
        assert best["target_scaler"] == trainer.target_scaler

    def test_validation_and_calibration_are_in_the_targets_units(self, tmp_path):
        torch.manual_seed(0)
        trainer = _trainer(tmp_path)
        trainer.fit()
        trainer.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in trainer.val_loader:
                out = trainer.model(batch.x, batch.edge_index, batch=batch.batch)
                preds.append(unscale_outputs(out, trainer.target_scaler))
                targets.append(batch.y)
        # val_loss is the mean of the batch losses
        expected = (
            torch.stack([nn.functional.mse_loss(p, t) for p, t in zip(preds, targets, strict=True)])
            .mean()
            .item()
        )
        assert trainer._validate_epoch()["val_loss"] == pytest.approx(expected, rel=1e-5)
        assert trainer.calibrate() == conformal_quantiles(
            torch.cat(targets), torch.cat(preds), alpha=0.1
        )

    def test_scaling_trains_both_targets(self, tmp_path):
        errors = {}
        for scaling in ("standard", None):
            torch.manual_seed(0)
            trainer = _trainer(tmp_path / str(scaling), scaling=scaling, epochs=60)
            trainer.fit(test=False)
            trainer.model.eval()
            batch = next(iter(DataLoader(_graphs(24, 1), batch_size=24)))
            with torch.no_grad():
                out = unscale_outputs(
                    trainer.model(batch.x, batch.edge_index, batch=batch.batch),
                    trainer.target_scaler,
                )
            errors[scaling] = ((out - batch.y).abs().mean(0) / SCALE).tolist()
        # the small target is learned with scaling; unscaled, the energy-like target dominates
        assert errors["standard"][1] < errors[None][1]

    def test_no_scaling_and_classification_record_nothing(self, tmp_path):
        trainer = _trainer(tmp_path, scaling=None, epochs=1)
        trainer.fit(test=False)
        assert trainer.target_scaler is None
        best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
        assert "target_scaler" not in best

    def test_an_unknown_scaling_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="target_scaling must be one of"):
            _trainer(tmp_path, scaling="robust")

    def test_resume_restores_the_scaler(self, tmp_path):
        torch.manual_seed(0)
        trainer = _trainer(tmp_path, epochs=1)
        trainer.fit(test=False)
        resumed = _trainer(tmp_path / "resumed", epochs=1)
        resumed.load_checkpoint(tmp_path / "checkpoints" / "best.pt")
        assert resumed.target_scaler == trainer.target_scaler


class TestPrediction:
    def test_predictor_returns_the_targets_units(self, tmp_path):
        torch.manual_seed(0)
        trainer = _trainer(tmp_path)
        trainer.fit(test=False)
        predictor = Predictor(
            model=trainer.model,
            working_root_dir=tmp_path,
            device=torch.device("cpu"),
            task_type="graph_regression",
            target_scaler=trainer.target_scaler,
        )
        batch = next(iter(DataLoader(_graphs(4, 5), batch_size=4)))
        with torch.no_grad():
            raw = trainer.model(batch.x, batch.edge_index, batch=batch.batch)
        assert torch.allclose(
            predictor.predict(batch), unscale_outputs(raw, trainer.target_scaler), atol=1e-5
        )

    def test_the_package_records_the_scaler_and_serves_in_units(self, tmp_path):
        pytest.importorskip("fastapi", reason="the `serve` extra is not installed")
        from milia_pipeline.models.deployment.package import (
            MODEL_CARD,
            PACKAGE_MANIFEST,
            package_checkpoint,
        )
        from milia_pipeline.models.deployment.serving import PackagedPredictor
        from milia_pipeline.models.factory.model_factory import ModelFactory
        from milia_pipeline.models.post_training import convert_to_pyg
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
                "out_channels": 2,
                "num_layers": 2,
            },
            task_type="graph_regression",
        )
        scaler = {"method": "standard", "n": 40, "mean": [-400.0, 3.0], "std": [50.0, 0.5]}
        checkpoint = CheckpointManager(working_root_dir=tmp_path).save(
            tmp_path / "checkpoints" / "best.pt",
            model,
            target_scaler=scaler,
            hyper_parameters={
                "model_name": "GCN",
                "task_type": "graph_regression",
                "hyperparameters": info["hyperparameters_values"],
                "model_info": info,
            },
        )
        package = package_checkpoint(checkpoint, tmp_path / "package")
        assert json.loads((package / PACKAGE_MANIFEST).read_text())["target_scaler"] == scaler
        assert "trained on targets standardized with the mean and std of 40" in (
            (package / MODEL_CARD).read_text()
        )
        served = PackagedPredictor.from_package(package)
        (graph,) = served.featurize(["CCO"])
        model.eval()
        with torch.no_grad():
            raw = model(graph.x, graph.edge_index, batch=torch.zeros(graph.num_nodes).long())
        assert torch.allclose(
            torch.tensor(served.predict_smiles(["CCO"])),
            unscale_outputs(raw, scaler),
            atol=1e-4,
        )


class TestDeprecationAndConfig:
    @pytest.mark.parametrize("name", ["StandardizeTargets", "NormalizeTargets"])
    def test_per_molecule_target_transforms_are_deprecated(self, name):
        from milia_pipeline.transformations import custom_transforms

        with pytest.warns(FutureWarning, match=f"The {name} transform is deprecated"):
            getattr(custom_transforms, name)()

    def test_the_shipped_configuration(self):
        models = yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]
        assert models["training"]["target_scaling"] == "standard"
        standard = yaml.safe_load((CONFIGS / "transformations.yaml").read_text())[
            "transformations"
        ]["standard_transforms"]
        assert [t["enabled"] for t in standard if t["name"] == "StandardizeTargets"] == [False]

    def test_standard_training_scales_by_default(self, tmp_path):
        import main

        class _Dataset(list):
            structural_features_config = None

        torch.manual_seed(0)
        dataset = _Dataset(_graphs(20, 7))
        config = {
            "global_paths": {"working_root_dir": str(tmp_path)},
            "models": {
                "selection": {
                    "mode": "single",
                    "model_name": "GCN",
                    "task_type": "graph_regression",
                },
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
        assert main._run_standard_training(args, logging.getLogger("test"), dataset, config) == 0
        best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
        assert best["target_scaler"]["method"] == "standard"
        assert len(best["target_scaler"]["mean"]) == 2
