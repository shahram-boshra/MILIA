"""Split conformal prediction intervals (milia_pipeline.models.post_training.conformal)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from checkpoint_records import IDENTIFIER_DATA_INFO
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.post_training.conformal import (
    conformal_quantiles,
    intervals,
    member_spread,
    recording_members,
)
from milia_pipeline.models.training.callbacks import ModelCheckpoint
from milia_pipeline.models.training.trainer import Trainer


class TestQuantiles:
    def test_the_rank_is_ceil_n_plus_one_times_one_minus_alpha(self):
        targets = torch.zeros(9, 1)
        predictions = torch.arange(1.0, 10.0).reshape(9, 1)  # scores 1..9
        calibration = conformal_quantiles(targets, predictions, alpha=0.2)
        # ceil(10 * 0.8) = 8 → the 8th smallest score
        assert calibration == {
            "method": "split_conformal_absolute",
            "alpha": 0.2,
            "n": 9,
            "quantiles": [8.0],
        }

    def test_too_few_samples_give_an_unbounded_interval(self):
        calibration = conformal_quantiles(torch.zeros(3, 2), torch.ones(3, 2), alpha=0.1)
        assert calibration["quantiles"] == [math.inf, math.inf]  # ceil(4 * 0.9) = 4 > 3

    def test_each_target_has_its_quantile(self):
        targets = torch.zeros(19, 2)
        predictions = torch.stack([torch.arange(1.0, 20.0), 10 * torch.arange(1.0, 20.0)], 1)
        q = conformal_quantiles(targets, predictions, alpha=0.1)["quantiles"]
        assert q == [18.0, 180.0]  # ceil(20 * 0.9) = 18

    def test_a_spread_normalizes_the_score(self):
        calibration = conformal_quantiles(
            torch.zeros(9, 1),
            torch.full((9, 1), 4.0),
            alpha=0.2,
            spread=torch.full((9, 1), 2.0),
        )
        assert calibration["method"] == "split_conformal_normalized"
        assert calibration["quantiles"] == [2.0]
        lower, upper = intervals(torch.tensor([[1.0]]), calibration, torch.tensor([[3.0]]))
        assert (lower.item(), upper.item()) == (-5.0, 7.0)

    def test_alpha_must_be_in_the_open_unit_interval(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            conformal_quantiles(torch.zeros(3, 1), torch.zeros(3, 1), alpha=1.0)

    def test_intervals_check_their_inputs(self):
        normalized = {
            "method": "split_conformal_normalized",
            "alpha": 0.1,
            "n": 9,
            "quantiles": [1],
        }
        with pytest.raises(ValueError, match="needs the ensemble's member spread"):
            intervals(torch.zeros(2, 1), normalized)
        with pytest.raises(ValueError, match="has 1 targets; the predictions have 2"):
            intervals(torch.zeros(2, 2), {**normalized, "method": "split_conformal_absolute"})

    def test_coverage_on_exchangeable_data_is_at_least_one_minus_alpha(self):
        generator = torch.Generator().manual_seed(0)
        noise = torch.randn(4000, 1, generator=generator)
        calibration = conformal_quantiles(noise[:1000], torch.zeros(1000, 1), alpha=0.1)
        lower, upper = intervals(torch.zeros(3000, 1), calibration)
        covered = ((noise[1000:] >= lower) & (noise[1000:] <= upper)).float().mean().item()
        assert 0.88 <= covered <= 0.93


def _ensemble(in_channels: int = 4):
    from milia_pipeline.models.builders.model_composer import ParallelEnsemble

    class _Member(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(in_channels, 2)

        def forward(self, x, edge_index=None, batch=None, **kwargs):
            return self.lin(x)

    torch.manual_seed(0)
    return ParallelEnsemble(
        models=[_Member(), _Member(), _Member()],
        weights=[1 / 3] * 3,
        fusion="mean",
        task_type="graph_regression",
    )


def _graphs(n: int, seed: int = 0) -> list[Data]:
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(3, 4, generator=generator)
        out.append(
            Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), y=x.sum(0, keepdim=True)[:, :2])
        )
    return out


class TestEnsembleSpread:
    def test_members_are_recorded_only_inside_the_block(self):
        ensemble = _ensemble().eval()
        batch = next(iter(DataLoader(_graphs(4), batch_size=4)))
        with recording_members(ensemble) as recorded:
            fused = ensemble(batch.x, batch.edge_index, batch=batch.batch)
            spread = member_spread(recorded)
        assert recorded is ensemble and spread.shape == fused.shape and (spread > 0).all()
        assert ensemble.last_member_outputs is None and ensemble.record_members is False
        ensemble(batch.x, batch.edge_index, batch=batch.batch)
        assert ensemble.last_member_outputs is None

    def test_a_single_model_has_no_spread(self):
        with recording_members(nn.Linear(2, 2)) as recorded:
            assert recorded is None and member_spread(recorded) is None


class _Pooled(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x, edge_index, batch=None):
        return self.lin(global_mean_pool(x, batch))


def _trainer(tmp_path, model, alpha=0.1):
    return Trainer(
        model=model,
        train_loader=DataLoader(_graphs(16, 1), batch_size=4),
        val_loader=DataLoader(_graphs(19, 2), batch_size=4),
        test_loader=DataLoader(_graphs(8, 3), batch_size=4),
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        callbacks=[ModelCheckpoint(dirpath=tmp_path / "checkpoints", verbose=False)],
        model_info={"task_type": "graph_regression"},
        max_epochs=2,
        calibration_alpha=alpha,
    )


class TestTrainerCalibration:
    def test_fit_stores_the_calibration_of_the_best_checkpoint(self, tmp_path):
        torch.manual_seed(0)
        trainer = _trainer(tmp_path, _Pooled())
        results = trainer.fit()
        best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
        assert best["calibration"] == results["calibration"]
        assert best["calibration"]["method"] == "split_conformal_absolute"
        assert best["calibration"]["n"] == 19 and len(best["calibration"]["quantiles"]) == 2
        assert trainer.calibrate(ckpt_path="best") == best["calibration"]

    def test_an_ensemble_is_calibrated_with_the_normalized_score(self, tmp_path):
        trainer = _trainer(tmp_path, _ensemble())
        results = trainer.fit()
        assert results["calibration"]["method"] == "split_conformal_normalized"

    def test_alpha_none_disables_calibration(self, tmp_path):
        results = _trainer(tmp_path, _Pooled(), alpha=None).fit()
        assert results["calibration"] is None
        best = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
        assert "calibration" not in best


class TestPackageAndPredictor:
    def _calibrated_package(self, tmp_path):
        from milia_pipeline.models.deployment.package import package_checkpoint
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
        calibration = {
            "method": "split_conformal_absolute",
            "alpha": 0.1,
            "n": 50,
            "quantiles": [0.5, 2.0],
        }
        checkpoint = CheckpointManager(working_root_dir=tmp_path).save(
            tmp_path / "checkpoints" / "best.pt",
            model,
            calibration=calibration,
            hyper_parameters={
                "model_name": "GCN",
                "task_type": "graph_regression",
                "hyperparameters": info["hyperparameters_values"],
                "model_info": info,
            },
            data_info=IDENTIFIER_DATA_INFO,
        )
        return package_checkpoint(checkpoint, tmp_path / "package"), calibration

    def test_the_package_records_and_describes_the_calibration(self, tmp_path):
        from milia_pipeline.models.deployment.package import MODEL_CARD, PACKAGE_MANIFEST

        package, calibration = self._calibrated_package(tmp_path)
        assert json.loads((package / PACKAGE_MANIFEST).read_text())["calibration"] == calibration
        card = (package / MODEL_CARD).read_text()
        assert "Split conformal intervals at 90% (alpha 0.1), fitted on 50" in card

    def test_predictor_and_server_return_the_intervals(self, tmp_path):
        pytest.importorskip("fastapi", reason="the `serve` extra is not installed")
        from milia_pipeline.models.deployment.serving import PackagedPredictor

        package, _ = self._calibrated_package(tmp_path)
        served = PackagedPredictor.from_package(package)
        rows, served_intervals = served.predict_smiles_with_intervals(["CCO", "c1ccccc1"])
        mean = np.asarray(rows)
        lower = np.asarray(served_intervals["lower"])
        upper = np.asarray(served_intervals["upper"])
        np.testing.assert_allclose(mean - lower, [[0.5, 2.0]] * 2, rtol=1e-5)
        np.testing.assert_allclose(upper - mean, [[0.5, 2.0]] * 2, rtol=1e-5)
        assert served_intervals["alpha"] == 0.1
