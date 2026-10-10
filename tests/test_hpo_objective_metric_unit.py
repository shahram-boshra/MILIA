#!/usr/bin/env python3
"""
Contract tests for the objective value of a trial (F53).

``_run_metric_value(results, metric)`` reads the configured metric from a ``Trainer.fit()`` result at
the epoch the trainer keeps (``best_epoch``, lowest validation loss). Previously any metric other than
a top-level key fell back to ``best_val_loss``, so a study configured for ``val_mae`` optimized the
validation loss.
"""

import logging

import pytest
import torch
from torch import nn

from milia_pipeline.exceptions import HPOError
from milia_pipeline.models.hpo.hpo_manager import _run_metric_value

# Three epochs: val_loss is lowest at epoch 1, val_mae lowest at epoch 2, last epoch is 2
RESULTS = {
    "train_metrics": {
        "train_loss": [0.9, 0.6, 0.4],
        "val_loss": [0.5, 0.2, 0.3],
        "val_mae": [0.9, 0.4, 0.1],
    },
    "test_metrics": {},
    "training_time": 12.5,
    "best_epoch": 1,
    "best_val_loss": 0.2,
}


@pytest.mark.contract
class TestRunMetricValue:
    def test_val_loss_equals_best_val_loss(self):
        assert _run_metric_value(RESULTS, "val_loss") == RESULTS["best_val_loss"]

    def test_other_metric_taken_at_kept_epoch(self):
        # not min(val_mae)=0.1, not last=0.1, not best_val_loss=0.2: the kept model's val_mae
        assert _run_metric_value(RESULTS, "val_mae") == 0.4

    def test_training_time_is_run_level(self):
        assert _run_metric_value(RESULTS, "training_time") == 12.5

    def test_unknown_metric_raises_and_lists_available(self):
        with pytest.raises(HPOError, match="Metric 'val_accuracy' is not produced") as excinfo:
            _run_metric_value(RESULTS, "val_accuracy")
        assert "train_loss, val_loss, val_mae, training_time" in str(excinfo.value)

    def test_best_val_loss_is_not_a_fallback(self):
        results = {**RESULTS, "train_metrics": {"train_loss": [0.1]}, "best_epoch": None}
        with pytest.raises(HPOError, match="is not produced"):
            _run_metric_value(results, "val_mae")

    @pytest.mark.parametrize("best_epoch", [None, 3, -1])
    def test_no_value_at_kept_epoch_raises(self, best_epoch):
        with pytest.raises(HPOError, match="has no value at the kept epoch"):
            _run_metric_value({**RESULTS, "best_epoch": best_epoch}, "val_mae")


@pytest.mark.contract
def test_real_trainer_result():
    """A real ``Trainer.fit()`` run: the value is read from its own result structure."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool

    from milia_pipeline.models.training.trainer import Trainer

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 1)

        def forward(self, x, edge_index, batch):
            return global_mean_pool(self.linear(x), batch)

    torch.manual_seed(0)
    graphs = [
        Data(x=torch.randn(4, 3), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.randn(1, 1))
        for _ in range(16)
    ]
    model = Model()
    logging.disable(logging.INFO)
    try:
        results = Trainer(
            model=model,
            train_loader=DataLoader(graphs[:12], batch_size=4),
            val_loader=DataLoader(graphs[12:], batch_size=4),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
            loss_fn=nn.MSELoss(),
            max_epochs=4,
        ).fit()
    finally:
        logging.disable(logging.NOTSET)
    val_losses = results["train_metrics"]["val_loss"]
    assert results["best_epoch"] == min(range(len(val_losses)), key=val_losses.__getitem__)
    assert _run_metric_value(results, "val_loss") == pytest.approx(results["best_val_loss"])
    with pytest.raises(HPOError, match="Available metrics: train_loss, val_loss, training_time"):
        _run_metric_value(results, "val_mae_typo")


@pytest.mark.contract
@pytest.mark.parametrize(("metric", "expected"), [("val_mae", 0.4), ("val_loss", 0.2)])
def test_objective_returns_configured_metric(metric, expected):
    """The trial objective (no CV) returns the configured metric at the kept epoch."""
    from unittest.mock import MagicMock, patch

    from milia_pipeline.models.hpo.hpo_config import HPOConfig, StudyConfig
    from milia_pipeline.models.hpo.hpo_manager import HPOManager

    module = "milia_pipeline.models.hpo.hpo_manager"
    trainer = MagicMock()
    trainer.fit.return_value = RESULTS
    factory = MagicMock()
    factory.create_model_with_info.return_value = (nn.Linear(2, 1), {})
    config = HPOConfig(enabled=True, study=StudyConfig(metric=metric))
    with (
        patch(f"{module}.get_backend", return_value=MagicMock()),
        patch(f"{module}.get_factory", return_value=factory),
        patch(f"{module}.infer_task_type", return_value="graph_regression"),
        patch(f"{module}._TARGET_SELECTION_AVAILABLE", False),
        patch(f"{module}.create_hpo_callback", return_value=MagicMock()),
        patch(f"{module}.DataSplitter") as splitter,
        patch(
            f"{module}._prepare_data_for_task_hpo",
            return_value=([MagicMock()], [MagicMock()], None),
        ),
        patch("milia_pipeline.models.training.loaders.make_loader", return_value=MagicMock()),
        patch(
            "milia_pipeline.models.acceleration.config_builder.build_acceleration",
            return_value=None,
        ),
        patch(f"{module}.Trainer", return_value=trainer),
    ):
        splitter.random_split.return_value = ([MagicMock()], [MagicMock()], [])
        manager = HPOManager(config)
        manager._model_factory = factory
        with patch.object(HPOManager, "_build_composite_hyperparameters", return_value={}):
            objective = manager._create_objective(
                model_name="GCN",
                dataset=[MagicMock()],
                base_hyperparameters={},
                trainer_kwargs={"max_epochs": 1},
                additional_callbacks=[],
            )
            trial = MagicMock(number=0)
            manager.backend.suggest_params.return_value = {}
            assert objective(trial) == expected
