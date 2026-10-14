#!/usr/bin/env python3
"""
Contract tests for ``pruner.type: wilcoxon`` — Optuna's ``WilcoxonPruner`` over cross-validation folds.

The pruner runs a paired one-sided Wilcoxon signed-rank test between the current trial's per-instance
values and the best trial's, the instance id being the reported step. MILIA's instances are the CV
folds (deterministic, ``random_seed=42``), reported at ``step=fold_idx``. Following the pruner's
Optuna documentation, the folds are evaluated in an order shuffled per trial, and a prune decision
ends the folds and returns their mean instead of raising ``TrialPruned``, so the sampler learns from
the trial (optuna 4.9.0 ``optuna/pruners/_wilcoxon.py``).
"""

import random
import warnings
from statistics import mean
from unittest.mock import MagicMock, patch

import optuna
import pytest
import torch.nn as nn
from optuna.trial import TrialState
from scipy import stats

from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend
from milia_pipeline.models.hpo.hpo_config import (
    HPOConfig,
    PrunerConfig,
    PrunerType,
    wilcoxon_earliest_prune_fold,
)

N_FOLDS = 6
FOLD_BASE = [0.1 * (k + 1) for k in range(N_FOLDS)]  # heterogeneous folds


def _run_cv(trial, offset, n_folds=N_FOLDS, instance_pruning=True, order=None):
    """Run ``_run_cross_validation`` on ``n_folds`` stub folds; fold k scores
    ``FOLD_BASE[k] + offset`` (val_loss). Appends the evaluated fold indices to ``order``."""
    from milia_pipeline.models.hpo.hpo_manager import _run_cross_validation

    folds = [([("fold", k)], [("fold", k)]) for k in range(n_folds)]
    splitter = MagicMock()
    splitter.k_fold_split.return_value = folds
    factory = MagicMock()
    factory.create_model_with_info.side_effect = lambda **kw: (nn.Linear(2, 1), {})

    def make_trainer(**kwargs):
        fold = kwargs["train_loader"][0][1]
        if order is not None:
            order.append(fold)
        trainer = MagicMock()
        value = FOLD_BASE[fold] + offset
        trainer.fit.return_value = {
            "train_metrics": {"val_loss": [value]},
            "test_metrics": {},
            "training_time": 1.0,
            "best_epoch": 0,
            "best_val_loss": value,
        }
        return trainer

    with (
        patch("milia_pipeline.models.hpo.hpo_manager.DataSplitter", splitter),
        patch("milia_pipeline.models.hpo.hpo_manager.Trainer", side_effect=make_trainer),
        patch(
            "milia_pipeline.models.training.loaders.make_loader",
            side_effect=lambda subset, **kw: subset,
        ),
    ):
        return _run_cross_validation(
            model_name="GCN",
            dataset=MagicMock(),
            model_params={},
            optimizer_params={},
            scheduler_params={},
            loss_params={},
            trainer_kwargs={},
            callbacks=[],
            n_folds=n_folds,
            metric="val_loss",
            aggregation="mean",
            factory=factory,
            task_type="graph_regression",
            trial=trial,
            metrics=("val_loss",),
            instance_pruning=instance_pruning,
        )


def _study():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        pruner = OptunaBackend().create_pruner("wilcoxon")
    return optuna.create_study(direction="minimize", pruner=pruner)


@pytest.mark.contract
class TestConfig:
    def test_accepted_with_cross_validation(self):
        config = HPOConfig.from_dict({"pruner": {"type": "wilcoxon"}, "cv_folds": 5})
        assert config.pruner.type is PrunerType.WILCOXON
        assert (config.pruner.p_threshold, config.pruner.n_startup_steps) == (0.1, 2)

    @pytest.mark.parametrize(
        ("hpo", "message"),
        [
            ({"cv_folds": 0}, "needs cross-validation"),
            ({"cv_folds": 4}, "only after 4 folds"),
            ({"cv_folds": 5, "cv_metric_aggregation": "median"}, "cv_metric_aggregation 'mean'"),
            ({"cv_folds": 5, "pruner": {"n_startup_steps": 5}}, "only after 5 folds"),
            ({"cv_folds": 5, "pruner": {"p_threshold": 0.0}}, "p_threshold must be in (0, 1]"),
            ({"cv_folds": 5, "pruner": {"p_threshold": 1.5}}, "p_threshold must be in (0, 1]"),
            ({"cv_folds": 5, "pruner": {"n_startup_steps": -1}}, "n_startup_steps must be"),
        ],
    )
    def test_rejected(self, hpo, message):
        pruner = {"type": "wilcoxon", **hpo.pop("pruner", {})}
        with pytest.raises(ValueError, match=message.replace("(", r"\(").replace("]", r"\]")):
            HPOConfig.from_dict({"pruner": pruner, **hpo})

    @pytest.mark.parametrize("p_threshold", [0.0, 1.5])
    def test_pruner_config_rejects_p_threshold_out_of_range(self, p_threshold):
        with pytest.raises(ValueError, match=r"p_threshold must be in \(0, 1\]"):
            PrunerConfig(type=PrunerType.WILCOXON, p_threshold=p_threshold)

    def test_larger_p_threshold_allows_fewer_folds(self):
        config = HPOConfig.from_dict(
            {"pruner": {"type": "wilcoxon", "p_threshold": 0.2}, "cv_folds": 4}
        )
        assert config.cv_folds == 4

    @pytest.mark.parametrize(
        ("p_threshold", "n_startup_steps", "expected"),
        [(0.1, 2, 4), (0.125, 2, 4), (0.2, 2, 3), (1.0, 0, 2), (0.1, 6, 6), (0.05, 2, 5)],
    )
    def test_earliest_prune_fold(self, p_threshold, n_startup_steps, expected):
        assert wilcoxon_earliest_prune_fold(p_threshold, n_startup_steps) == expected

    @pytest.mark.parametrize("p_threshold", [0.0, -0.1, 1.5])
    def test_earliest_prune_fold_rejects_p_threshold_out_of_range(self, p_threshold):
        with pytest.raises(ValueError, match=r"p_threshold must be in \(0, 1\]"):
            wilcoxon_earliest_prune_fold(p_threshold, 2)

    @pytest.mark.parametrize("p_threshold", [0.3, 0.2, 0.1, 0.05, 0.02])
    def test_earliest_prune_fold_matches_scipy(self, p_threshold):
        """The bound is the smallest fold count whose most extreme exact p-value (every fold worse)
        falls below p_threshold, as scipy computes it for the pruner."""
        earliest = wilcoxon_earliest_prune_fold(p_threshold, 0)

        def min_p(n):
            diffs = [0.1 * (i + 1) for i in range(n)]
            return stats.wilcoxon(diffs, alternative="greater", zero_method="zsplit").pvalue

        assert min_p(earliest) < p_threshold
        assert earliest == 2 or min_p(earliest - 1) >= p_threshold


@pytest.mark.contract
class TestBackend:
    def test_creates_wilcoxon_pruner_quietly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", optuna.exceptions.ExperimentalWarning)
            pruner = OptunaBackend().create_pruner(
                "wilcoxon", n_warmup_steps=10, p_threshold=0.05, n_startup_steps=3
            )
        assert isinstance(pruner, optuna.pruners.WilcoxonPruner)
        assert (pruner._p_threshold, pruner._n_startup_steps) == (0.05, 3)


@pytest.mark.contract
class TestFolds:
    def test_folds_run_in_a_per_trial_shuffled_order_at_fold_steps(self):
        study = optuna.create_study(pruner=optuna.pruners.NopPruner())
        for _ in range(3):
            trial = study.ask()
            order: list[int] = []
            _run_cv(trial, 0.0, order=order)
            expected = list(range(N_FOLDS))
            random.Random(trial.number).shuffle(expected)
            assert order == expected
            study.tell(trial, mean(FOLD_BASE))
        frozen = study.trials[0]
        assert sorted(frozen.intermediate_values) == list(range(N_FOLDS))
        assert frozen.intermediate_values == {k: FOLD_BASE[k] for k in range(N_FOLDS)}
        assert frozen.user_attrs["cv_folds_evaluated"] == N_FOLDS

    def test_other_pruners_keep_the_fold_order(self):
        study = optuna.create_study(pruner=optuna.pruners.NopPruner())
        trial = study.ask()
        trial = study.ask()  # number 1: its shuffle is not the identity
        order: list[int] = []
        _run_cv(trial, 0.0, instance_pruning=False, order=order)
        assert order == list(range(N_FOLDS))
        assert "cv_folds_evaluated" not in trial.user_attrs

    def test_worse_trial_stops_early_and_returns_its_fold_mean(self):
        study = _study()
        study.optimize(lambda t: _run_cv(t, 0.0)[0], n_trials=1)
        order: list[int] = []
        study.optimize(lambda t: _run_cv(t, 1.0, order=order)[0], n_trials=1)

        best, worse = study.trials
        earliest = wilcoxon_earliest_prune_fold(0.1, 2)
        assert worse.state is TrialState.COMPLETE  # value returned, not TrialPruned
        assert worse.user_attrs["cv_folds_evaluated"] == earliest < N_FOLDS
        assert len(order) == earliest
        assert worse.value == pytest.approx(mean(FOLD_BASE[k] + 1.0 for k in order))
        assert worse.value > best.value
        assert study.best_trial.number == best.number
        assert best.user_attrs["cv_folds_evaluated"] == N_FOLDS

    def test_better_trial_runs_every_fold(self):
        study = _study()
        study.optimize(lambda t: _run_cv(t, 0.0)[0], n_trials=1)
        study.optimize(lambda t: _run_cv(t, -0.05)[0], n_trials=1)
        better = study.trials[1]
        assert better.user_attrs["cv_folds_evaluated"] == N_FOLDS
        assert study.best_trial.number == better.number
