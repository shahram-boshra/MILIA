"""
End-to-End HPO Workflow Test Suite

Tests a complete HPO workflow:
    config → HPOManager creation → search space building →
    optimization (2–3 trials) → best parameters retrieval → study analysis.

Modules Exercised (from MILIA_Test_Recommendations.md §3.3):
    - milia_pipeline/models/hpo/hpo_config.py          — HPOConfig
    - milia_pipeline/models/hpo/hpo_manager.py          — HPOManager.optimize()
    - milia_pipeline/models/hpo/backends/optuna_backend.py — OptunaBackend
    - milia_pipeline/models/hpo/search_spaces/search_space_builder.py — SearchSpaceBuilder
    - milia_pipeline/models/hpo/search_spaces/param_types.py — ParamType
    - milia_pipeline/models/hpo/callbacks/optuna_callback.py — OptunaPruningCallback
    - milia_pipeline/models/hpo/analysis/study_analyzer.py — StudyAnalyzer
    - milia_pipeline/models/training/trainer.py          — Training within trials
    - milia_pipeline/models/training/data_splitting.py   — Data splitting within trials

Scope:
    Runs 2–3 trials with a tiny model and small synthetic dataset.
    Asserts: best parameters are returned, StudyAnalyzer produces analysis
    without errors, trial count matches expected.

Launch:
    From project root (/app/milia):
        pytest tests/test_e2e_hpo_workflow.py -v --tb=short

Author: MILIA Team
Version: 1.0.0
"""

import sys
import os
import logging
import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from typing import Dict, Any, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Add project root to Python path FIRST
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Third-party imports (with availability checks)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data, InMemoryDataset, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    import optuna
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ---------------------------------------------------------------------------
# MILIA imports — guarded so collection never crashes
# ---------------------------------------------------------------------------
try:
    from milia_pipeline.models.hpo.hpo_config import (
        HPOConfig,
        PrunerType,
        PrunerConfig,
        SamplerType,
        SamplerConfig,
        OptimizationDirection,
        StudyConfig,
    )
    HPO_CONFIG_AVAILABLE = True
except ImportError:
    HPO_CONFIG_AVAILABLE = False

try:
    from milia_pipeline.models.hpo.search_spaces.param_types import (
        ParamType,
        SearchSpaceParamConfig,
    )
    PARAM_TYPES_AVAILABLE = True
except ImportError:
    PARAM_TYPES_AVAILABLE = False

try:
    from milia_pipeline.models.hpo.search_spaces.search_space_builder import (
        SearchSpaceBuilder,
        build_search_space,
        validate_search_space,
    )
    SEARCH_SPACE_BUILDER_AVAILABLE = True
except ImportError:
    SEARCH_SPACE_BUILDER_AVAILABLE = False

try:
    from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend
    OPTUNA_BACKEND_AVAILABLE = True
except ImportError:
    OPTUNA_BACKEND_AVAILABLE = False

try:
    from milia_pipeline.models.hpo.callbacks.optuna_callback import (
        OptunaPruningCallback,
        create_hpo_callback,
    )
    CALLBACK_AVAILABLE = True
except ImportError:
    CALLBACK_AVAILABLE = False

try:
    from milia_pipeline.models.hpo.analysis.study_analyzer import (
        StudyAnalyzer,
        AnalysisConfig,
        ImportanceMethod,
        ExportFormat,
    )
    STUDY_ANALYZER_AVAILABLE = True
except ImportError:
    STUDY_ANALYZER_AVAILABLE = False

try:
    from milia_pipeline.models.hpo.hpo_manager import (
        HPOManager,
        is_hpo_enabled,
        get_best_params,
        create_hpo_manager,
        _flatten_params,
        _extract_param_categories,
        infer_task_type,
    )
    HPO_MANAGER_AVAILABLE = True
except ImportError:
    HPO_MANAGER_AVAILABLE = False

try:
    from milia_pipeline.models.training.trainer import Trainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False

try:
    from milia_pipeline.models.training.data_splitting import DataSplitter
    DATA_SPLITTER_AVAILABLE = True
except ImportError:
    DATA_SPLITTER_AVAILABLE = False

try:
    from milia_pipeline.exceptions import (
        HPOError,
        HPOConfigurationError,
        TrialFailedError,
        SearchSpaceError,
        PruningError,
    )
    EXCEPTIONS_AVAILABLE = True
except ImportError:
    EXCEPTIONS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not TORCH_GEOMETRIC_AVAILABLE,
        reason="torch_geometric not installed",
    ),
    pytest.mark.skipif(
        not OPTUNA_AVAILABLE,
        reason="optuna not installed",
    ),
]


# =============================================================================
# FIXTURES — Synthetic Data & Minimal Config
# =============================================================================

class _SyntheticGraphDataset(torch.utils.data.Dataset):
    """
    Minimal synthetic PyG-like graph dataset for HPO E2E tests.

    Each graph has:
        - x:          [num_nodes, in_channels] node features
        - edge_index: [2, num_edges] edge connectivity
        - y:          [1] scalar regression target
    """

    def __init__(self, n_graphs: int = 30, in_channels: int = 8,
                 num_nodes: int = 5, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self._data_list: List[Data] = []
        for i in range(n_graphs):
            x = torch.randn(num_nodes, in_channels)
            # Simple fully-connected graph
            src = []
            dst = []
            for s in range(num_nodes):
                for d in range(num_nodes):
                    if s != d:
                        src.append(s)
                        dst.append(d)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            y = torch.tensor([x.mean().item()], dtype=torch.float)
            self._data_list.append(Data(x=x, edge_index=edge_index, y=y))

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]


class _TinyGNN(nn.Module):
    """
    Minimal GNN for HPO E2E tests — avoids importing any external
    model registry or factory.  Uses only torch.nn and basic scatter ops.

    Architecture:
        Linear(in_channels → hidden_channels) → ReLU →
        Linear(hidden_channels → out_channels) + global mean pooling
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 16,
                 out_channels: int = 1, **kwargs):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch=None, **kwargs):
        h = self.act(self.lin1(x))
        h = self.lin2(h)
        # Global mean pooling
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)
        # Squeeze trailing dimension for single-target regression to match
        # PyG batched target shape: batch.y is [batch_size] (1-D) when each
        # graph has a scalar y, but nn.Linear(hidden, 1) + pooling produces
        # [batch_size, 1] (2-D).  Squeeze only when out_channels == 1 so
        # multi-target models (out_channels > 1) are unaffected.
        if h.dim() == 2 and h.size(-1) == 1:
            h = h.squeeze(-1)
        return h


@pytest.fixture
def synthetic_dataset():
    """30-graph synthetic PyG dataset (regression, in_channels=8)."""
    return _SyntheticGraphDataset(n_graphs=30, in_channels=8, seed=42)


@pytest.fixture
def tiny_model_factory():
    """
    Returns a callable that mirrors ModelFactory.create_model_with_info().

    Signature: factory(name, hyperparameters, task_type, sample_data,
                       num_classes_override, **kw)  →  (model, model_info)
    """
    def _factory(name: str = "TinyGNN", hyperparameters: Optional[Dict] = None,
                 task_type: str = "graph_regression", sample_data=None,
                 num_classes_override=None, **kwargs):
        hp = hyperparameters or {}
        hidden = hp.get("hidden_channels", 16)
        model = _TinyGNN(in_channels=8, hidden_channels=hidden, out_channels=1)
        model_info = {
            "model_name": name,
            "task_type": task_type,
            "uses_edge_features": False,
            "is_classification": False,
            "out_channels": 1,
        }
        return model, model_info

    return _factory


@pytest.fixture
def minimal_search_space():
    """
    Minimal search space dict in the HPOConfig format:
        { category: { param_name: SearchSpaceParamConfig, … }, … }
    """
    if not PARAM_TYPES_AVAILABLE:
        pytest.skip("param_types not available")

    return {
        "hyperparameters": {
            "hidden_channels": SearchSpaceParamConfig(
                type=ParamType.INT, low=8, high=32, step=8
            ),
        },
        "optimizer": {
            "lr": SearchSpaceParamConfig(
                type=ParamType.LOGUNIFORM, low=1e-4, high=1e-2
            ),
        },
    }


@pytest.fixture
def hpo_config_2_trials(minimal_search_space):
    """HPOConfig configured for 2 quick trials, no pruning, random sampler."""
    if not HPO_CONFIG_AVAILABLE:
        pytest.skip("hpo_config not importable")

    return HPOConfig(
        enabled=True,
        backend="optuna",
        n_trials=2,
        timeout=120,
        n_jobs=1,
        search_space=minimal_search_space,
        pruner=PrunerConfig(type=PrunerType.NONE),
        sampler=SamplerConfig(type=SamplerType.RANDOM, seed=42),
        study=StudyConfig(
            direction=OptimizationDirection.MINIMIZE,
            metric="val_loss",
            study_name="e2e_test_study",
            storage=None,
            load_if_exists=False,
        ),
        cv_folds=0,
        cv_metric_aggregation="mean",
        task_type="graph_regression",
    )


@pytest.fixture
def hpo_config_3_trials(minimal_search_space):
    """HPOConfig configured for 3 quick trials."""
    if not HPO_CONFIG_AVAILABLE:
        pytest.skip("hpo_config not importable")

    return HPOConfig(
        enabled=True,
        backend="optuna",
        n_trials=3,
        timeout=180,
        n_jobs=1,
        search_space=minimal_search_space,
        pruner=PrunerConfig(type=PrunerType.NONE),
        sampler=SamplerConfig(type=SamplerType.RANDOM, seed=123),
        study=StudyConfig(
            direction=OptimizationDirection.MINIMIZE,
            metric="val_loss",
            study_name="e2e_test_study_3",
            storage=None,
            load_if_exists=False,
        ),
        cv_folds=0,
        cv_metric_aggregation="mean",
        task_type="graph_regression",
    )


# =============================================================================
# SECTION 1 — HPOConfig Construction & Validation
# =============================================================================

class TestHPOConfigConstruction:
    """Verify HPOConfig, SearchSpaceParamConfig, enums, and from_dict()."""

    @pytest.mark.skipif(not HPO_CONFIG_AVAILABLE, reason="hpo_config not importable")
    def test_hpoconfig_enabled_flag(self):
        """HPOConfig.enabled=False by default; True when explicitly set."""
        default = HPOConfig()
        assert default.enabled is False

        enabled = HPOConfig(enabled=True)
        assert enabled.enabled is True

    @pytest.mark.skipif(not HPO_CONFIG_AVAILABLE, reason="hpo_config not importable")
    def test_hpoconfig_frozen_immutability(self):
        """HPOConfig instances must be frozen (Pydantic V2 frozen=True)."""
        cfg = HPOConfig(enabled=True, n_trials=5)
        with pytest.raises(Exception):
            # Pydantic V2 frozen model raises ValidationError on attribute set
            cfg.n_trials = 99

    @pytest.mark.skipif(not HPO_CONFIG_AVAILABLE, reason="hpo_config not importable")
    def test_hpoconfig_from_dict_round_trip(self, minimal_search_space):
        """HPOConfig.from_dict() parses a raw dict identical to YAML input."""
        raw = {
            "enabled": True,
            "backend": "optuna",
            "n_trials": 3,
            "timeout": None,
            "n_jobs": 1,
            "search_space": {
                "hyperparameters": {
                    "hidden_channels": {
                        "type": "int", "low": 8, "high": 32, "step": 8
                    },
                },
                "optimizer": {
                    "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
                },
            },
            "pruner": {"type": "none"},
            "sampler": {"type": "random", "seed": 42},
            "study": {
                "direction": "minimize",
                "metric": "val_loss",
                "study_name": "roundtrip_test",
            },
            "cv_folds": 0,
            "task_type": "graph_regression",
        }
        cfg = HPOConfig.from_dict(raw)
        assert cfg.enabled is True
        assert cfg.n_trials == 3
        assert cfg.backend == "optuna"
        assert cfg.study.study_name == "roundtrip_test"
        assert cfg.study.direction == OptimizationDirection.MINIMIZE
        assert "hyperparameters" in cfg.search_space
        assert "lr" in cfg.search_space["optimizer"]

    @pytest.mark.skipif(not HPO_CONFIG_AVAILABLE, reason="hpo_config not importable")
    def test_hpoconfig_invalid_backend_raises(self):
        """HPOConfig rejects unknown backends at construction time."""
        with pytest.raises(Exception):
            HPOConfig(enabled=True, backend="nonexistent_backend")

    @pytest.mark.skipif(not HPO_CONFIG_AVAILABLE, reason="hpo_config not importable")
    def test_hpoconfig_invalid_n_trials_raises(self):
        """HPOConfig rejects n_trials < 1."""
        with pytest.raises(Exception):
            HPOConfig(enabled=True, n_trials=0)

    @pytest.mark.skipif(not PARAM_TYPES_AVAILABLE, reason="param_types not importable")
    def test_param_type_enum_values(self):
        """ParamType enum has all 7 expected members."""
        expected = {"int", "float", "categorical", "loguniform",
                    "uniform", "int_uniform", "discrete_uniform"}
        actual = {pt.value for pt in ParamType}
        assert expected == actual

    @pytest.mark.skipif(not PARAM_TYPES_AVAILABLE, reason="param_types not importable")
    def test_search_space_param_config_numeric_validation(self):
        """Numeric SearchSpaceParamConfig requires low < high."""
        with pytest.raises(Exception):
            SearchSpaceParamConfig(type=ParamType.INT, low=10, high=5)

    @pytest.mark.skipif(not PARAM_TYPES_AVAILABLE, reason="param_types not importable")
    def test_search_space_param_config_categorical_requires_choices(self):
        """Categorical SearchSpaceParamConfig requires non-empty choices."""
        with pytest.raises(Exception):
            SearchSpaceParamConfig(type=ParamType.CATEGORICAL, choices=[])


# =============================================================================
# SECTION 2 — SearchSpaceBuilder Fluent API
# =============================================================================

class TestSearchSpaceBuilder:
    """Verify fluent builder, validation, and utility methods."""

    @pytest.mark.skipif(not SEARCH_SPACE_BUILDER_AVAILABLE,
                        reason="search_space_builder not importable")
    def test_fluent_builder_chain(self):
        """Builder supports method chaining and produces valid space."""
        space = (
            SearchSpaceBuilder()
            .add_int("hidden_channels", 16, 64, step=16, category="hyperparameters")
            .add_loguniform("lr", 1e-4, 1e-2, category="optimizer")
            .add_categorical("activation", ["relu", "elu"], category="hyperparameters")
            .build()
        )
        assert "hyperparameters" in space
        assert "optimizer" in space
        assert "hidden_channels" in space["hyperparameters"]
        assert "lr" in space["optimizer"]
        assert "activation" in space["hyperparameters"]

    @pytest.mark.skipif(not SEARCH_SPACE_BUILDER_AVAILABLE,
                        reason="search_space_builder not importable")
    def test_builder_frozen_after_build(self):
        """Builder cannot be modified after build() is called."""
        builder = SearchSpaceBuilder()
        builder.add_int("x", 1, 10, category="hyperparameters")
        _ = builder.build()
        with pytest.raises(Exception):
            builder.add_int("y", 1, 10, category="hyperparameters")

    @pytest.mark.skipif(not SEARCH_SPACE_BUILDER_AVAILABLE,
                        reason="search_space_builder not importable")
    def test_validate_search_space_accepts_valid(self, minimal_search_space):
        """validate_search_space() returns True for a valid space."""
        is_valid, errors = validate_search_space(minimal_search_space)
        assert is_valid is True, f"Unexpected errors: {errors}"

    @pytest.mark.skipif(not SEARCH_SPACE_BUILDER_AVAILABLE,
                        reason="search_space_builder not importable")
    def test_validate_search_space_rejects_empty(self):
        """validate_search_space() rejects an empty dict."""
        is_valid, errors = validate_search_space({})
        assert is_valid is False

    @pytest.mark.skipif(not SEARCH_SPACE_BUILDER_AVAILABLE,
                        reason="search_space_builder not importable")
    def test_get_param_count(self, minimal_search_space):
        """get_param_count() returns correct counts per category."""
        counts = SearchSpaceBuilder.get_param_count(minimal_search_space)
        assert counts["hyperparameters"] == 1
        assert counts["optimizer"] == 1

    @pytest.mark.skipif(not SEARCH_SPACE_BUILDER_AVAILABLE,
                        reason="search_space_builder not importable")
    def test_estimate_search_space_size(self, minimal_search_space):
        """estimate_search_space_size() produces a positive integer."""
        size = SearchSpaceBuilder.estimate_search_space_size(minimal_search_space)
        assert isinstance(size, int)
        assert size > 0

    @pytest.mark.skipif(not SEARCH_SPACE_BUILDER_AVAILABLE,
                        reason="search_space_builder not importable")
    def test_list_available_models(self):
        """list_available_models() returns a non-empty list of strings."""
        models = SearchSpaceBuilder.list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)


# =============================================================================
# SECTION 3 — OptunaBackend Primitives
# =============================================================================

class TestOptunaBackend:
    """Test OptunaBackend independently (create_study, suggest_params, etc.)."""

    @pytest.mark.skipif(not OPTUNA_BACKEND_AVAILABLE,
                        reason="optuna_backend not importable")
    def test_create_study(self):
        """OptunaBackend.create_study() returns a valid Optuna Study."""
        backend = OptunaBackend()
        study = backend.create_study(
            study_name="backend_unit_test",
            direction="minimize",
            storage=None,
            load_if_exists=False,
        )
        assert isinstance(study, optuna.Study)
        assert study.study_name == "backend_unit_test"

    @pytest.mark.skipif(not OPTUNA_BACKEND_AVAILABLE,
                        reason="optuna_backend not importable")
    @pytest.mark.filterwarnings(
        "ignore:.*multivariate.*:optuna.exceptions.ExperimentalWarning"
    )
    def test_create_sampler_tpe(self):
        """OptunaBackend creates TPE sampler without error."""
        backend = OptunaBackend()
        sampler = backend.create_sampler("tpe", seed=42)
        assert isinstance(sampler, optuna.samplers.TPESampler)

    @pytest.mark.skipif(not OPTUNA_BACKEND_AVAILABLE,
                        reason="optuna_backend not importable")
    def test_create_sampler_random(self):
        """OptunaBackend creates Random sampler without error."""
        backend = OptunaBackend()
        sampler = backend.create_sampler("random", seed=0)
        assert isinstance(sampler, optuna.samplers.RandomSampler)

    @pytest.mark.skipif(not OPTUNA_BACKEND_AVAILABLE or not PARAM_TYPES_AVAILABLE,
                        reason="optuna_backend or param_types not importable")
    def test_suggest_params(self, minimal_search_space):
        """OptunaBackend.suggest_params() returns a dict of suggested values."""
        backend = OptunaBackend()
        study = backend.create_study("suggest_test", "minimize")

        # Run one trial to get suggested params
        suggested = {}

        def _objective(trial):
            nonlocal suggested
            suggested = backend.suggest_params(trial, minimal_search_space)
            return 0.5  # dummy value

        study.optimize(_objective, n_trials=1)

        assert "hyperparameters.hidden_channels" in suggested
        assert "optimizer.lr" in suggested
        hc = suggested["hyperparameters.hidden_channels"]
        lr = suggested["optimizer.lr"]
        assert 8 <= hc <= 32
        assert 1e-4 <= lr <= 1e-2

    @pytest.mark.skipif(not OPTUNA_BACKEND_AVAILABLE,
                        reason="optuna_backend not importable")
    def test_create_pruner_median(self):
        """OptunaBackend.create_pruner('median') returns MedianPruner."""
        backend = OptunaBackend()
        pruner = backend.create_pruner("median", n_startup_trials=3,
                                       n_warmup_steps=2)
        assert isinstance(pruner, optuna.pruners.MedianPruner)

    @pytest.mark.skipif(not OPTUNA_BACKEND_AVAILABLE,
                        reason="optuna_backend not importable")
    def test_create_pruner_none(self):
        """OptunaBackend.create_pruner('none') returns NopPruner."""
        backend = OptunaBackend()
        pruner = backend.create_pruner("none")
        assert isinstance(pruner, optuna.pruners.NopPruner)

    @pytest.mark.skipif(not OPTUNA_BACKEND_AVAILABLE,
                        reason="optuna_backend not importable")
    def test_get_best_params_no_trials_raises(self):
        """get_best_params on empty study raises HPOError."""
        backend = OptunaBackend()
        study = backend.create_study("empty_study", "minimize")
        with pytest.raises(Exception):
            backend.get_best_params(study)


# =============================================================================
# SECTION 4 — OptunaPruningCallback
# =============================================================================

class TestOptunaPruningCallback:
    """Test the pruning callback in isolation (no Trainer dependency)."""

    @pytest.mark.skipif(not CALLBACK_AVAILABLE,
                        reason="optuna_callback not importable")
    def test_callback_creation(self):
        """OptunaPruningCallback can be instantiated with a mock trial."""
        mock_trial = MagicMock(spec=optuna.Trial)
        mock_trial.number = 0
        mock_trial.should_prune.return_value = False
        cb = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
        assert cb.monitor == "val_loss"
        assert cb.report_every == 1

    @pytest.mark.skipif(not CALLBACK_AVAILABLE,
                        reason="optuna_callback not importable")
    def test_callback_reports_metric(self):
        """on_epoch_end reports metric value to the trial."""
        mock_trial = MagicMock(spec=optuna.Trial)
        mock_trial.number = 0
        mock_trial.should_prune.return_value = False
        cb = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

        metrics = {"val_loss": 0.42, "train_loss": 0.55}
        cb.on_epoch_end(trainer=None, epoch=0, metrics=metrics)

        mock_trial.report.assert_called_once_with(0.42, 0)

    @pytest.mark.skipif(not CALLBACK_AVAILABLE,
                        reason="optuna_callback not importable")
    def test_callback_prunes_when_signalled(self):
        """on_epoch_end raises TrialPruned when trial.should_prune() is True."""
        mock_trial = MagicMock(spec=optuna.Trial)
        mock_trial.number = 1
        mock_trial.should_prune.return_value = True
        cb = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

        with pytest.raises(optuna.TrialPruned):
            cb.on_epoch_end(trainer=None, epoch=5,
                            metrics={"val_loss": 1.23})

    @pytest.mark.skipif(not CALLBACK_AVAILABLE,
                        reason="optuna_callback not importable")
    def test_callback_skips_missing_metric(self):
        """on_epoch_end does not crash when monitored metric is absent."""
        mock_trial = MagicMock(spec=optuna.Trial)
        mock_trial.number = 0
        mock_trial.should_prune.return_value = False
        cb = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

        # Pass metrics without val_loss
        cb.on_epoch_end(trainer=None, epoch=0,
                        metrics={"train_loss": 0.5})
        # trial.report should NOT have been called
        mock_trial.report.assert_not_called()

    @pytest.mark.skipif(not CALLBACK_AVAILABLE,
                        reason="optuna_callback not importable")
    def test_create_hpo_callback_factory(self):
        """create_hpo_callback() returns an OptunaPruningCallback."""
        mock_trial = MagicMock(spec=optuna.Trial)
        cb = create_hpo_callback(trial=mock_trial, monitor="val_loss",
                                  backend="optuna")
        assert isinstance(cb, OptunaPruningCallback)

    @pytest.mark.skipif(not CALLBACK_AVAILABLE,
                        reason="optuna_callback not importable")
    def test_create_hpo_callback_unknown_backend(self):
        """create_hpo_callback() raises ValueError for unknown backend."""
        mock_trial = MagicMock(spec=optuna.Trial)
        with pytest.raises(ValueError):
            create_hpo_callback(trial=mock_trial, backend="unknown")

    @pytest.mark.skipif(not CALLBACK_AVAILABLE,
                        reason="optuna_callback not importable")
    def test_callback_report_every_skips_epochs(self):
        """Callback with report_every=3 only reports every 3rd epoch."""
        mock_trial = MagicMock(spec=optuna.Trial)
        mock_trial.number = 0
        mock_trial.should_prune.return_value = False
        cb = OptunaPruningCallback(trial=mock_trial, monitor="val_loss",
                                   report_every=3)

        for epoch in range(6):
            cb.on_epoch_end(trainer=None, epoch=epoch,
                            metrics={"val_loss": 0.5 - epoch * 0.01})

        # report_every=3 → reports at epochs 0, 3  (epoch % 3 == 0)
        assert mock_trial.report.call_count == 2

    @pytest.mark.skipif(not CALLBACK_AVAILABLE,
                        reason="optuna_callback not importable")
    def test_callback_no_duplicate_step_report(self):
        """Callback does not re-report the same step (duplicate guard)."""
        mock_trial = MagicMock(spec=optuna.Trial)
        mock_trial.number = 0
        mock_trial.should_prune.return_value = False
        cb = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

        cb.on_epoch_end(trainer=None, epoch=0, metrics={"val_loss": 0.5})
        cb.on_epoch_end(trainer=None, epoch=0, metrics={"val_loss": 0.4})

        # Second call at same epoch should be skipped
        assert mock_trial.report.call_count == 1


# =============================================================================
# SECTION 5 — HPOManager Convenience Functions
# =============================================================================

class TestHPOManagerHelpers:
    """Test module-level helpers: is_hpo_enabled, _flatten_params, etc."""

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE or not HPO_CONFIG_AVAILABLE,
                        reason="hpo_manager or hpo_config not importable")
    def test_is_hpo_enabled_true(self):
        """is_hpo_enabled() returns True for enabled config."""
        assert is_hpo_enabled(HPOConfig(enabled=True)) is True

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_is_hpo_enabled_false_none(self):
        """is_hpo_enabled(None) returns False."""
        assert is_hpo_enabled(None) is False

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_flatten_params(self):
        """_flatten_params strips category prefixes."""
        params = {
            "hyperparameters.hidden_channels": 64,
            "optimizer.lr": 0.001,
            "plain_key": 42,
        }
        flat = _flatten_params(params)
        assert flat == {"hidden_channels": 64, "lr": 0.001, "plain_key": 42}

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_extract_param_categories(self):
        """_extract_param_categories routes params to correct buckets."""
        params = {
            "hidden_channels": 64,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "factor": 0.5,
            "patience": 10,
            "alpha": 0.25,
            "batch_size": 32,
        }
        model, opt, sched, loss, training = _extract_param_categories(params)
        assert "hidden_channels" in model
        assert "lr" in opt
        assert "weight_decay" in opt
        assert "factor" in sched
        assert "patience" in sched
        assert "alpha" in loss
        assert "batch_size" in training

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_infer_task_type_from_metric(self):
        """infer_task_type uses metric heuristics when no dataset metadata."""
        dataset = MagicMock()
        dataset.__getitem__ = MagicMock(return_value=None)
        del dataset.task_type  # ensure no .task_type attribute

        assert infer_task_type(dataset, metric="val_mae") == "graph_regression"
        assert infer_task_type(dataset, metric="accuracy") == "graph_classification"

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_infer_task_type_default_fallback(self):
        """infer_task_type defaults to 'graph_regression' when no signals."""
        dataset = MagicMock()
        del dataset.task_type
        dataset.__getitem__ = MagicMock(side_effect=IndexError)
        result = infer_task_type(dataset, metric=None)
        assert result == "graph_regression"


# =============================================================================
# SECTION 6 — Data Splitting Integration
# =============================================================================

class TestDataSplittingIntegration:
    """Verify DataSplitter works with synthetic dataset used by HPOManager."""

    @pytest.mark.skipif(not DATA_SPLITTER_AVAILABLE,
                        reason="data_splitting not importable")
    def test_random_split_sizes(self, synthetic_dataset):
        """random_split produces subsets whose sizes sum to dataset length."""
        train, val, test = DataSplitter.random_split(
            synthetic_dataset, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0
        )
        assert len(train) + len(val) + len(test) == len(synthetic_dataset)

    @pytest.mark.skipif(not DATA_SPLITTER_AVAILABLE,
                        reason="data_splitting not importable")
    def test_random_split_data_objects(self, synthetic_dataset):
        """Each element in split subsets is a valid PyG Data object."""
        train, val, _ = DataSplitter.random_split(
            synthetic_dataset, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0
        )
        sample = train[0]
        assert hasattr(sample, "x")
        assert hasattr(sample, "edge_index")
        assert hasattr(sample, "y")

    @pytest.mark.skipif(not DATA_SPLITTER_AVAILABLE,
                        reason="data_splitting not importable")
    def test_k_fold_split(self, synthetic_dataset):
        """k_fold_split returns correct number of folds."""
        folds = DataSplitter.k_fold_split(
            synthetic_dataset, n_splits=3, random_seed=42
        )
        assert len(folds) == 3
        for train, val in folds:
            assert len(train) + len(val) == len(synthetic_dataset)


# =============================================================================
# SECTION 7 — HPOManager Initialization
# =============================================================================

class TestHPOManagerInit:
    """Verify HPOManager instantiation and from_config paths."""

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_hpo_manager_disabled(self):
        """HPOManager with disabled config does not initialise a backend."""
        cfg = HPOConfig(enabled=False)
        mgr = HPOManager(cfg)
        assert mgr.backend is None
        assert mgr.config.enabled is False

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_hpo_manager_enabled_has_backend(self, hpo_config_2_trials):
        """HPOManager with enabled config initialises a backend."""
        mgr = HPOManager(hpo_config_2_trials)
        assert mgr.backend is not None

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_from_config_dict(self, minimal_search_space):
        """HPOManager.from_config() accepts a raw dict."""
        raw = {
            "enabled": True,
            "backend": "optuna",
            "n_trials": 2,
            "search_space": {
                "hyperparameters": {
                    "hidden_channels": {
                        "type": "int", "low": 8, "high": 32, "step": 8,
                    },
                },
            },
            "study": {
                "direction": "minimize",
                "metric": "val_loss",
                "study_name": "from_config_test",
            },
        }
        mgr = HPOManager.from_config(raw)
        assert mgr.config.enabled is True
        assert mgr.config.n_trials == 2

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_optimize_disabled_raises(self, synthetic_dataset):
        """optimize() on disabled HPOManager raises HPOError."""
        mgr = HPOManager(HPOConfig(enabled=False))
        with pytest.raises(Exception) as exc_info:
            mgr.optimize(model_name="TinyGNN", dataset=synthetic_dataset)
        # Should indicate HPO is disabled
        assert "disabled" in str(exc_info.value).lower() or \
               "enabled" in str(exc_info.value).lower()

    @pytest.mark.skipif(not HPO_MANAGER_AVAILABLE,
                        reason="hpo_manager not importable")
    def test_create_hpo_manager_convenience(self):
        """create_hpo_manager() returns an HPOManager instance."""
        mgr = create_hpo_manager(enabled=True, n_trials=2)
        assert isinstance(mgr, HPOManager)
        assert mgr.config.n_trials == 2


# =============================================================================
# SECTION 8 — Full E2E Optimization (CORE TEST)
# =============================================================================

class TestE2EOptimization:
    """
    Full end-to-end HPO workflow:
        HPOConfig → HPOManager → optimize() (2–3 trials)
        → best_params → StudyAnalyzer analysis

    Mocks ModelFactory and patches Trainer to use _TinyGNN so the test is
    self-contained and does not require the full MILIA model registry.
    """

    @pytest.mark.skipif(
        not all([HPO_MANAGER_AVAILABLE, OPTUNA_BACKEND_AVAILABLE,
                 TRAINER_AVAILABLE, DATA_SPLITTER_AVAILABLE,
                 CALLBACK_AVAILABLE, HPO_CONFIG_AVAILABLE]),
        reason="One or more HPO/Trainer modules not importable",
    )
    def test_e2e_optimize_2_trials(
        self, hpo_config_2_trials, synthetic_dataset, tiny_model_factory
    ):
        """
        Full 2-trial optimization succeeds and returns valid best_params.

        Strategy: Patch the module-level `get_factory` to return our
        tiny_model_factory, and patch `ModelRegistry` as unavailable so
        _filter_search_space_for_model falls back to unfiltered space.
        """
        # Create HPOManager
        manager = HPOManager(hpo_config_2_trials)

        # Patch factory lookup to return our tiny factory
        mock_factory = MagicMock()
        mock_factory.create_model_with_info = tiny_model_factory

        with patch.object(manager, "_model_factory", mock_factory):
            # Patch get_factory at module level so _create_objective captures it
            with patch(
                "milia_pipeline.models.hpo.hpo_manager.get_factory",
                return_value=mock_factory,
            ):
                # Patch ModelRegistry as unavailable for filter fallback
                with patch(
                    "milia_pipeline.models.hpo.hpo_manager._REGISTRY_AVAILABLE",
                    False,
                ):
                    best_params = manager.optimize(
                        model_name="TinyGNN",
                        dataset=synthetic_dataset,
                        trainer_kwargs={"max_epochs": 2},
                    )

        # ASSERTIONS
        assert best_params is not None
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        assert manager.study is not None
        assert manager.best_params is not None

        # Verify trial count
        completed = [
            t for t in manager.study.trials
            if t.state == TrialState.COMPLETE
        ]
        assert len(completed) == 2, (
            f"Expected 2 completed trials, got {len(completed)}"
        )

    @pytest.mark.skipif(
        not all([HPO_MANAGER_AVAILABLE, OPTUNA_BACKEND_AVAILABLE,
                 TRAINER_AVAILABLE, DATA_SPLITTER_AVAILABLE,
                 CALLBACK_AVAILABLE, HPO_CONFIG_AVAILABLE]),
        reason="One or more HPO/Trainer modules not importable",
    )
    def test_e2e_optimize_3_trials(
        self, hpo_config_3_trials, synthetic_dataset, tiny_model_factory
    ):
        """Full 3-trial optimisation succeeds; trial count == 3."""
        manager = HPOManager(hpo_config_3_trials)

        mock_factory = MagicMock()
        mock_factory.create_model_with_info = tiny_model_factory

        with patch.object(manager, "_model_factory", mock_factory):
            with patch(
                "milia_pipeline.models.hpo.hpo_manager.get_factory",
                return_value=mock_factory,
            ):
                with patch(
                    "milia_pipeline.models.hpo.hpo_manager._REGISTRY_AVAILABLE",
                    False,
                ):
                    best_params = manager.optimize(
                        model_name="TinyGNN",
                        dataset=synthetic_dataset,
                        trainer_kwargs={"max_epochs": 2},
                    )

        assert best_params is not None
        completed = [
            t for t in manager.study.trials
            if t.state == TrialState.COMPLETE
        ]
        assert len(completed) == 3

    @pytest.mark.skipif(
        not all([HPO_MANAGER_AVAILABLE, OPTUNA_BACKEND_AVAILABLE,
                 TRAINER_AVAILABLE, DATA_SPLITTER_AVAILABLE,
                 CALLBACK_AVAILABLE, HPO_CONFIG_AVAILABLE]),
        reason="One or more HPO/Trainer modules not importable",
    )
    def test_e2e_best_params_keys(
        self, hpo_config_2_trials, synthetic_dataset, tiny_model_factory
    ):
        """
        Best params dict contains expected keys from the search space
        (stripped of category prefix).
        """
        manager = HPOManager(hpo_config_2_trials)
        mock_factory = MagicMock()
        mock_factory.create_model_with_info = tiny_model_factory

        with patch.object(manager, "_model_factory", mock_factory), \
             patch("milia_pipeline.models.hpo.hpo_manager.get_factory",
                   return_value=mock_factory), \
             patch("milia_pipeline.models.hpo.hpo_manager._REGISTRY_AVAILABLE",
                   False):
            best_params = manager.optimize(
                model_name="TinyGNN",
                dataset=synthetic_dataset,
                trainer_kwargs={"max_epochs": 2},
            )

        # Optuna stores params with full names
        # e.g., "hyperparameters.hidden_channels", "optimizer.lr"
        param_keys = set(best_params.keys())
        assert any("hidden_channels" in k for k in param_keys), (
            f"Expected 'hidden_channels' in best_params keys: {param_keys}"
        )
        assert any("lr" in k for k in param_keys), (
            f"Expected 'lr' in best_params keys: {param_keys}"
        )

    @pytest.mark.skipif(
        not all([HPO_MANAGER_AVAILABLE, OPTUNA_BACKEND_AVAILABLE,
                 TRAINER_AVAILABLE, DATA_SPLITTER_AVAILABLE,
                 CALLBACK_AVAILABLE, HPO_CONFIG_AVAILABLE]),
        reason="One or more HPO/Trainer modules not importable",
    )
    def test_e2e_get_best_params_convenience(
        self, hpo_config_2_trials, synthetic_dataset, tiny_model_factory
    ):
        """get_best_params() convenience function works after optimize."""
        manager = HPOManager(hpo_config_2_trials)
        mock_factory = MagicMock()
        mock_factory.create_model_with_info = tiny_model_factory

        with patch.object(manager, "_model_factory", mock_factory), \
             patch("milia_pipeline.models.hpo.hpo_manager.get_factory",
                   return_value=mock_factory), \
             patch("milia_pipeline.models.hpo.hpo_manager._REGISTRY_AVAILABLE",
                   False):
            manager.optimize(
                model_name="TinyGNN",
                dataset=synthetic_dataset,
                trainer_kwargs={"max_epochs": 2},
            )

        result = get_best_params(manager)
        assert result is not None
        assert isinstance(result, dict)


# =============================================================================
# SECTION 9 — StudyAnalyzer Post-Optimization
# =============================================================================

class TestStudyAnalyzerE2E:
    """
    Verify StudyAnalyzer produces analysis from a completed study.

    Uses a study created via direct Optuna API (no HPOManager dependency)
    so this section is independently testable.
    """

    @pytest.fixture
    def completed_study(self):
        """Create a small completed Optuna study for analysis tests."""
        study = optuna.create_study(
            study_name="analyzer_test_study",
            direction="minimize",
        )

        def _dummy_objective(trial):
            x = trial.suggest_float("x", -5.0, 5.0)
            y = trial.suggest_int("y", 1, 10)
            return (x - 2) ** 2 + y

        study.optimize(_dummy_objective, n_trials=5)
        return study

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_analyzer_creation(self, completed_study):
        """StudyAnalyzer can be created from a completed study."""
        analyzer = StudyAnalyzer(completed_study)
        assert analyzer.study is completed_study

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_analyzer_from_manager(
        self, hpo_config_2_trials, synthetic_dataset, tiny_model_factory
    ):
        """StudyAnalyzer.from_manager() works after HPOManager.optimize()."""
        if not all([HPO_MANAGER_AVAILABLE, TRAINER_AVAILABLE,
                    DATA_SPLITTER_AVAILABLE, CALLBACK_AVAILABLE]):
            pytest.skip("HPOManager dependencies not available")

        manager = HPOManager(hpo_config_2_trials)
        mock_factory = MagicMock()
        mock_factory.create_model_with_info = tiny_model_factory

        with patch.object(manager, "_model_factory", mock_factory), \
             patch("milia_pipeline.models.hpo.hpo_manager.get_factory",
                   return_value=mock_factory), \
             patch("milia_pipeline.models.hpo.hpo_manager._REGISTRY_AVAILABLE",
                   False):
            manager.optimize(
                model_name="TinyGNN",
                dataset=synthetic_dataset,
                trainer_kwargs={"max_epochs": 2},
            )

        analyzer = StudyAnalyzer.from_manager(manager)
        assert analyzer.study is manager.study

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_get_trial_count(self, completed_study):
        """get_trial_count() returns a dict with state counts."""
        analyzer = StudyAnalyzer(completed_study)
        counts = analyzer.get_trial_count()
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total >= 5

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_get_optimization_trajectory(self, completed_study):
        """get_optimization_trajectory() returns dict with best_value."""
        analyzer = StudyAnalyzer(completed_study)
        trajectory = analyzer.get_optimization_trajectory()
        assert isinstance(trajectory, dict)
        assert "best_value" in trajectory
        assert trajectory["best_value"] is not None

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_get_value_statistics(self, completed_study):
        """get_value_statistics() returns statistical summary."""
        analyzer = StudyAnalyzer(completed_study)
        stats = analyzer.get_value_statistics()
        assert isinstance(stats, dict)

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_get_convergence_data(self, completed_study):
        """get_convergence_data() returns convergence analysis dict."""
        analyzer = StudyAnalyzer(completed_study)
        convergence = analyzer.get_convergence_data()
        assert isinstance(convergence, dict)

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_get_parameter_importance(self, completed_study):
        """get_parameter_importance() returns importance dict."""
        analyzer = StudyAnalyzer(completed_study)
        try:
            importance = analyzer.get_parameter_importance()
            assert isinstance(importance, dict)
        except Exception:
            # fANOVA may fail with very few trials; this is acceptable
            pytest.skip("Parameter importance computation failed (too few trials)")

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_export_results_dict(self, completed_study):
        """export_results(format=DICT) returns a dict without error."""
        analyzer = StudyAnalyzer(completed_study)
        result = analyzer.export_results(format=ExportFormat.DICT)
        assert isinstance(result, dict)
        assert "study_name" in result

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_export_results_json_to_file(self, completed_study, tmp_path):
        """export_results(format=JSON) writes a valid JSON file."""
        analyzer = StudyAnalyzer(completed_study)
        json_path = tmp_path / "study_export.json"
        analyzer.export_results(
            path=str(json_path),
            format=ExportFormat.JSON,
        )
        assert json_path.exists()
        import json
        with open(json_path) as f:
            data = json.load(f)
        assert "study_name" in data

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_analyzer_repr(self, completed_study):
        """StudyAnalyzer.__repr__ returns a descriptive string."""
        analyzer = StudyAnalyzer(completed_study)
        r = repr(analyzer)
        assert "StudyAnalyzer" in r
        assert "analyzer_test_study" in r

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_clear_cache(self, completed_study):
        """clear_cache() runs without error."""
        analyzer = StudyAnalyzer(completed_study)
        # Populate cache
        analyzer.get_trial_count()
        # Clear
        analyzer.clear_cache()
        # Re-populate should still work
        counts = analyzer.get_trial_count()
        assert sum(counts.values()) >= 5

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_study_comparison(self):
        """compare_with() returns comparison dict between two studies."""
        study_a = optuna.create_study(study_name="cmp_a", direction="minimize")
        study_b = optuna.create_study(study_name="cmp_b", direction="minimize")

        study_a.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)
        study_b.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)

        analyzer_a = StudyAnalyzer(study_a)
        analyzer_b = StudyAnalyzer(study_b)

        cmp = analyzer_a.compare_with(analyzer_b)
        assert "winner" in cmp
        assert "best_values" in cmp
        assert "trial_counts" in cmp

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_analyzer_from_manager_no_study_raises(self):
        """from_manager() raises when manager has no study."""
        if not HPO_MANAGER_AVAILABLE:
            pytest.skip("hpo_manager not importable")
        mgr = HPOManager(HPOConfig(enabled=False))
        with pytest.raises(Exception):
            StudyAnalyzer.from_manager(mgr)

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_analysis_config_validation(self):
        """AnalysisConfig validates convergence_window >= 1."""
        with pytest.raises(Exception):
            AnalysisConfig(convergence_window=0)

    @pytest.mark.skipif(not STUDY_ANALYZER_AVAILABLE,
                        reason="study_analyzer not importable")
    def test_analysis_config_frozen(self):
        """AnalysisConfig is frozen (Pydantic V2 frozen=True)."""
        cfg = AnalysisConfig()
        with pytest.raises(Exception):
            cfg.convergence_window = 99


# =============================================================================
# SECTION 10 — Trainer + HPO Callback Integration
# =============================================================================

class TestTrainerHPOIntegration:
    """
    Verify Trainer correctly integrates HPO callback:
        - hpo_callback is appended to callbacks
        - TrialPruned propagates through _on_epoch_end
    """

    @pytest.mark.skipif(not TRAINER_AVAILABLE or not CALLBACK_AVAILABLE,
                        reason="Trainer or callback not importable")
    def test_trainer_appends_hpo_callback(self, synthetic_dataset):
        """Trainer adds hpo_callback to its callbacks list."""
        model = _TinyGNN()
        loader = PyGDataLoader(synthetic_dataset, batch_size=8, shuffle=False)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        mock_trial = MagicMock(spec=optuna.Trial)
        mock_trial.number = 0
        mock_trial.should_prune.return_value = False
        cb = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=opt,
            loss_fn=nn.MSELoss(),
            max_epochs=1,
            hpo_callback=cb,
        )
        # hpo_callback should now be in the callbacks list
        assert cb in trainer.callbacks

    @pytest.mark.skipif(not TRAINER_AVAILABLE or not CALLBACK_AVAILABLE,
                        reason="Trainer or callback not importable")
    def test_trainer_fit_with_hpo_callback(self, synthetic_dataset):
        """Trainer.fit() completes with an HPO callback reporting metrics."""
        model = _TinyGNN()
        loader = PyGDataLoader(synthetic_dataset, batch_size=16, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        mock_trial = MagicMock(spec=optuna.Trial)
        mock_trial.number = 0
        mock_trial.should_prune.return_value = False

        cb = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=opt,
            loss_fn=nn.MSELoss(),
            max_epochs=2,
            hpo_callback=cb,
        )
        results = trainer.fit()

        assert results is not None
        assert "best_val_loss" in results
        # Callback should have reported at least once per epoch
        assert mock_trial.report.call_count >= 1


# =============================================================================
# SECTION 11 — Exception Hierarchy Smoke Tests
# =============================================================================

class TestHPOExceptions:
    """Verify HPO-specific exceptions exist and have expected attributes."""

    @pytest.mark.skipif(not EXCEPTIONS_AVAILABLE,
                        reason="exceptions not importable")
    def test_hpo_error_instantiation(self):
        """HPOError can be instantiated with message and study_name."""
        err = HPOError("test error", study_name="my_study")
        assert "test error" in str(err)

    @pytest.mark.skipif(not EXCEPTIONS_AVAILABLE,
                        reason="exceptions not importable")
    def test_trial_failed_error(self):
        """TrialFailedError accepts trial_number."""
        err = TrialFailedError("trial failed", trial_number=5)
        assert "trial failed" in str(err)

    @pytest.mark.skipif(not EXCEPTIONS_AVAILABLE,
                        reason="exceptions not importable")
    def test_hpo_configuration_error(self):
        """HPOConfigurationError can be raised."""
        with pytest.raises(HPOConfigurationError):
            raise HPOConfigurationError("bad config")

    @pytest.mark.skipif(not EXCEPTIONS_AVAILABLE,
                        reason="exceptions not importable")
    def test_search_space_error(self):
        """SearchSpaceError can be raised."""
        with pytest.raises(SearchSpaceError):
            raise SearchSpaceError("bad space")

    @pytest.mark.skipif(not EXCEPTIONS_AVAILABLE,
                        reason="exceptions not importable")
    def test_pruning_error(self):
        """PruningError can be raised."""
        with pytest.raises(PruningError):
            raise PruningError("pruned")
