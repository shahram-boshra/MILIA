#!/usr/bin/env python3
"""
Contract Tests: Model Registries (Section 2.4)

Verifies that model-related registries (LossRegistry, OptimizerRegistry,
SchedulerRegistry, MetricsRegistry) expose consistent APIs and all registered
entries can be instantiated with default/minimal parameters.

Contract guarantees tested:
  1. Each registry lists all entries via its list method.
  2. Every listed entry can be instantiated without error.
  3. Returned objects are instances of the expected base type.
  4. Invalid names raise ValueError with informative messages.
  5. get_*_info / get_valid_params introspection methods return dicts.
  6. Custom registration works and respects overwrite semantics.
  7. Parameter filtering silently drops unsupported params.
  8. Task-aware selection returns correct types (LossRegistry, MetricsRegistry).
  9. Convenience module-level functions delegate correctly.

Modules exercised:
  - milia_pipeline/models/training/loss_functions.py
  - milia_pipeline/models/training/optimizers.py
  - milia_pipeline/models/training/schedulers.py
  - milia_pipeline/models/training/metrics.py

Run from project root:
  pytest tests/test_contract_model_registries.py -v --tb=short

Author: MILIA Team
"""

import logging
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Path setup – add project root so `milia_pipeline` is importable when
# running from /app/milia (Docker) or any other working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module imports under test
# ---------------------------------------------------------------------------
from milia_pipeline.models.training.loss_functions import (
    FocalLoss,
    LossRegistry,
    get_default_loss_for_task,
    get_loss,
    get_loss_for_task,
    is_loss_compatible_with_task,
    list_losses,
)
from milia_pipeline.models.training.metrics import (
    TORCHMETRICS_AVAILABLE,
    MetricsRegistry,
    get_default_metrics_for_task,
    get_metric,
    get_metrics_for_task,
    is_metric_compatible_with_task,
    list_metrics,
)
from milia_pipeline.models.training.optimizers import (
    OptimizerRegistry,
    get_optimizer,
    list_optimizers,
)
from milia_pipeline.models.training.schedulers import (
    SchedulerRegistry,
    create_warmup_scheduler,
    get_scheduler,
    list_schedulers,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.contract


# ===========================================================================
# Shared fixtures
# ===========================================================================


@pytest.fixture
def dummy_model():
    """A minimal nn.Module whose parameters can feed optimizers / schedulers."""
    model = nn.Linear(4, 1)
    return model


@pytest.fixture
def dummy_optimizer(dummy_model):
    """An Adam optimizer on the dummy model (needed by schedulers)."""
    return torch.optim.Adam(dummy_model.parameters(), lr=0.01)


# ===========================================================================
# 1. LOSS REGISTRY CONTRACTS
# ===========================================================================


class TestLossRegistryContract:
    """Contract: LossRegistry exposes consistent API and all entries instantiate."""

    # ----- 1a. Listing contract -----

    def test_list_available_returns_sorted_list(self):
        """list_available() returns a sorted list of strings."""
        available = LossRegistry.list_available()
        assert isinstance(available, list)
        assert len(available) > 0
        assert available == sorted(available), "list_available() must return sorted names"
        assert all(isinstance(name, str) for name in available)

    def test_list_available_contains_expected_entries(self):
        """Registry must contain the documented 18 loss entries."""
        available = set(LossRegistry.list_available())
        # Regression losses
        for name in ("mse", "mae", "l1", "huber", "smooth_l1", "rmse", "weighted_mse"):
            assert name in available, f"Missing regression loss: {name}"
        # Classification losses
        for name in ("cross_entropy", "ce", "nll", "bce", "bce_with_logits", "focal"):
            assert name in available, f"Missing classification loss: {name}"
        # Other losses
        for name in (
            "multilabel_soft_margin",
            "margin_ranking",
            "triplet_margin",
            "kl_div",
            "poisson_nll",
            "cosine_embedding",
        ):
            assert name in available, f"Missing other loss: {name}"

    # ----- 1b. Instantiation contract -----

    def test_all_losses_instantiate_with_defaults(self):
        """Every registered loss can be instantiated with no explicit params."""
        for name in LossRegistry.list_available():
            loss_fn = LossRegistry.get_loss(name)
            assert isinstance(loss_fn, nn.Module), (
                f"Loss '{name}' did not return an nn.Module instance"
            )

    def test_focal_loss_instantiates_with_params(self):
        """FocalLoss accepts alpha, gamma, reduction."""
        loss_fn = LossRegistry.get_loss("focal", {"alpha": 0.5, "gamma": 3.0, "reduction": "sum"})
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha == 0.5
        assert loss_fn.gamma == 3.0
        assert loss_fn.reduction == "sum"

    def test_mse_loss_callable(self):
        """MSE loss produces a scalar tensor on dummy data."""
        loss_fn = LossRegistry.get_loss("mse")
        preds = torch.randn(8, 1)
        targets = torch.randn(8, 1)
        result = loss_fn(preds, targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0, "Loss should return a scalar"

    # ----- 1c. Error contract -----

    def test_unknown_loss_raises_valueerror(self):
        """Requesting a non-existent loss raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss function"):
            LossRegistry.get_loss("nonexistent_loss_xyz")

    # ----- 1d. Parameter filtering contract -----

    def test_invalid_params_filtered_silently(self):
        """Invalid parameters are filtered out — no crash."""
        loss_fn = LossRegistry.get_loss("mse", {"alpha": 0.5, "nonexistent": True})
        assert isinstance(loss_fn, nn.Module)

    # ----- 1e. Introspection contract -----

    def test_get_loss_info_returns_dict(self):
        """get_loss_info returns a dict with expected keys."""
        for name in ("mse", "focal", "cross_entropy"):
            info = LossRegistry.get_loss_info(name)
            assert isinstance(info, dict)
            assert "name" in info
            assert "class" in info
            assert "valid_params" in info

    def test_get_valid_params_returns_dict(self):
        """get_valid_params returns a dict for every registered loss."""
        for name in LossRegistry.list_available():
            params = LossRegistry.get_valid_params(name)
            assert isinstance(params, dict), f"get_valid_params('{name}') did not return dict"

    def test_get_loss_info_unknown_raises_valueerror(self):
        """get_loss_info raises ValueError for unknown loss."""
        with pytest.raises(ValueError, match="Unknown loss function"):
            LossRegistry.get_loss_info("nonexistent_loss_xyz")

    # ----- 1f. Task-aware selection contract -----

    def test_get_loss_for_task_regression(self):
        """Auto-selected loss for regression task is an nn.Module."""
        for task in ("graph_regression", "node_regression", "edge_regression"):
            loss_fn = LossRegistry.get_loss_for_task(task)
            assert isinstance(loss_fn, nn.Module), f"get_loss_for_task('{task}') failed"

    def test_get_loss_for_task_classification(self):
        """Auto-selected loss for classification task is an nn.Module."""
        for task in ("graph_classification", "node_classification", "edge_classification"):
            loss_fn = LossRegistry.get_loss_for_task(task)
            assert isinstance(loss_fn, nn.Module), f"get_loss_for_task('{task}') failed"

    def test_get_loss_for_task_link_prediction(self):
        """Auto-selected loss for link_prediction is an nn.Module."""
        loss_fn = LossRegistry.get_loss_for_task("link_prediction")
        assert isinstance(loss_fn, nn.Module)

    def test_get_default_loss_for_task_returns_string(self):
        """get_default_loss_for_task returns a string name for every known task."""
        for task in ("graph_regression", "graph_classification", "link_prediction"):
            default = LossRegistry.get_default_loss_for_task(task)
            assert isinstance(default, str)
            assert default in LossRegistry._losses, (
                f"Default loss '{default}' for task '{task}' is not registered"
            )

    def test_incompatible_loss_overridden_for_classification(self):
        """Regression loss for classification task is auto-corrected."""
        loss_fn = LossRegistry.get_loss_for_task("graph_classification", "mse")
        # Must NOT be MSELoss — should be overridden to CrossEntropyLoss
        assert not isinstance(loss_fn, nn.MSELoss), (
            "MSE should be overridden for classification tasks"
        )
        assert isinstance(loss_fn, nn.Module)

    def test_incompatible_loss_overridden_for_regression(self):
        """Classification loss for regression task is auto-corrected."""
        loss_fn = LossRegistry.get_loss_for_task("graph_regression", "cross_entropy")
        assert not isinstance(loss_fn, nn.CrossEntropyLoss), (
            "CrossEntropy should be overridden for regression tasks"
        )
        assert isinstance(loss_fn, nn.Module)

    # ----- 1g. Compatibility check contract -----

    def test_is_loss_compatible_with_task(self):
        """is_loss_compatible_with_task reflects correct categories."""
        assert LossRegistry.is_loss_compatible_with_task("mse", "graph_regression") is True
        assert LossRegistry.is_loss_compatible_with_task("mse", "graph_classification") is False
        assert (
            LossRegistry.is_loss_compatible_with_task("cross_entropy", "graph_classification")
            is True
        )
        assert (
            LossRegistry.is_loss_compatible_with_task("cross_entropy", "graph_regression") is False
        )

    # ----- 1h. Custom registration contract -----

    def test_register_custom_loss(self):
        """Custom nn.Module subclass can be registered and retrieved."""

        class _TestLoss(nn.Module):
            def forward(self, inp, tgt):
                return ((inp - tgt) ** 2).mean()

        name = "_contract_test_custom_loss"
        try:
            LossRegistry.register_custom_loss(name, _TestLoss)
            loss_fn = LossRegistry.get_loss(name)
            assert isinstance(loss_fn, _TestLoss)
        finally:
            # Cleanup
            LossRegistry._losses.pop(name, None)

    def test_register_custom_loss_duplicate_raises(self):
        """Re-registering without overwrite=True raises ValueError."""

        class _TestLoss2(nn.Module):
            def forward(self, inp, tgt):
                return torch.tensor(0.0)

        name = "_contract_test_dup_loss"
        try:
            LossRegistry.register_custom_loss(name, _TestLoss2)
            with pytest.raises(ValueError, match="already registered"):
                LossRegistry.register_custom_loss(name, _TestLoss2, overwrite=False)
        finally:
            LossRegistry._losses.pop(name, None)

    def test_register_non_module_raises_typeerror(self):
        """Registering a non-nn.Module class raises TypeError."""
        with pytest.raises(TypeError, match="subclass of nn.Module"):
            LossRegistry.register_custom_loss("_bad_loss", int)

    # ----- 1i. Convenience functions contract -----

    def test_convenience_get_loss(self):
        """Module-level get_loss delegates to LossRegistry."""
        loss_fn = get_loss("mse")
        assert isinstance(loss_fn, nn.Module)

    def test_convenience_list_losses(self):
        """Module-level list_losses delegates to LossRegistry."""
        result = list_losses()
        assert result == LossRegistry.list_available()

    def test_convenience_get_loss_for_task(self):
        """Module-level get_loss_for_task delegates to LossRegistry."""
        loss_fn = get_loss_for_task("graph_regression")
        assert isinstance(loss_fn, nn.Module)

    def test_convenience_get_default_loss_for_task(self):
        """Module-level get_default_loss_for_task delegates to LossRegistry."""
        result = get_default_loss_for_task("graph_classification")
        assert isinstance(result, str)

    def test_convenience_is_loss_compatible_with_task(self):
        """Module-level is_loss_compatible_with_task delegates to LossRegistry."""
        result = is_loss_compatible_with_task("mse", "graph_regression")
        assert result is True


# ===========================================================================
# 2. OPTIMIZER REGISTRY CONTRACTS
# ===========================================================================


class TestOptimizerRegistryContract:
    """Contract: OptimizerRegistry exposes consistent API and all entries instantiate."""

    # ----- 2a. Listing contract -----

    def test_list_available_returns_sorted_list(self):
        """list_available() returns a sorted list of strings."""
        available = OptimizerRegistry.list_available()
        assert isinstance(available, list)
        assert len(available) > 0
        assert available == sorted(available)

    def test_list_available_contains_expected_entries(self):
        """Registry must contain the documented 12 optimizer entries."""
        available = set(OptimizerRegistry.list_available())
        expected = {
            "adam",
            "adamw",
            "adamax",
            "adadelta",
            "adagrad",
            "rmsprop",
            "sgd",
            "asgd",
            "lbfgs",
            "rprop",
            "nadam",
            "radam",
        }
        for name in expected:
            assert name in available, f"Missing optimizer: {name}"

    # ----- 2b. Instantiation contract -----

    def test_all_optimizers_instantiate_with_defaults(self, dummy_model):
        """Every registered optimizer can be instantiated with model parameters."""
        for name in OptimizerRegistry.list_available():
            optimizer = OptimizerRegistry.get_optimizer(name, dummy_model.parameters())
            assert isinstance(optimizer, torch.optim.Optimizer), (
                f"Optimizer '{name}' did not return a torch.optim.Optimizer"
            )

    def test_adam_with_custom_lr(self, dummy_model):
        """Adam optimizer accepts custom lr parameter."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", dummy_model.parameters(), {"lr": 0.0001}
        )
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 0.0001

    # ----- 2c. Error contract -----

    def test_unknown_optimizer_raises_valueerror(self, dummy_model):
        """Requesting a non-existent optimizer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptimizerRegistry.get_optimizer("nonexistent_opt_xyz", dummy_model.parameters())

    # ----- 2d. Parameter filtering contract -----

    def test_invalid_params_filtered_silently(self, dummy_model):
        """Invalid parameters are filtered out — no crash."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", dummy_model.parameters(), {"lr": 0.001, "completely_invalid_param": 999}
        )
        assert isinstance(optimizer, torch.optim.Optimizer)

    # ----- 2e. Introspection contract -----

    def test_get_optimizer_info_returns_dict(self):
        """get_optimizer_info returns a dict with expected keys."""
        for name in ("adam", "sgd", "rmsprop"):
            info = OptimizerRegistry.get_optimizer_info(name)
            assert isinstance(info, dict)
            assert "name" in info
            assert "class" in info
            assert "valid_params" in info
            assert "default_params" in info

    def test_get_valid_params_returns_dict(self):
        """get_valid_params returns a dict for every registered optimizer."""
        for name in OptimizerRegistry.list_available():
            params = OptimizerRegistry.get_valid_params(name)
            assert isinstance(params, dict), f"get_valid_params('{name}') did not return dict"

    def test_get_default_params_returns_dict(self):
        """get_default_params returns a dict (may be empty for some optimizers)."""
        for name in ("adam", "adamw", "sgd", "rmsprop", "adagrad"):
            defaults = OptimizerRegistry.get_default_params(name)
            assert isinstance(defaults, dict)
            assert len(defaults) > 0, f"Expected non-empty defaults for '{name}'"

    def test_get_optimizer_info_unknown_raises_valueerror(self):
        """get_optimizer_info raises ValueError for unknown optimizer."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptimizerRegistry.get_optimizer_info("nonexistent_opt_xyz")

    # ----- 2f. Custom registration contract -----

    def test_register_custom_optimizer(self, dummy_model):
        """Custom Optimizer subclass can be registered and retrieved."""

        class _TestOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        name = "_contract_test_custom_opt"
        try:
            OptimizerRegistry.register_custom_optimizer(name, _TestOptimizer, {"lr": 0.01})
            opt = OptimizerRegistry.get_optimizer(name, dummy_model.parameters())
            assert isinstance(opt, _TestOptimizer)
        finally:
            OptimizerRegistry._optimizers.pop(name, None)
            OptimizerRegistry._defaults.pop(name, None)

    def test_register_custom_optimizer_duplicate_raises(self, dummy_model):
        """Re-registering without overwrite=True raises ValueError."""

        class _TestOptimizer2(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        name = "_contract_test_dup_opt"
        try:
            OptimizerRegistry.register_custom_optimizer(name, _TestOptimizer2)
            with pytest.raises(ValueError, match="already registered"):
                OptimizerRegistry.register_custom_optimizer(name, _TestOptimizer2, overwrite=False)
        finally:
            OptimizerRegistry._optimizers.pop(name, None)
            OptimizerRegistry._defaults.pop(name, None)

    def test_register_non_optimizer_raises_typeerror(self):
        """Registering a non-Optimizer class raises TypeError."""
        with pytest.raises(TypeError, match="subclass of torch.optim.Optimizer"):
            OptimizerRegistry.register_custom_optimizer("_bad_opt", nn.Linear)

    # ----- 2g. Convenience functions contract -----

    def test_convenience_get_optimizer(self, dummy_model):
        """Module-level get_optimizer delegates to OptimizerRegistry."""
        opt = get_optimizer("adam", dummy_model.parameters(), {"lr": 0.001})
        assert isinstance(opt, torch.optim.Optimizer)

    def test_convenience_list_optimizers(self):
        """Module-level list_optimizers delegates to OptimizerRegistry."""
        result = list_optimizers()
        assert result == OptimizerRegistry.list_available()


# ===========================================================================
# 3. SCHEDULER REGISTRY CONTRACTS
# ===========================================================================


class TestSchedulerRegistryContract:
    """Contract: SchedulerRegistry exposes consistent API and all entries instantiate."""

    # Schedulers that require complex first-positional args beyond optimizer
    # and cannot be instantiated with simple default params alone.
    _SKIP_INSTANTIATION = {"chained", "sequential"}

    # ----- 3a. Listing contract -----

    def test_list_available_returns_sorted_list(self):
        """list_available() returns a sorted list of strings."""
        available = SchedulerRegistry.list_available()
        assert isinstance(available, list)
        assert len(available) > 0
        assert available == sorted(available)

    def test_list_available_contains_expected_entries(self):
        """Registry must contain the documented 13 scheduler entries."""
        available = set(SchedulerRegistry.list_available())
        expected = {
            "reduce_on_plateau",
            "step_lr",
            "multistep_lr",
            "exponential_lr",
            "cosine_annealing",
            "cosine_annealing_warm_restarts",
            "cyclic_lr",
            "one_cycle",
            "polynomial_lr",
            "linear_lr",
            "chained",
            "sequential",
            "constant_lr",
        }
        for name in expected:
            assert name in available, f"Missing scheduler: {name}"

    # ----- 3b. Instantiation contract -----

    def test_instantiable_schedulers_with_defaults(self, dummy_optimizer):
        """Every instantiable scheduler can be created with defaults/minimal params.

        Note: 'chained' and 'sequential' require list-of-schedulers / milestones
        as mandatory first arguments and are tested separately.
        """
        # Schedulers that need specific required params beyond optimizer
        required_params = {
            "multistep_lr": {"milestones": [10, 20]},
            "one_cycle": {"max_lr": 0.01, "total_steps": 100},
            "cosine_annealing_warm_restarts": {"T_0": 10},
        }

        for name in SchedulerRegistry.list_available():
            if name in self._SKIP_INSTANTIATION:
                continue

            params = required_params.get(name)
            scheduler = SchedulerRegistry.get_scheduler(name, dummy_optimizer, params)
            assert scheduler is not None, f"Scheduler '{name}' returned None"

    def test_chained_scheduler_requires_special_args(self):
        """'chained' scheduler requires schedulers list — verify it's registered."""
        assert "chained" in SchedulerRegistry._schedulers
        assert SchedulerRegistry._schedulers["chained"] is torch.optim.lr_scheduler.ChainedScheduler

    def test_sequential_scheduler_requires_special_args(self):
        """'sequential' scheduler requires schedulers + milestones — verify it's registered."""
        assert "sequential" in SchedulerRegistry._schedulers
        assert SchedulerRegistry._schedulers["sequential"] is torch.optim.lr_scheduler.SequentialLR

    def test_step_lr_with_custom_params(self, dummy_optimizer):
        """StepLR accepts custom step_size and gamma."""
        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", dummy_optimizer, {"step_size": 5, "gamma": 0.5}
        )
        assert scheduler is not None
        assert scheduler.step_size == 5
        assert scheduler.gamma == 0.5

    # ----- 3c. Error contract -----

    def test_unknown_scheduler_raises_valueerror(self, dummy_optimizer):
        """Requesting a non-existent scheduler raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            SchedulerRegistry.get_scheduler("nonexistent_sched_xyz", dummy_optimizer)

    # ----- 3d. Parameter filtering contract -----

    def test_invalid_params_filtered_silently(self, dummy_optimizer):
        """Invalid parameters are filtered out — no crash."""
        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", dummy_optimizer, {"step_size": 10, "completely_invalid_param": 999}
        )
        assert scheduler is not None

    # ----- 3e. Metric-based detection contract -----

    def test_is_metric_based(self):
        """reduce_on_plateau is metric-based; others are not."""
        assert SchedulerRegistry.is_metric_based("reduce_on_plateau") is True
        assert SchedulerRegistry.is_metric_based("step_lr") is False
        assert SchedulerRegistry.is_metric_based("cosine_annealing") is False

    # ----- 3f. Introspection contract -----

    def test_get_scheduler_info_returns_dict(self):
        """get_scheduler_info returns a dict with expected keys."""
        for name in ("step_lr", "reduce_on_plateau", "cosine_annealing"):
            info = SchedulerRegistry.get_scheduler_info(name)
            assert isinstance(info, dict)
            assert "name" in info
            assert "class" in info
            assert "metric_based" in info
            assert "valid_params" in info

    def test_get_valid_params_returns_dict(self):
        """get_valid_params returns a dict for every registered scheduler."""
        for name in SchedulerRegistry.list_available():
            params = SchedulerRegistry.get_valid_params(name)
            assert isinstance(params, dict), f"get_valid_params('{name}') did not return dict"

    def test_get_default_params_returns_dict(self):
        """get_default_params returns a dict (may be empty for some schedulers)."""
        for name in ("reduce_on_plateau", "step_lr", "cosine_annealing"):
            defaults = SchedulerRegistry.get_default_params(name)
            assert isinstance(defaults, dict)
            assert len(defaults) > 0

    def test_get_scheduler_info_unknown_raises_valueerror(self):
        """get_scheduler_info raises ValueError for unknown scheduler."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            SchedulerRegistry.get_scheduler_info("nonexistent_sched_xyz")

    # ----- 3g. Custom registration contract -----

    def test_register_custom_scheduler(self, dummy_optimizer):
        """Custom scheduler class can be registered and retrieved."""

        class _TestScheduler(torch.optim.lr_scheduler.LRScheduler):
            def __init__(self, optimizer, my_param=0.5, last_epoch=-1):
                self.my_param = my_param
                super().__init__(optimizer, last_epoch)

            def get_lr(self):
                return [base_lr * self.my_param for base_lr in self.base_lrs]

        name = "_contract_test_custom_sched"
        try:
            SchedulerRegistry.register_custom_scheduler(name, _TestScheduler, {"my_param": 0.5})
            scheduler = SchedulerRegistry.get_scheduler(name, dummy_optimizer)
            assert scheduler is not None
        finally:
            SchedulerRegistry._schedulers.pop(name, None)
            SchedulerRegistry._defaults.pop(name, None)
            SchedulerRegistry._metric_based.discard(name)

    def test_register_custom_scheduler_duplicate_raises(self, dummy_optimizer):
        """Re-registering without overwrite=True raises ValueError."""

        class _TestScheduler2(torch.optim.lr_scheduler.LRScheduler):
            def __init__(self, optimizer, last_epoch=-1):
                super().__init__(optimizer, last_epoch)

            def get_lr(self):
                return self.base_lrs

        name = "_contract_test_dup_sched"
        try:
            SchedulerRegistry.register_custom_scheduler(name, _TestScheduler2)
            with pytest.raises(ValueError, match="already registered"):
                SchedulerRegistry.register_custom_scheduler(name, _TestScheduler2, overwrite=False)
        finally:
            SchedulerRegistry._schedulers.pop(name, None)
            SchedulerRegistry._defaults.pop(name, None)

    # ----- 3h. Warmup scheduler helper contract -----

    def test_create_warmup_scheduler(self, dummy_optimizer):
        """create_warmup_scheduler produces a SequentialLR."""
        scheduler = create_warmup_scheduler(
            dummy_optimizer,
            warmup_epochs=5,
            total_epochs=50,
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    # ----- 3i. Convenience functions contract -----

    def test_convenience_get_scheduler(self, dummy_optimizer):
        """Module-level get_scheduler delegates to SchedulerRegistry."""
        scheduler = get_scheduler("step_lr", dummy_optimizer, {"step_size": 10})
        assert scheduler is not None

    def test_convenience_list_schedulers(self):
        """Module-level list_schedulers delegates to SchedulerRegistry."""
        result = list_schedulers()
        assert result == SchedulerRegistry.list_available()


# ===========================================================================
# 4. METRICS REGISTRY CONTRACTS
# ===========================================================================


class TestMetricsRegistryContract:
    """Contract: MetricsRegistry exposes consistent API and all entries instantiate."""

    # ----- 4a. Listing contract -----

    def test_list_available_returns_sorted_list(self):
        """list_available() returns a sorted list of strings."""
        available = MetricsRegistry.list_available()
        assert isinstance(available, list)
        assert len(available) > 0
        assert available == sorted(available)

    def test_list_available_contains_expected_entries(self):
        """Registry must contain the documented 12 metric entries."""
        available = set(MetricsRegistry.list_available())
        expected = {
            "mse",
            "mae",
            "rmse",
            "r2",
            "mape",
            "explained_variance",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
        }
        for name in expected:
            assert name in available, f"Missing metric: {name}"

    # ----- 4b. Instantiation contract -----

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_regression_metrics_instantiate(self):
        """Every regression metric can be instantiated with no params."""
        regression = ["mse", "mae", "rmse", "r2", "mape", "explained_variance"]
        for name in regression:
            metric = MetricsRegistry.get_metric(name)
            assert isinstance(metric, nn.Module), (
                f"Metric '{name}' did not return an nn.Module instance"
            )

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_classification_metrics_instantiate_with_task_param(self):
        """Classification metrics instantiate when given the 'task' parameter."""
        classification = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
        for name in classification:
            metric = MetricsRegistry.get_metric(name, {"task": "binary"})
            assert isinstance(metric, nn.Module), (
                f"Metric '{name}' with task='binary' did not return an nn.Module"
            )

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_rmse_metric_callable(self):
        """RMSE metric produces a scalar tensor on dummy data."""
        metric = MetricsRegistry.get_metric("rmse")
        preds = torch.randn(8)
        targets = torch.randn(8)
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    # ----- 4c. Error contract -----

    def test_unknown_metric_raises_valueerror(self):
        """Requesting a non-existent metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricsRegistry.get_metric("nonexistent_metric_xyz")

    @pytest.mark.skipif(
        TORCHMETRICS_AVAILABLE, reason="Test only applies when TorchMetrics is missing"
    )
    def test_metric_without_torchmetrics_raises_runtime_error(self):
        """When TorchMetrics is unavailable, metrics that depend on it raise RuntimeError."""
        # This path is only reachable if TORCHMETRICS_AVAILABLE is False,
        # meaning the _metrics values are None for TorchMetrics-backed entries.
        with pytest.raises(RuntimeError, match="TorchMetrics"):
            MetricsRegistry.get_metric("mse")

    # ----- 4d. Parameter filtering contract -----

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_invalid_params_filtered_silently(self):
        """Invalid parameters are filtered out — no crash."""
        metric = MetricsRegistry.get_metric("mse", {"completely_invalid_param": 999})
        assert isinstance(metric, nn.Module)

    # ----- 4e. Introspection contract -----

    def test_get_metric_info_returns_dict(self):
        """get_metric_info returns a dict with expected keys."""
        for name in ("mse", "rmse"):
            info = MetricsRegistry.get_metric_info(name)
            assert isinstance(info, dict)
            assert "name" in info

    def test_get_valid_params_returns_dict(self):
        """get_valid_params returns a dict for every registered metric."""
        for name in MetricsRegistry.list_available():
            params = MetricsRegistry.get_valid_params(name)
            assert isinstance(params, dict), f"get_valid_params('{name}') did not return dict"

    def test_get_metric_info_unknown_raises_valueerror(self):
        """get_metric_info raises ValueError for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricsRegistry.get_metric_info("nonexistent_metric_xyz")

    # ----- 4f. Task-aware selection contract -----

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_get_metrics_for_task_regression(self):
        """Auto-selected metrics for regression tasks are dict of nn.Modules."""
        for task in ("graph_regression", "node_regression", "edge_regression"):
            metrics = MetricsRegistry.get_metrics_for_task(task)
            assert isinstance(metrics, dict)
            assert len(metrics) > 0
            for metric_name, metric_obj in metrics.items():
                assert isinstance(metric_obj, nn.Module), (
                    f"Metric '{metric_name}' for task '{task}' is not nn.Module"
                )

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_get_metrics_for_task_classification(self):
        """Auto-selected metrics for classification tasks are dict of nn.Modules."""
        for task in ("graph_classification", "node_classification"):
            metrics = MetricsRegistry.get_metrics_for_task(task, num_classes=2)
            assert isinstance(metrics, dict)
            assert len(metrics) > 0

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_get_metrics_for_task_link_prediction(self):
        """Auto-selected metrics for link_prediction are dict of nn.Modules."""
        metrics = MetricsRegistry.get_metrics_for_task("link_prediction")
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_get_default_metrics_for_task_returns_list(self):
        """get_default_metrics_for_task returns a list of strings."""
        for task in ("graph_regression", "graph_classification", "link_prediction"):
            defaults = MetricsRegistry.get_default_metrics_for_task(task)
            assert isinstance(defaults, list)
            assert len(defaults) > 0
            assert all(isinstance(n, str) for n in defaults)

    # ----- 4g. Compatibility check contract -----

    def test_is_metric_compatible_with_task(self):
        """is_metric_compatible_with_task reflects correct categories."""
        assert MetricsRegistry.is_metric_compatible_with_task("mse", "graph_regression") is True
        assert (
            MetricsRegistry.is_metric_compatible_with_task("mse", "graph_classification") is False
        )
        assert (
            MetricsRegistry.is_metric_compatible_with_task("accuracy", "graph_classification")
            is True
        )
        assert (
            MetricsRegistry.is_metric_compatible_with_task("accuracy", "graph_regression") is False
        )

    # ----- 4h. Custom registration contract -----

    def test_register_custom_metric(self):
        """Custom nn.Module metric can be registered and retrieved."""

        class _TestMetric(nn.Module):
            def forward(self, preds, target):
                return torch.mean(torch.abs(preds - target))

        name = "_contract_test_custom_metric"
        try:
            MetricsRegistry.register_custom_metric(
                name, _TestMetric, is_classification=False, is_regression=True
            )
            assert name in MetricsRegistry._metrics
            assert name in MetricsRegistry._regression_metrics
        finally:
            MetricsRegistry._metrics.pop(name, None)
            MetricsRegistry._regression_metrics.discard(name)
            MetricsRegistry._classification_metrics.discard(name)

    def test_register_custom_metric_duplicate_raises(self):
        """Re-registering without overwrite=True raises ValueError."""

        class _TestMetric2(nn.Module):
            def forward(self, preds, target):
                return torch.tensor(0.0)

        name = "_contract_test_dup_metric"
        try:
            MetricsRegistry.register_custom_metric(name, _TestMetric2)
            with pytest.raises(ValueError, match="already registered"):
                MetricsRegistry.register_custom_metric(name, _TestMetric2, overwrite=False)
        finally:
            MetricsRegistry._metrics.pop(name, None)
            MetricsRegistry._regression_metrics.discard(name)
            MetricsRegistry._classification_metrics.discard(name)

    def test_register_non_module_raises_typeerror(self):
        """Registering a non-Module / non-Metric class raises TypeError."""
        with pytest.raises(TypeError, match="subclass of nn.Module"):
            MetricsRegistry.register_custom_metric("_bad_metric", int)

    # ----- 4i. MetricCollection contract -----

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_create_metric_collection(self):
        """create_metric_collection returns a MetricCollection for regression.

        Note: RMSEMetric inherits from nn.Module (not torchmetrics.Metric),
        so MetricCollection rejects it. We exclude 'rmse' and use only
        pure TorchMetrics-backed entries to test the collection path.
        """
        from torchmetrics import MetricCollection as MC

        # Use only metrics that are torchmetrics.Metric subclasses
        collection = MetricsRegistry.create_metric_collection(
            "graph_regression",
            metric_names=["mse", "mae", "r2"],
        )
        assert isinstance(collection, MC)

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_create_metric_collection_with_prefix(self):
        """create_metric_collection applies prefix to metric names.

        Note: Excludes 'rmse' because RMSEMetric is an nn.Module wrapper,
        not a torchmetrics.Metric subclass, and MetricCollection requires
        all values to be torchmetrics.Metric instances.
        """
        from torchmetrics import MetricCollection as MC

        collection = MetricsRegistry.create_metric_collection(
            "graph_regression",
            metric_names=["mse", "mae", "r2"],
            prefix="val_",
        )
        assert isinstance(collection, MC)
        # All keys should start with "val_"
        for key in collection:
            assert key.startswith("val_"), f"Metric key '{key}' does not have prefix 'val_'"

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_create_metric_collection_rejects_non_torchmetric(self):
        """MetricCollection raises ValueError when given RMSEMetric (nn.Module, not Metric).

        This documents a known limitation: RMSEMetric wraps MeanSquaredError
        inside nn.Module instead of subclassing torchmetrics.Metric, so it
        cannot participate in MetricCollection. The default regression metric
        set includes 'rmse', triggering this error.
        """
        with pytest.raises(ValueError, match="is not an instance of"):
            MetricsRegistry.create_metric_collection("graph_regression")

    # ----- 4j. Convenience functions contract -----

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_convenience_get_metric(self):
        """Module-level get_metric delegates to MetricsRegistry."""
        metric = get_metric("mse")
        assert isinstance(metric, nn.Module)

    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics not installed")
    def test_convenience_get_metrics_for_task(self):
        """Module-level get_metrics_for_task delegates to MetricsRegistry."""
        metrics = get_metrics_for_task("graph_regression")
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_convenience_list_metrics(self):
        """Module-level list_metrics delegates to MetricsRegistry."""
        result = list_metrics()
        assert result == MetricsRegistry.list_available()

    def test_convenience_get_default_metrics_for_task(self):
        """Module-level get_default_metrics_for_task delegates to MetricsRegistry."""
        result = get_default_metrics_for_task("graph_regression")
        assert isinstance(result, list)

    def test_convenience_is_metric_compatible_with_task(self):
        """Module-level is_metric_compatible_with_task delegates to MetricsRegistry."""
        result = is_metric_compatible_with_task("mse", "graph_regression")
        assert result is True


# ===========================================================================
# 5. CROSS-REGISTRY CONSISTENCY CONTRACTS
# ===========================================================================


class TestCrossRegistryConsistency:
    """Cross-cutting contract: All four registries share consistent patterns."""

    def test_all_registries_have_list_method(self):
        """Every registry exposes a list method returning a sorted list."""
        registries = [
            ("LossRegistry", LossRegistry.list_available),
            ("OptimizerRegistry", OptimizerRegistry.list_available),
            ("SchedulerRegistry", SchedulerRegistry.list_available),
            ("MetricsRegistry", MetricsRegistry.list_available),
        ]
        for name, list_fn in registries:
            result = list_fn()
            assert isinstance(result, list), f"{name}.list_available() didn't return list"
            assert result == sorted(result), f"{name}.list_available() is not sorted"

    def test_all_registries_have_info_method(self):
        """Every registry exposes an info method that returns a dict."""
        test_cases = [
            ("LossRegistry", LossRegistry.get_loss_info, "mse"),
            ("OptimizerRegistry", OptimizerRegistry.get_optimizer_info, "adam"),
            ("SchedulerRegistry", SchedulerRegistry.get_scheduler_info, "step_lr"),
            ("MetricsRegistry", MetricsRegistry.get_metric_info, "mse"),
        ]
        for registry_name, info_fn, sample_entry in test_cases:
            info = info_fn(sample_entry)
            assert isinstance(info, dict), (
                f"{registry_name}.get_*_info('{sample_entry}') didn't return dict"
            )
            assert "name" in info, f"{registry_name}.get_*_info missing 'name' key"

    def test_all_registries_have_valid_params_method(self):
        """Every registry exposes get_valid_params that returns a dict."""
        test_cases = [
            ("LossRegistry", LossRegistry.get_valid_params, "mse"),
            ("OptimizerRegistry", OptimizerRegistry.get_valid_params, "adam"),
            ("SchedulerRegistry", SchedulerRegistry.get_valid_params, "step_lr"),
            ("MetricsRegistry", MetricsRegistry.get_valid_params, "mse"),
        ]
        for registry_name, params_fn, sample_entry in test_cases:
            params = params_fn(sample_entry)
            assert isinstance(params, dict), (
                f"{registry_name}.get_valid_params('{sample_entry}') didn't return dict"
            )

    def test_all_registries_have_filter_params(self):
        """Every registry has a _filter_params classmethod for param introspection."""
        for registry_cls in (LossRegistry, OptimizerRegistry, SchedulerRegistry, MetricsRegistry):
            assert hasattr(registry_cls, "_filter_params"), (
                f"{registry_cls.__name__} missing _filter_params"
            )
            assert callable(registry_cls._filter_params)

    def test_all_registries_reject_unknown_names(self):
        """All registries raise ValueError for unknown entry names."""
        bad_name = "absolutely_nonexistent_entry_xyz_123"

        with pytest.raises(ValueError):
            LossRegistry.get_loss(bad_name)

        with pytest.raises(ValueError):
            OptimizerRegistry.get_optimizer(bad_name, nn.Linear(1, 1).parameters())

        with pytest.raises(ValueError):
            SchedulerRegistry.get_scheduler(
                bad_name, torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.01)
            )

        with pytest.raises(ValueError):
            MetricsRegistry.get_metric(bad_name)

    def test_task_aware_registries_share_task_categories(self):
        """LossRegistry and MetricsRegistry agree on task type categories."""
        loss_classification = LossRegistry._classification_losses
        loss_regression = LossRegistry._regression_losses
        metric_classification = MetricsRegistry._classification_metrics
        metric_regression = MetricsRegistry._regression_metrics

        # Ensure they have entries (non-empty)
        assert len(loss_classification) > 0
        assert len(loss_regression) > 0
        assert len(metric_classification) > 0
        assert len(metric_regression) > 0

        # Classification and regression should be disjoint within each registry
        assert loss_classification.isdisjoint(loss_regression), (
            "LossRegistry: classification and regression sets overlap"
        )
        assert metric_classification.isdisjoint(metric_regression), (
            "MetricsRegistry: classification and regression sets overlap"
        )

    def test_task_default_mappings_reference_valid_entries(self):
        """Default task-to-entry mappings reference entries that exist in the registry."""
        # LossRegistry task defaults
        for task, loss_name in LossRegistry._task_to_default_loss.items():
            assert loss_name in LossRegistry._losses, (
                f"LossRegistry._task_to_default_loss['{task}'] = '{loss_name}' "
                f"but '{loss_name}' is not in the loss registry"
            )

        # MetricsRegistry task defaults
        for task, metric_names in MetricsRegistry._task_to_default_metrics.items():
            for mname in metric_names:
                assert mname in MetricsRegistry._metrics, (
                    f"MetricsRegistry._task_to_default_metrics['{task}'] references "
                    f"'{mname}' but it is not in the metrics registry"
                )
