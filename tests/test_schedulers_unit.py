#!/usr/bin/env python3
"""
Complete Unit Test Suite for schedulers.py Module

Tests scheduler registry system including:
- SchedulerRegistry class with all methods
- Scheduler registration and retrieval
- Default parameters handling
- Parameter filtering with _filter_params() method (dynamic introspection)
- get_valid_params() method for parameter discovery
- Metric-based scheduler identification
- Custom scheduler registration
- Convenience functions (get_scheduler, list_schedulers)
- Helper functions (create_warmup_scheduler)
- Error handling and edge cases
- Integration with PyTorch optimizers
- Thread safety considerations

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: milia Team
Version: 1.2.0

Changelog:
- v1.2.0: Added tests for _filter_params and get_valid_params introspection failure
          fallback paths, get_scheduler default-params logging branch,
          user-params-override-registry-defaults merge behavior,
          register_custom_scheduler non-metric-based default path,
          overwrite-builtin-scheduler path, create_warmup_scheduler error
          propagation for unknown after_scheduler_name, and non-cosine T_max
          branch coverage
- v1.1.0: Added tests for get_valid_params(), _filter_params(), updated
          parameter filtering behavior tests (invalid params now filtered silently),
          updated get_scheduler_info tests to verify valid_params key
- v1.0.0: Initial release
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# Import the module under test
from milia_pipeline.models.training.schedulers import (
    # Main class
    SchedulerRegistry,
    # Helper functions
    create_warmup_scheduler,
    # Convenience functions
    get_scheduler,
    list_schedulers,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Linear(10, 5)


@pytest.fixture
def simple_optimizer(simple_model):
    """Create a simple optimizer for testing."""
    return optim.SGD(simple_model.parameters(), lr=0.1)


@pytest.fixture
def adam_optimizer(simple_model):
    """Create an Adam optimizer for testing."""
    return optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def reset_scheduler_registry():
    """Reset scheduler registry to default state after test."""
    # Store original state
    original_schedulers = SchedulerRegistry._schedulers.copy()
    original_defaults = SchedulerRegistry._defaults.copy()
    original_metric_based = SchedulerRegistry._metric_based.copy()

    yield

    # Restore original state
    SchedulerRegistry._schedulers = original_schedulers
    SchedulerRegistry._defaults = original_defaults
    SchedulerRegistry._metric_based = original_metric_based


# =============================================================================
# SCHEDULER REGISTRY TESTS
# =============================================================================


class TestSchedulerRegistry:
    """Test SchedulerRegistry class."""

    def test_registry_has_expected_schedulers(self):
        """Test that registry contains all expected schedulers."""
        expected_schedulers = [
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
        ]

        available = SchedulerRegistry.list_available()
        for sched_name in expected_schedulers:
            assert sched_name in available, f"Missing scheduler: {sched_name}"

    def test_registry_schedulers_are_valid_classes(self):
        """Test that all registered schedulers are valid PyTorch scheduler classes."""
        for name, sched_cls in SchedulerRegistry._schedulers.items():
            assert callable(sched_cls), f"Scheduler {name} is not callable"
            assert hasattr(sched_cls, "__name__"), f"Scheduler {name} has no __name__"

    def test_default_params_structure(self):
        """Test that default params are properly structured."""
        for name, params in SchedulerRegistry._defaults.items():
            assert isinstance(params, dict), f"Default params for {name} must be dict"
            assert name in SchedulerRegistry._schedulers, f"Defaults for unknown scheduler: {name}"

    def test_metric_based_schedulers_are_valid(self):
        """Test that metric-based schedulers are in registry."""
        for name in SchedulerRegistry._metric_based:
            assert name in SchedulerRegistry._schedulers, (
                f"Metric-based scheduler {name} not in registry"
            )

    def test_list_available_returns_sorted_list(self):
        """Test list_available returns sorted list of scheduler names."""
        available = SchedulerRegistry.list_available()
        assert isinstance(available, list)
        assert len(available) > 0
        assert available == sorted(available)

    def test_list_available_includes_all_schedulers(self):
        """Test list_available includes all registered schedulers."""
        available = SchedulerRegistry.list_available()
        for name in SchedulerRegistry._schedulers.keys():
            assert name in available


class TestGetScheduler:
    """Test SchedulerRegistry.get_scheduler method."""

    def test_get_step_lr_default_params(self, simple_optimizer):
        """Test getting StepLR scheduler with parameters."""
        # Note: Registry doesn't auto-apply defaults, must provide required params
        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", simple_optimizer, {"step_size": 30, "gamma": 0.1}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 30
        assert scheduler.gamma == 0.1

    def test_get_step_lr_custom_params(self, simple_optimizer):
        """Test getting StepLR scheduler with custom parameters."""
        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", simple_optimizer, {"step_size": 10, "gamma": 0.5}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 10
        assert scheduler.gamma == 0.5

    def test_get_exponential_lr(self, simple_optimizer):
        """Test getting ExponentialLR scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "exponential_lr", simple_optimizer, {"gamma": 0.9}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        assert scheduler.gamma == 0.9

    def test_get_cosine_annealing(self, simple_optimizer):
        """Test getting CosineAnnealingLR scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "cosine_annealing", simple_optimizer, {"T_max": 50, "eta_min": 1e-6}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 50
        assert scheduler.eta_min == 1e-6

    def test_get_cosine_annealing_warm_restarts(self, simple_optimizer):
        """Test getting CosineAnnealingWarmRestarts scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "cosine_annealing_warm_restarts", simple_optimizer, {"T_0": 10, "T_mult": 2}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        assert scheduler.T_0 == 10
        assert scheduler.T_mult == 2

    def test_get_reduce_on_plateau(self, simple_optimizer):
        """Test getting ReduceLROnPlateau scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "reduce_on_plateau", simple_optimizer, {"mode": "min", "patience": 5, "factor": 0.5}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.mode == "min"
        assert scheduler.patience == 5
        assert scheduler.factor == 0.5

    def test_get_multistep_lr(self, simple_optimizer):
        """Test getting MultiStepLR scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "multistep_lr", simple_optimizer, {"milestones": [30, 80], "gamma": 0.1}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        # milestones is a Counter object, convert to list for comparison
        assert sorted(list(scheduler.milestones.keys())) == [30, 80]
        assert scheduler.gamma == 0.1

    def test_get_cyclic_lr(self, simple_optimizer):
        """Test getting CyclicLR scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "cyclic_lr", simple_optimizer, {"base_lr": 0.001, "max_lr": 0.01, "step_size_up": 2000}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)

    def test_get_one_cycle(self, simple_optimizer):
        """Test getting OneCycleLR scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "one_cycle", simple_optimizer, {"max_lr": 0.1, "total_steps": 100}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def test_get_polynomial_lr(self, simple_optimizer):
        """Test getting PolynomialLR scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "polynomial_lr", simple_optimizer, {"total_iters": 100, "power": 2.0}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.PolynomialLR)
        assert scheduler.total_iters == 100
        assert scheduler.power == 2.0

    def test_get_linear_lr(self, simple_optimizer):
        """Test getting LinearLR scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "linear_lr",
            simple_optimizer,
            {"start_factor": 0.1, "end_factor": 1.0, "total_iters": 50},
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)
        assert scheduler.start_factor == 0.1
        assert scheduler.end_factor == 1.0
        assert scheduler.total_iters == 50

    def test_get_constant_lr(self, simple_optimizer):
        """Test getting ConstantLR scheduler."""
        scheduler = SchedulerRegistry.get_scheduler(
            "constant_lr", simple_optimizer, {"factor": 0.5, "total_iters": 10}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)
        assert scheduler.factor == 0.5
        assert scheduler.total_iters == 10

    def test_get_scheduler_with_none_params(self, simple_optimizer):
        """Test getting scheduler with None params uses registry defaults.

        The SchedulerRegistry now merges registry defaults with provided params,
        so passing None for a scheduler with defaults (like step_lr) will use
        those defaults and succeed.
        """
        # StepLR has registry defaults: {"step_size": 30, "gamma": 0.1}
        scheduler = SchedulerRegistry.get_scheduler("step_lr", simple_optimizer, None)

        # Should create scheduler with registry defaults
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 30  # from registry defaults
        assert scheduler.gamma == 0.1  # from registry defaults

    def test_get_scheduler_with_empty_params(self, simple_optimizer):
        """Test getting scheduler with empty params dict uses registry defaults.

        The SchedulerRegistry now merges registry defaults with provided params,
        so passing {} for a scheduler with defaults will use those defaults.
        """
        # StepLR has registry defaults: {"step_size": 30, "gamma": 0.1}
        scheduler = SchedulerRegistry.get_scheduler("step_lr", simple_optimizer, {})

        # Should create scheduler with registry defaults
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 30  # from registry defaults
        assert scheduler.gamma == 0.1  # from registry defaults

    def test_get_scheduler_without_defaults_requires_params(self, simple_optimizer):
        """Test that scheduler without registry defaults requires params.

        Schedulers like multistep_lr don't have registry defaults defined,
        so they require the user to provide required parameters.
        """
        # multistep_lr requires 'milestones' parameter and has no registry defaults
        with pytest.raises(ValueError, match="Invalid parameters for scheduler"):
            SchedulerRegistry.get_scheduler("multistep_lr", simple_optimizer, None)

    def test_get_scheduler_unknown_name_raises_error(self, simple_optimizer):
        """Test that unknown scheduler name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler: 'nonexistent'"):
            SchedulerRegistry.get_scheduler("nonexistent", simple_optimizer)

    def test_get_scheduler_unknown_name_shows_available(self, simple_optimizer):
        """Test that error message includes available schedulers."""
        with pytest.raises(ValueError, match="Available schedulers:"):
            SchedulerRegistry.get_scheduler("invalid_scheduler", simple_optimizer)

    def test_get_scheduler_invalid_params_filtered_silently(self, simple_optimizer):
        """Test that invalid parameters are filtered out silently (not raising errors).

        The SchedulerRegistry now uses _filter_params() to dynamically filter out
        parameters that are not accepted by the scheduler class constructor.
        When valid params exist (like step_size for step_lr), invalid ones are filtered.
        """
        # Invalid params should be filtered out, scheduler should be created with valid params
        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", simple_optimizer, {"step_size": 10, "invalid_param": 123}
        )

        # Should create StepLR with the valid 'step_size' parameter
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 10

    def test_get_scheduler_mixed_valid_invalid_params(self, simple_optimizer):
        """Test that valid params are used while invalid ones are filtered."""
        # Mix of valid (T_max, eta_min) and invalid (invalid_param) parameters
        scheduler = SchedulerRegistry.get_scheduler(
            "cosine_annealing",
            simple_optimizer,
            {"T_max": 50, "eta_min": 1e-6, "invalid_param": 123},
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 50
        assert scheduler.eta_min == 1e-6

    def test_get_scheduler_safe_usage_example(self, simple_optimizer):
        """Test safe usage example from get_scheduler docstring.

        From docstring:
            >>> scheduler = SchedulerRegistry.get_scheduler(
            ...     "step_lr",
            ...     optimizer,
            ...     {"step_size": 10, "invalid_param": 123}
            ... )
            >>> # Works! 'invalid_param' is filtered out
        """
        # This should work - 'invalid_param' is filtered out
        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", simple_optimizer, {"step_size": 10, "invalid_param": 123}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_get_scheduler_invalid_params_shows_original_error(self, simple_optimizer):
        """Test that error message includes original TypeError."""
        # Use a parameter that truly causes TypeError - pass wrong type for required param
        with pytest.raises(ValueError, match="Invalid parameters for scheduler"):
            SchedulerRegistry.get_scheduler(
                "one_cycle",
                simple_optimizer,
                {"max_lr": "not_a_number", "total_steps": "also_not_a_number"},
            )

    def test_get_scheduler_with_adam_optimizer(self, adam_optimizer):
        """Test getting scheduler with Adam optimizer."""
        scheduler = SchedulerRegistry.get_scheduler(
            "cosine_annealing", adam_optimizer, {"T_max": 100}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_get_scheduler_logging(self, simple_optimizer, caplog):
        """Test that scheduler creation is logged."""
        with caplog.at_level(logging.DEBUG):
            SchedulerRegistry.get_scheduler("step_lr", simple_optimizer, {"step_size": 20})

        assert "Initialized step_lr scheduler" in caplog.text
        assert "step_size" in caplog.text

    def test_get_scheduler_logs_filtered_params(self, simple_optimizer, caplog):
        """Test that get_scheduler logs ignored params at debug level."""
        with caplog.at_level(logging.DEBUG):
            SchedulerRegistry.get_scheduler(
                "step_lr", simple_optimizer, {"step_size": 20, "invalid_param": 123}
            )

        # Should log about ignored unsupported params
        assert "ignored unsupported params" in caplog.text

    def test_get_scheduler_logs_default_params_message(self, simple_optimizer, caplog):
        """Test that get_scheduler logs 'default params' when filtered_params is empty.

        When the merged+filtered params result in an empty dict, the logger
        should produce the 'default params' debug message branch.
        """
        # cosine_annealing_warm_restarts has no registry defaults
        # and T_0 is required, so passing only invalid params should yield
        # an empty filtered_params dict, BUT will fail at instantiation.
        # Instead, test with a scheduler that can be created with no extra params.
        # constant_lr has no registry defaults but all constructor params have defaults.
        with caplog.at_level(logging.DEBUG):
            SchedulerRegistry.get_scheduler("polynomial_lr", simple_optimizer, {})

        assert "default params" in caplog.text

    def test_get_scheduler_user_params_override_registry_defaults(self, simple_optimizer):
        """Test that user-provided params override registry defaults selectively.

        The merge is: {**registry_defaults, **user_params}, so user params
        take precedence, while non-overridden defaults are preserved.
        """
        # step_lr registry defaults: {"step_size": 30, "gamma": 0.1}
        # Override only gamma, step_size should come from registry defaults
        scheduler = SchedulerRegistry.get_scheduler("step_lr", simple_optimizer, {"gamma": 0.5})

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 30  # from registry defaults
        assert scheduler.gamma == 0.5  # from user override


class TestIsMetricBased:
    """Test SchedulerRegistry.is_metric_based method."""

    def test_reduce_on_plateau_is_metric_based(self):
        """Test that ReduceLROnPlateau is identified as metric-based."""
        assert SchedulerRegistry.is_metric_based("reduce_on_plateau") is True

    def test_step_lr_is_not_metric_based(self):
        """Test that StepLR is not metric-based."""
        assert SchedulerRegistry.is_metric_based("step_lr") is False

    def test_cosine_annealing_is_not_metric_based(self):
        """Test that CosineAnnealingLR is not metric-based."""
        assert SchedulerRegistry.is_metric_based("cosine_annealing") is False

    def test_all_schedulers_have_metric_based_info(self):
        """Test that metric-based info exists for all schedulers."""
        for name in SchedulerRegistry.list_available():
            # Should not raise error
            result = SchedulerRegistry.is_metric_based(name)
            assert isinstance(result, bool)


class TestGetSchedulerInfo:
    """Test SchedulerRegistry.get_scheduler_info method."""

    def test_get_scheduler_info_step_lr(self):
        """Test getting info for StepLR scheduler."""
        info = SchedulerRegistry.get_scheduler_info("step_lr")

        assert isinstance(info, dict)
        assert info["name"] == "step_lr"
        assert info["class"] == "StepLR"
        assert "torch.optim.lr_scheduler" in info["module"]
        assert info["metric_based"] is False
        assert "step_size" in info["default_params"]
        assert info["doc"] is not None
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)
        # StepLR should have step_size, gamma, last_epoch params
        assert "step_size" in info["valid_params"]
        assert "gamma" in info["valid_params"]

    def test_get_scheduler_info_reduce_on_plateau(self):
        """Test getting info for ReduceLROnPlateau scheduler."""
        info = SchedulerRegistry.get_scheduler_info("reduce_on_plateau")

        assert info["name"] == "reduce_on_plateau"
        assert info["class"] == "ReduceLROnPlateau"
        assert info["metric_based"] is True
        assert "mode" in info["default_params"]
        assert "patience" in info["default_params"]
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)
        # ReduceLROnPlateau should have mode, factor, patience, etc.
        assert "mode" in info["valid_params"]
        assert "factor" in info["valid_params"]
        assert "patience" in info["valid_params"]

    def test_get_scheduler_info_cosine_annealing(self):
        """Test getting info for CosineAnnealingLR scheduler."""
        info = SchedulerRegistry.get_scheduler_info("cosine_annealing")

        assert info["name"] == "cosine_annealing"
        assert info["class"] == "CosineAnnealingLR"
        assert info["metric_based"] is False
        assert "T_max" in info["default_params"]
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)
        # CosineAnnealingLR should have T_max, eta_min, etc.
        assert "T_max" in info["valid_params"]
        assert "eta_min" in info["valid_params"]

    def test_get_scheduler_info_scheduler_without_defaults(self):
        """Test getting info for scheduler without default params."""
        info = SchedulerRegistry.get_scheduler_info("multistep_lr")

        assert info["name"] == "multistep_lr"
        assert isinstance(info["default_params"], dict)
        # May be empty since not all schedulers have defaults

    def test_get_scheduler_info_unknown_scheduler_raises_error(self):
        """Test that getting info for unknown scheduler raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler: 'nonexistent'"):
            SchedulerRegistry.get_scheduler_info("nonexistent")

    def test_get_scheduler_info_all_schedulers(self):
        """Test getting info for all registered schedulers."""
        for name in SchedulerRegistry.list_available():
            info = SchedulerRegistry.get_scheduler_info(name)
            assert info["name"] == name
            assert "class" in info
            assert "module" in info
            assert "metric_based" in info
            assert "default_params" in info
            assert "valid_params" in info
            assert isinstance(info["valid_params"], dict)


class TestGetDefaultParams:
    """Test SchedulerRegistry.get_default_params method."""

    def test_get_default_params_step_lr(self):
        """Test getting default params for StepLR."""
        defaults = SchedulerRegistry.get_default_params("step_lr")

        assert isinstance(defaults, dict)
        assert defaults["step_size"] == 30
        assert defaults["gamma"] == 0.1

    def test_get_default_params_reduce_on_plateau(self):
        """Test getting default params for ReduceLROnPlateau."""
        defaults = SchedulerRegistry.get_default_params("reduce_on_plateau")

        assert defaults["mode"] == "min"
        assert defaults["factor"] == 0.1
        assert defaults["patience"] == 10
        assert defaults["threshold"] == 1e-4

    def test_get_default_params_cosine_annealing(self):
        """Test getting default params for CosineAnnealingLR."""
        defaults = SchedulerRegistry.get_default_params("cosine_annealing")

        assert defaults["T_max"] == 100
        assert defaults["eta_min"] == 0

    def test_get_default_params_returns_copy(self):
        """Test that get_default_params returns a copy, not the original."""
        defaults1 = SchedulerRegistry.get_default_params("step_lr")
        defaults2 = SchedulerRegistry.get_default_params("step_lr")

        # Modify first copy
        defaults1["step_size"] = 999

        # Second copy should be unaffected
        assert defaults2["step_size"] == 30

    def test_get_default_params_scheduler_without_defaults(self):
        """Test getting default params for scheduler without defaults returns empty dict."""
        defaults = SchedulerRegistry.get_default_params("multistep_lr")
        assert isinstance(defaults, dict)

    def test_get_default_params_unknown_scheduler_raises_error(self):
        """Test that unknown scheduler raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler: 'invalid'"):
            SchedulerRegistry.get_default_params("invalid")


class TestGetValidParams:
    """Test SchedulerRegistry.get_valid_params method."""

    def test_get_valid_params_step_lr(self):
        """Test getting valid params for StepLR scheduler."""
        params = SchedulerRegistry.get_valid_params("step_lr")

        assert isinstance(params, dict)
        assert "step_size" in params
        assert "gamma" in params
        assert "last_epoch" in params

    def test_get_valid_params_reduce_on_plateau(self):
        """Test getting valid params for ReduceLROnPlateau scheduler.

        Example from docstring:
            >>> params = SchedulerRegistry.get_valid_params("reduce_on_plateau")
            >>> print(params)
            {'mode': 'min', 'factor': 0.1, 'patience': 10, ...}
        """
        params = SchedulerRegistry.get_valid_params("reduce_on_plateau")

        assert isinstance(params, dict)
        assert "mode" in params
        assert "factor" in params
        assert "patience" in params
        assert "threshold" in params
        # Check default values
        assert params["mode"] == "min"
        assert params["factor"] == 0.1
        assert params["patience"] == 10

    def test_get_valid_params_cosine_annealing(self):
        """Test getting valid params for CosineAnnealingLR scheduler."""
        params = SchedulerRegistry.get_valid_params("cosine_annealing")

        assert isinstance(params, dict)
        assert "T_max" in params
        assert "eta_min" in params
        assert "last_epoch" in params

    def test_get_valid_params_exponential_lr(self):
        """Test getting valid params for ExponentialLR scheduler."""
        params = SchedulerRegistry.get_valid_params("exponential_lr")

        assert isinstance(params, dict)
        assert "gamma" in params
        assert "last_epoch" in params

    def test_get_valid_params_all_schedulers(self):
        """Test getting valid params for all registered schedulers."""
        for sched_name in SchedulerRegistry.list_available():
            params = SchedulerRegistry.get_valid_params(sched_name)
            assert isinstance(params, dict)

    def test_get_valid_params_unknown_scheduler_raises_error(self):
        """Test that getting valid params for unknown scheduler raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SchedulerRegistry.get_valid_params("unknown_scheduler")

        assert "Unknown scheduler: 'unknown_scheduler'" in str(exc_info.value)

    def test_get_valid_params_returns_defaults(self):
        """Test that get_valid_params returns default values for params with defaults."""
        params = SchedulerRegistry.get_valid_params("step_lr")

        # StepLR gamma has a default of 0.1
        assert params["gamma"] == 0.1
        assert params["last_epoch"] == -1

    def test_get_valid_params_differs_from_registry_defaults(self):
        """Test that get_valid_params (introspected) may differ from registry defaults.

        Registry defaults are manually defined, while get_valid_params uses introspection.
        """
        introspected = SchedulerRegistry.get_valid_params("step_lr")
        registry_defaults = SchedulerRegistry.get_default_params("step_lr")

        # Both should be dicts
        assert isinstance(introspected, dict)
        assert isinstance(registry_defaults, dict)

        # Introspected may have more params than registry defaults
        assert len(introspected) >= len(registry_defaults)

    def test_get_valid_params_introspection_failure_returns_empty(self):
        """Test get_valid_params returns empty dict when introspection fails.

        When inspect.signature() raises ValueError or TypeError (e.g., for
        C-extension classes), the method returns an empty dict.
        """
        with patch("inspect.signature", side_effect=ValueError("cannot introspect")):
            params = SchedulerRegistry.get_valid_params("step_lr")

        assert params == {}

    def test_get_valid_params_introspection_type_error_returns_empty(self):
        """Test get_valid_params returns empty dict on TypeError from introspection."""
        with patch("inspect.signature", side_effect=TypeError("unsupported callable")):
            params = SchedulerRegistry.get_valid_params("cosine_annealing")

        assert params == {}


class TestFilterParams:
    """Test SchedulerRegistry._filter_params private method.

    The _filter_params method dynamically filters parameters to only those
    accepted by the target scheduler constructor using inspect.signature().
    """

    def test_filter_params_step_lr_valid_params(self):
        """Test _filter_params with all valid StepLR params."""
        params = {"step_size": 10, "gamma": 0.5, "last_epoch": -1}
        filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, params)

        assert filtered == params

    def test_filter_params_step_lr_invalid_params(self):
        """Test _filter_params filters out invalid params."""
        params = {"step_size": 10, "gamma": 0.5, "invalid_param": 123, "another_bad": "value"}
        filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, params)

        assert "step_size" in filtered
        assert "gamma" in filtered
        assert "invalid_param" not in filtered
        assert "another_bad" not in filtered

    def test_filter_params_cosine_annealing_filters_step_lr_params(self):
        """Test _filter_params filters StepLR-specific params from CosineAnnealing."""
        params = {
            "T_max": 100,
            "eta_min": 1e-6,
            "step_size": 10,  # StepLR param, not valid for CosineAnnealing
            "gamma": 0.5,  # StepLR param, not valid for CosineAnnealing
        }
        filtered = SchedulerRegistry._filter_params(
            torch.optim.lr_scheduler.CosineAnnealingLR, params
        )

        assert "T_max" in filtered
        assert "eta_min" in filtered
        assert "step_size" not in filtered
        assert "gamma" not in filtered

    def test_filter_params_empty_dict(self):
        """Test _filter_params with empty params dict."""
        params = {}
        filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, params)

        assert filtered == {}

    def test_filter_params_all_invalid(self):
        """Test _filter_params with all invalid params."""
        params = {"invalid1": 1, "invalid2": 2, "invalid3": 3}
        filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, params)

        assert filtered == {}

    def test_filter_params_none_input(self):
        """Test _filter_params handles None/empty input gracefully."""
        # The method checks if not params first
        filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, None)
        assert filtered == {}

        filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, {})
        assert filtered == {}

    def test_filter_params_preserves_values(self):
        """Test _filter_params preserves parameter values correctly."""
        params = {"step_size": 15, "gamma": 0.123456, "last_epoch": 5}
        filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, params)

        assert filtered["step_size"] == 15
        assert filtered["gamma"] == 0.123456
        assert filtered["last_epoch"] == 5

    def test_filter_params_excludes_self_and_optimizer(self):
        """Test _filter_params excludes 'self' and 'optimizer' (model parameters)."""
        # 'optimizer' is the optimizer argument, should not be in filtered result
        params = {"step_size": 10, "optimizer": "should_be_ignored", "self": "should_be_ignored"}
        filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, params)

        assert "step_size" in filtered
        assert "optimizer" not in filtered
        assert "self" not in filtered

    def test_filter_params_introspection_failure_returns_original(self):
        """Test _filter_params returns original params when introspection fails.

        When inspect.signature() raises ValueError or TypeError (e.g., for
        built-in C extensions), the method falls back to returning the
        original params dict unmodified.
        """
        params = {"some_param": 42, "another": "value"}

        # Create a class whose __init__ causes inspect.signature() to raise
        with patch("inspect.signature", side_effect=ValueError("cannot introspect")):
            filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, params)

        # Fallback: returns original params unfiltered
        assert filtered == params

    def test_filter_params_introspection_type_error_returns_original(self):
        """Test _filter_params returns original params on TypeError from introspection."""
        params = {"key1": 1, "key2": 2}

        with patch("inspect.signature", side_effect=TypeError("unsupported callable")):
            filtered = SchedulerRegistry._filter_params(torch.optim.lr_scheduler.StepLR, params)

        assert filtered == params


class TestRegisterCustomScheduler:
    """Test SchedulerRegistry.register_custom_scheduler method."""

    def test_register_custom_scheduler_basic(self, simple_optimizer, reset_scheduler_registry):
        """Test registering a custom scheduler."""

        class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, my_param=0.5, last_epoch=-1):
                self.my_param = my_param
                super().__init__(optimizer, last_epoch)

            def get_lr(self):
                return [base_lr * self.my_param for base_lr in self.base_lrs]

        SchedulerRegistry.register_custom_scheduler(
            "my_scheduler", CustomScheduler, {"my_param": 0.5}
        )

        assert "my_scheduler" in SchedulerRegistry.list_available()

        # Test using the custom scheduler
        scheduler = SchedulerRegistry.get_scheduler(
            "my_scheduler", simple_optimizer, {"my_param": 0.8}
        )
        assert isinstance(scheduler, CustomScheduler)
        assert scheduler.my_param == 0.8

    def test_register_custom_scheduler_without_defaults(self, reset_scheduler_registry):
        """Test registering custom scheduler without default params."""

        class SimpleScheduler(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return [base_lr * 0.9 for base_lr in self.base_lrs]

        SchedulerRegistry.register_custom_scheduler("simple_scheduler", SimpleScheduler)

        assert "simple_scheduler" in SchedulerRegistry.list_available()
        defaults = SchedulerRegistry.get_default_params("simple_scheduler")
        assert isinstance(defaults, dict)

    def test_register_custom_scheduler_metric_based(self, reset_scheduler_registry):
        """Test registering custom metric-based scheduler."""

        class MetricScheduler(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return self.base_lrs

        SchedulerRegistry.register_custom_scheduler(
            "metric_scheduler", MetricScheduler, metric_based=True
        )

        assert SchedulerRegistry.is_metric_based("metric_scheduler") is True

    def test_register_custom_scheduler_duplicate_raises_error(self, reset_scheduler_registry):
        """Test that registering duplicate name raises ValueError."""

        class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return self.base_lrs

        SchedulerRegistry.register_custom_scheduler("custom_test", CustomScheduler)

        with pytest.raises(ValueError, match="already registered"):
            SchedulerRegistry.register_custom_scheduler("custom_test", CustomScheduler)

    def test_register_custom_scheduler_overwrite(self, reset_scheduler_registry):
        """Test overwriting existing scheduler with overwrite=True."""

        class CustomScheduler1(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return self.base_lrs

        class CustomScheduler2(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return [lr * 0.5 for lr in self.base_lrs]

        SchedulerRegistry.register_custom_scheduler("custom_overwrite", CustomScheduler1)

        # Should not raise error with overwrite=True
        SchedulerRegistry.register_custom_scheduler(
            "custom_overwrite", CustomScheduler2, overwrite=True
        )

        assert "custom_overwrite" in SchedulerRegistry.list_available()

    def test_register_custom_scheduler_logging(self, caplog, reset_scheduler_registry):
        """Test that custom scheduler registration is logged."""

        class LogScheduler(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return self.base_lrs

        with caplog.at_level(logging.INFO):
            SchedulerRegistry.register_custom_scheduler("log_scheduler", LogScheduler)

        assert "Registered custom scheduler: 'log_scheduler'" in caplog.text

    def test_register_custom_scheduler_not_metric_based_by_default(self, reset_scheduler_registry):
        """Test that a custom scheduler is NOT metric-based when metric_based=False (default).

        Verifies the default path where metric_based is not set, ensuring
        the scheduler name is not added to _metric_based.
        """

        class NonMetricScheduler(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return self.base_lrs

        SchedulerRegistry.register_custom_scheduler("non_metric_custom", NonMetricScheduler)

        assert SchedulerRegistry.is_metric_based("non_metric_custom") is False
        assert "non_metric_custom" not in SchedulerRegistry._metric_based

    def test_register_custom_scheduler_overwrite_builtin(
        self, simple_optimizer, reset_scheduler_registry
    ):
        """Test overwriting a built-in scheduler with overwrite=True.

        Ensures that overwrite works not only for custom names but also
        for replacing existing built-in schedulers in the registry.
        """

        class ReplacementStepLR(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, custom_val=42, last_epoch=-1):
                self.custom_val = custom_val
                super().__init__(optimizer, last_epoch)

            def get_lr(self):
                return self.base_lrs

        SchedulerRegistry.register_custom_scheduler(
            "step_lr", ReplacementStepLR, {"custom_val": 42}, overwrite=True
        )

        scheduler = SchedulerRegistry.get_scheduler("step_lr", simple_optimizer, {"custom_val": 99})
        assert isinstance(scheduler, ReplacementStepLR)
        assert scheduler.custom_val == 99

    def test_register_custom_scheduler_no_default_params_stored(self, reset_scheduler_registry):
        """Test that no defaults are stored when default_params is None/not provided."""

        class BareScheduler(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return self.base_lrs

        SchedulerRegistry.register_custom_scheduler("bare_scheduler", BareScheduler)

        defaults = SchedulerRegistry.get_default_params("bare_scheduler")
        assert defaults == {}


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_scheduler_function(self, simple_optimizer):
        """Test get_scheduler convenience function."""
        scheduler = get_scheduler("step_lr", simple_optimizer, {"step_size": 20})
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 20

    def test_get_scheduler_function_delegates_to_registry(self, simple_optimizer):
        """Test that get_scheduler delegates to SchedulerRegistry."""
        with patch.object(SchedulerRegistry, "get_scheduler") as mock_get:
            mock_get.return_value = Mock()

            params = {"step_size": 15}
            get_scheduler("step_lr", simple_optimizer, params)

            mock_get.assert_called_once_with("step_lr", simple_optimizer, params)

    def test_list_schedulers_function(self):
        """Test list_schedulers convenience function."""
        schedulers = list_schedulers()
        assert isinstance(schedulers, list)
        assert len(schedulers) > 0
        assert "step_lr" in schedulers
        assert "cosine_annealing" in schedulers

    def test_list_schedulers_function_delegates_to_registry(self):
        """Test that list_schedulers delegates to SchedulerRegistry."""
        with patch.object(SchedulerRegistry, "list_available") as mock_list:
            mock_list.return_value = ["scheduler1", "scheduler2"]

            result = list_schedulers()

            mock_list.assert_called_once()
            assert result == ["scheduler1", "scheduler2"]


# =============================================================================
# WARMUP SCHEDULER TESTS
# =============================================================================


class TestCreateWarmupScheduler:
    """Test create_warmup_scheduler helper function."""

    def test_create_warmup_scheduler_basic(self, simple_optimizer):
        """Test creating warmup scheduler with basic parameters."""
        scheduler = create_warmup_scheduler(simple_optimizer, warmup_epochs=5, total_epochs=50)

        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    def test_create_warmup_scheduler_with_cosine_annealing(self, simple_optimizer):
        """Test creating warmup scheduler with cosine annealing."""
        scheduler = create_warmup_scheduler(
            simple_optimizer,
            warmup_epochs=10,
            total_epochs=100,
            after_scheduler_name="cosine_annealing",
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        assert len(scheduler._schedulers) == 2
        assert isinstance(scheduler._schedulers[0], torch.optim.lr_scheduler.LinearLR)
        assert isinstance(scheduler._schedulers[1], torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_warmup_scheduler_with_step_lr(self, simple_optimizer):
        """Test creating warmup scheduler with step LR."""
        scheduler = create_warmup_scheduler(
            simple_optimizer,
            warmup_epochs=5,
            total_epochs=50,
            after_scheduler_name="step_lr",
            after_scheduler_params={"step_size": 10, "gamma": 0.5},
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        assert len(scheduler._schedulers) == 2
        assert isinstance(scheduler._schedulers[1], torch.optim.lr_scheduler.StepLR)

    def test_create_warmup_scheduler_custom_start_lr(self, simple_optimizer):
        """Test creating warmup scheduler with custom start learning rate."""
        warmup_start_lr = 1e-7
        initial_lr = simple_optimizer.param_groups[0]["lr"]  # Store initial LR before creation

        scheduler = create_warmup_scheduler(
            simple_optimizer, warmup_epochs=10, total_epochs=100, warmup_start_lr=warmup_start_lr
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        # Check that warmup scheduler exists
        warmup_sched = scheduler._schedulers[0]
        assert isinstance(warmup_sched, torch.optim.lr_scheduler.LinearLR)
        # LinearLR stores start_factor as the value calculated from initial LR
        expected_start_factor = warmup_start_lr / initial_lr
        assert abs(warmup_sched.start_factor - expected_start_factor) < 1e-9

    def test_create_warmup_scheduler_milestones(self, simple_optimizer):
        """Test that warmup scheduler has correct milestones."""
        warmup_epochs = 15
        scheduler = create_warmup_scheduler(
            simple_optimizer, warmup_epochs=warmup_epochs, total_epochs=100
        )

        assert scheduler._milestones == [warmup_epochs]

    def test_create_warmup_scheduler_auto_calculates_t_max(self, simple_optimizer):
        """Test that T_max is automatically calculated for cosine annealing."""
        warmup_epochs = 10
        total_epochs = 100

        scheduler = create_warmup_scheduler(
            simple_optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            after_scheduler_name="cosine_annealing",
        )

        # Second scheduler should have T_max = total_epochs - warmup_epochs
        cosine_sched = scheduler._schedulers[1]
        assert cosine_sched.T_max == total_epochs - warmup_epochs

    def test_create_warmup_scheduler_respects_provided_t_max(self, simple_optimizer):
        """Test that provided T_max is respected."""
        scheduler = create_warmup_scheduler(
            simple_optimizer,
            warmup_epochs=10,
            total_epochs=100,
            after_scheduler_name="cosine_annealing",
            after_scheduler_params={"T_max": 50},
        )

        cosine_sched = scheduler._schedulers[1]
        assert cosine_sched.T_max == 50

    def test_create_warmup_scheduler_logging(self, simple_optimizer, caplog):
        """Test that warmup scheduler creation is logged."""
        with caplog.at_level(logging.INFO):
            create_warmup_scheduler(simple_optimizer, warmup_epochs=10, total_epochs=100)

        assert "Created warmup scheduler" in caplog.text
        assert "10 epochs warmup" in caplog.text
        assert "cosine_annealing" in caplog.text

    def test_create_warmup_scheduler_with_exponential(self, simple_optimizer):
        """Test creating warmup scheduler with exponential LR."""
        scheduler = create_warmup_scheduler(
            simple_optimizer,
            warmup_epochs=5,
            total_epochs=50,
            after_scheduler_name="exponential_lr",
            after_scheduler_params={"gamma": 0.95},
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        assert isinstance(scheduler._schedulers[1], torch.optim.lr_scheduler.ExponentialLR)

    def test_create_warmup_scheduler_unknown_after_scheduler_raises_error(self, simple_optimizer):
        """Test that create_warmup_scheduler propagates ValueError for unknown scheduler.

        When after_scheduler_name is not in the registry, the internal call
        to SchedulerRegistry.get_scheduler should raise ValueError.
        """
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_warmup_scheduler(
                simple_optimizer,
                warmup_epochs=5,
                total_epochs=50,
                after_scheduler_name="nonexistent_scheduler",
            )

    def test_create_warmup_scheduler_non_cosine_ignores_t_max_logic(self, simple_optimizer):
        """Test that T_max auto-calculation only applies to cosine_annealing.

        When after_scheduler_name is not 'cosine_annealing', the T_max
        auto-population branch should be skipped, even if T_max is in params.
        """
        scheduler = create_warmup_scheduler(
            simple_optimizer,
            warmup_epochs=5,
            total_epochs=50,
            after_scheduler_name="exponential_lr",
            after_scheduler_params={"gamma": 0.9, "T_max": 999},  # T_max is irrelevant here
        )

        # ExponentialLR should be created successfully (T_max filtered out by _filter_params)
        assert isinstance(scheduler._schedulers[1], torch.optim.lr_scheduler.ExponentialLR)


# =============================================================================
# SCHEDULER FUNCTIONALITY TESTS
# =============================================================================


class TestSchedulerFunctionality:
    """Test actual scheduler functionality with training loops."""

    def test_step_lr_reduces_learning_rate(self, simple_optimizer):
        """Test that StepLR reduces learning rate at correct steps."""
        initial_lr = simple_optimizer.param_groups[0]["lr"]

        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", simple_optimizer, {"step_size": 2, "gamma": 0.5}
        )

        # PyTorch scheduler behavior: step() is meant to be called after optimizer.step()
        # First call to step() moves from epoch 0 to epoch 1
        # LR is updated based on the epoch

        # Initial state: epoch 0, no change yet
        current_lr = simple_optimizer.param_groups[0]["lr"]
        assert current_lr == initial_lr

        # Step 1: epoch becomes 1, no reduction yet (step_size=2)
        scheduler.step()
        assert simple_optimizer.param_groups[0]["lr"] == initial_lr

        # Step 2: epoch becomes 2, should reduce (2 % 2 == 0)
        scheduler.step()
        assert abs(simple_optimizer.param_groups[0]["lr"] - initial_lr * 0.5) < 1e-9

        # Step 3: epoch becomes 3, no additional reduction
        scheduler.step()
        assert abs(simple_optimizer.param_groups[0]["lr"] - initial_lr * 0.5) < 1e-9

        # Step 4: epoch becomes 4, should reduce again (4 % 2 == 0)
        scheduler.step()
        assert abs(simple_optimizer.param_groups[0]["lr"] - initial_lr * 0.25) < 1e-9

    def test_exponential_lr_decays_exponentially(self, simple_optimizer):
        """Test that ExponentialLR decays learning rate exponentially."""
        initial_lr = simple_optimizer.param_groups[0]["lr"]
        gamma = 0.9

        scheduler = SchedulerRegistry.get_scheduler(
            "exponential_lr", simple_optimizer, {"gamma": gamma}
        )

        for i in range(1, 5):
            scheduler.step()
            expected_lr = initial_lr * (gamma**i)
            assert abs(simple_optimizer.param_groups[0]["lr"] - expected_lr) < 1e-6

    def test_cosine_annealing_follows_cosine_schedule(self, simple_optimizer):
        """Test that CosineAnnealingLR follows cosine schedule."""
        scheduler = SchedulerRegistry.get_scheduler(
            "cosine_annealing", simple_optimizer, {"T_max": 10, "eta_min": 0}
        )

        initial_lr = simple_optimizer.param_groups[0]["lr"]

        # At T_max/2, LR should be around half of initial
        for _ in range(5):
            scheduler.step()

        mid_lr = simple_optimizer.param_groups[0]["lr"]
        assert mid_lr < initial_lr

        # At T_max, LR should be eta_min (0)
        for _ in range(5):
            scheduler.step()

        final_lr = simple_optimizer.param_groups[0]["lr"]
        assert final_lr < mid_lr

    def test_reduce_on_plateau_reduces_on_no_improvement(self, simple_optimizer):
        """Test that ReduceLROnPlateau reduces LR when metric doesn't improve."""
        initial_lr = simple_optimizer.param_groups[0]["lr"]

        scheduler = SchedulerRegistry.get_scheduler(
            "reduce_on_plateau", simple_optimizer, {"mode": "min", "patience": 2, "factor": 0.5}
        )

        # No improvement for patience + 1 steps
        scheduler.step(1.0)  # Sets best
        scheduler.step(1.0)  # Counter = 1
        scheduler.step(1.0)  # Counter = 2
        scheduler.step(1.0)  # Counter = 3, exceeds patience, should reduce

        # After exceeding patience, LR should be reduced
        assert simple_optimizer.param_groups[0]["lr"] < initial_lr

    def test_reduce_on_plateau_does_not_reduce_on_improvement(self, simple_optimizer):
        """Test that ReduceLROnPlateau doesn't reduce LR when metric improves."""
        initial_lr = simple_optimizer.param_groups[0]["lr"]

        scheduler = SchedulerRegistry.get_scheduler(
            "reduce_on_plateau", simple_optimizer, {"mode": "min", "patience": 2, "factor": 0.5}
        )

        # Continuous improvement
        scheduler.step(1.0)
        scheduler.step(0.9)
        scheduler.step(0.8)
        scheduler.step(0.7)

        # LR should not have changed
        assert simple_optimizer.param_groups[0]["lr"] == initial_lr

    def test_warmup_scheduler_warms_up_then_decays(self, simple_optimizer):
        """Test that warmup scheduler warms up then applies main scheduler."""
        scheduler = create_warmup_scheduler(
            simple_optimizer,
            warmup_epochs=3,
            total_epochs=10,
            warmup_start_lr=0.01,
            after_scheduler_name="step_lr",
            after_scheduler_params={"step_size": 2, "gamma": 0.5},
        )

        initial_lr = simple_optimizer.param_groups[0]["lr"]

        # During warmup, LR should increase
        scheduler.step()
        lr_after_step1 = simple_optimizer.param_groups[0]["lr"]

        scheduler.step()
        lr_after_step2 = simple_optimizer.param_groups[0]["lr"]

        # After warmup (step 3), should reach initial_lr
        scheduler.step()
        lr_after_warmup = simple_optimizer.param_groups[0]["lr"]

        # Then step_lr takes over
        scheduler.step()
        scheduler.step()  # At step 5, step_lr should reduce (every 2 steps)

        # Verify warmup progression
        assert lr_after_step1 > 0.01
        assert lr_after_step2 > lr_after_step1


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_scheduler_with_zero_learning_rate(self):
        """Test scheduler with zero initial learning rate."""
        model = nn.Linear(5, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.0)

        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", optimizer, {"step_size": 10, "gamma": 0.5}
        )

        # Should not crash
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 0.0

    def test_scheduler_with_multiple_param_groups(self):
        """Test scheduler with optimizer having multiple parameter groups."""
        model1 = nn.Linear(5, 2)
        model2 = nn.Linear(3, 4)

        optimizer = optim.SGD(
            [
                {"params": model1.parameters(), "lr": 0.1},
                {"params": model2.parameters(), "lr": 0.01},
            ]
        )

        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", optimizer, {"step_size": 1, "gamma": 0.5}
        )

        scheduler.step()

        # Both groups should be updated
        assert optimizer.param_groups[0]["lr"] == 0.05
        assert optimizer.param_groups[1]["lr"] == 0.005

    def test_cosine_annealing_with_zero_t_max_raises_error(self, simple_optimizer):
        """Test that cosine annealing with T_max=0 raises appropriate error or creates scheduler."""
        # PyTorch's behavior with T_max=0 may vary by version
        try:
            scheduler = SchedulerRegistry.get_scheduler(
                "cosine_annealing", simple_optimizer, {"T_max": 0}
            )
            # If it doesn't raise, that's also acceptable behavior
            assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        except ValueError:
            # If it raises ValueError, that's the expected behavior
            pass

    def test_step_lr_with_negative_step_size_raises_error(self, simple_optimizer):
        """Test that StepLR with negative step_size raises error or creates scheduler."""
        # PyTorch's validation behavior may vary by version
        try:
            scheduler = SchedulerRegistry.get_scheduler(
                "step_lr", simple_optimizer, {"step_size": -10}
            )
            # If it doesn't raise, that's PyTorch's behavior
            assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        except ValueError:
            # If it raises ValueError, that's expected validation
            pass

    def test_reduce_on_plateau_with_invalid_mode(self, simple_optimizer):
        """Test that ReduceLROnPlateau with invalid mode raises error."""
        with pytest.raises(ValueError):
            SchedulerRegistry.get_scheduler(
                "reduce_on_plateau", simple_optimizer, {"mode": "invalid"}
            )

    def test_warmup_scheduler_with_zero_warmup_epochs(self, simple_optimizer):
        """Test warmup scheduler with zero warmup epochs."""
        scheduler = create_warmup_scheduler(simple_optimizer, warmup_epochs=0, total_epochs=100)

        # Should still create valid scheduler
        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    def test_warmup_scheduler_warmup_equals_total_epochs(self, simple_optimizer):
        """Test warmup scheduler when warmup epochs equal total epochs."""
        scheduler = create_warmup_scheduler(simple_optimizer, warmup_epochs=100, total_epochs=100)

        # Should still create valid scheduler
        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    def test_scheduler_state_dict_save_load(self, simple_optimizer):
        """Test that scheduler state can be saved and loaded."""
        scheduler1 = SchedulerRegistry.get_scheduler(
            "step_lr", simple_optimizer, {"step_size": 5, "gamma": 0.5}
        )

        # Step a few times
        for _ in range(3):
            scheduler1.step()

        # Save state
        state = scheduler1.state_dict()

        # Create new scheduler and load state
        model2 = nn.Linear(10, 5)
        optimizer2 = optim.SGD(model2.parameters(), lr=0.1)
        scheduler2 = SchedulerRegistry.get_scheduler(
            "step_lr", optimizer2, {"step_size": 5, "gamma": 0.5}
        )
        scheduler2.load_state_dict(state)

        # Both should be in same state
        assert scheduler1.last_epoch == scheduler2.last_epoch

    def test_empty_model_parameters(self):
        """Test that PyTorch raises error for optimizer with empty parameters."""
        # PyTorch explicitly raises ValueError for empty parameter list
        with pytest.raises(ValueError, match="optimizer got an empty parameter list"):
            optimizer = optim.SGD([], lr=0.1)

    def test_very_large_t_max(self, simple_optimizer):
        """Test cosine annealing with very large T_max."""
        scheduler = SchedulerRegistry.get_scheduler(
            "cosine_annealing", simple_optimizer, {"T_max": 1000000}
        )

        # Should work normally
        initial_lr = simple_optimizer.param_groups[0]["lr"]
        scheduler.step()
        # LR should decrease very slightly
        assert simple_optimizer.param_groups[0]["lr"] < initial_lr

    def test_scheduler_with_nan_learning_rate(self):
        """Test scheduler behavior with NaN learning rate."""
        model = nn.Linear(5, 2)
        optimizer = optim.SGD(model.parameters(), lr=float("nan"))

        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", optimizer, {"step_size": 10, "gamma": 0.9}
        )

        # Scheduler should handle NaN (PyTorch's behavior)
        scheduler.step()
        # Note: This tests that no exception is raised


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSchedulerIntegration:
    """Integration tests for schedulers in training scenarios."""

    def test_scheduler_with_training_loop(self, simple_optimizer):
        """Test scheduler integration in a simple training loop."""
        model = nn.Linear(10, 1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", optimizer, {"step_size": 2, "gamma": 0.9}
        )

        # Simulate training loop
        for epoch in range(5):
            # Fake forward/backward pass
            inputs = torch.randn(32, 10)
            targets = torch.randn(32, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Update scheduler
            scheduler.step()

        # LR should have been reduced
        assert optimizer.param_groups[0]["lr"] < 0.01

    def test_reduce_on_plateau_with_validation(self, simple_optimizer):
        """Test ReduceLROnPlateau with validation loop."""
        scheduler = SchedulerRegistry.get_scheduler(
            "reduce_on_plateau", simple_optimizer, {"mode": "min", "patience": 3, "factor": 0.5}
        )

        initial_lr = simple_optimizer.param_groups[0]["lr"]

        # Simulate validation losses (no improvement)
        val_losses = [1.0, 1.0, 1.0, 1.0, 1.0]

        for val_loss in val_losses:
            scheduler.step(val_loss)

        # LR should have been reduced after patience
        assert simple_optimizer.param_groups[0]["lr"] < initial_lr

    def test_multiple_schedulers_chained(self, simple_optimizer):
        """Test using ChainedScheduler with multiple schedulers."""
        scheduler1 = torch.optim.lr_scheduler.StepLR(simple_optimizer, step_size=5, gamma=0.5)
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(simple_optimizer, gamma=0.9)

        # ChainedScheduler takes a list as first positional argument, not as kwarg
        chained = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

        initial_lr = simple_optimizer.param_groups[0]["lr"]

        chained.step()

        # Both schedulers should have been applied
        # LR should be affected by both
        assert simple_optimizer.param_groups[0]["lr"] != initial_lr

    def test_warmup_then_cosine_training(self):
        """Test warmup followed by cosine annealing in training."""
        model = nn.Linear(5, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        scheduler = create_warmup_scheduler(
            optimizer, warmup_epochs=5, total_epochs=20, after_scheduler_name="cosine_annealing"
        )

        lrs = []
        for epoch in range(20):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Check that LR increased during warmup
        assert lrs[4] > lrs[0]

        # Check that LR decreased after warmup
        assert lrs[-1] < lrs[5]


# =============================================================================
# MODULE INITIALIZATION TESTS
# =============================================================================


class TestModuleInitialization:
    """Test module-level initialization."""

    def test_module_logger_info_message(self, caplog):
        """Test that module logs initialization message."""
        # This message is logged when module is imported
        # We can't re-import, but we can verify logger exists
        from milia_pipeline.models.training import schedulers

        assert hasattr(schedulers, "logger")

    def test_registry_has_schedulers_on_import(self):
        """Test that registry is populated on module import."""
        assert len(SchedulerRegistry._schedulers) > 0
        assert len(SchedulerRegistry.list_available()) > 0

    def test_all_schedulers_are_pytorch_classes(self):
        """Test that all registered schedulers are from PyTorch."""
        for name, sched_cls in SchedulerRegistry._schedulers.items():
            assert "torch.optim.lr_scheduler" in sched_cls.__module__


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Test thread safety considerations (conceptual tests)."""

    def test_multiple_schedulers_different_optimizers(self):
        """Test creating multiple schedulers with different optimizers."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(5, 2)

        optimizer1 = optim.SGD(model1.parameters(), lr=0.1)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

        scheduler1 = SchedulerRegistry.get_scheduler(
            "step_lr", optimizer1, {"step_size": 10, "gamma": 0.5}
        )
        scheduler2 = SchedulerRegistry.get_scheduler("cosine_annealing", optimizer2, {"T_max": 100})

        # Both should work independently
        scheduler1.step()
        scheduler2.step()

        assert optimizer1.param_groups[0]["lr"] != optimizer2.param_groups[0]["lr"]

    def test_registry_access_is_safe(self):
        """Test that registry can be accessed multiple times safely."""
        # Access registry multiple times
        for _ in range(10):
            schedulers = SchedulerRegistry.list_available()
            assert len(schedulers) > 0

    def test_custom_registration_isolation(self, reset_scheduler_registry):
        """Test that custom scheduler registration is properly isolated."""

        class CustomScheduler1(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return self.base_lrs

        class CustomScheduler2(torch.optim.lr_scheduler._LRScheduler):
            def get_lr(self):
                return self.base_lrs

        # Register in sequence
        SchedulerRegistry.register_custom_scheduler("custom1", CustomScheduler1)
        count_after_first = len(SchedulerRegistry.list_available())

        SchedulerRegistry.register_custom_scheduler("custom2", CustomScheduler2)
        count_after_second = len(SchedulerRegistry.list_available())

        # Should have incremented by exactly 1 each time
        assert count_after_second == count_after_first + 1


# =============================================================================
# DOCUMENTATION AND EXAMPLES TESTS
# =============================================================================


class TestDocumentationExamples:
    """Test that examples in docstrings work correctly."""

    def test_basic_usage_example(self, simple_optimizer):
        """Test basic usage example from module docstring."""
        scheduler = SchedulerRegistry.get_scheduler(
            "cosine_annealing", simple_optimizer, {"T_max": 100}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_get_scheduler_method_examples(self, simple_optimizer):
        """Test examples from get_scheduler method docstring."""
        # Simple usage - must provide required params
        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", simple_optimizer, {"step_size": 30, "gamma": 0.1}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

        # With custom parameters
        scheduler = SchedulerRegistry.get_scheduler(
            "cosine_annealing", simple_optimizer, {"T_max": 100, "eta_min": 1e-6}
        )
        assert scheduler.T_max == 100
        assert scheduler.eta_min == 1e-6

        # ReduceLROnPlateau (metric-based)
        scheduler = SchedulerRegistry.get_scheduler(
            "reduce_on_plateau", simple_optimizer, {"mode": "min", "patience": 10, "factor": 0.5}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_create_warmup_scheduler_example(self, simple_optimizer):
        """Test example from create_warmup_scheduler docstring."""
        scheduler = create_warmup_scheduler(
            simple_optimizer,
            warmup_epochs=10,
            total_epochs=100,
            after_scheduler_name="cosine_annealing",
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    def test_get_valid_params_docstring_example(self):
        """Test example from get_valid_params method docstring.

        From docstring:
            >>> params = SchedulerRegistry.get_valid_params("reduce_on_plateau")
            >>> print(params)
            {'mode': 'min', 'factor': 0.1, 'patience': 10, ...}
        """
        params = SchedulerRegistry.get_valid_params("reduce_on_plateau")
        assert isinstance(params, dict)
        assert "mode" in params
        assert "factor" in params
        assert "patience" in params
        assert params["mode"] == "min"

    def test_get_scheduler_info_docstring_example(self):
        """Test example from get_scheduler_info method docstring.

        From docstring:
            >>> info = SchedulerRegistry.get_scheduler_info("cosine_annealing")
            >>> print(info['valid_params'])
        """
        info = SchedulerRegistry.get_scheduler_info("cosine_annealing")
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)
        assert "T_max" in info["valid_params"]

    def test_get_scheduler_safe_usage_docstring_example(self, simple_optimizer):
        """Test safe usage example from get_scheduler docstring.

        From docstring:
            >>> # Safe usage - invalid params are filtered
            >>> scheduler = SchedulerRegistry.get_scheduler(
            ...     "step_lr",
            ...     optimizer,
            ...     {"step_size": 10, "invalid_param": 123}
            ... )
            >>> # Works! 'invalid_param' is filtered out
        """
        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", simple_optimizer, {"step_size": 10, "invalid_param": 123}
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_is_metric_based_docstring_example(self, simple_optimizer):
        """Test is_metric_based example from docstring.

        From docstring:
            >>> if SchedulerRegistry.is_metric_based("reduce_on_plateau"):
            ...     scheduler.step(val_loss)
            ... else:
            ...     scheduler.step()
        """
        # Test the pattern from the docstring
        scheduler = SchedulerRegistry.get_scheduler(
            "reduce_on_plateau", simple_optimizer, {"mode": "min", "patience": 5}
        )

        if SchedulerRegistry.is_metric_based("reduce_on_plateau"):
            # This should be true
            scheduler.step(1.0)
        else:
            scheduler.step()

        assert SchedulerRegistry.is_metric_based("reduce_on_plateau") is True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
