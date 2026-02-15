#!/usr/bin/env python3
"""
Complete Unit Test Suite for optimizers.py Module

Tests optimizer registry system including:
- OptimizerRegistry class with all methods
- Optimizer registration and retrieval (12 PyTorch optimizers)
- Default parameters handling
- Parameter filtering with _filter_params() method (dynamic introspection)
- get_valid_params() method for parameter discovery
- Parameter group support
- Custom optimizer registration
- Convenience functions (get_optimizer, list_optimizers)
- Error handling and edge cases
- Integration with PyTorch models
- Thread safety considerations

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: milia Team
Version: 1.2.0

Changelog:
- v1.2.0: Added defensive branch coverage for production-readiness:
          - _filter_params() introspection fallback when inspect.signature() fails
          - get_valid_params() introspection fallback when inspect.signature() fails
          - get_optimizer() TypeError wrapping (constructor TypeError → ValueError)
- v1.1.0: Added tests for get_valid_params(), _filter_params(), updated
          parameter filtering behavior tests (invalid params now filtered silently),
          updated get_optimizer_info tests to verify valid_params key
- v1.0.0: Initial release
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# Import the module under test
from milia_pipeline.models.training.optimizers import (
    # Main class
    OptimizerRegistry,
    # Convenience functions
    get_optimizer,
    list_optimizers,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Linear(10, 5)


@pytest.fixture
def multi_layer_model():
    """Create a multi-layer model for testing parameter groups."""

    class MultiLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
            self.decoder = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    return MultiLayerNet()


@pytest.fixture
def reset_optimizer_registry():
    """Reset optimizer registry to default state after test."""
    # Store original state
    original_optimizers = OptimizerRegistry._optimizers.copy()
    original_defaults = OptimizerRegistry._defaults.copy()

    yield

    # Restore original state
    OptimizerRegistry._optimizers = original_optimizers
    OptimizerRegistry._defaults = original_defaults


# =============================================================================
# OPTIMIZER REGISTRY TESTS
# =============================================================================


class TestOptimizerRegistry:
    """Test OptimizerRegistry class."""

    def test_registry_has_expected_optimizers(self):
        """Test that registry contains all expected optimizers."""
        expected_optimizers = [
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
        ]

        available = OptimizerRegistry.list_available()
        for opt_name in expected_optimizers:
            assert opt_name in available, f"Missing optimizer: {opt_name}"

    def test_registry_optimizers_are_valid_classes(self):
        """Test that all registered optimizers are valid PyTorch optimizer classes."""
        for name, opt_cls in OptimizerRegistry._optimizers.items():
            assert callable(opt_cls), f"Optimizer {name} is not callable"
            assert hasattr(opt_cls, "__name__"), f"Optimizer {name} has no __name__"
            assert issubclass(opt_cls, torch.optim.Optimizer), (
                f"Optimizer {name} is not a subclass of torch.optim.Optimizer"
            )

    def test_default_params_structure(self):
        """Test that default params are properly structured."""
        for name, params in OptimizerRegistry._defaults.items():
            assert isinstance(params, dict), f"Default params for {name} must be dict"
            assert name in OptimizerRegistry._optimizers, f"Defaults for unknown optimizer: {name}"

    def test_list_available_returns_sorted_list(self):
        """Test list_available returns sorted list of optimizer names."""
        available = OptimizerRegistry.list_available()
        assert isinstance(available, list)
        assert len(available) > 0
        assert available == sorted(available)

    def test_list_available_includes_all_optimizers(self):
        """Test list_available includes all registered optimizers."""
        available = OptimizerRegistry.list_available()
        for name in OptimizerRegistry._optimizers.keys():
            assert name in available

    def test_optimizer_count(self):
        """Test that expected number of optimizers are registered."""
        available = OptimizerRegistry.list_available()
        # 12 optimizers as per the module
        assert len(available) == 12


class TestGetOptimizer:
    """Test OptimizerRegistry.get_optimizer method."""

    def test_get_adam_default_params(self, simple_model):
        """Test getting Adam optimizer with default parameters."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 0.001}
        )
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_get_adam_custom_params(self, simple_model):
        """Test getting Adam optimizer with custom parameters."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adam",
            simple_model.parameters(),
            {"lr": 0.002, "betas": (0.95, 0.999), "weight_decay": 1e-5},
        )
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.002
        assert optimizer.param_groups[0]["betas"] == (0.95, 0.999)
        assert optimizer.param_groups[0]["weight_decay"] == 1e-5

    def test_get_adamw(self, simple_model):
        """Test getting AdamW optimizer."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adamw", simple_model.parameters(), {"lr": 0.001, "weight_decay": 0.01}
        )
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_get_sgd(self, simple_model):
        """Test getting SGD optimizer."""
        optimizer = OptimizerRegistry.get_optimizer(
            "sgd", simple_model.parameters(), {"lr": 0.01, "momentum": 0.9}
        )
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_get_sgd_with_nesterov(self, simple_model):
        """Test getting SGD optimizer with Nesterov momentum."""
        optimizer = OptimizerRegistry.get_optimizer(
            "sgd", simple_model.parameters(), {"lr": 0.01, "momentum": 0.9, "nesterov": True}
        )
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["nesterov"] is True

    def test_get_rmsprop(self, simple_model):
        """Test getting RMSprop optimizer."""
        optimizer = OptimizerRegistry.get_optimizer(
            "rmsprop", simple_model.parameters(), {"lr": 0.01, "alpha": 0.99}
        )
        assert isinstance(optimizer, torch.optim.RMSprop)
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[0]["alpha"] == 0.99

    def test_get_adagrad(self, simple_model):
        """Test getting Adagrad optimizer."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adagrad", simple_model.parameters(), {"lr": 0.01}
        )
        assert isinstance(optimizer, torch.optim.Adagrad)
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_get_adadelta(self, simple_model):
        """Test getting Adadelta optimizer."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adadelta", simple_model.parameters(), {"lr": 1.0, "rho": 0.9}
        )
        assert isinstance(optimizer, torch.optim.Adadelta)
        assert optimizer.param_groups[0]["lr"] == 1.0

    def test_get_adamax(self, simple_model):
        """Test getting Adamax optimizer."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adamax", simple_model.parameters(), {"lr": 0.002}
        )
        assert isinstance(optimizer, torch.optim.Adamax)
        assert optimizer.param_groups[0]["lr"] == 0.002

    def test_get_asgd(self, simple_model):
        """Test getting ASGD optimizer."""
        optimizer = OptimizerRegistry.get_optimizer("asgd", simple_model.parameters(), {"lr": 0.01})
        assert isinstance(optimizer, torch.optim.ASGD)
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_get_rprop(self, simple_model):
        """Test getting Rprop optimizer."""
        optimizer = OptimizerRegistry.get_optimizer(
            "rprop", simple_model.parameters(), {"lr": 0.01}
        )
        assert isinstance(optimizer, torch.optim.Rprop)
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_get_nadam(self, simple_model):
        """Test getting NAdam optimizer."""
        optimizer = OptimizerRegistry.get_optimizer(
            "nadam", simple_model.parameters(), {"lr": 0.002}
        )
        assert isinstance(optimizer, torch.optim.NAdam)
        assert optimizer.param_groups[0]["lr"] == 0.002

    def test_get_radam(self, simple_model):
        """Test getting RAdam optimizer."""
        optimizer = OptimizerRegistry.get_optimizer(
            "radam", simple_model.parameters(), {"lr": 0.001}
        )
        assert isinstance(optimizer, torch.optim.RAdam)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_get_lbfgs(self, simple_model):
        """Test getting LBFGS optimizer."""
        optimizer = OptimizerRegistry.get_optimizer("lbfgs", simple_model.parameters(), {"lr": 1})
        assert isinstance(optimizer, torch.optim.LBFGS)
        assert optimizer.param_groups[0]["lr"] == 1

    def test_get_optimizer_with_none_params(self, simple_model):
        """Test getting optimizer with None params (should work with defaults)."""
        optimizer = OptimizerRegistry.get_optimizer("adam", simple_model.parameters(), None)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_with_empty_params(self, simple_model):
        """Test getting optimizer with empty params dict."""
        optimizer = OptimizerRegistry.get_optimizer("adam", simple_model.parameters(), {})
        assert isinstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_with_parameter_groups(self, multi_layer_model):
        """Test getting optimizer with parameter groups."""
        param_groups = [
            {"params": multi_layer_model.encoder.parameters(), "lr": 0.001},
            {"params": multi_layer_model.decoder.parameters(), "lr": 0.01},
        ]

        optimizer = OptimizerRegistry.get_optimizer("adam", param_groups)
        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[1]["lr"] == 0.01

    def test_get_optimizer_unknown_name_raises_error(self, simple_model):
        """Test that unknown optimizer name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizerRegistry.get_optimizer("unknown_optimizer", simple_model.parameters())

        assert "Unknown optimizer: 'unknown_optimizer'" in str(exc_info.value)
        assert "Available optimizers:" in str(exc_info.value)

    def test_get_optimizer_invalid_params_filtered_silently(self, simple_model):
        """Test that invalid parameters are filtered out silently (not raising errors).

        The OptimizerRegistry now uses _filter_params() to dynamically filter out
        parameters that are not accepted by the optimizer class constructor.
        """
        # Invalid params should be filtered out, optimizer should be created with defaults
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 0.001, "invalid_param": 123}
        )

        # Should create Adam with the valid 'lr' parameter
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_get_optimizer_mixed_valid_invalid_params(self, simple_model):
        """Test that valid params are used while invalid ones are filtered."""
        # Mix of valid (lr, betas) and invalid (invalid_param) parameters
        optimizer = OptimizerRegistry.get_optimizer(
            "adam",
            simple_model.parameters(),
            {"lr": 0.002, "betas": (0.95, 0.999), "invalid_param": 123},
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.002
        assert optimizer.param_groups[0]["betas"] == (0.95, 0.999)

    def test_get_optimizer_safe_usage_example(self, simple_model):
        """Test safe usage example from get_optimizer docstring.

        From docstring:
            >>> optimizer = OptimizerRegistry.get_optimizer(
            ...     "adam",
            ...     model.parameters(),
            ...     {"lr": 0.001, "invalid_param": 123}
            ... )
            >>> # Works! 'invalid_param' is filtered out
        """
        # This should work - 'invalid_param' is filtered out
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 0.001, "invalid_param": 123}
        )
        assert isinstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_logs_debug_message(self, simple_model, caplog):
        """Test that optimizer initialization logs debug message."""
        with caplog.at_level(logging.DEBUG):
            OptimizerRegistry.get_optimizer("adam", simple_model.parameters(), {"lr": 0.001})

        # Check if debug message was logged
        debug_messages = [rec.message for rec in caplog.records if rec.levelname == "DEBUG"]
        assert any("Initialized adam optimizer" in msg for msg in debug_messages)

    def test_get_optimizer_logs_filtered_params(self, simple_model, caplog):
        """Test that get_optimizer logs ignored params at debug level."""
        with caplog.at_level(logging.DEBUG):
            OptimizerRegistry.get_optimizer(
                "adam", simple_model.parameters(), {"lr": 0.001, "invalid_param": 123}
            )

        # Should log about ignored unsupported params
        debug_messages = [rec.message for rec in caplog.records if rec.levelname == "DEBUG"]
        assert any("ignored unsupported params" in msg for msg in debug_messages)

    def test_get_optimizer_constructor_type_error_wraps_as_value_error(self, simple_model):
        """Test that TypeError from optimizer constructor is wrapped as ValueError.

        When the optimizer class constructor raises TypeError (e.g., due to
        incompatible parameter types that pass filtering but fail at construction),
        get_optimizer wraps it into a descriptive ValueError.
        """
        # Patch _filter_params to bypass filtering and let bad params through
        # to trigger the TypeError from the actual optimizer constructor
        with patch.object(
            OptimizerRegistry,
            "_filter_params",
            return_value={"lr": "not_a_number"},  # Will cause TypeError in Adam.__init__
        ):
            with pytest.raises(ValueError) as exc_info:
                OptimizerRegistry.get_optimizer(
                    "adam", simple_model.parameters(), {"lr": "not_a_number"}
                )

            assert "Invalid parameters for optimizer 'adam'" in str(exc_info.value)
            # Verify the original TypeError is chained
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, TypeError)


class TestGetOptimizerInfo:
    """Test OptimizerRegistry.get_optimizer_info method."""

    def test_get_optimizer_info_adam(self):
        """Test getting optimizer info for Adam."""
        info = OptimizerRegistry.get_optimizer_info("adam")

        assert isinstance(info, dict)
        assert info["name"] == "adam"
        assert info["class"] == "Adam"
        assert "torch.optim" in info["module"]
        assert isinstance(info["default_params"], dict)
        assert info["doc"] is not None
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)
        # Adam should have lr, betas, eps, weight_decay, etc.
        assert "lr" in info["valid_params"]

    def test_get_optimizer_info_sgd(self):
        """Test getting optimizer info for SGD."""
        info = OptimizerRegistry.get_optimizer_info("sgd")

        assert info["name"] == "sgd"
        assert info["class"] == "SGD"
        assert "default_params" in info
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)
        # SGD should have lr, momentum, dampening, weight_decay, nesterov
        assert "lr" in info["valid_params"]
        assert "momentum" in info["valid_params"]

    def test_get_optimizer_info_unknown_raises_error(self):
        """Test that getting info for unknown optimizer raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizerRegistry.get_optimizer_info("unknown_opt")

        assert "Unknown optimizer: 'unknown_opt'" in str(exc_info.value)

    def test_get_optimizer_info_has_all_keys(self):
        """Test that optimizer info contains all expected keys."""
        info = OptimizerRegistry.get_optimizer_info("adam")

        expected_keys = ["name", "class", "module", "default_params", "valid_params", "doc"]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

        assert isinstance(info["valid_params"], dict)


class TestGetDefaultParams:
    """Test OptimizerRegistry.get_default_params method."""

    def test_get_default_params_adam(self):
        """Test getting default params for Adam."""
        defaults = OptimizerRegistry.get_default_params("adam")

        assert isinstance(defaults, dict)
        assert "lr" in defaults
        assert "betas" in defaults
        assert "eps" in defaults
        assert "weight_decay" in defaults
        assert defaults["lr"] == 0.001
        assert defaults["betas"] == (0.9, 0.999)

    def test_get_default_params_sgd(self):
        """Test getting default params for SGD."""
        defaults = OptimizerRegistry.get_default_params("sgd")

        assert isinstance(defaults, dict)
        assert "lr" in defaults
        assert "momentum" in defaults
        assert defaults["lr"] == 0.01
        assert defaults["nesterov"] is False

    def test_get_default_params_adamw(self):
        """Test getting default params for AdamW."""
        defaults = OptimizerRegistry.get_default_params("adamw")

        assert isinstance(defaults, dict)
        assert "weight_decay" in defaults
        assert defaults["weight_decay"] == 0.01

    def test_get_default_params_returns_copy(self):
        """Test that get_default_params returns a copy, not reference."""
        defaults1 = OptimizerRegistry.get_default_params("adam")
        defaults2 = OptimizerRegistry.get_default_params("adam")

        # Modify one
        defaults1["lr"] = 999

        # Other should be unchanged
        assert defaults2["lr"] == 0.001

    def test_get_default_params_unknown_raises_error(self):
        """Test that getting default params for unknown optimizer raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizerRegistry.get_default_params("unknown_opt")

        assert "Unknown optimizer: 'unknown_opt'" in str(exc_info.value)

    def test_get_default_params_optimizer_without_defaults(self):
        """Test getting default params for optimizer without defined defaults."""
        # Some optimizers may not have defaults defined
        defaults = OptimizerRegistry.get_default_params("rprop")
        assert isinstance(defaults, dict)


class TestGetValidParams:
    """Test OptimizerRegistry.get_valid_params method."""

    def test_get_valid_params_adam(self):
        """Test getting valid params for Adam optimizer.

        Example from docstring:
            >>> params = OptimizerRegistry.get_valid_params("adam")
            >>> print(params)
            {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, ...}
        """
        params = OptimizerRegistry.get_valid_params("adam")

        assert isinstance(params, dict)
        assert "lr" in params
        assert "betas" in params
        assert "eps" in params
        assert "weight_decay" in params

    def test_get_valid_params_sgd(self):
        """Test getting valid params for SGD optimizer."""
        params = OptimizerRegistry.get_valid_params("sgd")

        assert isinstance(params, dict)
        assert "lr" in params
        assert "momentum" in params
        assert "dampening" in params
        assert "weight_decay" in params
        assert "nesterov" in params

    def test_get_valid_params_adamw(self):
        """Test getting valid params for AdamW optimizer."""
        params = OptimizerRegistry.get_valid_params("adamw")

        assert isinstance(params, dict)
        assert "lr" in params
        assert "betas" in params
        assert "weight_decay" in params

    def test_get_valid_params_rmsprop(self):
        """Test getting valid params for RMSprop optimizer."""
        params = OptimizerRegistry.get_valid_params("rmsprop")

        assert isinstance(params, dict)
        assert "lr" in params
        assert "alpha" in params
        assert "eps" in params
        assert "momentum" in params

    def test_get_valid_params_all_optimizers(self):
        """Test getting valid params for all registered optimizers."""
        for opt_name in OptimizerRegistry.list_available():
            params = OptimizerRegistry.get_valid_params(opt_name)
            assert isinstance(params, dict)
            # All optimizers should have 'lr' parameter
            assert "lr" in params

    def test_get_valid_params_unknown_optimizer_raises_error(self):
        """Test that getting valid params for unknown optimizer raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizerRegistry.get_valid_params("unknown_optimizer")

        assert "Unknown optimizer: 'unknown_optimizer'" in str(exc_info.value)

    def test_get_valid_params_returns_defaults(self):
        """Test that get_valid_params returns default values for params with defaults."""
        params = OptimizerRegistry.get_valid_params("adam")

        # Adam params should have sensible defaults
        assert params["lr"] == 0.001
        assert params["betas"] == (0.9, 0.999)
        assert params["eps"] == 1e-08

    def test_get_valid_params_differs_from_registry_defaults(self):
        """Test that get_valid_params (introspected) may differ from registry defaults.

        Registry defaults are manually defined, while get_valid_params uses introspection.
        """
        introspected = OptimizerRegistry.get_valid_params("adam")
        registry_defaults = OptimizerRegistry.get_default_params("adam")

        # Both should be dicts
        assert isinstance(introspected, dict)
        assert isinstance(registry_defaults, dict)

        # Introspected may have more params than registry defaults
        assert len(introspected) >= len(registry_defaults)

    def test_get_valid_params_introspection_fallback(self):
        """Test get_valid_params fallback when inspect.signature() fails.

        When inspect.signature() raises ValueError or TypeError (e.g., for
        built-in C extension classes whose __init__ cannot be introspected),
        get_valid_params should return an empty dict rather than crashing.
        """
        with patch("inspect.signature", side_effect=ValueError("no signature found")):
            params = OptimizerRegistry.get_valid_params("adam")

        # Fallback: returns empty dict
        assert params == {}
        assert isinstance(params, dict)

    def test_get_valid_params_introspection_fallback_type_error(self):
        """Test get_valid_params fallback when inspect.signature() raises TypeError."""
        with patch("inspect.signature", side_effect=TypeError("unsupported callable")):
            params = OptimizerRegistry.get_valid_params("sgd")

        # Fallback: returns empty dict
        assert params == {}
        assert isinstance(params, dict)


class TestFilterParams:
    """Test OptimizerRegistry._filter_params private method.

    The _filter_params method dynamically filters parameters to only those
    accepted by the target optimizer constructor using inspect.signature().
    """

    def test_filter_params_adam_valid_params(self):
        """Test _filter_params with all valid Adam params."""
        params = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
        filtered = OptimizerRegistry._filter_params(torch.optim.Adam, params)

        assert filtered == params

    def test_filter_params_adam_invalid_params(self):
        """Test _filter_params filters out invalid params."""
        params = {"lr": 0.001, "invalid_param": 123, "another_bad": "value"}
        filtered = OptimizerRegistry._filter_params(torch.optim.Adam, params)

        assert filtered == {"lr": 0.001}
        assert "invalid_param" not in filtered
        assert "another_bad" not in filtered

    def test_filter_params_sgd_filters_adam_params(self):
        """Test _filter_params filters Adam-specific params from SGD."""
        params = {
            "lr": 0.01,
            "momentum": 0.9,
            "betas": (0.9, 0.999),  # Adam param, not valid for SGD
            "eps": 1e-8,  # Adam param, not valid for SGD
        }
        filtered = OptimizerRegistry._filter_params(torch.optim.SGD, params)

        assert "lr" in filtered
        assert "momentum" in filtered
        assert "betas" not in filtered
        assert "eps" not in filtered

    def test_filter_params_empty_dict(self):
        """Test _filter_params with empty params dict."""
        params = {}
        filtered = OptimizerRegistry._filter_params(torch.optim.Adam, params)

        assert filtered == {}

    def test_filter_params_all_invalid(self):
        """Test _filter_params with all invalid params."""
        params = {"invalid1": 1, "invalid2": 2, "invalid3": 3}
        filtered = OptimizerRegistry._filter_params(torch.optim.Adam, params)

        assert filtered == {}

    def test_filter_params_none_input(self):
        """Test _filter_params handles None/empty input gracefully."""
        # The method checks if not params first
        filtered = OptimizerRegistry._filter_params(torch.optim.Adam, None)
        assert filtered == {}

        filtered = OptimizerRegistry._filter_params(torch.optim.Adam, {})
        assert filtered == {}

    def test_filter_params_preserves_values(self):
        """Test _filter_params preserves parameter values correctly."""
        params = {"lr": 0.123456, "betas": (0.85, 0.9999), "eps": 1e-10, "weight_decay": 0.05}
        filtered = OptimizerRegistry._filter_params(torch.optim.Adam, params)

        assert filtered["lr"] == 0.123456
        assert filtered["betas"] == (0.85, 0.9999)
        assert filtered["eps"] == 1e-10
        assert filtered["weight_decay"] == 0.05

    def test_filter_params_excludes_self_and_params(self):
        """Test _filter_params excludes 'self' and 'params' (model parameters)."""
        # 'params' is the model parameters argument, should not be in filtered result
        params = {
            "lr": 0.001,
            "params": "should_be_ignored",  # This is the model params arg
            "self": "should_be_ignored",
        }
        filtered = OptimizerRegistry._filter_params(torch.optim.Adam, params)

        assert "lr" in filtered
        assert "params" not in filtered
        assert "self" not in filtered

    def test_filter_params_introspection_fallback(self):
        """Test _filter_params fallback when inspect.signature() fails.

        When inspect.signature() raises ValueError or TypeError (e.g., for
        built-in C extension classes), _filter_params should return the
        original params dict as-is, ensuring graceful degradation.
        """
        params = {"lr": 0.001, "betas": (0.9, 0.999)}

        with patch("inspect.signature", side_effect=ValueError("no signature found")):
            filtered = OptimizerRegistry._filter_params(torch.optim.Adam, params)

        # Fallback: returns original params unchanged
        assert filtered == params

    def test_filter_params_introspection_fallback_type_error(self):
        """Test _filter_params fallback when inspect.signature() raises TypeError."""
        params = {"lr": 0.01, "momentum": 0.9}

        with patch("inspect.signature", side_effect=TypeError("unsupported callable")):
            filtered = OptimizerRegistry._filter_params(torch.optim.SGD, params)

        # Fallback: returns original params unchanged
        assert filtered == params


class TestRegisterCustomOptimizer:
    """Test OptimizerRegistry.register_custom_optimizer method."""

    def test_register_custom_optimizer_basic(self, reset_optimizer_registry):
        """Test registering a basic custom optimizer."""

        class CustomOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        OptimizerRegistry.register_custom_optimizer("custom_opt", CustomOptimizer, {"lr": 0.01})

        assert "custom_opt" in OptimizerRegistry.list_available()
        assert OptimizerRegistry._optimizers["custom_opt"] == CustomOptimizer

    def test_register_custom_optimizer_with_default_params(self, reset_optimizer_registry):
        """Test registering custom optimizer with default params."""

        class CustomOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01, momentum=0.9):
                defaults = dict(lr=lr, momentum=momentum)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        custom_defaults = {"lr": 0.02, "momentum": 0.95}

        OptimizerRegistry.register_custom_optimizer(
            "custom_momentum_opt", CustomOptimizer, custom_defaults
        )

        assert OptimizerRegistry._defaults["custom_momentum_opt"] == custom_defaults

    def test_register_custom_optimizer_without_defaults(self, reset_optimizer_registry):
        """Test registering custom optimizer without default params."""

        class SimpleOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        OptimizerRegistry.register_custom_optimizer("simple_opt", SimpleOptimizer)

        assert "simple_opt" in OptimizerRegistry.list_available()

    def test_register_custom_optimizer_duplicate_raises_error(self, reset_optimizer_registry):
        """Test that registering duplicate optimizer without overwrite raises error."""

        class CustomOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        OptimizerRegistry.register_custom_optimizer("dup_opt", CustomOptimizer)

        with pytest.raises(ValueError) as exc_info:
            OptimizerRegistry.register_custom_optimizer("dup_opt", CustomOptimizer)

        assert "already registered" in str(exc_info.value)
        assert "Use overwrite=True" in str(exc_info.value)

    def test_register_custom_optimizer_overwrite(self, reset_optimizer_registry):
        """Test overwriting existing optimizer."""

        class CustomOptimizer1(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        class CustomOptimizer2(torch.optim.Optimizer):
            def __init__(self, params, lr=0.02):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        OptimizerRegistry.register_custom_optimizer("my_opt", CustomOptimizer1)
        OptimizerRegistry.register_custom_optimizer("my_opt", CustomOptimizer2, overwrite=True)

        assert OptimizerRegistry._optimizers["my_opt"] == CustomOptimizer2

    def test_register_custom_optimizer_non_optimizer_class_raises_error(
        self, reset_optimizer_registry
    ):
        """Test that registering non-Optimizer class raises TypeError."""

        class NotAnOptimizer:
            pass

        with pytest.raises(TypeError) as exc_info:
            OptimizerRegistry.register_custom_optimizer("bad_opt", NotAnOptimizer)

        assert "must be a subclass of torch.optim.Optimizer" in str(exc_info.value)

    def test_register_custom_optimizer_logs_info(self, reset_optimizer_registry, caplog):
        """Test that registering custom optimizer logs info message."""

        class CustomOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        with caplog.at_level(logging.INFO):
            OptimizerRegistry.register_custom_optimizer("log_test_opt", CustomOptimizer)

        info_messages = [rec.message for rec in caplog.records if rec.levelname == "INFO"]
        assert any("Registered custom optimizer: 'log_test_opt'" in msg for msg in info_messages)

    def test_use_custom_optimizer_after_registration(self, reset_optimizer_registry, simple_model):
        """Test using custom optimizer after registration."""

        class MyCustomOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01, custom_param=1.0):
                defaults = dict(lr=lr, custom_param=custom_param)
                super().__init__(params, defaults)

            def step(self, closure=None):
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            p.data.add_(p.grad.data, alpha=-group["lr"])

        OptimizerRegistry.register_custom_optimizer(
            "my_custom", MyCustomOptimizer, {"lr": 0.01, "custom_param": 2.0}
        )

        optimizer = OptimizerRegistry.get_optimizer(
            "my_custom", simple_model.parameters(), {"lr": 0.005, "custom_param": 3.0}
        )

        assert isinstance(optimizer, MyCustomOptimizer)
        assert optimizer.param_groups[0]["lr"] == 0.005
        assert optimizer.param_groups[0]["custom_param"] == 3.0


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_optimizer_function(self, simple_model):
        """Test get_optimizer convenience function."""
        optimizer = get_optimizer("adam", simple_model.parameters(), {"lr": 0.001})
        assert isinstance(optimizer, torch.optim.Adam)

    def test_list_optimizers_function(self):
        """Test list_optimizers convenience function."""
        optimizers = list_optimizers()
        assert isinstance(optimizers, list)
        assert len(optimizers) > 0
        assert "adam" in optimizers
        assert "sgd" in optimizers

    def test_convenience_function_matches_registry(self, simple_model):
        """Test that convenience function matches registry method."""
        opt1 = get_optimizer("adam", simple_model.parameters(), {"lr": 0.001})
        opt2 = OptimizerRegistry.get_optimizer("adam", simple_model.parameters(), {"lr": 0.001})

        assert type(opt1) == type(opt2)
        assert opt1.param_groups[0]["lr"] == opt2.param_groups[0]["lr"]


# =============================================================================
# OPTIMIZER FUNCTIONALITY TESTS
# =============================================================================


class TestOptimizerFunctionality:
    """Test actual optimizer functionality."""

    def test_optimizer_step_updates_parameters(self, simple_model):
        """Test that optimizer step updates model parameters."""
        optimizer = OptimizerRegistry.get_optimizer("adam", simple_model.parameters(), {"lr": 0.01})

        # Store initial parameters
        initial_weight = simple_model.weight.data.clone()

        # Create dummy input and target
        input_data = torch.randn(5, 10)
        target = torch.randn(5, 5)

        # Forward pass
        output = simple_model(input_data)
        loss = nn.functional.mse_loss(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Parameters should have changed
        assert not torch.equal(initial_weight, simple_model.weight.data)

    def test_optimizer_zero_grad(self, simple_model):
        """Test that zero_grad clears gradients."""
        optimizer = OptimizerRegistry.get_optimizer("sgd", simple_model.parameters(), {"lr": 0.01})

        # Create gradients
        input_data = torch.randn(5, 10)
        target = torch.randn(5, 5)
        output = simple_model(input_data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Verify gradients exist
        assert simple_model.weight.grad is not None

        # Clear gradients
        optimizer.zero_grad()

        # Verify gradients are cleared
        assert simple_model.weight.grad is None or torch.all(simple_model.weight.grad == 0)

    def test_different_optimizers_produce_different_updates(self, simple_model):
        """Test that different optimizers produce different parameter updates."""
        # Create two identical models
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        model2.load_state_dict(model1.state_dict())

        # Use different optimizers
        opt1 = OptimizerRegistry.get_optimizer("adam", model1.parameters(), {"lr": 0.01})
        opt2 = OptimizerRegistry.get_optimizer("sgd", model2.parameters(), {"lr": 0.01})

        # Same input and target
        input_data = torch.randn(5, 10)
        target = torch.randn(5, 5)

        # Train both models
        for model, optimizer in [(model1, opt1), (model2, opt2)]:
            output = model(input_data)
            loss = nn.functional.mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Parameters should be different
        assert not torch.equal(model1.weight.data, model2.weight.data)

    def test_optimizer_with_parameter_groups_updates_separately(self, multi_layer_model):
        """Test that parameter groups are updated with different learning rates."""
        param_groups = [
            {"params": multi_layer_model.encoder.parameters(), "lr": 0.001},
            {"params": multi_layer_model.decoder.parameters(), "lr": 0.01},
        ]

        optimizer = OptimizerRegistry.get_optimizer("sgd", param_groups)

        # Store initial parameters
        initial_encoder_weight = list(multi_layer_model.encoder.parameters())[0].data.clone()
        initial_decoder_weight = list(multi_layer_model.decoder.parameters())[0].data.clone()

        # Training step
        input_data = torch.randn(5, 10)
        target = torch.randn(5, 5)
        output = multi_layer_model(input_data)
        loss = nn.functional.mse_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Both should be updated
        encoder_weight = list(multi_layer_model.encoder.parameters())[0].data
        decoder_weight = list(multi_layer_model.decoder.parameters())[0].data

        assert not torch.equal(initial_encoder_weight, encoder_weight)
        assert not torch.equal(initial_decoder_weight, decoder_weight)

    def test_sgd_with_momentum(self, simple_model):
        """Test SGD optimizer with momentum."""
        optimizer = OptimizerRegistry.get_optimizer(
            "sgd", simple_model.parameters(), {"lr": 0.01, "momentum": 0.9}
        )

        assert optimizer.param_groups[0]["momentum"] == 0.9

        # Run multiple steps
        input_data = torch.randn(5, 10)
        target = torch.randn(5, 5)

        for _ in range(3):
            output = simple_model(input_data)
            loss = nn.functional.mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_adam_with_weight_decay(self, simple_model):
        """Test Adam optimizer with weight decay."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 0.001, "weight_decay": 1e-5}
        )

        assert optimizer.param_groups[0]["weight_decay"] == 1e-5

        # Run training step
        input_data = torch.randn(5, 10)
        target = torch.randn(5, 5)
        output = simple_model(input_data)
        loss = nn.functional.mse_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestOptimizerIntegration:
    """Test optimizer integration with training loops."""

    def test_simple_training_loop(self, simple_model):
        """Test optimizer in a simple training loop."""
        optimizer = OptimizerRegistry.get_optimizer("adam", simple_model.parameters(), {"lr": 0.01})

        initial_loss = None
        final_loss = None

        for epoch in range(10):
            input_data = torch.randn(32, 10)
            target = torch.randn(32, 5)

            output = simple_model(input_data)
            loss = nn.functional.mse_loss(output, target)

            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 9:
                final_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should generally decrease (though not guaranteed for random data)
        # Just verify training ran without errors
        assert initial_loss is not None
        assert final_loss is not None

    def test_optimizer_state_dict(self, simple_model):
        """Test saving and loading optimizer state."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 0.001}
        )

        # Run a few steps
        for _ in range(3):
            input_data = torch.randn(5, 10)
            target = torch.randn(5, 5)
            output = simple_model(input_data)
            loss = nn.functional.mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, dict)
        assert "state" in state_dict
        assert "param_groups" in state_dict

        # Create new optimizer and load state
        new_optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 0.001}
        )
        new_optimizer.load_state_dict(state_dict)

        # State should match
        assert (
            new_optimizer.state_dict()["param_groups"][0]["lr"]
            == state_dict["param_groups"][0]["lr"]
        )

    def test_multiple_optimizers_same_model(self, multi_layer_model):
        """Test using multiple optimizers for different parts of model."""
        opt_encoder = OptimizerRegistry.get_optimizer(
            "adam", multi_layer_model.encoder.parameters(), {"lr": 0.001}
        )
        opt_decoder = OptimizerRegistry.get_optimizer(
            "sgd", multi_layer_model.decoder.parameters(), {"lr": 0.01}
        )

        # Training step
        input_data = torch.randn(5, 10)
        target = torch.randn(5, 5)
        output = multi_layer_model(input_data)
        loss = nn.functional.mse_loss(output, target)

        opt_encoder.zero_grad()
        opt_decoder.zero_grad()
        loss.backward()
        opt_encoder.step()
        opt_decoder.step()

    def test_optimizer_with_gradient_clipping(self, simple_model):
        """Test optimizer with gradient clipping."""
        optimizer = OptimizerRegistry.get_optimizer("adam", simple_model.parameters(), {"lr": 0.01})

        input_data = torch.randn(5, 10)
        target = torch.randn(5, 5)
        output = simple_model(input_data)
        loss = nn.functional.mse_loss(output, target)

        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(simple_model.parameters(), max_norm=1.0)

        optimizer.step()


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_model_parameters(self):
        """Test optimizer with empty parameters iterator."""
        empty_params = iter([])

        # Some optimizers may fail with empty parameters
        with pytest.raises((ValueError, RuntimeError)):
            OptimizerRegistry.get_optimizer("adam", empty_params, {"lr": 0.001})

    def test_optimizer_case_sensitivity(self, simple_model):
        """Test that optimizer names are case-sensitive."""
        # Should work with lowercase
        opt = OptimizerRegistry.get_optimizer("adam", simple_model.parameters(), {"lr": 0.001})
        assert isinstance(opt, torch.optim.Adam)

        # Should fail with uppercase (assuming registry uses lowercase)
        with pytest.raises(ValueError):
            OptimizerRegistry.get_optimizer("ADAM", simple_model.parameters(), {"lr": 0.001})

    def test_optimizer_with_very_small_lr(self, simple_model):
        """Test optimizer with very small learning rate."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 1e-10}
        )
        assert optimizer.param_groups[0]["lr"] == 1e-10

    def test_optimizer_with_very_large_lr(self, simple_model):
        """Test optimizer with very large learning rate."""
        optimizer = OptimizerRegistry.get_optimizer(
            "sgd", simple_model.parameters(), {"lr": 1000.0}
        )
        assert optimizer.param_groups[0]["lr"] == 1000.0

    def test_optimizer_with_zero_lr(self, simple_model):
        """Test optimizer with zero learning rate."""
        optimizer = OptimizerRegistry.get_optimizer("adam", simple_model.parameters(), {"lr": 0.0})

        initial_weight = simple_model.weight.data.clone()

        # Run training step
        input_data = torch.randn(5, 10)
        target = torch.randn(5, 5)
        output = simple_model(input_data)
        loss = nn.functional.mse_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Parameters should not change with lr=0
        assert torch.allclose(initial_weight, simple_model.weight.data, atol=1e-8)

    def test_optimizer_parameters_as_list(self, simple_model):
        """Test passing parameters as list instead of iterator."""
        params_list = list(simple_model.parameters())
        optimizer = OptimizerRegistry.get_optimizer("adam", params_list, {"lr": 0.001})
        assert isinstance(optimizer, torch.optim.Adam)

    def test_invalid_parameter_type(self):
        """Test that invalid parameter type raises appropriate error."""
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            OptimizerRegistry.get_optimizer("adam", "not_parameters", {"lr": 0.001})


# =============================================================================
# MODULE INITIALIZATION TESTS
# =============================================================================


class TestModuleInitialization:
    """Test module-level initialization."""

    def test_module_logger_info_message(self, caplog):
        """Test that module logs initialization message."""
        # This message is logged when module is imported
        from milia_pipeline.models.training import optimizers

        assert hasattr(optimizers, "logger")

    def test_registry_has_optimizers_on_import(self):
        """Test that registry is populated on module import."""
        assert len(OptimizerRegistry._optimizers) > 0
        assert len(OptimizerRegistry.list_available()) > 0

    def test_all_optimizers_are_pytorch_classes(self):
        """Test that all registered optimizers are from PyTorch."""
        for name, opt_cls in OptimizerRegistry._optimizers.items():
            assert "torch.optim" in opt_cls.__module__


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Test thread safety considerations (conceptual tests)."""

    def test_multiple_optimizers_different_models(self):
        """Test creating multiple optimizers with different models."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(5, 2)

        optimizer1 = OptimizerRegistry.get_optimizer("adam", model1.parameters(), {"lr": 0.001})
        optimizer2 = OptimizerRegistry.get_optimizer("sgd", model2.parameters(), {"lr": 0.01})

        # Both should work independently
        input1 = torch.randn(3, 10)
        input2 = torch.randn(3, 5)
        target1 = torch.randn(3, 5)
        target2 = torch.randn(3, 2)

        # Train model1
        output1 = model1(input1)
        loss1 = nn.functional.mse_loss(output1, target1)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        # Train model2
        output2 = model2(input2)
        loss2 = nn.functional.mse_loss(output2, target2)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

    def test_registry_access_is_safe(self):
        """Test that registry can be accessed multiple times safely."""
        # Access registry multiple times
        for _ in range(10):
            optimizers = OptimizerRegistry.list_available()
            assert len(optimizers) > 0

    def test_custom_registration_isolation(self, reset_optimizer_registry):
        """Test that custom optimizer registration is properly isolated."""

        class CustomOptimizer1(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        class CustomOptimizer2(torch.optim.Optimizer):
            def __init__(self, params, lr=0.02):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        # Register in sequence
        OptimizerRegistry.register_custom_optimizer("custom1", CustomOptimizer1)
        count_after_first = len(OptimizerRegistry.list_available())

        OptimizerRegistry.register_custom_optimizer("custom2", CustomOptimizer2)
        count_after_second = len(OptimizerRegistry.list_available())

        # Should have incremented by exactly 1 each time
        assert count_after_second == count_after_first + 1


# =============================================================================
# DOCUMENTATION AND EXAMPLES TESTS
# =============================================================================


class TestDocumentationExamples:
    """Test that examples in docstrings work correctly."""

    def test_basic_usage_example(self, simple_model):
        """Test basic usage example from module docstring."""
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 0.001, "weight_decay": 1e-5}
        )
        assert isinstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_method_examples(self, simple_model):
        """Test examples from get_optimizer method docstring."""
        # Simple usage with defaults
        optimizer = OptimizerRegistry.get_optimizer("adam", simple_model.parameters())
        assert isinstance(optimizer, torch.optim.Adam)

        # With custom parameters
        optimizer = OptimizerRegistry.get_optimizer(
            "adam",
            simple_model.parameters(),
            {"lr": 0.001, "weight_decay": 1e-5, "betas": (0.9, 0.999)},
        )
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == 1e-5

    def test_parameter_groups_example(self, multi_layer_model):
        """Test parameter groups example from docstring."""
        param_groups = [
            {"params": multi_layer_model.encoder.parameters(), "lr": 0.001},
            {"params": multi_layer_model.decoder.parameters(), "lr": 0.01},
        ]
        optimizer = OptimizerRegistry.get_optimizer("adam", param_groups)
        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) == 2

    def test_list_available_example(self):
        """Test list_available example from docstring."""
        optimizers = OptimizerRegistry.list_available()
        assert isinstance(optimizers, list)
        assert len(optimizers) > 0

    def test_get_default_params_example(self):
        """Test get_default_params example from docstring."""
        defaults = OptimizerRegistry.get_default_params("adam")
        assert isinstance(defaults, dict)
        assert "lr" in defaults
        assert defaults["lr"] == 0.001

    def test_register_custom_optimizer_example(self, reset_optimizer_registry, simple_model):
        """Test register_custom_optimizer example from docstring."""

        class MyOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)

            def step(self, closure=None):
                # Custom optimization step
                pass

        OptimizerRegistry.register_custom_optimizer("my_opt", MyOptimizer, {"lr": 0.01})

        assert "my_opt" in OptimizerRegistry.list_available()

    def test_convenience_function_example(self, simple_model):
        """Test convenience function example from docstring."""
        optimizer = get_optimizer("adam", simple_model.parameters(), {"lr": 0.001})
        assert isinstance(optimizer, torch.optim.Adam)

    def test_get_valid_params_docstring_example(self):
        """Test example from get_valid_params method docstring.

        From docstring:
            >>> params = OptimizerRegistry.get_valid_params("adam")
            >>> print(params)
            {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, ...}
        """
        params = OptimizerRegistry.get_valid_params("adam")
        assert isinstance(params, dict)
        assert "lr" in params
        assert "betas" in params
        assert "eps" in params

    def test_get_optimizer_info_docstring_example(self):
        """Test example from get_optimizer_info method docstring.

        From docstring:
            >>> info = OptimizerRegistry.get_optimizer_info("adam")
            >>> print(info['valid_params'])
        """
        info = OptimizerRegistry.get_optimizer_info("adam")
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)
        assert "lr" in info["valid_params"]

    def test_get_optimizer_safe_usage_docstring_example(self, simple_model):
        """Test safe usage example from get_optimizer docstring.

        From docstring:
            >>> # Safe usage - invalid params are filtered
            >>> optimizer = OptimizerRegistry.get_optimizer(
            ...     "adam",
            ...     model.parameters(),
            ...     {"lr": 0.001, "invalid_param": 123}
            ... )
            >>> # Works! 'invalid_param' is filtered out
        """
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 0.001, "invalid_param": 123}
        )
        assert isinstance(optimizer, torch.optim.Adam)


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================


class TestParameterValidation:
    """Test parameter validation and edge cases."""

    def test_adam_beta_validation(self, simple_model):
        """Test Adam beta parameter validation."""
        # Valid betas
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", simple_model.parameters(), {"lr": 0.001, "betas": (0.9, 0.999)}
        )
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)

    def test_sgd_momentum_range(self, simple_model):
        """Test SGD momentum parameter range."""
        # Valid momentum
        optimizer = OptimizerRegistry.get_optimizer(
            "sgd", simple_model.parameters(), {"lr": 0.01, "momentum": 0.9}
        )
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_rmsprop_alpha_parameter(self, simple_model):
        """Test RMSprop alpha parameter."""
        optimizer = OptimizerRegistry.get_optimizer(
            "rmsprop", simple_model.parameters(), {"lr": 0.01, "alpha": 0.99}
        )
        assert optimizer.param_groups[0]["alpha"] == 0.99

    def test_weight_decay_parameter(self, simple_model):
        """Test weight decay parameter for various optimizers."""
        for opt_name in ["adam", "adamw", "sgd"]:
            optimizer = OptimizerRegistry.get_optimizer(
                opt_name, simple_model.parameters(), {"lr": 0.01, "weight_decay": 1e-4}
            )
            assert optimizer.param_groups[0]["weight_decay"] == 1e-4


# =============================================================================
# COMPREHENSIVE OPTIMIZER TESTS
# =============================================================================


class TestAllOptimizers:
    """Test all registered optimizers can be instantiated."""

    def test_all_optimizers_instantiate(self, simple_model):
        """Test that all registered optimizers can be instantiated."""
        for opt_name in OptimizerRegistry.list_available():
            # Skip LBFGS as it may have special requirements
            if opt_name == "lbfgs":
                optimizer = OptimizerRegistry.get_optimizer(
                    opt_name, simple_model.parameters(), {"lr": 1}
                )
            else:
                optimizer = OptimizerRegistry.get_optimizer(
                    opt_name, simple_model.parameters(), {"lr": 0.01}
                )

            assert isinstance(optimizer, torch.optim.Optimizer)

    def test_all_optimizers_step(self, simple_model):
        """Test that all optimizers can perform optimization step."""
        for opt_name in OptimizerRegistry.list_available():
            # Create fresh model for each optimizer
            model = nn.Linear(10, 5)

            if opt_name == "lbfgs":
                optimizer = OptimizerRegistry.get_optimizer(opt_name, model.parameters(), {"lr": 1})
            else:
                optimizer = OptimizerRegistry.get_optimizer(
                    opt_name, model.parameters(), {"lr": 0.01}
                )

            # Training step
            input_data = torch.randn(5, 10)
            target = torch.randn(5, 5)

            def closure():
                optimizer.zero_grad()
                output = model(input_data)
                loss = nn.functional.mse_loss(output, target)
                loss.backward()
                return loss

            if opt_name == "lbfgs":
                optimizer.step(closure)
            else:
                output = model(input_data)
                loss = nn.functional.mse_loss(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
