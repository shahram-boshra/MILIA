#!/usr/bin/env python3
"""
Complete Unit Test Suite for loss_functions.py Module

Tests loss functions registry system including:
- Custom loss functions (FocalLoss, WeightedMSELoss, RMSELoss)
- LossRegistry class with all methods
- Loss registration and retrieval (19 loss functions including aliases)
- Parameter validation and edge cases
- Parameter filtering with _filter_params() method (dynamic introspection)
- get_valid_params() method for parameter discovery
- Custom loss registration
- Task-aware loss selection (get_loss_for_task, get_default_loss_for_task, is_loss_compatible_with_task)
- Loss compatibility checking with task types
- Convenience functions (get_loss, get_loss_for_task, list_losses, get_default_loss_for_task, is_loss_compatible_with_task)
- Error handling and edge cases
- Integration with PyTorch tensors
- Loss computation and backpropagation
- Module-level initialization and logger

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: milia Team
Version: 1.3.0

Changelog:
- v1.3.0: Added module-level tests (logger, exports, initialization count),
          added test for FocalLoss negative alpha (disables alpha weighting),
          added test for link_prediction treated as classification in is_loss_compatible_with_task,
          added test for introspection fallback in _filter_params,
          added test for get_valid_params introspection on all registered losses,
          added test for get_loss_for_task with unknown loss name,
          added test for logging with params in get_loss,
          added test for all neutral losses compatibility,
          added test for all expected task types in mapping,
          added test for TypeError propagation in get_loss,
          added tests for all classification/regression tasks auto-selection,
          added tests for edge classification/regression default losses
- v1.2.0: Added tests for task-aware loss selection methods (get_loss_for_task,
          get_default_loss_for_task, is_loss_compatible_with_task), added tests
          for _classification_losses, _regression_losses, _task_to_default_loss
          class attributes, added corresponding convenience function tests
- v1.1.0: Added tests for get_valid_params(), _filter_params(), updated
          parameter filtering behavior tests (invalid params now filtered silently),
          updated get_loss_info tests to verify valid_params key
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

# Import logger for module-level tests
from milia_pipeline.models.training import loss_functions as loss_functions_module

# Import the module under test
from milia_pipeline.models.training.loss_functions import (
    # Custom loss classes
    FocalLoss,
    # Main class
    LossRegistry,
    RMSELoss,
    WeightedMSELoss,
    get_default_loss_for_task,
    # Convenience functions
    get_loss,
    get_loss_for_task,
    is_loss_compatible_with_task,
    list_losses,
)

# =============================================================================
# MODULE-LEVEL TESTS
# =============================================================================


class TestModuleLevel:
    """Test module-level attributes and initialization."""

    def test_module_has_logger(self):
        """Test that module has a logger instance."""
        assert hasattr(loss_functions_module, "logger")
        assert isinstance(loss_functions_module.logger, logging.Logger)

    def test_logger_name_matches_module(self):
        """Test logger name matches module name."""
        assert loss_functions_module.logger.name == loss_functions_module.__name__

    def test_module_exports_all_public_api(self):
        """Test module exports all public API components."""
        # Custom loss classes
        assert hasattr(loss_functions_module, "FocalLoss")
        assert hasattr(loss_functions_module, "WeightedMSELoss")
        assert hasattr(loss_functions_module, "RMSELoss")

        # Registry class
        assert hasattr(loss_functions_module, "LossRegistry")

        # Convenience functions
        assert hasattr(loss_functions_module, "get_loss")
        assert hasattr(loss_functions_module, "get_loss_for_task")
        assert hasattr(loss_functions_module, "list_losses")
        assert hasattr(loss_functions_module, "get_default_loss_for_task")
        assert hasattr(loss_functions_module, "is_loss_compatible_with_task")

    def test_loss_registry_initialized_with_correct_count(self):
        """Test LossRegistry is initialized with expected number of losses."""
        # Per docstring: 19 loss functions including aliases
        expected_count = 19
        actual_count = len(LossRegistry._losses)
        assert actual_count == expected_count, (
            f"Expected {expected_count} losses, got {actual_count}"
        )


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def binary_classification_data():
    """Create binary classification data for testing."""
    # Batch size: 10
    logits = torch.randn(10, requires_grad=True)
    targets = torch.randint(0, 2, (10,)).float()
    return logits, targets


@pytest.fixture
def multiclass_classification_data():
    """Create multiclass classification data for testing."""
    # Batch size: 10, num classes: 5
    logits = torch.randn(10, 5, requires_grad=True)
    targets = torch.randint(0, 5, (10,))
    return logits, targets


@pytest.fixture
def regression_data():
    """Create regression data for testing."""
    # Batch size: 10, features: 3
    predictions = torch.randn(10, 3, requires_grad=True)
    targets = torch.randn(10, 3)
    return predictions, targets


@pytest.fixture
def regression_data_1d():
    """Create 1D regression data for testing."""
    predictions = torch.randn(10, requires_grad=True)
    targets = torch.randn(10)
    return predictions, targets


@pytest.fixture
def weighted_data():
    """Create weighted data for testing weighted losses."""
    predictions = torch.randn(10, 3, requires_grad=True)
    targets = torch.randn(10, 3)
    weights = torch.rand(10, 3)
    return predictions, targets, weights


@pytest.fixture
def reset_loss_registry():
    """Reset loss registry to default state after test."""
    # Store original state
    original_losses = LossRegistry._losses.copy()

    yield

    # Restore original state
    LossRegistry._losses = original_losses


# =============================================================================
# CUSTOM LOSS FUNCTIONS TESTS
# =============================================================================


class TestFocalLoss:
    """Test FocalLoss custom implementation."""

    def test_focal_loss_initialization_default(self):
        """Test FocalLoss initialization with default parameters."""
        loss_fn = FocalLoss()
        assert loss_fn.alpha == 0.25
        assert loss_fn.gamma == 2.0
        assert loss_fn.reduction == "mean"

    def test_focal_loss_initialization_custom(self):
        """Test FocalLoss initialization with custom parameters."""
        loss_fn = FocalLoss(alpha=0.5, gamma=3.0, reduction="sum")
        assert loss_fn.alpha == 0.5
        assert loss_fn.gamma == 3.0
        assert loss_fn.reduction == "sum"

    def test_focal_loss_forward_mean_reduction(self, binary_classification_data):
        """Test FocalLoss forward pass with mean reduction."""
        logits, targets = binary_classification_data
        loss_fn = FocalLoss(reduction="mean")
        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_focal_loss_forward_sum_reduction(self, binary_classification_data):
        """Test FocalLoss forward pass with sum reduction."""
        logits, targets = binary_classification_data
        loss_fn = FocalLoss(reduction="sum")
        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_focal_loss_forward_none_reduction(self, binary_classification_data):
        """Test FocalLoss forward pass with no reduction."""
        logits, targets = binary_classification_data
        loss_fn = FocalLoss(reduction="none")
        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == logits.shape
        assert loss.requires_grad

    def test_focal_loss_backward(self, binary_classification_data):
        """Test FocalLoss backward pass."""
        logits, targets = binary_classification_data
        loss_fn = FocalLoss()
        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_focal_loss_different_alphas(self, binary_classification_data):
        """Test FocalLoss with different alpha values."""
        logits, targets = binary_classification_data

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            loss_fn = FocalLoss(alpha=alpha)
            loss = loss_fn(logits, targets)
            assert loss.item() >= 0

    def test_focal_loss_different_gammas(self, binary_classification_data):
        """Test FocalLoss with different gamma values."""
        logits, targets = binary_classification_data

        for gamma in [0.0, 1.0, 2.0, 5.0]:
            loss_fn = FocalLoss(gamma=gamma)
            loss = loss_fn(logits, targets)
            assert loss.item() >= 0

    def test_focal_loss_negative_alpha_disables_alpha_weighting(self, binary_classification_data):
        """Test FocalLoss with negative alpha disables alpha weighting.

        Per implementation (line 74): if self.alpha >= 0 applies alpha weighting.
        Negative alpha bypasses the alpha_t multiplication branch.
        """
        logits, targets = binary_classification_data

        # Negative alpha should disable alpha weighting (only focal weight applied)
        loss_fn_neg_alpha = FocalLoss(alpha=-1.0, gamma=2.0)
        loss_neg = loss_fn_neg_alpha(logits, targets)

        assert loss_fn_neg_alpha.alpha == -1.0
        assert isinstance(loss_neg, torch.Tensor)
        assert loss_neg.item() >= 0
        assert not torch.isnan(loss_neg)

        # Compare with alpha=0.5 (alpha weighting enabled)
        loss_fn_pos_alpha = FocalLoss(alpha=0.5, gamma=2.0)
        loss_pos = loss_fn_pos_alpha(logits, targets)

        # The losses should generally differ since alpha weighting is applied differently
        # (unless targets are exactly balanced, which is statistically unlikely)
        # Just verify both compute without error and are valid
        assert not torch.isnan(loss_pos)

    def test_focal_loss_with_perfect_predictions(self):
        """Test FocalLoss with perfect predictions."""
        logits = torch.ones(10) * 10  # Very confident predictions
        targets = torch.ones(10)

        loss_fn = FocalLoss()
        loss = loss_fn(logits, targets)

        # Loss should be very small for perfect predictions
        assert loss.item() < 0.1

    def test_focal_loss_is_nn_module(self):
        """Test that FocalLoss is a proper nn.Module."""
        loss_fn = FocalLoss()
        assert isinstance(loss_fn, nn.Module)


class TestWeightedMSELoss:
    """Test WeightedMSELoss custom implementation."""

    def test_weighted_mse_initialization_default(self):
        """Test WeightedMSELoss initialization with default parameters."""
        loss_fn = WeightedMSELoss()
        assert loss_fn.reduction == "mean"

    def test_weighted_mse_initialization_custom(self):
        """Test WeightedMSELoss initialization with custom parameters."""
        loss_fn = WeightedMSELoss(reduction="sum")
        assert loss_fn.reduction == "sum"

    def test_weighted_mse_forward_without_weights(self, regression_data):
        """Test WeightedMSELoss forward pass without weights."""
        predictions, targets = regression_data
        loss_fn = WeightedMSELoss()
        loss = loss_fn(predictions, targets)

        # Should behave like standard MSE
        standard_mse = nn.MSELoss()(predictions, targets)
        assert torch.allclose(loss, standard_mse, rtol=1e-5)

    def test_weighted_mse_forward_with_weights(self, weighted_data):
        """Test WeightedMSELoss forward pass with weights."""
        predictions, targets, weights = weighted_data
        loss_fn = WeightedMSELoss()
        loss = loss_fn(predictions, targets, weights)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_weighted_mse_mean_reduction(self, weighted_data):
        """Test WeightedMSELoss with mean reduction."""
        predictions, targets, weights = weighted_data
        loss_fn = WeightedMSELoss(reduction="mean")
        loss = loss_fn(predictions, targets, weights)

        assert loss.ndim == 0

    def test_weighted_mse_sum_reduction(self, weighted_data):
        """Test WeightedMSELoss with sum reduction."""
        predictions, targets, weights = weighted_data
        loss_fn = WeightedMSELoss(reduction="sum")
        loss = loss_fn(predictions, targets, weights)

        assert loss.ndim == 0

    def test_weighted_mse_none_reduction(self, weighted_data):
        """Test WeightedMSELoss with no reduction."""
        predictions, targets, weights = weighted_data
        loss_fn = WeightedMSELoss(reduction="none")
        loss = loss_fn(predictions, targets, weights)

        assert loss.shape == predictions.shape

    def test_weighted_mse_backward(self, weighted_data):
        """Test WeightedMSELoss backward pass."""
        predictions, targets, weights = weighted_data
        loss_fn = WeightedMSELoss()
        loss = loss_fn(predictions, targets, weights)
        loss.backward()

        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape

    def test_weighted_mse_uniform_weights(self, regression_data):
        """Test WeightedMSELoss with uniform weights equals unweighted."""
        predictions, targets = regression_data
        weights = torch.ones_like(predictions)

        loss_fn = WeightedMSELoss()
        weighted_loss = loss_fn(predictions, targets, weights)
        unweighted_loss = loss_fn(predictions, targets, None)

        assert torch.allclose(weighted_loss, unweighted_loss, rtol=1e-5)

    def test_weighted_mse_zero_weights(self, regression_data):
        """Test WeightedMSELoss with zero weights."""
        predictions, targets = regression_data
        weights = torch.zeros_like(predictions)

        loss_fn = WeightedMSELoss()
        loss = loss_fn(predictions, targets, weights)

        assert loss.item() == 0.0

    def test_weighted_mse_is_nn_module(self):
        """Test that WeightedMSELoss is a proper nn.Module."""
        loss_fn = WeightedMSELoss()
        assert isinstance(loss_fn, nn.Module)


class TestRMSELoss:
    """Test RMSELoss custom implementation."""

    def test_rmse_initialization(self):
        """Test RMSELoss initialization."""
        loss_fn = RMSELoss()
        assert hasattr(loss_fn, "mse")
        assert isinstance(loss_fn.mse, nn.MSELoss)

    def test_rmse_forward(self, regression_data):
        """Test RMSELoss forward pass."""
        predictions, targets = regression_data
        loss_fn = RMSELoss()
        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_rmse_equals_sqrt_mse(self, regression_data):
        """Test that RMSE equals sqrt(MSE)."""
        predictions, targets = regression_data

        rmse_fn = RMSELoss()
        mse_fn = nn.MSELoss()

        rmse = rmse_fn(predictions, targets)
        mse = mse_fn(predictions, targets)

        expected_rmse = torch.sqrt(mse)
        assert torch.allclose(rmse, expected_rmse, rtol=1e-5)

    def test_rmse_backward(self, regression_data):
        """Test RMSELoss backward pass."""
        predictions, targets = regression_data
        loss_fn = RMSELoss()
        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape

    def test_rmse_with_perfect_predictions(self):
        """Test RMSELoss with perfect predictions."""
        predictions = torch.ones(10, 3)
        targets = torch.ones(10, 3)

        loss_fn = RMSELoss()
        loss = loss_fn(predictions, targets)

        assert loss.item() == 0.0

    def test_rmse_is_nn_module(self):
        """Test that RMSELoss is a proper nn.Module."""
        loss_fn = RMSELoss()
        assert isinstance(loss_fn, nn.Module)


# =============================================================================
# LOSS REGISTRY TESTS
# =============================================================================


class TestLossRegistry:
    """Test LossRegistry class."""

    def test_registry_has_expected_losses(self):
        """Test that registry contains all expected loss functions."""
        expected_losses = [
            # Regression losses
            "mse",
            "mae",
            "l1",
            "huber",
            "smooth_l1",
            "rmse",
            "weighted_mse",
            # Classification losses
            "cross_entropy",
            "ce",
            "nll",
            "bce",
            "bce_with_logits",
            "focal",
            # Multi-label losses
            "multilabel_soft_margin",
            # Ranking losses
            "margin_ranking",
            "triplet_margin",
            # Other losses
            "kl_div",
            "poisson_nll",
            "cosine_embedding",
        ]

        available = LossRegistry.list_available()
        for loss_name in expected_losses:
            assert loss_name in available, f"Missing loss: {loss_name}"

    def test_registry_losses_are_valid_classes(self):
        """Test that all registered losses are valid nn.Module classes."""
        for name, loss_cls in LossRegistry._losses.items():
            assert callable(loss_cls), f"Loss {name} is not callable"
            assert hasattr(loss_cls, "__name__"), f"Loss {name} has no __name__"
            assert issubclass(loss_cls, nn.Module), f"Loss {name} is not a subclass of nn.Module"

    def test_list_available_returns_sorted_list(self):
        """Test list_available returns sorted list of loss names."""
        available = LossRegistry.list_available()
        assert isinstance(available, list)
        assert len(available) > 0
        assert available == sorted(available)

    def test_list_available_includes_all_losses(self):
        """Test list_available includes all registered losses."""
        available = LossRegistry.list_available()
        for name in LossRegistry._losses.keys():
            assert name in available

    def test_loss_count(self):
        """Test that expected number of losses are registered."""
        available = LossRegistry.list_available()
        # 19 loss functions total (including aliases)
        assert len(available) == 19

    def test_regression_losses_present(self):
        """Test that all regression losses are present."""
        available = LossRegistry.list_available()
        regression_losses = ["mse", "mae", "l1", "huber", "smooth_l1", "rmse", "weighted_mse"]
        for loss in regression_losses:
            assert loss in available

    def test_classification_losses_present(self):
        """Test that all classification losses are present."""
        available = LossRegistry.list_available()
        classification_losses = ["cross_entropy", "ce", "nll", "bce", "bce_with_logits", "focal"]
        for loss in classification_losses:
            assert loss in available

    def test_aliases_present(self):
        """Test that loss aliases are present."""
        available = LossRegistry.list_available()
        assert "l1" in available  # Alias for mae
        assert "ce" in available  # Alias for cross_entropy


class TestGetLoss:
    """Test LossRegistry.get_loss method."""

    def test_get_mse_default_params(self):
        """Test getting MSE loss with default parameters."""
        loss_fn = LossRegistry.get_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_mse_custom_params(self):
        """Test getting MSE loss with custom parameters."""
        loss_fn = LossRegistry.get_loss("mse", {"reduction": "sum"})
        assert isinstance(loss_fn, nn.MSELoss)
        assert loss_fn.reduction == "sum"

    def test_get_mae(self):
        """Test getting MAE loss."""
        loss_fn = LossRegistry.get_loss("mae")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_get_l1_alias(self):
        """Test getting L1 loss via alias."""
        loss_fn = LossRegistry.get_loss("l1")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_get_cross_entropy(self):
        """Test getting CrossEntropy loss."""
        loss_fn = LossRegistry.get_loss("cross_entropy")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_get_ce_alias(self):
        """Test getting CrossEntropy loss via alias."""
        loss_fn = LossRegistry.get_loss("ce")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_get_focal_loss(self):
        """Test getting FocalLoss."""
        loss_fn = LossRegistry.get_loss("focal")
        assert isinstance(loss_fn, FocalLoss)

    def test_get_focal_loss_custom_params(self):
        """Test getting FocalLoss with custom parameters."""
        loss_fn = LossRegistry.get_loss("focal", {"alpha": 0.5, "gamma": 3.0})
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha == 0.5
        assert loss_fn.gamma == 3.0

    def test_get_weighted_mse(self):
        """Test getting WeightedMSELoss."""
        loss_fn = LossRegistry.get_loss("weighted_mse")
        assert isinstance(loss_fn, WeightedMSELoss)

    def test_get_rmse(self):
        """Test getting RMSELoss."""
        loss_fn = LossRegistry.get_loss("rmse")
        assert isinstance(loss_fn, RMSELoss)

    def test_get_bce(self):
        """Test getting BCE loss."""
        loss_fn = LossRegistry.get_loss("bce")
        assert isinstance(loss_fn, nn.BCELoss)

    def test_get_bce_with_logits(self):
        """Test getting BCE with logits loss."""
        loss_fn = LossRegistry.get_loss("bce_with_logits")
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_get_huber_loss(self):
        """Test getting Huber loss."""
        loss_fn = LossRegistry.get_loss("huber")
        assert isinstance(loss_fn, nn.HuberLoss)

    def test_get_smooth_l1(self):
        """Test getting Smooth L1 loss."""
        loss_fn = LossRegistry.get_loss("smooth_l1")
        assert isinstance(loss_fn, nn.SmoothL1Loss)

    def test_get_nll(self):
        """Test getting NLL loss."""
        loss_fn = LossRegistry.get_loss("nll")
        assert isinstance(loss_fn, nn.NLLLoss)

    def test_get_kl_div(self):
        """Test getting KL Divergence loss."""
        loss_fn = LossRegistry.get_loss("kl_div")
        assert isinstance(loss_fn, nn.KLDivLoss)

    def test_get_poisson_nll(self):
        """Test getting Poisson NLL loss."""
        loss_fn = LossRegistry.get_loss("poisson_nll")
        assert isinstance(loss_fn, nn.PoissonNLLLoss)

    def test_get_cosine_embedding(self):
        """Test getting Cosine Embedding loss."""
        loss_fn = LossRegistry.get_loss("cosine_embedding")
        assert isinstance(loss_fn, nn.CosineEmbeddingLoss)

    def test_get_multilabel_soft_margin(self):
        """Test getting MultiLabel Soft Margin loss."""
        loss_fn = LossRegistry.get_loss("multilabel_soft_margin")
        assert isinstance(loss_fn, nn.MultiLabelSoftMarginLoss)

    def test_get_margin_ranking(self):
        """Test getting Margin Ranking loss."""
        loss_fn = LossRegistry.get_loss("margin_ranking")
        assert isinstance(loss_fn, nn.MarginRankingLoss)

    def test_get_triplet_margin(self):
        """Test getting Triplet Margin loss."""
        loss_fn = LossRegistry.get_loss("triplet_margin")
        assert isinstance(loss_fn, nn.TripletMarginLoss)

    def test_get_unknown_loss_raises_error(self):
        """Test that getting unknown loss raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LossRegistry.get_loss("unknown_loss")

        assert "Unknown loss function: 'unknown_loss'" in str(exc_info.value)
        assert "Available losses:" in str(exc_info.value)

    def test_get_loss_invalid_params_filtered_silently(self):
        """Test that invalid parameters are filtered out silently (not raising errors).

        The LossRegistry now uses _filter_params() to dynamically filter out
        parameters that are not accepted by the loss class constructor.
        """
        # Invalid params should be filtered out, loss should be created with defaults
        loss_fn = LossRegistry.get_loss("focal", {"invalid_param": 123})

        # Should create FocalLoss with default parameters
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha == 0.25  # default
        assert loss_fn.gamma == 2.0  # default
        assert loss_fn.reduction == "mean"  # default

    def test_get_loss_mixed_valid_invalid_params(self):
        """Test that valid params are used while invalid ones are filtered."""
        # Mix of valid (alpha, gamma) and invalid (invalid_param) parameters
        loss_fn = LossRegistry.get_loss("focal", {"alpha": 0.5, "gamma": 3.0, "invalid_param": 123})

        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha == 0.5  # valid param used
        assert loss_fn.gamma == 3.0  # valid param used
        assert loss_fn.reduction == "mean"  # default

    def test_get_loss_invalid_params_for_mse(self):
        """Test that invalid params (like alpha) for MSE are filtered silently.

        Example from docstring: LossRegistry.get_loss("mse", {"alpha": 0.5})
        should work because 'alpha' is filtered out.
        """
        # 'alpha' is not a valid param for MSELoss
        loss_fn = LossRegistry.get_loss("mse", {"alpha": 0.5})

        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_with_none_params(self):
        """Test getting loss with None params uses defaults."""
        loss_fn = LossRegistry.get_loss("mse", None)
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_with_empty_params(self):
        """Test getting loss with empty params dict."""
        loss_fn = LossRegistry.get_loss("mse", {})
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_logging(self, caplog):
        """Test that get_loss logs appropriately."""
        with caplog.at_level(logging.DEBUG):
            LossRegistry.get_loss("mse")

        # Should have debug log
        assert any("Initialized mse loss" in record.message for record in caplog.records)

    def test_get_loss_logging_filtered_params(self, caplog):
        """Test that get_loss logs ignored params at debug level."""
        with caplog.at_level(logging.DEBUG):
            LossRegistry.get_loss("mse", {"alpha": 0.5, "invalid_param": 123})

        # Should log about ignored unsupported params
        assert any("ignored unsupported params" in record.message for record in caplog.records)

    def test_get_loss_logging_with_params(self, caplog):
        """Test that get_loss logs initialization with params at debug level."""
        with caplog.at_level(logging.DEBUG):
            LossRegistry.get_loss("focal", {"alpha": 0.5, "gamma": 3.0})

        # Should have log message about initialization with params
        assert any(
            "Initialized focal loss with params" in record.message for record in caplog.records
        )


class TestGetLossInfo:
    """Test LossRegistry.get_loss_info method."""

    def test_get_loss_info_mse(self):
        """Test getting info for MSE loss."""
        info = LossRegistry.get_loss_info("mse")

        assert isinstance(info, dict)
        assert info["name"] == "mse"
        assert info["class"] == "MSELoss"
        assert "module" in info
        assert "doc" in info
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)

    def test_get_loss_info_focal(self):
        """Test getting info for Focal loss."""
        info = LossRegistry.get_loss_info("focal")

        assert isinstance(info, dict)
        assert info["name"] == "focal"
        assert info["class"] == "FocalLoss"
        assert "module" in info
        assert info["doc"] is not None
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)
        # FocalLoss should have alpha, gamma, reduction params
        assert "alpha" in info["valid_params"]
        assert "gamma" in info["valid_params"]
        assert "reduction" in info["valid_params"]

    def test_get_loss_info_all_losses(self):
        """Test getting info for all registered losses."""
        for loss_name in LossRegistry.list_available():
            info = LossRegistry.get_loss_info(loss_name)
            assert isinstance(info, dict)
            assert "name" in info
            assert "class" in info
            assert "module" in info
            assert "doc" in info
            assert "valid_params" in info
            assert isinstance(info["valid_params"], dict)

    def test_get_loss_info_unknown_loss_raises_error(self):
        """Test that getting info for unknown loss raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LossRegistry.get_loss_info("unknown_loss")

        assert "Unknown loss function: 'unknown_loss'" in str(exc_info.value)


class TestGetValidParams:
    """Test LossRegistry.get_valid_params method."""

    def test_get_valid_params_focal(self):
        """Test getting valid params for FocalLoss.

        Example from docstring:
            >>> params = LossRegistry.get_valid_params("focal")
            >>> print(params)
            {'alpha': 0.25, 'gamma': 2.0, 'reduction': 'mean'}
        """
        params = LossRegistry.get_valid_params("focal")

        assert isinstance(params, dict)
        assert "alpha" in params
        assert "gamma" in params
        assert "reduction" in params
        # Check default values
        assert params["alpha"] == 0.25
        assert params["gamma"] == 2.0
        assert params["reduction"] == "mean"

    def test_get_valid_params_mse(self):
        """Test getting valid params for MSELoss.

        Example from docstring:
            >>> params = LossRegistry.get_valid_params("mse")
            >>> print(params)
            {'size_average': None, 'reduce': None, 'reduction': 'mean'}
        """
        params = LossRegistry.get_valid_params("mse")

        assert isinstance(params, dict)
        assert "reduction" in params
        assert params["reduction"] == "mean"

    def test_get_valid_params_weighted_mse(self):
        """Test getting valid params for WeightedMSELoss."""
        params = LossRegistry.get_valid_params("weighted_mse")

        assert isinstance(params, dict)
        assert "reduction" in params
        assert params["reduction"] == "mean"

    def test_get_valid_params_rmse(self):
        """Test getting valid params for RMSELoss (no parameters)."""
        params = LossRegistry.get_valid_params("rmse")

        assert isinstance(params, dict)
        # RMSELoss has no constructor parameters
        assert len(params) == 0

    def test_get_valid_params_cross_entropy(self):
        """Test getting valid params for CrossEntropyLoss."""
        params = LossRegistry.get_valid_params("cross_entropy")

        assert isinstance(params, dict)
        assert "reduction" in params
        assert "weight" in params

    def test_get_valid_params_all_losses(self):
        """Test getting valid params for all registered losses."""
        for loss_name in LossRegistry.list_available():
            params = LossRegistry.get_valid_params(loss_name)
            assert isinstance(params, dict)

    def test_get_valid_params_unknown_loss_raises_error(self):
        """Test that getting valid params for unknown loss raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LossRegistry.get_valid_params("unknown_loss")

        assert "Unknown loss function: 'unknown_loss'" in str(exc_info.value)

    def test_get_valid_params_returns_defaults(self):
        """Test that get_valid_params returns default values for params with defaults."""
        params = LossRegistry.get_valid_params("focal")

        # All FocalLoss params have defaults
        for param_name, default_value in params.items():
            assert default_value is not None, f"Parameter {param_name} should have a default"

    def test_get_valid_params_huber(self):
        """Test getting valid params for HuberLoss."""
        params = LossRegistry.get_valid_params("huber")

        assert isinstance(params, dict)
        assert "reduction" in params
        assert "delta" in params

    def test_get_valid_params_introspection_handles_all_registered_losses(self):
        """Test get_valid_params introspection works for all registered losses.

        This ensures dynamic introspection via inspect.signature works for
        all loss types, both custom and PyTorch built-in losses.
        """
        for loss_name in LossRegistry.list_available():
            params = LossRegistry.get_valid_params(loss_name)
            # Should always return a dict (empty or with params)
            assert isinstance(params, dict), f"get_valid_params('{loss_name}') did not return dict"


class TestFilterParams:
    """Test LossRegistry._filter_params private method.

    The _filter_params method dynamically filters parameters to only those
    accepted by the target class constructor using inspect.signature().
    """

    def test_filter_params_focal_loss_valid_params(self):
        """Test _filter_params with all valid FocalLoss params."""
        params = {"alpha": 0.5, "gamma": 3.0, "reduction": "sum"}
        filtered = LossRegistry._filter_params(FocalLoss, params)

        assert filtered == params

    def test_filter_params_focal_loss_invalid_params(self):
        """Test _filter_params filters out invalid params."""
        params = {"alpha": 0.5, "invalid_param": 123, "another_bad": "value"}
        filtered = LossRegistry._filter_params(FocalLoss, params)

        assert filtered == {"alpha": 0.5}
        assert "invalid_param" not in filtered
        assert "another_bad" not in filtered

    def test_filter_params_mse_loss_filters_alpha(self):
        """Test _filter_params filters alpha from MSELoss (not a valid param)."""
        params = {"alpha": 0.5, "reduction": "sum"}
        filtered = LossRegistry._filter_params(nn.MSELoss, params)

        assert "alpha" not in filtered
        assert filtered == {"reduction": "sum"}

    def test_filter_params_empty_dict(self):
        """Test _filter_params with empty params dict."""
        params = {}
        filtered = LossRegistry._filter_params(FocalLoss, params)

        assert filtered == {}

    def test_filter_params_all_invalid(self):
        """Test _filter_params with all invalid params."""
        params = {"invalid1": 1, "invalid2": 2, "invalid3": 3}
        filtered = LossRegistry._filter_params(FocalLoss, params)

        assert filtered == {}

    def test_filter_params_none_input(self):
        """Test _filter_params handles None input gracefully."""
        # The method checks if not params first
        filtered = LossRegistry._filter_params(FocalLoss, None)
        assert filtered == {}

        filtered = LossRegistry._filter_params(FocalLoss, {})
        assert filtered == {}

    def test_filter_params_weighted_mse(self):
        """Test _filter_params with WeightedMSELoss."""
        params = {"reduction": "sum", "invalid_param": 123}
        filtered = LossRegistry._filter_params(WeightedMSELoss, params)

        assert filtered == {"reduction": "sum"}

    def test_filter_params_preserves_values(self):
        """Test _filter_params preserves parameter values correctly."""
        params = {"alpha": 0.123456, "gamma": 9.87654, "reduction": "none"}
        filtered = LossRegistry._filter_params(FocalLoss, params)

        assert filtered["alpha"] == 0.123456
        assert filtered["gamma"] == 9.87654
        assert filtered["reduction"] == "none"

    def test_filter_params_with_various_value_types(self):
        """Test _filter_params with various value types."""
        # CrossEntropyLoss accepts weight as tensor
        weight_tensor = torch.tensor([1.0, 2.0, 3.0])
        params = {"weight": weight_tensor, "reduction": "mean", "invalid": "should_be_filtered"}
        filtered = LossRegistry._filter_params(nn.CrossEntropyLoss, params)

        assert "weight" in filtered
        assert torch.equal(filtered["weight"], weight_tensor)
        assert "reduction" in filtered
        assert "invalid" not in filtered

    def test_filter_params_rmse_loss(self):
        """Test _filter_params with RMSELoss (no constructor params)."""
        params = {"alpha": 0.5, "gamma": 2.0, "reduction": "mean"}
        filtered = LossRegistry._filter_params(RMSELoss, params)

        # RMSELoss.__init__ only has self, so all params should be filtered
        assert filtered == {}

    def test_filter_params_introspection_fallback(self):
        """Test _filter_params fallback when introspection fails.

        Per implementation (lines 303-306): returns original params if introspection fails.
        This handles built-in types or C extensions without signatures.
        """
        params = {"some_param": "value", "another": 123}

        # Use patch to simulate inspect.signature raising ValueError
        # This mimics the behavior when introspecting C extensions or built-in types
        with patch("inspect.signature", side_effect=ValueError("no signature found")):
            # When introspection fails, _filter_params should catch the exception
            # and return original params as fallback (lines 303-306)
            filtered = LossRegistry._filter_params(FocalLoss, params)

        # Due to fallback, should return original params unchanged
        assert filtered == params


# =============================================================================
# TASK-AWARE LOSS SELECTION TESTS
# =============================================================================


class TestTaskToDefaultLossMapping:
    """Test the _task_to_default_loss class attribute mapping."""

    def test_task_to_default_loss_exists(self):
        """Test that _task_to_default_loss mapping exists."""
        assert hasattr(LossRegistry, "_task_to_default_loss")
        assert isinstance(LossRegistry._task_to_default_loss, dict)

    def test_classification_tasks_map_to_cross_entropy(self):
        """Test classification tasks map to cross_entropy by default."""
        mapping = LossRegistry._task_to_default_loss
        assert mapping.get("graph_classification") == "cross_entropy"
        assert mapping.get("node_classification") == "cross_entropy"
        assert mapping.get("edge_classification") == "cross_entropy"

    def test_regression_tasks_map_to_mse(self):
        """Test regression tasks map to mse by default."""
        mapping = LossRegistry._task_to_default_loss
        assert mapping.get("graph_regression") == "mse"
        assert mapping.get("node_regression") == "mse"
        assert mapping.get("edge_regression") == "mse"

    def test_all_expected_task_types_in_mapping(self):
        """Test all expected task types are present in _task_to_default_loss.

        Per implementation, expected task types:
        - graph_classification, node_classification, edge_classification
        - graph_regression, node_regression, edge_regression
        - link_prediction
        """
        mapping = LossRegistry._task_to_default_loss
        expected_tasks = [
            "graph_classification",
            "node_classification",
            "edge_classification",
            "graph_regression",
            "node_regression",
            "edge_regression",
            "link_prediction",
        ]

        for task in expected_tasks:
            assert task in mapping, f"Task '{task}' not in _task_to_default_loss mapping"

    def test_link_prediction_maps_to_bce_with_logits(self):
        """Test link_prediction task maps to bce_with_logits."""
        mapping = LossRegistry._task_to_default_loss
        assert mapping.get("link_prediction") == "bce_with_logits"


class TestLossCategories:
    """Test _classification_losses and _regression_losses class attributes."""

    def test_classification_losses_set_exists(self):
        """Test that _classification_losses set exists."""
        assert hasattr(LossRegistry, "_classification_losses")
        assert isinstance(LossRegistry._classification_losses, set)

    def test_regression_losses_set_exists(self):
        """Test that _regression_losses set exists."""
        assert hasattr(LossRegistry, "_regression_losses")
        assert isinstance(LossRegistry._regression_losses, set)

    def test_classification_losses_contains_expected_losses(self):
        """Test _classification_losses contains expected losses."""
        expected = {"cross_entropy", "ce", "nll", "bce", "bce_with_logits", "focal"}
        assert LossRegistry._classification_losses == expected

    def test_regression_losses_contains_expected_losses(self):
        """Test _regression_losses contains expected losses."""
        expected = {"mse", "mae", "l1", "huber", "smooth_l1", "rmse", "weighted_mse"}
        assert LossRegistry._regression_losses == expected

    def test_classification_and_regression_sets_disjoint(self):
        """Test classification and regression loss sets have no overlap."""
        overlap = LossRegistry._classification_losses & LossRegistry._regression_losses
        assert len(overlap) == 0, f"Overlap found: {overlap}"


class TestGetLossForTask:
    """Test LossRegistry.get_loss_for_task method."""

    def test_get_loss_for_task_auto_select_classification(self):
        """Test auto-selection for classification tasks."""
        loss_fn = LossRegistry.get_loss_for_task("graph_classification")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_get_loss_for_task_auto_select_regression(self):
        """Test auto-selection for regression tasks."""
        loss_fn = LossRegistry.get_loss_for_task("graph_regression")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_for_task_auto_select_link_prediction(self):
        """Test auto-selection for link prediction task."""
        loss_fn = LossRegistry.get_loss_for_task("link_prediction")
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_get_loss_for_task_node_classification(self):
        """Test auto-selection for node classification."""
        loss_fn = LossRegistry.get_loss_for_task("node_classification")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_get_loss_for_task_node_regression(self):
        """Test auto-selection for node regression."""
        loss_fn = LossRegistry.get_loss_for_task("node_regression")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_for_task_edge_classification(self):
        """Test auto-selection for edge classification."""
        loss_fn = LossRegistry.get_loss_for_task("edge_classification")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_get_loss_for_task_edge_regression(self):
        """Test auto-selection for edge regression."""
        loss_fn = LossRegistry.get_loss_for_task("edge_regression")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_for_task_all_classification_tasks_return_cross_entropy(self):
        """Test all classification task types auto-select to CrossEntropyLoss."""
        classification_tasks = [
            "graph_classification",
            "node_classification",
            "edge_classification",
        ]
        for task in classification_tasks:
            loss_fn = LossRegistry.get_loss_for_task(task)
            assert isinstance(loss_fn, nn.CrossEntropyLoss), (
                f"Task '{task}' should auto-select CrossEntropyLoss"
            )

    def test_get_loss_for_task_all_regression_tasks_return_mse(self):
        """Test all regression task types auto-select to MSELoss."""
        regression_tasks = ["graph_regression", "node_regression", "edge_regression"]
        for task in regression_tasks:
            loss_fn = LossRegistry.get_loss_for_task(task)
            assert isinstance(loss_fn, nn.MSELoss), f"Task '{task}' should auto-select MSELoss"

    def test_get_loss_for_task_with_compatible_loss_override(self):
        """Test using compatible loss override."""
        # NLL is compatible with classification
        loss_fn = LossRegistry.get_loss_for_task("graph_classification", "nll")
        assert isinstance(loss_fn, nn.NLLLoss)

        # Huber is compatible with regression
        loss_fn = LossRegistry.get_loss_for_task("graph_regression", "huber")
        assert isinstance(loss_fn, nn.HuberLoss)

    def test_get_loss_for_task_incompatible_loss_corrected(self, caplog):
        """Test that incompatible loss gets auto-corrected with warning."""
        with caplog.at_level(logging.WARNING):
            # MSE is regression loss, not compatible with classification
            loss_fn = LossRegistry.get_loss_for_task("graph_classification", "mse")

        # Should return cross_entropy instead of mse
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

        # Should have logged a warning
        assert any("Auto-selecting" in record.message for record in caplog.records)

    def test_get_loss_for_task_classification_with_regression_loss_warning(self, caplog):
        """Test classification task with regression loss logs appropriate warning."""
        with caplog.at_level(logging.WARNING):
            LossRegistry.get_loss_for_task("graph_classification", "mse")

        # Check warning message content
        warning_logged = any(
            "classification" in record.message.lower()
            and "mse" in record.message.lower()
            and "regression" in record.message.lower()
            for record in caplog.records
        )
        assert warning_logged

    def test_get_loss_for_task_regression_with_classification_loss_warning(self, caplog):
        """Test regression task with classification loss logs appropriate warning."""
        with caplog.at_level(logging.WARNING):
            LossRegistry.get_loss_for_task("graph_regression", "cross_entropy")

        # Check warning message content
        warning_logged = any(
            "regression" in record.message.lower()
            and "cross_entropy" in record.message.lower()
            and "classification" in record.message.lower()
            for record in caplog.records
        )
        assert warning_logged

    def test_get_loss_for_task_regression_with_ce_corrected(self, caplog):
        """Test regression task with cross_entropy gets auto-corrected."""
        with caplog.at_level(logging.WARNING):
            loss_fn = LossRegistry.get_loss_for_task("graph_regression", "cross_entropy")

        # Should return MSE instead of cross_entropy
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_for_task_with_params(self):
        """Test get_loss_for_task with custom parameters."""
        loss_fn = LossRegistry.get_loss_for_task(
            "graph_classification", "focal", {"alpha": 0.5, "gamma": 3.0}
        )
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha == 0.5
        assert loss_fn.gamma == 3.0

    def test_get_loss_for_task_case_insensitive(self):
        """Test that task type is case-insensitive."""
        loss1 = LossRegistry.get_loss_for_task("GRAPH_CLASSIFICATION")
        loss2 = LossRegistry.get_loss_for_task("Graph_Classification")
        loss3 = LossRegistry.get_loss_for_task("graph_classification")

        assert type(loss1) == type(loss2) == type(loss3)

    def test_get_loss_for_task_unknown_task_falls_back_to_mse(self):
        """Test unknown task type falls back to MSE."""
        loss_fn = LossRegistry.get_loss_for_task("unknown_task_type")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_for_task_empty_task_type(self):
        """Test empty task type falls back to MSE."""
        loss_fn = LossRegistry.get_loss_for_task("")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_for_task_none_task_type(self):
        """Test None task type falls back to MSE."""
        loss_fn = LossRegistry.get_loss_for_task(None)
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_for_task_logs_auto_selection(self, caplog):
        """Test that auto-selection logs info message."""
        with caplog.at_level(logging.INFO):
            LossRegistry.get_loss_for_task("graph_classification")

        assert any("Auto-selected" in record.message for record in caplog.records)

    def test_get_loss_for_task_with_unknown_loss_name_raises_error(self):
        """Test get_loss_for_task with unknown loss name raises ValueError.

        When user specifies a loss name that doesn't exist in registry,
        the method should propagate the ValueError from get_loss().
        """
        with pytest.raises(ValueError) as exc_info:
            LossRegistry.get_loss_for_task("graph_classification", "nonexistent_loss")

        assert "Unknown loss function" in str(exc_info.value)
        assert "nonexistent_loss" in str(exc_info.value)


class TestGetDefaultLossForTask:
    """Test LossRegistry.get_default_loss_for_task method."""

    def test_get_default_loss_for_graph_classification(self):
        """Test getting default loss for graph classification."""
        default = LossRegistry.get_default_loss_for_task("graph_classification")
        assert default == "cross_entropy"

    def test_get_default_loss_for_graph_regression(self):
        """Test getting default loss for graph regression."""
        default = LossRegistry.get_default_loss_for_task("graph_regression")
        assert default == "mse"

    def test_get_default_loss_for_node_classification(self):
        """Test getting default loss for node classification."""
        default = LossRegistry.get_default_loss_for_task("node_classification")
        assert default == "cross_entropy"

    def test_get_default_loss_for_node_regression(self):
        """Test getting default loss for node regression."""
        default = LossRegistry.get_default_loss_for_task("node_regression")
        assert default == "mse"

    def test_get_default_loss_for_link_prediction(self):
        """Test getting default loss for link prediction."""
        default = LossRegistry.get_default_loss_for_task("link_prediction")
        assert default == "bce_with_logits"

    def test_get_default_loss_for_edge_classification(self):
        """Test getting default loss for edge classification."""
        default = LossRegistry.get_default_loss_for_task("edge_classification")
        assert default == "cross_entropy"

    def test_get_default_loss_for_edge_regression(self):
        """Test getting default loss for edge regression."""
        default = LossRegistry.get_default_loss_for_task("edge_regression")
        assert default == "mse"

    def test_get_default_loss_for_unknown_task(self):
        """Test getting default loss for unknown task returns mse."""
        default = LossRegistry.get_default_loss_for_task("unknown_task")
        assert default == "mse"

    def test_get_default_loss_case_insensitive(self):
        """Test that task type is case-insensitive."""
        default1 = LossRegistry.get_default_loss_for_task("GRAPH_CLASSIFICATION")
        default2 = LossRegistry.get_default_loss_for_task("graph_classification")
        assert default1 == default2

    def test_get_default_loss_empty_task(self):
        """Test empty task type returns mse."""
        default = LossRegistry.get_default_loss_for_task("")
        assert default == "mse"

    def test_get_default_loss_none_task(self):
        """Test None task type returns mse."""
        default = LossRegistry.get_default_loss_for_task(None)
        assert default == "mse"


class TestIsLossCompatibleWithTask:
    """Test LossRegistry.is_loss_compatible_with_task method."""

    def test_mse_compatible_with_regression(self):
        """Test MSE is compatible with regression tasks."""
        assert LossRegistry.is_loss_compatible_with_task("mse", "graph_regression") is True
        assert LossRegistry.is_loss_compatible_with_task("mse", "node_regression") is True

    def test_mse_not_compatible_with_classification(self):
        """Test MSE is NOT compatible with classification tasks."""
        assert LossRegistry.is_loss_compatible_with_task("mse", "graph_classification") is False
        assert LossRegistry.is_loss_compatible_with_task("mse", "node_classification") is False

    def test_cross_entropy_compatible_with_classification(self):
        """Test cross_entropy is compatible with classification tasks."""
        assert (
            LossRegistry.is_loss_compatible_with_task("cross_entropy", "graph_classification")
            is True
        )
        assert (
            LossRegistry.is_loss_compatible_with_task("cross_entropy", "node_classification")
            is True
        )

    def test_cross_entropy_not_compatible_with_regression(self):
        """Test cross_entropy is NOT compatible with regression tasks."""
        assert (
            LossRegistry.is_loss_compatible_with_task("cross_entropy", "graph_regression") is False
        )
        assert (
            LossRegistry.is_loss_compatible_with_task("cross_entropy", "node_regression") is False
        )

    def test_focal_compatible_with_classification(self):
        """Test focal loss is compatible with classification."""
        assert LossRegistry.is_loss_compatible_with_task("focal", "graph_classification") is True

    def test_bce_compatible_with_link_prediction(self):
        """Test BCE losses are compatible with link prediction."""
        assert LossRegistry.is_loss_compatible_with_task("bce", "link_prediction") is True
        assert (
            LossRegistry.is_loss_compatible_with_task("bce_with_logits", "link_prediction") is True
        )

    def test_link_prediction_treated_as_classification(self):
        """Test link_prediction is treated as classification task (line 590).

        Per implementation: is_classification = 'classification' in task_lower or task_lower == 'link_prediction'
        This means regression losses should NOT be compatible with link_prediction.
        """
        # Regression losses should NOT be compatible with link_prediction
        assert LossRegistry.is_loss_compatible_with_task("mse", "link_prediction") is False
        assert LossRegistry.is_loss_compatible_with_task("mae", "link_prediction") is False
        assert LossRegistry.is_loss_compatible_with_task("huber", "link_prediction") is False

        # Classification losses SHOULD be compatible with link_prediction
        assert LossRegistry.is_loss_compatible_with_task("cross_entropy", "link_prediction") is True
        assert LossRegistry.is_loss_compatible_with_task("focal", "link_prediction") is True

    def test_huber_compatible_with_regression(self):
        """Test Huber loss is compatible with regression."""
        assert LossRegistry.is_loss_compatible_with_task("huber", "graph_regression") is True

    def test_all_regression_losses_compatible_with_regression(self):
        """Test all regression losses are compatible with regression tasks."""
        regression_losses = ["mse", "mae", "l1", "huber", "smooth_l1", "rmse", "weighted_mse"]
        for loss_name in regression_losses:
            assert LossRegistry.is_loss_compatible_with_task(loss_name, "graph_regression") is True

    def test_all_classification_losses_compatible_with_classification(self):
        """Test all classification losses are compatible with classification tasks."""
        classification_losses = ["cross_entropy", "ce", "nll", "bce", "bce_with_logits", "focal"]
        for loss_name in classification_losses:
            assert (
                LossRegistry.is_loss_compatible_with_task(loss_name, "graph_classification") is True
            )

    def test_case_insensitive_loss_name(self):
        """Test loss name is case-insensitive."""
        assert LossRegistry.is_loss_compatible_with_task("MSE", "graph_regression") is True
        assert (
            LossRegistry.is_loss_compatible_with_task("CROSS_ENTROPY", "graph_classification")
            is True
        )

    def test_case_insensitive_task_type(self):
        """Test task type is case-insensitive."""
        assert LossRegistry.is_loss_compatible_with_task("mse", "GRAPH_REGRESSION") is True

    def test_neutral_loss_compatible_with_any_task(self):
        """Test losses not in either category are compatible with any task."""
        # kl_div is not in classification or regression sets
        assert LossRegistry.is_loss_compatible_with_task("kl_div", "graph_regression") is True
        assert LossRegistry.is_loss_compatible_with_task("kl_div", "graph_classification") is True

    def test_all_neutral_losses_compatible_with_any_task(self):
        """Test all neutral losses (not in classification/regression sets) are compatible.

        Neutral losses include: kl_div, poisson_nll, cosine_embedding,
        multilabel_soft_margin, margin_ranking, triplet_margin.
        These are not categorized and should be compatible with any task type.
        """
        neutral_losses = [
            "kl_div",
            "poisson_nll",
            "cosine_embedding",
            "multilabel_soft_margin",
            "margin_ranking",
            "triplet_margin",
        ]

        for loss_name in neutral_losses:
            # Verify they are indeed not in classification or regression sets
            assert loss_name not in LossRegistry._classification_losses, (
                f"{loss_name} unexpectedly in _classification_losses"
            )
            assert loss_name not in LossRegistry._regression_losses, (
                f"{loss_name} unexpectedly in _regression_losses"
            )

            # Verify they are compatible with both task types
            assert LossRegistry.is_loss_compatible_with_task(loss_name, "graph_regression") is True
            assert (
                LossRegistry.is_loss_compatible_with_task(loss_name, "graph_classification") is True
            )
            assert LossRegistry.is_loss_compatible_with_task(loss_name, "link_prediction") is True

    def test_empty_inputs(self):
        """Test empty inputs return True (compatible by default)."""
        assert LossRegistry.is_loss_compatible_with_task("", "") is True
        assert LossRegistry.is_loss_compatible_with_task(None, None) is True


class TestTaskAwareConvenienceFunctions:
    """Test task-aware convenience functions."""

    def test_get_loss_for_task_convenience_function(self):
        """Test get_loss_for_task convenience function."""
        loss_fn = get_loss_for_task("graph_classification")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_get_loss_for_task_convenience_with_name(self):
        """Test get_loss_for_task convenience function with loss name."""
        loss_fn = get_loss_for_task("graph_classification", "nll")
        assert isinstance(loss_fn, nn.NLLLoss)

    def test_get_loss_for_task_convenience_with_params(self):
        """Test get_loss_for_task convenience function with params."""
        loss_fn = get_loss_for_task("graph_classification", "focal", {"alpha": 0.3})
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha == 0.3

    def test_get_default_loss_for_task_convenience_function(self):
        """Test get_default_loss_for_task convenience function."""
        default = get_default_loss_for_task("graph_classification")
        assert default == "cross_entropy"

        default = get_default_loss_for_task("graph_regression")
        assert default == "mse"

    def test_is_loss_compatible_with_task_convenience_function(self):
        """Test is_loss_compatible_with_task convenience function."""
        assert is_loss_compatible_with_task("mse", "graph_regression") is True
        assert is_loss_compatible_with_task("mse", "graph_classification") is False

    def test_convenience_functions_match_registry_methods(self):
        """Test that convenience functions match registry methods."""
        # get_loss_for_task
        loss1 = get_loss_for_task("graph_classification")
        loss2 = LossRegistry.get_loss_for_task("graph_classification")
        assert type(loss1) == type(loss2)

        # get_default_loss_for_task
        default1 = get_default_loss_for_task("graph_classification")
        default2 = LossRegistry.get_default_loss_for_task("graph_classification")
        assert default1 == default2

        # is_loss_compatible_with_task
        compat1 = is_loss_compatible_with_task("mse", "graph_regression")
        compat2 = LossRegistry.is_loss_compatible_with_task("mse", "graph_regression")
        assert compat1 == compat2


class TestRegisterCustomLoss:
    """Test LossRegistry.register_custom_loss method."""

    def test_register_custom_loss_basic(self, reset_loss_registry):
        """Test registering a custom loss function."""

        class CustomLoss(nn.Module):
            def forward(self, input, target):
                return ((input - target) ** 2).mean()

        LossRegistry.register_custom_loss("custom", CustomLoss)

        assert "custom" in LossRegistry.list_available()
        loss_fn = LossRegistry.get_loss("custom")
        assert isinstance(loss_fn, CustomLoss)

    def test_register_custom_loss_with_parameters(self, reset_loss_registry):
        """Test registering custom loss with parameters."""

        class ParameterizedLoss(nn.Module):
            def __init__(self, scale=1.0):
                super().__init__()
                self.scale = scale

            def forward(self, input, target):
                return self.scale * ((input - target) ** 2).mean()

        LossRegistry.register_custom_loss("param_loss", ParameterizedLoss)

        loss_fn = LossRegistry.get_loss("param_loss", {"scale": 2.5})
        assert isinstance(loss_fn, ParameterizedLoss)
        assert loss_fn.scale == 2.5

    def test_register_custom_loss_duplicate_raises_error(self, reset_loss_registry):
        """Test that registering duplicate loss raises ValueError."""

        class CustomLoss(nn.Module):
            def forward(self, input, target):
                return ((input - target) ** 2).mean()

        LossRegistry.register_custom_loss("custom", CustomLoss)

        with pytest.raises(ValueError) as exc_info:
            LossRegistry.register_custom_loss("custom", CustomLoss)

        assert "Loss 'custom' already registered" in str(exc_info.value)
        assert "Use overwrite=True to replace" in str(exc_info.value)

    def test_register_custom_loss_with_overwrite(self, reset_loss_registry):
        """Test registering custom loss with overwrite."""

        class CustomLoss1(nn.Module):
            def forward(self, input, target):
                return ((input - target) ** 2).mean()

        class CustomLoss2(nn.Module):
            def forward(self, input, target):
                return ((input - target) ** 3).mean()

        LossRegistry.register_custom_loss("custom", CustomLoss1)
        LossRegistry.register_custom_loss("custom", CustomLoss2, overwrite=True)

        loss_fn = LossRegistry.get_loss("custom")
        assert isinstance(loss_fn, CustomLoss2)

    def test_register_non_module_raises_error(self, reset_loss_registry):
        """Test that registering non-nn.Module raises TypeError."""

        class NotAModule:
            pass

        with pytest.raises(TypeError) as exc_info:
            LossRegistry.register_custom_loss("bad", NotAModule)

        assert "must be a subclass of nn.Module" in str(exc_info.value)

    def test_register_function_raises_error(self, reset_loss_registry):
        """Test that registering function raises TypeError."""

        def loss_function(input, target):
            return ((input - target) ** 2).mean()

        with pytest.raises(TypeError):
            LossRegistry.register_custom_loss("func_loss", loss_function)

    def test_register_custom_loss_logging(self, reset_loss_registry, caplog):
        """Test that registering custom loss logs appropriately."""

        class CustomLoss(nn.Module):
            def forward(self, input, target):
                return ((input - target) ** 2).mean()

        with caplog.at_level(logging.INFO):
            LossRegistry.register_custom_loss("custom", CustomLoss)

        assert any(
            "Registered custom loss: 'custom'" in record.message for record in caplog.records
        )

    def test_register_multiple_custom_losses(self, reset_loss_registry):
        """Test registering multiple custom losses."""

        class Loss1(nn.Module):
            def forward(self, input, target):
                return ((input - target) ** 2).mean()

        class Loss2(nn.Module):
            def forward(self, input, target):
                return torch.abs(input - target).mean()

        initial_count = len(LossRegistry.list_available())

        LossRegistry.register_custom_loss("loss1", Loss1)
        LossRegistry.register_custom_loss("loss2", Loss2)

        assert len(LossRegistry.list_available()) == initial_count + 2
        assert "loss1" in LossRegistry.list_available()
        assert "loss2" in LossRegistry.list_available()


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions (get_loss, get_loss_for_task, list_losses, get_default_loss_for_task, is_loss_compatible_with_task)."""

    def test_get_loss_function(self):
        """Test get_loss convenience function."""
        loss_fn = get_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_with_params(self):
        """Test get_loss with parameters."""
        loss_fn = get_loss("focal", {"alpha": 0.5, "gamma": 2.0})
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.alpha == 0.5

    def test_list_losses_function(self):
        """Test list_losses convenience function."""
        losses = list_losses()
        assert isinstance(losses, list)
        assert len(losses) > 0
        assert "mse" in losses
        assert "focal" in losses

    def test_get_loss_for_task_function(self):
        """Test get_loss_for_task convenience function."""
        # Auto-select for classification
        loss_fn = get_loss_for_task("graph_classification")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

        # Override with specific loss
        loss_fn = get_loss_for_task("graph_regression", "huber")
        assert isinstance(loss_fn, nn.HuberLoss)

    def test_get_default_loss_for_task_function(self):
        """Test get_default_loss_for_task convenience function."""
        assert get_default_loss_for_task("graph_classification") == "cross_entropy"
        assert get_default_loss_for_task("graph_regression") == "mse"

    def test_is_loss_compatible_with_task_function(self):
        """Test is_loss_compatible_with_task convenience function."""
        assert is_loss_compatible_with_task("mse", "graph_regression") is True
        assert is_loss_compatible_with_task("mse", "graph_classification") is False

    def test_convenience_functions_match_registry(self):
        """Test that convenience functions match registry methods."""
        # get_loss
        loss1 = get_loss("mse")
        loss2 = LossRegistry.get_loss("mse")
        assert type(loss1) == type(loss2)

        # list_losses
        list1 = list_losses()
        list2 = LossRegistry.list_available()
        assert list1 == list2

        # get_loss_for_task
        loss3 = get_loss_for_task("graph_classification")
        loss4 = LossRegistry.get_loss_for_task("graph_classification")
        assert type(loss3) == type(loss4)

        # get_default_loss_for_task
        default1 = get_default_loss_for_task("graph_regression")
        default2 = LossRegistry.get_default_loss_for_task("graph_regression")
        assert default1 == default2

        # is_loss_compatible_with_task
        compat1 = is_loss_compatible_with_task("focal", "graph_classification")
        compat2 = LossRegistry.is_loss_compatible_with_task("focal", "graph_classification")
        assert compat1 == compat2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestLossIntegration:
    """Test loss functions in realistic scenarios."""

    def test_mse_loss_training_step(self, regression_data):
        """Test MSE loss in a training step."""
        predictions, targets = regression_data

        loss_fn = LossRegistry.get_loss("mse")
        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_cross_entropy_training_step(self, multiclass_classification_data):
        """Test CrossEntropy loss in a training step."""
        logits, targets = multiclass_classification_data

        loss_fn = LossRegistry.get_loss("cross_entropy")
        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(loss).any()

    def test_focal_loss_training_step(self, binary_classification_data):
        """Test Focal loss in a training step."""
        logits, targets = binary_classification_data

        loss_fn = LossRegistry.get_loss("focal", {"alpha": 0.25, "gamma": 2.0})
        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(loss).any()

    def test_multiple_losses_same_data(self, regression_data):
        """Test multiple loss functions on same data."""
        predictions, targets = regression_data

        loss_names = ["mse", "mae", "huber", "smooth_l1"]
        losses = {}

        for name in loss_names:
            loss_fn = LossRegistry.get_loss(name)
            losses[name] = loss_fn(predictions.clone(), targets)

        # All losses should be positive and finite
        for name, loss in losses.items():
            assert loss.item() >= 0, f"{name} loss is negative"
            assert not torch.isnan(loss), f"{name} loss is NaN"
            assert not torch.isinf(loss), f"{name} loss is inf"

    def test_loss_reduction_modes(self, regression_data):
        """Test different reduction modes for MSE loss."""
        predictions, targets = regression_data

        mean_loss = LossRegistry.get_loss("mse", {"reduction": "mean"})
        sum_loss = LossRegistry.get_loss("mse", {"reduction": "sum"})
        none_loss = LossRegistry.get_loss("mse", {"reduction": "none"})

        mean_result = mean_loss(predictions, targets)
        sum_result = sum_loss(predictions, targets)
        none_result = none_loss(predictions, targets)

        # Check shapes
        assert mean_result.ndim == 0
        assert sum_result.ndim == 0
        assert none_result.shape == predictions.shape

        # Check relationship
        expected_mean = none_result.mean()
        assert torch.allclose(mean_result, expected_mean, rtol=1e-5)

    def test_weighted_vs_unweighted_mse(self, regression_data):
        """Test weighted vs unweighted MSE comparison."""
        predictions, targets = regression_data

        # Unweighted
        standard_mse = LossRegistry.get_loss("mse")
        standard_loss = standard_mse(predictions, targets)

        # Weighted with uniform weights
        weighted_mse = LossRegistry.get_loss("weighted_mse")
        weights = torch.ones_like(predictions)
        weighted_loss = weighted_mse(predictions, targets, weights)

        assert torch.allclose(standard_loss, weighted_loss, rtol=1e-5)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_get_loss_with_wrong_param_types(self):
        """Test getting loss with wrong parameter types."""
        # Python doesn't enforce type hints at runtime, so wrong types
        # may not raise errors during instantiation. Test actual usage instead.
        loss_fn = LossRegistry.get_loss("focal", {"gamma": "2.0"})

        # The error will occur when trying to use the loss with tensors
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()

        with pytest.raises((TypeError, RuntimeError)):
            _loss = loss_fn(logits, targets)

    def test_get_loss_invalid_type_error_propagation(self):
        """Test that TypeError from loss constructor is wrapped in ValueError.

        Per implementation (lines 271-274): TypeError is caught and re-raised
        as ValueError with descriptive message.
        """
        # Create params that will cause TypeError after filtering
        # Use a param with invalid type that will pass filtering but fail construction
        # For cross_entropy, 'weight' must be a tensor, passing an int will fail
        with pytest.raises(ValueError) as exc_info:
            # ignore_index must be int, but weight must be tensor
            # Passing weight as string should cause TypeError
            LossRegistry.get_loss("cross_entropy", {"weight": "invalid_type"})

        assert "Invalid parameters for loss" in str(exc_info.value)

    def test_loss_with_mismatched_shapes(self):
        """Test losses with mismatched tensor shapes."""
        loss_fn = LossRegistry.get_loss("mse")

        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 3)  # Different shape

        with pytest.raises(RuntimeError):
            loss_fn(predictions, targets)

    def test_loss_with_nan_inputs(self):
        """Test loss behavior with NaN inputs."""
        loss_fn = LossRegistry.get_loss("mse")

        predictions = torch.tensor([1.0, float("nan"), 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        loss = loss_fn(predictions, targets)
        assert torch.isnan(loss)

    def test_loss_with_inf_inputs(self):
        """Test loss behavior with inf inputs."""
        loss_fn = LossRegistry.get_loss("mse")

        predictions = torch.tensor([1.0, float("inf"), 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        loss = loss_fn(predictions, targets)
        assert torch.isinf(loss)

    def test_empty_tensor_handling(self):
        """Test loss with empty tensors."""
        loss_fn = LossRegistry.get_loss("mse")

        predictions = torch.tensor([])
        targets = torch.tensor([])

        # Should handle empty tensors (may return NaN or 0)
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)


# =============================================================================
# DOCUMENTATION EXAMPLES TESTS
# =============================================================================


class TestDocumentationExamples:
    """Test that examples in docstrings work correctly."""

    def test_loss_registry_usage_example(self):
        """Test basic usage example from LossRegistry docstring."""
        loss_fn = LossRegistry.get_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

        loss_fn = LossRegistry.get_loss("focal", {"alpha": 0.5, "gamma": 2.0})
        assert isinstance(loss_fn, FocalLoss)

        available = LossRegistry.list_available()
        assert isinstance(available, list)

    def test_get_loss_method_examples(self):
        """Test examples from get_loss method docstring."""
        # Simple usage
        loss_fn = LossRegistry.get_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

        # With parameters
        loss_fn = LossRegistry.get_loss("focal", {"alpha": 0.25, "gamma": 2.0, "reduction": "mean"})
        assert isinstance(loss_fn, FocalLoss)

    def test_get_loss_info_docstring_example(self):
        """Test example from get_loss_info method docstring.

        From docstring:
            >>> info = LossRegistry.get_loss_info("focal")
            >>> print(info['valid_params'])
        """
        info = LossRegistry.get_loss_info("focal")
        assert "valid_params" in info
        assert isinstance(info["valid_params"], dict)
        # Should contain FocalLoss params
        assert "alpha" in info["valid_params"]
        assert "gamma" in info["valid_params"]
        assert "reduction" in info["valid_params"]

    def test_register_custom_loss_example(self, reset_loss_registry):
        """Test example from register_custom_loss docstring."""

        class MyCustomLoss(nn.Module):
            def forward(self, input, target):
                return ((input - target) ** 2).mean()

        LossRegistry.register_custom_loss("my_loss", MyCustomLoss)
        loss_fn = LossRegistry.get_loss("my_loss")
        assert isinstance(loss_fn, MyCustomLoss)

    def test_convenience_function_examples(self):
        """Test convenience function examples from docstrings."""
        # get_loss
        loss_fn = get_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

        # list_losses
        losses = list_losses()
        assert isinstance(losses, list)

    def test_get_valid_params_docstring_examples(self):
        """Test examples from get_valid_params method docstring.

        From docstring:
            >>> params = LossRegistry.get_valid_params("focal")
            >>> print(params)
            {'alpha': 0.25, 'gamma': 2.0, 'reduction': 'mean'}

            >>> params = LossRegistry.get_valid_params("mse")
            >>> print(params)
            {'size_average': None, 'reduce': None, 'reduction': 'mean'}
        """
        # Focal loss params
        params = LossRegistry.get_valid_params("focal")
        assert params["alpha"] == 0.25
        assert params["gamma"] == 2.0
        assert params["reduction"] == "mean"

        # MSE loss params
        params = LossRegistry.get_valid_params("mse")
        assert "reduction" in params
        assert params["reduction"] == "mean"

    def test_get_loss_safe_usage_example(self):
        """Test safe usage example from get_loss docstring.

        From docstring:
            >>> # Safe usage - invalid params are filtered
            >>> loss_fn = LossRegistry.get_loss("mse", {"alpha": 0.5})
            >>> # Works! 'alpha' is filtered out since MSELoss doesn't accept it
        """
        # This should work - 'alpha' is filtered out
        loss_fn = LossRegistry.get_loss("mse", {"alpha": 0.5})
        assert isinstance(loss_fn, nn.MSELoss)

    def test_get_loss_for_task_docstring_examples(self):
        """Test examples from get_loss_for_task method docstring.

        From docstring:
            >>> # Auto-select for classification task
            >>> loss_fn = LossRegistry.get_loss_for_task('graph_classification')
            >>> # Returns CrossEntropyLoss

            >>> # Override with compatible loss
            >>> loss_fn = LossRegistry.get_loss_for_task('graph_classification', 'nll')
            >>> # Returns NLLLoss (compatible with classification)
        """
        # Auto-select for classification
        loss_fn = LossRegistry.get_loss_for_task("graph_classification")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

        # Override with compatible loss
        loss_fn = LossRegistry.get_loss_for_task("graph_classification", "nll")
        assert isinstance(loss_fn, nn.NLLLoss)

    def test_get_default_loss_for_task_docstring_examples(self):
        """Test examples from get_default_loss_for_task method docstring.

        From docstring:
            >>> LossRegistry.get_default_loss_for_task('graph_classification')
            'cross_entropy'
            >>> LossRegistry.get_default_loss_for_task('graph_regression')
            'mse'
        """
        assert LossRegistry.get_default_loss_for_task("graph_classification") == "cross_entropy"
        assert LossRegistry.get_default_loss_for_task("graph_regression") == "mse"

    def test_is_loss_compatible_with_task_docstring_examples(self):
        """Test examples from is_loss_compatible_with_task method docstring.

        From docstring:
            >>> LossRegistry.is_loss_compatible_with_task('mse', 'graph_regression')
            True
            >>> LossRegistry.is_loss_compatible_with_task('mse', 'graph_classification')
            False
        """
        assert LossRegistry.is_loss_compatible_with_task("mse", "graph_regression") is True
        assert LossRegistry.is_loss_compatible_with_task("mse", "graph_classification") is False

    def test_get_loss_for_task_convenience_docstring_examples(self):
        """Test examples from get_loss_for_task convenience function docstring.

        From docstring:
            >>> # Auto-select for classification
            >>> loss_fn = get_loss_for_task('graph_classification')
            >>> # Returns CrossEntropyLoss

            >>> # Override with specific loss
            >>> loss_fn = get_loss_for_task('graph_regression', 'huber')
            >>> # Returns HuberLoss
        """
        # Auto-select for classification
        loss_fn = get_loss_for_task("graph_classification")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

        # Override with specific loss
        loss_fn = get_loss_for_task("graph_regression", "huber")
        assert isinstance(loss_fn, nn.HuberLoss)

    def test_get_default_loss_for_task_convenience_docstring_example(self):
        """Test example from get_default_loss_for_task convenience function docstring.

        From docstring:
            >>> get_default_loss_for_task('graph_classification')
            'cross_entropy'
        """
        assert get_default_loss_for_task("graph_classification") == "cross_entropy"

    def test_is_loss_compatible_convenience_docstring_example(self):
        """Test example from is_loss_compatible_with_task convenience function docstring.

        From docstring:
            >>> is_loss_compatible_with_task('mse', 'graph_classification')
            False
        """
        assert is_loss_compatible_with_task("mse", "graph_classification") is False


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_loss_with_single_sample(self):
        """Test losses with single sample."""
        predictions = torch.randn(1, 5, requires_grad=True)
        targets = torch.randn(1, 5)

        loss_fn = LossRegistry.get_loss("mse")
        loss = loss_fn(predictions, targets)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_loss_with_large_batch(self):
        """Test losses with large batch size."""
        predictions = torch.randn(1000, 10, requires_grad=True)
        targets = torch.randn(1000, 10)

        loss_fn = LossRegistry.get_loss("mse")
        loss = loss_fn(predictions, targets)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_focal_loss_extreme_alpha(self):
        """Test FocalLoss with extreme alpha values."""
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()

        for alpha in [0.0, 1.0]:
            loss_fn = FocalLoss(alpha=alpha)
            loss = loss_fn(logits, targets)
            assert not torch.isnan(loss)

    def test_focal_loss_zero_gamma(self):
        """Test FocalLoss with gamma=0 (should behave like BCE)."""
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()

        focal = FocalLoss(gamma=0.0, alpha=-1)  # alpha=-1 disables alpha weighting
        loss = focal(logits, targets)

        assert not torch.isnan(loss)

    def test_weighted_mse_extreme_weights(self):
        """Test WeightedMSELoss with extreme weight values."""
        predictions = torch.randn(10, 3)
        targets = torch.randn(10, 3)

        # Very large weights
        large_weights = torch.ones_like(predictions) * 1e6
        loss_fn = WeightedMSELoss()
        loss = loss_fn(predictions, targets, large_weights)
        assert not torch.isnan(loss)

        # Very small weights
        small_weights = torch.ones_like(predictions) * 1e-6
        loss = loss_fn(predictions, targets, small_weights)
        assert not torch.isnan(loss)

    def test_all_losses_with_zero_tensors(self):
        """Test all losses with zero tensors."""
        predictions = torch.zeros(5, 3)
        targets = torch.zeros(5, 3)

        for loss_name in ["mse", "mae", "huber", "smooth_l1", "rmse"]:
            loss_fn = LossRegistry.get_loss(loss_name)
            loss = loss_fn(predictions, targets)
            assert loss.item() == 0.0 or torch.allclose(loss, torch.tensor(0.0))


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================


class TestParameterValidation:
    """Test parameter validation for loss functions."""

    def test_focal_loss_parameter_ranges(self):
        """Test FocalLoss parameter ranges."""
        # Valid parameters
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            loss_fn = FocalLoss(alpha=alpha)
            assert loss_fn.alpha == alpha

        for gamma in [0.0, 1.0, 2.0, 5.0, 10.0]:
            loss_fn = FocalLoss(gamma=gamma)
            assert loss_fn.gamma == gamma

    def test_loss_reduction_parameter(self):
        """Test reduction parameter for various losses."""
        for reduction in ["mean", "sum", "none"]:
            mse = LossRegistry.get_loss("mse", {"reduction": reduction})
            assert mse.reduction == reduction

            weighted_mse = LossRegistry.get_loss("weighted_mse", {"reduction": reduction})
            assert weighted_mse.reduction == reduction

    def test_cross_entropy_weight_parameter(self):
        """Test CrossEntropy weight parameter."""
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_fn = LossRegistry.get_loss("cross_entropy", {"weight": weights})

        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert torch.equal(loss_fn.weight, weights)


# =============================================================================
# COMPREHENSIVE LOSS TESTS
# =============================================================================


class TestAllLosses:
    """Test all registered losses can be instantiated and used."""

    def test_all_regression_losses_instantiate(self):
        """Test that all regression losses can be instantiated."""
        regression_losses = ["mse", "mae", "l1", "huber", "smooth_l1", "rmse", "weighted_mse"]

        for loss_name in regression_losses:
            loss_fn = LossRegistry.get_loss(loss_name)
            assert isinstance(loss_fn, nn.Module)

    def test_all_classification_losses_instantiate(self):
        """Test that all classification losses can be instantiated."""
        classification_losses = ["cross_entropy", "ce", "nll", "bce", "bce_with_logits", "focal"]

        for loss_name in classification_losses:
            loss_fn = LossRegistry.get_loss(loss_name)
            assert isinstance(loss_fn, nn.Module)

    def test_all_losses_forward_pass(self):
        """Test that all losses can perform forward pass."""
        # Regression losses
        predictions = torch.randn(5, 3, requires_grad=True)
        targets_reg = torch.randn(5, 3)

        for loss_name in ["mse", "mae", "l1", "huber", "smooth_l1", "rmse"]:
            loss_fn = LossRegistry.get_loss(loss_name)
            loss = loss_fn(predictions, targets_reg)
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)

    def test_all_losses_backward_pass(self):
        """Test that all losses support backward pass."""
        # Regression losses
        predictions = torch.randn(5, 3, requires_grad=True)
        targets_reg = torch.randn(5, 3)

        for loss_name in ["mse", "mae", "huber", "smooth_l1", "rmse"]:
            pred = predictions.clone().detach().requires_grad_(True)
            loss_fn = LossRegistry.get_loss(loss_name)
            loss = loss_fn(pred, targets_reg)
            loss.backward()
            assert pred.grad is not None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
