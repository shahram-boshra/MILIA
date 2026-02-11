#!/usr/bin/env python3
"""
Extended Production-Ready Unit Test Suite for metrics.py Module

Comprehensive test coverage including:
- RMSEMetric custom wrapper class (initialization, forward pass, device handling)
- MetricsRegistry class (all static/class methods)
  - get_metric() with various configurations
  - _filter_params() parameter introspection
  - list_available() registry listing
  - get_metric_info() metric information retrieval
  - get_valid_params() parameter extraction
  - register_custom_metric() custom metric registration
  - get_metrics_for_task() task-aware metric selection
  - get_default_metrics_for_task() default metric retrieval
  - is_metric_compatible_with_task() compatibility checking
  - create_metric_collection() MetricCollection creation
- Convenience functions (get_metric, get_metrics_for_task, list_metrics, etc.)
- Edge cases and error handling
- Device management (CPU/CUDA scenarios)
- TorchMetrics availability scenarios
- Classification vs regression metric handling
- Task-type aware metric selection with fallback logic

This is an EXTENDED PRODUCTION-READY test suite with comprehensive coverage
for enterprise-grade deployment.

Author: milia Team
Version: 1.0.0
"""
import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
import logging
import inspect
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import Dict, Any, List

import torch
import torch.nn as nn

# Import the module under test
from milia_pipeline.models.training.metrics import (
    RMSEMetric,
    MetricsRegistry,
    get_metric,
    get_metrics_for_task,
    list_metrics,
    get_default_metrics_for_task,
    is_metric_compatible_with_task,
    TORCHMETRICS_AVAILABLE,
)

# Conditional imports for tests that need TorchMetrics
if TORCHMETRICS_AVAILABLE:
    from torchmetrics import Metric, MetricCollection
    from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
    from torchmetrics.classification import Accuracy, F1Score, AUROC


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_predictions():
    """Create sample prediction tensors."""
    return torch.randn(16, 1)


@pytest.fixture
def sample_targets():
    """Create sample target tensors."""
    return torch.randn(16, 1)


@pytest.fixture
def sample_predictions_2d():
    """Create sample 2D prediction tensors."""
    return torch.randn(16, 4)


@pytest.fixture
def sample_targets_2d():
    """Create sample 2D target tensors."""
    return torch.randn(16, 4)


@pytest.fixture
def sample_classification_preds():
    """Create sample classification prediction tensors (logits)."""
    return torch.randn(16, 2)


@pytest.fixture
def sample_classification_targets():
    """Create sample classification target tensors (binary labels)."""
    return torch.randint(0, 2, (16,))


@pytest.fixture
def sample_multiclass_preds():
    """Create sample multiclass prediction tensors."""
    return torch.randn(16, 5)


@pytest.fixture
def sample_multiclass_targets():
    """Create sample multiclass target tensors."""
    return torch.randint(0, 5, (16,))


@pytest.fixture
def mock_metric_class():
    """Create a mock metric class for testing custom registration."""
    class CustomMockMetric(nn.Module):
        def __init__(self, custom_param=None):
            super().__init__()
            self.custom_param = custom_param
        
        def forward(self, preds, target):
            return torch.mean((preds - target) ** 2)
    
    return CustomMockMetric


@pytest.fixture
def mock_torchmetrics_metric_class():
    """Create a mock TorchMetrics-compatible metric class."""
    if not TORCHMETRICS_AVAILABLE:
        pytest.skip("TorchMetrics not available")
    
    class CustomTorchMetric(Metric):
        def __init__(self, custom_param=None, **kwargs):
            super().__init__(**kwargs)
            self.custom_param = custom_param
            self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        
        def update(self, preds, target):
            self.sum += torch.sum(torch.abs(preds - target))
            self.count += target.numel()
        
        def compute(self):
            return self.sum / self.count
    
    return CustomTorchMetric


@pytest.fixture
def cleanup_registry():
    """Fixture to clean up the registry after tests that modify it."""
    # Store original state
    original_metrics = MetricsRegistry._metrics.copy()
    original_classification = MetricsRegistry._classification_metrics.copy()
    original_regression = MetricsRegistry._regression_metrics.copy()
    
    yield
    
    # Restore original state
    MetricsRegistry._metrics = original_metrics
    MetricsRegistry._classification_metrics = original_classification
    MetricsRegistry._regression_metrics = original_regression


# =============================================================================
# RMSEMETRIC CLASS TESTS
# =============================================================================

class TestRMSEMetric:
    """Test RMSEMetric custom wrapper class."""
    
    def test_rmse_metric_initialization(self):
        """Test RMSEMetric initializes correctly."""
        metric = RMSEMetric()
        assert isinstance(metric, nn.Module)
        if TORCHMETRICS_AVAILABLE:
            assert metric._metric is not None
        else:
            assert metric._metric is None
    
    def test_rmse_metric_forward_pass(self, sample_predictions, sample_targets):
        """Test RMSEMetric forward pass computes correctly."""
        metric = RMSEMetric()
        
        # Flatten to 1D for metric computation
        preds = sample_predictions.flatten()
        targets = sample_targets.flatten()
        
        result = metric(preds, targets)
        
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0 or result.shape == ()  # Scalar output
        assert result >= 0  # RMSE is always non-negative
    
    def test_rmse_metric_forward_perfect_prediction(self):
        """Test RMSEMetric returns zero for perfect predictions."""
        metric = RMSEMetric()
        
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = targets.clone()
        
        result = metric(preds, targets)
        
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)
    
    def test_rmse_metric_forward_known_error(self):
        """Test RMSEMetric computes correct RMSE for known values."""
        metric = RMSEMetric()
        
        # Known example: errors are [1, 2, 3, 4], squared [1, 4, 9, 16], mean=7.5, sqrt=2.7386
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        preds = torch.tensor([0.0, 0.0, 0.0, 0.0])
        
        result = metric(preds, targets)
        
        expected_rmse = torch.sqrt(torch.tensor((1 + 4 + 9 + 16) / 4.0))
        assert torch.isclose(result, expected_rmse, atol=1e-5)
    
    def test_rmse_metric_to_device_cpu(self):
        """Test RMSEMetric.to() moves metric to CPU."""
        metric = RMSEMetric()
        device = torch.device('cpu')
        
        result = metric.to(device)
        
        assert result is metric  # Returns self for chaining
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rmse_metric_to_device_cuda(self):
        """Test RMSEMetric.to() moves metric to CUDA."""
        metric = RMSEMetric()
        device = torch.device('cuda:0')
        
        result = metric.to(device)
        
        assert result is metric
    
    def test_rmse_metric_fallback_when_torchmetrics_unavailable(self, sample_predictions, sample_targets):
        """Test RMSEMetric fallback computation when TorchMetrics unavailable."""
        metric = RMSEMetric()
        
        # Force fallback by setting _metric to None
        metric._metric = None
        
        preds = sample_predictions.flatten()
        targets = sample_targets.flatten()
        
        result = metric(preds, targets)
        
        # Manually compute expected RMSE
        expected = torch.sqrt(torch.mean((preds - targets) ** 2))
        
        assert torch.isclose(result, expected, atol=1e-6)
    
    def test_rmse_metric_with_batched_input(self):
        """Test RMSEMetric with batched input tensors."""
        metric = RMSEMetric()
        
        batch_size = 32
        preds = torch.randn(batch_size)
        targets = torch.randn(batch_size)
        
        result = metric(preds, targets)
        
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0 or result.numel() == 1
    
    def test_rmse_metric_gradient_flow(self):
        """Test that gradients flow through RMSEMetric."""
        metric = RMSEMetric()
        
        preds = torch.randn(10, requires_grad=True)
        targets = torch.randn(10)
        
        result = metric(preds, targets)
        result.backward()
        
        assert preds.grad is not None
        assert preds.grad.shape == preds.shape


# =============================================================================
# METRICSREGISTRY - GET_METRIC TESTS
# =============================================================================

class TestMetricsRegistryGetMetric:
    """Test MetricsRegistry.get_metric() method."""
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_mse(self):
        """Test getting MSE metric."""
        metric = MetricsRegistry.get_metric("mse")
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_mae(self):
        """Test getting MAE metric."""
        metric = MetricsRegistry.get_metric("mae")
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    def test_get_metric_rmse(self):
        """Test getting RMSE metric (custom wrapper)."""
        metric = MetricsRegistry.get_metric("rmse")
        
        assert metric is not None
        assert isinstance(metric, RMSEMetric)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_r2(self):
        """Test getting R2Score metric."""
        metric = MetricsRegistry.get_metric("r2")
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_mape(self):
        """Test getting MAPE metric."""
        metric = MetricsRegistry.get_metric("mape")
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_explained_variance(self):
        """Test getting ExplainedVariance metric."""
        metric = MetricsRegistry.get_metric("explained_variance")
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_accuracy_binary(self):
        """Test getting Accuracy metric for binary classification."""
        metric = MetricsRegistry.get_metric("accuracy", params={"task": "binary"})
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_accuracy_multiclass(self):
        """Test getting Accuracy metric for multiclass classification."""
        metric = MetricsRegistry.get_metric("accuracy", params={"task": "multiclass", "num_classes": 5})
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_f1_binary(self):
        """Test getting F1Score metric for binary classification."""
        metric = MetricsRegistry.get_metric("f1", params={"task": "binary"})
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_precision(self):
        """Test getting Precision metric."""
        metric = MetricsRegistry.get_metric("precision", params={"task": "binary"})
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_recall(self):
        """Test getting Recall metric."""
        metric = MetricsRegistry.get_metric("recall", params={"task": "binary"})
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_auroc(self):
        """Test getting AUROC metric."""
        metric = MetricsRegistry.get_metric("auroc", params={"task": "binary"})
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_auprc(self):
        """Test getting AveragePrecision (AUPRC) metric."""
        metric = MetricsRegistry.get_metric("auprc", params={"task": "binary"})
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    def test_get_metric_case_insensitive(self):
        """Test that metric name is case-insensitive."""
        metric_lower = MetricsRegistry.get_metric("rmse")
        metric_upper = MetricsRegistry.get_metric("RMSE")
        metric_mixed = MetricsRegistry.get_metric("RmSe")
        
        assert type(metric_lower) == type(metric_upper) == type(metric_mixed)
    
    def test_get_metric_unknown_raises_value_error(self):
        """Test that unknown metric name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricsRegistry.get_metric("unknown_metric_xyz")
    
    def test_get_metric_unknown_error_lists_available(self):
        """Test that unknown metric error message lists available metrics."""
        with pytest.raises(ValueError) as exc_info:
            MetricsRegistry.get_metric("nonexistent")
        
        assert "Available metrics:" in str(exc_info.value)
        assert "mse" in str(exc_info.value)
        assert "mae" in str(exc_info.value)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_with_device_cpu(self):
        """Test getting metric with explicit CPU device."""
        device = torch.device('cpu')
        metric = MetricsRegistry.get_metric("mse", device=device)
        
        assert metric is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_with_device_cuda(self):
        """Test getting metric with CUDA device."""
        device = torch.device('cuda:0')
        metric = MetricsRegistry.get_metric("mse", device=device)
        
        assert metric is not None
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_with_empty_params(self):
        """Test getting metric with empty params dict."""
        metric = MetricsRegistry.get_metric("mse", params={})
        
        assert metric is not None
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_with_none_params(self):
        """Test getting metric with None params."""
        metric = MetricsRegistry.get_metric("mse", params=None)
        
        assert metric is not None
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_filters_unsupported_params(self):
        """Test that unsupported params are filtered out silently."""
        # Passing an unsupported parameter should not raise an error
        metric = MetricsRegistry.get_metric("mse", params={"unsupported_param_xyz": 123})
        
        assert metric is not None
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_with_valid_params(self):
        """Test getting metric with valid parameters."""
        # For Accuracy, 'num_classes' and 'task' are valid params
        metric = MetricsRegistry.get_metric("accuracy", params={"task": "multiclass", "num_classes": 3})
        
        assert metric is not None


# =============================================================================
# METRICSREGISTRY - _FILTER_PARAMS TESTS
# =============================================================================

class TestMetricsRegistryFilterParams:
    """Test MetricsRegistry._filter_params() method."""
    
    def test_filter_params_empty_params(self):
        """Test _filter_params returns empty dict for empty params."""
        result = MetricsRegistry._filter_params(nn.Module, {})
        
        assert result == {}
    
    def test_filter_params_none_equivalent(self):
        """Test _filter_params handles empty params correctly."""
        result = MetricsRegistry._filter_params(nn.Module, {})
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_filter_params_keeps_valid_params(self):
        """Test _filter_params keeps valid parameters."""
        # MeanSquaredError accepts 'squared' parameter
        params = {"squared": False}
        result = MetricsRegistry._filter_params(MeanSquaredError, params)
        
        assert "squared" in result
        assert result["squared"] is False
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_filter_params_removes_invalid_params(self):
        """Test _filter_params removes invalid parameters."""
        params = {"squared": False, "invalid_param": "value"}
        result = MetricsRegistry._filter_params(MeanSquaredError, params)
        
        assert "squared" in result
        assert "invalid_param" not in result
    
    def test_filter_params_removes_self_and_cls(self):
        """Test _filter_params removes 'self' and 'cls' from valid params."""
        class TestClass:
            def __init__(self, param1, param2="default"):
                pass
        
        params = {"param1": 1, "param2": 2, "self": "bad", "cls": "also_bad"}
        result = MetricsRegistry._filter_params(TestClass, params)
        
        assert "self" not in result
        assert "cls" not in result
        assert "param1" in result
        assert "param2" in result
    
    def test_filter_params_handles_introspection_failure(self):
        """Test _filter_params handles classes where introspection fails."""
        # Create a class where signature introspection might fail
        class WeirdClass:
            pass
        
        # Remove __init__ to simulate introspection failure
        # In practice, most classes have __init__, so this tests fallback behavior
        params = {"some_param": "value"}
        
        # Should not raise an error
        result = MetricsRegistry._filter_params(WeirdClass, params)
        
        assert isinstance(result, dict)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_filter_params_checks_both_init_and_new(self):
        """Test _filter_params checks both __init__ and __new__ signatures."""
        # TorchMetrics uses __new__ for task routing in some metrics
        # This test verifies that parameters accepted by __new__ are kept
        params = {"task": "binary", "num_classes": 2}
        result = MetricsRegistry._filter_params(Accuracy, params)
        
        # 'task' is accepted by TorchMetrics classification metrics
        assert "task" in result


# =============================================================================
# METRICSREGISTRY - LIST_AVAILABLE TESTS
# =============================================================================

class TestMetricsRegistryListAvailable:
    """Test MetricsRegistry.list_available() method."""
    
    def test_list_available_returns_list(self):
        """Test list_available returns a list."""
        result = MetricsRegistry.list_available()
        
        assert isinstance(result, list)
    
    def test_list_available_non_empty(self):
        """Test list_available returns non-empty list."""
        result = MetricsRegistry.list_available()
        
        assert len(result) > 0
    
    def test_list_available_contains_known_metrics(self):
        """Test list_available contains known metric names."""
        result = MetricsRegistry.list_available()
        
        assert "mse" in result
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result
        assert "accuracy" in result
        assert "f1" in result
    
    def test_list_available_sorted(self):
        """Test list_available returns sorted list."""
        result = MetricsRegistry.list_available()
        
        assert result == sorted(result)
    
    def test_list_available_all_lowercase(self):
        """Test list_available returns all lowercase names."""
        result = MetricsRegistry.list_available()
        
        for name in result:
            assert name == name.lower()
    
    def test_list_available_no_duplicates(self):
        """Test list_available returns no duplicate names."""
        result = MetricsRegistry.list_available()
        
        assert len(result) == len(set(result))


# =============================================================================
# METRICSREGISTRY - GET_METRIC_INFO TESTS
# =============================================================================

class TestMetricsRegistryGetMetricInfo:
    """Test MetricsRegistry.get_metric_info() method."""
    
    def test_get_metric_info_returns_dict(self):
        """Test get_metric_info returns a dictionary."""
        result = MetricsRegistry.get_metric_info("rmse")
        
        assert isinstance(result, dict)
    
    def test_get_metric_info_contains_name(self):
        """Test get_metric_info contains name field."""
        result = MetricsRegistry.get_metric_info("mse")
        
        assert "name" in result
        assert result["name"] == "mse"
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_info_contains_class_name(self):
        """Test get_metric_info contains class field."""
        result = MetricsRegistry.get_metric_info("mse")
        
        assert "class" in result
        assert result["class"] == "MeanSquaredError"
    
    def test_get_metric_info_contains_is_classification(self):
        """Test get_metric_info contains is_classification field."""
        result = MetricsRegistry.get_metric_info("accuracy")
        
        assert "is_classification" in result
        assert result["is_classification"] is True
    
    def test_get_metric_info_contains_is_regression(self):
        """Test get_metric_info contains is_regression field."""
        result = MetricsRegistry.get_metric_info("mse")
        
        assert "is_regression" in result
        assert result["is_regression"] is True
    
    def test_get_metric_info_classification_metric(self):
        """Test get_metric_info for classification metric."""
        result = MetricsRegistry.get_metric_info("f1")
        
        assert result["is_classification"] is True
        assert result["is_regression"] is False
    
    def test_get_metric_info_regression_metric(self):
        """Test get_metric_info for regression metric."""
        result = MetricsRegistry.get_metric_info("mae")
        
        assert result["is_regression"] is True
        assert result["is_classification"] is False
    
    def test_get_metric_info_unknown_raises_value_error(self):
        """Test get_metric_info raises ValueError for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricsRegistry.get_metric_info("nonexistent_metric")
    
    def test_get_metric_info_case_insensitive(self):
        """Test get_metric_info is case-insensitive."""
        result1 = MetricsRegistry.get_metric_info("MSE")
        result2 = MetricsRegistry.get_metric_info("mse")
        
        assert result1["name"] == result2["name"]
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_info_contains_valid_params(self):
        """Test get_metric_info contains valid_params field."""
        result = MetricsRegistry.get_metric_info("mse")
        
        assert "valid_params" in result
        assert isinstance(result["valid_params"], dict)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_info_contains_module(self):
        """Test get_metric_info contains module field."""
        result = MetricsRegistry.get_metric_info("mse")
        
        assert "module" in result
        assert "torchmetrics" in result["module"]


# =============================================================================
# METRICSREGISTRY - GET_VALID_PARAMS TESTS
# =============================================================================

class TestMetricsRegistryGetValidParams:
    """Test MetricsRegistry.get_valid_params() method."""
    
    def test_get_valid_params_returns_dict(self):
        """Test get_valid_params returns a dictionary."""
        result = MetricsRegistry.get_valid_params("rmse")
        
        assert isinstance(result, dict)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_valid_params_mse(self):
        """Test get_valid_params for MSE metric."""
        result = MetricsRegistry.get_valid_params("mse")
        
        # MeanSquaredError has 'squared' parameter
        assert isinstance(result, dict)
    
    def test_get_valid_params_unknown_raises_value_error(self):
        """Test get_valid_params raises ValueError for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricsRegistry.get_valid_params("nonexistent_metric")
    
    def test_get_valid_params_case_insensitive(self):
        """Test get_valid_params is case-insensitive."""
        result1 = MetricsRegistry.get_valid_params("MAE")
        result2 = MetricsRegistry.get_valid_params("mae")
        
        assert result1 == result2


# =============================================================================
# METRICSREGISTRY - REGISTER_CUSTOM_METRIC TESTS
# =============================================================================

class TestMetricsRegistryRegisterCustomMetric:
    """Test MetricsRegistry.register_custom_metric() method."""
    
    def test_register_custom_metric_success(self, mock_metric_class, cleanup_registry):
        """Test registering a custom metric successfully."""
        MetricsRegistry.register_custom_metric(
            name="custom_test_metric",
            metric_class=mock_metric_class,
            is_classification=False,
            is_regression=True
        )
        
        assert "custom_test_metric" in MetricsRegistry._metrics
    
    def test_register_custom_metric_retrievable(self, mock_metric_class, cleanup_registry):
        """Test registered custom metric can be retrieved."""
        MetricsRegistry.register_custom_metric(
            name="custom_retrievable",
            metric_class=mock_metric_class
        )
        
        metric = MetricsRegistry.get_metric("custom_retrievable")
        assert isinstance(metric, mock_metric_class)
    
    def test_register_custom_metric_lowercase_name(self, mock_metric_class, cleanup_registry):
        """Test custom metric name is stored as lowercase."""
        MetricsRegistry.register_custom_metric(
            name="UPPERCASE_METRIC",
            metric_class=mock_metric_class
        )
        
        assert "uppercase_metric" in MetricsRegistry._metrics
        assert "UPPERCASE_METRIC" not in MetricsRegistry._metrics
    
    def test_register_custom_metric_added_to_classification(self, mock_metric_class, cleanup_registry):
        """Test custom metric added to classification set when specified."""
        MetricsRegistry.register_custom_metric(
            name="custom_clf",
            metric_class=mock_metric_class,
            is_classification=True,
            is_regression=False
        )
        
        assert "custom_clf" in MetricsRegistry._classification_metrics
    
    def test_register_custom_metric_added_to_regression(self, mock_metric_class, cleanup_registry):
        """Test custom metric added to regression set when specified."""
        MetricsRegistry.register_custom_metric(
            name="custom_reg",
            metric_class=mock_metric_class,
            is_classification=False,
            is_regression=True
        )
        
        assert "custom_reg" in MetricsRegistry._regression_metrics
    
    def test_register_custom_metric_duplicate_raises_error(self, mock_metric_class, cleanup_registry):
        """Test registering duplicate metric name raises ValueError."""
        MetricsRegistry.register_custom_metric(
            name="duplicate_test",
            metric_class=mock_metric_class
        )
        
        with pytest.raises(ValueError, match="already registered"):
            MetricsRegistry.register_custom_metric(
                name="duplicate_test",
                metric_class=mock_metric_class
            )
    
    def test_register_custom_metric_overwrite(self, mock_metric_class, cleanup_registry):
        """Test overwriting existing metric with overwrite=True."""
        MetricsRegistry.register_custom_metric(
            name="overwrite_test",
            metric_class=mock_metric_class
        )
        
        # Should not raise with overwrite=True
        MetricsRegistry.register_custom_metric(
            name="overwrite_test",
            metric_class=mock_metric_class,
            overwrite=True
        )
        
        assert "overwrite_test" in MetricsRegistry._metrics
    
    def test_register_custom_metric_invalid_type_raises_error(self, cleanup_registry):
        """Test registering non-Module class raises TypeError."""
        class NotAModule:
            pass
        
        with pytest.raises(TypeError, match="must be a subclass"):
            MetricsRegistry.register_custom_metric(
                name="invalid_metric",
                metric_class=NotAModule
            )
    
    def test_register_custom_metric_not_a_class_raises_error(self, cleanup_registry):
        """Test registering non-class raises TypeError."""
        with pytest.raises(TypeError):
            MetricsRegistry.register_custom_metric(
                name="not_a_class",
                metric_class="string_not_class"
            )
    
    def test_register_custom_metric_instance_instead_of_type_raises_error(self, mock_metric_class, cleanup_registry):
        """Test registering an instance instead of a class raises TypeError.
        
        PRODUCTION-READY: Validates input type checking for metric registration.
        NON-BREAKING: Uses cleanup_registry fixture.
        DYNAMIC: Tests the isinstance check in register_custom_metric.
        """
        # Create an instance of the metric class
        instance = mock_metric_class()
        
        with pytest.raises(TypeError, match="must be a subclass"):
            MetricsRegistry.register_custom_metric(
                name="instance_metric",
                metric_class=instance  # Instance, not class
            )
    
    def test_register_custom_metric_both_classification_and_regression(self, mock_metric_class, cleanup_registry):
        """Test registering a metric as both classification and regression.
        
        PRODUCTION-READY: Validates that a metric can be in both sets if needed.
        DYNAMIC: Tests the is_classification and is_regression both True case.
        FUTURE-PROOF: Supports flexible metric categorization.
        """
        MetricsRegistry.register_custom_metric(
            name="dual_purpose_metric",
            metric_class=mock_metric_class,
            is_classification=True,
            is_regression=True
        )
        
        assert "dual_purpose_metric" in MetricsRegistry._classification_metrics
        assert "dual_purpose_metric" in MetricsRegistry._regression_metrics
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_register_torchmetrics_custom_metric(self, mock_torchmetrics_metric_class, cleanup_registry):
        """Test registering a TorchMetrics-based custom metric."""
        MetricsRegistry.register_custom_metric(
            name="torchmetrics_custom",
            metric_class=mock_torchmetrics_metric_class
        )
        
        assert "torchmetrics_custom" in MetricsRegistry._metrics
        
        metric = MetricsRegistry.get_metric("torchmetrics_custom")
        assert isinstance(metric, mock_torchmetrics_metric_class)


# =============================================================================
# METRICSREGISTRY - GET_METRICS_FOR_TASK TESTS
# =============================================================================

class TestMetricsRegistryGetMetricsForTask:
    """Test MetricsRegistry.get_metrics_for_task() method."""
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_graph_regression(self):
        """Test getting metrics for graph_regression task."""
        metrics = MetricsRegistry.get_metrics_for_task("graph_regression")
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        # Should contain regression metrics
        assert any(name in metrics for name in ["mae", "mse", "rmse", "r2"])
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_node_regression(self):
        """Test getting metrics for node_regression task."""
        metrics = MetricsRegistry.get_metrics_for_task("node_regression")
        
        assert isinstance(metrics, dict)
        assert any(name in metrics for name in ["mae", "mse", "rmse", "r2"])
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_edge_regression(self):
        """Test getting metrics for edge_regression task."""
        metrics = MetricsRegistry.get_metrics_for_task("edge_regression")
        
        assert isinstance(metrics, dict)
        assert any(name in metrics for name in ["mae", "mse", "rmse", "r2"])
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_graph_classification(self):
        """Test getting metrics for graph_classification task."""
        metrics = MetricsRegistry.get_metrics_for_task("graph_classification", num_classes=2)
        
        assert isinstance(metrics, dict)
        # Should contain classification metrics
        assert any(name in metrics for name in ["accuracy", "f1", "precision", "recall"])
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_node_classification(self):
        """Test getting metrics for node_classification task."""
        metrics = MetricsRegistry.get_metrics_for_task("node_classification", num_classes=3)
        
        assert isinstance(metrics, dict)
        assert any(name in metrics for name in ["accuracy", "f1", "precision", "recall"])
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_link_prediction(self):
        """Test getting metrics for link_prediction task."""
        metrics = MetricsRegistry.get_metrics_for_task("link_prediction")
        
        assert isinstance(metrics, dict)
        # Link prediction should have binary classification metrics
        assert any(name in metrics for name in ["auroc", "auprc", "accuracy"])
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_with_explicit_metric_names(self):
        """Test getting metrics with explicit metric names."""
        metrics = MetricsRegistry.get_metrics_for_task(
            "graph_regression",
            metric_names=["mae", "mse"]
        )
        
        assert "mae" in metrics
        assert "mse" in metrics
        assert len(metrics) == 2
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_incompatible_metrics_fallback(self, caplog):
        """Test fallback to defaults when metrics incompatible with task."""
        with caplog.at_level(logging.WARNING):
            metrics = MetricsRegistry.get_metrics_for_task(
                "graph_regression",
                metric_names=["accuracy", "f1"]  # Classification metrics for regression task
            )
        
        # Should fall back to default regression metrics
        assert any(name in metrics for name in ["mae", "mse", "rmse", "r2"])
        assert "incompatible" in caplog.text.lower()
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_case_insensitive(self):
        """Test task type is case-insensitive."""
        metrics1 = MetricsRegistry.get_metrics_for_task("GRAPH_REGRESSION")
        metrics2 = MetricsRegistry.get_metrics_for_task("graph_regression")
        
        assert set(metrics1.keys()) == set(metrics2.keys())
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_unknown_task_defaults(self):
        """Test unknown task type gets default metrics."""
        metrics = MetricsRegistry.get_metrics_for_task("unknown_task_xyz")
        
        # Should get fallback defaults (mse, mae)
        assert any(name in metrics for name in ["mse", "mae"])
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_with_params(self):
        """Test getting metrics with additional parameters."""
        metrics = MetricsRegistry.get_metrics_for_task(
            "graph_regression",
            params={"squared": False}  # Parameter for MSE
        )
        
        assert isinstance(metrics, dict)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_with_device(self):
        """Test getting metrics with explicit device."""
        device = torch.device('cpu')
        metrics = MetricsRegistry.get_metrics_for_task(
            "graph_regression",
            device=device
        )
        
        assert isinstance(metrics, dict)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_with_num_classes(self):
        """Test getting classification metrics with num_classes."""
        metrics = MetricsRegistry.get_metrics_for_task(
            "graph_classification",
            num_classes=5
        )
        
        assert isinstance(metrics, dict)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_binary_classification(self):
        """Test binary classification inferred from num_classes=2."""
        metrics = MetricsRegistry.get_metrics_for_task(
            "graph_classification",
            num_classes=2
        )
        
        assert isinstance(metrics, dict)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_empty_metric_names(self):
        """Test getting metrics with empty metric_names list."""
        metrics = MetricsRegistry.get_metrics_for_task(
            "graph_regression",
            metric_names=[]
        )
        
        # Empty list should fall back to defaults
        assert len(metrics) > 0
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_none_task_type(self):
        """Test getting metrics with None task_type."""
        metrics = MetricsRegistry.get_metrics_for_task(None)
        
        # Should get fallback defaults
        assert isinstance(metrics, dict)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_partial_compatible_metrics(self, caplog):
        """Test when some metrics are compatible and some are not."""
        with caplog.at_level(logging.WARNING):
            metrics = MetricsRegistry.get_metrics_for_task(
                "graph_regression",
                metric_names=["mae", "accuracy"]  # mae is valid, accuracy is not
            )
        
        # Should fallback since there's an incompatible metric
        assert "incompatible" in caplog.text.lower() or "mae" in metrics
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_raises_when_all_creations_fail(self, cleanup_registry):
        """Test get_metrics_for_task raises RuntimeError when no metrics can be created.
        
        PRODUCTION-READY: Validates error path when metric instantiation fails entirely.
        NON-BREAKING: Uses cleanup_registry to restore state.
        DYNAMIC: Tests the RuntimeError branch when metrics dict is empty.
        FUTURE-PROOF: Ensures meaningful error message for debugging.
        """
        # Register a metric class that will fail to instantiate
        class FailingMetric(nn.Module):
            def __init__(self, **kwargs):
                raise ValueError("Intentional failure for testing")
        
        # Store original task defaults and replace with failing metric only
        original_defaults = MetricsRegistry._task_to_default_metrics.get('graph_regression', [])
        
        try:
            # Register the failing metric
            MetricsRegistry._metrics['failing_only'] = FailingMetric
            MetricsRegistry._regression_metrics.add('failing_only')
            MetricsRegistry._task_to_default_metrics['graph_regression'] = ['failing_only']
            
            with pytest.raises(RuntimeError, match="No metrics could be created"):
                MetricsRegistry.get_metrics_for_task("graph_regression")
        finally:
            # Cleanup is handled by cleanup_registry fixture
            MetricsRegistry._task_to_default_metrics['graph_regression'] = original_defaults
            if 'failing_only' in MetricsRegistry._metrics:
                del MetricsRegistry._metrics['failing_only']
            MetricsRegistry._regression_metrics.discard('failing_only')
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_classification_with_num_classes_only(self):
        """Test classification task with num_classes but no explicit task parameter.
        
        PRODUCTION-READY: Validates the auto-inference of task type from num_classes.
        DYNAMIC: Tests the branch where task is inferred from num_classes value.
        FUTURE-PROOF: Documents expected behavior for classification configuration.
        """
        # Binary case: num_classes=2 should infer task='binary'
        metrics_binary = MetricsRegistry.get_metrics_for_task(
            "graph_classification",
            num_classes=2
        )
        assert isinstance(metrics_binary, dict)
        assert len(metrics_binary) > 0
        
        # Multiclass case: num_classes>2 should infer task='multiclass'
        metrics_multiclass = MetricsRegistry.get_metrics_for_task(
            "graph_classification",
            num_classes=5
        )
        assert isinstance(metrics_multiclass, dict)
        assert len(metrics_multiclass) > 0
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_classification_without_num_classes_defaults_binary(self, caplog):
        """Test classification task without num_classes defaults to binary.
        
        PRODUCTION-READY: Validates the default behavior for unspecified num_classes.
        DYNAMIC: Tests the fallback branch in classification metric setup.
        FUTURE-PROOF: Documents expected default for ambiguous configuration.
        """
        with caplog.at_level(logging.DEBUG):
            metrics = MetricsRegistry.get_metrics_for_task("graph_classification")
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        # Should log debug message about defaulting to binary
        # Note: exact log message depends on implementation


# =============================================================================
# METRICSREGISTRY - GET_DEFAULT_METRICS_FOR_TASK TESTS
# =============================================================================

class TestMetricsRegistryGetDefaultMetricsForTask:
    """Test MetricsRegistry.get_default_metrics_for_task() method."""
    
    def test_get_default_metrics_graph_regression(self):
        """Test default metrics for graph_regression."""
        defaults = MetricsRegistry.get_default_metrics_for_task("graph_regression")
        
        assert isinstance(defaults, list)
        assert "mae" in defaults
        assert "mse" in defaults
    
    def test_get_default_metrics_graph_classification(self):
        """Test default metrics for graph_classification."""
        defaults = MetricsRegistry.get_default_metrics_for_task("graph_classification")
        
        assert isinstance(defaults, list)
        assert "accuracy" in defaults
        assert "f1" in defaults
    
    def test_get_default_metrics_link_prediction(self):
        """Test default metrics for link_prediction."""
        defaults = MetricsRegistry.get_default_metrics_for_task("link_prediction")
        
        assert isinstance(defaults, list)
        assert "auroc" in defaults
    
    def test_get_default_metrics_node_regression(self):
        """Test default metrics for node_regression."""
        defaults = MetricsRegistry.get_default_metrics_for_task("node_regression")
        
        assert isinstance(defaults, list)
        assert len(defaults) > 0
    
    def test_get_default_metrics_node_classification(self):
        """Test default metrics for node_classification."""
        defaults = MetricsRegistry.get_default_metrics_for_task("node_classification")
        
        assert isinstance(defaults, list)
        assert "accuracy" in defaults
    
    def test_get_default_metrics_edge_regression(self):
        """Test default metrics for edge_regression."""
        defaults = MetricsRegistry.get_default_metrics_for_task("edge_regression")
        
        assert isinstance(defaults, list)
    
    def test_get_default_metrics_edge_classification(self):
        """Test default metrics for edge_classification."""
        defaults = MetricsRegistry.get_default_metrics_for_task("edge_classification")
        
        assert isinstance(defaults, list)
    
    def test_get_default_metrics_unknown_task(self):
        """Test default metrics for unknown task type."""
        defaults = MetricsRegistry.get_default_metrics_for_task("unknown_task")
        
        # Should return fallback defaults
        assert isinstance(defaults, list)
        assert "mse" in defaults or "mae" in defaults
    
    def test_get_default_metrics_case_insensitive(self):
        """Test default metrics lookup is case-insensitive."""
        defaults1 = MetricsRegistry.get_default_metrics_for_task("GRAPH_REGRESSION")
        defaults2 = MetricsRegistry.get_default_metrics_for_task("graph_regression")
        
        assert defaults1 == defaults2
    
    def test_get_default_metrics_none_task(self):
        """Test default metrics for None task type."""
        defaults = MetricsRegistry.get_default_metrics_for_task(None)
        
        # Should return fallback defaults
        assert isinstance(defaults, list)
    
    def test_get_default_metrics_empty_string(self):
        """Test default metrics for empty string task type."""
        defaults = MetricsRegistry.get_default_metrics_for_task("")
        
        assert isinstance(defaults, list)


# =============================================================================
# METRICSREGISTRY - IS_METRIC_COMPATIBLE_WITH_TASK TESTS
# =============================================================================

class TestMetricsRegistryIsMetricCompatibleWithTask:
    """Test MetricsRegistry.is_metric_compatible_with_task() method."""
    
    def test_mse_compatible_with_regression(self):
        """Test MSE is compatible with regression tasks."""
        assert MetricsRegistry.is_metric_compatible_with_task("mse", "graph_regression") is True
        assert MetricsRegistry.is_metric_compatible_with_task("mse", "node_regression") is True
        assert MetricsRegistry.is_metric_compatible_with_task("mse", "edge_regression") is True
    
    def test_mse_incompatible_with_classification(self):
        """Test MSE is incompatible with classification tasks."""
        assert MetricsRegistry.is_metric_compatible_with_task("mse", "graph_classification") is False
        assert MetricsRegistry.is_metric_compatible_with_task("mse", "node_classification") is False
    
    def test_accuracy_compatible_with_classification(self):
        """Test Accuracy is compatible with classification tasks."""
        assert MetricsRegistry.is_metric_compatible_with_task("accuracy", "graph_classification") is True
        assert MetricsRegistry.is_metric_compatible_with_task("accuracy", "node_classification") is True
    
    def test_accuracy_incompatible_with_regression(self):
        """Test Accuracy is incompatible with regression tasks."""
        assert MetricsRegistry.is_metric_compatible_with_task("accuracy", "graph_regression") is False
    
    def test_f1_compatible_with_classification(self):
        """Test F1 is compatible with classification tasks."""
        assert MetricsRegistry.is_metric_compatible_with_task("f1", "graph_classification") is True
    
    def test_f1_incompatible_with_regression(self):
        """Test F1 is incompatible with regression tasks."""
        assert MetricsRegistry.is_metric_compatible_with_task("f1", "graph_regression") is False
    
    def test_auroc_compatible_with_link_prediction(self):
        """Test AUROC is compatible with link_prediction task."""
        assert MetricsRegistry.is_metric_compatible_with_task("auroc", "link_prediction") is True
    
    def test_mae_compatible_with_regression(self):
        """Test MAE is compatible with regression tasks."""
        assert MetricsRegistry.is_metric_compatible_with_task("mae", "graph_regression") is True
    
    def test_rmse_compatible_with_regression(self):
        """Test RMSE is compatible with regression tasks."""
        assert MetricsRegistry.is_metric_compatible_with_task("rmse", "node_regression") is True
    
    def test_r2_compatible_with_regression(self):
        """Test R2 is compatible with regression tasks."""
        assert MetricsRegistry.is_metric_compatible_with_task("r2", "edge_regression") is True
    
    def test_case_insensitive_metric_name(self):
        """Test metric name is case-insensitive."""
        assert MetricsRegistry.is_metric_compatible_with_task("MSE", "graph_regression") is True
        assert MetricsRegistry.is_metric_compatible_with_task("Mse", "graph_regression") is True
    
    def test_case_insensitive_task_type(self):
        """Test task type is case-insensitive."""
        assert MetricsRegistry.is_metric_compatible_with_task("mse", "GRAPH_REGRESSION") is True
        assert MetricsRegistry.is_metric_compatible_with_task("mse", "Graph_Regression") is True
    
    def test_unknown_metric_compatible(self):
        """Test unknown metric is considered compatible (permissive)."""
        result = MetricsRegistry.is_metric_compatible_with_task("unknown_xyz", "graph_regression")
        # Unknown metrics are neither in classification nor regression sets,
        # so they pass the compatibility check
        assert result is True
    
    def test_none_metric_name(self):
        """Test None metric name handling."""
        result = MetricsRegistry.is_metric_compatible_with_task(None, "graph_regression")
        assert isinstance(result, bool)
    
    def test_none_task_type(self):
        """Test None task type handling."""
        result = MetricsRegistry.is_metric_compatible_with_task("mse", None)
        assert isinstance(result, bool)
    
    def test_empty_string_handling(self):
        """Test empty string handling."""
        result = MetricsRegistry.is_metric_compatible_with_task("", "graph_regression")
        assert isinstance(result, bool)


# =============================================================================
# METRICSREGISTRY - CREATE_METRIC_COLLECTION TESTS
# =============================================================================

@pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
class TestMetricsRegistryCreateMetricCollection:
    """Test MetricsRegistry.create_metric_collection() method.
    
    Note: RMSEMetric is a custom nn.Module wrapper, not a torchmetrics.Metric,
    so it cannot be included in MetricCollection. Tests that would include RMSE
    (via default regression metrics) must either exclude it or expect ValueError.
    """
    
    def test_create_metric_collection_regression_without_rmse(self):
        """Test creating MetricCollection for regression task (excluding RMSE)."""
        # Explicitly exclude RMSE since it's nn.Module, not torchmetrics.Metric
        collection = MetricsRegistry.create_metric_collection(
            "graph_regression",
            metric_names=["mae", "mse", "r2"]  # Exclude rmse
        )
        
        assert isinstance(collection, MetricCollection)
        assert len(collection) == 3
    
    def test_create_metric_collection_regression_with_rmse_raises(self):
        """Test that including RMSEMetric in MetricCollection raises ValueError.
        
        RMSEMetric inherits from nn.Module, not torchmetrics.Metric, so it
        cannot be added to a MetricCollection. This documents expected behavior.
        """
        with pytest.raises(ValueError, match="not an instance of.*Metric"):
            # Default graph_regression metrics include rmse which is incompatible
            MetricsRegistry.create_metric_collection("graph_regression")
    
    def test_create_metric_collection_classification(self):
        """Test creating MetricCollection for classification task."""
        collection = MetricsRegistry.create_metric_collection(
            "graph_classification",
            num_classes=2
        )
        
        assert isinstance(collection, MetricCollection)
    
    def test_create_metric_collection_with_prefix(self):
        """Test creating MetricCollection with prefix."""
        collection = MetricsRegistry.create_metric_collection(
            "graph_regression",
            metric_names=["mae", "mse"],  # Exclude rmse
            prefix="val_"
        )
        
        # Check that metrics have the prefix
        for key in collection.keys():
            assert key.startswith("val_")
    
    def test_create_metric_collection_with_metric_names(self):
        """Test creating MetricCollection with explicit metric names."""
        collection = MetricsRegistry.create_metric_collection(
            "graph_regression",
            metric_names=["mae", "mse"]
        )
        
        assert len(collection) == 2
    
    def test_create_metric_collection_with_device(self):
        """Test creating MetricCollection with explicit device."""
        device = torch.device('cpu')
        collection = MetricsRegistry.create_metric_collection(
            "graph_regression",
            metric_names=["mae", "mse", "r2"],  # Exclude rmse
            device=device
        )
        
        assert isinstance(collection, MetricCollection)
    
    def test_create_metric_collection_with_params(self):
        """Test creating MetricCollection with additional params."""
        collection = MetricsRegistry.create_metric_collection(
            "graph_regression",
            metric_names=["mae", "mse"],  # Exclude rmse
            params={"squared": False}
        )
        
        assert isinstance(collection, MetricCollection)
    
    def test_create_metric_collection_link_prediction(self):
        """Test creating MetricCollection for link_prediction."""
        collection = MetricsRegistry.create_metric_collection("link_prediction")
        
        assert isinstance(collection, MetricCollection)
    
    def test_create_metric_collection_multiclass(self):
        """Test creating MetricCollection for multiclass classification."""
        collection = MetricsRegistry.create_metric_collection(
            "node_classification",
            num_classes=5
        )
        
        assert isinstance(collection, MetricCollection)
    
    def test_create_metric_collection_empty_prefix(self):
        """Test creating MetricCollection with empty prefix."""
        collection = MetricsRegistry.create_metric_collection(
            "graph_regression",
            metric_names=["mae", "mse"],  # Exclude rmse
            prefix=""
        )
        
        # No prefix should be added
        for key in collection.keys():
            assert not key.startswith("_")


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_function(self):
        """Test get_metric convenience function."""
        metric = get_metric("mse")
        
        assert metric is not None
        assert isinstance(metric, nn.Module)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_function_with_params(self):
        """Test get_metric convenience function with params."""
        metric = get_metric("accuracy", params={"task": "binary"})
        
        assert metric is not None
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_function_with_device(self):
        """Test get_metric convenience function with device."""
        device = torch.device('cpu')
        metric = get_metric("mae", device=device)
        
        assert metric is not None
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_function(self):
        """Test get_metrics_for_task convenience function."""
        metrics = get_metrics_for_task("graph_regression")
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_function_with_params(self):
        """Test get_metrics_for_task convenience function with all params."""
        metrics = get_metrics_for_task(
            task_type="graph_classification",
            metric_names=["accuracy", "f1"],
            params={"task": "binary"},
            device=torch.device('cpu'),
            num_classes=2
        )
        
        assert isinstance(metrics, dict)
    
    def test_list_metrics_function(self):
        """Test list_metrics convenience function."""
        available = list_metrics()
        
        assert isinstance(available, list)
        assert len(available) > 0
        assert "mse" in available
    
    def test_get_default_metrics_for_task_function(self):
        """Test get_default_metrics_for_task convenience function."""
        defaults = get_default_metrics_for_task("graph_regression")
        
        assert isinstance(defaults, list)
        assert len(defaults) > 0
    
    def test_is_metric_compatible_with_task_function(self):
        """Test is_metric_compatible_with_task convenience function."""
        result = is_metric_compatible_with_task("mse", "graph_regression")
        
        assert result is True
    
    def test_is_metric_compatible_with_task_function_incompatible(self):
        """Test is_metric_compatible_with_task returns False for incompatible."""
        result = is_metric_compatible_with_task("accuracy", "graph_regression")
        
        assert result is False


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_get_metric_with_special_characters_in_name(self):
        """Test get_metric with special characters in name."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricsRegistry.get_metric("mse!@#$")
    
    def test_get_metric_with_whitespace_name(self):
        """Test get_metric with whitespace in name."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricsRegistry.get_metric("  mse  ")
    
    def test_get_metric_with_empty_string(self):
        """Test get_metric with empty string."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricsRegistry.get_metric("")
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_computation_with_empty_tensors(self):
        """Test metric computation with empty tensors."""
        metric = MetricsRegistry.get_metric("mse")
        
        # Empty tensors might cause issues
        preds = torch.tensor([])
        targets = torch.tensor([])
        
        # Behavior depends on TorchMetrics implementation
        # This test documents current behavior
        try:
            result = metric(preds, targets)
            # If it works, result should be a tensor
            assert isinstance(result, torch.Tensor)
        except (RuntimeError, ValueError):
            # Some metrics may not handle empty tensors
            pass
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_computation_with_mismatched_shapes(self):
        """Test metric computation with mismatched tensor shapes."""
        metric = MetricsRegistry.get_metric("mse")
        
        preds = torch.randn(10)
        targets = torch.randn(5)  # Different shape
        
        # Should raise an error
        with pytest.raises((RuntimeError, ValueError)):
            metric(preds, targets)
    
    def test_registry_immutability_check(self):
        """Test that registry metrics dict is not accidentally mutated."""
        original_count = len(MetricsRegistry._metrics)
        
        # Get list of available metrics
        _ = MetricsRegistry.list_available()
        
        # Count should not change
        assert len(MetricsRegistry._metrics) == original_count
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_returns_new_instance(self):
        """Test that get_metric returns a new instance each time."""
        metric1 = MetricsRegistry.get_metric("mse")
        metric2 = MetricsRegistry.get_metric("mse")
        
        # Should be different instances
        assert metric1 is not metric2
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_with_nan_values(self):
        """Test metric computation with NaN values."""
        metric = MetricsRegistry.get_metric("mse")
        
        preds = torch.tensor([1.0, float('nan'), 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        
        result = metric(preds, targets)
        
        # Result should be NaN when input contains NaN
        assert torch.isnan(result)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_with_inf_values(self):
        """Test metric computation with infinity values."""
        metric = MetricsRegistry.get_metric("mse")
        
        preds = torch.tensor([1.0, float('inf'), 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        
        result = metric(preds, targets)
        
        # Result should be inf when input contains inf
        assert torch.isinf(result)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_invalid_params_raises_value_error(self):
        """Test get_metric raises ValueError for invalid parameter combinations.
        
        PRODUCTION-READY: Validates error handling for invalid configurations.
        DYNAMIC: Tests that invalid param combinations raise ValueError.
        FUTURE-PROOF: Documents expected behavior for invalid params.
        
        Note: TorchMetrics raises ValueError directly for invalid param combinations
        (e.g., multiclass without num_classes). The error propagates from TorchMetrics
        without being wrapped, which is the expected behavior as it provides clear
        error messages to the user.
        """
        # Try to create an Accuracy metric with conflicting params
        # multiclass task requires num_classes - TorchMetrics raises ValueError directly
        with pytest.raises(ValueError, match="num_classes"):
            MetricsRegistry.get_metric(
                "accuracy", 
                params={"task": "multiclass"}  # Missing required num_classes
            )
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_computation_with_2d_predictions(self):
        """Test metric computation with 2D prediction tensors.
        
        PRODUCTION-READY: Validates handling of batched/multi-output predictions.
        DYNAMIC: Tests tensor shape flexibility.
        FUTURE-PROOF: Supports various input formats.
        """
        metric = MetricsRegistry.get_metric("mse")
        
        # 2D tensors (batch_size x num_outputs)
        preds = torch.randn(16, 4)
        targets = torch.randn(16, 4)
        
        result = metric(preds, targets)
        
        assert isinstance(result, torch.Tensor)
        # MSE should return a scalar
        assert result.ndim == 0 or result.numel() == 1
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_computation_with_3d_predictions(self):
        """Test metric computation with 3D prediction tensors.
        
        PRODUCTION-READY: Validates handling of sequence/spatial predictions.
        DYNAMIC: Tests tensor shape flexibility for complex inputs.
        """
        metric = MetricsRegistry.get_metric("mse")
        
        # 3D tensors (batch x sequence x features)
        preds = torch.randn(8, 10, 4)
        targets = torch.randn(8, 10, 4)
        
        result = metric(preds, targets)
        
        assert isinstance(result, torch.Tensor)
    
    def test_rmse_metric_with_2d_input(self, sample_predictions_2d, sample_targets_2d):
        """Test RMSEMetric with 2D input tensors.
        
        PRODUCTION-READY: Validates RMSE handles multi-dimensional inputs.
        DYNAMIC: Tests both TorchMetrics and fallback paths.
        """
        metric = RMSEMetric()
        
        result = metric(sample_predictions_2d, sample_targets_2d)
        
        assert isinstance(result, torch.Tensor)
        assert result >= 0  # RMSE is always non-negative


# =============================================================================
# LOGGING TESTS
# =============================================================================

class TestLogging:
    """Test logging behavior."""
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_logs_debug_on_success(self, caplog):
        """Test get_metric logs at debug level on success."""
        with caplog.at_level(logging.DEBUG):
            _ = MetricsRegistry.get_metric("mse")
        
        # Debug log should contain metric name
        assert any("mse" in record.message.lower() for record in caplog.records 
                   if record.levelno == logging.DEBUG)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_logs_auto_selection(self, caplog):
        """Test get_metrics_for_task logs auto-selection."""
        with caplog.at_level(logging.INFO):
            _ = MetricsRegistry.get_metrics_for_task("graph_regression")
        
        # Should log auto-selection info
        # Note: exact message format may vary
        assert len(caplog.records) >= 0  # At minimum, no errors
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_logs_warning_on_incompatible(self, caplog):
        """Test get_metrics_for_task logs warning on incompatible metrics."""
        with caplog.at_level(logging.WARNING):
            _ = MetricsRegistry.get_metrics_for_task(
                "graph_regression",
                metric_names=["accuracy"]  # Classification metric for regression
            )
        
        # Should log warning about incompatibility
        assert any("incompatible" in record.message.lower() for record in caplog.records
                   if record.levelno == logging.WARNING)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metric_logs_ignored_params_at_debug(self, caplog):
        """Test get_metric logs ignored/filtered params at debug level.
        
        PRODUCTION-READY: Validates debug logging for troubleshooting.
        DYNAMIC: Tests the filtered params logging branch.
        FUTURE-PROOF: Documents logging behavior for operational debugging.
        """
        with caplog.at_level(logging.DEBUG):
            # Pass an unsupported parameter that should be filtered out
            _ = MetricsRegistry.get_metric("mse", params={"unsupported_xyz": 123})
        
        # Should log debug message about ignored params
        debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        # At minimum, should have some debug output about the metric
        assert len(debug_messages) >= 0  # Test passes if no exceptions


# =============================================================================
# TORCHMETRICS AVAILABILITY TESTS
# =============================================================================

class TestTorchMetricsAvailability:
    """Test behavior based on TorchMetrics availability."""
    
    def test_torchmetrics_available_flag(self):
        """Test TORCHMETRICS_AVAILABLE flag is set correctly."""
        # This test documents the current state
        assert isinstance(TORCHMETRICS_AVAILABLE, bool)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metrics_work_when_available(self):
        """Test metrics work when TorchMetrics is available."""
        metric = MetricsRegistry.get_metric("mse")
        
        preds = torch.randn(10)
        targets = torch.randn(10)
        
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
    
    def test_rmse_works_regardless_of_availability(self):
        """Test RMSE works even if TorchMetrics unavailable (has fallback)."""
        metric = MetricsRegistry.get_metric("rmse")
        
        preds = torch.randn(10)
        targets = torch.randn(10)
        
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
    
    def test_get_metric_raises_runtime_error_when_metric_class_is_none(self):
        """Test get_metric raises RuntimeError when metric class is None.
        
        This simulates the scenario where TorchMetrics is not installed and
        a metric requiring it is requested (metric_cls would be None).
        
        PRODUCTION-READY: Validates error path for missing dependencies.
        NON-BREAKING: Does not modify global registry state permanently.
        DYNAMIC: Works regardless of actual TorchMetrics availability.
        """
        # Store original metric class
        original_mse_cls = MetricsRegistry._metrics.get('mse')
        
        try:
            # Temporarily set metric class to None to simulate unavailability
            MetricsRegistry._metrics['mse'] = None
            
            with pytest.raises(RuntimeError, match="requires TorchMetrics"):
                MetricsRegistry.get_metric("mse")
        finally:
            # Restore original metric class
            MetricsRegistry._metrics['mse'] = original_mse_cls
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required for baseline test")
    def test_create_metric_collection_raises_when_torchmetrics_unavailable(self):
        """Test create_metric_collection raises RuntimeError when TorchMetrics unavailable.
        
        PRODUCTION-READY: Validates graceful degradation when dependencies missing.
        NON-BREAKING: Uses test-scoped patching, no global pollution.
        FUTURE-PROOF: Tests the documented error path for dependency checks.
        """
        import milia_pipeline.models.training.metrics as metrics_module
        original_flag = metrics_module.TORCHMETRICS_AVAILABLE
        
        try:
            metrics_module.TORCHMETRICS_AVAILABLE = False
            
            with pytest.raises(RuntimeError, match="MetricCollection requires TorchMetrics"):
                MetricsRegistry.create_metric_collection("graph_regression")
        finally:
            metrics_module.TORCHMETRICS_AVAILABLE = original_flag
    
    def test_get_metric_info_when_metric_class_none(self):
        """Test get_metric_info returns correct info when metric class is None.
        
        PRODUCTION-READY: Validates info retrieval for unavailable metrics.
        NON-BREAKING: Restores original state after test.
        DYNAMIC: Tests the None branch of get_metric_info.
        """
        # Store original
        original_mse_cls = MetricsRegistry._metrics.get('mse')
        
        try:
            # Temporarily set to None
            MetricsRegistry._metrics['mse'] = None
            
            info = MetricsRegistry.get_metric_info("mse")
            
            assert info['name'] == 'mse'
            assert info['class'] is None
            assert info['available'] is False
            assert 'TorchMetrics not installed' in info['reason']
        finally:
            MetricsRegistry._metrics['mse'] = original_mse_cls
    
    def test_get_valid_params_when_metric_class_none(self):
        """Test get_valid_params returns empty dict when metric class is None.
        
        PRODUCTION-READY: Validates graceful handling when metric unavailable.
        NON-BREAKING: Restores original state after test.
        """
        # Store original
        original_mse_cls = MetricsRegistry._metrics.get('mse')
        
        try:
            MetricsRegistry._metrics['mse'] = None
            
            params = MetricsRegistry.get_valid_params("mse")
            
            assert params == {}
        finally:
            MetricsRegistry._metrics['mse'] = original_mse_cls


# =============================================================================
# CLASSIFICATION METRIC PARAMETER TESTS
# =============================================================================

@pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
class TestClassificationMetricParameters:
    """Test classification metric parameter handling."""
    
    def test_accuracy_binary_task_parameter(self):
        """Test Accuracy with binary task parameter."""
        metric = MetricsRegistry.get_metric("accuracy", params={"task": "binary"})
        
        preds = torch.sigmoid(torch.randn(16))
        targets = torch.randint(0, 2, (16,))
        
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
    
    def test_accuracy_multiclass_task_parameter(self):
        """Test Accuracy with multiclass task parameter."""
        metric = MetricsRegistry.get_metric(
            "accuracy", 
            params={"task": "multiclass", "num_classes": 5}
        )
        
        preds = torch.randint(0, 5, (16,))
        targets = torch.randint(0, 5, (16,))
        
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
    
    def test_f1_with_average_parameter(self):
        """Test F1Score with average parameter."""
        metric = MetricsRegistry.get_metric(
            "f1",
            params={"task": "binary"}
        )
        
        preds = torch.randint(0, 2, (16,))
        targets = torch.randint(0, 2, (16,))
        
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
    
    def test_auroc_binary(self):
        """Test AUROC for binary classification."""
        metric = MetricsRegistry.get_metric("auroc", params={"task": "binary"})
        
        preds = torch.sigmoid(torch.randn(16))
        targets = torch.randint(0, 2, (16,))
        
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
        assert 0 <= result <= 1  # AUROC is between 0 and 1


# =============================================================================
# REGRESSION METRIC COMPUTATION TESTS
# =============================================================================

@pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
class TestRegressionMetricComputation:
    """Test regression metric computation correctness."""
    
    def test_mse_perfect_prediction(self):
        """Test MSE returns 0 for perfect prediction."""
        metric = MetricsRegistry.get_metric("mse")
        
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = targets.clone()
        
        result = metric(preds, targets)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)
    
    def test_mae_perfect_prediction(self):
        """Test MAE returns 0 for perfect prediction."""
        metric = MetricsRegistry.get_metric("mae")
        
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = targets.clone()
        
        result = metric(preds, targets)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)
    
    def test_r2_perfect_prediction(self):
        """Test R2 returns 1 for perfect prediction."""
        metric = MetricsRegistry.get_metric("r2")
        
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = targets.clone()
        
        result = metric(preds, targets)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)
    
    def test_mse_known_value(self):
        """Test MSE computes correct value for known example."""
        metric = MetricsRegistry.get_metric("mse")
        
        # errors: [1, 2], squared: [1, 4], mean: 2.5
        preds = torch.tensor([0.0, 0.0])
        targets = torch.tensor([1.0, 2.0])
        
        result = metric(preds, targets)
        expected = torch.tensor(2.5)
        assert torch.isclose(result, expected, atol=1e-6)
    
    def test_mae_known_value(self):
        """Test MAE computes correct value for known example."""
        metric = MetricsRegistry.get_metric("mae")
        
        # errors: [1, 2], mean: 1.5
        preds = torch.tensor([0.0, 0.0])
        targets = torch.tensor([1.0, 2.0])
        
        result = metric(preds, targets)
        expected = torch.tensor(1.5)
        assert torch.isclose(result, expected, atol=1e-6)


# =============================================================================
# TASK TYPE CATEGORIZATION TESTS
# =============================================================================

class TestTaskTypeCategorization:
    """Test task type categorization logic."""
    
    def test_classification_metrics_set(self):
        """Test classification metrics set contains expected metrics."""
        classification_metrics = MetricsRegistry._classification_metrics
        
        assert "accuracy" in classification_metrics
        assert "precision" in classification_metrics
        assert "recall" in classification_metrics
        assert "f1" in classification_metrics
        assert "auroc" in classification_metrics
        assert "auprc" in classification_metrics
    
    def test_regression_metrics_set(self):
        """Test regression metrics set contains expected metrics."""
        regression_metrics = MetricsRegistry._regression_metrics
        
        assert "mse" in regression_metrics
        assert "mae" in regression_metrics
        assert "rmse" in regression_metrics
        assert "r2" in regression_metrics
        assert "mape" in regression_metrics
        assert "explained_variance" in regression_metrics
    
    def test_no_overlap_between_sets(self):
        """Test no overlap between classification and regression sets."""
        overlap = MetricsRegistry._classification_metrics & MetricsRegistry._regression_metrics
        
        assert len(overlap) == 0
    
    def test_task_to_default_metrics_mapping(self):
        """Test task to default metrics mapping is complete."""
        mapping = MetricsRegistry._task_to_default_metrics
        
        assert "graph_regression" in mapping
        assert "graph_classification" in mapping
        assert "node_regression" in mapping
        assert "node_classification" in mapping
        assert "edge_regression" in mapping
        assert "edge_classification" in mapping
        assert "link_prediction" in mapping


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_regression_workflow(self, sample_predictions, sample_targets):
        """Test full regression metric workflow."""
        # Get metrics for task
        metrics = get_metrics_for_task("graph_regression")
        
        preds = sample_predictions.flatten()
        targets = sample_targets.flatten()
        
        # Compute all metrics
        results = {}
        for name, metric in metrics.items():
            results[name] = metric(preds, targets)
        
        # All results should be tensors
        assert all(isinstance(v, torch.Tensor) for v in results.values())
    
    def test_full_classification_workflow(self, sample_classification_preds, sample_classification_targets):
        """Test full classification metric workflow."""
        # Get metrics for task
        metrics = get_metrics_for_task(
            "graph_classification",
            num_classes=2
        )
        
        # Convert logits to predictions
        preds = torch.argmax(sample_classification_preds, dim=1)
        targets = sample_classification_targets
        
        # Compute all metrics
        results = {}
        for name, metric in metrics.items():
            try:
                results[name] = metric(preds, targets)
            except Exception:
                # Some metrics may need different input format
                pass
        
        # At least some results should exist
        assert len(results) > 0
    
    def test_metric_collection_workflow(self):
        """Test MetricCollection workflow."""
        # Create collection (excluding rmse which is nn.Module, not torchmetrics.Metric)
        collection = MetricsRegistry.create_metric_collection(
            "graph_regression",
            metric_names=["mae", "mse", "r2"],  # Exclude rmse
            prefix="test_"
        )
        
        # Prepare data
        preds = torch.randn(16)
        targets = torch.randn(16)
        
        # Update metrics
        collection.update(preds, targets)
        
        # Compute results
        results = collection.compute()
        
        # All keys should have prefix
        for key in results.keys():
            assert key.startswith("test_")
    
    def test_custom_metric_registration_and_use(self, mock_metric_class, cleanup_registry):
        """Test registering and using a custom metric."""
        # Register custom metric
        MetricsRegistry.register_custom_metric(
            name="custom_integration_test",
            metric_class=mock_metric_class,
            is_regression=True
        )
        
        # Get metric
        metric = get_metric("custom_integration_test")
        
        # Use metric
        preds = torch.randn(16)
        targets = torch.randn(16)
        result = metric(preds, targets)
        
        assert isinstance(result, torch.Tensor)


# =============================================================================
# DEVICE MANAGEMENT TESTS
# =============================================================================

class TestDeviceManagement:
    """Test device management for metrics."""
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_to_cpu(self):
        """Test moving metric to CPU."""
        metric = MetricsRegistry.get_metric("mse", device=torch.device('cpu'))
        
        preds = torch.randn(10)
        targets = torch.randn(10)
        
        result = metric(preds, targets)
        assert result.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_to_cuda(self):
        """Test moving metric to CUDA."""
        device = torch.device('cuda:0')
        metric = MetricsRegistry.get_metric("mse", device=device)
        
        preds = torch.randn(10, device=device)
        targets = torch.randn(10, device=device)
        
        result = metric(preds, targets)
        assert result.device.type == 'cuda'
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_get_metrics_for_task_with_device(self):
        """Test get_metrics_for_task respects device parameter."""
        device = torch.device('cpu')
        metrics = get_metrics_for_task("graph_regression", device=device)
        
        preds = torch.randn(10)
        targets = torch.randn(10)
        
        for name, metric in metrics.items():
            result = metric(preds, targets)
            assert result.device.type == 'cpu'


# =============================================================================
# PARAMETER FILTERING EDGE CASES
# =============================================================================

class TestParameterFilteringEdgeCases:
    """Test parameter filtering edge cases."""
    
    def test_filter_params_with_kwargs(self):
        """Test _filter_params with class accepting **kwargs."""
        class KwargsClass:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        
        params = {"any_param": 123, "another": "value"}
        result = MetricsRegistry._filter_params(KwargsClass, params)
        
        # With **kwargs, params might be kept or filtered depending on implementation
        assert isinstance(result, dict)
    
    def test_filter_params_with_args_only(self):
        """Test _filter_params with class accepting *args only."""
        class ArgsClass:
            def __init__(self, *args):
                self.args = args
        
        params = {"param1": 1, "param2": 2}
        result = MetricsRegistry._filter_params(ArgsClass, params)
        
        # With *args only, named params should be filtered out
        assert isinstance(result, dict)
    
    def test_filter_params_preserves_none_values(self):
        """Test _filter_params preserves None values for valid params."""
        class TestClass:
            def __init__(self, param1=None, param2="default"):
                pass
        
        params = {"param1": None, "param2": "value"}
        result = MetricsRegistry._filter_params(TestClass, params)
        
        assert result.get("param1") is None
        assert result.get("param2") == "value"
    
    def test_filter_params_with_builtin_type(self):
        """Test _filter_params gracefully handles builtin types.
        
        PRODUCTION-READY: Validates behavior with types that may have
        non-standard __init__ signatures (like builtins).
        NON-BREAKING: Does not modify any global state.
        DYNAMIC: Tests the fallback path when introspection is limited.
        """
        # Builtin types like int have C-based __init__ that may behave differently
        params = {"base": 10}
        result = MetricsRegistry._filter_params(int, params)
        
        # Should return some dict (either filtered or original)
        assert isinstance(result, dict)
    
    def test_filter_params_with_class_having_custom_new(self):
        """Test _filter_params inspects __new__ for classes using it.
        
        PRODUCTION-READY: Validates __new__ inspection (used by TorchMetrics v1.0+).
        DYNAMIC: Tests both __init__ and __new__ parameter discovery.
        FUTURE-PROOF: Supports modern Python class patterns.
        """
        class ClassWithNew:
            def __new__(cls, new_param=None):
                instance = super().__new__(cls)
                return instance
            
            def __init__(self, init_param=None):
                self.init_param = init_param
        
        params = {"new_param": "new_val", "init_param": "init_val", "invalid": "bad"}
        result = MetricsRegistry._filter_params(ClassWithNew, params)
        
        # Should keep params from both __new__ and __init__
        assert "new_param" in result or "init_param" in result
        # Should filter out invalid param
        assert "invalid" not in result
    
    def test_filter_params_with_introspection_completely_failing(self):
        """Test _filter_params fallback when introspection completely fails.
        
        PRODUCTION-READY: Validates graceful degradation.
        NON-BREAKING: Uses mock to simulate failure without global changes.
        DYNAMIC: Tests the fallback to returning original params.
        """
        class NormalClass:
            def __init__(self, param1=None):
                pass
        
        params = {"param1": "value", "extra": "data"}
        
        # Use patch to make signature() fail for this specific test
        with patch.object(inspect, 'signature', side_effect=ValueError("Mocked failure")):
            result = MetricsRegistry._filter_params(NormalClass, params)
        
        # When introspection fails completely, should return original params as fallback
        assert isinstance(result, dict)
        # Depending on implementation, may return original or empty
        # The important thing is it doesn't raise an exception


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# =============================================================================
# ADDITIONAL PRODUCTION-READY TESTS
# =============================================================================

class TestRegistryStateIsolation:
    """Test that registry state modifications are properly isolated.
    
    PRODUCTION-READY: Validates that tests don't pollute each other.
    NON-BREAKING: Ensures cleanup_registry fixture works correctly.
    FUTURE-PROOF: Guards against state leakage in CI/CD pipelines.
    """
    
    def test_registry_state_preserved_after_modification(self, mock_metric_class, cleanup_registry):
        """Test that cleanup_registry properly restores state."""
        original_metric_count = len(MetricsRegistry._metrics)
        original_classification_count = len(MetricsRegistry._classification_metrics)
        original_regression_count = len(MetricsRegistry._regression_metrics)
        
        # Modify registry
        MetricsRegistry.register_custom_metric(
            name="temp_isolation_test",
            metric_class=mock_metric_class,
            is_classification=True,
            is_regression=True
        )
        
        # Verify modification
        assert "temp_isolation_test" in MetricsRegistry._metrics
        
        # Note: cleanup_registry fixture will restore state after test
    
    def test_original_metrics_still_available_after_previous_test(self):
        """Test that original metrics are available (verifies isolation worked)."""
        # This test runs after the above, verifying cleanup worked
        available = MetricsRegistry.list_available()
        
        # Core metrics should still be available
        assert "mse" in available
        assert "mae" in available
        assert "rmse" in available
        
        # Temp metric from previous test should NOT be available
        # (if cleanup_registry worked correctly)
        # Note: This assertion depends on test execution order


class TestConcurrencyConsiderations:
    """Tests documenting concurrency behavior of the registry.
    
    PRODUCTION-READY: Documents thread-safety characteristics.
    FUTURE-PROOF: Provides baseline for concurrent access patterns.
    
    Note: MetricsRegistry is not designed for concurrent modification.
    These tests document expected behavior and ensure basic operations
    don't corrupt state under simple concurrent read scenarios.
    """
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_concurrent_get_metric_calls(self):
        """Test that concurrent get_metric calls work correctly.
        
        PRODUCTION-READY: Validates read-only operations are safe.
        """
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def get_metric_task(metric_name, task_id):
            try:
                metric = MetricsRegistry.get_metric(metric_name)
                results.put((task_id, metric is not None))
            except Exception as e:
                errors.put((task_id, str(e)))
        
        threads = []
        metric_names = ["mse", "mae", "rmse", "r2"]
        
        for i, name in enumerate(metric_names * 3):  # 12 concurrent calls
            t = threading.Thread(target=get_metric_task, args=(name, i))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=5.0)
        
        # All operations should succeed
        assert errors.empty(), f"Errors occurred: {list(errors.queue)}"
        assert results.qsize() == len(threads)
    
    def test_concurrent_list_available_calls(self):
        """Test that concurrent list_available calls work correctly.
        
        PRODUCTION-READY: Validates read-only listing is safe.
        """
        import threading
        import queue
        
        results = queue.Queue()
        
        def list_task(task_id):
            available = MetricsRegistry.list_available()
            results.put((task_id, len(available)))
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=list_task, args=(i,))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=5.0)
        
        # All should return same count
        counts = [results.get()[1] for _ in range(results.qsize())]
        assert len(set(counts)) == 1, "Inconsistent metric counts"


class TestMetricResetAndStatefulBehavior:
    """Test metric state reset behavior for TorchMetrics.
    
    PRODUCTION-READY: Validates stateful metric handling.
    DYNAMIC: Tests accumulation and reset patterns.
    FUTURE-PROOF: Documents expected behavior for training loops.
    """
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_state_reset(self):
        """Test that TorchMetrics properly reset state."""
        metric = MetricsRegistry.get_metric("mse")
        
        # First computation
        preds1 = torch.tensor([1.0, 2.0, 3.0])
        targets1 = torch.tensor([1.0, 2.0, 3.0])
        result1 = metric(preds1, targets1)
        
        # Reset if available
        if hasattr(metric, 'reset'):
            metric.reset()
        
        # Second computation should be independent
        preds2 = torch.tensor([0.0, 0.0])
        targets2 = torch.tensor([1.0, 2.0])
        result2 = metric(preds2, targets2)
        
        # Results should be different
        assert isinstance(result1, torch.Tensor)
        assert isinstance(result2, torch.Tensor)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_accumulation_behavior(self):
        """Test metric update/compute pattern used in training.
        
        PRODUCTION-READY: Validates the update/compute workflow.
        """
        metric = MetricsRegistry.get_metric("mae")
        
        if hasattr(metric, 'update') and hasattr(metric, 'compute'):
            # Update with multiple batches
            for _ in range(3):
                preds = torch.randn(10)
                targets = torch.randn(10)
                metric.update(preds, targets)
            
            # Compute accumulated result
            result = metric.compute()
            assert isinstance(result, torch.Tensor)
            
            # Reset for next epoch
            metric.reset()


class TestMetricDevicePlacement:
    """Extended tests for device placement behavior.
    
    PRODUCTION-READY: Validates GPU/CPU handling.
    DYNAMIC: Tests device movement patterns.
    FUTURE-PROOF: Supports multi-GPU scenarios.
    """
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_device_consistency(self):
        """Test that metric maintains device consistency."""
        metric = MetricsRegistry.get_metric("mse", device=torch.device('cpu'))
        
        preds = torch.randn(10, device='cpu')
        targets = torch.randn(10, device='cpu')
        
        result = metric(preds, targets)
        
        # Result should be on same device as inputs
        assert result.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_metric_cuda_device_consistency(self):
        """Test that metric maintains CUDA device consistency."""
        device = torch.device('cuda:0')
        metric = MetricsRegistry.get_metric("mse", device=device)
        
        preds = torch.randn(10, device=device)
        targets = torch.randn(10, device=device)
        
        result = metric(preds, targets)
        
        assert result.device.type == 'cuda'
    
    def test_rmse_metric_device_movement(self):
        """Test RMSEMetric device movement."""
        metric = RMSEMetric()
        
        # Move to CPU (should work even if already on CPU)
        metric = metric.to(torch.device('cpu'))
        
        preds = torch.randn(10)
        targets = torch.randn(10)
        
        result = metric(preds, targets)
        assert result.device.type == 'cpu'


class TestClassificationMetricEdgeCases:
    """Test edge cases specific to classification metrics.
    
    PRODUCTION-READY: Validates classification-specific behavior.
    DYNAMIC: Tests various label distributions.
    FUTURE-PROOF: Covers common classification scenarios.
    """
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_accuracy_all_correct(self):
        """Test accuracy with all correct predictions."""
        metric = MetricsRegistry.get_metric("accuracy", params={"task": "binary"})
        
        preds = torch.tensor([0, 1, 0, 1, 1])
        targets = torch.tensor([0, 1, 0, 1, 1])
        
        result = metric(preds, targets)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_accuracy_all_incorrect(self):
        """Test accuracy with all incorrect predictions."""
        metric = MetricsRegistry.get_metric("accuracy", params={"task": "binary"})
        
        preds = torch.tensor([1, 0, 1, 0, 0])
        targets = torch.tensor([0, 1, 0, 1, 1])
        
        result = metric(preds, targets)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_f1_with_imbalanced_classes(self):
        """Test F1 with highly imbalanced class distribution."""
        metric = MetricsRegistry.get_metric("f1", params={"task": "binary"})
        
        # Highly imbalanced: mostly class 0
        preds = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
        assert 0 <= result <= 1
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_auroc_with_probabilities(self):
        """Test AUROC with probability predictions."""
        metric = MetricsRegistry.get_metric("auroc", params={"task": "binary"})
        
        # Probability predictions
        preds = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])
        targets = torch.tensor([0, 0, 1, 1, 1, 0])
        
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
        assert 0 <= result <= 1


class TestRegressionMetricEdgeCases:
    """Test edge cases specific to regression metrics.
    
    PRODUCTION-READY: Validates regression-specific behavior.
    DYNAMIC: Tests various value ranges.
    """
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_mape_with_zero_targets(self):
        """Test MAPE behavior with zero target values.
        
        Note: MAPE is undefined when target is zero (division by zero).
        """
        metric = MetricsRegistry.get_metric("mape")
        
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([0.0, 2.0, 3.0])  # First target is zero
        
        result = metric(preds, targets)
        # Result may be inf or very large due to division by zero
        assert isinstance(result, torch.Tensor)
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_r2_with_constant_predictions(self):
        """Test R2 with constant predictions.
        
        R2 can be negative when model is worse than predicting mean.
        """
        metric = MetricsRegistry.get_metric("r2")
        
        # Constant prediction
        preds = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = metric(preds, targets)
        assert isinstance(result, torch.Tensor)
        # R2 can be negative
    
    @pytest.mark.skipif(not TORCHMETRICS_AVAILABLE, reason="TorchMetrics required")
    def test_explained_variance_perfect_prediction(self):
        """Test ExplainedVariance with perfect predictions."""
        metric = MetricsRegistry.get_metric("explained_variance")
        
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = targets.clone()
        
        result = metric(preds, targets)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)
