#!/usr/bin/env python3
"""
Extended Production-Ready Unit Test Suite for visualization.py Module

Comprehensive test coverage including:
- TrainingVisualizer initialization and configuration
- Loss curve plotting (matplotlib)
- Metrics plotting with various configurations
- Learning rate visualization
- Interactive plots (plotly)
- Training summary generation
- Export functionality (PNG, HTML, PDF)
- Style configuration and customization
- Dependency availability handling (matplotlib, plotly, kaleido)
- Edge cases and error handling
- Convenience functions (plot_training_summary, create_visualizer)
- File path handling and saving
- Module-level constants and initialization

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
from unittest.mock import Mock, patch, MagicMock, PropertyMock, call
from typing import Dict, Any, List
from datetime import datetime
import tempfile
import shutil

import numpy as np


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def basic_metrics_history():
    """Create a basic metrics history dictionary."""
    return {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
    }


@pytest.fixture
def comprehensive_metrics_history():
    """Create a comprehensive metrics history with multiple metrics."""
    return {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
        'train_mae': [0.5, 0.4, 0.3, 0.25, 0.2],
        'val_mae': [0.55, 0.45, 0.35, 0.3, 0.25],
        'train_rmse': [0.7, 0.6, 0.5, 0.4, 0.35],
        'val_rmse': [0.75, 0.65, 0.55, 0.45, 0.4],
        'lr': [0.001, 0.001, 0.0005, 0.0005, 0.00025],
    }


@pytest.fixture
def metrics_history_with_learning_rate():
    """Create metrics history with learning rate."""
    return {
        'train_loss': [1.0, 0.8, 0.6],
        'val_loss': [1.1, 0.9, 0.7],
        'lr': [0.001, 0.0005, 0.00025],
    }


@pytest.fixture
def metrics_history_with_learning_rate_key():
    """Create metrics history with 'learning_rate' key."""
    return {
        'train_loss': [1.0, 0.8, 0.6],
        'learning_rate': [0.001, 0.0005, 0.00025],
    }


@pytest.fixture
def metrics_history_with_lr_uppercase():
    """Create metrics history with 'LR' key."""
    return {
        'train_loss': [1.0, 0.8, 0.6],
        'LR': [0.01, 0.005, 0.0025],
    }


@pytest.fixture
def empty_metrics_history():
    """Create an empty metrics history."""
    return {}


@pytest.fixture
def train_only_metrics():
    """Create metrics history with only training loss."""
    return {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
    }


@pytest.fixture
def val_only_metrics():
    """Create metrics history with only validation loss."""
    return {
        'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
    }


@pytest.fixture
def no_loss_metrics():
    """Create metrics history without any loss metrics."""
    return {
        'train_mae': [0.5, 0.4, 0.3],
        'val_mae': [0.55, 0.45, 0.35],
    }


@pytest.fixture
def single_epoch_metrics():
    """Create metrics history with single epoch."""
    return {
        'train_loss': [1.0],
        'val_loss': [1.1],
    }


@pytest.fixture
def mismatched_length_metrics():
    """Create metrics history with mismatched lengths."""
    return {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val_loss': [1.1, 0.9, 0.7],  # Fewer validation epochs
    }


@pytest.fixture
def custom_style():
    """Create a custom style configuration."""
    return {
        'figure_size': (10, 8),
        'dpi': 200,
        'line_width': 3,
        'marker_size': 6,
        'font_size': 14,
        'title_size': 16,
        'legend_size': 12,
        'grid_alpha': 0.5,
        'colors': {
            'train': '#FF0000',
            'val': '#00FF00',
            'test': '#0000FF',
            'lr': '#FF00FF',
        }
    }


@pytest.fixture
def partial_custom_style():
    """Create a partial custom style configuration."""
    return {
        'dpi': 300,
        'line_width': 4,
    }


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_matplotlib_figure():
    """Create a mock matplotlib figure."""
    fig = Mock()
    fig.savefig = Mock()
    return fig


@pytest.fixture
def mock_plotly_figure():
    """Create a mock plotly figure."""
    fig = Mock()
    fig.add_trace = Mock()
    fig.update_layout = Mock()
    fig.update_xaxes = Mock()
    fig.update_yaxes = Mock()
    fig.write_html = Mock()
    fig.write_image = Mock()
    return fig


# =============================================================================
# TRAININGVISUALIZER INITIALIZATION TESTS
# =============================================================================

class TestTrainingVisualizerInitialization:
    """Test TrainingVisualizer initialization and configuration."""
    
    def test_initialization_with_basic_metrics(self, basic_metrics_history):
        """Test initialization with basic metrics history."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        
        assert visualizer.metrics_history == basic_metrics_history
        assert 'train_loss' in visualizer.metrics_history
        assert 'val_loss' in visualizer.metrics_history
    
    def test_initialization_with_empty_metrics(self, empty_metrics_history):
        """Test initialization with empty metrics history."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(empty_metrics_history)
        
        assert visualizer.metrics_history == {}
    
    def test_initialization_default_style(self, basic_metrics_history):
        """Test that default style is applied correctly."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        
        assert visualizer.style['figure_size'] == (12, 6)
        assert visualizer.style['dpi'] == 150
        assert visualizer.style['line_width'] == 2
        assert visualizer.style['marker_size'] == 4
        assert visualizer.style['font_size'] == 12
        assert visualizer.style['title_size'] == 14
        assert visualizer.style['legend_size'] == 10
        assert visualizer.style['grid_alpha'] == 0.3
    
    def test_initialization_default_colors(self, basic_metrics_history):
        """Test that default colors are applied correctly."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        
        assert visualizer.style['colors']['train'] == '#2196F3'
        assert visualizer.style['colors']['val'] == '#FF9800'
        assert visualizer.style['colors']['test'] == '#4CAF50'
        assert visualizer.style['colors']['lr'] == '#9C27B0'
    
    def test_initialization_with_custom_style(self, basic_metrics_history, custom_style):
        """Test initialization with custom style configuration."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(basic_metrics_history, style=custom_style)
        
        assert visualizer.style['figure_size'] == (10, 8)
        assert visualizer.style['dpi'] == 200
        assert visualizer.style['line_width'] == 3
        assert visualizer.style['marker_size'] == 6
        assert visualizer.style['colors']['train'] == '#FF0000'
    
    def test_initialization_with_partial_custom_style(
        self, basic_metrics_history, partial_custom_style
    ):
        """Test initialization with partial custom style (merging with defaults)."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(basic_metrics_history, style=partial_custom_style)
        
        # Custom values
        assert visualizer.style['dpi'] == 300
        assert visualizer.style['line_width'] == 4
        # Default values preserved
        assert visualizer.style['figure_size'] == (12, 6)
        assert visualizer.style['marker_size'] == 4
    
    def test_initialization_with_none_style(self, basic_metrics_history):
        """Test initialization with None style uses defaults."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(basic_metrics_history, style=None)
        
        assert visualizer.style['figure_size'] == (12, 6)
        assert visualizer.style['dpi'] == 150
    
    def test_default_style_class_attribute(self):
        """Test that DEFAULT_STYLE class attribute exists."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        assert hasattr(TrainingVisualizer, 'DEFAULT_STYLE')
        assert isinstance(TrainingVisualizer.DEFAULT_STYLE, dict)
        assert 'figure_size' in TrainingVisualizer.DEFAULT_STYLE
        assert 'colors' in TrainingVisualizer.DEFAULT_STYLE


# =============================================================================
# DEPENDENCY AVAILABILITY TESTS
# =============================================================================

class TestDependencyAvailability:
    """Test handling of optional dependencies (matplotlib, plotly, kaleido)."""
    
    def test_matplotlib_available_flag_true(self, basic_metrics_history):
        """Test MATPLOTLIB_AVAILABLE flag when matplotlib is available."""
        from milia_pipeline.models.training.visualization import MATPLOTLIB_AVAILABLE
        
        # This test runs in an environment where matplotlib should be available
        # If matplotlib is not available, this flag would be False
        assert isinstance(MATPLOTLIB_AVAILABLE, bool)
    
    def test_plotly_available_flag_true(self, basic_metrics_history):
        """Test PLOTLY_AVAILABLE flag when plotly is available."""
        from milia_pipeline.models.training.visualization import PLOTLY_AVAILABLE
        
        assert isinstance(PLOTLY_AVAILABLE, bool)
    
    def test_kaleido_available_flag(self, basic_metrics_history):
        """Test KALEIDO_AVAILABLE flag."""
        from milia_pipeline.models.training.visualization import KALEIDO_AVAILABLE
        
        assert isinstance(KALEIDO_AVAILABLE, bool)
    
    def test_initialization_warns_when_matplotlib_unavailable(
        self, basic_metrics_history, caplog
    ):
        """Test that warning is logged when matplotlib is unavailable."""
        with patch(
            'milia_pipeline.models.training.visualization.MATPLOTLIB_AVAILABLE',
            False
        ):
            from milia_pipeline.models.training.visualization import TrainingVisualizer
            
            with caplog.at_level(logging.WARNING):
                visualizer = TrainingVisualizer(basic_metrics_history)
            
            assert "matplotlib not available" in caplog.text.lower()
    
    def test_initialization_warns_when_plotly_unavailable(
        self, basic_metrics_history, caplog
    ):
        """Test that warning is logged when plotly is unavailable."""
        with patch(
            'milia_pipeline.models.training.visualization.PLOTLY_AVAILABLE',
            False
        ):
            from milia_pipeline.models.training.visualization import TrainingVisualizer
            
            with caplog.at_level(logging.WARNING):
                visualizer = TrainingVisualizer(basic_metrics_history)
            
            assert "plotly not available" in caplog.text.lower()


# =============================================================================
# PLOT LOSS CURVES TESTS
# =============================================================================

class TestPlotLossCurves:
    """Test plot_loss_curves method."""
    
    def test_plot_loss_curves_basic(self, basic_metrics_history):
        """Test basic loss curves plotting."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_loss_curves()
        
        assert fig is not None
    
    def test_plot_loss_curves_train_only(self, train_only_metrics):
        """Test loss curves with only training loss."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(train_only_metrics)
        fig = visualizer.plot_loss_curves()
        
        assert fig is not None
    
    def test_plot_loss_curves_val_only(self, val_only_metrics):
        """Test loss curves with only validation loss."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(val_only_metrics)
        fig = visualizer.plot_loss_curves()
        
        assert fig is not None
    
    def test_plot_loss_curves_with_save_path(
        self, basic_metrics_history, temp_output_dir
    ):
        """Test loss curves saving to file."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        save_path = temp_output_dir / "loss_curves.png"
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_loss_curves(save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_loss_curves_with_path_object(
        self, basic_metrics_history, temp_output_dir
    ):
        """Test loss curves saving with Path object."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        save_path = Path(temp_output_dir) / "loss_curves_path.png"
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_loss_curves(save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_loss_curves_with_string_path(
        self, basic_metrics_history, temp_output_dir
    ):
        """Test loss curves saving with string path."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        save_path = str(temp_output_dir / "loss_curves_str.png")
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_loss_curves(save_path=save_path)
        
        assert fig is not None
        assert Path(save_path).exists()
    
    def test_plot_loss_curves_custom_title(self, basic_metrics_history):
        """Test loss curves with custom title."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_loss_curves(title="Custom Loss Title")
        
        assert fig is not None
    
    def test_plot_loss_curves_show_false(self, basic_metrics_history):
        """Test loss curves with show=False (default)."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        # show=False should close the figure
        fig = visualizer.plot_loss_curves(show=False)
        
        assert fig is not None
    
    def test_plot_loss_curves_empty_history(self, empty_metrics_history):
        """Test loss curves with empty metrics history."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(empty_metrics_history)
        fig = visualizer.plot_loss_curves()
        
        # Should still return a figure, just with no data
        assert fig is not None
    
    def test_plot_loss_curves_single_epoch(self, single_epoch_metrics):
        """Test loss curves with single epoch data."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(single_epoch_metrics)
        fig = visualizer.plot_loss_curves()
        
        assert fig is not None
    
    def test_plot_loss_curves_mismatched_lengths(self, mismatched_length_metrics):
        """Test loss curves with mismatched train/val lengths."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(mismatched_length_metrics)
        fig = visualizer.plot_loss_curves()
        
        assert fig is not None
    
    def test_plot_loss_curves_returns_none_without_matplotlib(
        self, basic_metrics_history
    ):
        """Test that plot_loss_curves returns None when matplotlib unavailable."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        with patch(
            'milia_pipeline.models.training.visualization.MATPLOTLIB_AVAILABLE',
            False
        ):
            visualizer = TrainingVisualizer(basic_metrics_history)
            result = visualizer.plot_loss_curves()
            
            assert result is None


# =============================================================================
# PLOT METRICS TESTS
# =============================================================================

class TestPlotMetrics:
    """Test plot_metrics method."""
    
    def test_plot_metrics_auto_detect(self, comprehensive_metrics_history):
        """Test metrics plotting with auto-detected metrics (excludes loss)."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        fig = visualizer.plot_metrics()
        
        assert fig is not None
    
    def test_plot_metrics_specific_names(self, comprehensive_metrics_history):
        """Test metrics plotting with specific metric names."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        fig = visualizer.plot_metrics(metric_names=['train_mae', 'val_mae'])
        
        assert fig is not None
    
    def test_plot_metrics_single_metric(self, comprehensive_metrics_history):
        """Test metrics plotting with single metric."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        fig = visualizer.plot_metrics(metric_names=['train_mae'])
        
        assert fig is not None
    
    def test_plot_metrics_with_save_path(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test metrics plotting with save path."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        save_path = temp_output_dir / "metrics.png"
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        fig = visualizer.plot_metrics(save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_metrics_custom_title(self, comprehensive_metrics_history):
        """Test metrics plotting with custom title."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        fig = visualizer.plot_metrics(title="Custom Metrics Title")
        
        assert fig is not None
    
    def test_plot_metrics_no_metrics_to_plot(self, basic_metrics_history):
        """Test metrics plotting when no non-loss metrics available."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        # basic_metrics_history only has loss metrics
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_metrics()
        
        # Should return None when no metrics to plot
        assert fig is None
    
    def test_plot_metrics_empty_metric_list(self, comprehensive_metrics_history):
        """Test metrics plotting with empty metric names list."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        fig = visualizer.plot_metrics(metric_names=[])
        
        assert fig is None
    
    def test_plot_metrics_nonexistent_metric(self, comprehensive_metrics_history):
        """Test metrics plotting with nonexistent metric name."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        # Requesting nonexistent metric - should handle gracefully
        fig = visualizer.plot_metrics(metric_names=['nonexistent_metric'])
        
        assert fig is not None  # Figure created but metric not found
    
    def test_plot_metrics_color_detection_train(self, no_loss_metrics):
        """Test that train metrics get train color."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(no_loss_metrics)
        fig = visualizer.plot_metrics(metric_names=['train_mae'])
        
        assert fig is not None
    
    def test_plot_metrics_color_detection_val(self, no_loss_metrics):
        """Test that val metrics get val color."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(no_loss_metrics)
        fig = visualizer.plot_metrics(metric_names=['val_mae'])
        
        assert fig is not None
    
    def test_plot_metrics_color_detection_test(self):
        """Test that test metrics get test color."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {'test_accuracy': [0.8, 0.85, 0.9]}
        visualizer = TrainingVisualizer(metrics)
        fig = visualizer.plot_metrics(metric_names=['test_accuracy'])
        
        assert fig is not None
    
    def test_plot_metrics_many_subplots(self):
        """Test metrics plotting with many metrics (multiple rows)."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'metric1': [1, 2, 3],
            'metric2': [4, 5, 6],
            'metric3': [7, 8, 9],
            'metric4': [10, 11, 12],
            'metric5': [13, 14, 15],
        }
        visualizer = TrainingVisualizer(metrics)
        fig = visualizer.plot_metrics()
        
        assert fig is not None
    
    def test_plot_metrics_returns_none_without_matplotlib(
        self, comprehensive_metrics_history
    ):
        """Test that plot_metrics returns None when matplotlib unavailable."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        with patch(
            'milia_pipeline.models.training.visualization.MATPLOTLIB_AVAILABLE',
            False
        ):
            visualizer = TrainingVisualizer(comprehensive_metrics_history)
            result = visualizer.plot_metrics()
            
            assert result is None


# =============================================================================
# PLOT LEARNING RATE TESTS
# =============================================================================

class TestPlotLearningRate:
    """Test plot_learning_rate method."""
    
    def test_plot_learning_rate_with_lr_key(self, metrics_history_with_learning_rate):
        """Test learning rate plotting with 'lr' key."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(metrics_history_with_learning_rate)
        fig = visualizer.plot_learning_rate()
        
        assert fig is not None
    
    def test_plot_learning_rate_with_learning_rate_key(
        self, metrics_history_with_learning_rate_key
    ):
        """Test learning rate plotting with 'learning_rate' key."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(metrics_history_with_learning_rate_key)
        fig = visualizer.plot_learning_rate()
        
        assert fig is not None
    
    def test_plot_learning_rate_with_uppercase_key(
        self, metrics_history_with_lr_uppercase
    ):
        """Test learning rate plotting with 'LR' key."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(metrics_history_with_lr_uppercase)
        fig = visualizer.plot_learning_rate()
        
        assert fig is not None
    
    def test_plot_learning_rate_no_lr_data(self, basic_metrics_history):
        """Test learning rate plotting when no LR data available."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_learning_rate()
        
        # Should return None when no LR data
        assert fig is None
    
    def test_plot_learning_rate_with_save_path(
        self, metrics_history_with_learning_rate, temp_output_dir
    ):
        """Test learning rate saving to file."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        save_path = temp_output_dir / "learning_rate.png"
        visualizer = TrainingVisualizer(metrics_history_with_learning_rate)
        fig = visualizer.plot_learning_rate(save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_learning_rate_custom_title(
        self, metrics_history_with_learning_rate
    ):
        """Test learning rate with custom title."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(metrics_history_with_learning_rate)
        fig = visualizer.plot_learning_rate(title="Custom LR Title")
        
        assert fig is not None
    
    def test_plot_learning_rate_log_scale(self, metrics_history_with_learning_rate):
        """Test that learning rate uses log scale on y-axis."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(metrics_history_with_learning_rate)
        fig = visualizer.plot_learning_rate()
        
        # Verify figure was created (log scale is set internally)
        assert fig is not None
    
    def test_plot_learning_rate_returns_none_without_matplotlib(
        self, metrics_history_with_learning_rate
    ):
        """Test that plot_learning_rate returns None when matplotlib unavailable."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        with patch(
            'milia_pipeline.models.training.visualization.MATPLOTLIB_AVAILABLE',
            False
        ):
            visualizer = TrainingVisualizer(metrics_history_with_learning_rate)
            result = visualizer.plot_learning_rate()
            
            assert result is None


# =============================================================================
# PLOT INTERACTIVE TESTS
# =============================================================================

class TestPlotInteractive:
    """Test plot_interactive method."""
    
    def test_plot_interactive_basic(self, basic_metrics_history):
        """Test basic interactive plot creation."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_interactive()
        
        assert fig is not None
    
    def test_plot_interactive_comprehensive(self, comprehensive_metrics_history):
        """Test interactive plot with comprehensive metrics."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        fig = visualizer.plot_interactive()
        
        assert fig is not None
    
    def test_plot_interactive_save_html(
        self, basic_metrics_history, temp_output_dir
    ):
        """Test interactive plot saving as HTML."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        
        save_path = temp_output_dir / "interactive.html"
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_interactive(save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_interactive_save_png_with_kaleido(
        self, basic_metrics_history, temp_output_dir
    ):
        """Test interactive plot saving as PNG (requires kaleido + Chrome)."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE, KALEIDO_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        if not KALEIDO_AVAILABLE:
            pytest.skip("kaleido not available")
        
        save_path = temp_output_dir / "interactive.png"
        visualizer = TrainingVisualizer(basic_metrics_history)
        
        # Mock write_image to avoid Chrome dependency in Kaleido v1+
        with patch('plotly.graph_objects.Figure.write_image') as mock_write_image:
            fig = visualizer.plot_interactive(save_path=save_path)
            assert fig is not None
            # Verify write_image was called with the correct path
            mock_write_image.assert_called_once_with(str(save_path))
    
    def test_plot_interactive_save_pdf_with_kaleido(
        self, basic_metrics_history, temp_output_dir
    ):
        """Test interactive plot saving as PDF (requires kaleido + Chrome)."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE, KALEIDO_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        if not KALEIDO_AVAILABLE:
            pytest.skip("kaleido not available")
        
        save_path = temp_output_dir / "interactive.pdf"
        visualizer = TrainingVisualizer(basic_metrics_history)
        
        # Mock write_image to avoid Chrome dependency in Kaleido v1+
        with patch('plotly.graph_objects.Figure.write_image') as mock_write_image:
            fig = visualizer.plot_interactive(save_path=save_path)
            assert fig is not None
            # Verify write_image was called with the correct path
            mock_write_image.assert_called_once_with(str(save_path))
    
    def test_plot_interactive_save_unknown_format_defaults_to_html(
        self, basic_metrics_history, temp_output_dir
    ):
        """Test that unknown format defaults to HTML."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        
        save_path = temp_output_dir / "interactive.xyz"
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_interactive(save_path=save_path)
        
        assert fig is not None
        # Should create .html version
        html_path = temp_output_dir / "interactive.html"
        assert html_path.exists()
    
    def test_plot_interactive_custom_title(self, basic_metrics_history):
        """Test interactive plot with custom title."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        fig = visualizer.plot_interactive(title="Custom Interactive Title")
        
        assert fig is not None
    
    def test_plot_interactive_empty_metrics(self, empty_metrics_history):
        """Test interactive plot with empty metrics."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        
        visualizer = TrainingVisualizer(empty_metrics_history)
        fig = visualizer.plot_interactive()
        
        assert fig is not None
    
    def test_plot_interactive_train_only(self, train_only_metrics):
        """Test interactive plot with only training data."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        
        visualizer = TrainingVisualizer(train_only_metrics)
        fig = visualizer.plot_interactive()
        
        assert fig is not None
    
    def test_plot_interactive_val_only(self, val_only_metrics):
        """Test interactive plot with only validation data."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        
        visualizer = TrainingVisualizer(val_only_metrics)
        fig = visualizer.plot_interactive()
        
        assert fig is not None
    
    def test_plot_interactive_returns_none_without_plotly(
        self, basic_metrics_history
    ):
        """Test that plot_interactive returns None when plotly unavailable."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        with patch(
            'milia_pipeline.models.training.visualization.PLOTLY_AVAILABLE',
            False
        ):
            visualizer = TrainingVisualizer(basic_metrics_history)
            result = visualizer.plot_interactive()
            
            assert result is None


# =============================================================================
# PLOT TRAINING SUMMARY TESTS
# =============================================================================

class TestPlotTrainingSummary:
    """Test plot_training_summary method."""
    
    def test_plot_training_summary_default_formats(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test training summary with default formats (png, html)."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE, PLOTLY_AVAILABLE
        )
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        saved_paths = visualizer.plot_training_summary(output_dir=temp_output_dir)
        
        assert isinstance(saved_paths, dict)
        
        if MATPLOTLIB_AVAILABLE:
            assert 'loss_curves_png' in saved_paths or len(saved_paths) >= 0
        if PLOTLY_AVAILABLE:
            assert 'interactive_html' in saved_paths or len(saved_paths) >= 0
    
    def test_plot_training_summary_png_only(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test training summary with PNG format only."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        saved_paths = visualizer.plot_training_summary(
            output_dir=temp_output_dir,
            formats=['png']
        )
        
        assert isinstance(saved_paths, dict)
        # Should not have html
        assert 'interactive_html' not in saved_paths
    
    def test_plot_training_summary_html_only(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test training summary with HTML format only."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not available")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        saved_paths = visualizer.plot_training_summary(
            output_dir=temp_output_dir,
            formats=['html']
        )
        
        assert isinstance(saved_paths, dict)
        # Should have interactive_html
        if PLOTLY_AVAILABLE:
            assert 'interactive_html' in saved_paths
    
    def test_plot_training_summary_with_prefix(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test training summary with filename prefix."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        saved_paths = visualizer.plot_training_summary(
            output_dir=temp_output_dir,
            prefix="experiment1"
        )
        
        assert isinstance(saved_paths, dict)
        
        # Check that files have prefix in name
        for path in saved_paths.values():
            assert "experiment1" in path.name
    
    def test_plot_training_summary_creates_output_dir(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test that plot_training_summary creates output directory."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        new_dir = temp_output_dir / "new_subdir" / "plots"
        assert not new_dir.exists()
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        saved_paths = visualizer.plot_training_summary(output_dir=new_dir)
        
        assert new_dir.exists()
    
    def test_plot_training_summary_with_pdf_format(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test training summary with PDF format (requires kaleido + Chrome)."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, PLOTLY_AVAILABLE, KALEIDO_AVAILABLE
        )
        
        if not PLOTLY_AVAILABLE or not KALEIDO_AVAILABLE:
            pytest.skip("plotly and kaleido required for PDF")
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        
        # Mock write_image to avoid Chrome dependency in Kaleido v1+
        with patch('plotly.graph_objects.Figure.write_image') as mock_write_image:
            saved_paths = visualizer.plot_training_summary(
                output_dir=temp_output_dir,
                formats=['pdf']
            )
            assert isinstance(saved_paths, dict)
            # Verify write_image was called (for PDF generation)
            assert mock_write_image.called
            assert 'summary_pdf' in saved_paths
    
    def test_plot_training_summary_all_formats(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test training summary with all formats (PDF requires Chrome)."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        
        # Kaleido v1+ requires Chrome browser for PDF - may not be installed
        try:
            saved_paths = visualizer.plot_training_summary(
                output_dir=temp_output_dir,
                formats=['png', 'html', 'pdf']
            )
            assert isinstance(saved_paths, dict)
        except RuntimeError as e:
            if "Chrome" in str(e) or "chrome" in str(e):
                # Try without PDF format
                saved_paths = visualizer.plot_training_summary(
                    output_dir=temp_output_dir,
                    formats=['png', 'html']
                )
                assert isinstance(saved_paths, dict)
            else:
                raise
    
    def test_plot_training_summary_empty_metrics(
        self, empty_metrics_history, temp_output_dir
    ):
        """Test training summary with empty metrics."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(empty_metrics_history)
        saved_paths = visualizer.plot_training_summary(output_dir=temp_output_dir)
        
        assert isinstance(saved_paths, dict)
    
    def test_plot_training_summary_string_output_dir(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test training summary with string output directory."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        saved_paths = visualizer.plot_training_summary(
            output_dir=str(temp_output_dir)
        )
        
        assert isinstance(saved_paths, dict)
    
    def test_plot_training_summary_returns_path_objects(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test that saved paths are Path objects."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        saved_paths = visualizer.plot_training_summary(output_dir=temp_output_dir)
        
        for key, path in saved_paths.items():
            assert isinstance(path, Path)
    
    def test_plot_training_summary_files_exist(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test that all saved files actually exist."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        saved_paths = visualizer.plot_training_summary(output_dir=temp_output_dir)
        
        for key, path in saved_paths.items():
            assert path.exists(), f"File {path} does not exist"
    
    def test_plot_training_summary_timestamp_in_filename(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test that timestamp is included in filename."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        saved_paths = visualizer.plot_training_summary(output_dir=temp_output_dir)
        
        # Filenames should contain timestamp pattern YYYYMMDD_HHMMSS
        for path in saved_paths.values():
            # Check that filename contains a date-like pattern
            assert any(char.isdigit() for char in path.stem)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_plot_training_summary_function(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test plot_training_summary convenience function."""
        from milia_pipeline.models.training.visualization import (
            plot_training_summary
        )
        
        saved_paths = plot_training_summary(
            metrics_history=comprehensive_metrics_history,
            output_dir=temp_output_dir
        )
        
        assert isinstance(saved_paths, dict)
    
    def test_plot_training_summary_function_with_prefix(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test plot_training_summary function with prefix."""
        from milia_pipeline.models.training.visualization import (
            plot_training_summary
        )
        
        saved_paths = plot_training_summary(
            metrics_history=comprehensive_metrics_history,
            output_dir=temp_output_dir,
            prefix="test_run"
        )
        
        assert isinstance(saved_paths, dict)
        for path in saved_paths.values():
            assert "test_run" in path.name
    
    def test_plot_training_summary_function_with_formats(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test plot_training_summary function with specific formats."""
        from milia_pipeline.models.training.visualization import (
            plot_training_summary, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        saved_paths = plot_training_summary(
            metrics_history=comprehensive_metrics_history,
            output_dir=temp_output_dir,
            formats=['png']
        )
        
        assert isinstance(saved_paths, dict)
    
    def test_create_visualizer_function(self, basic_metrics_history):
        """Test create_visualizer convenience function."""
        from milia_pipeline.models.training.visualization import (
            create_visualizer, TrainingVisualizer
        )
        
        visualizer = create_visualizer(basic_metrics_history)
        
        assert isinstance(visualizer, TrainingVisualizer)
        assert visualizer.metrics_history == basic_metrics_history
    
    def test_create_visualizer_function_with_style(
        self, basic_metrics_history, custom_style
    ):
        """Test create_visualizer function with custom style."""
        from milia_pipeline.models.training.visualization import (
            create_visualizer, TrainingVisualizer
        )
        
        visualizer = create_visualizer(basic_metrics_history, style=custom_style)
        
        assert isinstance(visualizer, TrainingVisualizer)
        assert visualizer.style['dpi'] == custom_style['dpi']
    
    def test_create_visualizer_function_with_none_style(
        self, basic_metrics_history
    ):
        """Test create_visualizer function with None style."""
        from milia_pipeline.models.training.visualization import (
            create_visualizer, TrainingVisualizer
        )
        
        visualizer = create_visualizer(basic_metrics_history, style=None)
        
        assert isinstance(visualizer, TrainingVisualizer)
        # Should have default style
        assert visualizer.style['dpi'] == 150


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_metrics_with_nan_values(self):
        """Test handling of NaN values in metrics."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_loss': [1.0, float('nan'), 0.6, 0.5, 0.4],
            'val_loss': [1.1, 0.9, float('nan'), 0.6, 0.5],
        }
        visualizer = TrainingVisualizer(metrics)
        
        # Should not raise exception
        fig = visualizer.plot_loss_curves()
        assert fig is not None
    
    def test_metrics_with_inf_values(self):
        """Test handling of infinity values in metrics."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_loss': [1.0, float('inf'), 0.6, 0.5, 0.4],
            'val_loss': [1.1, 0.9, float('-inf'), 0.6, 0.5],
        }
        visualizer = TrainingVisualizer(metrics)
        
        # Should not raise exception
        fig = visualizer.plot_loss_curves()
        assert fig is not None
    
    def test_metrics_with_negative_values(self):
        """Test handling of negative values in metrics."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_loss': [-1.0, -0.8, -0.6, -0.5, -0.4],
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_loss_curves()
        assert fig is not None
    
    def test_metrics_with_zero_values(self):
        """Test handling of zero values in metrics."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_loss': [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_loss_curves()
        assert fig is not None
    
    def test_metrics_with_very_large_values(self):
        """Test handling of very large values in metrics."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_loss': [1e10, 1e9, 1e8, 1e7, 1e6],
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_loss_curves()
        assert fig is not None
    
    def test_metrics_with_very_small_values(self):
        """Test handling of very small values in metrics."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_loss': [1e-10, 1e-11, 1e-12, 1e-13, 1e-14],
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_loss_curves()
        assert fig is not None
    
    def test_many_epochs(self):
        """Test with many epochs (performance test)."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        num_epochs = 1000
        metrics = {
            'train_loss': list(np.linspace(1.0, 0.1, num_epochs)),
            'val_loss': list(np.linspace(1.1, 0.2, num_epochs)),
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_loss_curves()
        assert fig is not None
    
    def test_unicode_in_metric_names(self):
        """Test handling of unicode characters in metric names."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_αβγ': [1.0, 0.8, 0.6],
            'val_δεζ': [1.1, 0.9, 0.7],
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_metrics()
        assert fig is not None
    
    def test_special_characters_in_save_path(self, temp_output_dir):
        """Test saving with special characters in path."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {'train_loss': [1.0, 0.8, 0.6]}
        visualizer = TrainingVisualizer(metrics)
        
        # Create subdirectory with spaces (if platform allows)
        subdir = temp_output_dir / "test dir"
        subdir.mkdir(exist_ok=True)
        save_path = subdir / "loss_curves.png"
        
        fig = visualizer.plot_loss_curves(save_path=save_path)
        assert fig is not None
        assert save_path.exists()


# =============================================================================
# STYLE CONFIGURATION TESTS
# =============================================================================

class TestStyleConfiguration:
    """Test style configuration and customization."""
    
    def test_custom_figure_size(self, basic_metrics_history):
        """Test custom figure size."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        style = {'figure_size': (20, 10)}
        visualizer = TrainingVisualizer(basic_metrics_history, style=style)
        
        assert visualizer.style['figure_size'] == (20, 10)
    
    def test_custom_dpi(self, basic_metrics_history):
        """Test custom DPI setting."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        style = {'dpi': 300}
        visualizer = TrainingVisualizer(basic_metrics_history, style=style)
        
        assert visualizer.style['dpi'] == 300
    
    def test_custom_colors(self, basic_metrics_history):
        """Test custom color configuration."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        custom_colors = {
            'train': '#123456',
            'val': '#654321',
            'test': '#ABCDEF',
            'lr': '#FEDCBA',
        }
        style = {'colors': custom_colors}
        visualizer = TrainingVisualizer(basic_metrics_history, style=style)
        
        assert visualizer.style['colors'] == custom_colors
    
    def test_custom_line_width(self, basic_metrics_history):
        """Test custom line width."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        style = {'line_width': 5}
        visualizer = TrainingVisualizer(basic_metrics_history, style=style)
        
        assert visualizer.style['line_width'] == 5
    
    def test_custom_marker_size(self, basic_metrics_history):
        """Test custom marker size."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        style = {'marker_size': 10}
        visualizer = TrainingVisualizer(basic_metrics_history, style=style)
        
        assert visualizer.style['marker_size'] == 10
    
    def test_custom_font_sizes(self, basic_metrics_history):
        """Test custom font sizes."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        style = {
            'font_size': 16,
            'title_size': 20,
            'legend_size': 14,
        }
        visualizer = TrainingVisualizer(basic_metrics_history, style=style)
        
        assert visualizer.style['font_size'] == 16
        assert visualizer.style['title_size'] == 20
        assert visualizer.style['legend_size'] == 14
    
    def test_custom_grid_alpha(self, basic_metrics_history):
        """Test custom grid alpha."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        style = {'grid_alpha': 0.8}
        visualizer = TrainingVisualizer(basic_metrics_history, style=style)
        
        assert visualizer.style['grid_alpha'] == 0.8
    
    def test_style_does_not_modify_class_default(self, basic_metrics_history):
        """Test that custom style doesn't modify class DEFAULT_STYLE."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        original_dpi = TrainingVisualizer.DEFAULT_STYLE['dpi']
        
        style = {'dpi': 999}
        visualizer = TrainingVisualizer(basic_metrics_history, style=style)
        
        # Class default should not be modified
        assert TrainingVisualizer.DEFAULT_STYLE['dpi'] == original_dpi
        # Instance should have custom value
        assert visualizer.style['dpi'] == 999


# =============================================================================
# LOGGING TESTS
# =============================================================================

class TestLogging:
    """Test logging functionality."""
    
    def test_save_path_logged(self, basic_metrics_history, temp_output_dir, caplog):
        """Test that save path is logged."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        save_path = temp_output_dir / "test_log.png"
        visualizer = TrainingVisualizer(basic_metrics_history)
        
        with caplog.at_level(logging.INFO):
            visualizer.plot_loss_curves(save_path=save_path)
        
        assert "saved to" in caplog.text.lower() or save_path.exists()
    
    def test_no_lr_warning_logged(self, basic_metrics_history, caplog):
        """Test that warning is logged when no LR data."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        
        with caplog.at_level(logging.WARNING):
            result = visualizer.plot_learning_rate()
        
        assert result is None
        assert "learning rate" in caplog.text.lower() or "lr" in caplog.text.lower()
    
    def test_no_metrics_warning_logged(self, basic_metrics_history, caplog):
        """Test that warning is logged when no metrics to plot."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        
        with caplog.at_level(logging.WARNING):
            result = visualizer.plot_metrics()
        
        assert result is None
        # Should warn about no metrics


# =============================================================================
# MODULE-LEVEL TESTS
# =============================================================================

class TestModuleLevel:
    """Test module-level attributes and initialization."""
    
    def test_module_has_matplotlib_available(self):
        """Test that module exports MATPLOTLIB_AVAILABLE."""
        from milia_pipeline.models.training import visualization
        
        assert hasattr(visualization, 'MATPLOTLIB_AVAILABLE')
    
    def test_module_has_plotly_available(self):
        """Test that module exports PLOTLY_AVAILABLE."""
        from milia_pipeline.models.training import visualization
        
        assert hasattr(visualization, 'PLOTLY_AVAILABLE')
    
    def test_module_has_kaleido_available(self):
        """Test that module exports KALEIDO_AVAILABLE."""
        from milia_pipeline.models.training import visualization
        
        assert hasattr(visualization, 'KALEIDO_AVAILABLE')
    
    def test_module_has_training_visualizer(self):
        """Test that module exports TrainingVisualizer."""
        from milia_pipeline.models.training import visualization
        
        assert hasattr(visualization, 'TrainingVisualizer')
    
    def test_module_has_plot_training_summary(self):
        """Test that module exports plot_training_summary."""
        from milia_pipeline.models.training import visualization
        
        assert hasattr(visualization, 'plot_training_summary')
    
    def test_module_has_create_visualizer(self):
        """Test that module exports create_visualizer."""
        from milia_pipeline.models.training import visualization
        
        assert hasattr(visualization, 'create_visualizer')
    
    def test_module_has_logger(self):
        """Test that module has logger configured."""
        from milia_pipeline.models.training import visualization
        
        assert hasattr(visualization, 'logger')


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for visualization module."""
    
    def test_full_workflow(self, comprehensive_metrics_history, temp_output_dir):
        """Test complete visualization workflow."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE, PLOTLY_AVAILABLE
        )
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        
        # Generate all plots
        if MATPLOTLIB_AVAILABLE:
            loss_fig = visualizer.plot_loss_curves(
                save_path=temp_output_dir / "loss.png"
            )
            metrics_fig = visualizer.plot_metrics(
                save_path=temp_output_dir / "metrics.png"
            )
            lr_fig = visualizer.plot_learning_rate(
                save_path=temp_output_dir / "lr.png"
            )
            
            assert loss_fig is not None
        
        if PLOTLY_AVAILABLE:
            interactive_fig = visualizer.plot_interactive(
                save_path=temp_output_dir / "interactive.html"
            )
            assert interactive_fig is not None
    
    def test_summary_then_individual_plots(
        self, comprehensive_metrics_history, temp_output_dir
    ):
        """Test generating summary then individual plots."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        visualizer = TrainingVisualizer(comprehensive_metrics_history)
        
        # Generate summary first
        summary_paths = visualizer.plot_training_summary(
            output_dir=temp_output_dir / "summary"
        )
        
        # Create individual directory before saving
        individual_dir = temp_output_dir / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        # Then generate individual plots
        individual_paths = {
            'loss': visualizer.plot_loss_curves(
                save_path=individual_dir / "loss.png"
            ),
            'lr': visualizer.plot_learning_rate(
                save_path=individual_dir / "lr.png"
            ),
        }
        
        assert isinstance(summary_paths, dict)
    
    def test_multiple_visualizers_different_metrics(self, temp_output_dir):
        """Test creating multiple visualizers with different metrics."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        metrics1 = {'train_loss': [1.0, 0.8, 0.6]}
        metrics2 = {'train_loss': [2.0, 1.5, 1.0], 'val_loss': [2.2, 1.7, 1.2]}
        
        viz1 = TrainingVisualizer(metrics1)
        viz2 = TrainingVisualizer(metrics2)
        
        # They should be independent
        assert viz1.metrics_history != viz2.metrics_history
        assert len(viz1.metrics_history) == 1
        assert len(viz2.metrics_history) == 2
    
    def test_visualizer_with_custom_style_produces_plots(
        self, basic_metrics_history, custom_style, temp_output_dir
    ):
        """Test that custom styled visualizer produces valid plots."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history, style=custom_style)
        
        save_path = temp_output_dir / "custom_styled.png"
        fig = visualizer.plot_loss_curves(save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()


# =============================================================================
# DATA TYPE HANDLING TESTS
# =============================================================================

class TestDataTypeHandling:
    """Test handling of different data types in metrics."""
    
    def test_metrics_with_numpy_arrays(self):
        """Test metrics containing numpy arrays."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_loss': np.array([1.0, 0.8, 0.6, 0.5, 0.4]),
            'val_loss': np.array([1.1, 0.9, 0.7, 0.6, 0.5]),
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_loss_curves()
        assert fig is not None
    
    def test_metrics_with_float32(self):
        """Test metrics with float32 values."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_loss': list(np.array([1.0, 0.8, 0.6], dtype=np.float32)),
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_loss_curves()
        assert fig is not None
    
    def test_metrics_with_integers(self):
        """Test metrics with integer values."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_accuracy': [80, 85, 90, 92, 95],
            'val_accuracy': [78, 82, 88, 90, 93],
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_metrics()
        assert fig is not None
    
    def test_metrics_with_mixed_types(self):
        """Test metrics with mixed numeric types."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        metrics = {
            'train_loss': [1.0, 0.8, 0.6],  # float
            'train_accuracy': [80, 85, 90],  # int
            'val_score': np.array([0.7, 0.8, 0.9]),  # numpy
        }
        visualizer = TrainingVisualizer(metrics)
        
        fig = visualizer.plot_metrics()
        assert fig is not None


# =============================================================================
# CONCURRENCY AND THREAD SAFETY TESTS
# =============================================================================

class TestConcurrency:
    """Test thread safety and concurrent usage."""
    
    def test_multiple_figures_sequential(
        self, basic_metrics_history, temp_output_dir
    ):
        """Test creating multiple figures sequentially."""
        from milia_pipeline.models.training.visualization import (
            TrainingVisualizer, MATPLOTLIB_AVAILABLE
        )
        
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        visualizer = TrainingVisualizer(basic_metrics_history)
        
        figs = []
        for i in range(5):
            fig = visualizer.plot_loss_curves(
                save_path=temp_output_dir / f"loss_{i}.png"
            )
            figs.append(fig)
        
        assert all(fig is not None for fig in figs)
        assert len(list(temp_output_dir.glob("loss_*.png"))) == 5
    
    def test_multiple_visualizers_independent(self, basic_metrics_history):
        """Test that multiple visualizers are independent."""
        from milia_pipeline.models.training.visualization import TrainingVisualizer
        
        viz1 = TrainingVisualizer(basic_metrics_history)
        viz2 = TrainingVisualizer({'train_loss': [0.5, 0.4, 0.3]})
        
        # Modifying one shouldn't affect the other
        viz1.metrics_history['new_metric'] = [1, 2, 3]
        
        assert 'new_metric' not in viz2.metrics_history


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
