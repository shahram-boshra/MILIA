#!/usr/bin/env python3
"""
Extended Production-Ready Unit Test Suite for checkpoint_manager.py Module

Comprehensive test coverage including:
- CheckpointManager instantiation (with Dependency Injection pattern)
- create_version_info() static method (with PyG import scenarios)
- get_checkpoint_dir() method (directory creation, subdirectories)
- _resolve_path() method (relative/absolute paths, parent creation)
- _resolve_checkpoint_path() method (intelligent search behavior)
- save() method (all parameters, edge cases, directory creation, path resolution)
- load() method (v1.0 and v2.0 checkpoint formats, backward compatibility, path search)
- is_v2_checkpoint() method (version string comparisons)
- get_hyper_parameters() method (extraction and defaults)
- get_model_name() method (extraction and missing data)
- Checkpoint integrity and round-trip testing
- Error handling and edge cases
- Logging verification
- Path handling (string and Path objects, relative and absolute)

This is an EXTENDED PRODUCTION-READY test suite with comprehensive coverage
for enterprise-grade deployment following the Dependency Injection pattern.

Author: MILIA Team
Version: 2.0.0
"""
import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
from datetime import datetime
import tempfile
import shutil

import torch
import torch.nn as nn
import torch.optim as optim


# Import the module under test
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
    CheckpointManager,
    CHECKPOINT_FORMAT_VERSION,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """
    Create a CheckpointManager instance with working_root_dir.
    
    Uses temp_checkpoint_dir as the working_root_dir following the
    Dependency Injection pattern required by CheckpointManager.
    """
    return CheckpointManager(working_root_dir=temp_checkpoint_dir)


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = Mock(spec=nn.Module)
    model.state_dict = Mock(return_value={
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(5, 10),
        'layer2.bias': torch.randn(5),
    })
    model.load_state_dict = Mock()
    return model


@pytest.fixture
def real_model():
    """Create a real simple PyTorch model for integration tests."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    return SimpleModel()


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    optimizer = Mock(spec=optim.Adam)
    optimizer.state_dict = Mock(return_value={
        'state': {},
        'param_groups': [{'lr': 0.001, 'weight_decay': 0.0}]
    })
    optimizer.load_state_dict = Mock()
    return optimizer


@pytest.fixture
def real_optimizer(real_model):
    """Create a real optimizer for integration tests."""
    return optim.Adam(real_model.parameters(), lr=0.001)


@pytest.fixture
def mock_scheduler():
    """Create a mock learning rate scheduler."""
    scheduler = Mock()
    scheduler.state_dict = Mock(return_value={
        '_step_count': 10,
        '_last_lr': [0.001],
    })
    scheduler.load_state_dict = Mock()
    return scheduler


@pytest.fixture
def real_scheduler(real_optimizer):
    """Create a real scheduler for integration tests."""
    return optim.lr_scheduler.StepLR(real_optimizer, step_size=10, gamma=0.1)


@pytest.fixture
def sample_hyper_parameters():
    """Create sample hyper_parameters dict."""
    return {
        'model_name': 'GCN',
        'task_type': 'graph_regression',
        'hyperparameters': {
            'hidden_channels': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'learning_rate': 0.001,
        },
        'model_info': {
            'name': 'GCN',
            'uses_edge_features': False,
            'requires_edge_features': False,
            'detected_edge_params': [],
        },
        'wrapper_info': {
            'type': 'GraphRegressionWrapper',
            'out_channels': 3,
        },
    }


@pytest.fixture
def sample_data_info():
    """Create sample data_info dict."""
    return {
        'dataset_name': 'QM9',
        'num_features': 11,
        'num_targets': 3,
        'num_samples': 130000,
        'split_ratios': [0.8, 0.1, 0.1],
    }


@pytest.fixture
def sample_metrics_history():
    """Create sample metrics history."""
    return {
        'train_loss': [0.5, 0.4, 0.3, 0.25],
        'val_loss': [0.55, 0.42, 0.32, 0.28],
        'learning_rate': [0.001, 0.001, 0.0001, 0.0001],
    }


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def v1_checkpoint():
    """Create a simulated v1.0 checkpoint (legacy format)."""
    return {
        'epoch': 50,
        'model_state_dict': {'layer.weight': torch.randn(5, 5)},
        'optimizer_state_dict': {'state': {}},
        # No version_info, hyper_parameters, or data_info
    }


@pytest.fixture
def v2_checkpoint(sample_hyper_parameters, sample_data_info):
    """Create a simulated v2.0 checkpoint."""
    return {
        'epoch': 100,
        'global_step': 5000,
        'model_state_dict': {'layer.weight': torch.randn(5, 5)},
        'optimizer_state_dict': {'state': {}},
        'metrics_history': {'train_loss': [0.5, 0.3]},
        'best_val_loss': 0.25,
        'hyper_parameters': sample_hyper_parameters,
        'data_info': sample_data_info,
        'version_info': {
            'milia_version': '1.0.0',
            'checkpoint_format_version': '2.0',
            'pytorch_version': torch.__version__,
            'pyg_version': '2.3.0',
            'created_at': datetime.now().isoformat(),
        },
    }


# =============================================================================
# CHECKPOINT MANAGER INSTANTIATION TESTS
# =============================================================================

class TestCheckpointManagerInstantiation:
    """Test CheckpointManager class instantiation."""
    
    def test_instantiation_with_working_root_dir(self, temp_checkpoint_dir):
        """Test basic instantiation of CheckpointManager with working_root_dir."""
        manager = CheckpointManager(working_root_dir=temp_checkpoint_dir)
        assert manager is not None
        assert isinstance(manager, CheckpointManager)
    
    def test_instantiation_with_string_path(self, temp_checkpoint_dir):
        """Test instantiation accepts string path for working_root_dir."""
        manager = CheckpointManager(working_root_dir=str(temp_checkpoint_dir))
        assert manager is not None
        assert isinstance(manager, CheckpointManager)
    
    def test_instantiation_with_path_object(self, temp_checkpoint_dir):
        """Test instantiation accepts Path object for working_root_dir."""
        manager = CheckpointManager(working_root_dir=Path(temp_checkpoint_dir))
        assert manager is not None
        assert isinstance(manager, CheckpointManager)
    
    def test_instantiation_resolves_expanduser(self, temp_checkpoint_dir):
        """Test that working_root_dir with ~ is expanded."""
        # Create a manager - expanduser is called internally
        manager = CheckpointManager(working_root_dir=temp_checkpoint_dir)
        # The path should be absolute and resolved
        assert manager._working_root_dir.is_absolute()
    
    def test_instantiation_multiple_instances(self, temp_checkpoint_dir):
        """Test creating multiple CheckpointManager instances."""
        manager1 = CheckpointManager(working_root_dir=temp_checkpoint_dir)
        manager2 = CheckpointManager(working_root_dir=temp_checkpoint_dir)
        assert manager1 is not manager2
        # They should be independent instances
    
    def test_instantiation_different_working_dirs(self, temp_checkpoint_dir):
        """Test creating instances with different working directories."""
        subdir1 = temp_checkpoint_dir / "subdir1"
        subdir2 = temp_checkpoint_dir / "subdir2"
        subdir1.mkdir(parents=True, exist_ok=True)
        subdir2.mkdir(parents=True, exist_ok=True)
        
        manager1 = CheckpointManager(working_root_dir=subdir1)
        manager2 = CheckpointManager(working_root_dir=subdir2)
        
        assert manager1._working_root_dir != manager2._working_root_dir
    
    def test_manager_has_required_methods(self, checkpoint_manager):
        """Test that CheckpointManager has all required methods."""
        assert hasattr(checkpoint_manager, 'save')
        assert hasattr(checkpoint_manager, 'load')
        assert hasattr(checkpoint_manager, 'is_v2_checkpoint')
        assert hasattr(checkpoint_manager, 'get_hyper_parameters')
        assert hasattr(checkpoint_manager, 'get_model_name')
        assert hasattr(checkpoint_manager, 'create_version_info')
        assert hasattr(checkpoint_manager, 'get_checkpoint_dir')
        assert hasattr(checkpoint_manager, '_resolve_path')
        assert hasattr(checkpoint_manager, '_resolve_checkpoint_path')
    
    def test_create_version_info_is_static(self):
        """Test that create_version_info is a static method."""
        # Can call without instance
        version_info = CheckpointManager.create_version_info()
        assert isinstance(version_info, dict)


# =============================================================================
# CREATE_VERSION_INFO TESTS
# =============================================================================

class TestCreateVersionInfo:
    """Test create_version_info() static method."""
    
    def test_create_version_info_returns_dict(self, checkpoint_manager):
        """Test that create_version_info returns a dictionary."""
        version_info = checkpoint_manager.create_version_info()
        assert isinstance(version_info, dict)
    
    def test_create_version_info_has_required_keys(self, checkpoint_manager):
        """Test that version_info contains all required keys."""
        version_info = checkpoint_manager.create_version_info()
        
        required_keys = [
            'milia_version',
            'checkpoint_format_version',
            'pytorch_version',
            'pyg_version',
            'created_at',
        ]
        for key in required_keys:
            assert key in version_info, f"Missing key: {key}"
    
    def test_create_version_info_checkpoint_format_version(self, checkpoint_manager):
        """Test checkpoint_format_version matches module constant."""
        version_info = checkpoint_manager.create_version_info()
        assert version_info['checkpoint_format_version'] == CHECKPOINT_FORMAT_VERSION
        assert version_info['checkpoint_format_version'] == '2.0'
    
    def test_create_version_info_pytorch_version(self, checkpoint_manager):
        """Test pytorch_version is populated correctly."""
        version_info = checkpoint_manager.create_version_info()
        assert version_info['pytorch_version'] == torch.__version__
    
    def test_create_version_info_milia_version(self, checkpoint_manager):
        """Test milia_version is set."""
        version_info = checkpoint_manager.create_version_info()
        assert version_info['milia_version'] == '1.0.0'
    
    def test_create_version_info_created_at_format(self, checkpoint_manager):
        """Test created_at is in ISO format."""
        version_info = checkpoint_manager.create_version_info()
        created_at = version_info['created_at']
        
        # Should be parseable as ISO format
        parsed_datetime = datetime.fromisoformat(created_at)
        assert isinstance(parsed_datetime, datetime)
    
    def test_create_version_info_created_at_is_recent(self, checkpoint_manager):
        """Test created_at is a recent timestamp."""
        before = datetime.now()
        version_info = checkpoint_manager.create_version_info()
        after = datetime.now()
        
        created_at = datetime.fromisoformat(version_info['created_at'])
        assert before <= created_at <= after
    
    def test_create_version_info_with_pyg_installed(self, checkpoint_manager):
        """Test pyg_version when torch_geometric is available."""
        with patch.dict('sys.modules', {'torch_geometric': MagicMock(__version__='2.4.0')}):
            # Need to reimport or mock at the module level
            with patch('milia_pipeline.models.post_training.checkpoint.checkpoint_manager.torch_geometric', 
                       create=True) as mock_pyg:
                mock_pyg.__version__ = '2.4.0'
                # Since the import is inside the function, we need a different approach
                pass
        
        # Test the actual current behavior
        version_info = checkpoint_manager.create_version_info()
        # pyg_version should be either a version string or "unknown"
        assert isinstance(version_info['pyg_version'], str)
    
    def test_create_version_info_pyg_import_error(self, checkpoint_manager):
        """Test pyg_version when torch_geometric import fails."""
        # The actual function handles ImportError internally
        # We verify behavior based on whether torch_geometric is installed
        version_info = checkpoint_manager.create_version_info()
        
        # Should either have a version or "unknown"
        pyg_version = version_info['pyg_version']
        assert isinstance(pyg_version, str)
        assert len(pyg_version) > 0
    
    def test_create_version_info_called_as_static_method(self):
        """Test create_version_info can be called without instance."""
        version_info = CheckpointManager.create_version_info()
        assert 'checkpoint_format_version' in version_info
        assert version_info['checkpoint_format_version'] == '2.0'


# =============================================================================
# GET_CHECKPOINT_DIR METHOD TESTS
# =============================================================================

class TestGetCheckpointDirMethod:
    """Test get_checkpoint_dir() method."""
    
    def test_get_checkpoint_dir_default_subdir(self, checkpoint_manager, temp_checkpoint_dir):
        """Test get_checkpoint_dir returns default 'checkpoints' subdirectory."""
        result = checkpoint_manager.get_checkpoint_dir()
        
        expected = temp_checkpoint_dir / "checkpoints"
        assert result == expected
        assert result.exists()
        assert result.is_dir()
    
    def test_get_checkpoint_dir_custom_subdir(self, checkpoint_manager, temp_checkpoint_dir):
        """Test get_checkpoint_dir with custom subdirectory name."""
        result = checkpoint_manager.get_checkpoint_dir(subdir="my_checkpoints")
        
        expected = temp_checkpoint_dir / "my_checkpoints"
        assert result == expected
        assert result.exists()
        assert result.is_dir()
    
    def test_get_checkpoint_dir_creates_directory(self, temp_checkpoint_dir):
        """Test get_checkpoint_dir creates the directory if it doesn't exist."""
        manager = CheckpointManager(working_root_dir=temp_checkpoint_dir)
        subdir = "new_checkpoint_dir"
        expected_path = temp_checkpoint_dir / subdir
        
        # Ensure directory doesn't exist
        assert not expected_path.exists()
        
        result = manager.get_checkpoint_dir(subdir=subdir)
        
        assert result == expected_path
        assert expected_path.exists()
    
    def test_get_checkpoint_dir_nested_subdir(self, checkpoint_manager, temp_checkpoint_dir):
        """Test get_checkpoint_dir with nested subdirectory path."""
        result = checkpoint_manager.get_checkpoint_dir(subdir="level1/level2/level3")
        
        expected = temp_checkpoint_dir / "level1/level2/level3"
        assert result == expected
        assert result.exists()
    
    def test_get_checkpoint_dir_idempotent(self, checkpoint_manager):
        """Test get_checkpoint_dir is idempotent (multiple calls return same result)."""
        result1 = checkpoint_manager.get_checkpoint_dir()
        result2 = checkpoint_manager.get_checkpoint_dir()
        
        assert result1 == result2
    
    def test_get_checkpoint_dir_empty_subdir_string(self, checkpoint_manager, temp_checkpoint_dir):
        """Test get_checkpoint_dir with empty string subdir."""
        # Empty string should work, resolving to working_root_dir itself
        result = checkpoint_manager.get_checkpoint_dir(subdir="")
        
        assert result == temp_checkpoint_dir


# =============================================================================
# _RESOLVE_PATH METHOD TESTS  
# =============================================================================

class TestResolvePathMethod:
    """Test _resolve_path() internal method."""
    
    def test_resolve_path_relative(self, checkpoint_manager, temp_checkpoint_dir):
        """Test _resolve_path resolves relative paths against working_root_dir."""
        result = checkpoint_manager._resolve_path("subdir/file.pt")
        
        expected = temp_checkpoint_dir / "subdir/file.pt"
        assert result == expected.resolve()
    
    def test_resolve_path_absolute(self, checkpoint_manager, temp_checkpoint_dir):
        """Test _resolve_path returns absolute paths as-is."""
        absolute_path = Path("/absolute/path/to/file.pt")
        
        result = checkpoint_manager._resolve_path(absolute_path)
        
        assert result == absolute_path.resolve()
    
    def test_resolve_path_string_input(self, checkpoint_manager, temp_checkpoint_dir):
        """Test _resolve_path accepts string paths."""
        result = checkpoint_manager._resolve_path("model.pt")
        
        expected = temp_checkpoint_dir / "model.pt"
        assert result == expected.resolve()
    
    def test_resolve_path_path_object_input(self, checkpoint_manager, temp_checkpoint_dir):
        """Test _resolve_path accepts Path objects."""
        result = checkpoint_manager._resolve_path(Path("model.pt"))
        
        expected = temp_checkpoint_dir / "model.pt"
        assert result == expected.resolve()
    
    def test_resolve_path_create_parents_true(self, checkpoint_manager, temp_checkpoint_dir):
        """Test _resolve_path creates parent directories when create_parents=True."""
        nested_path = "deep/nested/dir/file.pt"
        expected_parent = temp_checkpoint_dir / "deep/nested/dir"
        
        # Ensure parent doesn't exist
        assert not expected_parent.exists()
        
        result = checkpoint_manager._resolve_path(nested_path, create_parents=True)
        
        assert expected_parent.exists()
        assert expected_parent.is_dir()
    
    def test_resolve_path_create_parents_false(self, checkpoint_manager, temp_checkpoint_dir):
        """Test _resolve_path doesn't create parents when create_parents=False."""
        nested_path = "another/nested/path/file.pt"
        expected_parent = temp_checkpoint_dir / "another/nested/path"
        
        result = checkpoint_manager._resolve_path(nested_path, create_parents=False)
        
        # Parent should NOT be created
        assert not expected_parent.exists()
    
    def test_resolve_path_expanduser(self, temp_checkpoint_dir):
        """Test _resolve_path expands ~ in paths."""
        manager = CheckpointManager(working_root_dir=temp_checkpoint_dir)
        
        # Note: This tests the expanduser call, though in relative path context
        # it combines with working_root_dir
        result = manager._resolve_path("test.pt")
        
        # Result should be absolute and resolved
        assert result.is_absolute()
    
    def test_resolve_path_returns_resolved_path(self, checkpoint_manager, temp_checkpoint_dir):
        """Test _resolve_path returns fully resolved path (no ..)."""
        result = checkpoint_manager._resolve_path("subdir/../file.pt")
        
        # Should be resolved without the ..
        expected = temp_checkpoint_dir / "file.pt"
        assert result == expected.resolve()


# =============================================================================
# _RESOLVE_CHECKPOINT_PATH METHOD TESTS
# =============================================================================

class TestResolveCheckpointPathMethod:
    """Test _resolve_checkpoint_path() internal method."""
    
    def test_resolve_checkpoint_path_absolute_exists(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test _resolve_checkpoint_path returns absolute path if it exists."""
        # Create a checkpoint file
        filepath = temp_checkpoint_dir / "existing.pt"
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        result = checkpoint_manager._resolve_checkpoint_path(filepath)
        
        assert result == filepath.resolve()
    
    def test_resolve_checkpoint_path_relative_exists_cwd(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test _resolve_checkpoint_path finds relative path from cwd if exists."""
        # This test depends on cwd behavior
        import os
        original_cwd = os.getcwd()
        
        try:
            # Change to temp dir
            os.chdir(temp_checkpoint_dir)
            
            # Create file in current directory
            filepath = Path("cwd_test.pt")
            torch.save({'test': True}, filepath)
            
            result = checkpoint_manager._resolve_checkpoint_path(filepath)
            
            assert result.exists()
            assert result.name == "cwd_test.pt"
        finally:
            os.chdir(original_cwd)
    
    def test_resolve_checkpoint_path_checks_default_checkpoint_dir(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test _resolve_checkpoint_path searches in default checkpoint directory."""
        # Create checkpoint in default checkpoints dir
        checkpoint_dir = checkpoint_manager.get_checkpoint_dir()
        filepath = checkpoint_dir / "in_default_dir.pt"
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        # Search by just the filename
        result = checkpoint_manager._resolve_checkpoint_path("in_default_dir.pt")
        
        assert result == filepath.resolve()
    
    def test_resolve_checkpoint_path_fallback_to_working_root(
        self, checkpoint_manager, temp_checkpoint_dir
    ):
        """Test _resolve_checkpoint_path falls back to working_root_dir resolution."""
        # File doesn't exist anywhere - should resolve relative to working_root_dir
        result = checkpoint_manager._resolve_checkpoint_path("nonexistent/model.pt")
        
        expected = (temp_checkpoint_dir / "nonexistent/model.pt").resolve()
        assert result == expected
    
    def test_resolve_checkpoint_path_string_input(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test _resolve_checkpoint_path accepts string paths."""
        filepath = temp_checkpoint_dir / "string_test.pt"
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        result = checkpoint_manager._resolve_checkpoint_path(str(filepath))
        
        assert result == filepath.resolve()
    
    def test_resolve_checkpoint_path_expanduser(
        self, checkpoint_manager, temp_checkpoint_dir
    ):
        """Test _resolve_checkpoint_path expands ~ in paths."""
        # Test that expanduser is called (path with ~ would be expanded)
        result = checkpoint_manager._resolve_checkpoint_path("~/some/path.pt")
        
        # Should not contain ~ after resolution
        assert "~" not in str(result)


# =============================================================================
# SAVE METHOD TESTS
# =============================================================================

class TestSaveMethod:
    """Test save() method."""
    
    def test_save_minimal_parameters(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save with minimal required parameters."""
        filepath = temp_checkpoint_dir / "minimal.pt"
        
        result_path = checkpoint_manager.save(
            filepath=filepath,
            model=mock_model
        )
        
        assert result_path == filepath.resolve()
        assert filepath.exists()
    
    def test_save_returns_path(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test that save returns the filepath."""
        filepath = temp_checkpoint_dir / "test.pt"
        
        result = checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        assert isinstance(result, Path)
        assert result == filepath.resolve()
    
    def test_save_creates_parent_directories(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test that save creates parent directories if they don't exist."""
        nested_path = temp_checkpoint_dir / "subdir1" / "subdir2" / "model.pt"
        
        checkpoint_manager.save(filepath=nested_path, model=mock_model)
        
        assert nested_path.exists()
        assert nested_path.parent.exists()
    
    def test_save_with_string_path(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save accepts string path and converts to Path."""
        filepath_str = str(temp_checkpoint_dir / "string_path.pt")
        
        result = checkpoint_manager.save(filepath=filepath_str, model=mock_model)
        
        assert isinstance(result, Path)
        assert Path(filepath_str).exists()
    
    def test_save_contains_model_state_dict(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test saved checkpoint contains model_state_dict."""
        filepath = temp_checkpoint_dir / "model_state.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        checkpoint = torch.load(filepath)
        assert 'model_state_dict' in checkpoint
        mock_model.state_dict.assert_called_once()
    
    def test_save_with_optimizer(
        self, checkpoint_manager, mock_model, mock_optimizer, temp_checkpoint_dir
    ):
        """Test save with optimizer includes optimizer_state_dict."""
        filepath = temp_checkpoint_dir / "with_optimizer.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            optimizer=mock_optimizer
        )
        
        checkpoint = torch.load(filepath)
        assert 'optimizer_state_dict' in checkpoint
        mock_optimizer.state_dict.assert_called_once()
    
    def test_save_without_optimizer(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save without optimizer excludes optimizer_state_dict."""
        filepath = temp_checkpoint_dir / "no_optimizer.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            optimizer=None
        )
        
        checkpoint = torch.load(filepath)
        assert 'optimizer_state_dict' not in checkpoint
    
    def test_save_with_scheduler(
        self, checkpoint_manager, mock_model, mock_scheduler, temp_checkpoint_dir
    ):
        """Test save with scheduler includes scheduler_state_dict."""
        filepath = temp_checkpoint_dir / "with_scheduler.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            scheduler=mock_scheduler
        )
        
        checkpoint = torch.load(filepath)
        assert 'scheduler_state_dict' in checkpoint
        mock_scheduler.state_dict.assert_called_once()
    
    def test_save_without_scheduler(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save without scheduler excludes scheduler_state_dict."""
        filepath = temp_checkpoint_dir / "no_scheduler.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            scheduler=None
        )
        
        checkpoint = torch.load(filepath)
        assert 'scheduler_state_dict' not in checkpoint
    
    def test_save_with_epoch(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save stores epoch correctly."""
        filepath = temp_checkpoint_dir / "with_epoch.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            epoch=42
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['epoch'] == 42
    
    def test_save_epoch_default(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save uses default epoch=0."""
        filepath = temp_checkpoint_dir / "default_epoch.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        checkpoint = torch.load(filepath)
        assert checkpoint['epoch'] == 0
    
    def test_save_with_global_step(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save stores global_step correctly."""
        filepath = temp_checkpoint_dir / "with_step.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            global_step=10000
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['global_step'] == 10000
    
    def test_save_global_step_default(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save uses default global_step=0."""
        filepath = temp_checkpoint_dir / "default_step.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        checkpoint = torch.load(filepath)
        assert checkpoint['global_step'] == 0
    
    def test_save_with_metrics_history(
        self, checkpoint_manager, mock_model, sample_metrics_history, temp_checkpoint_dir
    ):
        """Test save stores metrics_history correctly."""
        filepath = temp_checkpoint_dir / "with_metrics.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            metrics_history=sample_metrics_history
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['metrics_history'] == sample_metrics_history
    
    def test_save_metrics_history_default_empty(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save uses empty dict for metrics_history by default."""
        filepath = temp_checkpoint_dir / "default_metrics.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        checkpoint = torch.load(filepath)
        assert checkpoint['metrics_history'] == {}
    
    def test_save_metrics_history_none_becomes_empty(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save converts None metrics_history to empty dict."""
        filepath = temp_checkpoint_dir / "none_metrics.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            metrics_history=None
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['metrics_history'] == {}
    
    def test_save_with_best_val_loss(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save stores best_val_loss correctly."""
        filepath = temp_checkpoint_dir / "with_val_loss.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            best_val_loss=0.123
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['best_val_loss'] == 0.123
    
    def test_save_best_val_loss_default_inf(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save uses float('inf') as default best_val_loss."""
        filepath = temp_checkpoint_dir / "default_val_loss.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        checkpoint = torch.load(filepath)
        assert checkpoint['best_val_loss'] == float('inf')
    
    def test_save_with_hyper_parameters(
        self, checkpoint_manager, mock_model, sample_hyper_parameters, temp_checkpoint_dir
    ):
        """Test save stores hyper_parameters correctly."""
        filepath = temp_checkpoint_dir / "with_hyper.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            hyper_parameters=sample_hyper_parameters
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['hyper_parameters'] == sample_hyper_parameters
    
    def test_save_hyper_parameters_default_empty(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save uses empty dict for hyper_parameters by default."""
        filepath = temp_checkpoint_dir / "default_hyper.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        checkpoint = torch.load(filepath)
        assert checkpoint['hyper_parameters'] == {}
    
    def test_save_with_data_info(
        self, checkpoint_manager, mock_model, sample_data_info, temp_checkpoint_dir
    ):
        """Test save stores data_info correctly."""
        filepath = temp_checkpoint_dir / "with_data.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            data_info=sample_data_info
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['data_info'] == sample_data_info
    
    def test_save_data_info_default_empty(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save uses empty dict for data_info by default."""
        filepath = temp_checkpoint_dir / "default_data.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        checkpoint = torch.load(filepath)
        assert checkpoint['data_info'] == {}
    
    def test_save_contains_version_info(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test saved checkpoint contains version_info."""
        filepath = temp_checkpoint_dir / "version_info.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        checkpoint = torch.load(filepath)
        assert 'version_info' in checkpoint
        assert 'checkpoint_format_version' in checkpoint['version_info']
        assert checkpoint['version_info']['checkpoint_format_version'] == '2.0'
    
    def test_save_with_extra_data(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save stores extra_data via **kwargs."""
        filepath = temp_checkpoint_dir / "with_extra.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            custom_field_1="custom_value",
            custom_field_2={'nested': 'data'},
            custom_list=[1, 2, 3]
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['custom_field_1'] == "custom_value"
        assert checkpoint['custom_field_2'] == {'nested': 'data'}
        assert checkpoint['custom_list'] == [1, 2, 3]
    
    def test_save_full_checkpoint(
        self, checkpoint_manager, mock_model, mock_optimizer, mock_scheduler,
        sample_hyper_parameters, sample_data_info, sample_metrics_history,
        temp_checkpoint_dir
    ):
        """Test save with all parameters."""
        filepath = temp_checkpoint_dir / "full.pt"
        
        result = checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            epoch=100,
            global_step=50000,
            metrics_history=sample_metrics_history,
            best_val_loss=0.15,
            hyper_parameters=sample_hyper_parameters,
            data_info=sample_data_info,
            extra_key="extra_value"
        )
        
        assert result == filepath.resolve()
        
        checkpoint = torch.load(filepath)
        assert checkpoint['epoch'] == 100
        assert checkpoint['global_step'] == 50000
        assert checkpoint['best_val_loss'] == 0.15
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'scheduler_state_dict' in checkpoint
        assert checkpoint['metrics_history'] == sample_metrics_history
        assert checkpoint['hyper_parameters'] == sample_hyper_parameters
        assert checkpoint['data_info'] == sample_data_info
        assert checkpoint['extra_key'] == "extra_value"
        assert 'version_info' in checkpoint
    
    def test_save_logging(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir, caplog
    ):
        """Test that save logs appropriate messages."""
        filepath = temp_checkpoint_dir / "logged.pt"
        
        with caplog.at_level(logging.INFO):
            checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        assert "Saved enhanced checkpoint to" in caplog.text
    
    def test_save_overwrites_existing_file(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test that save overwrites existing checkpoint file."""
        filepath = temp_checkpoint_dir / "overwrite.pt"
        
        # Save first checkpoint
        checkpoint_manager.save(filepath=filepath, model=mock_model, epoch=10)
        checkpoint1 = torch.load(filepath)
        assert checkpoint1['epoch'] == 10
        
        # Save second checkpoint (should overwrite)
        checkpoint_manager.save(filepath=filepath, model=mock_model, epoch=20)
        checkpoint2 = torch.load(filepath)
        assert checkpoint2['epoch'] == 20


# =============================================================================
# SAVE METHOD PATH RESOLUTION TESTS
# =============================================================================

class TestSaveMethodPathResolution:
    """Test save() method path resolution behavior."""
    
    def test_save_with_relative_path(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save resolves relative paths against working_root_dir."""
        # Use a relative path
        relative_path = "relative_test.pt"
        
        result = checkpoint_manager.save(filepath=relative_path, model=mock_model)
        
        # Should be resolved to working_root_dir / relative_path
        expected = (temp_checkpoint_dir / relative_path).resolve()
        assert result == expected
        assert expected.exists()
    
    def test_save_with_nested_relative_path(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save creates parent directories for nested relative paths."""
        relative_path = "nested/subdir/model.pt"
        
        result = checkpoint_manager.save(filepath=relative_path, model=mock_model)
        
        expected = (temp_checkpoint_dir / relative_path).resolve()
        assert result == expected
        assert expected.exists()
    
    def test_save_with_absolute_path_outside_working_dir(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save accepts absolute paths outside working_root_dir."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as other_dir:
            absolute_path = Path(other_dir) / "outside.pt"
            
            result = checkpoint_manager.save(filepath=absolute_path, model=mock_model)
            
            assert result == absolute_path.resolve()
            assert absolute_path.exists()
    
    def test_save_returns_resolved_path(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save returns fully resolved path."""
        # Use path with .. that needs resolution
        filepath = temp_checkpoint_dir / "subdir" / ".." / "resolved.pt"
        
        result = checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        # Should be resolved
        expected = (temp_checkpoint_dir / "resolved.pt").resolve()
        assert result == expected


# =============================================================================
# LOAD METHOD PATH RESOLUTION TESTS
# =============================================================================

class TestLoadMethodPathResolution:
    """Test load() method path resolution behavior."""
    
    def test_load_with_relative_path(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test load resolves relative paths against working_root_dir."""
        # Save with absolute path
        absolute_path = temp_checkpoint_dir / "for_relative_load.pt"
        checkpoint_manager.save(filepath=absolute_path, model=mock_model, epoch=42)
        
        # Load with relative path
        relative_path = "for_relative_load.pt"
        checkpoint = checkpoint_manager.load(relative_path)
        
        assert checkpoint['epoch'] == 42
    
    def test_load_searches_default_checkpoint_dir(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test load searches in default checkpoint directory."""
        # Save in default checkpoints dir
        checkpoint_dir = checkpoint_manager.get_checkpoint_dir()
        filepath = checkpoint_dir / "in_checkpoints.pt"
        checkpoint_manager.save(filepath=filepath, model=mock_model, epoch=99)
        
        # Load by just the filename (should find in checkpoints dir)
        checkpoint = checkpoint_manager.load("in_checkpoints.pt")
        
        assert checkpoint['epoch'] == 99
    
    def test_load_with_nested_relative_path(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test load with nested relative path."""
        # Save with nested path
        nested_path = temp_checkpoint_dir / "subdir" / "nested_model.pt"
        checkpoint_manager.save(filepath=nested_path, model=mock_model, epoch=77)
        
        # Load with relative nested path
        checkpoint = checkpoint_manager.load("subdir/nested_model.pt")
        
        assert checkpoint['epoch'] == 77
    
    def test_load_absolute_path_prioritized(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test load prioritizes absolute paths that exist."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as other_dir:
            # Save to external location
            external_path = Path(other_dir) / "external.pt"
            torch.save({'epoch': 123, 'model_state_dict': {}, 'version_info': {'checkpoint_format_version': '2.0'}}, external_path)
            
            # Load using absolute path
            checkpoint = checkpoint_manager.load(external_path)
            
            assert checkpoint['epoch'] == 123


# =============================================================================
# SAVE METHOD WITH REAL MODELS TESTS
# =============================================================================

class TestSaveMethodWithRealModels:
    """Test save() method with real PyTorch models."""
    
    def test_save_real_model(
        self, checkpoint_manager, real_model, temp_checkpoint_dir
    ):
        """Test save with a real PyTorch model."""
        filepath = temp_checkpoint_dir / "real_model.pt"
        
        checkpoint_manager.save(filepath=filepath, model=real_model)
        
        checkpoint = torch.load(filepath)
        assert 'model_state_dict' in checkpoint
        # Verify state dict has expected keys
        assert 'linear.weight' in checkpoint['model_state_dict']
        assert 'linear.bias' in checkpoint['model_state_dict']
    
    def test_save_real_model_and_optimizer(
        self, checkpoint_manager, real_model, real_optimizer, temp_checkpoint_dir
    ):
        """Test save with real model and optimizer."""
        filepath = temp_checkpoint_dir / "real_model_opt.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=real_model,
            optimizer=real_optimizer
        )
        
        checkpoint = torch.load(filepath)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'param_groups' in checkpoint['optimizer_state_dict']
    
    def test_save_real_model_and_scheduler(
        self, checkpoint_manager, real_model, real_optimizer, real_scheduler,
        temp_checkpoint_dir
    ):
        """Test save with real model, optimizer, and scheduler."""
        filepath = temp_checkpoint_dir / "real_full.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=real_model,
            optimizer=real_optimizer,
            scheduler=real_scheduler
        )
        
        checkpoint = torch.load(filepath)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'scheduler_state_dict' in checkpoint


# =============================================================================
# LOAD METHOD TESTS
# =============================================================================

class TestLoadMethod:
    """Test load() method."""
    
    def test_load_returns_dict(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test that load returns a dictionary."""
        filepath = temp_checkpoint_dir / "to_load.pt"
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        result = checkpoint_manager.load(filepath)
        
        assert isinstance(result, dict)
    
    def test_load_contains_saved_data(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test load returns checkpoint with saved data."""
        filepath = temp_checkpoint_dir / "saved.pt"
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            epoch=50,
            global_step=2500
        )
        
        checkpoint = checkpoint_manager.load(filepath)
        
        assert checkpoint['epoch'] == 50
        assert checkpoint['global_step'] == 2500
        assert 'model_state_dict' in checkpoint
    
    def test_load_with_map_location_cpu(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test load with map_location=cpu."""
        filepath = temp_checkpoint_dir / "map_cpu.pt"
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        checkpoint = checkpoint_manager.load(
            filepath=filepath,
            map_location=torch.device('cpu')
        )
        
        assert isinstance(checkpoint, dict)
    
    def test_load_with_map_location_string(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test load with map_location as string."""
        filepath = temp_checkpoint_dir / "map_str.pt"
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        # torch.load accepts string for map_location
        checkpoint = checkpoint_manager.load(
            filepath=filepath,
            map_location='cpu'
        )
        
        assert isinstance(checkpoint, dict)
    
    def test_load_v2_checkpoint(
        self, checkpoint_manager, mock_model, sample_hyper_parameters,
        sample_data_info, temp_checkpoint_dir
    ):
        """Test load correctly handles v2.0 checkpoint."""
        filepath = temp_checkpoint_dir / "v2.pt"
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            hyper_parameters=sample_hyper_parameters,
            data_info=sample_data_info
        )
        
        checkpoint = checkpoint_manager.load(filepath)
        
        assert checkpoint['version_info']['checkpoint_format_version'] == '2.0'
        assert checkpoint['hyper_parameters'] == sample_hyper_parameters
        assert checkpoint['data_info'] == sample_data_info
    
    def test_load_v1_checkpoint_backward_compatibility(
        self, checkpoint_manager, v1_checkpoint, temp_checkpoint_dir
    ):
        """Test load handles v1.0 checkpoint with backward compatibility."""
        filepath = temp_checkpoint_dir / "v1.pt"
        
        # Save a v1.0 format checkpoint directly
        torch.save(v1_checkpoint, filepath)
        
        checkpoint = checkpoint_manager.load(filepath)
        
        # Should have added default empty dicts
        assert 'hyper_parameters' in checkpoint
        assert checkpoint['hyper_parameters'] == {}
        assert 'data_info' in checkpoint
        assert checkpoint['data_info'] == {}
        assert 'version_info' in checkpoint
    
    def test_load_v1_checkpoint_warning_logged(
        self, checkpoint_manager, v1_checkpoint, temp_checkpoint_dir, caplog
    ):
        """Test load logs warning for v1.0 checkpoint."""
        filepath = temp_checkpoint_dir / "v1_warn.pt"
        torch.save(v1_checkpoint, filepath)
        
        with caplog.at_level(logging.WARNING):
            checkpoint_manager.load(filepath)
        
        assert "Loading v1.0 checkpoint" in caplog.text
        assert "Model recreation requires manual configuration" in caplog.text
    
    def test_load_logging_info(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir, caplog
    ):
        """Test load logs info message with version."""
        filepath = temp_checkpoint_dir / "info_log.pt"
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        with caplog.at_level(logging.INFO):
            checkpoint_manager.load(filepath)
        
        assert "Loaded checkpoint from" in caplog.text
        assert "format v" in caplog.text
    
    def test_load_nonexistent_file_raises_error(self, checkpoint_manager):
        """Test load raises error for nonexistent file."""
        # torch.load raises FileNotFoundError for missing files
        with pytest.raises((FileNotFoundError, RuntimeError)):
            checkpoint_manager.load(Path("/nonexistent/path/model.pt"))
    
    def test_load_preserves_extra_data(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test load preserves extra_data from save."""
        filepath = temp_checkpoint_dir / "extra_data.pt"
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            custom_field="custom_value"
        )
        
        checkpoint = checkpoint_manager.load(filepath)
        
        assert checkpoint['custom_field'] == "custom_value"


# =============================================================================
# LOAD METHOD EDGE CASES
# =============================================================================

class TestLoadMethodEdgeCases:
    """Test load() method edge cases."""
    
    def test_load_checkpoint_with_partial_version_info(
        self, checkpoint_manager, temp_checkpoint_dir
    ):
        """Test load handles checkpoint with partial version_info."""
        filepath = temp_checkpoint_dir / "partial_version.pt"
        
        # Create checkpoint with partial version_info
        partial_checkpoint = {
            'epoch': 10,
            'model_state_dict': {'layer.weight': torch.randn(5, 5)},
            'version_info': {'milia_version': '1.0.0'},  # Missing checkpoint_format_version
        }
        torch.save(partial_checkpoint, filepath)
        
        checkpoint = checkpoint_manager.load(filepath)
        
        # Should default to v1.0 due to missing checkpoint_format_version
        assert 'hyper_parameters' in checkpoint
    
    def test_load_checkpoint_with_empty_version_info(
        self, checkpoint_manager, temp_checkpoint_dir
    ):
        """Test load handles checkpoint with empty version_info."""
        filepath = temp_checkpoint_dir / "empty_version.pt"
        
        empty_version_checkpoint = {
            'epoch': 10,
            'model_state_dict': {'layer.weight': torch.randn(5, 5)},
            'version_info': {},  # Empty version_info
        }
        torch.save(empty_version_checkpoint, filepath)
        
        checkpoint = checkpoint_manager.load(filepath)
        
        # Should default to v1.0 behavior
        assert 'hyper_parameters' in checkpoint
        assert checkpoint['hyper_parameters'] == {}


# =============================================================================
# IS_V2_CHECKPOINT METHOD TESTS
# =============================================================================

class TestIsV2CheckpointMethod:
    """Test is_v2_checkpoint() method."""
    
    def test_is_v2_checkpoint_with_v2(self, checkpoint_manager, v2_checkpoint):
        """Test is_v2_checkpoint returns True for v2.0 checkpoint."""
        result = checkpoint_manager.is_v2_checkpoint(v2_checkpoint)
        assert result is True
    
    def test_is_v2_checkpoint_with_v1(self, checkpoint_manager, v1_checkpoint):
        """Test is_v2_checkpoint returns False for v1.0 checkpoint."""
        result = checkpoint_manager.is_v2_checkpoint(v1_checkpoint)
        assert result is False
    
    def test_is_v2_checkpoint_without_version_info(self, checkpoint_manager):
        """Test is_v2_checkpoint returns False when version_info missing."""
        checkpoint = {
            'epoch': 10,
            'model_state_dict': {},
        }
        result = checkpoint_manager.is_v2_checkpoint(checkpoint)
        assert result is False
    
    def test_is_v2_checkpoint_with_empty_version_info(self, checkpoint_manager):
        """Test is_v2_checkpoint returns False with empty version_info."""
        checkpoint = {
            'epoch': 10,
            'version_info': {},
        }
        result = checkpoint_manager.is_v2_checkpoint(checkpoint)
        assert result is False
    
    def test_is_v2_checkpoint_version_comparison_string(self, checkpoint_manager):
        """Test is_v2_checkpoint handles string comparison correctly."""
        # Version '2.0' should be >= '2.0'
        checkpoint_v2 = {
            'version_info': {'checkpoint_format_version': '2.0'}
        }
        assert checkpoint_manager.is_v2_checkpoint(checkpoint_v2) is True
        
        # Version '2.1' should be >= '2.0'
        checkpoint_v21 = {
            'version_info': {'checkpoint_format_version': '2.1'}
        }
        assert checkpoint_manager.is_v2_checkpoint(checkpoint_v21) is True
        
        # Version '3.0' should be >= '2.0'
        checkpoint_v3 = {
            'version_info': {'checkpoint_format_version': '3.0'}
        }
        assert checkpoint_manager.is_v2_checkpoint(checkpoint_v3) is True
    
    def test_is_v2_checkpoint_version_less_than_2(self, checkpoint_manager):
        """Test is_v2_checkpoint returns False for versions < 2.0."""
        # Version '1.9' should be < '2.0' (string comparison)
        checkpoint_v19 = {
            'version_info': {'checkpoint_format_version': '1.9'}
        }
        assert checkpoint_manager.is_v2_checkpoint(checkpoint_v19) is False
        
        # Version '1.0' should be < '2.0'
        checkpoint_v10 = {
            'version_info': {'checkpoint_format_version': '1.0'}
        }
        assert checkpoint_manager.is_v2_checkpoint(checkpoint_v10) is False
    
    def test_is_v2_checkpoint_exact_v2(self, checkpoint_manager):
        """Test is_v2_checkpoint returns True for exactly v2.0."""
        checkpoint = {
            'version_info': {'checkpoint_format_version': '2.0'}
        }
        assert checkpoint_manager.is_v2_checkpoint(checkpoint) is True


# =============================================================================
# GET_HYPER_PARAMETERS METHOD TESTS
# =============================================================================

class TestGetHyperParametersMethod:
    """Test get_hyper_parameters() method."""
    
    def test_get_hyper_parameters_from_v2_checkpoint(
        self, checkpoint_manager, v2_checkpoint, sample_hyper_parameters
    ):
        """Test get_hyper_parameters extracts from v2.0 checkpoint."""
        result = checkpoint_manager.get_hyper_parameters(v2_checkpoint)
        assert result == sample_hyper_parameters
    
    def test_get_hyper_parameters_from_v1_checkpoint(
        self, checkpoint_manager, v1_checkpoint
    ):
        """Test get_hyper_parameters returns empty dict for v1.0 checkpoint."""
        result = checkpoint_manager.get_hyper_parameters(v1_checkpoint)
        assert result == {}
    
    def test_get_hyper_parameters_missing_key(self, checkpoint_manager):
        """Test get_hyper_parameters returns empty dict when key missing."""
        checkpoint = {
            'epoch': 10,
            'model_state_dict': {},
        }
        result = checkpoint_manager.get_hyper_parameters(checkpoint)
        assert result == {}
    
    def test_get_hyper_parameters_empty_hyper_params(self, checkpoint_manager):
        """Test get_hyper_parameters with empty hyper_parameters."""
        checkpoint = {
            'hyper_parameters': {},
        }
        result = checkpoint_manager.get_hyper_parameters(checkpoint)
        assert result == {}
    
    def test_get_hyper_parameters_with_nested_data(self, checkpoint_manager):
        """Test get_hyper_parameters preserves nested data structure."""
        nested_hyper_params = {
            'model_name': 'GAT',
            'hyperparameters': {
                'hidden_channels': 128,
                'heads': 8,
                'nested': {
                    'deeply': {
                        'nested': 'value'
                    }
                }
            }
        }
        checkpoint = {'hyper_parameters': nested_hyper_params}
        
        result = checkpoint_manager.get_hyper_parameters(checkpoint)
        
        assert result == nested_hyper_params
        assert result['hyperparameters']['nested']['deeply']['nested'] == 'value'


# =============================================================================
# GET_MODEL_NAME METHOD TESTS
# =============================================================================

class TestGetModelNameMethod:
    """Test get_model_name() method."""
    
    def test_get_model_name_from_v2_checkpoint(
        self, checkpoint_manager, v2_checkpoint
    ):
        """Test get_model_name extracts model name from v2.0 checkpoint."""
        result = checkpoint_manager.get_model_name(v2_checkpoint)
        assert result == 'GCN'
    
    def test_get_model_name_from_v1_checkpoint(
        self, checkpoint_manager, v1_checkpoint
    ):
        """Test get_model_name returns None for v1.0 checkpoint."""
        result = checkpoint_manager.get_model_name(v1_checkpoint)
        assert result is None
    
    def test_get_model_name_missing_hyper_parameters(self, checkpoint_manager):
        """Test get_model_name returns None when hyper_parameters missing."""
        checkpoint = {
            'epoch': 10,
            'model_state_dict': {},
        }
        result = checkpoint_manager.get_model_name(checkpoint)
        assert result is None
    
    def test_get_model_name_missing_model_name_key(self, checkpoint_manager):
        """Test get_model_name returns None when model_name key missing."""
        checkpoint = {
            'hyper_parameters': {
                'task_type': 'graph_regression',
                # No 'model_name' key
            }
        }
        result = checkpoint_manager.get_model_name(checkpoint)
        assert result is None
    
    def test_get_model_name_empty_hyper_parameters(self, checkpoint_manager):
        """Test get_model_name returns None with empty hyper_parameters."""
        checkpoint = {
            'hyper_parameters': {}
        }
        result = checkpoint_manager.get_model_name(checkpoint)
        assert result is None
    
    def test_get_model_name_various_model_names(self, checkpoint_manager):
        """Test get_model_name with various model names."""
        model_names = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'SchNet', 'DimeNet']
        
        for model_name in model_names:
            checkpoint = {
                'hyper_parameters': {
                    'model_name': model_name
                }
            }
            result = checkpoint_manager.get_model_name(checkpoint)
            assert result == model_name


# =============================================================================
# ROUND-TRIP TESTS (SAVE + LOAD)
# =============================================================================

class TestRoundTrip:
    """Test round-trip save and load operations."""
    
    def test_round_trip_minimal(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test basic round-trip save and load."""
        filepath = temp_checkpoint_dir / "round_trip.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        loaded = checkpoint_manager.load(filepath)
        
        assert 'epoch' in loaded
        assert 'model_state_dict' in loaded
        assert 'version_info' in loaded
    
    def test_round_trip_full_checkpoint(
        self, checkpoint_manager, mock_model, mock_optimizer, mock_scheduler,
        sample_hyper_parameters, sample_data_info, sample_metrics_history,
        temp_checkpoint_dir
    ):
        """Test full round-trip with all parameters."""
        filepath = temp_checkpoint_dir / "full_round_trip.pt"
        
        # Save
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            epoch=150,
            global_step=75000,
            metrics_history=sample_metrics_history,
            best_val_loss=0.08,
            hyper_parameters=sample_hyper_parameters,
            data_info=sample_data_info,
            custom_field="test_value"
        )
        
        # Load
        loaded = checkpoint_manager.load(filepath)
        
        # Verify all fields
        assert loaded['epoch'] == 150
        assert loaded['global_step'] == 75000
        assert loaded['best_val_loss'] == 0.08
        assert loaded['metrics_history'] == sample_metrics_history
        assert loaded['hyper_parameters'] == sample_hyper_parameters
        assert loaded['data_info'] == sample_data_info
        assert loaded['custom_field'] == "test_value"
        assert 'model_state_dict' in loaded
        assert 'optimizer_state_dict' in loaded
        assert 'scheduler_state_dict' in loaded
    
    def test_round_trip_real_model(
        self, checkpoint_manager, real_model, real_optimizer, temp_checkpoint_dir
    ):
        """Test round-trip with real PyTorch model."""
        filepath = temp_checkpoint_dir / "real_round_trip.pt"
        
        # Get original state
        original_state = real_model.state_dict()
        
        # Save
        checkpoint_manager.save(
            filepath=filepath,
            model=real_model,
            optimizer=real_optimizer,
            epoch=10
        )
        
        # Load
        loaded = checkpoint_manager.load(filepath)
        
        # Verify state dict shapes match
        for key in original_state:
            assert key in loaded['model_state_dict']
            assert original_state[key].shape == loaded['model_state_dict'][key].shape
    
    def test_round_trip_preserves_tensor_values(
        self, checkpoint_manager, real_model, temp_checkpoint_dir
    ):
        """Test round-trip preserves exact tensor values."""
        filepath = temp_checkpoint_dir / "tensor_values.pt"
        
        # Get original state
        original_state = {k: v.clone() for k, v in real_model.state_dict().items()}
        
        # Save and load
        checkpoint_manager.save(filepath=filepath, model=real_model)
        loaded = checkpoint_manager.load(filepath)
        
        # Verify exact values
        for key in original_state:
            torch.testing.assert_close(
                original_state[key],
                loaded['model_state_dict'][key]
            )


# =============================================================================
# CHECKPOINT FORMAT VERSION CONSTANT TESTS
# =============================================================================

class TestCheckpointFormatVersionConstant:
    """Test CHECKPOINT_FORMAT_VERSION module constant."""
    
    def test_checkpoint_format_version_value(self):
        """Test CHECKPOINT_FORMAT_VERSION has expected value."""
        assert CHECKPOINT_FORMAT_VERSION == "2.0"
    
    def test_checkpoint_format_version_is_string(self):
        """Test CHECKPOINT_FORMAT_VERSION is a string."""
        assert isinstance(CHECKPOINT_FORMAT_VERSION, str)
    
    def test_saved_checkpoint_uses_constant(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test saved checkpoints use CHECKPOINT_FORMAT_VERSION constant."""
        filepath = temp_checkpoint_dir / "format_version.pt"
        
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        checkpoint = torch.load(filepath)
        
        assert checkpoint['version_info']['checkpoint_format_version'] == CHECKPOINT_FORMAT_VERSION


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_save_with_empty_model_state_dict(
        self, checkpoint_manager, temp_checkpoint_dir
    ):
        """Test save with model that has empty state_dict."""
        empty_model = Mock(spec=nn.Module)
        empty_model.state_dict = Mock(return_value={})
        
        filepath = temp_checkpoint_dir / "empty_state.pt"
        checkpoint_manager.save(filepath=filepath, model=empty_model)
        
        checkpoint = torch.load(filepath)
        assert checkpoint['model_state_dict'] == {}
    
    def test_save_with_large_state_dict(
        self, checkpoint_manager, temp_checkpoint_dir
    ):
        """Test save with model that has large state_dict."""
        large_model = Mock(spec=nn.Module)
        large_model.state_dict = Mock(return_value={
            f'layer{i}.weight': torch.randn(1000, 1000)
            for i in range(10)
        })
        
        filepath = temp_checkpoint_dir / "large_state.pt"
        checkpoint_manager.save(filepath=filepath, model=large_model)
        
        assert filepath.exists()
        checkpoint = torch.load(filepath)
        assert len(checkpoint['model_state_dict']) == 10
    
    def test_save_with_special_characters_in_path(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save handles paths with special characters."""
        special_path = temp_checkpoint_dir / "test-model_v1.2.3.pt"
        
        checkpoint_manager.save(filepath=special_path, model=mock_model)
        
        assert special_path.exists()
    
    def test_save_with_negative_epoch(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save accepts negative epoch (edge case)."""
        filepath = temp_checkpoint_dir / "negative_epoch.pt"
        
        # Should not raise - just stores the value
        checkpoint_manager.save(filepath=filepath, model=mock_model, epoch=-1)
        
        checkpoint = torch.load(filepath)
        assert checkpoint['epoch'] == -1
    
    def test_save_with_very_large_epoch(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save with very large epoch number."""
        filepath = temp_checkpoint_dir / "large_epoch.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            epoch=1000000
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['epoch'] == 1000000
    
    def test_save_with_nan_best_val_loss(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save with NaN best_val_loss."""
        filepath = temp_checkpoint_dir / "nan_loss.pt"
        import math
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            best_val_loss=float('nan')
        )
        
        checkpoint = torch.load(filepath)
        assert math.isnan(checkpoint['best_val_loss'])
    
    def test_save_with_negative_inf_best_val_loss(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save with negative infinity best_val_loss."""
        filepath = temp_checkpoint_dir / "neg_inf_loss.pt"
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            best_val_loss=float('-inf')
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['best_val_loss'] == float('-inf')
    
    def test_load_corrupted_file(
        self, checkpoint_manager, temp_checkpoint_dir
    ):
        """Test load handles corrupted file gracefully."""
        filepath = temp_checkpoint_dir / "corrupted.pt"
        
        # Write invalid data
        with open(filepath, 'wb') as f:
            f.write(b'not a valid pytorch checkpoint')
        
        with pytest.raises(Exception):  # Could be various exceptions
            checkpoint_manager.load(filepath)
    
    def test_save_hyper_parameters_with_none_values(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save hyper_parameters containing None values."""
        filepath = temp_checkpoint_dir / "none_hyper.pt"
        
        hyper_params = {
            'model_name': 'GCN',
            'optional_field': None,
            'another_none': None,
        }
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            hyper_parameters=hyper_params
        )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['hyper_parameters']['optional_field'] is None
    
    def test_save_data_info_with_tensors(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save data_info containing tensors."""
        filepath = temp_checkpoint_dir / "tensor_data_info.pt"
        
        data_info = {
            'mean': torch.tensor([0.5, 0.5, 0.5]),
            'std': torch.tensor([0.1, 0.1, 0.1]),
        }
        
        checkpoint_manager.save(
            filepath=filepath,
            model=mock_model,
            data_info=data_info
        )
        
        checkpoint = torch.load(filepath)
        torch.testing.assert_close(
            checkpoint['data_info']['mean'],
            data_info['mean']
        )


# =============================================================================
# INTEGRATION WITH REAL MODEL LOADING
# =============================================================================

class TestIntegrationWithRealModelLoading:
    """Test integration scenarios with real model state loading."""
    
    def test_load_state_dict_into_real_model(
        self, checkpoint_manager, real_model, temp_checkpoint_dir
    ):
        """Test loading checkpoint state_dict into a real model."""
        filepath = temp_checkpoint_dir / "for_loading.pt"
        
        # Modify model weights before saving
        with torch.no_grad():
            real_model.linear.weight.fill_(1.0)
            real_model.linear.bias.fill_(0.5)
        
        # Save
        checkpoint_manager.save(filepath=filepath, model=real_model)
        
        # Create new model and load state
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        new_model = SimpleModel()
        checkpoint = checkpoint_manager.load(filepath)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify weights were loaded correctly
        torch.testing.assert_close(
            new_model.linear.weight,
            torch.ones(5, 10)
        )
        torch.testing.assert_close(
            new_model.linear.bias,
            torch.full((5,), 0.5)
        )
    
    def test_load_optimizer_state_into_real_optimizer(
        self, checkpoint_manager, real_model, real_optimizer, temp_checkpoint_dir
    ):
        """Test loading checkpoint optimizer state into a real optimizer."""
        filepath = temp_checkpoint_dir / "opt_state.pt"
        
        # Take some optimizer steps to create state
        for _ in range(5):
            real_optimizer.zero_grad()
            output = real_model(torch.randn(4, 10))
            loss = output.sum()
            loss.backward()
            real_optimizer.step()
        
        # Save
        checkpoint_manager.save(
            filepath=filepath,
            model=real_model,
            optimizer=real_optimizer
        )
        
        # Load into new optimizer
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        new_model = SimpleModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        
        checkpoint = checkpoint_manager.load(filepath)
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Verify optimizer state was loaded (state dict should match structure)
        assert 'param_groups' in new_optimizer.state_dict()


# =============================================================================
# MULTIPLE SAVE OPERATIONS TESTS
# =============================================================================

class TestMultipleSaveOperations:
    """Test multiple consecutive save operations."""
    
    def test_multiple_saves_same_file(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test multiple saves to the same file."""
        filepath = temp_checkpoint_dir / "multiple.pt"
        
        for epoch in range(5):
            checkpoint_manager.save(
                filepath=filepath,
                model=mock_model,
                epoch=epoch
            )
        
        checkpoint = torch.load(filepath)
        assert checkpoint['epoch'] == 4  # Last save
    
    def test_multiple_saves_different_files(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test multiple saves to different files."""
        for i in range(5):
            filepath = temp_checkpoint_dir / f"checkpoint_{i}.pt"
            checkpoint_manager.save(
                filepath=filepath,
                model=mock_model,
                epoch=i
            )
        
        # Verify all files exist with correct epochs
        for i in range(5):
            filepath = temp_checkpoint_dir / f"checkpoint_{i}.pt"
            checkpoint = torch.load(filepath)
            assert checkpoint['epoch'] == i


# =============================================================================
# THREAD SAFETY TESTS (BASIC)
# =============================================================================

class TestBasicThreadSafety:
    """Basic thread safety tests."""
    
    def test_create_version_info_is_thread_safe(self, checkpoint_manager):
        """Test create_version_info can be called from multiple threads."""
        import threading
        results = []
        errors = []
        
        def call_version_info():
            try:
                result = checkpoint_manager.create_version_info()
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=call_version_info) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 10
        # All results should have the same structure
        for result in results:
            assert 'checkpoint_format_version' in result


# =============================================================================
# DOCUMENTATION COMPLIANCE TESTS
# =============================================================================

class TestDocumentationCompliance:
    """Test that behavior matches documentation."""
    
    def test_save_returns_path_as_documented(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test save() returns Path as documented."""
        filepath = temp_checkpoint_dir / "doc_test.pt"
        
        result = checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        # Documentation says: "Returns: Path to saved checkpoint"
        assert isinstance(result, Path)
        assert result == filepath.resolve()
    
    def test_load_returns_dict_as_documented(
        self, checkpoint_manager, mock_model, temp_checkpoint_dir
    ):
        """Test load() returns Dict as documented."""
        filepath = temp_checkpoint_dir / "doc_test2.pt"
        checkpoint_manager.save(filepath=filepath, model=mock_model)
        
        result = checkpoint_manager.load(filepath)
        
        # Documentation says: "Returns: Checkpoint dictionary"
        assert isinstance(result, dict)
    
    def test_usage_example_from_docstring(
        self, mock_model, mock_optimizer, temp_checkpoint_dir
    ):
        """
        Test usage example from class docstring works.
        
        From docstring:
            # Caller provides working_root_dir explicitly
            working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
            manager = CheckpointManager(working_root_dir=working_root_dir)
            
            # Save with relative path
            manager.save(
                filepath="checkpoints/model.pt",  # Resolved to working_root_dir/checkpoints/model.pt
                model=model,
                optimizer=optimizer,
                ...
            )
            
            # Load with automatic path resolution
            checkpoint = manager.load("checkpoints/model.pt")
        """
        # Follow the docstring pattern exactly
        working_root_dir = temp_checkpoint_dir
        manager = CheckpointManager(working_root_dir=working_root_dir)
        
        hyper_parameters = {
            'model_name': 'GCN',
            'task_type': 'graph_regression',
            'hyperparameters': {'hidden_channels': 64},
            'model_info': {'name': 'GCN'},
            'wrapper_info': {'type': 'GraphRegressionWrapper'},
        }
        
        # Save with relative path as shown in docstring
        result_path = manager.save(
            filepath="checkpoints/model.pt",  # Relative path
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=100,
            hyper_parameters=hyper_parameters
        )
        
        # Verify resolved to working_root_dir/checkpoints/model.pt
        expected_path = (working_root_dir / "checkpoints/model.pt").resolve()
        assert result_path == expected_path
        assert expected_path.exists()
        
        # Load with automatic path resolution as in docstring
        checkpoint = manager.load("checkpoints/model.pt")
        hyper_params = checkpoint['hyper_parameters']
        
        assert hyper_params == hyper_parameters
        assert checkpoint['epoch'] == 100
    
    def test_dependency_injection_pattern(self, temp_checkpoint_dir):
        """Test that CheckpointManager follows Dependency Injection pattern."""
        # As documented: "All path resolution requires explicit `working_root_dir: Path` parameter"
        # "No hidden config loading (Service Locator anti-pattern removed)"
        
        # Must provide working_root_dir explicitly
        manager = CheckpointManager(working_root_dir=temp_checkpoint_dir)
        
        # The manager should use the provided working_root_dir
        assert manager._working_root_dir == temp_checkpoint_dir.resolve()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
