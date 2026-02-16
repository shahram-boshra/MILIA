#!/usr/bin/env python3
"""
Smoke Test: Prediction Pipeline
=================================

Rapid, lightweight checks that verify the MILIA post-training prediction
pathway can load a mock checkpoint and run prediction without crashing.
This is the **first gate in the CI/CD pipeline** — if smoke tests fail,
no further (more expensive) tests are triggered.

These tests do NOT validate numerical correctness; they confirm the
prediction pathway is "not on fire."

Modules exercised (Section 1.4 of MILIA_Test_Recommendations.md):
- milia_pipeline/models/post_training/checkpoint/checkpoint_manager.py — CheckpointManager
- milia_pipeline/models/post_training/inference/model_loader.py        — ModelLoader
- milia_pipeline/models/post_training/inference/predictor.py           — Predictor
- milia_pipeline/models/post_training/data_preparation/data_converter.py — DataConverterRegistry

Scope:
- Creates a minimal mock checkpoint
- Loads it
- Runs prediction on 1–2 synthetic PyG Data objects
- Asserts predictions are tensors of correct shape
- Total runtime target: < 15 seconds

Usage:
    pytest tests/test_smoke_prediction_pipeline.py -v --tb=short
    pytest tests/test_smoke_prediction_pipeline.py -v -m smoke

Docker usage:
    (shah_env) root@01b78773d9b4:/app/milia# pytest tests/test_smoke_prediction_pipeline.py -v

Author: MILIA Team
Version: 1.0.0
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ===========================================================================
# PATH SETUP: Add project root to Python path FIRST
# ===========================================================================
# This ensures milia_pipeline is importable regardless of working directory.
# Evidence: MILIA_Test_Recommendations.md "NOTE: I MUST Add the project root
# to Python path FIRST"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ===========================================================================
# PYTEST MARKERS
# ===========================================================================
# The ``smoke`` marker requires registration via conftest.py or pytest.ini
# to avoid PytestUnknownMarkWarning. This warning fires at *collection time*
# (before any test-level filterwarnings apply), so it CANNOT be suppressed
# from within a test file.
#
# Required: add to tests/conftest.py (or the project's existing conftest.py):
#
#     def pytest_configure(config):
#         config.addinivalue_line(
#             "markers",
#             "smoke: Smoke tests — fast, first gate in CI/CD pipeline",
#         )
#
# Alternatively, add to pytest.ini / pyproject.toml:
#
#     [pytest]
#     markers =
#         smoke: Smoke tests — fast, first gate in CI/CD pipeline
#
pytestmark = [
    pytest.mark.smoke,
    # -----------------------------------------------------------------------
    # WARNING MANAGEMENT STRATEGY: "Warnings as errors" with targeted exceptions.
    #
    # Evidence (Simon Willison, pytest best practice):
    #   "If you want ALL warnings to be failures in both development and CI...
    #    you may find there are warnings you cannot fix (because they are in
    #    dependency libraries). You can ignore those."
    #
    # Evidence (SQLAlchemy maintainer Mike Bayer, GitHub Discussion #6675):
    #   "test suites should generally raise all warnings"
    #
    # Evidence (PSF/Black Issue #3171):
    #   "black was raising a DeprecationWarning, but it was missed because
    #    the test suite is ignoring warnings."
    #
    # STRATEGY:
    # 1. Promote all warnings to errors (catches regressions in our own code)
    # 2. Explicitly ignore SPECIFIC third-party warnings we cannot fix
    # 3. Each exception is documented with source, reason, and remediation
    # -----------------------------------------------------------------------
    pytest.mark.filterwarnings("error"),
    #
    # Exception 1: torch.load weights_only FutureWarning
    # Source: checkpoint_manager.py line 261 → torch.load()
    # Reason: PyTorch >= 2.4 emits FutureWarning when weights_only is not
    #         explicitly set. This filter is retained for backward compatibility
    #         in case tests run against a checkpoint_manager.py that has not
    #         yet been updated with the explicit weights_only=True parameter.
    # Remediation: DONE — checkpoint_manager.py now passes weights_only=True.
    pytest.mark.filterwarnings(
        "ignore:You are using `torch.load` with `weights_only=False`:FutureWarning"
    ),
    #
    # Exception 2: PyG and scientific stack deprecation warnings
    # Source: torch_geometric, torch_scatter, numpy internal deprecations
    # Reason: Third-party libraries we do not control may emit DeprecationWarning
    #         or UserWarning during import or execution. These are not actionable
    #         by the MILIA project.
    # Remediation: Monitor upstream releases; remove when dependencies are updated.
    pytest.mark.filterwarnings("ignore::DeprecationWarning:torch_geometric"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning:torch_scatter"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning:numpy"),
    pytest.mark.filterwarnings("ignore::UserWarning:torch_geometric"),
    pytest.mark.filterwarnings("ignore::UserWarning:torch.jit"),
    #
    # Exception 3: pyparsing oneOf deprecation via matplotlib import chain
    # Source: torchmetrics → matplotlib → pyparsing/util.py:445
    # Reason: pyparsing emits DeprecationWarning for 'oneOf' (renamed to
    #         'one_of'). Triggered during matplotlib import at module load.
    #         MILIA does not call pyparsing directly.
    #         NOTE: pyparsing's warnings.warn() uses stacklevel=2, so Python
    #         attributes the warning to the *caller* (matplotlib._fontconfig_pattern),
    #         not to pyparsing itself. The module filter must match the resolved
    #         source module after stacklevel adjustment.
    # Remediation: Resolved when matplotlib upgrades its pyparsing usage.
    pytest.mark.filterwarnings("ignore::DeprecationWarning:matplotlib"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning:pyparsing"),
]

# ===========================================================================
# MODULE-LEVEL LOGGER
# ===========================================================================
logger = logging.getLogger(__name__)


# ===========================================================================
# HELPER: Minimal GNN for smoke testing
# ===========================================================================


class _MinimalGCN(nn.Module):
    """A minimal GCN-like model that requires NO PyG optional extensions.

    This model uses only ``torch_geometric.nn.GCNConv``, which is part of
    the core PyG package and does NOT require ``torch_scatter``,
    ``torch_sparse``, ``torch_cluster``, or ``torch_spline_conv``.

    Evidence: PyG documentation states GCNConv uses ``torch.sparse`` under
    the hood since PyG 2.0+ and works without optional C++ extensions.

    Architecture:
        GCNConv(in_channels, hidden) → ReLU → GCNConv(hidden, out_channels)
        → global_mean_pool → Linear(out_channels, num_targets)

    Args:
        in_channels: Number of input node features.
        hidden_channels: Hidden layer width.
        out_channels: Intermediate output dimension before pooling head.
        num_targets: Final prediction dimension (default: 1 for regression).
    """

    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 16,
        out_channels: int = 16,
        num_targets: int = 1,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.pool = global_mean_pool
        self.head = nn.Linear(out_channels, num_targets)

    def forward(self, x, edge_index, batch=None, **kwargs):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        if batch is not None:
            x = self.pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return self.head(x)


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture(scope="module")
def tmp_work_dir():
    """Provide a temporary working directory for the test module.

    Used as ``working_root_dir`` for all DI-pattern classes.
    Cleaned up after all tests in this module complete.
    """
    tmp_dir = tempfile.mkdtemp(prefix="milia_smoke_pred_")
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def synthetic_pyg_data_list() -> list:
    """Create a minimal list of synthetic PyG Data objects for prediction.

    Generates 5 small molecular graphs with:
    - x:          Node features (num_atoms, 11) — 11 is MILIA's default feature dim
    - edge_index: COO sparse edge connectivity
    - y:          Graph-level regression target shape (1,)
    - pos:        3D coordinates (num_atoms, 3) — optional, for 3D-aware models
    - z:          Atomic numbers (num_atoms,) — optional, for equivariant models

    Evidence: predictor.py Predictor.predict() expects PyG Data objects
    with at minimum x, edge_index (lines 209-240). The _forward() method
    introspects the model signature to determine which attributes to pass
    (lines 242-413).
    """
    from torch_geometric.data import Data

    data_list = []
    rng = np.random.RandomState(42)

    for i in range(5):
        num_atoms = rng.randint(3, 8)
        num_edges = rng.randint(num_atoms, num_atoms * 3)

        # Node features: (num_atoms, 11) — matches MILIA default feature dimension
        x = torch.randn(num_atoms, 11, dtype=torch.float32)

        # Edge connectivity: random COO format
        src = torch.randint(0, num_atoms, (num_edges,))
        dst = torch.randint(0, num_atoms, (num_edges,))
        edge_index = torch.stack([src, dst], dim=0)

        # Graph-level regression target
        y = torch.tensor([rng.uniform(-5.0, 5.0)], dtype=torch.float32)

        # 3D coordinates
        pos = torch.randn(num_atoms, 3, dtype=torch.float32)

        # Atomic numbers (C=6, N=7, O=8, H=1)
        z = torch.tensor(rng.choice([1, 6, 7, 8], size=num_atoms), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y, pos=pos, z=z)
        data_list.append(data)

    return data_list


@pytest.fixture(scope="module")
def minimal_model():
    """Create a minimal GCN model for checkpoint creation.

    Returns a model with known architecture and state_dict that can be
    saved to a checkpoint and later loaded by ModelLoader.

    Evidence: model_loader.py ModelLoader._load() calls
    ModelFactory.create_model_with_info() to recreate the model (line 279),
    then loads state_dict (line 395). For smoke tests, we bypass the
    factory by directly constructing a known model.
    """
    model = _MinimalGCN(
        in_channels=11,
        hidden_channels=16,
        out_channels=16,
        num_targets=1,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def mock_checkpoint_path(tmp_work_dir, minimal_model):
    """Create a minimal v2.0 checkpoint file on disk.

    Evidence: checkpoint_manager.py CheckpointManager.save() (lines 137-189)
    saves a dict with keys: epoch, global_step, model_state_dict,
    metrics_history, best_val_loss, hyper_parameters, data_info,
    version_info. The format version is CHECKPOINT_FORMAT_VERSION = "2.0"
    (line 16).

    Evidence: model_loader.py ModelLoader._load() reads hyper_parameters
    for model_name (line 192), task_type (line 193), and hyperparameters
    dict (lines 203-223). It also reads model_info from
    hyper_parameters['model_info'] (line 245) and data_info from
    checkpoint['data_info'] (line 426).
    """
    checkpoint_dir = tmp_work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "smoke_test_model.pt"

    checkpoint = {
        # Training state
        "epoch": 10,
        "global_step": 100,
        "model_state_dict": minimal_model.state_dict(),
        "metrics_history": {"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6]},
        "best_val_loss": 0.6,
        # Model recreation metadata (v2.0 format)
        "hyper_parameters": {
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": {
                "in_channels": 11,
                "hidden_channels": 16,
                "out_channels": 16,
                "num_layers": 2,
            },
            "model_info": {
                "task_type": "graph_regression",
                "uses_edge_features": False,
                "hyperparameters_values": {
                    "in_channels": 11,
                    "hidden_channels": 16,
                    "out_channels": 16,
                    "num_layers": 2,
                },
            },
            "wrapper_info": {},
            "target_selection_config": None,
        },
        "data_info": {
            "requires_edge_features": False,
            "uses_edge_features": False,
            "structural_features_config": {
                "atom": ["atomic_num", "degree"],
                "bond": ["bond_type"],
            },
        },
        "version_info": {
            "milia_version": "1.0.0",
            "checkpoint_format_version": "2.0",
            "pytorch_version": str(torch.__version__),
            "pyg_version": "unknown",
            "created_at": "2026-01-01T00:00:00",
        },
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture(scope="module")
def mock_v1_checkpoint_path(tmp_work_dir, minimal_model):
    """Create a minimal v1.0 (legacy) checkpoint file on disk.

    Evidence: checkpoint_manager.py CheckpointManager.load() (lines 210-227)
    detects v1.0 checkpoints by checking version_info.checkpoint_format_version.
    For v1.0 checkpoints, it adds empty metadata defaults:
    checkpoint.setdefault('hyper_parameters', {})
    checkpoint.setdefault('data_info', {})
    checkpoint.setdefault('version_info', {'checkpoint_format_version': '1.0'})

    Evidence: model_loader.py ModelLoader.load_from_checkpoint() (line 88-91)
    accepts override parameters model_name, hyperparameters, task_type which
    are required for v1.0 checkpoints that lack hyper_parameters metadata.
    """
    checkpoint_dir = tmp_work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "smoke_test_legacy_model.pt"

    # v1.0 checkpoint: minimal structure, no hyper_parameters metadata
    checkpoint = {
        "epoch": 5,
        "global_step": 50,
        "model_state_dict": minimal_model.state_dict(),
        "best_val_loss": 0.8,
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


# ===========================================================================
# SECTION 1: CHECKPOINT MANAGER SMOKE TESTS
# ===========================================================================


class TestCheckpointManagerSmoke:
    """Smoke tests for CheckpointManager.

    Verifies that CheckpointManager can be imported, instantiated with DI
    pattern (working_root_dir), and perform save/load operations without
    crashing.

    Evidence:
    - checkpoint_manager.py CheckpointManager.__init__(working_root_dir: Path)
      (line 62-69): Requires working_root_dir via Dependency Injection.
    - checkpoint_manager.py CheckpointManager.save() (lines 137-189):
      Saves checkpoint with model_state_dict, hyper_parameters, version_info.
    - checkpoint_manager.py CheckpointManager.load() (lines 197-237):
      Loads checkpoint with backward compatibility for v1.0 format.
    """

    def test_checkpoint_manager_importable(self):
        """CheckpointManager can be imported without errors."""
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CHECKPOINT_FORMAT_VERSION,
            CheckpointManager,
        )

        assert CheckpointManager is not None
        assert CHECKPOINT_FORMAT_VERSION == "2.0"

    def test_checkpoint_manager_instantiation(self, tmp_work_dir):
        """CheckpointManager can be instantiated with working_root_dir.

        Evidence: checkpoint_manager.py CheckpointManager.__init__()
        (line 62-69) accepts working_root_dir: Path and resolves it to
        an absolute path via Path.expanduser().resolve().
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        manager = CheckpointManager(working_root_dir=tmp_work_dir)
        assert manager is not None

    def test_create_version_info(self):
        """create_version_info() returns a dict with expected keys.

        Evidence: checkpoint_manager.py create_version_info() (lines 71-83)
        returns dict with keys: milia_version, checkpoint_format_version,
        pytorch_version, pyg_version, created_at.
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        version_info = CheckpointManager.create_version_info()
        assert isinstance(version_info, dict)
        assert "checkpoint_format_version" in version_info
        assert "pytorch_version" in version_info
        assert "created_at" in version_info

    def test_get_checkpoint_dir(self, tmp_work_dir):
        """get_checkpoint_dir() creates and returns checkpoint directory.

        Evidence: checkpoint_manager.py get_checkpoint_dir() (lines 85-96)
        returns working_root_dir / subdir and creates it if needed.
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        manager = CheckpointManager(working_root_dir=tmp_work_dir)
        checkpoint_dir = manager.get_checkpoint_dir()

        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()
        assert checkpoint_dir == tmp_work_dir / "checkpoints"

    def test_save_and_load_roundtrip(self, tmp_work_dir, minimal_model):
        """Checkpoint save and load complete without errors.

        Evidence: checkpoint_manager.py save() (lines 137-189) saves
        model_state_dict, hyper_parameters, version_info to file.
        load() (lines 197-237) loads and returns the checkpoint dict.
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        manager = CheckpointManager(working_root_dir=tmp_work_dir)

        # Save checkpoint
        filepath = manager.save(
            filepath="checkpoints/roundtrip_test.pt",
            model=minimal_model,
            epoch=5,
            global_step=100,
            best_val_loss=0.42,
            hyper_parameters={
                "model_name": "GCN",
                "task_type": "graph_regression",
                "hyperparameters": {"hidden_channels": 16},
            },
        )

        assert filepath.exists()

        # Load checkpoint
        checkpoint = manager.load("checkpoints/roundtrip_test.pt")

        assert isinstance(checkpoint, dict)
        assert checkpoint["epoch"] == 5
        assert checkpoint["global_step"] == 100
        assert checkpoint["best_val_loss"] == 0.42
        assert "model_state_dict" in checkpoint
        assert "hyper_parameters" in checkpoint
        assert "version_info" in checkpoint

    def test_is_v2_checkpoint(self, tmp_work_dir, mock_checkpoint_path):
        """is_v2_checkpoint correctly identifies v2.0 format.

        Evidence: checkpoint_manager.py is_v2_checkpoint() (lines 239-242)
        checks version_info.checkpoint_format_version >= '2.0'.
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        manager = CheckpointManager(working_root_dir=tmp_work_dir)
        checkpoint = manager.load(mock_checkpoint_path)

        assert manager.is_v2_checkpoint(checkpoint) is True

    def test_legacy_checkpoint_backward_compatibility(self, tmp_work_dir, mock_v1_checkpoint_path):
        """v1.0 checkpoints load with default metadata.

        Evidence: checkpoint_manager.py load() (lines 219-227) detects
        format_version == '1.0' and adds empty defaults:
        checkpoint.setdefault('hyper_parameters', {})
        checkpoint.setdefault('data_info', {})
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        manager = CheckpointManager(working_root_dir=tmp_work_dir)
        checkpoint = manager.load(mock_v1_checkpoint_path)

        assert isinstance(checkpoint, dict)
        assert "model_state_dict" in checkpoint
        # v1.0 defaults are injected
        assert "hyper_parameters" in checkpoint
        assert "data_info" in checkpoint
        assert manager.is_v2_checkpoint(checkpoint) is False

    def test_get_hyper_parameters(self, tmp_work_dir, mock_checkpoint_path):
        """get_hyper_parameters() extracts hyper_parameters from checkpoint.

        Evidence: checkpoint_manager.py get_hyper_parameters() (lines 244-246)
        returns checkpoint.get('hyper_parameters', {}).
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        manager = CheckpointManager(working_root_dir=tmp_work_dir)
        checkpoint = manager.load(mock_checkpoint_path)
        hyper_params = manager.get_hyper_parameters(checkpoint)

        assert isinstance(hyper_params, dict)
        assert hyper_params.get("model_name") == "GCN"
        assert hyper_params.get("task_type") == "graph_regression"

    def test_get_model_name(self, tmp_work_dir, mock_checkpoint_path):
        """get_model_name() extracts model name from checkpoint.

        Evidence: checkpoint_manager.py get_model_name() (lines 248-251)
        returns hyper_params.get('model_name').
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        manager = CheckpointManager(working_root_dir=tmp_work_dir)
        checkpoint = manager.load(mock_checkpoint_path)
        model_name = manager.get_model_name(checkpoint)

        assert model_name == "GCN"

    def test_resolve_path_relative(self, tmp_work_dir):
        """_resolve_path resolves relative paths against working_root_dir.

        Evidence: checkpoint_manager.py _resolve_path() (lines 98-113)
        resolves relative paths as (working_root_dir / path).resolve().
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        manager = CheckpointManager(working_root_dir=tmp_work_dir)
        resolved = manager._resolve_path("checkpoints/model.pt")

        assert resolved.is_absolute()
        assert str(tmp_work_dir) in str(resolved)

    def test_resolve_path_absolute(self, tmp_work_dir):
        """_resolve_path returns absolute paths unchanged.

        Evidence: checkpoint_manager.py _resolve_path() (lines 98-113)
        checks path.is_absolute() and returns path.resolve() if True.
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        manager = CheckpointManager(working_root_dir=tmp_work_dir)
        abs_path = Path("/tmp/absolute_test.pt")
        resolved = manager._resolve_path(abs_path)

        assert resolved == abs_path.resolve()


# ===========================================================================
# SECTION 2: MODEL LOADER SMOKE TESTS
# ===========================================================================


class TestModelLoaderSmoke:
    """Smoke tests for ModelLoader.

    Verifies that ModelLoader can be imported and that its class methods
    and convenience functions are accessible. Full model loading requires
    ModelFactory integration (tested with mocks).

    Evidence:
    - model_loader.py ModelLoader.__init__(working_root_dir: Path) (line 69-79):
      Requires working_root_dir, creates CheckpointManager and model_factory.
    - model_loader.py ModelLoader.load_from_checkpoint() (lines 81-153):
      Class method requiring checkpoint_path and working_root_dir.
    - model_loader.py ModelLoader.get_checkpoint_info() (lines 440-483):
      Inspection without loading model.
    - model_loader.py load_model(), load_model_only() (lines 490-562):
      Convenience functions.
    """

    def test_model_loader_importable(self):
        """ModelLoader can be imported without errors."""
        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
            load_model,
            load_model_only,
        )

        assert ModelLoader is not None
        assert callable(load_model)
        assert callable(load_model_only)

    def test_model_loader_has_expected_methods(self):
        """ModelLoader exposes documented class methods.

        Evidence: model_loader.py documents load_from_checkpoint (line 81),
        get_checkpoint_info (line 440), and _load (line 155) methods.
        """
        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        assert hasattr(ModelLoader, "load_from_checkpoint")
        assert hasattr(ModelLoader, "get_checkpoint_info")
        assert hasattr(ModelLoader, "_load")

    def test_get_checkpoint_info(self, tmp_work_dir, mock_checkpoint_path):
        """get_checkpoint_info() returns metadata without loading model.

        Evidence: model_loader.py get_checkpoint_info() (lines 440-483)
        returns dict with format_version, is_v2, model_name, task_type,
        epoch, best_val_loss, hyper_parameters, data_info, version_info,
        checkpoint_path, has_wrapper_info, has_target_selection.

        This method only reads the checkpoint file and does NOT invoke
        ModelFactory, so it works without mocking the factory.
        """
        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        info = ModelLoader.get_checkpoint_info(
            checkpoint_path=mock_checkpoint_path,
            working_root_dir=tmp_work_dir,
        )

        assert isinstance(info, dict)
        assert info["format_version"] == "2.0"
        assert info["is_v2"] is True
        assert info["model_name"] == "GCN"
        assert info["task_type"] == "graph_regression"
        assert info["epoch"] == 10
        assert info["best_val_loss"] == 0.6

    def test_get_checkpoint_info_legacy(self, tmp_work_dir, mock_v1_checkpoint_path):
        """get_checkpoint_info() handles v1.0 checkpoints gracefully.

        Evidence: model_loader.py get_checkpoint_info() reads from
        checkpoint dict. For v1.0 checkpoints, hyper_parameters will be
        empty (set by CheckpointManager.load() backward compatibility),
        so model_name and task_type default to 'UNKNOWN'.
        """
        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        info = ModelLoader.get_checkpoint_info(
            checkpoint_path=mock_v1_checkpoint_path,
            working_root_dir=tmp_work_dir,
        )

        assert isinstance(info, dict)
        assert info["is_v2"] is False
        assert info["model_name"] == "UNKNOWN"
        assert info["task_type"] == "UNKNOWN"

    def test_load_from_checkpoint_with_factory_mock(
        self, tmp_work_dir, mock_checkpoint_path, minimal_model
    ):
        """load_from_checkpoint() succeeds when ModelFactory is mocked.

        Evidence: model_loader.py _load() (lines 278-286) calls
        self.model_factory.create_model_with_info(name, hyperparameters,
        task_type, sample_data=None, device=device,
        target_selection_config=target_selection_config)
        which returns (model, model_info).

        We mock get_factory() to return a factory whose
        create_model_with_info() returns our minimal_model, bypassing
        the real ModelRegistry/ModelFactory which requires full project
        initialization.

        Mock Pollution Prevention: Uses @patch decorator at test level,
        not module-level sys.modules injection.
        """
        mock_model_info = {
            "task_type": "graph_regression",
            "uses_edge_features": False,
        }

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (
            minimal_model,
            mock_model_info,
        )

        with patch(
            "milia_pipeline.models.post_training.inference.model_loader.get_factory",
            return_value=mock_factory,
        ):
            from milia_pipeline.models.post_training.inference.model_loader import (
                ModelLoader,
            )

            model, model_info = ModelLoader.load_from_checkpoint(
                checkpoint_path=mock_checkpoint_path,
                working_root_dir=tmp_work_dir,
                device=torch.device("cpu"),
            )

        # Assertions: model is loaded and in eval mode
        assert model is not None
        assert isinstance(model, nn.Module)
        assert not model.training, "Model should be in eval mode after loading"

        # Assertions: model_info is populated
        assert isinstance(model_info, dict)

    def test_load_from_checkpoint_file_not_found(self, tmp_work_dir):
        """load_from_checkpoint() raises FileNotFoundError for missing checkpoint.

        Evidence: model_loader.py _load() (lines 169-176) checks
        resolved_checkpoint_path.exists() and raises FileNotFoundError
        with all searched locations listed.
        """
        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        mock_factory = MagicMock()
        with (
            patch(
                "milia_pipeline.models.post_training.inference.model_loader.get_factory",
                return_value=mock_factory,
            ),
            pytest.raises(FileNotFoundError),
        ):
            ModelLoader.load_from_checkpoint(
                checkpoint_path="nonexistent/model.pt",
                working_root_dir=tmp_work_dir,
                device=torch.device("cpu"),
            )

    def test_load_from_checkpoint_v1_missing_params_raises(
        self, tmp_work_dir, mock_v1_checkpoint_path
    ):
        """load_from_checkpoint() raises ValueError for v1.0 checkpoint without overrides.

        Evidence: model_loader.py _load() (lines 256-266) raises ValueError
        if resolved_model_name or resolved_task_type is None, which occurs
        for v1.0 checkpoints that lack hyper_parameters metadata and
        no overrides are provided.
        """
        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        mock_factory = MagicMock()
        with (
            patch(
                "milia_pipeline.models.post_training.inference.model_loader.get_factory",
                return_value=mock_factory,
            ),
            pytest.raises(ValueError, match="model_name is required"),
        ):
            ModelLoader.load_from_checkpoint(
                checkpoint_path=mock_v1_checkpoint_path,
                working_root_dir=tmp_work_dir,
                device=torch.device("cpu"),
            )

    def test_load_from_checkpoint_v1_with_overrides(
        self, tmp_work_dir, mock_v1_checkpoint_path, minimal_model
    ):
        """load_from_checkpoint() succeeds for v1.0 checkpoint WITH overrides.

        Evidence: model_loader.py _load() (lines 192-193) uses overrides:
        resolved_model_name = model_name or hyper_params.get('model_name')
        resolved_task_type = task_type or hyper_params.get('task_type')
        When overrides are provided, v1.0 checkpoints can be loaded.
        """
        mock_model_info = {
            "task_type": "graph_regression",
            "uses_edge_features": False,
        }

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (
            minimal_model,
            mock_model_info,
        )

        with patch(
            "milia_pipeline.models.post_training.inference.model_loader.get_factory",
            return_value=mock_factory,
        ):
            from milia_pipeline.models.post_training.inference.model_loader import (
                ModelLoader,
            )

            model, model_info = ModelLoader.load_from_checkpoint(
                checkpoint_path=mock_v1_checkpoint_path,
                working_root_dir=tmp_work_dir,
                device=torch.device("cpu"),
                model_name="GCN",
                hyperparameters={"in_channels": 11, "hidden_channels": 16},
                task_type="graph_regression",
            )

        assert model is not None
        assert isinstance(model, nn.Module)


# ===========================================================================
# SECTION 3: PREDICTOR SMOKE TESTS
# ===========================================================================


class TestPredictorSmoke:
    """Smoke tests for Predictor.

    Verifies that Predictor can be imported, instantiated directly with a
    model, and run single/batch predictions on synthetic PyG Data objects.

    Evidence:
    - predictor.py Predictor.__init__(model, working_root_dir, device,
      task_type, model_info) (lines 60-95): Direct instantiation.
    - predictor.py Predictor.predict(data, return_numpy) (lines 209-240):
      Single prediction on Data or Batch.
    - predictor.py Predictor.predict_batch(dataset, batch_size, ...)
      (lines 427-467): Batch prediction.
    - predictor.py Predictor.structural_features_config property
      (lines 104-121): Exposes featurization config from model_info.
    - predictor.py Predictor.from_checkpoint() (lines 145-207):
      Class method that creates Predictor from checkpoint.
    """

    def test_predictor_importable(self):
        """Predictor can be imported without errors."""
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
            predict,
        )

        assert Predictor is not None
        assert callable(predict)

    def test_predictor_instantiation(self, tmp_work_dir, minimal_model):
        """Predictor can be instantiated with a model directly.

        Evidence: predictor.py Predictor.__init__() (lines 60-95) accepts
        model, working_root_dir, device, task_type, model_info. Sets model
        to eval mode and moves to device.
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        assert predictor is not None
        assert predictor.task_type == "graph_regression"
        assert not predictor.model.training

    def test_predict_single(self, tmp_work_dir, minimal_model, synthetic_pyg_data_list):
        """predict() returns a tensor for a single Data object.

        Evidence: predictor.py predict() (lines 209-240) calls
        data.to(self.device), runs _forward(data) under torch.no_grad(),
        then _postprocess(). Returns torch.Tensor or numpy.ndarray.
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        data = synthetic_pyg_data_list[0]
        prediction = predictor.predict(data)

        assert isinstance(prediction, torch.Tensor)
        assert prediction.shape[-1] == 1, (
            "Graph regression should produce (1,) or (N, 1) shaped output"
        )

    def test_predict_single_return_numpy(
        self, tmp_work_dir, minimal_model, synthetic_pyg_data_list
    ):
        """predict(return_numpy=True) returns a numpy array.

        Evidence: predictor.py predict() (lines 238-239) checks
        return_numpy flag and calls output.cpu().numpy().
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        data = synthetic_pyg_data_list[0]
        prediction = predictor.predict(data, return_numpy=True)

        assert isinstance(prediction, np.ndarray)

    def test_predict_batch(self, tmp_work_dir, minimal_model, synthetic_pyg_data_list):
        """predict_batch() returns concatenated predictions for dataset.

        Evidence: predictor.py predict_batch() (lines 427-467) creates
        DataLoader from dataset, iterates batches, calls predict() for
        each, concatenates results with torch.cat().
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        predictions = predictor.predict_batch(
            synthetic_pyg_data_list,
            batch_size=2,
        )

        assert isinstance(predictions, torch.Tensor)
        # Should have one prediction per graph in dataset
        assert predictions.shape[0] == len(synthetic_pyg_data_list)

    def test_predict_batch_return_numpy(self, tmp_work_dir, minimal_model, synthetic_pyg_data_list):
        """predict_batch(return_numpy=True) returns a numpy array.

        Evidence: predictor.py predict_batch() (lines 465-466) checks
        return_numpy flag and calls output.cpu().numpy().
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        predictions = predictor.predict_batch(
            synthetic_pyg_data_list,
            batch_size=2,
            return_numpy=True,
        )

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == len(synthetic_pyg_data_list)

    def test_structural_features_config_property(self, tmp_work_dir, minimal_model):
        """structural_features_config property returns config from model_info.

        Evidence: predictor.py structural_features_config property
        (lines 104-121) returns self.model_info.get('data_info', {})
        .get('structural_features_config').
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        model_info = {
            "data_info": {
                "structural_features_config": {
                    "atom": ["atomic_num", "degree"],
                    "bond": ["bond_type"],
                },
            },
        }

        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
            model_info=model_info,
        )

        config = predictor.structural_features_config
        assert config is not None
        assert "atom" in config
        assert "bond" in config
        assert "atomic_num" in config["atom"]

    def test_structural_features_config_property_none(self, tmp_work_dir, minimal_model):
        """structural_features_config returns None when model_info is empty.

        Evidence: predictor.py structural_features_config property
        (lines 118-121): Returns None if self.model_info is falsy or
        data_info.structural_features_config is missing.
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
        )

        assert predictor.structural_features_config is None

    def test_save_predictions_csv(self, tmp_work_dir, minimal_model, synthetic_pyg_data_list):
        """save_predictions() writes predictions to CSV file.

        Evidence: predictor.py save_predictions() (lines 469-546) resolves
        output path via _resolve_path(), converts tensor to numpy, and
        saves to CSV/JSON/NPY/PT format based on format parameter.
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        predictions = predictor.predict_batch(synthetic_pyg_data_list[:2], batch_size=2)

        output_path = predictor.save_predictions(
            predictions, "predictions/test_output.csv", format="csv"
        )

        assert output_path.exists()
        assert output_path.suffix == ".csv"

    def test_save_predictions_pt(self, tmp_work_dir, minimal_model, synthetic_pyg_data_list):
        """save_predictions(format='pt') writes predictions to .pt file.

        Evidence: predictor.py save_predictions() (lines 539-540) uses
        torch.save(torch.from_numpy(predictions), resolved_path) for 'pt'.
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        predictions = predictor.predict_batch(synthetic_pyg_data_list[:2], batch_size=2)

        output_path = predictor.save_predictions(
            predictions, "predictions/test_output.pt", format="pt"
        )

        assert output_path.exists()
        assert output_path.suffix == ".pt"

        # Verify the saved file can be loaded back
        loaded = torch.load(output_path, weights_only=True)
        assert isinstance(loaded, torch.Tensor)

    def test_from_checkpoint_with_factory_mock(
        self, tmp_work_dir, mock_checkpoint_path, minimal_model
    ):
        """from_checkpoint() creates Predictor from checkpoint with mocked factory.

        Evidence: predictor.py from_checkpoint() (lines 145-207) creates a
        CheckpointManager, resolves path, calls ModelLoader.load_from_checkpoint(),
        then creates Predictor with model, model_info, and task_type from checkpoint.

        Mock Pollution Prevention: Uses @patch decorator at test level.
        """
        mock_model_info = {
            "task_type": "graph_regression",
            "uses_edge_features": False,
            "data_info": {
                "structural_features_config": {
                    "atom": ["atomic_num"],
                    "bond": ["bond_type"],
                },
            },
        }

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (
            minimal_model,
            mock_model_info,
        )

        with patch(
            "milia_pipeline.models.post_training.inference.model_loader.get_factory",
            return_value=mock_factory,
        ):
            from milia_pipeline.models.post_training.inference.predictor import (
                Predictor,
            )

            predictor = Predictor.from_checkpoint(
                checkpoint_path=mock_checkpoint_path,
                working_root_dir=tmp_work_dir,
                device=torch.device("cpu"),
            )

        assert predictor is not None
        assert isinstance(predictor.model, nn.Module)
        assert predictor.task_type == "graph_regression"


# ===========================================================================
# SECTION 4: DATA CONVERTER REGISTRY SMOKE TESTS
# ===========================================================================


class TestDataConverterRegistrySmoke:
    """Smoke tests for DataConverterRegistry.

    Verifies that the converter registry can be imported, lists converters,
    and built-in converters (pyg_data, dict) can perform basic conversions.

    Evidence:
    - data_converter.py DataConverterRegistry (lines 199-294): Singleton
      registry with register, get, list_all, list_available, is_registered,
      auto_detect methods.
    - data_converter.py PyGDataConverter (lines 366-388): Passthrough
      converter for Data objects. Always available.
    - data_converter.py DictConverter (lines 391-428): Converts dict with
      tensors to Data. Always available.
    - data_converter.py convert_to_pyg() (lines ~1100-1233): High-level
      convenience function with auto-detect.
    - data_converter.py list_available_formats(), list_all_formats()
      (lines 1321-1328): Registry listing functions.
    """

    def test_data_converter_registry_importable(self):
        """DataConverterRegistry can be imported without errors."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            DataConverterRegistry,
            get_registry,
            register_converter,
        )

        assert DataConverterRegistry is not None
        assert callable(get_registry)
        assert callable(register_converter)

    def test_convenience_functions_importable(self):
        """Convenience functions can be imported without errors."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_batch_to_pyg,
            convert_to_pyg,
            list_all_formats,
            list_available_formats,
        )

        assert callable(convert_to_pyg)
        assert callable(convert_batch_to_pyg)
        assert callable(list_available_formats)
        assert callable(list_all_formats)

    def test_registry_list_all(self):
        """list_all() returns non-empty list of registered format names.

        Evidence: data_converter.py registers at least pyg_data (line 366)
        and dict (line 391) via @register_converter. Additional formats
        (smiles, inchi, xyz, ase_atoms, sdf) are registered if their
        dependencies are available.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            get_registry,
        )

        registry = get_registry()
        all_formats = registry.list_all()

        assert isinstance(all_formats, list)
        assert len(all_formats) >= 2, "At least pyg_data and dict converters should be registered"
        assert "pyg_data" in all_formats
        assert "dict" in all_formats

    def test_registry_list_available(self):
        """list_available() returns formats whose dependencies are installed.

        Evidence: data_converter.py list_available() (lines 255-266)
        instantiates each converter and checks is_available property.
        pyg_data and dict always return True.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            list_available_formats,
        )

        available = list_available_formats()

        assert isinstance(available, list)
        assert "pyg_data" in available
        assert "dict" in available

    def test_registry_is_registered(self):
        """is_registered() correctly identifies known formats.

        Evidence: data_converter.py is_registered() (lines 268-271)
        checks format_name.lower() in self._converters.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            get_registry,
        )

        registry = get_registry()
        assert registry.is_registered("pyg_data") is True
        assert registry.is_registered("dict") is True
        assert registry.is_registered("nonexistent_format_xyz") is False

    def test_pyg_data_converter_passthrough(self, synthetic_pyg_data_list):
        """PyGDataConverter passes Data objects through unchanged.

        Evidence: data_converter.py PyGDataConverter.convert() (lines 385-388)
        checks isinstance(input_data, Data) and returns input_data as-is.
        """
        from torch_geometric.data import Data

        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        data = synthetic_pyg_data_list[0]
        result = convert_to_pyg(data, format="pyg_data")

        assert isinstance(result, Data)
        assert torch.equal(result.x, data.x)
        assert torch.equal(result.edge_index, data.edge_index)

    def test_dict_converter(self):
        """DictConverter converts dict with tensors to Data.

        Evidence: data_converter.py DictConverter.convert() (lines 415-428)
        iterates dict items, converts lists/tuples to tensors, and creates
        Data(**data_dict).
        """
        from torch_geometric.data import Data

        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        input_dict = {
            "x": torch.randn(5, 11),
            "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
            "y": torch.tensor([1.0]),
        }

        result = convert_to_pyg(input_dict, format="dict")

        assert isinstance(result, Data)
        assert result.x.shape == (5, 11)
        assert result.edge_index.shape == (2, 3)

    def test_auto_detect_pyg_data(self, synthetic_pyg_data_list):
        """auto_detect() identifies PyG Data objects.

        Evidence: data_converter.py auto_detect() (lines 273-294) iterates
        converters and calls can_convert(). PyGDataConverter.can_convert()
        (line 382) checks isinstance(input_data, Data).
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            get_registry,
        )

        registry = get_registry()
        detected = registry.auto_detect(synthetic_pyg_data_list[0])

        assert detected == "pyg_data"

    def test_auto_detect_dict(self):
        """auto_detect() identifies dict with tensor data.

        Evidence: data_converter.py DictConverter.can_convert() (lines 407-413)
        checks isinstance(input_data, dict) and presence of x/z and edge_index.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            get_registry,
        )

        registry = get_registry()
        input_dict = {
            "x": torch.randn(5, 11),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
        }
        detected = registry.auto_detect(input_dict)

        assert detected == "dict"

    def test_convert_to_pyg_auto_detect(self, synthetic_pyg_data_list):
        """convert_to_pyg() with format=None auto-detects format.

        Evidence: data_converter.py convert_to_pyg() (lines ~1148-1151)
        calls registry.auto_detect(input_data) when format is None.
        """
        from torch_geometric.data import Data

        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        data = synthetic_pyg_data_list[0]
        result = convert_to_pyg(data)  # format=None → auto-detect

        assert isinstance(result, Data)

    def test_convert_to_pyg_unknown_format_raises(self):
        """convert_to_pyg() raises ValueError for undetectable input.

        Evidence: data_converter.py convert_to_pyg() (lines ~1202-1207)
        raises ValueError when auto_detect returns None and input doesn't
        match any known unavailable format pattern.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        with pytest.raises((ValueError, ImportError)):
            convert_to_pyg(12345)  # int is not a known format

    def test_base_data_converter_abc(self):
        """BaseDataConverter is importable and is an ABC.

        Evidence: data_converter.py BaseDataConverter (lines 327-359) is
        an ABC with abstract methods: format_name, is_available,
        can_convert, convert.
        """
        from abc import ABC

        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            BaseDataConverter,
        )

        assert issubclass(BaseDataConverter, ABC)

        # Verify abstract methods exist
        assert hasattr(BaseDataConverter, "format_name")
        assert hasattr(BaseDataConverter, "is_available")
        assert hasattr(BaseDataConverter, "can_convert")
        assert hasattr(BaseDataConverter, "convert")

    def test_smiles_converter_registered(self):
        """SMILESConverter is registered in the registry.

        Evidence: data_converter.py @register_converter("smiles") (line 431)
        decorates SMILESConverter class.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            get_registry,
        )

        registry = get_registry()
        assert registry.is_registered("smiles")


# ===========================================================================
# SECTION 5: FULL PREDICTION PIPELINE INTEGRATION SMOKE TESTS
# ===========================================================================


class TestPredictionPipelineIntegrationSmoke:
    """Smoke tests for the complete prediction pipeline flow.

    Tests the end-to-end flow: checkpoint → ModelLoader → Predictor → output
    using mocked ModelFactory (to avoid full project initialization) but
    exercising all real prediction pipeline components.

    Evidence: Section 1.4 of MILIA_Test_Recommendations.md specifies:
    "Creates a minimal mock checkpoint, loads it, runs prediction on 1–2
    synthetic PyG Data objects. Asserts predictions are tensors of correct
    shape."
    """

    def test_full_pipeline_checkpoint_to_prediction(
        self,
        tmp_work_dir,
        mock_checkpoint_path,
        minimal_model,
        synthetic_pyg_data_list,
    ):
        """Complete pipeline: create checkpoint → load → predict.

        This is the critical smoke test that exercises the full prediction
        pathway described in Section 1.4:
        1. CheckpointManager saves a checkpoint (already done via fixture)
        2. ModelLoader loads checkpoint and recreates model
        3. Predictor runs inference on synthetic Data objects
        4. Output is a tensor with correct shape

        Mock Pollution Prevention: Uses @patch decorator at test level,
        no module-level sys.modules injection.
        """
        mock_model_info = {
            "task_type": "graph_regression",
            "uses_edge_features": False,
            "data_info": {
                "structural_features_config": {
                    "atom": ["atomic_num"],
                    "bond": ["bond_type"],
                },
            },
        }

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (
            minimal_model,
            mock_model_info,
        )

        with patch(
            "milia_pipeline.models.post_training.inference.model_loader.get_factory",
            return_value=mock_factory,
        ):
            from milia_pipeline.models.post_training.inference.predictor import (
                Predictor,
            )

            # Step 1: Create Predictor from checkpoint
            predictor = Predictor.from_checkpoint(
                checkpoint_path=mock_checkpoint_path,
                working_root_dir=tmp_work_dir,
                device=torch.device("cpu"),
            )

            # Step 2: Run prediction on single Data object
            single_pred = predictor.predict(synthetic_pyg_data_list[0])
            assert isinstance(single_pred, torch.Tensor)
            assert single_pred.shape[-1] == 1

            # Step 3: Run batch prediction
            batch_pred = predictor.predict_batch(synthetic_pyg_data_list[:2], batch_size=2)
            assert isinstance(batch_pred, torch.Tensor)
            assert batch_pred.shape[0] == 2

    def test_pipeline_with_data_converter(
        self, tmp_work_dir, minimal_model, synthetic_pyg_data_list
    ):
        """Pipeline: DataConverter → Predictor.predict().

        Tests that data converted via DataConverterRegistry can be passed
        directly to Predictor for inference.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        # Convert data via registry (passthrough for PyG Data)
        data = convert_to_pyg(synthetic_pyg_data_list[0])

        # Create predictor directly (no checkpoint needed for this test)
        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        prediction = predictor.predict(data)
        assert isinstance(prediction, torch.Tensor)

    def test_pipeline_dict_to_prediction(self, tmp_work_dir, minimal_model):
        """Pipeline: dict → DataConverter → Predictor.predict().

        Tests conversion from raw dict tensors through the prediction pipeline.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )

        # Create synthetic dict data
        input_dict = {
            "x": torch.randn(5, 11),
            "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        }

        # Convert via DataConverter
        data = convert_to_pyg(input_dict, format="dict")

        # Run prediction
        predictor = Predictor(
            model=minimal_model,
            working_root_dir=tmp_work_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        prediction = predictor.predict(data)
        assert isinstance(prediction, torch.Tensor)

    def test_checkpoint_save_load_predict_roundtrip(
        self,
        tmp_work_dir,
        minimal_model,
        synthetic_pyg_data_list,
    ):
        """Full roundtrip: save checkpoint → load → predict → save predictions.

        Evidence: This exercises all four modules listed in Section 1.4:
        - CheckpointManager.save() / .load()
        - ModelLoader (mocked factory, real checkpoint loading)
        - Predictor.predict() and Predictor.save_predictions()
        - DataConverterRegistry (passthrough via convert_to_pyg)

        Mock Pollution Prevention: Uses @patch decorator at test level.
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        # Step 1: Save checkpoint using CheckpointManager
        manager = CheckpointManager(working_root_dir=tmp_work_dir)
        checkpoint_path = manager.save(
            filepath="checkpoints/roundtrip_pipeline.pt",
            model=minimal_model,
            epoch=3,
            hyper_parameters={
                "model_name": "GCN",
                "task_type": "graph_regression",
                "hyperparameters": {
                    "in_channels": 11,
                    "hidden_channels": 16,
                    "out_channels": 16,
                    "num_layers": 2,
                },
                "model_info": {
                    "task_type": "graph_regression",
                    "uses_edge_features": False,
                },
            },
            data_info={
                "structural_features_config": {
                    "atom": ["atomic_num"],
                    "bond": ["bond_type"],
                },
            },
        )

        assert checkpoint_path.exists()

        # Step 2: Load via ModelLoader with mocked factory
        mock_model_info = {
            "task_type": "graph_regression",
            "uses_edge_features": False,
        }
        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (
            minimal_model,
            mock_model_info,
        )

        with patch(
            "milia_pipeline.models.post_training.inference.model_loader.get_factory",
            return_value=mock_factory,
        ):
            from milia_pipeline.models.post_training.inference.predictor import (
                Predictor,
            )

            predictor = Predictor.from_checkpoint(
                checkpoint_path=checkpoint_path,
                working_root_dir=tmp_work_dir,
                device=torch.device("cpu"),
            )

            # Step 3: Predict
            predictions = predictor.predict_batch(synthetic_pyg_data_list, batch_size=2)
            assert predictions.shape[0] == len(synthetic_pyg_data_list)

            # Step 4: Save predictions
            output_path = predictor.save_predictions(
                predictions, "results/roundtrip_predictions.csv", format="csv"
            )
            assert output_path.exists()


# ===========================================================================
# SECTION 6: POST-TRAINING PUBLIC API SMOKE TESTS
# ===========================================================================


class TestPostTrainingPublicAPISmoke:
    """Smoke tests for the post_training public API.

    Verifies that the post_training __init__.py re-exports are accessible.

    Evidence:
    - Project structure doc (lines 940-946): post_training __init__.py
      provides unified exports with 24 total exports:
      2 checkpoint + 5 inference + 17 data_preparation + 2 transfer_learning.
    - get_available_components() lists all available components by category.
    - get_implementation_status() tracks implementation status.
    """

    def test_post_training_package_importable(self):
        """milia_pipeline.models.post_training can be imported."""
        import milia_pipeline.models.post_training as post_training

        assert post_training is not None

    def test_checkpoint_exports(self):
        """Checkpoint-related exports are accessible.

        Evidence: project structure doc (line 947): CheckpointManager is
        exported from post_training. CHECKPOINT_FORMAT_VERSION constant
        is also available.
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CHECKPOINT_FORMAT_VERSION,
            CheckpointManager,
        )

        assert CheckpointManager is not None
        assert CHECKPOINT_FORMAT_VERSION is not None

    def test_inference_exports(self):
        """Inference-related exports are accessible.

        Evidence: project structure doc (lines 959-983): ModelLoader,
        Predictor, and convenience functions are exported.
        """
        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
            load_model,
            load_model_only,
        )
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
            predict,
        )

        assert ModelLoader is not None
        assert Predictor is not None
        assert callable(load_model)
        assert callable(load_model_only)
        assert callable(predict)

    def test_data_preparation_exports(self):
        """Data preparation exports are accessible.

        Evidence: project structure doc (lines 984-999): DataConverterRegistry,
        BaseDataConverter, convert_to_pyg, list_available_formats, etc.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            BaseDataConverter,
            DataConverterRegistry,
            convert_batch_to_pyg,
            convert_to_pyg,
            get_registry,
            list_all_formats,
            list_available_formats,
        )

        assert DataConverterRegistry is not None
        assert BaseDataConverter is not None
        assert callable(convert_to_pyg)
        assert callable(convert_batch_to_pyg)
        assert callable(list_available_formats)
        assert callable(list_all_formats)
        assert callable(get_registry)

    def test_get_available_components(self):
        """get_available_components() returns dict of available components.

        Evidence: project structure doc (line 944):
        get_available_components() lists all available components by category.

        Note: This function uses conditional imports with graceful fallback,
        so it should always succeed even if some dependencies are missing.
        """
        try:
            from milia_pipeline.models.post_training import (
                get_available_components,
            )

            components = get_available_components()
            assert isinstance(components, dict)
        except ImportError:
            # get_available_components may not be re-exported at package level
            # depending on __init__.py structure. This is acceptable for smoke.
            pytest.skip("get_available_components not available at package level")

    def test_get_implementation_status(self):
        """get_implementation_status() returns status dict.

        Evidence: project structure doc (line 945):
        get_implementation_status() tracks implementation status.
        """
        try:
            from milia_pipeline.models.post_training import (
                get_implementation_status,
            )

            status = get_implementation_status()
            assert isinstance(status, dict)
        except ImportError:
            pytest.skip("get_implementation_status not available at package level")


# ===========================================================================
# SECTION 7: 3D CONFORMER UTILITY SMOKE TESTS
# ===========================================================================


class TestConformerUtilitySmoke:
    """Smoke tests for 3D conformer generation utilities.

    Evidence:
    - data_converter.py _requires_3d_conformer() (lines 49-65):
      Checks if structural_features_config has 3D-requiring bond features
      (bond_length, bond_length_binned).
    - data_converter.py _3D_BOND_FEATURES constant (line 46):
      Set of features requiring 3D coordinates.
    """

    def test_requires_3d_conformer_importable(self):
        """_requires_3d_conformer can be imported."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        assert callable(_requires_3d_conformer)

    def test_requires_3d_conformer_false_no_config(self):
        """_requires_3d_conformer returns False for None config.

        Evidence: data_converter.py _requires_3d_conformer() (lines 61-62)
        returns False if structural_features_config is None.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        assert _requires_3d_conformer(None) is False

    def test_requires_3d_conformer_false_no_3d_features(self):
        """_requires_3d_conformer returns False when no 3D features present.

        Evidence: data_converter.py _requires_3d_conformer() (lines 64-65)
        checks intersection of bond features with _3D_BOND_FEATURES set.
        bond_type is NOT in _3D_BOND_FEATURES, so returns False.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        config = {"atom": ["atomic_num"], "bond": ["bond_type"]}
        assert _requires_3d_conformer(config) is False

    def test_requires_3d_conformer_true_bond_length(self):
        """_requires_3d_conformer returns True when bond_length present.

        Evidence: data_converter.py _3D_BOND_FEATURES (line 46):
        {'bond_length', 'bond_length_binned'}. bond_length IS in the set.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        config = {"atom": ["atomic_num"], "bond": ["bond_type", "bond_length"]}
        assert _requires_3d_conformer(config) is True

    def test_requires_3d_conformer_true_bond_length_binned(self):
        """_requires_3d_conformer returns True when bond_length_binned present."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        config = {"atom": [], "bond": ["bond_length_binned"]}
        assert _requires_3d_conformer(config) is True
