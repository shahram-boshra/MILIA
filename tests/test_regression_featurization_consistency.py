"""
Test Suite: Regression — Featurization Consistency (Section 4.3)

Validates the structural features configuration persistence fix (v1.6.0):
Training-time featurization config is saved in checkpoints and correctly
applied during prediction to avoid feature dimension mismatches.

Modules exercised:
- milia_pipeline/models/training/callbacks.py
    → ModelCheckpoint._save_checkpoint() saves structural_features_config in data_info
- milia_pipeline/models/post_training/inference/predictor.py
    → Predictor.structural_features_config property
- milia_pipeline/models/post_training/data_preparation/data_converter.py
    → convert_to_pyg(..., structural_features_config=)
    → _apply_structural_features_if_available()
    → _requires_3d_conformer()
    → _ensure_3d_conformer_for_prediction()
- milia_pipeline/molecules/mol_structural_features.py
    → add_structural_features()
    → get_available_features()

Test Design:
- Tests use @patch decorators for mocking (no sys.modules pollution)
- All fixtures are test-scoped or session-scoped with proper cleanup
- Synthetic data and mock checkpoints avoid filesystem/model dependencies
- Each test class targets a specific module boundary in the featurization chain

Author: MILIA Team
Version: 1.0.0
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Pytest marker: all tests in this module are regression tests
pytestmark = pytest.mark.regression

# ---------------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES — Synthetic Data, Configs, and Mock Objects
# =============================================================================


@pytest.fixture
def sample_structural_features_config_atom_only():
    """Structural features config with only atom features (no 3D requirements)."""
    return {
        "atom": ["degree", "hybridization", "is_aromatic"],
        "bond": [],
    }


@pytest.fixture
def sample_structural_features_config_atom_and_bond():
    """Structural features config with both atom and 2D bond features."""
    return {
        "atom": ["degree", "hybridization", "is_aromatic", "is_in_ring"],
        "bond": ["bond_type", "is_conjugated", "is_in_any_ring"],
    }


@pytest.fixture
def sample_structural_features_config_with_3d():
    """Structural features config requiring 3D conformer (bond_length)."""
    return {
        "atom": ["degree", "hybridization"],
        "bond": ["bond_type", "bond_length"],
    }


@pytest.fixture
def sample_structural_features_config_with_3d_binned():
    """Structural features config requiring 3D conformer (bond_length_binned)."""
    return {
        "atom": ["degree"],
        "bond": ["bond_type", "bond_length_binned"],
    }


@pytest.fixture
def empty_structural_features_config():
    """Structural features config with no features configured."""
    return {
        "atom": [],
        "bond": [],
    }


@pytest.fixture
def full_structural_features_config():
    """Structural features config with all available features."""
    return {
        "atom": [
            "degree",
            "total_degree",
            "hybridization",
            "total_valence",
            "is_aromatic",
            "is_in_ring",
            "partial_charge",
            "num_aromatic_bonds",
            "chirality",
        ],
        "bond": [
            "bond_type",
            "is_conjugated",
            "is_aromatic",
            "is_in_any_ring",
            "stereo",
        ],
    }


@pytest.fixture
def mock_model_info_with_structural_config(sample_structural_features_config_atom_and_bond):
    """model_info dict as it would appear after ModelLoader extracts from checkpoint."""
    return {
        "name": "GCN",
        "task_type": "graph_regression",
        "data_info": {
            "requires_edge_features": False,
            "uses_edge_features": False,
            "structural_features_config": sample_structural_features_config_atom_and_bond,
        },
        "hyperparameters_values": {
            "hidden_channels": 64,
            "num_layers": 3,
        },
    }


@pytest.fixture
def mock_model_info_without_structural_config():
    """model_info dict without structural_features_config (pre-v1.6.0 checkpoint)."""
    return {
        "name": "GCN",
        "task_type": "graph_regression",
        "data_info": {
            "requires_edge_features": False,
            "uses_edge_features": False,
        },
        "hyperparameters_values": {
            "hidden_channels": 64,
        },
    }


@pytest.fixture
def mock_model_info_empty_structural_config():
    """model_info dict with empty structural_features_config."""
    return {
        "name": "GCN",
        "task_type": "graph_regression",
        "data_info": {
            "requires_edge_features": False,
            "uses_edge_features": False,
            "structural_features_config": {},
        },
    }


@pytest.fixture
def synthetic_pyg_data():
    """Minimal synthetic PyG Data object representing a simple molecule (ethanol-like)."""
    from torch_geometric.data import Data

    num_atoms = 3
    # Node features: [atomic_num_one_hot (simplified)]
    x = torch.tensor([[6.0], [6.0], [8.0]], dtype=torch.float)
    # Edges: fully connected bidirectional for 3 atoms
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
        dtype=torch.long,
    )
    # Positions (3D coordinates)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [2.31, 0.97, 0.0]],
        dtype=torch.float,
    )

    data = Data(x=x, edge_index=edge_index, pos=pos)
    data.smiles = "CCO"
    return data


@pytest.fixture
def synthetic_pyg_data_no_smiles():
    """Synthetic PyG Data without smiles attribute (XYZ/dict origin)."""
    from torch_geometric.data import Data

    x = torch.tensor([[6.0], [8.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def mock_trainer(tmp_path, sample_structural_features_config_atom_and_bond):
    """
    Mock Trainer object with the attributes accessed by ModelCheckpoint._save_checkpoint().

    This mock replicates the exact attribute structure the real Trainer exposes,
    as verified from callbacks.py lines 535-608.
    """
    trainer = MagicMock()
    trainer.global_step = 100
    trainer.current_epoch = 5
    trainer.best_val_loss = 0.0234

    # Mock model state
    simple_model = torch.nn.Linear(10, 1)
    trainer.model = simple_model
    trainer.optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    trainer.scheduler = None

    # Metrics history as defaultdict (as used by Trainer)
    trainer.metrics_history = defaultdict(
        list,
        {
            "train_loss": [0.5, 0.3, 0.2, 0.15, 0.1],
            "val_loss": [0.6, 0.4, 0.25, 0.18, 0.12],
        },
    )

    # model_info with structural_features_config — the key payload for FIX 17
    trainer.model_info = {
        "name": "GCN",
        "task_type": "graph_regression",
        "hyperparameters_values": {"hidden_channels": 64, "num_layers": 3},
        "requires_edge_features": False,
        "uses_edge_features": False,
        "structural_features_config": sample_structural_features_config_atom_and_bond,
        "wrapper_info": {},
        "target_selection": None,
    }

    return trainer


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint output."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


# =============================================================================
# CLASS 1: ModelCheckpoint saves structural_features_config (callbacks.py)
# =============================================================================


class TestModelCheckpointSavesStructuralFeaturesConfig:
    """
    Verify ModelCheckpoint._save_checkpoint() persists structural_features_config
    in checkpoint['data_info'] (FIX 17, callbacks.py lines 581-592).

    This is the FIRST link in the featurization consistency chain:
    Training → Checkpoint → structural_features_config persisted.
    """

    def test_checkpoint_contains_data_info_key(self, mock_trainer, checkpoint_dir):
        """Saved checkpoint must contain 'data_info' top-level key."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        assert "data_info" in checkpoint, "Checkpoint missing 'data_info' key — FIX 17 regression"

    def test_data_info_contains_structural_features_config(self, mock_trainer, checkpoint_dir):
        """data_info must contain 'structural_features_config' sub-key."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        data_info = checkpoint["data_info"]
        assert "structural_features_config" in data_info, (
            "data_info missing 'structural_features_config' — FIX 17 regression"
        )

    def test_structural_features_config_matches_trainer_model_info(
        self,
        mock_trainer,
        checkpoint_dir,
        sample_structural_features_config_atom_and_bond,
    ):
        """Saved structural_features_config must be identical to trainer.model_info value."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        saved_config = checkpoint["data_info"]["structural_features_config"]

        assert saved_config == sample_structural_features_config_atom_and_bond, (
            f"structural_features_config mismatch!\n"
            f"Expected: {sample_structural_features_config_atom_and_bond}\n"
            f"Got:      {saved_config}"
        )

    def test_structural_features_config_atom_list_preserved_exactly(
        self, mock_trainer, checkpoint_dir
    ):
        """Atom feature list order and contents must be preserved byte-for-byte."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        saved_atom = checkpoint["data_info"]["structural_features_config"]["atom"]
        expected_atom = mock_trainer.model_info["structural_features_config"]["atom"]
        assert saved_atom == expected_atom, (
            f"Atom features order/content changed.\nExpected: {expected_atom}\nGot: {saved_atom}"
        )

    def test_structural_features_config_bond_list_preserved_exactly(
        self, mock_trainer, checkpoint_dir
    ):
        """Bond feature list order and contents must be preserved byte-for-byte."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        saved_bond = checkpoint["data_info"]["structural_features_config"]["bond"]
        expected_bond = mock_trainer.model_info["structural_features_config"]["bond"]
        assert saved_bond == expected_bond, (
            f"Bond features order/content changed.\nExpected: {expected_bond}\nGot: {saved_bond}"
        )

    def test_data_info_contains_edge_feature_flags(self, mock_trainer, checkpoint_dir):
        """data_info must also contain requires_edge_features and uses_edge_features."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        data_info = checkpoint["data_info"]
        assert "requires_edge_features" in data_info
        assert "uses_edge_features" in data_info

    def test_checkpoint_format_version_is_2_0(self, mock_trainer, checkpoint_dir):
        """Checkpoint must use format version 2.0 (which includes data_info)."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        assert "version_info" in checkpoint
        assert checkpoint["version_info"]["checkpoint_format_version"] == "2.0"

    def test_empty_structural_config_when_model_info_missing(self, checkpoint_dir):
        """When trainer has no model_info, structural_features_config defaults to {}."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        trainer = MagicMock()
        trainer.global_step = 0
        trainer.current_epoch = 0
        trainer.best_val_loss = float("inf")
        trainer.model = torch.nn.Linear(10, 1)
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters())
        trainer.scheduler = None
        trainer.metrics_history = defaultdict(list)
        # No model_info attribute
        trainer.model_info = None
        del trainer.model_info

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(trainer, ckpt_path, epoch=0, score=1.0)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        data_info = checkpoint["data_info"]
        assert data_info["structural_features_config"] == {}

    def test_empty_structural_config_when_model_info_is_empty_dict(self, checkpoint_dir):
        """When trainer.model_info is empty dict, structural_features_config defaults to {}."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        trainer = MagicMock()
        trainer.global_step = 0
        trainer.current_epoch = 0
        trainer.best_val_loss = float("inf")
        trainer.model = torch.nn.Linear(10, 1)
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters())
        trainer.scheduler = None
        trainer.metrics_history = defaultdict(list)
        trainer.model_info = {}

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(trainer, ckpt_path, epoch=0, score=1.0)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        assert checkpoint["data_info"]["structural_features_config"] == {}

    def test_best_checkpoint_also_contains_structural_features_config(
        self, mock_trainer, checkpoint_dir
    ):
        """best.pt checkpoint (is_best=True) must also persist structural_features_config."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        best_path = checkpoint_dir / "best.pt"
        mc._save_checkpoint(mock_trainer, best_path, epoch=5, score=0.12, is_best=True)

        checkpoint = torch.load(best_path, weights_only=False)
        assert "data_info" in checkpoint
        assert "structural_features_config" in checkpoint["data_info"]
        assert checkpoint["is_best"] is True

    def test_hyper_parameters_also_saved_alongside_data_info(self, mock_trainer, checkpoint_dir):
        """hyper_parameters section coexists with data_info in the same checkpoint."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "test_checkpoint.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        assert "hyper_parameters" in checkpoint
        assert "data_info" in checkpoint
        # Both must be non-empty
        assert checkpoint["hyper_parameters"]
        assert checkpoint["data_info"]


# =============================================================================
# CLASS 2: Predictor.structural_features_config property (predictor.py)
# =============================================================================


class TestPredictorStructuralFeaturesConfigProperty:
    """
    Verify Predictor.structural_features_config property (FIX 19)
    correctly reads from model_info['data_info']['structural_features_config'].

    This is the SECOND link: Checkpoint → Predictor → config accessible.
    """

    def test_property_returns_config_when_present(
        self, mock_model_info_with_structural_config, tmp_path
    ):
        """Property should return the config dict when present in model_info."""
        from milia_pipeline.models.post_training.inference.predictor import Predictor

        model = torch.nn.Linear(10, 1)
        predictor = Predictor(
            model=model,
            working_root_dir=tmp_path,
            model_info=mock_model_info_with_structural_config,
        )

        config = predictor.structural_features_config
        assert config is not None
        assert "atom" in config
        assert "bond" in config

    def test_property_returns_exact_config_values(
        self,
        mock_model_info_with_structural_config,
        sample_structural_features_config_atom_and_bond,
        tmp_path,
    ):
        """Property must return the exact same dict that was embedded in model_info."""
        from milia_pipeline.models.post_training.inference.predictor import Predictor

        model = torch.nn.Linear(10, 1)
        predictor = Predictor(
            model=model,
            working_root_dir=tmp_path,
            model_info=mock_model_info_with_structural_config,
        )

        config = predictor.structural_features_config
        assert config == sample_structural_features_config_atom_and_bond

    def test_property_returns_none_when_model_info_empty(self, tmp_path):
        """Property should return None when model_info is empty dict."""
        from milia_pipeline.models.post_training.inference.predictor import Predictor

        model = torch.nn.Linear(10, 1)
        predictor = Predictor(
            model=model,
            working_root_dir=tmp_path,
            model_info={},
        )

        assert predictor.structural_features_config is None

    def test_property_returns_none_when_model_info_is_none(self, tmp_path):
        """Property should return None when model_info is None (pre-v1.6.0 path)."""
        from milia_pipeline.models.post_training.inference.predictor import Predictor

        model = torch.nn.Linear(10, 1)
        predictor = Predictor(
            model=model,
            working_root_dir=tmp_path,
            model_info=None,
        )

        assert predictor.structural_features_config is None

    def test_property_returns_none_when_data_info_missing(self, tmp_path):
        """Property should return None when data_info key is absent from model_info."""
        from milia_pipeline.models.post_training.inference.predictor import Predictor

        model_info = {"name": "GCN", "task_type": "graph_regression"}
        model = torch.nn.Linear(10, 1)
        predictor = Predictor(
            model=model,
            working_root_dir=tmp_path,
            model_info=model_info,
        )

        assert predictor.structural_features_config is None

    def test_property_returns_none_when_structural_config_missing_from_data_info(
        self, mock_model_info_without_structural_config, tmp_path
    ):
        """
        Property should return None when data_info exists but has no
        structural_features_config key (backward compatibility).
        """
        from milia_pipeline.models.post_training.inference.predictor import Predictor

        model = torch.nn.Linear(10, 1)
        predictor = Predictor(
            model=model,
            working_root_dir=tmp_path,
            model_info=mock_model_info_without_structural_config,
        )

        assert predictor.structural_features_config is None

    def test_property_with_empty_structural_config(
        self, mock_model_info_empty_structural_config, tmp_path
    ):
        """
        Property should return {} when structural_features_config is empty dict.
        This is distinct from None (which means 'not present').
        """
        from milia_pipeline.models.post_training.inference.predictor import Predictor

        model = torch.nn.Linear(10, 1)
        predictor = Predictor(
            model=model,
            working_root_dir=tmp_path,
            model_info=mock_model_info_empty_structural_config,
        )

        config = predictor.structural_features_config
        assert config == {}

    def test_model_info_stored_on_predictor_instance(
        self, mock_model_info_with_structural_config, tmp_path
    ):
        """model_info must be accessible as predictor.model_info attribute."""
        from milia_pipeline.models.post_training.inference.predictor import Predictor

        model = torch.nn.Linear(10, 1)
        predictor = Predictor(
            model=model,
            working_root_dir=tmp_path,
            model_info=mock_model_info_with_structural_config,
        )

        assert predictor.model_info == mock_model_info_with_structural_config


# =============================================================================
# CLASS 3: convert_to_pyg with structural_features_config (data_converter.py)
# =============================================================================


class TestConvertToPygWithStructuralFeaturesConfig:
    """
    Verify convert_to_pyg() applies structural_features_config via
    _apply_structural_features_if_available() post-processing (FIX 20).

    This is the THIRD link: Predictor config → convert_to_pyg → features applied.
    """

    def test_convert_to_pyg_accepts_structural_features_config_parameter(self):
        """convert_to_pyg() must accept structural_features_config keyword arg."""
        import inspect

        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        sig = inspect.signature(convert_to_pyg)
        assert "structural_features_config" in sig.parameters, (
            "convert_to_pyg() missing 'structural_features_config' parameter — FIX 20 regression"
        )

    def test_pyg_data_passthrough_without_structural_config(self, synthetic_pyg_data):
        """PyG Data passthrough should work when structural_features_config is None."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        result = convert_to_pyg(synthetic_pyg_data, format="pyg_data")
        assert result is synthetic_pyg_data

    @patch(
        "milia_pipeline.models.post_training.data_preparation.data_converter."
        "_apply_structural_features_if_available"
    )
    def test_structural_features_post_processing_called_when_config_provided(
        self,
        mock_apply,
        synthetic_pyg_data,
        sample_structural_features_config_atom_and_bond,
    ):
        """
        When structural_features_config is non-None, _apply_structural_features_if_available
        must be called as post-processing step.
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        mock_apply.return_value = synthetic_pyg_data

        convert_to_pyg(
            synthetic_pyg_data,
            format="pyg_data",
            structural_features_config=sample_structural_features_config_atom_and_bond,
        )

        mock_apply.assert_called_once_with(
            synthetic_pyg_data,
            sample_structural_features_config_atom_and_bond,
        )

    @patch(
        "milia_pipeline.models.post_training.data_preparation.data_converter."
        "_apply_structural_features_if_available"
    )
    def test_structural_features_not_applied_when_config_is_none(
        self, mock_apply, synthetic_pyg_data
    ):
        """When structural_features_config is None, post-processing must NOT be called."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        convert_to_pyg(
            synthetic_pyg_data,
            format="pyg_data",
            structural_features_config=None,
        )

        mock_apply.assert_not_called()

    @patch(
        "milia_pipeline.models.post_training.data_preparation.data_converter."
        "_apply_structural_features_if_available"
    )
    def test_structural_features_not_applied_when_config_is_falsy(
        self, mock_apply, synthetic_pyg_data
    ):
        """When structural_features_config is empty dict (falsy), post-processing is skipped."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        convert_to_pyg(
            synthetic_pyg_data,
            format="pyg_data",
            structural_features_config={},
        )

        # Empty dict is falsy, so _apply should NOT be called
        mock_apply.assert_not_called()


# =============================================================================
# CLASS 4: _apply_structural_features_if_available (data_converter.py)
# =============================================================================


class TestApplyStructuralFeaturesIfAvailable:
    """
    Verify _apply_structural_features_if_available() correctly:
    1. Skips when config is None or has no features
    2. Reconstructs RDKit mol from SMILES/InChI stored in Data
    3. Calls add_structural_features() with correct arguments
    4. Returns original data on failure (graceful fallback)
    """

    def test_returns_original_data_when_config_is_none(self, synthetic_pyg_data):
        """Must return original data unchanged when config is None."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _apply_structural_features_if_available,
        )

        result = _apply_structural_features_if_available(synthetic_pyg_data, None)
        assert result is synthetic_pyg_data

    def test_returns_original_data_when_config_has_no_features(
        self, synthetic_pyg_data, empty_structural_features_config
    ):
        """Must return original data when both atom and bond lists are empty."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _apply_structural_features_if_available,
        )

        result = _apply_structural_features_if_available(
            synthetic_pyg_data, empty_structural_features_config
        )
        assert result is synthetic_pyg_data

    @patch("milia_pipeline.molecules.mol_structural_features.add_structural_features")
    def test_calls_add_structural_features_with_correct_config(
        self,
        mock_add_features,
        synthetic_pyg_data,
        sample_structural_features_config_atom_and_bond,
    ):
        """Must call add_structural_features with the exact config dict."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _apply_structural_features_if_available,
        )

        mock_add_features.return_value = synthetic_pyg_data

        _apply_structural_features_if_available(
            synthetic_pyg_data,
            sample_structural_features_config_atom_and_bond,
        )

        mock_add_features.assert_called_once()
        call_kwargs = mock_add_features.call_args
        # Verify feature_config argument
        assert (
            call_kwargs.kwargs.get("feature_config")
            == sample_structural_features_config_atom_and_bond
            or call_kwargs[1].get("feature_config")
            == sample_structural_features_config_atom_and_bond
            or (
                len(call_kwargs[0]) >= 3
                and call_kwargs[0][2] == sample_structural_features_config_atom_and_bond
            )
        )

    def test_returns_original_data_when_no_smiles_or_inchi(
        self,
        synthetic_pyg_data_no_smiles,
        sample_structural_features_config_atom_and_bond,
    ):
        """
        Must return original data when Data has no smiles/inchi attribute
        (cannot reconstruct RDKit mol).
        """
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _apply_structural_features_if_available,
        )

        result = _apply_structural_features_if_available(
            synthetic_pyg_data_no_smiles,
            sample_structural_features_config_atom_and_bond,
        )
        # Should return original data without modification
        assert result is synthetic_pyg_data_no_smiles

    @patch(
        "milia_pipeline.molecules.mol_structural_features.add_structural_features",
        side_effect=Exception("RDKit processing error"),
    )
    def test_graceful_fallback_on_add_structural_features_failure(
        self,
        mock_add_features,
        synthetic_pyg_data,
        sample_structural_features_config_atom_and_bond,
    ):
        """Must return original data (not raise) when add_structural_features fails."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _apply_structural_features_if_available,
        )

        result = _apply_structural_features_if_available(
            synthetic_pyg_data,
            sample_structural_features_config_atom_and_bond,
        )
        # Graceful fallback: returns original data
        assert result is synthetic_pyg_data


# =============================================================================
# CLASS 5: 3D Conformer Detection (data_converter.py)
# =============================================================================


class TestRequires3dConformer:
    """
    Verify _requires_3d_conformer() correctly identifies configs that need
    3D conformer generation (FIX 21, data_converter.py lines 46-65).
    """

    def test_returns_false_for_none_config(self):
        """None config means no 3D requirement."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        assert _requires_3d_conformer(None) is False

    def test_returns_false_for_2d_only_bond_features(
        self, sample_structural_features_config_atom_and_bond
    ):
        """Config with only 2D bond features (bond_type, etc.) should return False."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        assert _requires_3d_conformer(sample_structural_features_config_atom_and_bond) is False

    def test_returns_true_for_bond_length_feature(self, sample_structural_features_config_with_3d):
        """Config with 'bond_length' in bond features must return True."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        assert _requires_3d_conformer(sample_structural_features_config_with_3d) is True

    def test_returns_true_for_bond_length_binned_feature(
        self, sample_structural_features_config_with_3d_binned
    ):
        """Config with 'bond_length_binned' in bond features must return True."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        assert _requires_3d_conformer(sample_structural_features_config_with_3d_binned) is True

    def test_returns_false_for_empty_bond_list(self, sample_structural_features_config_atom_only):
        """Config with empty bond feature list should return False."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        assert _requires_3d_conformer(sample_structural_features_config_atom_only) is False

    def test_returns_false_for_empty_config(self, empty_structural_features_config):
        """Config with both empty lists should return False."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        assert _requires_3d_conformer(empty_structural_features_config) is False

    def test_returns_false_for_config_without_bond_key(self):
        """Config dict missing 'bond' key should return False (no KeyError)."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        config = {"atom": ["degree"]}
        assert _requires_3d_conformer(config) is False


# =============================================================================
# CLASS 6: get_available_features (mol_structural_features.py)
# =============================================================================


class TestGetAvailableFeatures:
    """
    Verify get_available_features() returns the complete feature catalog.
    This is used for validation that configs reference valid feature names.
    """

    def test_returns_dict_with_atom_and_bond_keys(self):
        """Must return a dict with both 'atom' and 'bond' keys."""
        from milia_pipeline.molecules.mol_structural_features import (
            get_available_features,
        )

        features = get_available_features()
        assert isinstance(features, dict)
        assert "atom" in features
        assert "bond" in features

    def test_atom_features_include_core_features(self):
        """Atom features must include the core set used in structural_features_config."""
        from milia_pipeline.molecules.mol_structural_features import (
            get_available_features,
        )

        features = get_available_features()
        atom_features = features["atom"]
        core_atom = ["degree", "hybridization", "is_aromatic", "is_in_ring"]
        for feat in core_atom:
            assert feat in atom_features, f"Core atom feature '{feat}' missing from catalog"

    def test_bond_features_include_core_features(self):
        """Bond features must include the core set used in structural_features_config."""
        from milia_pipeline.molecules.mol_structural_features import (
            get_available_features,
        )

        features = get_available_features()
        bond_features = features["bond"]
        core_bond = ["bond_type", "is_conjugated", "is_in_any_ring"]
        for feat in core_bond:
            assert feat in bond_features, f"Core bond feature '{feat}' missing from catalog"

    def test_3d_bond_features_listed(self):
        """3D bond features (bond_length, bond_length_binned) must be in catalog."""
        from milia_pipeline.molecules.mol_structural_features import (
            get_available_features,
        )

        features = get_available_features()
        bond_features = features["bond"]
        assert "bond_length" in bond_features
        assert "bond_length_binned" in bond_features

    def test_all_atom_features_are_strings(self):
        """All atom feature names must be strings."""
        from milia_pipeline.molecules.mol_structural_features import (
            get_available_features,
        )

        features = get_available_features()
        for feat in features["atom"]:
            assert isinstance(feat, str), f"Non-string atom feature: {feat}"

    def test_all_bond_features_are_strings(self):
        """All bond feature names must be strings."""
        from milia_pipeline.molecules.mol_structural_features import (
            get_available_features,
        )

        features = get_available_features()
        for feat in features["bond"]:
            assert isinstance(feat, str), f"Non-string bond feature: {feat}"


# =============================================================================
# CLASS 7: End-to-End Featurization Chain Consistency
# =============================================================================


class TestEndToEndFeaturizationChain:
    """
    End-to-end regression test: Simulate the full chain from checkpoint save
    to prediction-time featurization to verify dimension consistency.

    Chain:  trainer.model_info  →  ModelCheckpoint._save_checkpoint()
            →  checkpoint on disk  →  Predictor.structural_features_config
            →  convert_to_pyg(structural_features_config=...)
    """

    def test_roundtrip_save_load_structural_features_config(
        self,
        mock_trainer,
        checkpoint_dir,
        sample_structural_features_config_atom_and_bond,
    ):
        """
        Full roundtrip: save checkpoint → load → extract config →
        verify config matches original training-time config.
        """
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "roundtrip_test.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        # Simulate what ModelLoader does: load checkpoint, extract data_info
        checkpoint = torch.load(ckpt_path, weights_only=False)
        loaded_data_info = checkpoint["data_info"]
        loaded_config = loaded_data_info["structural_features_config"]

        # Simulate what Predictor does: model_info = {..., 'data_info': loaded_data_info}
        model_info = {
            "name": "GCN",
            "data_info": loaded_data_info,
        }

        from milia_pipeline.models.post_training.inference.predictor import Predictor

        model = torch.nn.Linear(10, 1)
        predictor = Predictor(
            model=model,
            working_root_dir=checkpoint_dir,
            model_info=model_info,
        )

        # The final property must match the original training config
        assert (
            predictor.structural_features_config == sample_structural_features_config_atom_and_bond
        )

    def test_config_survives_multiple_checkpoint_saves(
        self,
        mock_trainer,
        checkpoint_dir,
        sample_structural_features_config_atom_and_bond,
    ):
        """Config must be identical across multiple checkpoint saves (epoch 1..N)."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")

        configs_from_checkpoints = []
        for epoch in range(5):
            ckpt_path = checkpoint_dir / f"epoch_{epoch}.pt"
            mc._save_checkpoint(mock_trainer, ckpt_path, epoch=epoch, score=0.5 - epoch * 0.1)
            checkpoint = torch.load(ckpt_path, weights_only=False)
            configs_from_checkpoints.append(checkpoint["data_info"]["structural_features_config"])

        # All 5 checkpoints must have identical structural_features_config
        for i, config in enumerate(configs_from_checkpoints):
            assert config == sample_structural_features_config_atom_and_bond, (
                f"Config changed at checkpoint {i}"
            )

    def test_config_deep_copy_independence(
        self,
        mock_trainer,
        checkpoint_dir,
        sample_structural_features_config_atom_and_bond,
    ):
        """
        Modifying the original config after saving must NOT affect saved checkpoint.
        Ensures checkpoint contains a copy, not a reference.
        """
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "independence_test.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=5, score=0.12)

        # Mutate the original config in trainer
        mock_trainer.model_info["structural_features_config"]["atom"].append("MUTATED")

        # Loaded config must NOT contain the mutation
        checkpoint = torch.load(ckpt_path, weights_only=False)
        saved_config = checkpoint["data_info"]["structural_features_config"]
        assert "MUTATED" not in saved_config.get("atom", [])


# =============================================================================
# CLASS 8: DataConverterRegistry integration with structural_features_config
# =============================================================================


class TestDataConverterRegistryIntegration:
    """
    Verify the DataConverterRegistry and convenience functions properly
    propagate structural_features_config through the conversion pipeline.
    """

    def test_convert_to_pyg_function_exists(self):
        """convert_to_pyg must be importable from data_converter module."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        assert callable(convert_to_pyg)

    def test_apply_structural_features_if_available_function_exists(self):
        """_apply_structural_features_if_available must be importable."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _apply_structural_features_if_available,
        )

        assert callable(_apply_structural_features_if_available)

    def test_requires_3d_conformer_function_exists(self):
        """_requires_3d_conformer must be importable."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _requires_3d_conformer,
        )

        assert callable(_requires_3d_conformer)

    def test_3d_bond_features_constant_exists(self):
        """_3D_BOND_FEATURES constant must be defined."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            _3D_BOND_FEATURES,
        )

        assert isinstance(_3D_BOND_FEATURES, set)
        assert "bond_length" in _3D_BOND_FEATURES
        assert "bond_length_binned" in _3D_BOND_FEATURES

    def test_dict_converter_works_without_structural_config(self):
        """DictConverter should work normally without structural_features_config."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        dict_data = {
            "x": torch.tensor([[1.0], [2.0]]),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
        }
        result = convert_to_pyg(dict_data, format="dict")
        assert result.x is not None
        assert result.edge_index is not None

    def test_pyg_data_converter_passthrough_preserves_smiles(self, synthetic_pyg_data):
        """PyG Data passthrough must preserve smiles attribute for later feature extraction."""
        from milia_pipeline.models.post_training.data_preparation.data_converter import (
            convert_to_pyg,
        )

        result = convert_to_pyg(synthetic_pyg_data, format="pyg_data")
        assert hasattr(result, "smiles")
        assert result.smiles == "CCO"


# =============================================================================
# CLASS 9: Structural Features Config Validation Helpers
# =============================================================================


class TestStructuralFeaturesConfigValidation:
    """
    Verify that configs used in tests reference only valid feature names
    from the get_available_features() catalog. This prevents test configs
    from drifting away from what the real system supports.
    """

    @pytest.fixture
    def available_features(self):
        """Get the complete catalog of available features."""
        from milia_pipeline.molecules.mol_structural_features import (
            get_available_features,
        )

        return get_available_features()

    def test_atom_only_config_uses_valid_features(
        self,
        sample_structural_features_config_atom_only,
        available_features,
    ):
        """All atom features in config must exist in the available catalog."""
        valid_atom = set(available_features["atom"])
        for feat in sample_structural_features_config_atom_only["atom"]:
            assert feat in valid_atom, f"Invalid atom feature in config: '{feat}'"

    def test_atom_and_bond_config_uses_valid_features(
        self,
        sample_structural_features_config_atom_and_bond,
        available_features,
    ):
        """All atom and bond features in config must exist in the available catalog."""
        valid_atom = set(available_features["atom"])
        valid_bond = set(available_features["bond"])
        for feat in sample_structural_features_config_atom_and_bond["atom"]:
            assert feat in valid_atom, f"Invalid atom feature: '{feat}'"
        for feat in sample_structural_features_config_atom_and_bond["bond"]:
            assert feat in valid_bond, f"Invalid bond feature: '{feat}'"

    def test_3d_config_uses_valid_features(
        self,
        sample_structural_features_config_with_3d,
        available_features,
    ):
        """3D config features must exist in the available catalog."""
        valid_atom = set(available_features["atom"])
        valid_bond = set(available_features["bond"])
        for feat in sample_structural_features_config_with_3d["atom"]:
            assert feat in valid_atom, f"Invalid atom feature: '{feat}'"
        for feat in sample_structural_features_config_with_3d["bond"]:
            assert feat in valid_bond, f"Invalid bond feature: '{feat}'"

    def test_full_config_uses_valid_features(
        self,
        full_structural_features_config,
        available_features,
    ):
        """Full config features must all exist in the available catalog."""
        valid_atom = set(available_features["atom"])
        valid_bond = set(available_features["bond"])
        for feat in full_structural_features_config["atom"]:
            assert feat in valid_atom, f"Invalid atom feature: '{feat}'"
        for feat in full_structural_features_config["bond"]:
            assert feat in valid_bond, f"Invalid bond feature: '{feat}'"


# =============================================================================
# CLASS 10: Checkpoint data_info Structure Integrity
# =============================================================================


class TestCheckpointDataInfoStructureIntegrity:
    """
    Verify the complete data_info structure within checkpoints maintains
    backward-compatible fields alongside the new structural_features_config.
    """

    def test_data_info_has_exactly_three_keys(self, mock_trainer, checkpoint_dir):
        """
        data_info must contain exactly: requires_edge_features,
        uses_edge_features, structural_features_config.
        """
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "structure_test.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=0, score=1.0)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        data_info = checkpoint["data_info"]
        expected_keys = {
            "requires_edge_features",
            "uses_edge_features",
            "structural_features_config",
        }
        assert set(data_info.keys()) == expected_keys, (
            f"data_info keys mismatch.\nExpected: {expected_keys}\nGot: {set(data_info.keys())}"
        )

    def test_requires_edge_features_is_bool(self, mock_trainer, checkpoint_dir):
        """requires_edge_features must be a boolean."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "type_test.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=0, score=1.0)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        assert isinstance(checkpoint["data_info"]["requires_edge_features"], bool)

    def test_uses_edge_features_is_bool(self, mock_trainer, checkpoint_dir):
        """uses_edge_features must be a boolean."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "type_test.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=0, score=1.0)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        assert isinstance(checkpoint["data_info"]["uses_edge_features"], bool)

    def test_structural_features_config_is_dict(self, mock_trainer, checkpoint_dir):
        """structural_features_config must be a dict."""
        from milia_pipeline.models.training.callbacks import ModelCheckpoint

        mc = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
        ckpt_path = checkpoint_dir / "type_test.pt"
        mc._save_checkpoint(mock_trainer, ckpt_path, epoch=0, score=1.0)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        assert isinstance(checkpoint["data_info"]["structural_features_config"], dict)
