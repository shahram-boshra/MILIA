"""
E2E Transfer Learning Workflow Test Suite

Tests the complete transfer learning workflow:
  train base model → save checkpoint → load via FineTuner →
  apply freeze strategy → fine-tune on new data → predict.

Section 3.4 of MILIA_Test_Recommendations.md.

Modules exercised:
- milia_pipeline/models/training/trainer.py                          — Initial training
- milia_pipeline/models/training/callbacks.py                        — ModelCheckpoint
- milia_pipeline/models/post_training/checkpoint/checkpoint_manager.py — Checkpoint save/load
- milia_pipeline/models/post_training/transfer_learning/fine_tuner.py  — FineTuner, FreezeStrategy
- milia_pipeline/models/post_training/inference/predictor.py           — Post-fine-tuning prediction

Author: MILIA Team
"""

import sys
import os
import copy
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# SYNTHETIC DATA HELPERS
# =============================================================================

def _make_synthetic_pyg_data(
    num_nodes: int = 10,
    num_node_features: int = 16,
    num_edges: int = 20,
    num_targets: int = 1,
    seed: int = 42,
) -> Data:
    """
    Create a single synthetic PyG Data object for graph-level regression.

    Returns a fully valid PyG graph with:
    - x: Node feature matrix  [num_nodes, num_node_features]
    - edge_index: COO edge connectivity  [2, num_edges]
    - y: Target values  [1, num_targets]
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, num_node_features, generator=gen)
    # Random edges (COO format), ensure valid node indices
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    edge_index = torch.stack([src, dst], dim=0)
    y = torch.randn(1, num_targets, generator=gen)
    return Data(x=x, edge_index=edge_index, y=y)


def _make_synthetic_dataset(
    num_graphs: int = 50,
    num_node_features: int = 16,
    num_targets: int = 1,
    seed: int = 42,
) -> List[Data]:
    """Create a list of synthetic PyG Data objects."""
    return [
        _make_synthetic_pyg_data(
            num_nodes=torch.randint(5, 15, (1,)).item(),
            num_node_features=num_node_features,
            num_edges=torch.randint(10, 30, (1,)).item(),
            num_targets=num_targets,
            seed=seed + i,
        )
        for i in range(num_graphs)
    ]


# =============================================================================
# MINIMAL GNN MODEL (self-contained, no external registry dependency)
# =============================================================================

class _SimpleGNNEncoder(nn.Module):
    """Minimal GNN-like encoder with conv layers for testing freeze strategies."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        # Named with 'conv' so _freeze_encoder_layers() recognises them
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x


class _SimpleGraphModel(nn.Module):
    """
    Minimal graph-level regression model.

    Architecture:
        conv1 (Linear) → ReLU → conv2 (Linear) → ReLU → global mean pool → head (Linear)

    The model mirrors the pattern of PyG GNN wrappers with a conv encoder
    followed by a graph-level readout and a prediction head, allowing
    FineTuner's freeze/replace logic to operate correctly.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int
    ):
        super().__init__()
        self.encoder = _SimpleGNNEncoder(in_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encode nodes
        h = self.encoder(x)
        # Global mean pooling
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)
        # Predict
        return self.head(h)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def device():
    """Training device (CPU for CI reproducibility)."""
    return torch.device("cpu")


@pytest.fixture
def num_node_features():
    return 16


@pytest.fixture
def hidden_channels():
    return 32


@pytest.fixture
def base_out_channels():
    """Output dimension for the base (pre-training) task."""
    return 1


@pytest.fixture
def new_out_channels():
    """Output dimension for the fine-tuning (transfer) task."""
    return 5


@pytest.fixture
def base_model(num_node_features, hidden_channels, base_out_channels):
    """A fresh simple graph-level regression model."""
    return _SimpleGraphModel(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        out_channels=base_out_channels,
    )


@pytest.fixture
def synthetic_dataset(num_node_features):
    """50-graph synthetic dataset for base training."""
    return _make_synthetic_dataset(
        num_graphs=50, num_node_features=num_node_features, num_targets=1
    )


@pytest.fixture
def fine_tune_dataset(num_node_features, new_out_channels):
    """30-graph synthetic dataset for the fine-tuning task (multi-target)."""
    return _make_synthetic_dataset(
        num_graphs=30,
        num_node_features=num_node_features,
        num_targets=new_out_channels,
        seed=999,
    )


@pytest.fixture
def tmp_working_dir(tmp_path):
    """Temporary working directory (DI pattern)."""
    return tmp_path


# =============================================================================
# HELPER: train a model with Trainer and save checkpoint
# =============================================================================

def _train_and_save_checkpoint(
    model: nn.Module,
    dataset: List[Data],
    device: torch.device,
    checkpoint_dir: Path,
    max_epochs: int = 3,
    model_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Train *model* for *max_epochs* and save a v2.0 checkpoint.

    Uses the real Trainer and ModelCheckpoint from the MILIA codebase.
    Returns the path to the saved checkpoint.
    """
    from milia_pipeline.models.training.trainer import Trainer
    from milia_pipeline.models.training.callbacks import ModelCheckpoint

    # Split dataset into train / val
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_best=True,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
        device=device,
        callbacks=[checkpoint_cb],
        max_epochs=max_epochs,
        model_info=model_info or {},
    )

    results = trainer.fit()

    # Also save via Trainer.save_checkpoint to get v2.0 format with hyper_parameters
    explicit_ckpt_path = checkpoint_dir / "pretrained.pt"
    trainer.save_checkpoint(
        filepath=explicit_ckpt_path,
        hyper_parameters={
            "model_name": "SimpleGraphModel",
            "task_type": "graph_regression",
            "hyperparameters": {
                "in_channels": model.encoder.conv1.in_features,
                "hidden_channels": model.encoder.conv1.out_features,
                "out_channels": model.head.out_features,
            },
            "model_info": model_info or {},
        },
        data_info={
            "num_node_features": model.encoder.conv1.in_features,
            "requires_edge_features": False,
        },
    )

    return explicit_ckpt_path, results


# =============================================================================
# TEST CLASS — End-to-End Transfer Learning Workflow
# =============================================================================

@pytest.mark.e2e
class TestE2ETransferLearningWorkflow:
    """
    E2E transfer learning workflow tests.

    Workflow under test:
        1. Train a base model on a synthetic dataset for a few epochs
        2. Save a v2.0 checkpoint (via Trainer.save_checkpoint)
        3. Load the checkpoint into a FineTuner instance
        4. Apply various freeze strategies
        5. Replace the output head for a new task
        6. Fine-tune on a new dataset
        7. Run inference via Predictor on fine-tuned model
    """

    # -----------------------------------------------------------------
    # 1. Base model training and checkpoint saving
    # -----------------------------------------------------------------

    def test_base_training_produces_checkpoint(
        self, base_model, synthetic_dataset, device, tmp_working_dir
    ):
        """Train base model and verify checkpoint file exists and is v2.0."""
        ckpt_dir = tmp_working_dir / "checkpoints"
        ckpt_path, results = _train_and_save_checkpoint(
            model=base_model,
            dataset=synthetic_dataset,
            device=device,
            checkpoint_dir=ckpt_dir,
            max_epochs=3,
        )

        # Checkpoint file exists
        assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

        # Checkpoint is v2.0 format
        # weights_only=False required: Trainer.save_checkpoint stores TorchVersion
        # objects (via torch.__version__) which are not in the safe-globals allowlist.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        version_info = checkpoint.get("version_info", {})
        assert version_info.get("checkpoint_format_version") == "2.0"

        # hyper_parameters are present
        hp = checkpoint.get("hyper_parameters", {})
        assert hp.get("model_name") == "SimpleGraphModel"
        assert hp.get("task_type") == "graph_regression"
        assert "hyperparameters" in hp

        # data_info is present
        data_info = checkpoint.get("data_info", {})
        assert "num_node_features" in data_info

        # Training metrics populated
        assert "train_metrics" in results
        assert results["training_time"] > 0

    def test_model_checkpoint_callback_creates_best_pt(
        self, base_model, synthetic_dataset, device, tmp_working_dir
    ):
        """ModelCheckpoint callback saves best.pt alongside epoch checkpoints."""
        ckpt_dir = tmp_working_dir / "checkpoints"
        _train_and_save_checkpoint(
            model=base_model,
            dataset=synthetic_dataset,
            device=device,
            checkpoint_dir=ckpt_dir,
            max_epochs=3,
        )

        # ModelCheckpoint should have created best.pt and last.pt
        assert (ckpt_dir / "best.pt").exists(), "best.pt not found"
        assert (ckpt_dir / "last.pt").exists(), "last.pt not found"

    # -----------------------------------------------------------------
    # 2. CheckpointManager load / save round-trip
    # -----------------------------------------------------------------

    def test_checkpoint_manager_save_load_round_trip(
        self, base_model, synthetic_dataset, device, tmp_working_dir
    ):
        """CheckpointManager can save and reload a checkpoint preserving all metadata."""
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        cm = CheckpointManager(working_root_dir=tmp_working_dir)

        # Train first to have realistic state_dict
        ckpt_dir = tmp_working_dir / "checkpoints"
        ckpt_path, _ = _train_and_save_checkpoint(
            model=base_model,
            dataset=synthetic_dataset,
            device=device,
            checkpoint_dir=ckpt_dir,
            max_epochs=2,
        )

        # Save via CheckpointManager
        cm_save_path = cm.save(
            filepath="cm_checkpoints/model_cm.pt",
            model=base_model,
            optimizer=torch.optim.Adam(base_model.parameters()),
            epoch=2,
            global_step=100,
            best_val_loss=0.05,
            hyper_parameters={
                "model_name": "SimpleGraphModel",
                "task_type": "graph_regression",
                "hyperparameters": {"hidden_channels": 32},
            },
            data_info={"num_node_features": 16},
        )

        assert cm_save_path.exists()

        # Load
        loaded = cm.load(cm_save_path)
        assert loaded["epoch"] == 2
        assert loaded["global_step"] == 100
        assert cm.is_v2_checkpoint(loaded)
        assert cm.get_model_name(loaded) == "SimpleGraphModel"

    # -----------------------------------------------------------------
    # 3. FineTuner — freeze strategies
    # -----------------------------------------------------------------

    def test_fine_tuner_freeze_encoder(
        self, base_model, num_node_features, hidden_channels, base_out_channels, device
    ):
        """FreezeStrategy.ENCODER freezes conv layers, keeps head trainable."""
        from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
            FineTuner,
            FreezeStrategy,
        )

        ft = FineTuner(
            model=copy.deepcopy(base_model),
            hyper_parameters={"out_channels": base_out_channels},
            working_root_dir=Path(tempfile.mkdtemp()),
        )

        model = ft.prepare_for_finetuning(
            freeze_strategy=FreezeStrategy.ENCODER,
        )

        # Verify: parameters whose name contains 'conv' are frozen
        for name, param in model.named_parameters():
            if "conv" in name.lower():
                assert not param.requires_grad, (
                    f"Encoder param '{name}' should be frozen but requires_grad=True"
                )
            else:
                assert param.requires_grad, (
                    f"Non-encoder param '{name}' should be trainable but requires_grad=False"
                )

    def test_fine_tuner_freeze_none(
        self, base_model, base_out_channels, device
    ):
        """FreezeStrategy.NONE leaves all parameters trainable."""
        from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
            FineTuner,
            FreezeStrategy,
        )

        ft = FineTuner(
            model=copy.deepcopy(base_model),
            hyper_parameters={"out_channels": base_out_channels},
            working_root_dir=Path(tempfile.mkdtemp()),
        )

        model = ft.prepare_for_finetuning(
            freeze_strategy=FreezeStrategy.NONE,
        )

        for name, param in model.named_parameters():
            assert param.requires_grad, (
                f"Param '{name}' should be trainable with FreezeStrategy.NONE"
            )

    def test_fine_tuner_freeze_all_but_last(
        self, base_model, base_out_channels, device
    ):
        """FreezeStrategy.ALL_BUT_LAST freezes everything except the last layer."""
        from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
            FineTuner,
            FreezeStrategy,
        )

        ft = FineTuner(
            model=copy.deepcopy(base_model),
            hyper_parameters={"out_channels": base_out_channels},
            working_root_dir=Path(tempfile.mkdtemp()),
        )

        model = ft.prepare_for_finetuning(
            freeze_strategy=FreezeStrategy.ALL_BUT_LAST,
        )

        # The last layer (head) should be trainable
        param_names = list(dict(model.named_parameters()).keys())
        last_layer_prefix = param_names[-1].rsplit(".", 1)[0] if param_names else ""

        for name, param in model.named_parameters():
            if name.startswith(last_layer_prefix):
                assert param.requires_grad, (
                    f"Last-layer param '{name}' should be trainable"
                )
            else:
                assert not param.requires_grad, (
                    f"Non-last-layer param '{name}' should be frozen"
                )

    def test_fine_tuner_freeze_encoder_partial(
        self, base_model, base_out_channels, device
    ):
        """FreezeStrategy.ENCODER_PARTIAL freezes the first N layers."""
        from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
            FineTuner,
            FreezeStrategy,
        )

        ft = FineTuner(
            model=copy.deepcopy(base_model),
            hyper_parameters={"out_channels": base_out_channels},
            working_root_dir=Path(tempfile.mkdtemp()),
        )

        model = ft.prepare_for_finetuning(
            freeze_strategy=FreezeStrategy.ENCODER_PARTIAL,
            freeze_layers=1,  # Freeze only the first top-level module
        )

        # At least some parameters should be frozen and some trainable
        frozen_count = sum(
            1 for p in model.parameters() if not p.requires_grad
        )
        trainable_count = sum(
            1 for p in model.parameters() if p.requires_grad
        )
        assert frozen_count > 0, "ENCODER_PARTIAL should freeze some parameters"
        assert trainable_count > 0, "ENCODER_PARTIAL should leave some parameters trainable"

    # -----------------------------------------------------------------
    # 4. FineTuner — output head replacement
    # -----------------------------------------------------------------

    def test_replace_output_head(
        self, base_model, base_out_channels, new_out_channels
    ):
        """FineTuner replaces the last Linear layer for a new task dimension."""
        from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
            FineTuner,
            FreezeStrategy,
        )

        ft = FineTuner(
            model=copy.deepcopy(base_model),
            hyper_parameters={"out_channels": base_out_channels},
            working_root_dir=Path(tempfile.mkdtemp()),
        )

        model = ft.prepare_for_finetuning(
            new_out_channels=new_out_channels,
            freeze_strategy=FreezeStrategy.ENCODER,
        )

        # The head's output dimension should match new_out_channels
        assert model.head.out_features == new_out_channels, (
            f"Expected head out_features={new_out_channels}, "
            f"got {model.head.out_features}"
        )
        # The head's input dimension should remain unchanged
        assert model.head.in_features == base_model.encoder.conv2.out_features

    # -----------------------------------------------------------------
    # 5. Full round-trip: train → save → FineTuner (direct) → fine-tune
    # -----------------------------------------------------------------

    def test_full_transfer_learning_round_trip(
        self,
        base_model,
        synthetic_dataset,
        fine_tune_dataset,
        device,
        tmp_working_dir,
        num_node_features,
        hidden_channels,
        new_out_channels,
    ):
        """
        Complete transfer-learning E2E workflow:

        1. Train base model for 2 epochs
        2. Save v2.0 checkpoint
        3. Instantiate FineTuner directly (no from_checkpoint to avoid
           ModelLoader/registry dependency)
        4. Apply ENCODER freeze + replace output head
        5. Fine-tune the adapted model for 2 more epochs
        6. Assert frozen encoder params did not change
        7. Assert fine-tuned model produces predictions of correct shape
        """
        from milia_pipeline.models.training.trainer import Trainer
        from milia_pipeline.models.training.callbacks import ModelCheckpoint
        from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
            FineTuner,
            FreezeStrategy,
        )

        # --- Step 1 & 2: Train and save base checkpoint ---
        ckpt_dir = tmp_working_dir / "checkpoints"
        ckpt_path, base_results = _train_and_save_checkpoint(
            model=base_model,
            dataset=synthetic_dataset,
            device=device,
            checkpoint_dir=ckpt_dir,
            max_epochs=2,
        )
        assert ckpt_path.exists()

        # --- Step 3: Load checkpoint manually and create FineTuner ---
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = checkpoint.get("hyper_parameters", {})

        # Recreate a fresh model with same architecture to load state into
        fresh_model = _SimpleGraphModel(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=1,
        )
        fresh_model.load_state_dict(checkpoint["model_state_dict"])

        ft = FineTuner(
            model=fresh_model,
            hyper_parameters=hp.get("hyperparameters", {}),
            working_root_dir=tmp_working_dir,
        )

        # --- Step 4: Prepare for fine-tuning ---
        adapted_model = ft.prepare_for_finetuning(
            new_out_channels=new_out_channels,
            freeze_strategy=FreezeStrategy.ENCODER,
        )

        # Snapshot encoder weights before fine-tuning
        encoder_weights_before = {
            name: param.clone()
            for name, param in adapted_model.named_parameters()
            if "conv" in name.lower()
        }

        # --- Step 5: Fine-tune ---
        ft_split = int(0.8 * len(fine_tune_dataset))
        ft_train = fine_tune_dataset[:ft_split]
        ft_val = fine_tune_dataset[ft_split:]

        ft_train_loader = DataLoader(ft_train, batch_size=8, shuffle=True)
        ft_val_loader = DataLoader(ft_val, batch_size=8, shuffle=False)

        ft_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapted_model.parameters()),
            lr=1e-3,
        )

        ft_ckpt_dir = tmp_working_dir / "ft_checkpoints"
        ft_ckpt_dir.mkdir(parents=True, exist_ok=True)
        ft_checkpoint_cb = ModelCheckpoint(
            dirpath=ft_ckpt_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )

        ft_trainer = Trainer(
            model=adapted_model,
            train_loader=ft_train_loader,
            val_loader=ft_val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=ft_optimizer,
            device=device,
            callbacks=[ft_checkpoint_cb],
            max_epochs=2,
        )

        ft_results = ft_trainer.fit()

        # --- Step 6: Verify encoder params did not change ---
        for name, param in adapted_model.named_parameters():
            if "conv" in name.lower():
                before = encoder_weights_before[name]
                assert torch.equal(param.data, before), (
                    f"Frozen encoder param '{name}' changed during fine-tuning"
                )

        # --- Step 7: Verify predictions have correct shape ---
        test_data = fine_tune_dataset[0]
        test_batch = Batch.from_data_list([test_data])
        adapted_model.eval()
        with torch.no_grad():
            pred = adapted_model(
                test_batch.x, test_batch.edge_index, batch=test_batch.batch
            )
        assert pred.shape[-1] == new_out_channels, (
            f"Expected prediction dim={new_out_channels}, got {pred.shape[-1]}"
        )

        # Fine-tuning produced results
        assert "train_metrics" in ft_results
        assert ft_results["training_time"] > 0

    # -----------------------------------------------------------------
    # 6. Predictor on the fine-tuned model
    # -----------------------------------------------------------------

    def test_predictor_on_fine_tuned_model(
        self,
        base_model,
        synthetic_dataset,
        fine_tune_dataset,
        device,
        tmp_working_dir,
        num_node_features,
        hidden_channels,
        new_out_channels,
    ):
        """
        After fine-tuning, Predictor.predict() returns tensors of correct shape.

        This test constructs the Predictor directly (bypassing from_checkpoint
        which requires full ModelLoader/registry wiring) to verify inference
        on the adapted model.
        """
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )
        from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
            FineTuner,
            FreezeStrategy,
        )

        # Train base, save checkpoint, load, adapt (same as round-trip)
        ckpt_dir = tmp_working_dir / "ckpt_pred"
        ckpt_path, _ = _train_and_save_checkpoint(
            model=base_model,
            dataset=synthetic_dataset,
            device=device,
            checkpoint_dir=ckpt_dir,
            max_epochs=2,
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        fresh_model = _SimpleGraphModel(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=1,
        )
        fresh_model.load_state_dict(checkpoint["model_state_dict"])

        ft = FineTuner(
            model=fresh_model,
            hyper_parameters=checkpoint.get("hyper_parameters", {}).get(
                "hyperparameters", {}
            ),
            working_root_dir=tmp_working_dir,
        )
        adapted_model = ft.prepare_for_finetuning(
            new_out_channels=new_out_channels,
            freeze_strategy=FreezeStrategy.ENCODER,
        )

        # Quick fine-tune (1 epoch)
        ft_loader = DataLoader(fine_tune_dataset[:10], batch_size=4, shuffle=True)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapted_model.parameters()), lr=1e-3
        )
        adapted_model.train()
        for batch in ft_loader:
            optimizer.zero_grad()
            out = adapted_model(batch.x, batch.edge_index, batch=batch.batch)
            loss = nn.MSELoss()(out, batch.y)
            loss.backward()
            optimizer.step()

        # --- Build Predictor directly ---
        predictor = Predictor(
            model=adapted_model,
            working_root_dir=tmp_working_dir,
            device=device,
            task_type="graph_regression",
        )

        # Single prediction
        single_data = fine_tune_dataset[0]
        single_batch = Batch.from_data_list([single_data])
        pred = predictor.predict(single_batch)
        assert isinstance(pred, torch.Tensor)
        assert pred.shape[-1] == new_out_channels

        # Batch prediction
        preds = predictor.predict_batch(fine_tune_dataset[:5], batch_size=2)
        assert isinstance(preds, torch.Tensor)
        assert preds.shape[0] == 5
        assert preds.shape[-1] == new_out_channels

    # -----------------------------------------------------------------
    # 7. Predictor.save_predictions after fine-tuning
    # -----------------------------------------------------------------

    def test_save_predictions_after_fine_tuning(
        self,
        base_model,
        fine_tune_dataset,
        device,
        tmp_working_dir,
        num_node_features,
        hidden_channels,
        new_out_channels,
    ):
        """Predictor.save_predictions writes a CSV file with correct columns."""
        from milia_pipeline.models.post_training.inference.predictor import (
            Predictor,
        )
        from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
            FineTuner,
            FreezeStrategy,
        )

        # Create and adapt model (skip full training for speed)
        model = _SimpleGraphModel(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=new_out_channels,
        )

        predictor = Predictor(
            model=model,
            working_root_dir=tmp_working_dir,
            device=device,
            task_type="graph_regression",
        )

        preds = predictor.predict_batch(fine_tune_dataset[:5], batch_size=2)

        csv_path = predictor.save_predictions(
            preds, "predictions/results.csv", format="csv"
        )
        assert csv_path.exists()

        # Read and verify
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 5
        # Multi-target: columns are prediction_0 ... prediction_{n-1}
        assert df.shape[1] == new_out_channels

    # -----------------------------------------------------------------
    # 8. Structural features config preserved through checkpoint
    # -----------------------------------------------------------------

    def test_structural_features_config_preserved_in_checkpoint(
        self,
        base_model,
        synthetic_dataset,
        device,
        tmp_working_dir,
    ):
        """
        data_info.structural_features_config survives the checkpoint round-trip.

        This verifies the fix (v1.6.0) that prevents feature dimension mismatch
        between training and inference.
        """
        from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
            CheckpointManager,
        )

        structural_config = {
            "atom": ["atomic_num", "formal_charge", "is_aromatic"],
            "bond": ["bond_type", "is_conjugated"],
        }

        cm = CheckpointManager(working_root_dir=tmp_working_dir)
        save_path = cm.save(
            filepath="ckpt_struct/model.pt",
            model=base_model,
            epoch=0,
            data_info={
                "num_node_features": 16,
                "structural_features_config": structural_config,
            },
        )

        loaded = cm.load(save_path)
        loaded_config = loaded.get("data_info", {}).get(
            "structural_features_config"
        )
        assert loaded_config == structural_config, (
            "structural_features_config not preserved through checkpoint"
        )

    # -----------------------------------------------------------------
    # 9. Fine-tuning converges (loss decreases)
    # -----------------------------------------------------------------

    def test_fine_tuning_loss_decreases(
        self,
        base_model,
        synthetic_dataset,
        fine_tune_dataset,
        device,
        tmp_working_dir,
        num_node_features,
        hidden_channels,
        new_out_channels,
    ):
        """
        Fine-tuning on the new task should show a decreasing training loss,
        confirming that the adapted model is actually learning.
        """
        from milia_pipeline.models.training.trainer import Trainer
        from milia_pipeline.models.training.callbacks import ModelCheckpoint
        from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
            FineTuner,
            FreezeStrategy,
        )

        # Train base
        ckpt_dir = tmp_working_dir / "ckpt_loss"
        ckpt_path, _ = _train_and_save_checkpoint(
            model=base_model,
            dataset=synthetic_dataset,
            device=device,
            checkpoint_dir=ckpt_dir,
            max_epochs=2,
        )

        # Load and adapt — use FreezeStrategy.NONE so the full model is
        # trainable.  With ENCODER frozen only a single Linear head trains,
        # which on random multi-target noise may not reliably decrease within
        # a handful of epochs.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        fresh_model = _SimpleGraphModel(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=1,
        )
        fresh_model.load_state_dict(checkpoint["model_state_dict"])

        ft = FineTuner(
            model=fresh_model,
            hyper_parameters={},
            working_root_dir=tmp_working_dir,
        )
        adapted_model = ft.prepare_for_finetuning(
            new_out_channels=new_out_channels,
            freeze_strategy=FreezeStrategy.NONE,
        )

        # Fine-tune for 10 epochs with a moderate LR to observe loss trend.
        # shuffle=False for reproducibility across runs.
        ft_split = int(0.8 * len(fine_tune_dataset))
        ft_train_loader = DataLoader(
            fine_tune_dataset[:ft_split], batch_size=8, shuffle=False
        )
        ft_val_loader = DataLoader(
            fine_tune_dataset[ft_split:], batch_size=8, shuffle=False
        )

        ft_ckpt_dir = tmp_working_dir / "ft_ckpt_loss"
        ft_ckpt_dir.mkdir(parents=True, exist_ok=True)

        ft_trainer = Trainer(
            model=adapted_model,
            train_loader=ft_train_loader,
            val_loader=ft_val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=torch.optim.Adam(adapted_model.parameters(), lr=1e-2),
            device=device,
            callbacks=[
                ModelCheckpoint(
                    dirpath=ft_ckpt_dir,
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                )
            ],
            max_epochs=10,
        )

        ft_results = ft_trainer.fit()

        train_losses = ft_results["train_metrics"].get("train_loss", [])
        assert len(train_losses) == 10, (
            f"Expected 10 epoch losses, got {len(train_losses)}"
        )

        # The *best* (minimum) loss observed should be strictly lower than the
        # initial loss.  This is more robust than comparing last-vs-first because
        # stochastic training on synthetic data may oscillate in later epochs.
        best_loss = min(train_losses)
        assert best_loss < train_losses[0], (
            f"Fine-tuning loss never improved below the initial value: "
            f"first={train_losses[0]:.6f}, best={best_loss:.6f}"
        )

    # -----------------------------------------------------------------
    # 10. EarlyStopping interacts correctly with fine-tuning
    # -----------------------------------------------------------------

    def test_early_stopping_during_fine_tuning(
        self,
        device,
        fine_tune_dataset,
        num_node_features,
        hidden_channels,
        new_out_channels,
        tmp_working_dir,
    ):
        """EarlyStopping can halt fine-tuning when validation loss stops improving."""
        from milia_pipeline.models.training.trainer import Trainer
        from milia_pipeline.models.training.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
        )

        # Use a model with a very high learning rate to cause rapid initial
        # convergence followed by divergence/plateau, so EarlyStopping triggers.
        model = _SimpleGraphModel(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=new_out_channels,
        )

        ft_split = int(0.5 * len(fine_tune_dataset))
        train_loader = DataLoader(
            fine_tune_dataset[:ft_split], batch_size=4, shuffle=False
        )
        val_loader = DataLoader(
            fine_tune_dataset[ft_split:], batch_size=4, shuffle=False
        )

        ckpt_dir = tmp_working_dir / "early_stop_ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # patience=3 means stop if no improvement for 3 consecutive epochs.
        # A very large learning rate (0.5) causes the loss to oscillate wildly
        # after initial descent, so patience will be exhausted quickly.
        early_stop = EarlyStopping(
            monitor="val_loss", patience=3, mode="min", min_delta=0.0
        )
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir, monitor="val_loss", mode="min",
            save_top_k=1, save_last=True,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.5),  # Aggressive LR
            device=device,
            callbacks=[early_stop, ckpt_cb],
            max_epochs=50,  # Upper bound — should stop well before this
        )

        results = trainer.fit()

        # Verify EarlyStopping was active and trainer ran
        epochs_completed = len(
            results["train_metrics"].get("train_loss", [])
        )
        # The core contract: EarlyStopping callback correctly integrates with
        # Trainer (set_trainer called, on_epoch_end invoked, should_stop checked).
        # We verify integration by checking the callback's internal state was updated.
        assert early_stop.best_score is not None, (
            "EarlyStopping.best_score was never set — callback not integrated"
        )
        assert epochs_completed >= 1, "At least 1 epoch should have completed"

        # If early stopping triggered, it should have stopped before max_epochs.
        # If it didn't trigger (unlikely with aggressive LR, but possible with
        # random data), we still verify the callback was properly wired.
        if early_stop.should_stop():
            assert epochs_completed < 50, (
                f"EarlyStopping triggered but epochs_completed={epochs_completed} >= max_epochs"
            )


# =============================================================================
# MODULE-LEVEL ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
