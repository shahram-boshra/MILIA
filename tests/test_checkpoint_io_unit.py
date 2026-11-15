"""
Checkpoint loading never falls back to full unpickling (F50).

A file that ``torch.load(weights_only=True)`` rejects — corrupt, or carrying a global outside the allowlist —
must raise from every MILIA checkpoint loader without its payload running. Only an explicit
``weights_only=False`` from the caller performs full unpickling. Checkpoints MILIA writes load under the default.
"""

import pickle
from pathlib import Path

import pytest
import torch

from milia_pipeline.models.acceleration.distributed_strategies import DistributedManager
from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import CheckpointManager
from milia_pipeline.models.training.trainer import Trainer


class _Payload:
    """Pickles to a call of ``Path.touch(marker)``: unpickling it creates the marker file."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


class _TrustedValue:
    """A plain custom object, loadable only when the caller opts into full unpickling."""

    def __init__(self, value: int):
        self.value = value


@pytest.fixture
def crafted(tmp_path):
    """A checkpoint-shaped file whose metadata carries a payload; returns (path, marker)."""
    marker = tmp_path / "payload-ran"
    path = tmp_path / "crafted.pt"
    torch.save({"model_state_dict": {}, "hyper_parameters": {"x": _Payload(marker)}}, path)
    assert not marker.exists()
    return path, marker


def test_rejected_file_raises_with_opt_out_note_and_payload_never_runs(crafted):
    path, marker = crafted
    with pytest.raises(pickle.UnpicklingError) as excinfo:
        load_checkpoint_file(path)
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("weights_only=False explicitly" in note for note in notes)
    assert not marker.exists()


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a valid pytorch checkpoint")
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint_file(path)


def test_explicit_opt_out_loads_a_trusted_file(tmp_path):
    path = tmp_path / "trusted.pt"
    torch.save({"value": _TrustedValue(7)}, path)
    assert load_checkpoint_file(path, weights_only=False)["value"].value == 7


def test_checkpoint_manager_does_not_retry_unsafely(crafted, tmp_path):
    path, marker = crafted
    with pytest.raises(pickle.UnpicklingError):
        CheckpointManager(working_root_dir=tmp_path).load(path)
    assert not marker.exists()


def test_trainer_loaders_do_not_retry_unsafely(crafted):
    path, marker = crafted
    with pytest.raises(pickle.UnpicklingError):
        Trainer.get_checkpoint_info(path)
    assert Trainer.is_v2_checkpoint(path) is False
    assert not marker.exists()


def test_distributed_loader_does_not_retry_unsafely(crafted):
    path, marker = crafted
    with pytest.raises(pickle.UnpicklingError):
        DistributedManager(verbose=False).load_checkpoint(torch.nn.Linear(2, 1), path)
    assert not marker.exists()


def test_checkpoint_manager_output_loads_under_weights_only(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    manager = CheckpointManager(working_root_dir=tmp_path)
    saved = manager.save(
        tmp_path / "ckpt.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics_history={"val_loss": [0.5, 0.4]},
        best_val_loss=0.4,
        hyper_parameters={"model_name": "GCN", "hyperparameters": {"hidden_channels": 16}},
        data_info={"structural_features_config": {"atom": ["atomic_num"], "bond": []}},
    )
    loaded = torch.load(saved, weights_only=True)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["model_state_dict"]["weight"], model.state_dict()["weight"])
