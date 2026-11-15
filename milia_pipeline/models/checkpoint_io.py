"""
Checkpoint file loading for MILIA models.

Every checkpoint is read with ``torch.load(weights_only=True)``: only tensors, primitive containers and the types
registered with ``torch.serialization.add_safe_globals`` (PyTorch's own, PyG's data classes, MILIA's allowlist in
``CheckpointManager``) are rebuilt. A file that needs anything else is rejected. MILIA never retries with full
unpickling, which "can result in arbitrary code execution" (PyTorch serialization notes) — a corrupt or crafted
file must not reach it. A caller who trusts a file (for example a legacy checkpoint it wrote itself) passes
``weights_only=False`` explicitly.

Checkpoints written by MILIA hold only state dicts, tensors and plain metadata, so they load under the default.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

__all__ = ["load_checkpoint_file"]

_REJECTION_NOTE = (
    "MILIA reads checkpoints with torch.load(weights_only=True) and never falls back to full unpickling, which can "
    "run code embedded in the file. If you trust this file (for example a checkpoint you created), load it with "
    "weights_only=False explicitly."
)


def load_checkpoint_file(
    filepath: str | Path,
    map_location: str | torch.device | None = None,
    *,
    weights_only: bool = True,
) -> Any:
    """Load a ``torch.save`` file with restricted unpickling unless the caller opts out.

    Args:
        filepath: Path of the file written by ``torch.save``.
        map_location: Passed to ``torch.load``.
        weights_only: ``True`` (default) restricts unpickling to allowlisted types. ``False`` performs full
            unpickling and must only be used for files the caller trusts.

    Returns:
        The deserialized object.

    Raises:
        pickle.UnpicklingError: Under ``weights_only=True``, the file is corrupt or needs a global outside the
            allowlist; the exception carries a note explaining the explicit opt-out.
    """
    try:
        return torch.load(filepath, map_location=map_location, weights_only=weights_only)
    except pickle.UnpicklingError as exc:
        if weights_only:
            exc.add_note(_REJECTION_NOTE)
        raise
