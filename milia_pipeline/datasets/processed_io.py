"""
Weights-only storage for MILIA's processed datasets and processing chunks.

The processed file uses the layout of PyG's ``InMemoryDataset.save``, i.e.
``(data.to_dict(), slices, data class)``, and each processing chunk stores
``(list of data.to_dict(), data class)``. Every stored value is a tensor, a primitive or a
container of them, and the data class is one PyG registers with
``torch.serialization.add_safe_globals``, so both load with ``torch.load(weights_only=True)``.

A file in the older layout (a pickled ``Data`` object, which needs PyG's storage classes such
as ``GlobalStorage``) or an unreadable file raises :class:`StaleProcessedFileError`. Callers
rebuild it from the raw data; MILIA never retries with full unpickling, which can execute code
embedded in the file (PyTorch serialization notes).
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

__all__ = [
    "StaleProcessedFileError",
    "load_chunk",
    "load_processed",
    "save_chunk",
    "save_processed",
]

# The global torch's weights-only unpickler names when it rejects one ("GLOBAL <qualified name>").
_REJECTED_GLOBAL = re.compile(r"GLOBAL (\S+)")


class StaleProcessedFileError(Exception):
    """The file needs full unpickling (older layout) or is unreadable; rebuild it from raw data."""


def _load_weights_only(path: str | Path) -> Any:
    try:
        return torch.load(path, weights_only=True)
    except OSError:
        raise  # missing or unreadable on disk: an I/O problem, not a stale file
    except Exception as exc:
        # Any other failure means the bytes are not a weights-only file: a rejected global
        # (UnpicklingError), a truncated or damaged file (struct.error, EOFError, RuntimeError,
        # ...). The message names the rejected global or the error type; torch's own text is
        # not relayed, because it advises weights_only=False or allowlisting the class. The
        # original exception stays chained as __cause__.
        found = (
            _REJECTED_GLOBAL.search(str(exc)) if isinstance(exc, pickle.UnpicklingError) else None
        )
        kind = type(exc)
        error_name = (
            kind.__qualname__
            if kind.__module__ == "builtins"
            else f"{kind.__module__}.{kind.__qualname__}"
        )
        detail = f"rejected global {found.group(1)}" if found else error_name
        raise StaleProcessedFileError(
            f"{path} cannot be read with torch.load(weights_only=True) ({detail})"
        ) from exc


def _data_class(candidate: Any, path: str | Path) -> type[BaseData]:
    if isinstance(candidate, type) and issubclass(candidate, BaseData):
        return candidate
    raise StaleProcessedFileError(f"{path} does not name a PyG data class (found {candidate!r})")


def save_chunk(data_list: Sequence[BaseData], path: str | Path) -> None:
    """Write one processing chunk as plain dictionaries plus their (single) data class."""
    classes = {type(data) for data in data_list}
    if len(classes) > 1:
        raise ValueError(
            f"a chunk must hold one data class, found {sorted(c.__name__ for c in classes)}"
        )
    data_cls = classes.pop() if classes else Data
    torch.save(([data.to_dict() for data in data_list], data_cls), path)


def load_chunk(path: str | Path) -> list[BaseData]:
    """Read a chunk written by :func:`save_chunk` with ``weights_only=True``."""
    stored = _load_weights_only(path)
    if not (isinstance(stored, tuple) and len(stored) == 2 and isinstance(stored[0], list)):
        raise StaleProcessedFileError(
            f"{path} is not in the chunk layout (list of dicts, data class)"
        )
    items, data_cls = stored
    data_cls = _data_class(data_cls, path)
    return [data_cls.from_dict(item) for item in items]


def save_processed(
    data: BaseData, slices: dict[str, torch.Tensor] | None, path: str | Path
) -> None:
    """Write collated data in ``InMemoryDataset.save`` layout.

    The file holds ``(data.to_dict(), slices, data class)``. The caller collates:
    ``miliaDataset.collate`` is an instance method, so PyG's classmethod
    ``InMemoryDataset.save``, which calls ``cls.collate``, cannot be used for it.
    """
    torch.save((data.to_dict(), slices, data.__class__), path)


def load_processed(path: str | Path) -> tuple[BaseData, dict[str, torch.Tensor] | None]:
    """Read a processed file in ``InMemoryDataset.save`` layout with ``weights_only=True``.

    Returns:
        The collated data object and its slices, as ``InMemoryDataset.load`` would set them.

    Raises:
        StaleProcessedFileError: The file is in the older pickled-``Data`` layout or cannot be read.
    """
    stored = _load_weights_only(path)
    if not (isinstance(stored, tuple) and len(stored) == 3 and isinstance(stored[0], dict)):
        raise StaleProcessedFileError(
            f"{path} is in an older layout; MILIA now writes (data.to_dict(), slices, data class)"
        )
    mapping, slices, data_cls = stored
    return _data_class(data_cls, path).from_dict(mapping), slices
