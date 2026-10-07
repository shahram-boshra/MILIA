"""Shared DataLoader factory for MILIA's training paths (PA-2).

Every training path (main, HPO trial / final model / CV folds, ``create_training_pipeline``) builds
its loaders through :func:`make_loader`, so ``models.acceleration.computation.dataloader`` takes effect
in one place.

* :class:`LoaderOptions` defaults equal the loaders MILIA used before (single-process, unpinned), so
  loaders change only when acceleration is enabled and configured.
* Options are normalized to torch's ``DataLoader`` contract (torch 2.4, ``torch/utils/data/dataloader.py``):
  ``prefetch_factor`` must be ``None`` and ``persistent_workers`` false unless ``num_workers > 0``;
  ``pin_memory`` only matters when batches are copied to a CUDA device.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric import loader as pyg_loader


@dataclass(frozen=True)
class LoaderOptions:
    """Worker / memory options for :func:`make_loader`. Defaults reproduce torch's defaults."""

    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False

    def as_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``DataLoader``; worker-only options only when workers are used."""
        kwargs: dict[str, Any] = {"num_workers": self.num_workers, "pin_memory": self.pin_memory}
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs


def loader_options_from_config(models_config: Mapping[str, Any] | None) -> LoaderOptions:
    """``LoaderOptions`` from ``models.acceleration``; the current defaults when it is disabled.

    Raises:
        ConfigurationError: the acceleration section is invalid (same validation as PA-1e).
    """
    from milia_pipeline.models.acceleration.config_builder import load_acceleration_config

    config = load_acceleration_config(models_config)
    if not config.enabled:
        return LoaderOptions()

    dataloader = config.computation.dataloader
    workers = dataloader.num_workers
    device_type = config.device.type
    targets_cuda = device_type.startswith("cuda") or device_type == "auto"
    return LoaderOptions(
        num_workers=workers,
        pin_memory=bool(dataloader.pin_memory and targets_cuda and torch.cuda.is_available()),
        prefetch_factor=dataloader.prefetch_factor if workers > 0 else None,
        persistent_workers=bool(dataloader.persistent_workers and workers > 0),
    )


def make_loader(
    dataset: Sequence[Any] | Any,
    *,
    batch_size: int,
    shuffle: bool,
    options: LoaderOptions | None = None,
) -> pyg_loader.DataLoader:
    """Build a PyG ``DataLoader`` with the shared options (``None`` → current defaults).

    The class is looked up on ``torch_geometric.loader`` at call time (as the replaced call sites
    did), so patching ``torch_geometric.loader.DataLoader`` keeps working as a test seam.
    """
    return pyg_loader.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **(options or LoaderOptions()).as_kwargs()
    )
