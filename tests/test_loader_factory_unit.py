"""PA-2: shared DataLoader factory and models.acceleration.computation.dataloader mapping."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.exceptions import ConfigurationError
from milia_pipeline.models.training.loaders import (
    LoaderOptions,
    loader_options_from_config,
    make_loader,
)


def _graphs(n: int = 6) -> list[Data]:
    gen = torch.Generator().manual_seed(0)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    return [Data(x=torch.randn(3, 4, generator=gen), edge_index=ei) for _ in range(n)]


def _cfg(enabled: bool = True, device: str = "cpu", **dataloader):
    return {
        "acceleration": {
            "enabled": enabled,
            "device": {"type": device},
            "computation": {"dataloader": dataloader},
        }
    }


# --- defaults = the loaders MILIA built before PA-2 ------------------------------------------------
def test_default_options_match_torch_defaults():
    assert LoaderOptions().as_kwargs() == {"num_workers": 0, "pin_memory": False}


def test_default_loader_is_single_process_and_unpinned():
    loader = make_loader(_graphs(), batch_size=4, shuffle=False)
    assert loader.num_workers == 0
    assert loader.pin_memory is False
    assert loader.prefetch_factor is None
    assert loader.persistent_workers is False
    assert [b.num_graphs for b in loader] == [4, 2]


def test_worker_options_only_passed_with_workers():
    kwargs = LoaderOptions(num_workers=2, prefetch_factor=4, persistent_workers=True).as_kwargs()
    assert kwargs == {
        "num_workers": 2,
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }


def test_multi_worker_loader_iterates_pyg_batches():
    loader = make_loader(
        _graphs(), batch_size=3, shuffle=False, options=LoaderOptions(num_workers=1)
    )
    assert [b.num_graphs for b in loader] == [3, 3]


def test_class_is_resolved_at_call_time():
    """`torch_geometric.loader.DataLoader` stays patchable (the seam existing tests use)."""
    with patch("torch_geometric.loader.DataLoader") as fake:
        make_loader([1, 2], batch_size=1, shuffle=True)
    fake.assert_called_once_with(
        [1, 2], batch_size=1, shuffle=True, num_workers=0, pin_memory=False
    )


# --- config mapping ------------------------------------------------------------------------------
@pytest.mark.parametrize("models_config", [None, {}, _cfg(enabled=False, num_workers=8)])
def test_disabled_or_absent_config_keeps_defaults(models_config):
    assert loader_options_from_config(models_config) == LoaderOptions()


def test_enabled_config_maps_worker_options():
    options = loader_options_from_config(
        _cfg(num_workers=2, prefetch_factor=3, persistent_workers=True, pin_memory=True)
    )
    assert options == LoaderOptions(
        num_workers=2, pin_memory=False, prefetch_factor=3, persistent_workers=True
    )  # cpu device → no pinning


def test_shipped_prefetch_default_with_zero_workers_is_not_passed():
    """torch raises for prefetch_factor with num_workers=0; the shipped default (2) is dropped."""
    options = loader_options_from_config(_cfg(num_workers=0, prefetch_factor=2))
    assert options.prefetch_factor is None
    assert make_loader(_graphs(), batch_size=2, shuffle=False, options=options).num_workers == 0


@pytest.mark.parametrize(
    ("device", "cuda", "pinned"),
    [("cuda", True, True), ("auto", True, True), ("cuda", False, False), ("cpu", True, False)],
)
def test_pin_memory_only_for_cuda_targets(device, cuda, pinned):
    with patch("torch.cuda.is_available", return_value=cuda):
        options = loader_options_from_config(_cfg(device=device, num_workers=0, pin_memory=True))
    assert options.pin_memory is pinned


# --- validation (enabled only) ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dataloader", "message"),
    [
        ({"num_workers": -1}, "num_workers must be >= 0"),
        (
            {"num_workers": 0, "persistent_workers": True},
            "persistent_workers needs num_workers > 0",
        ),
        ({"num_workers": 2, "prefetch_factor": -1}, "prefetch_factor must be >= 0"),
    ],
)
def test_invalid_enabled_dataloader_rejected(dataloader, message):
    with pytest.raises(ConfigurationError, match=message):
        loader_options_from_config(_cfg(**dataloader))
