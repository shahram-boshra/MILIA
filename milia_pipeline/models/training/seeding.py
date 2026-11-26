"""Seeding a training run, so that the same code, data and configuration give the same model.

The data split has its own seed (``data_split.random_seed``). Training draws random numbers too:
weight initialization, the shuffling of the training loader, dropout. ``training.seed`` seeds
them: :func:`seed_everything` seeds Python's ``random``, NumPy and PyTorch ("You can use
torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)"), and the training
loader shuffles with its own generator (:func:`loader_generator`), its workers seeded from it
(:func:`seed_worker`), as the PyTorch reproducibility notes describe
(https://docs.pytorch.org/docs/stable/notes/randomness.html). ``seed: null`` draws a seed, which
is recorded (``run_info.training_seed``) like a given one, so every run can be repeated.

Seeding makes a run repeatable on the same platform and versions; PyTorch does not guarantee
identical results across releases, platforms or nondeterministic (e.g. CUDA atomic) kernels.
"""

from __future__ import annotations

import random
import secrets
from typing import Any

import numpy as np
import torch

from milia_pipeline.exceptions import ConfigurationError

__all__ = ["SEED_LIMIT", "loader_generator", "resolve_seed", "seed_everything", "seed_worker"]

#: NumPy's legacy global seed accepts 0 <= seed < 2**32
SEED_LIMIT = 2**32


def resolve_seed(seed: Any) -> int:
    """``training.seed`` as an integer: a given one is checked, ``None`` draws one.

    Raises:
        ConfigurationError: Not an integer in ``[0, 2**32)``.
    """
    if seed is None:
        return secrets.randbelow(SEED_LIMIT)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < SEED_LIMIT:
        raise ConfigurationError(
            f"models.training.seed must be an integer in [0, 2**32) or null, got {seed!r}"
        )
    return seed


def seed_everything(seed: int) -> int:
    """Seed Python's ``random``, NumPy and PyTorch (all devices) with ``seed``; returns it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def loader_generator(seed: int) -> torch.Generator:
    """A generator for a loader's shuffling, independent of the global PyTorch generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn``: seed a loader worker's ``random`` and NumPy from its PyTorch seed
    (which the loader derives from its generator)."""
    worker_seed = torch.initial_seed() % SEED_LIMIT
    np.random.seed(worker_seed)
    random.seed(worker_seed)
