# Location: milia_pipeline/models/hpo/seeding.py

"""
Per-worker sampler seeds for parallel HPO (P2-3b, blueprint S3 / F8).

Several processes optimizing one study with the same sampler seed draw the same parameters: two workers
with ``TPESampler(seed=42)`` produced 10 trials but only 5 distinct configurations (R5, R62). Each
worker therefore gets its own seed, derived deterministically from the configured seed and its worker
index.

Derivation follows NumPy's parallel random-generation guidance: ``root_seed + worker_id`` is documented
as unsafe (runs with different root seeds share worker seeds; close seeds can yield identical results);
``SeedSequence([worker_id, root_seed])`` mixes both into a high-quality, independent state. Optuna's
samplers take an ``int`` seed for ``numpy.random.RandomState``, whose range is ``[0, 2**32 - 1]``, so
one 32-bit word of the derived state is used. Values are stable across NumPy 1.26 and 2.x (R62).

Optuna does not make distributed runs reproducible (FAQ); derived seeds make workers distinct and the
seed each worker uses traceable, not the interleaving of trials.
"""

from __future__ import annotations

import numpy as np

from milia_pipeline.exceptions import HPOConfigurationError

__all__ = ["derive_worker_seed", "validate_worker_index"]


def validate_worker_index(worker_index: object) -> int | None:
    """Return ``worker_index`` if it is ``None`` or an ``int >= 0``; raise otherwise.

    ``bool`` is rejected although it subclasses ``int`` (``True`` is not a worker index).
    """
    if worker_index is None:
        return None
    if isinstance(worker_index, bool) or not isinstance(worker_index, int) or worker_index < 0:
        raise HPOConfigurationError(
            f"worker_index must be a non-negative integer or None, got {worker_index!r}",
            config_key="worker_index",
        )
    return worker_index


def derive_worker_seed(base_seed: int | None, worker_index: int | None) -> int | None:
    """Return the sampler seed for one worker.

    Args:
        base_seed: Configured sampler seed (``models.hpo.sampler.seed``); ``None`` = unseeded
        worker_index: This worker's index (``None`` = not a worker of a parallel study)

    Returns:
        ``base_seed`` unchanged when either argument is ``None`` (single-process behaviour, or unseeded
        workers drawing from OS entropy); otherwise a deterministic ``int`` in ``[0, 2**32 - 1]`` from
        ``SeedSequence([worker_index, base_seed])``.
    """
    worker_index = validate_worker_index(worker_index)
    if base_seed is None or worker_index is None:
        return base_seed
    state = np.random.SeedSequence([worker_index, base_seed]).generate_state(1, dtype=np.uint32)
    return int(state[0])
