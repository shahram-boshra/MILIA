# Location: milia_pipeline/models/hpo/backends/__init__.py

"""
HPO Backends Package

Provides backend implementations for hyperparameter optimization.
Supports Optuna (primary) with Ray Tune available for future scale-out.

Exports:
    HPOBackendProtocol: Protocol defining the backend interface (7 methods)
    OptunaBackend: Primary HPO backend using Optuna's TPE sampler
    get_backend: Factory function to get backend by name
    OPTUNA_AVAILABLE: Boolean indicating if Optuna is installed

Usage:
    >>> from milia_pipeline.models.hpo.backends import get_backend, OptunaBackend
    >>>
    >>> # Factory usage (recommended)
    >>> backend = get_backend("optuna")
    >>> study = backend.create_study("my_study", "minimize")
    >>>
    >>> # Direct instantiation
    >>> backend = OptunaBackend()
    >>> sampler = backend.create_sampler("tpe", seed=42)
    >>> pruner = backend.create_pruner("median", n_startup_trials=5)

Pattern: Follows handlers/__init__.py export pattern
"""

from .base import (
    OPTUNA_AVAILABLE,
    HPOBackendProtocol,
    get_backend,
)
from .optuna_backend import OptunaBackend

__all__ = [
    # Protocol
    "HPOBackendProtocol",
    # Factory function
    "get_backend",
    # Backend implementations
    "OptunaBackend",
    # Availability flags
    "OPTUNA_AVAILABLE",
]
