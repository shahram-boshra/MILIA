# milia_pipeline/models/hpo/callbacks/__init__.py

"""
HPO Callbacks Subpackage

Provides callback implementations for integrating HPO backends with MILIA's
training system. Callbacks report intermediate metrics and handle pruning
decisions during hyperparameter optimization trials.

Primary Exports:
    - OptunaPruningCallback: Callback for Optuna trial pruning
    - create_hpo_callback: Factory function for backend-specific callbacks

Availability Flags:
    - OPTUNA_AVAILABLE: True if Optuna is installed

Pattern: Follows handlers/__init__.py export structure
Reference: MILIA_HPO_Implementation_Blueprint.md (lines 267, 325-329)

Usage:
    >>> from milia_pipeline.models.hpo.callbacks import (
    ...     OptunaPruningCallback,
    ...     create_hpo_callback,
    ... )
    >>>
    >>> # Direct usage with Optuna trial
    >>> callback = OptunaPruningCallback(trial=trial, monitor="val_loss")
    >>>
    >>> # Factory function usage
    >>> callback = create_hpo_callback(
    ...     trial=trial,
    ...     monitor="val_loss",
    ...     backend="optuna"
    ... )

Integration Point:
    These callbacks are automatically created by HPOManager and injected
    into the Trainer's callback list during HPO trials.
"""

# =============================================================================
# IMPORTS FROM OPTUNA CALLBACK MODULE
# =============================================================================

from .optuna_callback import (
    OPTUNA_AVAILABLE,
    OptunaPruningCallback,
    create_hpo_callback,
)

# =============================================================================
# FUTURE: RAY TUNE CALLBACK (INACTIVE)
# =============================================================================
# When Ray Tune backend is activated, uncomment:
# from .ray_tune_callback import (
#     RayTuneReportCallback,
#     RAY_TUNE_AVAILABLE,
# )

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Primary callback class
    "OptunaPruningCallback",
    # Factory function
    "create_hpo_callback",
    # Availability flag
    "OPTUNA_AVAILABLE",
]

# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "MILIA Team"
