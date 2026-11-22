"""Regression targets standardized with statistics of the training split, undone at prediction.

A network trains better on targets of unit scale, and on several targets of very different
magnitude (a total energy in Hartree next to a dipole in Debye) an unscaled loss is dominated by
the largest. The scaler is fit on the training split only, so no validation or test statistics
reach the model (scikit-learn ``TransformedTargetRegressor``: ``transformer.fit`` on the training
targets, ``inverse_transform`` of every prediction; Chemprop: target scaler fit on the training
data, its inverse stored with the model and applied at inference).

Training (``Trainer``) computes the loss on standardized targets, and so does ``val_loss``, the
validation objective that early stopping, the best checkpoint and the deploy gate use: every
target weighs alike, whatever its units (Chemprop scores its validation loss on the scaled
targets, with the scaler fit on the training split). The other validation metrics, the test
metrics, conformal calibration and prediction work in the targets' own units (the model's outputs
are unscaled first), overall and per target (``val_mae_<target>``, ``test_rmse_<target>``).
The scaler is stored in every checkpoint as ``target_scaler``::

    {"method": "standard", "n": 1200, "mean": [...], "std": [...]}   # one value per target

As ``StandardScaler`` (scikit-learn, Chemprop): per-target mean and population standard
deviation over the training molecules (or nodes, for node-level tasks), missing values (NaN)
ignored; a target with zero or undefined spread keeps std 1.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "TARGET_SCALING_METHODS",
    "fit_target_scaler",
    "scale_targets",
    "unscale_outputs",
    "unscale_spread",
]

TARGET_SCALING_METHODS = ("standard",)


def fit_target_scaler(targets: torch.Tensor, method: str = "standard") -> dict[str, Any]:
    """Fit the scaler on training targets (``[n]`` or ``[n, k]``; one column per target).

    Raises:
        ValueError: An unknown method, or no targets.
    """
    if method not in TARGET_SCALING_METHODS:
        raise ValueError(
            f"Unknown target scaling {method!r}; expected one of {TARGET_SCALING_METHODS}"
        )
    rows = targets.detach().double()
    rows = rows.reshape(rows.size(0), -1) if rows.dim() > 1 else rows.reshape(-1, 1)
    if rows.size(0) == 0:
        raise ValueError("Cannot fit a target scaler on no targets")
    mean = torch.nanmean(rows, dim=0)
    std = torch.sqrt(torch.nanmean((rows - mean) ** 2, dim=0))
    mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
    std = torch.where(torch.isnan(std) | (std == 0), torch.ones_like(std), std)
    return {
        "method": method,
        "n": int(rows.size(0)),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def _stats(scaler: dict[str, Any], like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(scaler["mean"], dtype=like.dtype, device=like.device)
    std = torch.tensor(scaler["std"], dtype=like.dtype, device=like.device)
    width = like.size(-1) if like.dim() > 1 else 1  # a 1-D tensor holds one target
    if width != mean.numel():
        raise ValueError(
            f"The target scaler has {mean.numel()} targets; the tensor has {width} columns"
        )
    if like.dim() <= 1:
        return mean.reshape(()), std.reshape(())
    return mean, std


def scale_targets(targets: torch.Tensor, scaler: dict[str, Any] | None) -> torch.Tensor:
    """``(y - mean) / std`` per target (``scaler`` ``None``: unchanged)."""
    if scaler is None:
        return targets
    mean, std = _stats(scaler, targets)
    return (targets - mean) / std


def unscale_outputs(outputs: torch.Tensor, scaler: dict[str, Any] | None) -> torch.Tensor:
    """``y_scaled * std + mean`` per target: model outputs in the targets' own units."""
    if scaler is None:
        return outputs
    mean, std = _stats(scaler, outputs)
    return outputs * std + mean


def unscale_spread(spread: torch.Tensor | None, scaler: dict[str, Any] | None) -> Any:
    """A spread (standard deviation) of model outputs in the targets' own units."""
    if spread is None or scaler is None:
        return spread
    _, std = _stats(scaler, spread)
    return spread * std
