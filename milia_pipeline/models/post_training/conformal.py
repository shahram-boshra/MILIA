"""Split conformal prediction intervals for MILIA regression models.

After training, the best checkpoint predicts the validation set (the calibration set, which
training did not fit). For each target column the nonconformity scores are sorted and the
conformal quantile is q̂ = the ⌈(n+1)(1−α)⌉-th smallest score. A new molecule's interval is

- ``ŷ ± q̂`` for a single model (score ``|y − ŷ|``);
- ``ŷ ± q̂·σ̂`` for a :class:`ParallelEnsemble` (score ``|y − ŷ| / σ̂``, σ̂ = the standard deviation
  across its members), so the width follows the members' disagreement.

Under exchangeability of the calibration set and the new molecules, each target's interval
covers the true value with probability at least 1−α, whatever the model (split conformal;
Angelopoulos & Bates, https://arxiv.org/abs/2107.07511). The guarantee is marginal (on average
over molecules), holds per target and not jointly, and says nothing about molecules unlike the
calibration set. The members of a MILIA ensemble are trained jointly, so their spread on its own
understates the error (Jeffares et al. 2023, https://arxiv.org/abs/2301.11323); conformal scaling
is what makes it usable.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "CALIBRATION_METHODS",
    "DEFAULT_ALPHA",
    "conformal_quantiles",
    "intervals",
    "member_spread",
    "recording_members",
]

DEFAULT_ALPHA = 0.1
CALIBRATION_METHODS = ("split_conformal_absolute", "split_conformal_normalized")
_MIN_SPREAD = 1e-12


def _output_ensemble(model: nn.Module) -> nn.Module | None:
    """The ParallelEnsemble whose fused output is the model's output (the model itself)."""
    from milia_pipeline.models.builders.model_composer import ParallelEnsemble
    from milia_pipeline.models.training.module_utils import unwrap_compiled

    root = unwrap_compiled(model)
    return root if isinstance(root, ParallelEnsemble) and root.readout else None


@contextlib.contextmanager
def recording_members(model: nn.Module) -> Iterator[nn.Module | None]:
    """Record the member outputs of the model's output ensemble during the block.

    Yields the ensemble, or ``None`` when the model is not a :class:`ParallelEnsemble`.
    """
    ensemble = _output_ensemble(model)
    if ensemble is None:
        yield None
        return
    ensemble.record_members, ensemble.last_member_outputs = True, None
    try:
        yield ensemble
    finally:
        ensemble.record_members, ensemble.last_member_outputs = False, None


def member_spread(ensemble: nn.Module | None) -> torch.Tensor | None:
    """Standard deviation across the members of the last recorded forward pass."""
    outputs = getattr(ensemble, "last_member_outputs", None)
    if outputs is None or outputs.size(0) < 2:
        return None
    return outputs.float().std(dim=0, unbiased=True).clamp_min(_MIN_SPREAD)


def conformal_quantiles(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    alpha: float = DEFAULT_ALPHA,
    spread: torch.Tensor | None = None,
) -> dict[str, Any]:
    """The split conformal calibration of a model on a calibration set.

    Args:
        targets, predictions: ``[n, k]`` (or ``[n]``) true and predicted values.
        alpha: Miscoverage level, ``0 < alpha < 1``.
        spread: ``[n, k]`` member spread; given, the score is normalized by it.

    Returns:
        ``{"method", "alpha", "n", "quantiles"}``; a quantile is ``inf`` when
        ⌈(n+1)(1−α)⌉ > n (too few calibration samples for this α).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    targets = targets.detach().float().reshape(predictions.size(0), -1)
    predictions = predictions.detach().float().reshape(targets.shape)
    scores = (targets - predictions).abs()
    method = CALIBRATION_METHODS[0]
    if spread is not None:
        scores = scores / spread.detach().float().reshape(targets.shape)
        method = CALIBRATION_METHODS[1]
    n = scores.size(0)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        quantiles = [math.inf] * scores.size(1)
    else:
        quantiles = scores.sort(dim=0).values[rank - 1].tolist()
    return {"method": method, "alpha": float(alpha), "n": int(n), "quantiles": quantiles}


def intervals(
    predictions: torch.Tensor, calibration: dict[str, Any], spread: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lower, upper)`` for predictions ``[n, k]`` under a stored calibration.

    Raises:
        ValueError: A normalized calibration without a member spread, or a column count that
            differs from the calibration's.
    """
    predictions = predictions.detach().float()
    q = torch.tensor(calibration["quantiles"], dtype=predictions.dtype)
    rows = predictions.reshape(predictions.size(0), -1)
    if rows.size(1) != q.numel():
        raise ValueError(
            f"The calibration has {q.numel()} targets; the predictions have {rows.size(1)}"
        )
    if calibration["method"] == CALIBRATION_METHODS[1]:
        if spread is None:
            raise ValueError("A normalized calibration needs the ensemble's member spread")
        half = q * spread.detach().float().reshape(rows.shape)
    else:
        half = q.expand_as(rows)
    return rows - half, rows + half
