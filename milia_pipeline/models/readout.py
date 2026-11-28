"""The graph-level readout: how a model's per-atom outputs become one prediction per molecule.

A graph-level task (``graph_*``) with a model whose output is per node (PyG's message-passing
models, model plugins, MILIA's ensemble) pools those outputs over each molecule's atoms:

* ``mean``: the average; independent of how many atoms a molecule has (an intensive property);
* ``add``: the sum; grows with the atoms present (an extensive property: a total energy is a sum
  of atomic contributions);
* ``max``: the per-feature maximum.

A mean "captures the distribution (proportions) of elements in a multiset, but not the exact
multiset", while a sum can tell multisets apart (Xu et al., arXiv:1810.00826); SchNet sums atomic
contributions for extensive properties and averages them for orbital energies (arXiv:1712.06113).
A model with its own readout (SchNet, DimeNet: ``produces_graph_output``) keeps it.

The readout is ``models.selection.pooling_method`` (default ``mean``), validated when the
configuration loads; an explicit ``pooling_method`` in a model's hyperparameters, or in the
ensemble configuration, takes precedence. :func:`with_readout` puts it where the model factory
reads it, for standard training and HPO alike; the factory records it in a model's
``model_info["readout"]`` and its hyperparameters, so a model rebuilt from its checkpoint
(prediction, packages, serving, the registry) pools as it was trained.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

__all__ = [
    "DEFAULT_READOUT",
    "READOUTS",
    "READOUT_KEY",
    "configured_readout",
    "global_pool",
    "validate_readout",
    "with_readout",
]

READOUT_KEY = "pooling_method"
DEFAULT_READOUT = "mean"
READOUTS = ("mean", "add", "max")
ENSEMBLE = "ensemble"


def validate_readout(name: Any, where: str = READOUT_KEY) -> str:
    """``name`` if it is a readout MILIA has, else :class:`ValueError` naming the options."""
    if not isinstance(name, str) or name not in READOUTS:
        raise ValueError(f"{where}: {name!r} is not a readout; use one of {', '.join(READOUTS)}")
    return name


def _pool(name: str) -> Callable[..., torch.Tensor]:
    from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

    return {"mean": global_mean_pool, "add": global_add_pool, "max": global_max_pool}[name]


def global_pool(
    x: torch.Tensor, batch: torch.Tensor | None, readout: str, size: int | None = None
) -> torch.Tensor:
    """Pool node rows ``x`` to one row per graph with ``readout`` (validated).

    ``batch`` maps each node to its graph; ``None`` is one graph. ``size`` (the graph count)
    saves PyG inferring it from ``batch`` (a device-to-host sync).
    """
    pool = _pool(validate_readout(readout))
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        size = 1
    return pool(x, batch, size=size)


def configured_readout(models_config: Mapping[str, Any] | None) -> str:
    """``models.selection.pooling_method`` validated, else the default.

    Raises:
        ValueError: The configured value is not a readout.
    """
    selection = (models_config or {}).get("selection") or {}
    if READOUT_KEY not in selection or selection[READOUT_KEY] is None:
        return DEFAULT_READOUT
    return validate_readout(selection[READOUT_KEY], f"models.selection.{READOUT_KEY}")


def with_readout(
    model_name: str, hyperparameters: Mapping[str, Any], readout: str
) -> dict[str, Any]:
    """``hyperparameters`` with ``readout`` where the factory reads it, unless one is set there.

    A single model reads ``hyperparameters["pooling_method"]``; MILIA's ensemble reads its
    ``ensemble_config["pooling_method"]``. An explicit value is kept (validated).

    Raises:
        ValueError: ``readout``, or an explicit value, is not a readout.
    """
    validate_readout(readout)
    result = dict(hyperparameters)
    if str(model_name).lower() == ENSEMBLE:
        ensemble = dict(result.get("ensemble_config") or {})
        ensemble[READOUT_KEY] = validate_readout(
            ensemble.get(READOUT_KEY, readout), f"ensemble_config.{READOUT_KEY}"
        )
        result["ensemble_config"] = ensemble
    else:
        result[READOUT_KEY] = validate_readout(result.get(READOUT_KEY, readout))
    return result
