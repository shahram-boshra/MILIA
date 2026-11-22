"""Input drift: requests compared with the distribution the model was trained on.

Training records a reference of its inputs (:func:`build_input_reference`, stored as
``input_reference`` in the checkpoint and the package manifest): for every node-feature column of
``x`` (after the recorded pre-transforms, as the model saw it) and for the number of atoms per
molecule, a histogram of the training split. Serving (:class:`DriftMonitor`) keeps the same
histograms over a sliding window of the most recent molecules and, once the window is full,
compares each feature with the reference.

Method (the defaults of the monitoring services):

* A feature with at most ``CATEGORICAL_MAX_VALUES`` distinct reference values is categorical (its
  values plus one bucket for values the reference never had); any other is numerical, binned at
  the reference deciles with open outer bins.
* The distance is the Jensen-Shannon distance (base 2, in [0, 1]) between the window's and the
  reference's bin frequencies, for both kinds (Vertex AI Model Monitoring scores numerical
  features with the Jensen-Shannon divergence; Evidently's default for categorical features above
  1000 observations is the Jensen-Shannon distance with threshold 0.1).
* A feature drifts when its distance reaches the threshold (``MILIA_DRIFT_THRESHOLD``, default
  ``0.1``). Distances are computed only once the window holds ``MILIA_DRIFT_WINDOW`` molecules
  (default 1000): on fewer, sampling noise alone moves a 10-bin distance by about 0.1. Distances
  are descriptive, not tests: atoms of one molecule are not independent observations. The
  reference is noisy in the same way when few molecules trained the model (the report states
  ``reference_molecules``).

Each change of the drift status is logged as one JSON line (logger
``milia_pipeline.models.deployment.drift``); ``GET /monitoring/drift`` returns the current report.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from milia_pipeline.exceptions import DeploymentError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW",
    "DRIFT_METHOD",
    "DriftMonitor",
    "build_input_reference",
    "drift_settings",
    "jensen_shannon_distance",
]

DRIFT_METHOD = "jensen_shannon_distance"
REFERENCE_VERSION = 1
DEFAULT_THRESHOLD = 0.1
DEFAULT_WINDOW = 1000
CATEGORICAL_MAX_VALUES = 10
QUANTILES = [i / 10 for i in range(1, 10)]  # decile edges of a numerical feature
MAX_REFERENCE_ROWS = 100_000  # uniform sample of the training atoms (and molecules) kept


def _keep_sample(kept: torch.Tensor | None, keys: torch.Tensor | None, rows, generator):
    """Bottom-k sampling: rows with the ``MAX_REFERENCE_ROWS`` smallest random keys (uniform)."""
    new_keys = torch.rand(rows.size(0), generator=generator)
    rows = rows if kept is None else torch.cat([kept, rows])
    new_keys = new_keys if keys is None else torch.cat([keys, new_keys])
    if rows.size(0) > MAX_REFERENCE_ROWS:
        order = torch.topk(new_keys, MAX_REFERENCE_ROWS, largest=False).indices
        rows, new_keys = rows[order], new_keys[order]
    return rows, new_keys


def _feature(values: torch.Tensor) -> dict[str, Any]:
    values = values.double()
    distinct = torch.unique(values)
    if distinct.numel() <= CATEGORICAL_MAX_VALUES:
        counts = torch.stack([(values == v).sum() for v in distinct]).double()
        probabilities = (counts / counts.sum()).tolist() + [0.0]  # + values never seen
        return {"kind": "categorical", "values": distinct.tolist(), "probabilities": probabilities}
    edges = torch.unique(torch.quantile(values, torch.tensor(QUANTILES, dtype=values.dtype)))
    counts = torch.bincount(torch.bucketize(values, edges, right=True), minlength=edges.numel() + 1)
    counts = counts.double()
    return {
        "kind": "numerical",
        "edges": edges.tolist(),
        "probabilities": (counts / counts.sum()).tolist(),
    }


def build_input_reference(batches: Iterable[Any], seed: int = 0) -> dict[str, Any] | None:
    """The training inputs' reference: one histogram per ``x`` column and for atoms per molecule.

    Args:
        batches: PyG batches of the training split (``x``; ``batch`` for several molecules).
        seed: Seed of the sampling generator (deterministic reference).

    Returns:
        ``{"version", "method", "molecules", "atoms", "features": {name: histogram}}``, or
        ``None`` when the batches have no node features.
    """
    generator = torch.Generator().manual_seed(seed)
    rows = row_keys = sizes = size_keys = None
    atoms = molecules = 0
    for batch in batches:
        x = getattr(batch, "x", None)
        if x is None or x.dim() != 2 or x.size(0) == 0:
            continue
        x = x.detach().float().cpu()
        index = getattr(batch, "batch", None)
        per_molecule = (
            torch.bincount(index.cpu()).double()
            if index is not None
            else torch.tensor([float(x.size(0))], dtype=torch.float64)
        )
        atoms += x.size(0)
        molecules += per_molecule.numel()
        rows, row_keys = _keep_sample(rows, row_keys, x, generator)
        sizes, size_keys = _keep_sample(sizes, size_keys, per_molecule, generator)
    if rows is None:
        return None
    features = {f"x_{i}": _feature(rows[:, i]) for i in range(rows.size(1))}
    features["num_atoms"] = _feature(sizes)
    return {
        "version": REFERENCE_VERSION,
        "method": DRIFT_METHOD,
        "molecules": molecules,
        "atoms": atoms,
        "features": features,
    }


def _bin_counts(histogram: dict[str, Any], values: torch.Tensor) -> torch.Tensor:
    values = values.double()
    bins = len(histogram["probabilities"])
    if histogram["kind"] == "categorical":
        known = torch.tensor(histogram["values"], dtype=torch.float64)
        match = values.unsqueeze(-1) == known
        index = torch.where(match.any(-1), match.double().argmax(-1), torch.tensor(bins - 1))
    else:
        index = torch.bucketize(
            values, torch.tensor(histogram["edges"], dtype=torch.float64), right=True
        )
    return torch.bincount(index, minlength=bins).double()


def jensen_shannon_distance(
    p: Sequence[float] | torch.Tensor, q: Sequence[float] | torch.Tensor
) -> float:
    """Jensen-Shannon distance (base 2) of two frequency vectors: 0 (same) to 1 (disjoint)."""
    p = torch.as_tensor(p, dtype=torch.float64)
    q = torch.as_tensor(q, dtype=torch.float64)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def _kl(a: torch.Tensor) -> torch.Tensor:
        mask = a > 0
        return (a[mask] * torch.log2(a[mask] / m[mask])).sum()

    divergence = max(float((_kl(p) + _kl(q)) / 2), 0.0)
    return math.sqrt(min(divergence, 1.0))


def drift_settings() -> tuple[float, int]:
    """``MILIA_DRIFT_THRESHOLD`` (0 < t <= 1) and ``MILIA_DRIFT_WINDOW`` (molecules, >= 1).

    Raises:
        DeploymentError: A value that is not a number in range.
    """
    raw_threshold = os.environ.get("MILIA_DRIFT_THRESHOLD", str(DEFAULT_THRESHOLD))
    raw_window = os.environ.get("MILIA_DRIFT_WINDOW", str(DEFAULT_WINDOW))
    try:
        threshold, window = float(raw_threshold), int(raw_window)
    except ValueError as exc:
        raise DeploymentError(
            f"MILIA_DRIFT_THRESHOLD must be a number and MILIA_DRIFT_WINDOW an integer, got "
            f"{raw_threshold!r} and {raw_window!r}"
        ) from exc
    if not 0 < threshold <= 1 or window < 1:
        raise DeploymentError(
            f"MILIA_DRIFT_THRESHOLD must be in (0, 1] and MILIA_DRIFT_WINDOW at least 1, got "
            f"{threshold} and {window}"
        )
    return threshold, window


class DriftMonitor:
    """Sliding-window histograms of served molecules, compared with the training reference.

    Thread-safe: the server featurizes requests in a thread pool.
    """

    def __init__(
        self,
        reference: dict[str, Any] | None,
        threshold: float = DEFAULT_THRESHOLD,
        window: int = DEFAULT_WINDOW,
    ):
        if reference is not None and reference.get("version") != REFERENCE_VERSION:
            raise DeploymentError(
                f"Unsupported input reference version {reference.get('version')!r} "
                f"(this MILIA reads version {REFERENCE_VERSION})"
            )
        self.reference = reference
        self.threshold = threshold
        self.window = window
        self._lock = threading.Lock()
        self._molecules: deque[dict[str, torch.Tensor]] = deque()
        self._totals: dict[str, torch.Tensor] = {}
        self._status = "not_recorded" if reference is None else "collecting"

    def observe(self, graphs: Sequence[Any]) -> None:
        """Add featurized (and pre-transformed) molecules to the window."""
        if self.reference is None:
            return
        features = self.reference["features"]
        columns = [name for name in features if name.startswith("x_")]
        for graph in graphs:
            x = graph.x.detach().float().cpu()
            counts = {
                name: _bin_counts(features[name], x[:, int(name[2:])])
                for name in columns
                if int(name[2:]) < x.size(1)
            }
            counts["num_atoms"] = _bin_counts(
                features["num_atoms"], torch.tensor([float(x.size(0))])
            )
            with self._lock:
                self._molecules.append(counts)
                for name, value in counts.items():
                    self._totals[name] = self._totals.get(name, 0) + value
                if len(self._molecules) > self.window:
                    for name, value in self._molecules.popleft().items():
                        self._totals[name] = self._totals[name] - value
        self._log_status_change()

    def report(self) -> dict[str, Any]:
        """The drift status, the distance of each feature and the features that drift."""
        with self._lock:
            molecules = len(self._molecules)
            totals = {name: value.clone() for name, value in self._totals.items()}
        report: dict[str, Any] = {
            "method": DRIFT_METHOD,
            "threshold": self.threshold,
            "window": self.window,
            "molecules": molecules,
            # the training molecules behind the reference: few make the reference itself noisy
            "reference_molecules": (self.reference or {}).get("molecules"),
        }
        if self.reference is None:
            return {**report, "status": "not_recorded", "drifted": [], "distances": {}}
        if molecules < self.window:
            return {**report, "status": "collecting", "drifted": [], "distances": {}}
        features = self.reference["features"]
        distances = {
            name: round(jensen_shannon_distance(totals[name], features[name]["probabilities"]), 6)
            for name in features
            if name in totals and float(totals[name].sum()) > 0
        }
        drifted = sorted(name for name, d in distances.items() if d >= self.threshold)
        return {
            **report,
            "status": "drift" if drifted else "no_drift",
            "drifted": drifted,
            "distances": distances,
        }

    def _log_status_change(self) -> None:
        report = self.report()
        if report["status"] == self._status:
            return
        self._status = report["status"]
        record = {"event": "input_drift", **{k: report[k] for k in ("status", "molecules")}}
        record["drifted"] = {name: report["distances"][name] for name in report["drifted"]}
        level = logging.WARNING if report["status"] == "drift" else logging.INFO
        logger.log(level, json.dumps(record, sort_keys=True))
