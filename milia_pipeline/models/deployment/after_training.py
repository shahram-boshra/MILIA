"""Deployment at the end of training (``models.deployment.enabled: true``).

After training and the test evaluation, MILIA:

1. **Gates** the best checkpoint (:func:`evaluate_gate`): ``gate.metric`` must meet
   ``gate.threshold`` (when set) and, with ``gate.compare_to_current``, beat the same metric recorded
   by the newest published version of this model. ``gate.mode`` says whether lower (``min``) or
   higher (``max``) is better. The decision is written to ``gate.json`` next to the training
   results, whether it passes or not.
2. When the gate passes, **deploys** it (:func:`.deploy.deploy`): a verified, versioned package with
   the target's ``Dockerfile`` and, with ``onnx.enabled``, ``model.onnx``.

A failed gate is a valid outcome, not a failed run: nothing is deployed, a warning is logged and
training still succeeds. The same pattern as TFX's Pusher, which pushes only a blessed model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.deployment.deploy import DeployResult, deploy
from milia_pipeline.models.deployment.package import (
    PACKAGE_MANIFEST,
    checkpoint_scores,
    manifest_scores,
    model_folder_name,
)
from milia_pipeline.models.deployment.settings import DeploymentSettings, GateSettings

logger = logging.getLogger(__name__)

__all__ = ["GATE_FILE", "GateDecision", "deploy_after_training", "evaluate_gate"]

GATE_FILE = "gate.json"


@dataclass(frozen=True)
class GateDecision:
    """Whether a checkpoint may be deployed, and why (written to ``gate.json``)."""

    checkpoint: str
    metric: str
    mode: str
    value: float | None
    threshold: float | None
    compare_to_current: bool
    current_version: str | None
    current_value: float | None
    passed: bool
    reasons: list[str] = field(default_factory=list)
    decided_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _better(value: float, reference: float, mode: str) -> bool:
    return value < reference if mode == "min" else value > reference


def _at_least(value: float, threshold: float, mode: str) -> bool:
    return value <= threshold if mode == "min" else value >= threshold


def _newest_version(model_dir: Path) -> Path | None:
    if not model_dir.is_dir():
        return None
    versions = [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(versions, key=lambda p: int(p.name), default=None)


def _model_folder(checkpoint: Path) -> str:
    hyper_parameters = (load_checkpoint_file(checkpoint, map_location="cpu") or {}).get(
        "hyper_parameters"
    ) or {}
    return model_folder_name(str(hyper_parameters.get("model_name", "model")))


def evaluate_gate(
    checkpoint: str | Path, gate: GateSettings, output_dir: str | Path
) -> GateDecision:
    """Decide whether ``checkpoint`` may be deployed under ``output_dir``.

    Raises:
        DeploymentError: The checkpoint cannot be read.
    """
    checkpoint = Path(checkpoint)
    scores = checkpoint_scores(checkpoint)
    value = scores.get(gate.metric)
    reasons: list[str] = []
    current_version = current_value = None

    if value is None:
        reasons.append(
            f"the checkpoint records no {gate.metric!r} (it records: {', '.join(sorted(scores))})"
        )
    if (
        value is not None
        and gate.threshold is not None
        and not _at_least(value, gate.threshold, gate.mode)
    ):
        bound = "at most" if gate.mode == "min" else "at least"
        reasons.append(f"{gate.metric} = {value:.6g}; the threshold is {bound} {gate.threshold:g}")
    if gate.compare_to_current:
        newest = _newest_version(Path(output_dir) / _model_folder(checkpoint))
        if newest is not None:
            current_version = str(newest)
            manifest = json.loads((newest / PACKAGE_MANIFEST).read_text())
            current_value = manifest_scores(manifest.get("metrics")).get(gate.metric)
            if current_value is None:
                logger.info(f"{newest} records no {gate.metric}: nothing to compare with")
            elif value is not None and not _better(value, current_value, gate.mode):
                reasons.append(
                    f"{gate.metric} = {value:.6g} does not beat {current_value:.6g} of {newest}"
                )
    return GateDecision(
        checkpoint=str(checkpoint),
        metric=gate.metric,
        mode=gate.mode,
        value=value,
        threshold=gate.threshold,
        compare_to_current=gate.compare_to_current,
        current_version=current_version,
        current_value=current_value,
        passed=not reasons,
        reasons=reasons,
    )


def deploy_after_training(
    checkpoint: str | Path,
    settings: DeploymentSettings,
    working_root_dir: str | Path,
    results_dir: str | Path,
) -> tuple[GateDecision, DeployResult | None]:
    """Gate ``checkpoint`` and deploy it when the gate passes; write ``gate.json``.

    Returns:
        The gate decision, and the deployment (``None`` when the gate failed).

    Raises:
        DeploymentError: Reading the checkpoint, packaging or verification failed.
        ExportError: The requested ONNX export failed.
    """
    output_dir = Path(settings.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(working_root_dir) / output_dir
    decision = evaluate_gate(checkpoint, settings.gate, output_dir)
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    (results / GATE_FILE).write_text(json.dumps(asdict(decision), indent=2) + "\n")
    if not decision.passed:
        logger.warning(
            f"Deployment skipped: the gate on {decision.metric} failed "
            f"({'; '.join(decision.reasons)}). Decision: {results / GATE_FILE}"
        )
        return decision, None
    return decision, deploy(checkpoint, settings, working_root_dir)
