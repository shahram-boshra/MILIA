"""What a training run records about itself in every checkpoint it writes (``run_info``).

A checkpoint is packaged and served long after its run, possibly by another MILIA version and
without the run's configuration or data. ``run_info`` keeps what describing the model needs and
cannot be recovered from the weights: the dataset, the target names, a hash of the
resolved configuration, the data split, the source data's declared units, the code version (MILIA, git commit, libraries) and a few
training molecules as SMILES (the package's input example). MLflow records the same facts with a
logged model: its signature and input example, and the run's source commit.

:func:`collect_run_info` never raises for a missing optional fact: it records ``None`` instead,
so the manifest and model card can say "not recorded".
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import platform
import subprocess
from collections.abc import Iterable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RUN_INFO_VERSION",
    "collect_run_info",
    "config_sha256",
    "example_smiles",
    "target_names_for",
]

RUN_INFO_VERSION = 1
MAX_INPUT_EXAMPLES = 5
_MAX_SCANNED = 50  # molecules read to find the examples
_LIBRARIES = ("milia-py", "torch", "torch_geometric", "rdkit")


def config_sha256(config: Any) -> str:
    """SHA-256 of the resolved configuration as canonical JSON (sorted keys, ``str`` fallback)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def example_smiles(graphs: Iterable[Any], limit: int = MAX_INPUT_EXAMPLES) -> list[str]:
    """Up to ``limit`` SMILES of the given molecules (their ``smiles``, else their ``inchi``).

    An InChI is converted with RDKit; a molecule RDKit cannot read is skipped. At most
    50 molecules are read.
    """
    found: list[str] = []
    for graph in itertools.islice(graphs, _MAX_SCANNED):
        if len(found) >= limit:
            break
        smiles = getattr(graph, "smiles", None)
        if not (isinstance(smiles, str) and smiles and smiles != "N/A"):
            smiles = _smiles_from_inchi(getattr(graph, "inchi", None))
        if smiles:
            found.append(smiles)
    return found


def _smiles_from_inchi(inchi: Any) -> str | None:
    if not (isinstance(inchi, str) and inchi.startswith("InChI=")):
        return None
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromInchi(inchi)
        finally:
            RDLogger.EnableLog("rdApp.*")
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def target_names_for(selection: dict[str, Any] | None, sample: Any) -> list[str] | None:
    """Names of the model's outputs, in order, or ``None`` when the run cannot tell.

    The dataset stores the names of ``y``'s columns on each molecule (``y_property_names``). A
    target selection (``TargetSelectionConfig.to_dict()``) keeps its ``resolved_names``, or the
    columns at its ``resolved_indices``.
    """
    available = getattr(sample, "y_property_names", None)
    available = list(available) if isinstance(available, list | tuple) and available else None
    if not selection:
        return available
    if selection.get("resolved_names"):
        return list(selection["resolved_names"])
    indices = selection.get("resolved_indices")
    if available and indices and all(0 <= i < len(available) for i in indices):
        return [available[i] for i in indices]
    return None


def _git_commit() -> str | None:
    """The commit of the MILIA source tree, when it runs from a git checkout."""
    source = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and len(commit) == 40 else None


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _dataset_units(dataset_type: str | None) -> dict[str, str | None]:
    """The units of a registered dataset's source data (``DatasetSchema``), or ``None`` each.

    These are the raw data's units; a configured transformation may convert the targets.
    """
    units: dict[str, str | None] = {"energy": None, "coordinates": None}
    if not dataset_type:
        return units
    try:
        from milia_pipeline.datasets.registry import get as get_dataset

        schema = get_dataset(dataset_type).schema
        units["energy"] = getattr(schema, "energy_units", None)
        units["coordinates"] = getattr(schema, "coordinate_units", None)
    except Exception as exc:
        logger.debug(f"No units recorded for dataset {dataset_type!r}: {exc}")
    return units


def collect_run_info(
    config: dict[str, Any],
    *,
    task_type: str,
    out_channels: int | None,
    target_names: Sequence[str] | None,
    data_split: dict[str, Any] | None,
    train_graphs: Iterable[Any] = (),
) -> dict[str, Any]:
    """The ``run_info`` entry of a training run's checkpoints.

    Args:
        config: The resolved configuration the run uses.
        task_type: The run's task type.
        out_channels: The model's output width.
        target_names: Names of the model's outputs in order; recorded only when there is one name
            per output.
        data_split: The split settings (ratios, seed).
        train_graphs: Training molecules; up to five become the input example.
    """
    dataset_type = config.get("dataset_type")
    names = list(target_names) if target_names else None
    if names is not None and out_channels is not None and len(names) != out_channels:
        logger.warning(
            f"{len(names)} target names for {out_channels} outputs; target names not recorded"
        )
        names = None
    return {
        "run_info_version": RUN_INFO_VERSION,
        "dataset": {"type": dataset_type, "source_units": _dataset_units(dataset_type)},
        "task_type": task_type,
        "targets": {"names": names, "count": out_channels},
        "data_split": dict(data_split) if data_split else None,
        "config_sha256": config_sha256(config),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "input_example": example_smiles(train_graphs),
    }
