"""
Self-describing deployment packages built from MILIA training checkpoints.

A bare ``state_dict`` cannot be served: rebuilding the network needs the model class and the
hyperparameters it was created with. A MILIA v2 checkpoint carries both
(``hyper_parameters``: model name, task type, the complete hyperparameters dict and
``model_info``), and ``ModelLoader``
(``milia_pipeline.models.post_training.inference.model_loader``) rebuilds the model from it.
A package is a directory holding:

* ``model.pt``: the checkpoint reduced to what inference needs (model weights, recreation
  metadata, data compatibility and version information), with no optimizer, scheduler or
  training history;
* ``deployment.json`` (format ``milia-deployment/3``): the MILIA version that wrote it, the model
  name and task type and the SHA-256 of ``model.pt``, plus

  - ``signature``: the inputs (the input kinds the model accepts and why others are refused,
    ``models/input_contract.py``; identifiers are featurized with the run's
    ``structural_features_config`` into ``node_features`` columns) and the outputs (count, names,
    task type);
  - ``input_example``: up to five training molecules as SMILES;
  - ``structure_example``: up to five training molecules as structure inputs (atomic numbers,
    coordinates in ångström, charge, identifier; ``StructureInput.to_dict``), for a model that
    accepts structures;
  - ``metrics``: the checkpoint's epoch, its monitored metric, its validation metrics at that
    epoch, and its test metrics (when training tested this checkpoint);
  - ``lineage``: the source checkpoint's SHA-256, when and with which versions it was written,
    the dataset, the data split, the configuration's SHA-256 and the git commit;

* ``MODEL_CARD.md``: the same facts for people (model details, inputs and outputs, training data,
  evaluation, limitations, how to load).

The signature, input example and lineage come from the checkpoint's ``run_info``, which MILIA's
trainer records (``milia_pipeline.models.training.run_info``). A checkpoint without
it still packages; those fields are ``null`` and the card says "not recorded".

:func:`package_checkpoint` verifies what it wrote: the package, loaded as the server loads it,
must predict its examples as the source checkpoint does, or its files are removed (the input
example for a model that accepts identifiers, the structure example for one that accepts
structures; the checksums are verified for every package). A model that accepts structures but
whose checkpoint records no structure example still packages, and is not served structures.
:func:`publish_package` writes each package to a new integer version folder of its model
family (``<output_dir>/<family>/<N>/``) through a temporary sibling and a rename, so a version is
either complete and verified or absent, and is never overwritten. A family (:func:`model_family`)
is what the model predicts: the dataset type, the task type, the targets (names and count) and
the source data's units. Versions of one family may come from different architectures, as the
versions of one registered model in a registry (MLflow: "A registered model has a unique name,
contains versions, aliases, tags"); the folder records its family in ``family.json``.

:func:`load_package` checks the format (``milia-deployment/1`` and ``/2`` packages still load) and
the digest, then rebuilds the model with ``ModelLoader``. Every read uses
``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.checkpoint_io import load_checkpoint_file
from milia_pipeline.models.input_contract import IDENTIFIER, STRUCTURE, InputContract

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_SMILES_LENGTH",
    "checkpoint_scores",
    "manifest_scores",
    "MODEL_CARD",
    "PACKAGE_CHECKPOINT",
    "PACKAGE_FORMAT",
    "PACKAGE_MANIFEST",
    "SUPPORTED_FORMATS",
    "is_package",
    "load_package",
    "model_card",
    "FAMILY_FILE",
    "checkpoint_family",
    "manifest_family",
    "model_family",
    "model_folder_name",
    "package_checkpoint",
    "publish_package",
    "verify_package",
]

# 3: signature.inputs.kinds (the input contract); 4: structure_example (verified when recorded)
PACKAGE_FORMAT = "milia-deployment/4"
SUPPORTED_FORMATS = (
    "milia-deployment/1",
    "milia-deployment/2",
    "milia-deployment/3",
    PACKAGE_FORMAT,
)
PACKAGE_MANIFEST = "deployment.json"
PACKAGE_CHECKPOINT = "model.pt"
MODEL_CARD = "MODEL_CARD.md"
MAX_SMILES_LENGTH = 2000  # longest SMILES the serving app accepts

# Checkpoint entries inference needs; training state (optimizer, scheduler, history) is dropped.
_INFERENCE_KEYS = (
    "model_state_dict",
    "hyper_parameters",
    "data_info",
    "version_info",
    "calibration",
    "target_scaler",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_checkpoint(
    checkpoint_path: str | Path,
    save_path: str | Path,
    *,
    verify: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Path:
    """Write a deployment package for a MILIA v2 checkpoint, and verify it.

    Args:
        checkpoint_path: A checkpoint written by MILIA's trainer (for example
            ``{working_root_dir}/checkpoints/best.pt``).
        save_path: Directory to create the package in.
        verify: For a model that accepts identifier inputs, reload the written package, predict
            its input example (``"CCO"`` when none is recorded) and compare with the model built
            from ``checkpoint_path`` (``rtol``/``atol``, as :func:`~.onnx_export.export_onnx`). A
            mismatch removes the files this call wrote and raises.

    Returns:
        The package directory.

    Raises:
        DeploymentError: The checkpoint is missing, unreadable, or lacks the model weights or the
            model name and task type needed to rebuild the model; or the written package does not
            reproduce the checkpoint's predictions.
    """
    source = Path(checkpoint_path).expanduser()
    if not source.is_file():
        raise DeploymentError(f"Checkpoint not found: {source}")
    try:
        checkpoint = load_checkpoint_file(source, map_location="cpu")
    except Exception as exc:
        raise DeploymentError(f"Cannot read checkpoint {source}: {exc}") from exc

    hyper_parameters = checkpoint.get("hyper_parameters") if isinstance(checkpoint, dict) else None
    missing = [
        name
        for name, present in (
            ("model_state_dict", isinstance(checkpoint, dict) and "model_state_dict" in checkpoint),
            ("hyper_parameters.model_name", bool((hyper_parameters or {}).get("model_name"))),
            ("hyper_parameters.task_type", bool((hyper_parameters or {}).get("task_type"))),
        )
        if not present
    ]
    if missing:
        raise DeploymentError(
            f"{source} is not a MILIA v2 checkpoint that can rebuild its model "
            f"(missing: {', '.join(missing)}). Package a checkpoint written by MILIA's trainer."
        )

    from milia_pipeline import __version__

    contract = _input_contract(source)
    package = Path(save_path).expanduser()
    package.mkdir(parents=True, exist_ok=True)
    target = package / PACKAGE_CHECKPOINT
    torch.save({key: checkpoint[key] for key in _INFERENCE_KEYS if key in checkpoint}, target)

    run_info = checkpoint.get("run_info") or {}
    version_info = checkpoint.get("version_info") or {}
    manifest = {
        "format": PACKAGE_FORMAT,
        "milia_version": __version__,
        "checkpoint": PACKAGE_CHECKPOINT,
        "sha256": _sha256(target),
        "model_name": hyper_parameters["model_name"],
        "task_type": hyper_parameters["task_type"],
        "checkpoint_format_version": version_info.get("checkpoint_format_version"),
        "signature": _signature(checkpoint, run_info, contract),
        "input_example": list(run_info.get("input_example") or []),
        "structure_example": (
            list(run_info.get("structure_example") or []) if contract.accepts(STRUCTURE) else []
        ),
        "metrics": _metrics(checkpoint),
        "calibration": checkpoint.get("calibration"),
        "target_scaler": checkpoint.get("target_scaler"),
        "input_reference": checkpoint.get("input_reference"),
        "lineage": {
            "source_checkpoint_sha256": _sha256(source),
            "checkpoint_created_at": version_info.get("created_at"),
            "torch_version": version_info.get("pytorch_version"),
            "pyg_version": version_info.get("pyg_version"),
            "run_info_version": run_info.get("run_info_version"),
            "dataset": run_info.get("dataset"),
            "data_split": run_info.get("data_split"),
            "config_sha256": run_info.get("config_sha256"),
            "git_commit": run_info.get("git_commit"),
            "versions": run_info.get("versions"),
            "model_source": run_info.get("model_source"),
        },
    }
    (package / PACKAGE_MANIFEST).write_text(json.dumps(manifest, indent=2, default=_json) + "\n")
    (package / MODEL_CARD).write_text(model_card(manifest))
    # Reproduction is checked on each input kind the model accepts and has an example for; a
    # model accepting neither (QM inputs from its dataset, records missing) by its checksums only.
    if verify:
        try:
            if contract.accepts(IDENTIFIER):
                _check_reproduces(source, package, manifest["input_example"], rtol=rtol, atol=atol)
            if manifest["structure_example"]:
                _check_reproduces_structures(
                    source, package, manifest["structure_example"], rtol=rtol, atol=atol
                )
        except DeploymentError:
            for name in (PACKAGE_CHECKPOINT, PACKAGE_MANIFEST, MODEL_CARD):
                (package / name).unlink(missing_ok=True)
            raise
    return package


_FALLBACK_EXAMPLE = ("CCO",)


def _input_contract(checkpoint: Path) -> InputContract:
    """The input contract of the model a checkpoint rebuilds (``models/input_contract.py``)."""
    from milia_pipeline.models.post_training.inference.predictor import Predictor

    try:
        predictor = Predictor.from_checkpoint(
            checkpoint_path=checkpoint,
            working_root_dir=checkpoint.parent,
            device=torch.device("cpu"),
        )
    except Exception as exc:
        raise DeploymentError(f"Cannot rebuild the model of {checkpoint}: {exc}") from exc
    return predictor.input_contract


def _predictions(checkpoint: Path, smiles: list[str]) -> torch.Tensor:
    from milia_pipeline.models.post_training import convert_to_pyg
    from milia_pipeline.models.post_training.inference.predictor import Predictor

    predictor = Predictor.from_checkpoint(
        checkpoint_path=checkpoint, working_root_dir=checkpoint.parent, device=torch.device("cpu")
    )
    config = predictor.structural_features_config
    graphs = predictor.prepare_inputs(
        [
            convert_to_pyg(
                s,
                format="smiles",
                structural_features_config=config,
                **predictor.featurization_kwargs,
            )
            for s in smiles
        ]
    )
    # As the model returns them: one row per molecule, or per atom for a node-level model
    return torch.as_tensor(predictor.predict_batch(graphs, return_numpy=True))


def _check_reproduces(
    source: Path, package: Path, example: list[str], *, rtol: float, atol: float
) -> None:
    """The package, loaded as a server loads it, predicts as the source checkpoint does."""
    smiles = list(example) or list(_FALLBACK_EXAMPLE)
    try:
        checkpoint, _ = verify_package(package)
        expected = _predictions(source, smiles)
        actual = _predictions(checkpoint, smiles)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except Exception as exc:
        raise DeploymentError(
            f"The package in {package} does not reproduce the predictions of {source} on "
            f"{smiles}: {exc}"
        ) from exc


def _structure_predictions(checkpoint: Path, example: list[dict[str, Any]]) -> torch.Tensor:
    from milia_pipeline.models.post_training.inference.predictor import Predictor
    from milia_pipeline.molecules.structure_inputs import StructureInput

    predictor = Predictor.from_checkpoint(
        checkpoint_path=checkpoint, working_root_dir=checkpoint.parent, device=torch.device("cpu")
    )
    graphs = predictor.prepare_structures([StructureInput.from_dict(s) for s in example])
    # As the model returns them: one row per molecule, or per atom for a node-level model
    return torch.as_tensor(predictor.predict_batch(graphs, return_numpy=True))


def _check_reproduces_structures(
    source: Path, package: Path, example: list[dict[str, Any]], *, rtol: float, atol: float
) -> None:
    """The package, loaded as a server loads it, predicts the structure example as the source."""
    try:
        checkpoint, _ = verify_package(package)
        expected = _structure_predictions(source, example)
        actual = _structure_predictions(checkpoint, example)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except Exception as exc:
        raise DeploymentError(
            f"The package in {package} does not reproduce the predictions of {source} on its "
            f"{len(example)} structure example(s): {exc}"
        ) from exc


def model_folder_name(model_name: str) -> str:
    """A model name as one safe path component."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", model_name).strip(".")
    return name or "model"


FAMILY_FILE = "family.json"


def model_family(
    *,
    dataset_type: str | None,
    task_type: str | None,
    target_names: list[str] | None,
    target_count: int | None,
    source_units: Any = None,
) -> dict[str, Any]:
    """The model family of a model: what it predicts, and a folder name for it.

    Two models are of one family when they predict the same targets of the same dataset type for
    the same task, in the same source units; only then do their metrics measure the same thing.
    The architecture is not part of it (a GCN and a GAT of one family are versions of one
    family). ``id`` is ``<dataset>-<task>-<digest>``: the digest is the first 12 hex digits of
    SHA-256 over the canonical JSON of ``key``.
    """
    key = {
        "dataset_type": dataset_type,
        "task_type": task_type,
        "targets": {
            "names": list(target_names) if target_names else None,
            "count": target_count,
        },
        "source_units": source_units,
    }
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), default=_json)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    slug = model_folder_name(f"{dataset_type or 'unknown'}-{task_type or 'unknown'}")
    return {"id": f"{slug}-{digest}", "key": key}


def checkpoint_family(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The family (:func:`model_family`) of a loaded checkpoint, from what its package records."""
    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    values = hyper_parameters.get("hyperparameters") or {}
    model_info = hyper_parameters.get("model_info") or {}
    run_info = checkpoint.get("run_info") or {}
    dataset = run_info.get("dataset") or {}
    return model_family(
        dataset_type=dataset.get("type"),
        task_type=hyper_parameters.get("task_type"),
        target_names=(run_info.get("targets") or {}).get("names"),
        target_count=model_info.get("out_channels", values.get("out_channels")),
        source_units=dataset.get("source_units"),
    )


def manifest_family(manifest: dict[str, Any]) -> dict[str, Any]:
    """The family (:func:`model_family`) of a package, from its manifest."""
    dataset = (manifest.get("lineage") or {}).get("dataset") or {}
    outputs = (manifest.get("signature") or {}).get("outputs") or {}
    return model_family(
        dataset_type=dataset.get("type"),
        task_type=manifest.get("task_type"),
        target_names=outputs.get("names"),
        target_count=outputs.get("count"),
        source_units=dataset.get("source_units"),
    )


def _family_dir(output: Path, family: dict[str, Any]) -> Path:
    """``<output>/<family id>/``, created with its ``family.json``; an existing folder must record
    the same family (a digest collision, or a folder edited by hand, is refused)."""
    folder = output / family["id"]
    folder.mkdir(exist_ok=True)
    record = folder / FAMILY_FILE
    if record.is_file():
        recorded = json.loads(record.read_text())
        if recorded.get("key") != json.loads(json.dumps(family["key"], default=_json)):
            raise DeploymentError(
                f"{folder} records another model family ({record}); not publishing into it"
            )
    else:
        record.write_text(json.dumps(family, indent=2, default=_json) + "\n")
    return folder


def _next_version(model_dir: Path) -> int:
    versions = [int(p.name) for p in model_dir.iterdir() if p.name.isdigit()]
    return max(versions, default=0) + 1


def publish_package(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    prepare: Callable[[Path], None] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Path:
    """Write a verified package as a new version of its family: ``<output_dir>/<family>/<N>/``.

    The package is written to a temporary sibling, verified (:func:`package_checkpoint`) and
    then renamed into the next free integer version, so a version folder is either complete and
    verified or absent, and an existing version is never overwritten. Versions start at 1.

    Args:
        checkpoint_path: A MILIA checkpoint, or an existing package (verified with
            :func:`verify_package`, then copied: its manifest, card and checkpoint unchanged).
        output_dir: The folder holding ``<family>/<N>/`` (:func:`model_family`).
        prepare: Called with the staged package after verification and before the rename, to
            add files to the version (a serving ``Dockerfile``, an ONNX export); an exception
            from it publishes nothing.

    Raises:
        DeploymentError: As :func:`package_checkpoint` or :func:`verify_package`; nothing is
            left behind.
    """
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=output))
    try:
        source = Path(checkpoint_path).expanduser()
        if is_package(source):
            checkpoint, _ = verify_package(source)
            for name in (checkpoint.name, PACKAGE_MANIFEST, MODEL_CARD):
                if (source / name).is_file():
                    shutil.copy2(source / name, staging / name)
            verify_package(staging)
        else:
            package_checkpoint(source, staging, rtol=rtol, atol=atol)
        if prepare is not None:
            prepare(staging)
        manifest = json.loads((staging / PACKAGE_MANIFEST).read_text())
        model_dir = _family_dir(output, manifest_family(manifest))
        while True:
            target = model_dir / str(_next_version(model_dir))
            if target.exists():
                continue  # another writer took it between the scan and the check
            try:
                staging.rename(target)
            except OSError as exc:
                if target.exists():
                    continue  # taken between the check and the rename
                raise DeploymentError(f"Cannot move the package to {target}: {exc}") from exc
            logger.info(f"Published deployment package {target}")
            return target
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _json(value: Any) -> Any:
    """JSON for values a checkpoint may hold (tensors, numpy scalars, paths)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _signature(
    checkpoint: dict[str, Any], run_info: dict[str, Any], contract: InputContract
) -> dict[str, Any]:
    """What the model takes and returns, from the checkpoint, its ``run_info`` and contract."""
    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    values = hyper_parameters.get("hyperparameters") or {}
    model_info = hyper_parameters.get("model_info") or {}
    data_info = checkpoint.get("data_info") or {}
    targets = run_info.get("targets") or {}
    return {
        "inputs": {
            # Accepted input kinds, and the reason each other kind is refused
            "kinds": contract.to_dict(),
            # Identifier inputs: SMILES strings up to this length (the serving app's limit)
            "max_length": MAX_SMILES_LENGTH,
            "featurization": data_info.get("structural_features_config") or None,
            "node_features": values.get("in_channels"),
            "uses_edge_features": bool(
                data_info.get("uses_edge_features", model_info.get("uses_edge_features"))
            ),
            # The training data's pre-transforms, applied to every featurized input
            # (pre_transform_record.py); None: none, or not recorded (transforms_recorded)
            "transforms": (data_info.get("input_transforms") or {}).get("repr"),
            "transforms_recorded": "input_transforms" in data_info,
            # Hydrogens as nodes in the training graphs, added to featurized molecules
            # (molecules/featurization_record.py); None: not recorded
            "explicit_hydrogens": data_info.get("explicit_hydrogens"),
            # Features from the dataset's QM data, and where each came from
            "qm_input_features": list(data_info.get("qm_input_features") or []),
            "qm_input_sources": data_info.get("qm_input_sources"),
            # How the training dataset built each molecule (what a structure input must match)
            "molecule_creation": data_info.get("molecule_creation"),
        },
        "outputs": {
            "count": model_info.get("out_channels", values.get("out_channels")),
            "names": targets.get("names"),
            "task_type": hyper_parameters.get("task_type"),
        },
    }


def _metrics(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint's own scores: validation at its epoch, and test when it was tested.

    ``metrics_history`` holds one value per epoch up to the checkpoint's epoch, so the last value
    of each ``val_*`` series is the checkpoint's.
    """
    history = checkpoint.get("metrics_history") or {}
    validation = {
        name: float(values[-1])
        for name, values in sorted(history.items())
        if name.startswith("val_") and isinstance(values, list | tuple) and values
    }
    test = checkpoint.get("test_metrics")
    monitored = checkpoint.get("monitored_metric")
    return {
        "epoch": checkpoint.get("epoch"),
        "monitored": (
            {"name": monitored, "value": float(checkpoint["monitored_score"])}
            if monitored and checkpoint.get("monitored_score") is not None
            else None
        ),
        "validation": validation or None,
        "test": {name: float(value) for name, value in test.items()} if test else None,
    }


def manifest_scores(metrics: dict[str, Any] | None) -> dict[str, float]:
    """A manifest's ``metrics`` as one name → value mapping (validation, test, monitored)."""
    metrics = metrics or {}
    scores = dict(metrics.get("validation") or {})
    scores.update(metrics.get("test") or {})
    monitored = metrics.get("monitored") or {}
    if monitored.get("name") and monitored.get("value") is not None:
        scores.setdefault(monitored["name"], monitored["value"])
    return {name: float(value) for name, value in scores.items()}


def checkpoint_scores(checkpoint_path: str | Path) -> dict[str, float]:
    """The scores a package of this checkpoint would record (:func:`manifest_scores`).

    Raises:
        DeploymentError: The checkpoint cannot be read.
    """
    path = Path(checkpoint_path).expanduser()
    try:
        checkpoint = load_checkpoint_file(path, map_location="cpu")
    except Exception as exc:
        raise DeploymentError(f"Cannot read checkpoint {path}: {exc}") from exc
    return manifest_scores(_metrics(checkpoint))


def _structures_line(manifest: dict[str, Any]) -> str:
    examples = manifest.get("structure_example") or []
    accepted = (((manifest.get("signature") or {}).get("inputs") or {}).get("kinds") or {}).get(
        "accepted"
    ) or []
    if examples:
        names = ", ".join(f"`{s.get('name')}` ({len(s['atomic_numbers'])} atoms)" for s in examples)
        return f"{len(examples)} training molecules, verified at packaging: {names}."
    if STRUCTURE in accepted:
        return (
            "not recorded (trained before MILIA recorded them): structures are not served; "
            "retrain to serve them."
        )
    return "none (the model does not accept structure inputs)."


def _show(value: Any) -> str:
    if value is None or value == [] or value == {}:
        return "not recorded"
    if isinstance(value, list | tuple):
        return ", ".join(f"`{item}`" for item in value)
    return f"`{value}`"


def _table(scores: dict[str, float] | None) -> list[str]:
    if not scores:
        return ["Not recorded."]
    return ["| Metric | Value |", "|---|---|"] + [
        f"| `{name}` | {value:.6g} |" for name, value in scores.items()
    ]


def _calibration_lines(calibration: dict[str, Any] | None) -> list[str]:
    if not calibration:
        return [
            "Not recorded: no prediction intervals (the checkpoint has no conformal calibration)."
        ]
    alpha = calibration["alpha"]
    scaled = calibration["method"] == "split_conformal_normalized"
    return [
        f"Split conformal intervals at {1 - alpha:.0%} (alpha {alpha:g}), fitted on "
        f"{calibration['n']} validation molecules"
        + (", scaled by the spread across the ensemble's members." if scaled else ".")
        + " Each target's interval covers the true value with probability at least "
        f"{1 - alpha:.0%} on average over molecules exchangeable with the validation set; "
        "the guarantee is per target, not joint, and not for molecules unlike it.",
    ]


def _drift_line(reference: dict[str, Any] | None) -> str:
    if not reference:
        return "not recorded (the server reports no drift)"
    return (
        f"histograms of {len(reference['features']) - 1} node-feature columns and of atoms per "
        f"molecule over {reference['molecules']} training molecules; the server compares served "
        "molecules with them (Jensen-Shannon distance, `GET /monitoring/drift`)"
    )


def _target_scaling_line(scaler: dict[str, Any] | None) -> str:
    if not scaler:
        return "none recorded (the model's outputs are its predictions)"
    return (
        f"trained on targets standardized with the mean and std of {scaler['n']} training "
        "values per target; MILIA returns predictions in the targets' own units. A model "
        "exported to ONNX outputs the standardized values: multiply by `target_scaler.std` and "
        f"add `target_scaler.mean` from `{PACKAGE_MANIFEST}`."
    )


def _qm_line(features: list[str] | None) -> str:
    if not features:
        return "none (every feature is computed from the molecule's identifier)"
    return f"{_show(features)} come from the training dataset's QM data"


def _kinds_lines(kinds: dict[str, Any] | None) -> list[str]:
    if not kinds:
        return ["- Accepted inputs: not recorded (a package written before the input contract)."]
    lines = [f"- Accepted inputs: {_show(kinds.get('accepted'))}"]
    lines += [
        f"- Refused `{kind}` inputs: {reason}"
        for kind, reason in (kinds.get("refused") or {}).items()
    ]
    if "identifier" not in (kinds.get("accepted") or []):
        lines.append("- The server does not start for this model: it serves identifier inputs.")
    return lines


def _hydrogens_line(explicit: bool | None) -> str:
    if explicit is None:
        return "not recorded (molecules are featurized with implicit hydrogens)"
    if explicit:
        return "yes: the training graphs hold hydrogens as atoms; featurized molecules get them too"
    return "no: the training graphs hold heavy atoms only"


def _transforms_line(inputs: dict[str, Any]) -> str:
    if not inputs.get("transforms_recorded"):
        return (
            "not recorded (the checkpoint predates the record; inputs are used as featurized, "
            "which matches training only if its data had no pre-transform)"
        )
    if not inputs.get("transforms"):
        return "none (the training data had no pre-transform)"
    names = re.findall(r"^\s+(\w+)\(", inputs["transforms"], flags=re.MULTILINE) or [
        inputs["transforms"]
    ]
    return (
        f"{_show(names)}, the training data's pre-transforms, applied to every featurized "
        "molecule before the model (an ONNX export does not include them)"
    )


def _source_line(source: dict[str, Any] | None) -> str:
    """The model card's line on where the model's classes come from (``lineage.model_source``)."""
    if not source:
        return "not recorded (trained before MILIA recorded it)"
    records = source.get("members") or [source]
    kinds = ", ".join(
        f"`{r.get('model')}` ({r.get('kind')}"
        + (f", {r['path']}, SHA-256 `{r.get('sha256')}`" if r.get("kind") == "directory" else "")
        + ")"
        for r in records
    )
    pins = ", ".join(f"`{pin}`" for pin in source.get("requirements") or []) or "none"
    return f"{kinds}; install: {pins}"


def model_card(manifest: dict[str, Any]) -> str:
    """``MODEL_CARD.md`` for a package manifest (Hugging Face / Mitchell et al. sections)."""
    signature = manifest.get("signature") or {}
    inputs = signature.get("inputs") or {}
    outputs = signature.get("outputs") or {}
    metrics = manifest.get("metrics") or {}
    lineage = manifest.get("lineage") or {}
    dataset = lineage.get("dataset") or {}
    units = dataset.get("source_units") or {}
    featurization = inputs.get("featurization") or {}
    split = lineage.get("data_split") or {}
    monitored = metrics.get("monitored") or {}
    versions = lineage.get("versions") or {}
    lines = [
        f"# {manifest.get('model_name')} ({manifest.get('task_type')})",
        "",
        f"MILIA deployment package (`{manifest.get('format')}`), written by MILIA "
        f"{manifest.get('milia_version')}. `{PACKAGE_MANIFEST}` holds the same facts as data.",
        "",
        "## Model details",
        "",
        f"- Architecture: `{manifest.get('model_name')}`; task: `{manifest.get('task_type')}`",
        f"- Model code: {_source_line((manifest.get('lineage') or {}).get('model_source'))}",
        f"- Weights: `{manifest.get('checkpoint')}` (SHA-256 `{manifest.get('sha256')}`)",
        f"- Checkpoint epoch: {_show(metrics.get('epoch'))}; written "
        f"{_show(lineage.get('checkpoint_created_at'))} with PyTorch "
        f"{_show(lineage.get('torch_version'))} and PyG {_show(lineage.get('pyg_version'))}",
        "",
        "## Inputs and outputs",
        "",
        *_kinds_lines(inputs.get("kinds")),
        f"- Identifier input: a SMILES string per molecule, at most {inputs.get('max_length')} "
        f"characters. MILIA featurizes it into {_show(inputs.get('node_features'))} node features "
        "with "
        + (
            f"the training run's settings (atom: {_show(featurization.get('atom'))}; "
            f"bond: {_show(featurization.get('bond'))})."
            if featurization
            else "MILIA's default featurization (the checkpoint records no settings)."
        ),
        f"- Input transforms: {_transforms_line(inputs)}",
        f"- Explicit hydrogens: {_hydrogens_line(inputs.get('explicit_hydrogens'))}",
        f"- QM inputs: {_qm_line(inputs.get('qm_input_features'))}",
        f"- Output: {_show(outputs.get('count'))} values per molecule; names: "
        f"{_show(outputs.get('names'))}.",
        f"- Target scaling: {_target_scaling_line(manifest.get('target_scaler'))}",
        f"- Input drift reference: {_drift_line(manifest.get('input_reference'))}",
        f"- Example input: {_show(manifest.get('input_example'))}",
        f"- Example structures: {_structures_line(manifest)}",
        "",
        "## Training data",
        "",
        f"- Dataset: {_show(dataset.get('type'))}; source data units (dataset schema): energy "
        f"{_show(units.get('energy'))}, coordinates {_show(units.get('coordinates'))}. "
        "A configured transformation may convert the targets; the configuration is identified "
        "below.",
        f"- Split: {_show(split or None)}",
        "",
        "## Evaluation",
        "",
        f"Monitored metric: {_show(monitored.get('name'))} = "
        f"{_show(monitored.get('value'))} at the checkpoint's epoch.",
        "",
        "Validation set, at the checkpoint's epoch:",
        "",
        *_table(metrics.get("validation")),
        "",
        "Test set, scored with these weights:",
        "",
        *_table(metrics.get("test")),
        "",
        "## Uncertainty",
        "",
        *_calibration_lines(manifest.get("calibration")),
        "",
        "## Limitations",
        "",
        "- Predictions are only as reliable as the molecule is similar to the training data; "
        "this package does not assess a molecule's applicability domain.",
        "- An identifier input is a molecular graph: properties that depend on a 3D conformation "
        "are predicted from the 2D graph MILIA builds from it.",
        "",
        "## Reproducibility",
        "",
        f"- Source checkpoint SHA-256: {_show(lineage.get('source_checkpoint_sha256'))}",
        f"- Configuration SHA-256: {_show(lineage.get('config_sha256'))}",
        f"- Git commit: {_show(lineage.get('git_commit'))}",
        f"- Versions: {_show([f'{k} {v}' for k, v in versions.items()] or None)}",
        "",
        "## How to load",
        "",
        "```python",
        "from milia_pipeline.models.deployment.package import load_package",
        "",
        'model, model_info = load_package("<this directory>")',
        "```",
        "",
        "Serve it with `python -m milia_pipeline.models.deployment.serving --package "
        "<this directory>`.",
        "",
    ]
    return "\n".join(lines)


def is_package(path: str | Path) -> bool:
    """Whether ``path`` is a directory holding a deployment manifest."""
    return (Path(path) / PACKAGE_MANIFEST).is_file()


def verify_package(package_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Check a package's manifest and checkpoint digest.

    Returns:
        ``(checkpoint_path, manifest)``.

    Raises:
        DeploymentError: The manifest is missing, unreadable or of another format, names a
            checkpoint outside the package, or ``model.pt`` does not match the recorded SHA-256.
    """
    package = Path(package_dir).expanduser()
    manifest_path = package / PACKAGE_MANIFEST
    if not manifest_path.is_file():
        raise DeploymentError(f"No {PACKAGE_MANIFEST} in {package}: not a MILIA deployment package")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise DeploymentError(f"Cannot read {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") not in SUPPORTED_FORMATS:
        found = manifest.get("format") if isinstance(manifest, dict) else manifest
        raise DeploymentError(
            f"{manifest_path} has format {found!r}; expected one of {SUPPORTED_FORMATS}"
        )

    name = str(manifest.get("checkpoint", PACKAGE_CHECKPOINT))
    if Path(name).name != name:
        raise DeploymentError(f"{manifest_path} names a checkpoint outside the package: {name!r}")
    checkpoint = package / name
    if not checkpoint.is_file() or _sha256(checkpoint) != manifest.get("sha256"):
        raise DeploymentError(
            f"{checkpoint} is missing or does not match the SHA-256 in {manifest_path}"
        )
    return checkpoint, manifest


def load_package(
    package_dir: str | Path, device: torch.device | None = None
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model of a deployment package, in eval mode.

    Returns:
        ``(model, model_info)`` as returned by ``ModelLoader.load_from_checkpoint``.

    Raises:
        DeploymentError: As :func:`verify_package`.
    """
    package = Path(package_dir).expanduser()
    checkpoint, _ = verify_package(package)

    from milia_pipeline.models.post_training.inference.model_loader import ModelLoader

    model, model_info = ModelLoader.load_from_checkpoint(
        checkpoint_path=checkpoint, working_root_dir=package, device=device
    )
    model.eval()
    return model, model_info
