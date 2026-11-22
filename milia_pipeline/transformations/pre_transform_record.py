"""The dataset pre-transforms a model was trained on, recorded so that inference applies them too.

A PyG dataset applies its ``pre_transform`` once, when it processes the raw data, so every
training graph carries its effect: ``NormalizeFeatures`` scales each row of ``x`` to sum to one.
A molecule featurized at inference (``convert_to_pyg``) has not been through that pipeline, so
without it the model sees inputs scaled differently from the ones it was trained on
(training-serving skew; Google, Rules of ML #32: re-use the training code at serving time).

Training records the pipeline in the checkpoint (``data_info["input_transforms"]``,
:func:`record_pre_transforms`); inference rebuilds it with the code ``miliaDataset`` uses
(:func:`instantiate_transforms`), checks that it rebuilt the same pipeline, and applies it to
every graph it featurizes (:func:`load_input_transform`).

Record::

    {"version": 1,
     "transforms": [{"name": "NormalizeFeatures", "kwargs": {"attrs": ["x"]}}, ...] | None,
     "repr": "Compose([\\n  NormalizeFeatures()\\n])"}

``transforms`` is ``None`` when the dataset's pipeline is not described by a configuration MILIA
can rebuild (a dataset that is not a ``miliaDataset``); inference then refuses raw inputs rather
than predicting on inputs the model was not trained on. ``repr`` is PyG's own description of the
pipeline (the string PyG stores in ``processed/pre_transform.pt``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from milia_pipeline.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = [
    "RECORD_VERSION",
    "instantiate_transforms",
    "load_input_transform",
    "pipeline_repr",
    "record_pre_transforms",
]

RECORD_VERSION = 1


def pipeline_repr(transform: Any) -> str:
    """PyG's description of a pre-transform (``torch_geometric.data.dataset._repr``).

    PyG writes this string to ``processed/pre_transform.pt`` and compares it on load; memory
    addresses (``<... at 0x...>``) are removed so the same pipeline always gives the same string.
    """
    if transform is None:
        return "None"
    return re.sub("(<.*?)\\s.*(>)", r"\1\2", str(transform))


def instantiate_transforms(specs: Sequence[dict[str, Any]], registry: Any = None) -> Any:
    """Build a ``Compose`` from transform specs (``{"name", "kwargs"}``), as ``miliaDataset`` does.

    Each name is looked up in MILIA's transform ``registry`` first and in
    ``torch_geometric.transforms`` otherwise.

    Returns:
        A ``torch_geometric.transforms.Compose``, or ``None`` for an empty list.

    Raises:
        Exception: The error of the last lookup or constructor that failed.
    """
    import torch_geometric.transforms
    from torch_geometric.transforms import Compose

    transforms = []
    for spec in specs:
        name = spec["name"]
        kwargs = dict(spec.get("kwargs") or {})
        try:
            if registry is None:
                raise LookupError("no transform registry")
            transform_class = registry.get_transform_class(name)
            transforms.append(transform_class(**kwargs))
        except Exception as exc:
            logger.debug(f"Registry creation failed for {name}: {exc}")
            transforms.append(getattr(torch_geometric.transforms, name)(**kwargs))
    return Compose(transforms) if transforms else None


def _processed_record(dataset: Any) -> str | None:
    """The pipeline description PyG stored when it processed ``dataset`` (``None``: none)."""
    processed_dir = getattr(dataset, "processed_dir", None)
    if not processed_dir:
        return None
    path = Path(processed_dir) / "pre_transform.pt"
    if not path.is_file():
        return None
    from torch_geometric.io import fs

    value = fs.torch_load(str(path))
    return value if isinstance(value, str) else None


def record_pre_transforms(dataset: Any) -> dict[str, Any] | None:
    """What a model trained on ``dataset`` needs inference to apply to raw inputs.

    Args:
        dataset: The training dataset (a PyG ``Dataset``; a ``miliaDataset`` records its
            configuration as ``pre_transform_config``).

    Returns:
        The record (see the module docstring), or ``None`` when the dataset has no
        pre-transform.

    Raises:
        ConfigurationError: The dataset was processed with another pipeline than the one it is
            configured with: its graphs do not carry the configured transforms, so no record
            would describe them. Re-process the dataset (``force_reload``).
    """
    transform = getattr(dataset, "pre_transform", None)
    description = pipeline_repr(transform)
    processed = _processed_record(dataset)
    if processed is not None and processed != description:
        raise ConfigurationError(
            f"The dataset in {dataset.processed_dir} was processed with the pre-transform "
            f"{processed!r}, but is configured with {description!r}: its graphs do not carry "
            "the configured transforms. Re-process the dataset (force_reload) or restore the "
            "configuration it was processed with."
        )
    if transform is None:
        return None
    specs = getattr(dataset, "pre_transform_config", None)
    transforms = (
        [{"name": s["name"], "kwargs": dict(s.get("kwargs") or {})} for s in specs]
        if specs
        else None
    )
    return {"version": RECORD_VERSION, "transforms": transforms, "repr": description}


def load_input_transform(record: dict[str, Any] | None) -> Any:
    """Rebuild a checkpoint's recorded pre-transform for inference.

    The built-in custom transforms are registered first (a serving process does not run the
    CLI's startup registration). The rebuilt pipeline must describe itself as the recorded one.

    Args:
        record: ``data_info["input_transforms"]`` (``None``: the dataset had no pre-transform).

    Returns:
        The pipeline to call on each featurized graph, or ``None`` when there is none.

    Raises:
        ConfigurationError: The record has no rebuildable configuration, names a transform this
            MILIA does not have, or rebuilds a different pipeline.
    """
    if record is None:
        return None
    if record.get("version") != RECORD_VERSION:
        raise ConfigurationError(
            f"Unsupported input-transform record version {record.get('version')!r} "
            f"(this MILIA reads version {RECORD_VERSION})"
        )
    recorded = record.get("repr")
    specs = record.get("transforms")
    if specs is None:
        raise ConfigurationError(
            f"The model was trained on graphs pre-transformed with {recorded!r}, which its "
            "checkpoint does not record as a configuration; raw inputs cannot be prepared the "
            "same way."
        )
    from milia_pipeline.transformations.graph_transforms import (
        get_graph_transforms,
        register_all_custom_transforms,
    )

    register_all_custom_transforms()
    try:
        transform = instantiate_transforms(specs, registry=get_graph_transforms().registry)
    except Exception as exc:
        raise ConfigurationError(
            f"Cannot rebuild the input transforms {recorded!r} the model was trained with: {exc}"
        ) from exc
    rebuilt = pipeline_repr(transform)
    if rebuilt != recorded:
        raise ConfigurationError(
            f"The input transforms rebuilt from the checkpoint ({rebuilt!r}) differ from the "
            f"recorded ones ({recorded!r})"
        )
    return transform
