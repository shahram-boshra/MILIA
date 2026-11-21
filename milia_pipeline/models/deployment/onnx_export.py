"""
ONNX export of MILIA models with PyTorch's ``torch.export``-based exporter (``dynamo=True``).

PyTorch made this exporter the default in 2.9 and recommends it; the TorchScript-based one is
legacy. It needs the ``onnx`` extra (``onnx``, ``onnxscript``; ``onnxruntime`` to verify and run
the result): ``pip install 'milia-py[onnx]'``.

A MILIA graph model takes a batch of molecular graphs, whose node and edge counts change from
batch to batch. :func:`export_onnx` therefore exports it from a PyG ``Data``/``Batch`` example
with named tensor inputs (``x``, ``edge_index``, ``edge_attr`` when present, ``batch``) and
dynamic node and edge dimensions, so the ONNX model accepts any batch. The output has one row
per graph. Before returning, the exported model is run with ONNX Runtime on the example and
compared with PyTorch, so a file that computes different values is never left behind.

MILIA captures the model with ``torch.export`` in strict mode (TorchDynamo) and passes the
``ExportedProgram`` to the ONNX exporter (:mod:`._graph_capture`, shared with :mod:`.pt2_export`).
Strict mode keeps the graph count, which depends on the values of ``batch``, symbolic; non-strict
capture (the exporter's first attempt) raises on it, so it is only the fallback. A model
``torch.export`` cannot capture either way raises ``ExportError`` naming each attempt's cause.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from milia_pipeline.exceptions import ExportError
from milia_pipeline.models.deployment._graph_capture import (
    GraphModelInputs,
    capture,
    first_line,
    graph_example_inputs,
    prepare,
)

logger = logging.getLogger(__name__)

__all__ = ["GraphModelInputs", "export_onnx", "graph_example_inputs"]

ONNX_EXTRA_HINT = (
    "ONNX export needs the `onnx` extra (onnx, onnxscript, onnxruntime): "
    "pip install 'milia-py[onnx]'"
)


def _require(*packages: str) -> None:
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        raise ExportError(f"{ONNX_EXTRA_HINT}; missing: {', '.join(missing)}")


def export_onnx(
    model: nn.Module,
    filepath: str | Path,
    example: Any,
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    dynamic_shapes: Any = None,
    dynamic_axes: dict[str, Any] | None = None,
    opset_version: int | None = None,
    verify: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Path:
    """Export ``model`` to ONNX with the ``torch.export``-based exporter and verify it.

    Args:
        model: Model to export (put in eval mode).
        filepath: Output ``.onnx`` file.
        example: For a graph model, a PyG ``Data``/``Batch``: the inputs become ``x``,
            ``edge_index``, ``edge_attr`` (when present) and ``batch``, with dynamic node and
            edge counts. For any other module, a tensor or a tuple of tensors.
        input_names: ONNX input names (graph models: set from the example).
        output_names: ONNX output names (default ``["output"]``).
        dynamic_shapes: ``torch.export`` dynamic shapes (graph models: set automatically).
        dynamic_axes: Legacy-exporter form of dynamic dimensions; PyTorch converts it. Not with
            ``dynamic_shapes``.
        opset_version: ONNX opset; ``None`` uses the exporter's default.
        verify: Run the exported model with ONNX Runtime on ``example`` and compare it with
            PyTorch (``rtol``/``atol``); a mismatch removes the file and raises.

    Returns:
        ``filepath``.

    Raises:
        ExportError: Missing ``onnx`` extra, an unsuitable example, a model ``torch.export``
            cannot export, or a verification mismatch.
    """
    if dynamic_shapes is not None and dynamic_axes is not None:
        raise ExportError("Pass dynamic_shapes or dynamic_axes, not both")
    _require("onnx", "onnxscript", *(("onnxruntime",) if verify else ()))
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    module, args, names, dynamic_shapes = prepare(model, example, dynamic_shapes)
    if names is not None:
        input_names = names

    # dynamic_axes (legacy form) is converted by torch.onnx.export itself, which needs the module;
    # otherwise MILIA captures the ExportedProgram (strict first) and hands that to the exporter.
    source = (
        module if dynamic_axes is not None else capture(module, args, dynamic_shapes, "ONNX export")
    )
    try:
        program = torch.onnx.export(
            source,
            args,
            str(filepath),
            dynamo=True,
            input_names=list(input_names) if input_names else None,
            output_names=list(output_names or ["output"]),
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
    except Exception as exc:  # torch.export / translation failures carry their own type
        raise ExportError(f"ONNX export failed: {first_line(exc)}") from exc

    if verify:
        try:
            with torch.no_grad():
                expected = module(*args)
            actual = program(*args)[0]
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except Exception as exc:
            filepath.unlink(missing_ok=True)
            raise ExportError(
                f"The exported ONNX model does not reproduce PyTorch's output: {first_line(exc)}"
            ) from exc
        finally:
            program.release()
    logger.info(f"Exported model to ONNX: {filepath}")
    return filepath
