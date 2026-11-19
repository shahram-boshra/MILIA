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
``ExportedProgram`` to the ONNX exporter. Strict mode keeps the graph count, which depends on
the values of ``batch``, symbolic; non-strict capture (the exporter's first attempt) raises on
it, so it is only the fallback. A model ``torch.export`` cannot capture either way raises
``ExportError`` naming each attempt's cause.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.export import Dim

from milia_pipeline.exceptions import ExportError
from milia_pipeline.models._graph_trace import is_graph_model

logger = logging.getLogger(__name__)

__all__ = ["GraphModelInputs", "export_onnx", "graph_example_inputs"]

ONNX_EXTRA_HINT = (
    "ONNX export needs the `onnx` extra (onnx, onnxscript, onnxruntime): "
    "pip install 'milia-py[onnx]'"
)
GRAPH_INPUTS = ("x", "edge_index", "edge_attr", "batch")


class GraphModelInputs(nn.Module):
    """A MILIA graph model with positional tensor inputs, which ONNX needs.

    ONNX inputs are named tensors: no PyG ``Data`` objects and no ``None``. The model is called
    as MILIA's ``Predictor`` calls it: ``model(x, edge_index, edge_attr=..., batch=...)``.
    """

    def __init__(self, model: nn.Module, input_names: Sequence[str]):
        super().__init__()
        self.model = model
        self.input_names = tuple(input_names)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        named = dict(zip(self.input_names, tensors, strict=True))
        return self.model(named.pop("x"), named.pop("edge_index"), **named)


def graph_example_inputs(data: Any) -> tuple[tuple[str, ...], tuple[torch.Tensor, ...]]:
    """Input names and tensors of a PyG ``Data``/``Batch``, in :data:`GRAPH_INPUTS` order.

    A single ``Data`` gets an all-zero ``batch`` (one graph), so the exported model always takes
    a ``batch`` vector and returns one row per graph.
    """
    if getattr(data, "x", None) is None or getattr(data, "edge_index", None) is None:
        raise ExportError("ONNX export of a graph model needs an example with `x` and `edge_index`")
    tensors = {"x": data.x, "edge_index": data.edge_index}
    if getattr(data, "edge_attr", None) is not None:
        tensors["edge_attr"] = data.edge_attr
    batch = getattr(data, "batch", None)
    tensors["batch"] = batch if batch is not None else torch.zeros(data.x.size(0), dtype=torch.long)
    names = tuple(name for name in GRAPH_INPUTS if name in tensors)
    return names, tuple(tensors[name] for name in names)


def _graph_dynamic_shapes(input_names: Sequence[str]) -> tuple[tuple[dict[int, Dim], ...]]:
    nodes, edges = Dim("num_nodes"), Dim("num_edges")
    spec = {"x": {0: nodes}, "edge_index": {1: edges}, "edge_attr": {0: edges}, "batch": {0: nodes}}
    return (tuple(spec[name] for name in input_names),)  # one entry: the *tensors argument


def _require(*packages: str) -> None:
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        raise ExportError(f"{ONNX_EXTRA_HINT}; missing: {', '.join(missing)}")


def _is_graph_data(example: Any) -> bool:
    try:
        from torch_geometric.data.data import BaseData
    except ImportError:
        return False
    return isinstance(example, BaseData)


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

    if _is_graph_data(example):
        names, args = graph_example_inputs(example)
        module: nn.Module = GraphModelInputs(model, names).eval()
        input_names = names
        if dynamic_shapes is None and dynamic_axes is None:
            dynamic_shapes = _graph_dynamic_shapes(names)
    elif is_graph_model(model):
        raise ExportError(
            "A graph model is exported from a PyG Data or Batch example (its x, edge_index, "
            "edge_attr and batch become the ONNX inputs); a tensor example cannot describe a batch "
            "of graphs"
        )
    else:
        module = model
        args = example if isinstance(example, tuple) else (example,)

    # dynamic_axes (legacy form) is converted by torch.onnx.export itself, which needs the module;
    # otherwise MILIA captures the ExportedProgram (strict first) and hands that to the exporter.
    source = module if dynamic_axes is not None else _capture(module, args, dynamic_shapes)
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
        raise ExportError(f"ONNX export failed: {_first_line(exc)}") from exc

    if verify:
        try:
            with torch.no_grad():
                expected = module(*args)
            actual = program(*args)[0]
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except Exception as exc:
            filepath.unlink(missing_ok=True)
            raise ExportError(
                f"The exported ONNX model does not reproduce PyTorch's output: {_first_line(exc)}"
            ) from exc
        finally:
            program.release()
    logger.info(f"Exported model to ONNX: {filepath}")
    return filepath


def _capture(module: nn.Module, args: tuple, dynamic_shapes: Any) -> torch.export.ExportedProgram:
    """``torch.export`` the module: strict (TorchDynamo) first, then non-strict.

    Strict mode traces Python control flow with TorchDynamo and keeps sizes derived from tensor
    values (the graph count of a batch) symbolic; non-strict tracing raises on them
    (``GuardOnDataDependentSymNode`` in PyG's scatter). Each failed attempt is reported.
    """
    failures = []
    for strict in (True, False):
        try:
            return torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=strict)
        except Exception as exc:
            failures.append(f"strict={strict}: {_first_line(exc)}")
    raise ExportError(
        f"ONNX export failed: torch.export could not capture the model ({'; '.join(failures)})"
    )


def _first_line(exc: BaseException) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    summary = next(  # torch.export puts the cause after a header
        (line for line in lines if line.startswith("<class ") or "Error" in line.split(":", 1)[0]),
        lines[0] if lines else type(exc).__name__,
    )
    location = next((line for line in lines if line.startswith("Caused by:")), None)
    return f"{summary} ({location})" if location else summary
