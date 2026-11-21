"""``torch.export`` capture of MILIA models, shared by the ONNX and ``.pt2`` exporters.

A MILIA graph model takes a batch of molecular graphs, whose node and edge counts change from
batch to batch. It is captured from a PyG ``Data``/``Batch`` example with positional tensor
inputs (``x``, ``edge_index``, ``edge_attr`` when present, ``batch``) and dynamic node and edge
dimensions, so the exported program accepts any batch and returns one row per graph.

Capture runs ``torch.export`` in strict mode (TorchDynamo) first: it keeps the graph count, which
depends on the values of ``batch``, symbolic. Non-strict tracing raises on it
(``GuardOnDataDependentSymNode`` in PyG's scatter), so it is only the fallback. A model neither
mode can capture raises ``ExportError`` naming each attempt's cause.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.export import Dim

from milia_pipeline.exceptions import ExportError
from milia_pipeline.models._graph_trace import is_graph_model

GRAPH_INPUTS = ("x", "edge_index", "edge_attr", "batch")


class GraphModelInputs(nn.Module):
    """A MILIA graph model with positional tensor inputs, which an exported program needs.

    Exported inputs are tensors: no PyG ``Data`` objects and no ``None``. The model is called as
    MILIA's ``Predictor`` calls it: ``model(x, edge_index, edge_attr=..., batch=...)``.
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
        raise ExportError("Exporting a graph model needs an example with `x` and `edge_index`")
    tensors = {"x": data.x, "edge_index": data.edge_index}
    if getattr(data, "edge_attr", None) is not None:
        tensors["edge_attr"] = data.edge_attr
    batch = getattr(data, "batch", None)
    tensors["batch"] = batch if batch is not None else torch.zeros(data.x.size(0), dtype=torch.long)
    names = tuple(name for name in GRAPH_INPUTS if name in tensors)
    return names, tuple(tensors[name] for name in names)


def graph_dynamic_shapes(input_names: Sequence[str]) -> tuple[tuple[dict[int, Dim], ...]]:
    """``torch.export`` dynamic shapes for graph inputs: dynamic node and edge counts."""
    nodes, edges = Dim("num_nodes"), Dim("num_edges")
    spec = {"x": {0: nodes}, "edge_index": {1: edges}, "edge_attr": {0: edges}, "batch": {0: nodes}}
    return (tuple(spec[name] for name in input_names),)  # one entry: the *tensors argument


def is_graph_data(example: Any) -> bool:
    """Whether ``example`` is a PyG ``Data``/``Batch``."""
    try:
        from torch_geometric.data.data import BaseData
    except ImportError:
        return False
    return isinstance(example, BaseData)


def prepare(
    model: nn.Module, example: Any, dynamic_shapes: Any = None
) -> tuple[nn.Module, tuple, tuple[str, ...] | None, Any]:
    """The module, positional arguments, input names and dynamic shapes to export.

    A graph model gets :class:`GraphModelInputs` around it and, unless ``dynamic_shapes`` is
    given, dynamic node and edge counts. Any other module is exported as it is, from a tensor or
    a tuple of tensors (input names ``None``).

    Raises:
        ExportError: A graph model with an example that is not a PyG ``Data``/``Batch``.
    """
    if is_graph_data(example):
        names, args = graph_example_inputs(example)
        return (
            GraphModelInputs(model, names).eval(),
            args,
            names,
            graph_dynamic_shapes(names) if dynamic_shapes is None else dynamic_shapes,
        )
    if is_graph_model(model):
        raise ExportError(
            "A graph model is exported from a PyG Data or Batch example (its x, edge_index, "
            "edge_attr and batch become the exported inputs); a tensor example cannot describe a "
            "batch of graphs"
        )
    return model, example if isinstance(example, tuple) else (example,), None, dynamic_shapes


def capture(
    module: nn.Module, args: tuple, dynamic_shapes: Any, what: str
) -> torch.export.ExportedProgram:
    """``torch.export`` the module: strict (TorchDynamo) first, then non-strict.

    Each failed attempt is reported in the ``ExportError``, which names ``what`` failed.
    """
    failures = []
    for strict in (True, False):
        try:
            return torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=strict)
        except Exception as exc:
            failures.append(f"strict={strict}: {first_line(exc)}")
    raise ExportError(
        f"{what} failed: torch.export could not capture the model ({'; '.join(failures)})"
    )


def first_line(exc: BaseException) -> str:
    """The cause line of an export error (``torch.export`` puts it after a header)."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    summary = next(
        (line for line in lines if line.startswith("<class ") or "Error" in line.split(":", 1)[0]),
        lines[0] if lines else type(exc).__name__,
    )
    location = next((line for line in lines if line.startswith("Caused by:")), None)
    return f"{summary} ({location})" if location else summary
