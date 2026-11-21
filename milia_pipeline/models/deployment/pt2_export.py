"""``.pt2`` export of MILIA models with ``torch.export``, PyTorch's replacement for TorchScript.

:func:`export_pt2` captures the model as an ``ExportedProgram`` (:mod:`._graph_capture`, the same
capture as ONNX export: strict mode first, dynamic node and edge counts for graph models) and
writes one of two artifacts:

- ``aot_compile=False`` (default): the ``ExportedProgram``, saved with ``torch.export.save``.
  Load it with ``torch.export.load(path).module()``; it runs in Python with PyTorch, without
  MILIA's model code. MILIA's metadata (versions, input names) is stored in the file as
  ``milia.json`` and read back with ``torch.export.load(path, extra_files={"milia.json": ""})``.
- ``aot_compile=True``: an AOTInductor package, compiled ahead of time to native code with
  ``torch._inductor.aoti_compile_and_package``. Load it with
  ``torch._inductor.aoti_load_package(path)``. Compiling needs a C++ compiler (the MILIA images
  include gcc and g++).

Before returning, the written file is loaded back and run on the example, and its output is
compared with the eager model, so a file that computes different values is never left behind.

Two properties of ``.pt2`` files come from PyTorch:

- ``torch.export.load`` uses pickle: load only files from a trusted source.
- The format is under active development; a file may not load with a newer PyTorch. The metadata
  records the PyTorch version that wrote it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from milia_pipeline.exceptions import ExportError
from milia_pipeline.models.deployment._graph_capture import capture, first_line, prepare

logger = logging.getLogger(__name__)

__all__ = ["METADATA_FILE", "export_pt2"]

METADATA_FILE = "milia.json"


def export_pt2(
    model: nn.Module,
    filepath: str | Path,
    example: Any,
    *,
    dynamic_shapes: Any = None,
    aot_compile: bool = False,
    inductor_configs: dict[str, Any] | None = None,
    verify: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Path:
    """Export ``model`` to a ``.pt2`` file with ``torch.export`` and verify it.

    Args:
        model: Model to export (put in eval mode).
        filepath: Output file; its suffix must be ``.pt2``.
        example: For a graph model, a PyG ``Data``/``Batch``: the inputs become ``x``,
            ``edge_index``, ``edge_attr`` (when present) and ``batch``, with dynamic node and
            edge counts. For any other module, a tensor or a tuple of tensors.
        dynamic_shapes: ``torch.export`` dynamic shapes (graph models: set automatically).
        aot_compile: Write an AOTInductor package instead of the ``ExportedProgram``.
        inductor_configs: Inductor options for ``aot_compile=True``.
        verify: Load the written file, run it on ``example`` and compare it with the eager model
            (``rtol``/``atol``); a mismatch removes the file and raises.

    Returns:
        ``filepath``.

    Raises:
        ExportError: A suffix other than ``.pt2``, an unsuitable example, a model
            ``torch.export`` cannot capture, a failed compilation, or a verification mismatch.
    """
    filepath = Path(filepath)
    if filepath.suffix != ".pt2":
        raise ExportError(f"A .pt2 export needs a path ending in .pt2, got {filepath.name}")
    if inductor_configs is not None and not aot_compile:
        raise ExportError("inductor_configs applies only with aot_compile=True")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    module, args, names, dynamic_shapes = prepare(model, example, dynamic_shapes)
    program = capture(module, args, dynamic_shapes, ".pt2 export")
    try:
        if aot_compile:
            torch._inductor.aoti_compile_and_package(
                program, package_path=str(filepath), inductor_configs=inductor_configs
            )
        else:
            torch.export.save(program, filepath, extra_files={METADATA_FILE: _metadata(names)})
    except Exception as exc:
        filepath.unlink(missing_ok=True)
        step = "AOTInductor compilation" if aot_compile else "Saving the ExportedProgram"
        raise ExportError(f".pt2 export failed: {step}: {first_line(exc)}") from exc

    if verify:
        try:
            with torch.no_grad():
                expected = module(*args)
                if aot_compile:
                    actual = torch._inductor.aoti_load_package(str(filepath))(*args)
                else:
                    actual = torch.export.load(filepath).module()(*args)
            if isinstance(expected, torch.Tensor) and isinstance(actual, list | tuple):
                (actual,) = actual  # an AOTInductor model returns its outputs as a list
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except Exception as exc:
            filepath.unlink(missing_ok=True)
            raise ExportError(
                f"The exported .pt2 model does not reproduce PyTorch's output: {first_line(exc)}"
            ) from exc
    kind = "AOTInductor package" if aot_compile else "ExportedProgram"
    logger.info(f"Exported model to .pt2 ({kind}): {filepath}")
    return filepath


def _metadata(input_names: tuple[str, ...] | None) -> str:
    from milia_pipeline import __version__

    return json.dumps(
        {
            "format": "torch.export.ExportedProgram",
            "milia_version": __version__,
            "torch_version": torch.__version__,
            "input_names": list(input_names) if input_names is not None else None,
        }
    )
