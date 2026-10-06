"""Module helpers shared by the Trainer and its callbacks (PA-1e)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def unwrap_compiled(module: Any) -> Any:
    """Return the original module of a ``torch.compile`` wrapper, else ``module`` unchanged.

    ``torch.compile`` returns an ``OptimizedModule`` that registers the original as child
    ``_orig_mod`` (``torch/_dynamo/eval_frame.py``), so its ``state_dict()`` keys are prefixed
    ``_orig_mod.``. Both share the same parameters: saving from / loading into the original keeps
    checkpoints loadable by an uncompiled model. The check is structural (a real ``_modules`` dict
    holding an ``nn.Module``), so mocks and plain modules pass through unchanged.
    """
    modules = getattr(module, "_modules", None)
    if isinstance(modules, dict) and isinstance(modules.get("_orig_mod"), nn.Module):
        return modules["_orig_mod"]
    return module
