"""The ``batch`` vector belongs to the module that pools by graph, whatever the model it wraps accepts.

``GraphLevelModelWrapper`` turns node outputs into one row per graph with ``batch`` (PyG's
``global_*_pool(x, batch)``; without it PyG pools every row into one graph). The model it wraps
may not take ``batch`` at all (PyG ``PMLP.forward(x, edge_index=None)``), so callers must not
decide from the wrapped model's signature whether ``batch`` is passed.

The wrapper declares ``consumes_batch = True`` on its class (the same explicit opt-in as PA-1d's
``accepts_num_graphs``); the Trainer and the Predictor pass ``batch`` to a model whose class
declares it, and the wrapper forwards ``batch`` only to a model whose ``forward`` accepts it.
"""

from __future__ import annotations

import inspect
from typing import Any

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def consumes_batch(module: Any) -> bool:
    """True when ``module``'s class declares that it consumes ``batch`` (``consumes_batch = True``).

    Read from the class (``is True``), so proxies whose every attribute is truthy (mocks) do not
    qualify. Pass the module without a ``torch.compile`` wrapper.
    """
    return getattr(type(module), "consumes_batch", False) is True


def forward_accepts(module: Any, name: str) -> bool:
    """True when ``module.forward`` accepts keyword ``name`` (a named parameter or ``**kwargs``).

    Resolved once by the caller (at construction), not per call: signature introspection is
    Python-level work that a traced forward (``torch.export``) should not repeat.
    """
    try:
        parameters = inspect.signature(module.forward).parameters.values()
    except (AttributeError, TypeError, ValueError):
        return False
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD or (p.name == name and p.kind in _KEYWORD_KINDS)
        for p in parameters
    )
