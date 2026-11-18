"""Graph models are not traced with ``torch.jit.trace``.

``torch.jit.trace`` records only the operations run on the example inputs: control flow is not
recorded, and sizes read from the example become constants, so "you may silently get incorrect
results on subsequent invocations of the model" (PyTorch ``torch.jit.trace`` documentation). A
MILIA graph model reads the batch's node, edge and graph counts and branches on them, so a trace
returns wrong shapes or fails for any other batch. MILIA refuses to trace such a model.
"""

from __future__ import annotations

GRAPH_TRACE_ERROR = (
    "torch.jit.trace cannot convert a graph model: the trace keeps the example batch's node, edge "
    "and graph counts and the branches taken for it, so other batches give wrong results or fail "
    "(PyTorch torch.jit.trace documentation). Use torch.compile (compile_model=True) instead."
)

_GRAPH_MODULE_PREFIXES = ("torch_geometric.", "milia_pipeline.")


def is_graph_model(model) -> bool:
    """Return True if ``model`` (an ``nn.Module``) contains a PyTorch Geometric or MILIA module.

    Both take graph batches: PyG layers and pooling, and MILIA's graph- and edge-level wrappers,
    ensembles and built architectures. A registry model that is a plain ``torch.nn`` layer (for
    example ``Linear``) becomes one through the wrapper that pools it per graph.
    """
    return any(
        type(module).__module__.startswith(_GRAPH_MODULE_PREFIXES) for module in model.modules()
    )
