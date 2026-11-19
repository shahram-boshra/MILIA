"""Graph-level output declared by a module, not inferred from the output's shape.

MILIA's graph-level wrapper (``GraphLevelModelWrapper``) and its ensembles (``ParallelEnsemble``
that reads out, ``SequentialStack``, ``HierarchicalComposition``) pool node outputs per graph.
They declare it with a ``produces_graph_output`` property, so a caller that wraps or fuses them
never pools their output again and never has to infer "one row per graph" from row counts.

That inference compares the number of graphs, which depends on the values of ``batch``, with
the number of nodes. It is correct in eager mode, but it is a data-dependent branch, which
``torch.export`` cannot trace (``GuardOnDataDependentSymNode``). PyTorch's guidance for such
errors is to rewrite the branch so it does not depend on data; a declared property is static.
A model built by the factory also declares it through its attached capability tags (PyG models
with a built-in readout: AttentiveFP, NeuralFingerprint, SchNet, DimeNet, ViSNet, ...). Modules
that declare nothing keep the shape rule.
"""

from __future__ import annotations

from typing import Any


def produces_graph_output(module: Any) -> bool:
    """True when ``module`` declares that its output already has one row per graph.

    Declared by a ``produces_graph_output`` property (MILIA's wrapper and ensembles) or by the
    capability tags the factory attached to a built model (primary output at graph level).
    """
    if getattr(module, "produces_graph_output", False) is True:
        return True
    from milia_pipeline.models.capabilities import attached_capabilities

    built = attached_capabilities(module)
    return built is not None and built.primary.level == "graph"
