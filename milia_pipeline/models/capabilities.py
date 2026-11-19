"""Model capability tags: what a model produces and needs, and the tasks that follow from it.

A model's capabilities are typed annotations that MILIA reads instead of guessing from names
(the pattern of scikit-learn's estimator tags: "annotations of estimators that allow programmatic
inspection of their capabilities", which "can depend on estimator parameters"). They are resolved
for a model class and its hyperparameters, before the model is built, in this order:

1. the class's own declaration: a ``milia_capabilities(hyperparameters)`` classmethod (plugin
   models and MILIA's own models declare themselves this way);
2. a provider registered for the class or one of its bases (``register_capabilities``) - how MILIA
   describes third-party classes such as PyG's, whose outputs it cannot infer from a signature;
3. a default derived from the ``forward`` signature: per-node features.

The tasks a model supports are computed from its capabilities (``supported_tasks``), and
``check_model_capabilities`` builds a model and verifies its capabilities against its actual
outputs (the role of scikit-learn's ``check_estimator``): a declaration is checked, not trusted.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

__all__ = [
    "GRAPH_TASKS",
    "NODE_TASKS",
    "EDGE_TASKS",
    "OutputSpec",
    "ModelCapabilities",
    "ModelCapabilityError",
    "register_capabilities",
    "get_model_capabilities",
    "supported_tasks",
    "check_model_capabilities",
    "default_capabilities",
    "requires_positions",
    "select_primary_output",
    "attach_capabilities",
    "attached_capabilities",
]

OutputLevel = Literal["node", "graph"]
OutputKind = Literal["features", "property", "vector", "embedding"]
CapabilityProvider = Callable[[Mapping[str, Any]], "ModelCapabilities"]

NODE_TASKS = ("node_classification", "node_regression")
GRAPH_TASKS = ("graph_classification", "graph_regression")
EDGE_TASKS = ("link_prediction", "edge_regression", "edge_classification")

# (level, kind) of the primary output -> the tasks it can serve.
# - node features: a representation per node; MILIA heads turn it into node predictions, pool it
#   into graph predictions, or decode node pairs into edge predictions.
# - graph property: one prediction per graph (the model pools internally); graph tasks only.
# - node vector: a physical vector per node (e.g. atomic forces); regressed per node, never pooled.
# - node embedding: a transductive embedding model (graph autoencoders, random-walk embeddings).
_TASKS: dict[tuple[str, str], tuple[str, ...]] = {
    ("node", "features"): NODE_TASKS + GRAPH_TASKS + EDGE_TASKS,
    ("graph", "property"): GRAPH_TASKS,
    ("node", "vector"): ("node_regression",),
    ("node", "embedding"): ("link_prediction", "node_embedding"),
}


class ModelCapabilityError(ValueError):
    """A capability declaration is invalid, or a model does not behave as declared."""


@dataclass(frozen=True)
class OutputSpec:
    """One output of a model's forward, in the order the forward returns them.

    Args:
        name: What the output is (e.g. ``"features"``, ``"energy"``, ``"forces"``).
        level: ``"node"`` (one row per node) or ``"graph"`` (one row per graph).
        kind: ``"features"``, ``"property"``, ``"vector"`` or ``"embedding"``.
        width: Size of the last dimension when the model fixes it (forces: 3), else ``None``.
    """

    name: str
    level: OutputLevel
    kind: OutputKind
    width: int | None = None


@dataclass(frozen=True)
class ModelCapabilities:
    """Typed capability tags of a model class for given hyperparameters.

    Args:
        outputs: The forward's outputs in return order; the first is the one MILIA trains on. A
            forward that returns a tuple declares one entry per element.
        requires_positions: The forward takes atomic numbers ``z`` and positions ``pos``.
        uses_graph_structure: The forward reads ``edge_index``.
        accepts_edge_attr: The forward can use edge features.
        differentiates_inputs: The forward differentiates an output with respect to its inputs
            (forces = -dE/dpos), so it cannot run under ``torch.no_grad``.
    """

    outputs: tuple[OutputSpec, ...]
    requires_positions: bool = False
    uses_graph_structure: bool = True
    accepts_edge_attr: bool = False
    differentiates_inputs: bool = False
    _tasks: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ModelCapabilityError("ModelCapabilities needs at least one output")
        primary = self.outputs[0]
        key = (primary.level, primary.kind)
        if key not in _TASKS:
            allowed = ", ".join(f"{level} {kind}" for level, kind in _TASKS)
            raise ModelCapabilityError(
                f"Primary output {primary.name!r} is a {primary.level} {primary.kind}; "
                f"supported primary outputs: {allowed}"
            )
        object.__setattr__(self, "_tasks", _TASKS[key])

    @property
    def primary(self) -> OutputSpec:
        """The output MILIA trains on and predicts."""
        return self.outputs[0]


def supported_tasks(capabilities: ModelCapabilities) -> tuple[str, ...]:
    """The MILIA task types a model with these capabilities can serve."""
    return capabilities._tasks


_PROVIDERS: dict[str, CapabilityProvider] = {}


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def register_capabilities(
    *targets: type | str,
) -> Callable[[CapabilityProvider], CapabilityProvider]:
    """Register a provider of capabilities for classes MILIA does not own.

    A target is a class or its fully qualified name (``module.QualName``); a name avoids importing
    a module only to describe it. The provider receives the hyperparameters the model will be
    built with (empty when unknown) and returns ``ModelCapabilities``. It also applies to
    subclasses of a target that register nothing themselves.
    """

    def decorator(provider: CapabilityProvider) -> CapabilityProvider:
        for target in targets:
            key = target if isinstance(target, str) else _qualified_name(target)
            if key in _PROVIDERS and _PROVIDERS[key] is not provider:
                raise ModelCapabilityError(f"Capabilities already registered for {key}")
            _PROVIDERS[key] = provider
        return provider

    return decorator


def _forward_parameters(model_class: type) -> list[str]:
    try:
        parameters = inspect.signature(model_class.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return []
    return [name for name in parameters if name != "self"]


def requires_positions(model_class: type) -> bool:
    """The forward takes ``z`` and ``pos`` among its first three parameters (3D molecular models)."""
    first = _forward_parameters(model_class)[:3]
    return "z" in first and "pos" in first


def default_capabilities(model_class: type) -> ModelCapabilities:
    """Capabilities read from the forward signature: one feature row per node."""
    params = _forward_parameters(model_class)
    return ModelCapabilities(
        outputs=(OutputSpec("features", "node", "features"),),
        requires_positions=requires_positions(model_class),
        uses_graph_structure="edge_index" in params,
        accepts_edge_attr=getattr(model_class, "supports_edge_attr", "edge_attr" in params) is True,
    )


def get_model_capabilities(
    model_class: type, hyperparameters: Mapping[str, Any] | None = None
) -> ModelCapabilities:
    """Resolve the capabilities of ``model_class`` built with ``hyperparameters``.

    Order: the class's ``milia_capabilities`` classmethod, then a provider registered for the class
    or its nearest base, then ``default_capabilities``.
    """
    params: Mapping[str, Any] = dict(hyperparameters or {})
    declare = getattr(model_class, "milia_capabilities", None)
    if callable(declare):
        return _validated(declare(params), model_class)
    for cls in model_class.__mro__:
        provider = _PROVIDERS.get(_qualified_name(cls))
        if provider is not None:
            return _validated(provider(params), model_class)
    return default_capabilities(model_class)


def _validated(value: Any, model_class: type) -> ModelCapabilities:
    if not isinstance(value, ModelCapabilities):
        raise ModelCapabilityError(
            f"Capabilities of {model_class.__name__} must be ModelCapabilities, "
            f"got {type(value).__name__}"
        )
    return value


_ATTRIBUTE = "_milia_capabilities"


def attach_capabilities(model: torch.nn.Module, capabilities: ModelCapabilities) -> None:
    """Record on a built model the capabilities resolved for the hyperparameters it was built with.

    The class alone cannot say what an instance does: PyG ``PNA`` and ``GAT`` accept edge features
    only when built with ``edge_dim`` (``PNAConv`` creates ``edge_encoder`` and ``GATConv``
    ``lin_edge`` only then). The factory attaches them; callers read them with
    ``attached_capabilities``. Stored as a plain attribute (not a parameter or buffer), so it
    never enters a ``state_dict``.
    """
    if not isinstance(capabilities, ModelCapabilities):
        raise ModelCapabilityError(
            f"attach_capabilities needs ModelCapabilities, got {type(capabilities).__name__}"
        )
    object.__setattr__(model, _ATTRIBUTE, capabilities)


def attached_capabilities(model: Any) -> ModelCapabilities | None:
    """The capabilities attached to ``model`` itself when it was built, else ``None``.

    Read from the instance only (a wrapper that delegates attribute access to the model it holds
    does not report that model's capabilities as its own); a ``torch.compile`` wrapper is looked
    through.
    """
    module = getattr(model, "_orig_mod", model)
    value = vars(module).get(_ATTRIBUTE) if hasattr(module, "__dict__") else None
    return value if isinstance(value, ModelCapabilities) else None


def select_primary_output(model: torch.nn.Module, result: Any) -> Any:
    """The output MILIA trains on, from what ``model``'s forward returned.

    ``model`` is the module that produced ``result`` (a ``torch.compile`` wrapper is looked
    through to the compiled module's class). A tensor is returned as is. A tuple is accepted only when the class declares that many outputs
    (PyG ViSNet: ``(energy, forces or None)``); its first element, the primary output, is returned.
    Any other tuple raises ``ModelCapabilityError`` naming the class: taking an element MILIA cannot
    identify would silently train on the wrong quantity.
    """
    if not isinstance(result, tuple):
        return result
    model_class = type(getattr(model, "_orig_mod", model))
    declared = get_model_capabilities(model_class).outputs
    if len(result) != len(declared):
        raise ModelCapabilityError(
            f"{model_class.__name__}.forward returned a tuple of {len(result)}, but its capabilities "
            f"declare {len(declared)} output(s) ({', '.join(spec.name for spec in declared)}). "
            "Declare its outputs with a milia_capabilities classmethod "
            "(milia_pipeline.models.capabilities)."
        )
    return result[0]


# ---------------------------------------------------------------------------------------------
# Verification (scikit-learn ``check_estimator`` pattern)
# ---------------------------------------------------------------------------------------------
_INPUT_NAMES = ("x", "z", "pos", "edge_index", "edge_attr", "batch")


def _call_by_name(model: torch.nn.Module, batch: Any) -> Any:
    """Call ``model`` with the batch attributes its forward names (bound by parameter name)."""
    params = inspect.signature(model.forward).parameters
    kwargs = {}
    for name in params:
        if name in _INPUT_NAMES and getattr(batch, name, None) is not None:
            kwargs[name] = getattr(batch, name)
    return model(**kwargs)


def check_model_capabilities(
    model_class: type,
    hyperparameters: Mapping[str, Any],
    graphs: Sequence[Any],
) -> ModelCapabilities:
    """Build ``model_class`` and verify its resolved capabilities against its outputs.

    ``graphs`` are at least two PyG ``Data`` objects with different node counts (so one row per
    node can never be mistaken for one row per graph). The model runs once in eval mode. Raises ``ModelCapabilityError`` naming every mismatch; returns the capabilities.
    """
    from torch_geometric.data import Batch

    if len(graphs) < 2 or len({g.num_nodes for g in graphs}) < 2:
        raise ModelCapabilityError("check_model_capabilities needs graphs of different sizes")
    capabilities = get_model_capabilities(model_class, hyperparameters)
    batch = Batch.from_data_list(list(graphs))
    model = model_class(**hyperparameters).eval()
    # Gradients stay enabled: a model may differentiate its output (ViSNet forces = -dE/dpos).
    result = _call_by_name(model, batch)
    results = result if isinstance(result, tuple) else (result,)

    problems = []
    if len(results) != len(capabilities.outputs):
        problems.append(
            f"declares {len(capabilities.outputs)} output(s), forward returned {len(results)}"
        )
    rows = {"node": batch.num_nodes, "graph": batch.num_graphs}
    for spec, value in zip(capabilities.outputs, results, strict=False):
        if value is None:
            continue  # an optional output the hyperparameters switch off (e.g. ViSNet forces)
        if not torch.is_tensor(value):
            problems.append(f"output {spec.name!r} is {type(value).__name__}, not a tensor")
            continue
        if value.size(0) != rows[spec.level]:
            problems.append(
                f"output {spec.name!r} declared {spec.level}-level ({rows[spec.level]} rows), "
                f"got {value.size(0)} rows"
            )
        if spec.width is not None and value.size(-1) != spec.width:
            problems.append(f"output {spec.name!r} width {value.size(-1)}, declared {spec.width}")
    if problems:
        raise ModelCapabilityError(f"{model_class.__name__}: " + "; ".join(problems))
    return capabilities


# Providers for third-party classes register on import of this module.
from milia_pipeline.models import _pyg_capabilities  # noqa: E402, F401
