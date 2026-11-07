"""Every model mode with every task type: build, run, check the output shape, backpropagate.

Modes: a single PyG model, a single plugin model, a custom architecture, and ensembles
with each strategy (PyG members, and PyG + plugin members). Each cell is created the way ``main.py`` creates it,
including ``num_classes`` for classification. A cell that does not work yet is a strict
``xfail`` naming its finding, so fixing it turns the cell red until the mark is removed.
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch_geometric.data import Batch, Data

from milia_pipeline.models import model_plugin_system as mps
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.registry import model_registry

TASKS = (
    "node_regression",
    "node_classification",
    "graph_regression",
    "graph_classification",
    "link_prediction",
    "edge_regression",
    "edge_classification",
)
STRATEGIES = ("sequential", "parallel", "hierarchical")
NUM_GRAPHS, NUM_NODES, NUM_FEATURES, NUM_CLASSES = 3, 6, 4, 3

# Known gaps, each a strict xfail until its fix lands (findings in the plugin strategy).
CUSTOM_NOT_TASK_ADAPTIVE = (
    "F30: custom architectures get no task head; the custom mode is deprecated (D10, removal 2.0.0)"
)


def _graphs(task_type: str) -> list[Data]:
    """Data shaped as ``prepare_data_for_task`` leaves it for ``task_type``."""
    generator = torch.Generator().manual_seed(3)
    ring = torch.arange(NUM_NODES)
    edge_index = torch.cat(
        [torch.stack([ring, (ring + 1) % NUM_NODES]), torch.stack([(ring + 1) % NUM_NODES, ring])],
        dim=1,
    )
    num_edges = edge_index.size(1)
    graphs = []
    for _ in range(NUM_GRAPHS):
        data = Data(
            x=torch.randn(NUM_NODES, NUM_FEATURES, generator=generator),
            edge_index=edge_index,
            edge_attr=torch.randn(num_edges, 3, generator=generator),
        )
        if task_type == "node_regression":
            data.y = torch.randn(NUM_NODES, 1, generator=generator)
        elif task_type == "node_classification":
            data.y = torch.randint(0, NUM_CLASSES, (NUM_NODES,), generator=generator)
        elif task_type == "graph_regression":
            data.y = torch.randn(1, 1, generator=generator)
        elif task_type == "graph_classification":
            data.y = torch.randint(0, NUM_CLASSES, (1,), generator=generator)
        elif task_type == "edge_regression":
            data.edge_y = torch.randn(num_edges, 1, generator=generator)
        elif task_type == "edge_classification":
            data.edge_y = torch.randint(0, NUM_CLASSES, (num_edges,), generator=generator)
        else:  # link_prediction
            data.edge_label_index = torch.randint(0, NUM_NODES, (2, 4), generator=generator)
            data.edge_label = torch.randint(0, 2, (4,), generator=generator).float()
        graphs.append(data)
    return graphs


def _expected_shape(task_type: str, batch: Batch) -> tuple[int, int]:
    if task_type.startswith("node_"):
        rows = batch.num_nodes
    elif task_type.startswith("graph_"):
        rows = batch.num_graphs
    elif task_type == "link_prediction":
        rows = batch.edge_label_index.size(1)
    else:
        rows = batch.edge_index.size(1)
    return rows, NUM_CLASSES if "classification" in task_type else 1


def _ensemble_config(strategy: str, members: list[str]) -> dict:
    return {
        "name": "MatrixEnsemble",
        "strategy": strategy,
        "fusion": {"method": "mean"},
        "models": [
            {
                "name": name,
                "level": level if strategy == "hierarchical" else 0,
                "hyperparameters": {"hidden_channels": 16},
            }
            for level, name in enumerate(members)
        ],
    }


def _create(mode: str, task_type: str, sample: Data) -> torch.nn.Module:
    """Create the model as ``main.py`` does for ``mode``."""
    num_classes = NUM_CLASSES if "classification" in task_type else None
    factory = ModelFactory()
    if mode in ("single", "single/plugin"):
        name = "GCN" if mode == "single" else "ExampleGNN"
        model, _ = factory.create_model_with_info(
            name, {"hidden_channels": 16}, task_type, sample, num_classes_override=num_classes
        )
        return model
    if mode == "custom":
        layers = [
            {"type": "GCNConv", "params": {"out_channels": 16}},
            {"type": "ReLU", "params": {}},
            {"type": "GCNConv", "params": {"out_channels": 16}},
        ]
        hyperparameters = {"architecture_config": {"name": "MatrixCustom", "layers": layers}}
        if num_classes is not None:
            hyperparameters["out_channels"] = num_classes
        return factory.create_model("custom", hyperparameters, task_type, sample_data=sample)

    kind, strategy = mode.split("/")
    members = ["GCN", "GraphSAGE"] if kind == "ensemble" else ["GCN", "ExampleGNN"]
    hyperparameters = {"ensemble_config": _ensemble_config(strategy, members)}
    if num_classes is not None:
        key = "num_classes" if task_type == "edge_classification" else "out_channels"
        hyperparameters[key] = num_classes
    return factory.create_model("ensemble", hyperparameters, task_type, sample_data=sample)


def _cells():
    for task_type in TASKS:
        yield task_type, "single", None
        yield task_type, "single/plugin", None
        yield task_type, "custom", CUSTOM_NOT_TASK_ADAPTIVE
        for strategy in STRATEGIES:
            yield task_type, f"ensemble/{strategy}", None
        for strategy in STRATEGIES:
            yield task_type, f"mixed/{strategy}", None


def _marks(mode: str, gap: str | None) -> list:
    marks = [pytest.mark.xfail(strict=True, reason=gap)] if gap else []
    if mode == "custom":  # deprecated mode: its FutureWarning is expected here
        marks.append(pytest.mark.filterwarnings("ignore:Custom architectures:FutureWarning"))
    return marks


@pytest.fixture(scope="module")
def plugins():
    """The shipped example plugin, discovered fresh; restored afterwards."""
    saved = (mps.ModelPluginLoader._instance, mps._plugin_loader)
    mps.ModelPluginLoader._instance = mps._plugin_loader = None
    mps._reset_model_plugin_discovery()
    registry = model_registry.registry
    before = set(registry.list_available_models())
    saved_plugin_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith(f"{mps.PLUGIN_MODULE_NAMESPACE}.")
    }
    mps.ensure_model_plugins_discovered()
    yield
    for name in set(registry.list_available_models()) - before:
        registry.unregister_model(name)
    # Restore the plugin modules as they were: models registered before this
    # fixture still import from them.
    for name in [n for n in sys.modules if n.startswith(f"{mps.PLUGIN_MODULE_NAMESPACE}.")]:
        del sys.modules[name]
    sys.modules.update(saved_plugin_modules)
    mps._reset_model_plugin_discovery()
    mps.ModelPluginLoader._instance, mps._plugin_loader = saved


@pytest.mark.parametrize(
    ("task_type", "mode"),
    [pytest.param(task, mode, marks=_marks(mode, gap)) for task, mode, gap in _cells()],
)
def test_mode_and_task(plugins, task_type, mode):
    torch.manual_seed(0)
    graphs = _graphs(task_type)
    batch = Batch.from_data_list(graphs)

    model = _create(mode, task_type, graphs[0])

    kwargs = {"batch": batch.batch}
    if task_type == "link_prediction":
        kwargs["edge_label_index"] = batch.edge_label_index
    if getattr(type(model), "accepts_num_graphs", False) is True:
        kwargs["num_graphs"] = batch.num_graphs
    out = model(batch.x, batch.edge_index, **kwargs)

    rows, columns = _expected_shape(task_type, batch)
    assert out.size(0) == rows
    assert (out.size(-1) if out.dim() > 1 else 1) == columns
    out.float().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
