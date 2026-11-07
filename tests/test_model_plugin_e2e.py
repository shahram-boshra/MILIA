"""Model plugins end to end: the shipped example plugin, the user template, mixed ensembles."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

import milia_pipeline
from milia_pipeline.models import model_plugin_system as mps
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.registry import model_registry
from milia_pipeline.plugins import get_model_plugins_directory

REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TEMPLATE_DIR = get_model_plugins_directory() / "user_template"


@pytest.fixture
def fresh(tmp_path):
    """Fresh automatic discovery and loader; registered models and plugin modules are removed."""
    saved_instance = mps.ModelPluginLoader._instance
    saved_module_loader = mps._plugin_loader
    mps.ModelPluginLoader._instance = None
    mps._plugin_loader = None
    mps._reset_model_plugin_discovery()
    registry = model_registry.registry
    before = set(registry.list_available_models())
    try:
        yield
    finally:
        for name in set(registry.list_available_models()) - before:
            registry.unregister_model(name)
        for name in list(sys.modules):
            if name.startswith(f"{mps.PLUGIN_MODULE_NAMESPACE}."):
                del sys.modules[name]
        mps._reset_model_plugin_discovery()
        mps.ModelPluginLoader._instance = saved_instance
        mps._plugin_loader = saved_module_loader


@pytest.fixture
def example_models(fresh):
    """The shipped configuration discovers the shipped example plugin."""
    assert "example_models" in mps.ensure_model_plugins_discovered(REPO_CONFIGS)


def _graphs(num_graphs: int = 3, num_nodes: int = 5, features: int = 4) -> list[Data]:
    generator = torch.Generator().manual_seed(7)
    ring = torch.arange(num_nodes)
    edge_index = torch.stack([ring, (ring + 1) % num_nodes])
    return [
        Data(
            x=torch.randn(num_nodes, features, generator=generator),
            edge_index=edge_index,
            y=torch.randn(1, 1, generator=generator),
        )
        for _ in range(num_graphs)
    ]


def _one_step_lowers_the_loss(model: torch.nn.Module, batch: Batch, target: torch.Tensor):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    def loss() -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            model(batch.x, batch.edge_index, batch=batch.batch), target
        )

    before = loss()
    optimizer.zero_grad()
    before.backward()
    optimizer.step()
    with torch.no_grad():
        assert loss().item() < before.item()


def _members(model: torch.nn.Module, class_name: str) -> list[torch.nn.Module]:
    return [module for module in model.modules() if type(module).__name__ == class_name]


def test_example_plugin_builds_runs_and_trains(example_models):
    torch.manual_seed(0)
    graphs = _graphs()
    batch = Batch.from_data_list(graphs)

    model = ModelFactory().create_model(
        "ExampleGNN", {"hidden_channels": 16}, "graph_regression", sample_data=graphs[0]
    )

    (regressor,) = _members(model, "ExampleGNN")
    assert (regressor.in_channels, regressor.hidden_channels) == (4, 16)
    out = model(batch.x, batch.edge_index, batch=batch.batch)
    assert out.shape == (3, 1)
    _one_step_lowers_the_loss(model, batch, torch.cat([g.y for g in graphs]))


def test_example_plugin_is_deterministic_under_a_seed(example_models):
    graphs = _graphs()
    outputs = []
    for _ in range(2):
        torch.manual_seed(123)
        model = ModelFactory().create_model(
            "ExampleGNN", {}, "graph_regression", sample_data=graphs[0]
        )
        outputs.append(model(graphs[0].x, graphs[0].edge_index))
    assert torch.equal(outputs[0], outputs[1])


def test_copied_template_registers_its_model(fresh, tmp_path, monkeypatch):
    """The user template, copied and renamed, is a valid plugin as shipped."""
    # The template targets the release that introduces manifest schema version 2.
    monkeypatch.setattr(milia_pipeline, "__version__", "1.19.0")
    plugin_dir = tmp_path / "plugins" / "my_model"
    shutil.copytree(TEMPLATE_DIR, plugin_dir)
    for template in plugin_dir.glob("*.template"):
        template.rename(template.with_suffix(""))

    loader = mps.get_plugin_loader()
    assert loader.discover_plugins([tmp_path / "plugins"]) == ["your_model_plugin_name"]
    assert loader.load_plugin("your_model_plugin_name") is True
    declaration = loader._plugins["your_model_plugin_name"].model_declarations[0]
    assert (declaration.composable, declaration.chainable) == (True, True)
    assert model_registry.registry.has_model("YourModelName")


@pytest.mark.parametrize(
    ("strategy", "task_type", "plugin_level"),
    [
        ("hierarchical", "node_regression", 1),
        ("hierarchical", "graph_regression", 1),
        ("sequential", "graph_regression", 0),
        ("parallel", "graph_regression", 0),
    ],
)
def test_mixed_pyg_and_plugin_ensemble(example_models, strategy, task_type, plugin_level):
    """A PyG model and the plugin model in one ensemble: built, chained, run and trained."""
    torch.manual_seed(0)
    graphs = _graphs()
    if task_type == "node_regression":
        for graph in graphs:
            graph.y = torch.randn(graph.num_nodes, 1)
    batch = Batch.from_data_list(graphs)
    ensemble_config = {
        "name": "MixedEnsemble",
        "strategy": strategy,
        "fusion": {"method": "mean"},
        "models": [
            {
                "name": "GCN",
                "level": 0,
                "hyperparameters": {"hidden_channels": 16, "out_channels": 8, "num_layers": 2},
            },
            {
                "name": "ExampleGNN",
                "level": plugin_level,
                "hyperparameters": {"hidden_channels": 16, "out_channels": 8},
            },
        ],
    }

    model = ModelFactory().create_model(
        "ensemble", {"ensemble_config": ensemble_config}, task_type, sample_data=graphs[0]
    )

    (gcn,) = _members(model, "GCN")
    (regressor,) = _members(model, "ExampleGNN")
    # Hierarchical and sequential: the plugin member's input is the GCN output (chained).
    assert regressor.in_channels == (gcn.out_channels if strategy != "parallel" else 4)
    out = model(batch.x, batch.edge_index, batch=batch.batch)
    expected_rows = batch.num_nodes if task_type == "node_regression" else batch.num_graphs
    assert out.shape[0] == expected_rows
    _one_step_lowers_the_loss(model, batch, torch.randn(out.shape))


@pytest.mark.parametrize("task_type", ["link_prediction", "edge_regression", "edge_classification"])
def test_plugin_model_outputs_embeddings_for_edge_tasks(example_models, task_type):
    """For the edge decoder the plugin model outputs hidden_channels-wide node embeddings.

    Its own out_channels default (1) must not become the embedding width; the class count
    for edge classification goes to the decoder.
    """
    graphs = _graphs()
    num_classes = 3 if task_type == "edge_classification" else None
    model, _ = ModelFactory().create_model_with_info(
        "ExampleGNN",
        {"hidden_channels": 16},
        task_type,
        graphs[0],
        num_classes_override=num_classes,
    )

    (gnn,) = _members(model, "ExampleGNN")
    assert gnn.out_channels == 16
    if num_classes is not None:
        out = model(graphs[0].x, graphs[0].edge_index)
        assert out.shape == (graphs[0].edge_index.size(1), num_classes)


def test_a_name_is_not_a_3d_signature(example_models):
    """``ExampleGNN`` contains "egnn" but takes (x, edge_index): it is not a 3D molecular model."""
    graphs = _graphs()
    model = ModelFactory().create_model("ExampleGNN", {}, "graph_regression", graphs[0])
    assert model(graphs[0].x, graphs[0].edge_index).shape == (1, 1)
