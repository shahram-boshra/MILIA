"""Models that return several outputs: MILIA trains on the declared primary output.

PyG ViSNet returns ``(energy, forces or None)``. The graph-level wrapper called ``.size`` on the
tuple, and the ensemble's 3D path returned the tuple to the fusion step, so ViSNet could neither
train nor predict through MILIA. The capability tags declare the outputs; MILIA selects the first
(primary) one, and a tuple from a model that declares no such outputs raises naming the model.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch_geometric.typing as pyg_typing
from torch_geometric.data import Batch

from milia_pipeline.models.builders.model_composer import ParallelEnsemble
from milia_pipeline.models.capabilities import (
    ModelCapabilities,
    ModelCapabilityError,
    OutputSpec,
    select_primary_output,
)
from milia_pipeline.models.factory.model_factory import GraphLevelModelWrapper, ModelFactory
from milia_pipeline.models.post_training import convert_to_pyg
from milia_pipeline.models.post_training.inference.predictor import Predictor


class EnergyAndForces(nn.Module):
    """Declares (energy per graph, forces per atom), like ViSNet with derivative=True."""

    @classmethod
    def milia_capabilities(cls, hyperparameters):
        return ModelCapabilities(
            outputs=(
                OutputSpec("energy", "graph", "property"),
                OutputSpec("forces", "node", "vector", width=3),
            ),
            requires_positions=True,
            uses_graph_structure=False,
        )

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 1)

    def forward(self, z, pos, batch=None):
        from torch_geometric.nn import global_add_pool

        energy = global_add_pool(self.lin(pos), batch)
        return energy, -pos


class UndeclaredTuple(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x, edge_index, batch=None):
        out = self.lin(x)
        return out, out


def _batch() -> Batch:
    gen = torch.Generator().manual_seed(0)
    graphs = [convert_to_pyg(s, format="smiles") for s in ("CCO", "c1ccccc1O")]
    for g in graphs:
        g.z = torch.randint(1, 10, (g.num_nodes,), generator=gen)
        g.pos = torch.randn(g.num_nodes, 3, generator=gen)
        g.x = g.x[:, :4]
    return Batch.from_data_list(graphs)


# --- select_primary_output -----------------------------------------------------------------------
def test_tensor_passes_through():
    t = torch.ones(2, 1)
    assert select_primary_output(UndeclaredTuple(), t) is t


def test_declared_tuple_gives_the_primary_output():
    energy, forces = torch.ones(2, 1), torch.zeros(5, 3)
    assert select_primary_output(EnergyAndForces(), (energy, forces)) is energy
    assert select_primary_output(EnergyAndForces(), (energy, None)) is energy


def test_undeclared_tuple_raises_naming_the_model():
    with pytest.raises(ModelCapabilityError, match="UndeclaredTuple.forward returned a tuple of 2"):
        select_primary_output(UndeclaredTuple(), (torch.ones(1), torch.ones(1)))


def test_compiled_module_is_looked_through():
    class Compiled(nn.Module):  # stands in for torch.compile's OptimizedModule
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    energy = torch.ones(2, 1)
    assert select_primary_output(Compiled(EnergyAndForces()), (energy, None)) is energy


# --- callers -----------------------------------------------------------------------------------
def test_graph_wrapper_uses_the_primary_output():
    b = _batch()
    wrapper = GraphLevelModelWrapper(EnergyAndForces(), "graph_regression")
    out = wrapper(b.z, b.pos, batch=b.batch, num_graphs=b.num_graphs)
    assert out.shape == (b.num_graphs, 1)


def test_graph_wrapper_rejects_an_undeclared_tuple():
    b = _batch()
    wrapper = GraphLevelModelWrapper(UndeclaredTuple(), "graph_regression")
    with pytest.raises(ModelCapabilityError, match="UndeclaredTuple"):
        wrapper(b.x, b.edge_index, batch=b.batch)


def test_ensemble_3d_member_uses_the_primary_output():
    b = _batch()
    ensemble = ParallelEnsemble(
        models=[EnergyAndForces()], weights=[1.0], fusion="mean", task_type="graph_regression"
    )
    out = ensemble._call_model_with_signature(
        ensemble.models[0], b.x, b.edge_index, None, b.batch, None, z=b.z, pos=b.pos
    )
    assert torch.is_tensor(out) and out.shape == (b.num_graphs, 1)


def _molecules_3d():
    gen = torch.Generator().manual_seed(0)
    graphs = [convert_to_pyg(s, format="smiles") for s in ("CCO", "c1ccccc1O", "CC(=O)O")]
    for g in graphs:
        g.z = torch.randint(1, 10, (g.num_nodes,), generator=gen)
        g.pos = torch.randn(g.num_nodes, 3, generator=gen) * 1.5
    return graphs


@pytest.mark.skipif(not pyg_typing.WITH_RADIUS, reason="ViSNet's radius_graph needs pyg-lib")
def test_visnet_trains_and_predicts_through_milia(tmp_path):
    graphs = _molecules_3d()
    torch.manual_seed(0)
    model = ModelFactory().create_model(
        name="ViSNet",
        hyperparameters={"hidden_channels": 16, "num_layers": 2},
        task_type="graph_regression",
        sample_data=graphs[0],
    )
    batch = Batch.from_data_list(graphs)
    out = model(batch.z, batch.pos, batch=batch.batch, num_graphs=batch.num_graphs)
    assert out.shape[0] == len(graphs)
    out.sum().backward()  # trainable: gradients reach the parameters
    assert any(p.grad is not None for p in model.parameters())

    model.eval()
    predictor = Predictor(model, working_root_dir=tmp_path, task_type="graph_regression")
    with torch.no_grad():  # as Predictor.predict and the Trainer's evaluation run
        predictions = predictor._forward(batch)
    assert predictions.shape[0] == len(graphs)


def test_visnet_with_derivative_is_rejected_at_build():
    # derivative=True computes forces with torch.autograd.grad inside forward, which fails under
    # the torch.no_grad of MILIA's evaluation and prediction; MILIA trains on the energy only.
    from milia_pipeline.models.factory.model_factory import ModelValidationError

    with pytest.raises(ModelValidationError, match="ViSNet: derivative=False"):
        ModelFactory().create_model(
            name="ViSNet",
            hyperparameters={"hidden_channels": 16, "num_layers": 2, "derivative": True},
            task_type="graph_regression",
            sample_data=_molecules_3d()[0],
        )


def test_visnet_capabilities_follow_derivative():
    from torch_geometric.nn.models import ViSNet

    from milia_pipeline.models.capabilities import get_model_capabilities

    assert not get_model_capabilities(ViSNet, {}).differentiates_inputs
    assert get_model_capabilities(ViSNet, {"derivative": True}).differentiates_inputs
