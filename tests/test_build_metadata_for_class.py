"""``build_metadata_for_class`` builds model metadata from an imported class.

``generate_model_metadata`` imports a PyG class and delegates to it, so both give
the same metadata for every discovered PyG model. It also works for classes that
are not part of PyG, such as plugin models.
"""

from __future__ import annotations

import importlib

import torch

from milia_pipeline.models.registry import build_metadata_for_class
from milia_pipeline.models.registry.pyg_introspector import (
    generate_model_metadata,
    get_introspector,
    get_model_metadata,
)


def _import_class(import_path: str) -> type:
    module_path, class_name = import_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def test_matches_generate_model_metadata_for_every_pyg_model():
    compared = 0
    for name in get_introspector().get_all_model_names():
        cached = get_model_metadata(name)
        if cached is None:  # the class could not be imported in this environment
            continue
        path = cached.import_path

        built = build_metadata_for_class(name, _import_class(path), path)

        assert built.model_dump() == generate_model_metadata(name, path).model_dump(), name
        compared += 1

    assert compared > 0


class _PluginStyleNet(torch.nn.Module):
    """Plugin-style model outside PyG."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        return self.lin(x)


def test_builds_metadata_for_a_class_outside_pyg():
    import_path = f"{__name__}._PluginStyleNet"

    metadata = build_metadata_for_class("PluginStyleNet", _PluginStyleNet, import_path)

    assert metadata.name == "PluginStyleNet"
    assert metadata.import_path == import_path
    assert metadata.description == "Plugin-style model outside PyG."
    assert set(metadata.parameters) == {"in_channels", "hidden_channels", "out_channels"}
    assert metadata.parameters["hidden_channels"].default == 32
    assert set(metadata.hyperparameters) == {"in_channels", "hidden_channels", "out_channels"}
