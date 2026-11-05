"""Consumers see every model through ``ModelRegistry``, whatever its origin.

PyG discovery, plugins and ``register_model`` all feed the registry. Search
spaces, model listings and statistics are read from it, so a registered plugin
model is treated like a PyG model. For PyG models the registry gives the same
results as the PyG introspector it is fed by.
"""

from __future__ import annotations

import ast
import logging
import uuid
from pathlib import Path

import pytest
import torch

from milia_pipeline.models.hpo.search_spaces.search_space_builder import SearchSpaceBuilder
from milia_pipeline.models.registry.model_registry import ModelRegistry
from milia_pipeline.models.registry.pyg_introspector import (
    DynamicModelMetadata,
    ModelCategory,
    get_introspector,
    introspect_model_signature,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "milia_pipeline"
REGISTRY_PACKAGE = PACKAGE_ROOT / "models" / "registry"

# Backend lookups that only the registry may make (pure helpers such as
# generate_search_space and the introspect_* functions stay allowed).
BACKEND_LOOKUPS = frozenset(
    {
        "get_introspector",
        "get_model_metadata",
        "get_all_model_names",
        "get_models_by_task",
        "get_models_by_category",
        "get_models_by_tag",
        "search_models",
        "get_category_statistics",
    }
)


class _OriginProbeNet(torch.nn.Module):
    """A non-PyG model with a parameter the generic fallback space lacks."""

    def __init__(self, in_channels: int = 4, num_blocks: int = 4, out_channels: int = 1):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.num_blocks = num_blocks

    def forward(self, x, edge_index, batch=None):
        return self.lin(x)


@pytest.fixture
def registry():
    return ModelRegistry.get_instance()


@pytest.fixture
def registered_model(registry):
    name = f"OriginProbeNet_{uuid.uuid4().hex[:8]}"
    metadata = DynamicModelMetadata(
        name=name,
        category=ModelCategory.GENERAL,
        import_path=f"{__name__}._OriginProbeNet",
        supported_tasks=["graph_regression"],
        parameters=introspect_model_signature(_OriginProbeNet),
    )
    assert registry.register_model(name, _OriginProbeNet, metadata, plugin_name="origin_probe")
    yield name
    registry.unregister_model(name)


def test_pyg_search_spaces_equal_the_introspector(registry):
    introspector = get_introspector()

    for name in introspector.get_all_model_names():
        assert registry.get_search_space(name) == introspector.get_search_space(name), name


def test_listing_starts_with_the_introspector_names_in_order(registry):
    discovered = get_introspector().get_all_model_names()

    assert registry.list_discovered_model_names()[: len(discovered)] == discovered


def test_registered_model_gets_its_own_search_space_and_is_listed(registered_model, caplog):
    caplog.set_level(logging.WARNING)

    space = SearchSpaceBuilder.for_model(registered_model, include_optimizer=False)

    assert "num_blocks" in space["hyperparameters"]
    assert "No predefined space" not in caplog.text
    assert "not found in the model registry" not in caplog.text
    assert registered_model in SearchSpaceBuilder.list_available_models()


def test_category_breakdown_comes_from_the_registry(registry, registered_model):
    from milia_pipeline.models import get_module_info

    breakdown = get_module_info()["category_breakdown"]

    assert breakdown == registry.get_statistics()["by_category"]
    assert registered_model in registry.list_by_category()[ModelCategory.GENERAL.value]


def _backend_lookup_calls(path: Path) -> list[str]:
    calls = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in BACKEND_LOOKUPS:
            calls.append(f"{path.relative_to(PACKAGE_ROOT.parent)}:{node.lineno}: {name}()")
    return calls


def test_only_the_registry_package_calls_the_discovery_backend():
    offenders = [
        call
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if REGISTRY_PACKAGE not in path.parents
        for call in _backend_lookup_calls(path)
    ]

    assert offenders == [], "Resolve models through ModelRegistry:\n" + "\n".join(offenders)
