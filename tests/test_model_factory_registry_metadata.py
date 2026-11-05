"""``ModelFactory`` resolves model metadata through ``ModelRegistry``.

The registry holds metadata for every model origin. For PyG models it stores the
introspector's own metadata object, so reading it from the registry changes nothing
for them.
"""

from __future__ import annotations

import pytest

from milia_pipeline.exceptions import ModelInstantiationError
from milia_pipeline.models.factory.model_factory import ModelFactory
from milia_pipeline.models.registry.model_registry import ModelRegistry
from milia_pipeline.models.registry.pyg_introspector import get_model_metadata


def test_registry_metadata_for_pyg_model_is_the_introspector_object():
    registry = ModelRegistry.get_instance()
    if not registry.has_model("GCN"):  # another test may have reset the registry
        registry.auto_discover_pyg_models()

    introspected = get_model_metadata("GCN")

    assert introspected is not None
    assert registry.get_registration("GCN").metadata is introspected
    assert registry.get_metadata("GCN") is introspected


def test_model_without_metadata_raises_instantiation_error(monkeypatch):
    factory = ModelFactory()
    monkeypatch.setattr(factory.registry, "has_model", lambda name: True)
    monkeypatch.setattr(factory.registry, "get_metadata", lambda name: None)

    with pytest.raises(ModelInstantiationError, match="No metadata found for model 'Ghost'"):
        factory.create_model(name="Ghost", hyperparameters={}, task_type="graph_regression")
