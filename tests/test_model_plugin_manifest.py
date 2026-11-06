"""Model plugin manifest: schema version 2 and the legacy adapter."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import pytest
import yaml
from packaging.specifiers import SpecifierSet

from milia_pipeline.exceptions import PluginError, PluginValidationError
from milia_pipeline.models._model_plugin_manifest import (
    KNOWN_TASKS,
    PluginManifestV2,
    parse_manifest,
)
from milia_pipeline.models.registry.pyg_introspector import get_introspector, get_model_metadata

SOURCE = "/plugins/my_models/plugin.yaml"
TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "milia_pipeline/plugins/models/user_template/plugin.yaml.template"
)

V2_MANIFEST = {
    "schema_version": 2,
    "plugin_name": "my_models",
    "version": "1.2.0",
    "author": "Ada",
    "description": "Example models",
    "license": "MIT",
    "plugin_type": "user",
    "homepage": "https://example.org/my-models",
    "requirements": {"milia": ">=1.19.0", "python": ">=3.10", "torch_geometric": ">=2.6"},
    "dependencies": ["torch>=2.4", "scikit-learn"],
    "requires_extra": "descriptors-cdk",
    "models": [
        {
            "name": "MyNet",
            "class": "nets.my_net:MyNet",
            "category": "basic_gnn",
            "supported_tasks": ["graph_regression", "node_classification"],
            "description": "A network",
            "requires_edge_index": True,
            "requires_edge_features": True,
            "requires_edge_weights": False,
            "supports_heterogeneous": False,
            "hyperparameters": {"hidden_channels": {"type": "integer", "default": 32}},
            "reference": {"paper": "Doe 2026", "url": "https://example.org/paper"},
        }
    ],
}


def _legacy_manifest(**top_level) -> dict:
    manifest = {
        "plugin_name": "my_models",
        "version": "1.0.0",
        "author": "Ada",
        "description": "Example models",
        "license": "MIT",
        "models": [
            {
                "name": "MyNet",
                "class_name": "MyNet",
                "module_path": "model",
                "category": "basic_gnn",
                "supported_tasks": ["graph_regression"],
            }
        ],
    }
    manifest.update(top_level)
    return manifest


def _parse_quietly(data: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return parse_manifest(data, SOURCE)


def _errors(data: dict) -> list[str]:
    with pytest.raises(PluginValidationError) as caught:
        _parse_quietly(data)
    return caught.value.validation_errors


# --- schema version 2 --------------------------------------------------------


def test_v2_manifest_round_trips_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        manifest = parse_manifest(copy.deepcopy(V2_MANIFEST), SOURCE)

    assert isinstance(manifest, PluginManifestV2)
    assert manifest.model_dump(by_alias=True) == V2_MANIFEST
    entry = manifest.models[0]
    assert (entry.module, entry.class_name) == ("nets.my_net", "MyNet")


def test_v2_defaults_infer_tasks_and_use_the_general_category():
    data = {
        "schema_version": 2,
        "plugin_name": "my_models",
        "version": "1.0.0",
        "models": [{"name": "MyNet", "class": "model:MyNet"}],
    }

    entry = parse_manifest(data, SOURCE).models[0]

    assert entry.supported_tasks is None
    assert entry.category == "general"


@pytest.mark.parametrize(
    "path",
    [(), ("models", 0), ("requirements",), ("models", 0, "reference")],
    ids=["top-level", "model", "requirements", "reference"],
)
def test_v2_rejects_unknown_keys(path):
    data = copy.deepcopy(V2_MANIFEST)
    target = data
    for key in path:
        target = target[key]
    target["surprise"] = 1

    errors = _errors(data)

    assert errors == [
        ".".join(str(p) for p in (*path, "surprise")) + ": Extra inputs are not permitted"
    ]


@pytest.mark.parametrize(
    ("path", "value", "location"),
    [
        (("plugin_name",), "My-Models", "plugin_name"),
        (("version",), "1.0", "version"),
        (("models", 0, "name"), "my-net", "models.0.name"),
        (("models", 0, "class"), "my-model:MyNet", "models.0.class"),
        (("models", 0, "class"), "model", "models.0.class"),
        (("models", 0, "category"), "custom", "models.0.category"),
        (("models", 0, "supported_tasks"), ["regression"], "models.0.supported_tasks"),
        (("models", 0, "supported_tasks"), [], "models.0.supported_tasks"),
        (("requirements", "milia"), "=>1.0", "requirements.milia"),
        (("dependencies",), ["bad spec !!"], "dependencies"),
        (("requires_extra",), "Descriptors_CDK", "requires_extra"),
        (("plugin_type",), "user_experimental", "plugin_type"),
        (("models", 0, "hyperparameters"), {"hidden_channels": 32}, "models.0.hyperparameters"),
        (
            ("models", 0, "hyperparameters"),
            {"hidden_channels": {"default": 32, "step": 2}},
            "models.0.hyperparameters",
        ),
        (
            ("models", 0, "hyperparameters"),
            {"hidden_channels": {"type": "int"}},
            "models.0.hyperparameters",
        ),
    ],
)
def test_v2_rejects_invalid_values_with_the_field_location(path, value, location):
    data = copy.deepcopy(V2_MANIFEST)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    errors = _errors(data)

    assert len(errors) == 1 and errors[0].startswith(f"{location}: "), errors


def test_v2_rejects_duplicate_model_names():
    data = copy.deepcopy(V2_MANIFEST)
    data["models"].append(copy.deepcopy(data["models"][0]))

    assert _errors(data) == ["<manifest>: duplicate model names: ['MyNet']"]


def test_unsupported_schema_version_is_rejected():
    assert _errors({**V2_MANIFEST, "schema_version": 3}) == [
        "schema_version: unsupported value 3; expected 2"
    ]


def test_validation_error_is_a_plugin_error_naming_the_manifest():
    data = {**copy.deepcopy(V2_MANIFEST), "plugin_name": "My-Models"}

    with pytest.raises(PluginError) as caught:
        parse_manifest(data, SOURCE)

    assert isinstance(caught.value, PluginValidationError)
    assert SOURCE in str(caught.value) and "plugin_name: must match" in str(caught.value)


# --- legacy adapter ----------------------------------------------------------


def _model(**fields) -> dict:
    model = _legacy_manifest()["models"][0]
    model.update(fields)
    return model


@pytest.mark.parametrize(
    ("legacy", "check"),
    [
        (
            _legacy_manifest(),
            lambda m: m.models[0].class_ref == "model:MyNet",
        ),
        (
            _legacy_manifest(models=[_model(module_path="model.py")]),
            lambda m: m.models[0].class_ref == "model:MyNet",
        ),
        (_legacy_manifest(milia_version=">=1.2"), lambda m: m.requirements.milia == ">=1.2"),
        (_legacy_manifest(python_version=">=3.10"), lambda m: m.requirements.python == ">=3.10"),
        (
            _legacy_manifest(pyg_version=">=2.5"),
            lambda m: m.requirements.torch_geometric == ">=2.5",
        ),
        (
            _legacy_manifest(min_pyg_version="2.6.0"),
            lambda m: m.requirements.torch_geometric == ">=2.6.0",
        ),
        (
            _legacy_manifest(pyg_version=">=2.5", models=[_model(min_pyg_version="2.6.0")]),
            lambda m: SpecifierSet(m.requirements.torch_geometric) == SpecifierSet(">=2.5,>=2.6.0"),
        ),
        (
            _legacy_manifest(requirements={"milia_version": ">=1.0.0"}),
            lambda m: m.requirements.milia == ">=1.0.0",
        ),
        (
            _legacy_manifest(
                models=[_model(reference_paper="Doe 2026", reference_url="https://x.org")]
            ),
            lambda m: (
                (m.models[0].reference.paper, m.models[0].reference.url)
                == ("Doe 2026", "https://x.org")
            ),
        ),
        (_legacy_manifest(email="ada@example.org"), lambda m: "email" not in m.model_dump()),
        (
            _legacy_manifest(models=[_model(version="1.0.0")]),
            lambda m: "version" not in m.models[0].model_dump(),
        ),
        (
            _legacy_manifest(plugin_type="user_experimental"),
            lambda m: m.plugin_type == "user",
        ),
    ],
    ids=[
        "class_name+module_path",
        "module_path .py suffix",
        "milia_version",
        "python_version",
        "pyg_version",
        "min_pyg_version",
        "per-model min_pyg_version",
        "nested requirements",
        "reference_paper/url",
        "email ignored",
        "per-model version ignored",
        "legacy plugin_type",
    ],
)
def test_legacy_key_is_mapped_and_warns_against_the_manifest_file(legacy, check):
    with pytest.warns(DeprecationWarning, match="removed in MILIA 2.0.0") as record:
        manifest = parse_manifest(copy.deepcopy(legacy), SOURCE)

    assert check(manifest)
    assert record and all(w.filename == SOURCE for w in record)


def test_legacy_hyperparameter_forms_load_with_a_warning():
    """A legacy bare value becomes ``{default: value}``; an unknown type is kept, with a warning."""
    legacy = _legacy_manifest(
        models=[_model(hyperparameters={"hidden": 64, "depth": {"type": "int"}})]
    )

    with pytest.warns(DeprecationWarning) as record:
        manifest = parse_manifest(legacy, SOURCE)

    assert manifest.models[0].hyperparameters == {
        "hidden": {"default": 64},
        "depth": {"type": "int"},
    }
    messages = " ".join(str(w.message) for w in record)
    assert "bare value is read as {default: 64}" in messages and "unknown type 'int'" in messages


def test_legacy_manifest_without_class_keys_is_rejected():
    legacy = _legacy_manifest(models=[{"name": "MyNet", "category": "basic_gnn"}])

    assert _errors(legacy) == ["models.0.class: Field required"]


def test_legacy_plugin_name_that_is_an_identifier_warns_instead_of_failing():
    with pytest.warns(DeprecationWarning, match="does not match"):
        manifest = parse_manifest(_legacy_manifest(plugin_name="MyModels"), SOURCE)

    assert manifest.plugin_name == "MyModels"


def test_user_template_parses():
    with TEMPLATE.open() as f:
        data = yaml.safe_load(f)

    manifest = _parse_quietly(data)

    assert manifest.models[0].class_ref == "model:YourModelClass"
    assert manifest.dependencies == []  # every item is commented out in the template


def test_known_tasks_cover_the_tasks_inferred_for_pyg_models():
    inferred = set()
    for name in get_introspector().get_all_model_names():
        metadata = get_model_metadata(name)
        if metadata is not None:
            inferred.update(metadata.supported_tasks)

    assert inferred and inferred <= KNOWN_TASKS
