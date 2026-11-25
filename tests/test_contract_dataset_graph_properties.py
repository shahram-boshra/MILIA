"""Contract: every dataset builds its graph-level properties the same way, from its configuration.

Rows come from ``datasets.registry.list_all()``, so a dataset added later is tested without
edits. For each dataset, a synthetic molecule is built from the dataset's own shipped
``data_config.property_selection`` and ``property_availability`` and passed through its real
handler's ``enrich_pyg_data``:

(a) no handler overrides ``enrich_pyg_data``;
(b) ``y`` holds the effective scalar targets (configured ∩ ``provided_properties``), named in
    ``y_property_names``; every configured vector and variable-length property is attached;
(c) a configured property missing for one molecule refuses that molecule;
(d) a selection outside the dataset's schema, or a name with two roles, is refused at start;
(e) the effective selection is fixed for the run and logged once;
(f) shapes declared in ``property_availability.<dataset>.shapes`` are enforced;
(g) nothing is appended to ``y``; the atomization energy is in the dataset's energy unit.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from milia_pipeline.config import (
    get_dataset_atomic_references,
    get_graph_property_shapes,
    validate_graph_property_selection,
)
from milia_pipeline.config.config_containers import ProcessingConfig
from milia_pipeline.config.config_loader import load_config
from milia_pipeline.datasets import registry
from milia_pipeline.exceptions import ConfigurationError, HandlerError, PropertyEnrichmentError
from milia_pipeline.handlers.base_handler import DatasetHandler

pytestmark = pytest.mark.contract

DATASETS = sorted(registry.list_all())
ATOMS = torch.tensor([8, 1, 1])  # every declared reference table covers H and O
VARYING = 3  # length given to a dimension that varies between molecules


def _selection(dataset: str) -> dict:
    return load_config()["data_config"]["property_selection"][dataset]


def _availability(dataset: str) -> dict:
    return load_config()["property_availability"][dataset]


def _handler(dataset: str, **changes) -> DatasetHandler:
    selection = {**_selection(dataset), **changes}
    processing = ProcessingConfig(
        scalar_graph_targets=list(selection.get("scalar_graph_targets_to_include") or []),
        node_features=[],
        node_targets=[],
        vector_graph_properties=list(selection.get("vector_graph_properties_to_include") or []),
        variable_len_graph_properties=list(
            selection.get("variable_len_graph_properties_to_include") or []
        ),
        calculate_atomization_energy_from=selection.get("calculate_atomization_energy_from"),
        atomization_energy_key_name=selection.get("atomization_energy_key_name"),
        vibration_refinement={"comparison_tolerance": 1e-4},
    )
    dataset_config = MagicMock(name="DatasetConfig")
    dataset_config.dataset_type = dataset
    dataset_config.is_uncertainty_enabled = False
    dataset_config.uncertainty_config = None
    filter_config = MagicMock(name="FilterConfig")
    filter_config.handler_filters = {}
    return registry.get(dataset).create_handler(
        dataset_config, filter_config, processing, MagicMock(spec=logging.Logger)
    )


def _value(name: str, shapes: dict, rng: np.random.Generator):
    shape = shapes.get(name)
    if shape is None:
        return np.asarray(rng.normal(), dtype=np.float64)
    if name == "vibmodes":  # one (n_atoms, 3) mode per frequency
        return rng.normal(size=(VARYING, len(ATOMS), 3))
    return rng.normal(size=tuple(VARYING if dim is None else dim for dim in shape))


def _molecule(dataset: str) -> dict:
    """Every property the dataset lists, with its declared shape (scalars otherwise)."""
    rng = np.random.default_rng(0)
    availability = _availability(dataset)
    shapes = get_graph_property_shapes(dataset)
    properties = {
        name: _value(name, shapes, rng)
        for key in ("scalar_graph_targets", "vector_graph_properties")
        for name in availability.get(key) or []
    }
    for name in availability.get("variable_len_graph_properties") or []:
        properties[name] = _value(name, {**{name: (None,)}, **shapes}, rng)
    if "freqs" in properties:  # positive, distinct frequencies (refinement keeps them)
        properties["freqs"] = np.array([1000.0, 2000.0, 3000.0])
    properties["molecular_charge"] = 0
    properties["inchi"] = "InChI=1S/H2O/h1H2"
    return properties


def _effective_targets(handler: DatasetHandler, properties: dict) -> list[str]:
    provided = handler.provided_properties(properties)
    config = handler.processing_config
    derived = config.atomization_energy_key_name
    return [
        name
        for name in config.scalar_graph_targets
        if provided is None or name in provided or name == derived
    ]


def _enrich(handler: DatasetHandler, properties: dict, index: int = 0) -> Data:
    graph = Data(z=ATOMS.clone(), pos=torch.rand(3, 3), molecular_charge=torch.tensor([0]))
    return handler.enrich_pyg_data(graph, properties, index)


def test_every_registered_dataset_is_covered():
    assert len(DATASETS) >= 11


@pytest.mark.parametrize("dataset", DATASETS)
def test_a_handler_does_not_override_enrich_pyg_data(dataset):
    handler_class = type(_handler(dataset))
    assert "enrich_pyg_data" not in vars(handler_class)
    assert handler_class.enrich_pyg_data is DatasetHandler.enrich_pyg_data


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_shipped_selection_is_in_the_dataset_schema(dataset):
    selection = _selection(dataset)
    validate_graph_property_selection(
        dataset,
        selection,
        selection.get("calculate_atomization_energy_from"),
        selection.get("atomization_energy_key_name"),
        [
            *(selection.get("node_features_to_add") or []),
            *(selection.get("node_targets_to_include") or []),
        ],
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_configured_properties_are_attached_with_names(dataset):
    handler = _handler(dataset)
    properties = _molecule(dataset)
    data = _enrich(handler, properties)
    targets = _effective_targets(handler, properties)
    if targets:
        assert data.y_property_names == targets
        assert tuple(data.y.shape) == (len(targets),)  # (g) nothing appended
    else:
        assert getattr(data, "y", None) is None
    shapes = get_graph_property_shapes(dataset)
    for name in [
        *handler.processing_config.vector_graph_properties,
        *handler.processing_config.variable_len_graph_properties,
    ]:
        value = getattr(data, name)
        stacked = torch.stack(value) if isinstance(value, list) else value
        declared = shapes.get(name)
        if declared is not None:
            assert stacked.dim() == len(declared)
            assert all(d is None or d == s for d, s in zip(declared, stacked.shape, strict=True))
    assert data.dataset_type == dataset


@pytest.mark.parametrize("dataset", DATASETS)
def test_a_missing_configured_property_refuses_the_molecule(dataset):
    handler = _handler(dataset)
    properties = _molecule(dataset)
    config = handler.processing_config
    configured = [
        *_effective_targets(handler, properties),
        *config.vector_graph_properties,
        *config.variable_len_graph_properties,
    ]
    configured = [name for name in configured if name != config.atomization_energy_key_name]
    if not configured:
        pytest.skip(f"{dataset} configures no graph-level property")
    _enrich(handler, properties)  # fixes the run's selection
    del properties[configured[-1]]
    with pytest.raises((PropertyEnrichmentError, HandlerError)):
        _enrich(handler, properties, index=1)


@pytest.mark.parametrize("dataset", DATASETS)
def test_an_unknown_property_is_refused_at_start(dataset):
    selection = dict(_selection(dataset))
    selection["scalar_graph_targets_to_include"] = [
        *(selection.get("scalar_graph_targets_to_include") or []),
        "not_a_property",
    ]
    with pytest.raises(ConfigurationError, match="not_a_property"):
        validate_graph_property_selection(dataset, selection, None, None, [])


@pytest.mark.parametrize("dataset", DATASETS)
def test_a_name_with_two_roles_is_refused_at_start(dataset):
    vectors = _availability(dataset).get("vector_graph_properties") or []
    if not vectors:
        pytest.skip(f"{dataset} has no vector property")
    selection = {
        "scalar_graph_targets_to_include": [],
        "vector_graph_properties_to_include": [vectors[0]],
    }
    with pytest.raises(ConfigurationError, match="role"):
        validate_graph_property_selection(dataset, selection, None, None, [vectors[0]])


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_selection_is_fixed_for_the_run_and_logged_once(dataset):
    handler = _handler(dataset)
    first = _molecule(dataset)
    _enrich(handler, first)
    second = {**_molecule(dataset), "_feature_tier": "complete"}
    data = _enrich(handler, second, index=1)
    targets = _effective_targets(handler, first)
    assert getattr(data, "y_property_names", []) in (targets, [])
    logged = [
        call
        for call in handler.logger.info.call_args_list
        if "graph properties for this run" in call.args[0]
    ]
    assert len(logged) == 1


@pytest.mark.parametrize("dataset", DATASETS)
def test_declared_shapes_are_enforced(dataset):
    handler = _handler(dataset)
    shapes = get_graph_property_shapes(dataset)
    config = handler.processing_config
    fixed = [
        name
        for name in config.vector_graph_properties
        if shapes.get(name) and all(dim is not None for dim in shapes[name])
    ]
    if not fixed:
        pytest.skip(f"{dataset} configures no vector property with a declared shape")
    properties = _molecule(dataset)
    properties[fixed[0]] = np.zeros(shapes[fixed[0]][0] + 1)
    with pytest.raises((PropertyEnrichmentError, HandlerError)):
        _enrich(handler, properties)


@pytest.mark.parametrize("dataset", DATASETS)
def test_atomization_energy_is_in_the_dataset_energy_unit(dataset):
    selection = _selection(dataset)
    energy_key = selection.get("calculate_atomization_energy_from")
    references = get_dataset_atomic_references(dataset)
    table = references.for_property(energy_key) if references and energy_key else None
    if table is None:
        pytest.skip(f"{dataset} computes no atomization energy")
    handler = _handler(dataset)
    properties = _molecule(dataset)
    data = _enrich(handler, properties)
    attribute = selection["atomization_energy_key_name"]
    expected = float(properties[energy_key]) - sum(table[int(z)] for z in ATOMS)
    assert float(getattr(data, attribute)) == pytest.approx(expected, abs=1e-5)


def _config_with(monkeypatch, availability: dict) -> None:
    """Patch the configuration the two functions read (their own module, even if reloaded)."""
    config = {"property_availability": {"X": availability}}
    for function in (get_graph_property_shapes, validate_graph_property_selection):
        monkeypatch.setitem(function.__globals__, "load_config", lambda: config)


@pytest.mark.parametrize("shape", [[], [0], [3.0], "3", [True], [-1]])
def test_a_malformed_declared_shape_is_refused(monkeypatch, shape):
    _config_with(monkeypatch, {"vector_graph_properties": ["v"], "shapes": {"v": shape}})
    with pytest.raises(ConfigurationError, match="shapes"):
        get_graph_property_shapes("X")


def test_null_dimensions_declare_varying_lengths(monkeypatch):
    _config_with(monkeypatch, {"shapes": {"m": [None, None, 3]}})
    assert get_graph_property_shapes("X") == {"m": (None, None, 3)}


def test_a_shape_for_a_property_the_dataset_lacks_is_refused(monkeypatch):
    _config_with(monkeypatch, {"vector_graph_properties": ["v"], "shapes": {"w": [3]}})
    with pytest.raises(ConfigurationError, match="'w'"):
        validate_graph_property_selection("X", {}, None, None, [])


def test_atomization_energy_needs_both_keys_and_a_scalar_source(monkeypatch):
    _config_with(monkeypatch, {"scalar_graph_targets": ["E"]})
    with pytest.raises(ConfigurationError, match="together"):
        validate_graph_property_selection("X", {}, "E", None, [])
    with pytest.raises(ConfigurationError, match="not a scalar property"):
        validate_graph_property_selection("X", {}, "F", "F_ATOM", [])
    with pytest.raises(ConfigurationError, match="already a property"):
        validate_graph_property_selection("X", {}, "E", "E", [])


def test_the_atomization_energy_may_be_a_named_target(monkeypatch):
    _config_with(monkeypatch, {"scalar_graph_targets": ["E"]})
    selection = {"scalar_graph_targets_to_include": ["E", "E_ATOM"]}
    validate_graph_property_selection("X", selection, "E", "E_ATOM", [])
    with pytest.raises(ConfigurationError, match="E_ATOM"):
        validate_graph_property_selection("X", selection, None, None, [])


def test_a_repeated_name_is_refused(monkeypatch):
    _config_with(monkeypatch, {"scalar_graph_targets": ["E"]})
    with pytest.raises(ConfigurationError, match="more than once"):
        validate_graph_property_selection(
            "X", {"scalar_graph_targets_to_include": ["E", "E"]}, None, None, []
        )


def test_a_graph_property_that_is_also_per_atom_is_refused(monkeypatch):
    _config_with(monkeypatch, {"scalar_graph_targets": ["E"]})
    with pytest.raises(ConfigurationError, match="role"):
        validate_graph_property_selection(
            "X", {"scalar_graph_targets_to_include": ["E"]}, None, None, ["E"]
        )
