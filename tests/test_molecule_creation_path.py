"""One creation path per strategy, and the charge every graph is built with.

Rows come from ``datasets.registry.list_all()``, so a dataset added later is tested without
edits. Each dataset's real handler builds a molecule through ``MoleculeDataConverter``:

(a) the graph carries ``molecular_charge``, the net charge of the molecule as built;
(b) identifier-based creation records the identifier's own charge (InChI ``/q`` and ``/p``);
(c) coordinate-based creation determines bonds for the dataset's charge, and a molecule that
    cannot carry it is refused, never built another way;
(d) a creation failure refuses the molecule (no second path).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rdkit import Chem

from milia_pipeline.config.config_containers import ProcessingConfig
from milia_pipeline.config.config_loader import load_config
from milia_pipeline.datasets import registry
from milia_pipeline.exceptions import PyGDataCreationError, RDKitConversionError
from milia_pipeline.molecules.mol_conversion_utils import (
    built_molecular_charge,
    create_mol_with_dataset_support,
)
from milia_pipeline.molecules.molecule_converter_core import MoleculeDataConverter

pytestmark = pytest.mark.contract

DATASETS = sorted(registry.list_all())
WATER = {
    "coordinates": np.array([[0.0, 0.0, 0.12], [0.0, 0.76, -0.48], [0.0, -0.76, -0.48]]),
    "atomic_numbers": np.array([8, 1, 1]),
}
HYDROXIDE = {k: v[:2] for k, v in WATER.items()}
IDENTIFIERS = {
    ("inchi", 0): "InChI=1S/H2O/h1H2",
    ("inchi", -1): "InChI=1S/H2O/h1H2/p-1",
    ("smiles", 0): "O",
    ("smiles", -1): "[OH-]",
}


def _handler(dataset: str):
    selection = load_config()["data_config"]["property_selection"][dataset]
    processing = ProcessingConfig(
        scalar_graph_targets=list(selection.get("scalar_graph_targets_to_include") or []),
        node_features=[],
        node_targets=[],
        vector_graph_properties=[],
        variable_len_graph_properties=[],
        calculate_atomization_energy_from=None,
        atomization_energy_key_name=None,
    )
    dataset_config = MagicMock(name="DatasetConfig")
    dataset_config.dataset_type = dataset
    dataset_config.is_uncertainty_enabled = False
    dataset_config.uncertainty_config = None
    filter_config = MagicMock(name="FilterConfig")
    filter_config.handler_filters = {}
    handler = registry.get(dataset).create_handler(
        dataset_config, filter_config, processing, logging.getLogger("test.creation")
    )
    return handler, dataset_config


def _converter(dataset: str) -> MoleculeDataConverter:
    """The converter's creation step alone (configuration loading is tested elsewhere)."""
    handler, dataset_config = _handler(dataset)
    converter = MoleculeDataConverter.__new__(MoleculeDataConverter)
    converter._dataset_handler = handler
    converter._dataset_config = dataset_config
    converter.dataset_type = dataset
    converter.logger = logging.getLogger("test.creation")
    return converter


def _identifier_type(dataset: str) -> str:
    keys = registry.get(dataset).get_identifier_keys()
    kind = keys[0][1] if keys else "smiles"
    return kind if kind in ("inchi", "smiles") else "smiles"  # e.g. a compound id: unused


def _build(dataset: str, charge: int, raw: dict):
    id_type = _identifier_type(dataset)
    atoms = WATER if charge == 0 else HYDROXIDE
    mol_data = {"identifier": IDENTIFIERS[(id_type, charge)], "id_type": id_type, **atoms}
    return _converter(dataset)._create_base_pyg_data(0, mol_data, raw)


def test_every_registered_dataset_is_covered():
    assert DATASETS, "no dataset registered"


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_dataset_builds_a_neutral_molecule_with_its_charge(dataset):
    raw = {"molecular_charge": 0, "charge": 0, "n_electrons": 10}
    graph = _build(dataset, 0, raw)
    assert graph.molecular_charge.tolist() == [0]
    assert graph.num_nodes == 3 and graph.edge_index.shape[1] == 4  # two O-H bonds


@pytest.mark.parametrize("dataset", DATASETS)
def test_an_ion_is_built_with_its_charge_or_refused(dataset):
    """Identifier-based: the identifier's charge. Coordinate-based: the dataset's charge,
    for which bonds are determined; a dataset that reports neutral cannot build an ion."""
    dataset_class = registry.get(dataset)
    raw = {"molecular_charge": -1, "charge": -1, "n_electrons": 11}
    handler_charge = _handler(dataset)[0].get_molecular_charge(raw, HYDROXIDE["atomic_numbers"], "")
    if (
        dataset_class.get_molecule_creation_strategy() == "coordinate_based"
        and handler_charge != -1
    ):
        with pytest.raises((PyGDataCreationError, RDKitConversionError)):
            _build(dataset, -1, raw)
        return
    assert _build(dataset, -1, raw).molecular_charge.tolist() == [-1]


@pytest.mark.parametrize("dataset", DATASETS)
def test_a_creation_failure_refuses_the_molecule(dataset):
    module = "milia_pipeline.molecules.molecule_converter_core"
    with (
        patch(f"{module}.create_rdkit_mol", side_effect=RuntimeError("cannot build")),
        pytest.raises(PyGDataCreationError, match="cannot build"),
    ):
        _build(dataset, 0, {"molecular_charge": 0, "charge": 0, "n_electrons": 10})
    assert not hasattr(MoleculeDataConverter, "_create_pyg_data_with_enhanced_utils")


def test_a_coordinate_built_molecule_must_carry_the_charge_its_bonds_were_determined_for():
    handler = MagicMock()
    handler.get_molecule_creation_strategy.return_value = "coordinate_based"
    neutral = Chem.MolFromSmiles("O")
    assert built_molecular_charge(neutral, 0, handler) == 0
    with pytest.raises(RDKitConversionError, match="does not carry the dataset's charge"):
        built_molecular_charge(neutral, -1, handler)
    handler.get_molecule_creation_strategy.return_value = "identifier_coordinate_based"
    assert built_molecular_charge(Chem.MolFromSmiles("[OH-]"), 0, handler) == -1


def test_the_public_builder_uses_the_dataset_charge():
    """create_mol_with_dataset_support builds as the converter does (bonds for the charge)."""
    handler, _ = _handler("QDPi")
    graph = create_mol_with_dataset_support(
        mol_identifier="[OH-]",
        logger=logging.getLogger("test.creation"),
        molecule_index=0,
        handler=handler,
        mol_id_type="smiles",
        raw_data_dict={"molecular_charge": -1, "energy": -75.7},
        **HYDROXIDE,
    )
    assert graph.molecular_charge.tolist() == [-1]
