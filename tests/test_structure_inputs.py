"""Structure inputs: built as the training dataset built its molecules.

Matrix rows come from ``datasets.registry.list_all()`` (a dataset added later is tested without
edits), so both creation strategies and both coordinate units are covered:

(a) every registered dataset builds a structure through ``StructureBuilder``;
(b) the structure graph equals the training graph of the same molecule, built by the converter
    steps ``convert`` runs (creation, structural features), on ``x``, ``edge_index``,
    ``edge_attr``, ``pos``, ``z`` and ``molecular_charge``, for a neutral molecule and an ion.

Then the charge rules, the readers (XYZ, SDF/MOL with hydrogens kept), the input contract and
``Predictor.prepare_structures``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from milia_pipeline.config.config_constants import get_bohr_to_angstrom
from milia_pipeline.config.config_containers import DatasetConfig, FilterConfig, ProcessingConfig
from milia_pipeline.config.config_loader import load_config
from milia_pipeline.datasets import registry
from milia_pipeline.exceptions import ConfigurationError, StructureInputError
from milia_pipeline.models.input_contract import (
    DATASET,
    IDENTIFIER,
    STRUCTURE,
    STRUCTURE_RECORDS,
    input_contract,
)
from milia_pipeline.molecules.featurization_record import (
    creation_switches,
    molecule_creation_record,
)
from milia_pipeline.molecules.molecule_converter_core import MoleculeDataConverter
from milia_pipeline.molecules.structure_inputs import (
    StructureBuilder,
    StructureInput,
    read_sdf,
    read_structures,
    read_xyz,
)

pytestmark = pytest.mark.contract

DATASETS = sorted(registry.list_all())
COMPARED = ("x", "edge_index", "edge_attr", "pos", "z", "molecular_charge")
# Features that read the given coordinates (bond_length) and stereochemistry (chirality, stereo),
# none reading a value only the dataset holds
FEATURES = {
    "atom": ["degree", "hybridization", "is_aromatic", "chirality"],
    "bond": ["bond_type", "stereo", "bond_length"],
}
WATER = (
    np.array([8, 1, 1]),
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]),
)
HYDROXIDE = (np.array([8, 1]), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]))
IDENTIFIERS = {
    ("inchi", 0): "InChI=1S/H2O/h1H2",
    ("inchi", -1): "InChI=1S/H2O/h1H2/p-1",
    ("smiles", 0): "O",
    ("smiles", -1): "[OH-]",
}


def _configs(dataset):
    dataset_config = DatasetConfig(dataset_type=dataset)
    return dataset_config, FilterConfig(), ProcessingConfig(scalar_graph_targets=[])


def _identifier(dataset, charge):
    keys = registry.get(dataset).get_identifier_keys()
    if not keys or keys[0][1] not in ("inchi", "smiles"):
        return None, None, None
    key, kind = keys[0]
    return key, kind, IDENTIFIERS[(kind, charge)]


def _entry(dataset, atoms, angstrom, charge):
    """A dataset entry: every scalar property the dataset lists, its atoms, its coordinates."""
    rng = np.random.default_rng(0)
    availability = load_config()["property_availability"][dataset]
    entry = {name: np.float64(rng.normal()) for name in availability["scalar_graph_targets"]}
    units = registry.get(dataset).get_coordinate_units()
    coordinates = angstrom / get_bohr_to_angstrom() if units == "bohr" else angstrom
    electrons = int(atoms.sum()) - charge
    entry.update(
        atoms=atoms,
        coordinates=coordinates,
        molecular_charge=charge,
        charge=charge,
        n_electrons=electrons,
    )
    key, _, identifier = _identifier(dataset, charge)
    if key is not None:
        entry[key] = identifier
    return entry


def _training_graph(dataset, entry):
    """The training graph's inputs, by the converter steps ``convert`` runs to build them."""
    dataset_config, filter_config, processing_config = _configs(dataset)
    handler = registry.get(dataset).create_handler(
        dataset_config, filter_config, processing_config, logging.getLogger("test")
    )
    converter = MoleculeDataConverter(
        dataset_handler=handler,
        dataset_config=dataset_config,
        filter_config=filter_config,
        processing_config=processing_config,
        structural_features_config=FEATURES,
    )
    _, kind, identifier = _identifier(dataset, int(entry["charge"]))
    mol_data = {
        "identifier": identifier or "molecule_0",
        "id_type": kind or "index",
        "coordinates": entry["coordinates"],
        "atomic_numbers": entry["atoms"],
    }
    graph = converter._create_base_pyg_data(0, mol_data, entry)
    return converter._add_structural_features_enhanced(graph, 0, mol_data["identifier"], entry)


def _builder(dataset, charges=(0,)):
    creation = molecule_creation_record(dataset)
    creation["charges"] = list(charges)
    return StructureBuilder(creation, FEATURES)


def _structure(dataset, atoms, angstrom, charge, state_charge=True):
    _, kind, identifier = _identifier(dataset, charge)
    return StructureInput.create(
        atoms,
        angstrom,
        charge=charge if state_charge else None,
        identifier=identifier,
        identifier_type=kind,
    )


def test_every_registered_dataset_is_covered():
    assert DATASETS
    strategies = {registry.get(name).get_molecule_creation_strategy() for name in DATASETS}
    assert strategies == {"identifier_coordinate_based", "coordinate_based"}


@pytest.mark.parametrize("dataset", DATASETS)
def test_a_structure_gives_the_training_graph_of_its_molecule(dataset):
    atoms, angstrom = WATER
    training = _training_graph(dataset, _entry(dataset, atoms, angstrom, 0))
    graph = _builder(dataset).build_one(_structure(dataset, atoms, angstrom, 0))
    for name in COMPARED:
        assert torch.equal(getattr(graph, name), getattr(training, name)), name
    assert graph.x.shape == (3, training.x.shape[1]) and "y" not in graph


@pytest.mark.parametrize(
    "dataset",
    [
        name
        for name in DATASETS
        # an ion is built where the dataset's charge rule gives it: identifier-based (the
        # identifier's charge) and a coordinate-based dataset that stores charges (QDPi)
        if registry.get(name).get_molecule_creation_strategy() == "identifier_coordinate_based"
        or name == "QDPi"
    ],
)
def test_an_ion_gives_the_training_graph_with_its_charge(dataset):
    atoms, angstrom = HYDROXIDE
    training = _training_graph(dataset, _entry(dataset, atoms, angstrom, -1))
    graph = _builder(dataset, charges=(-1, 0)).build_one(_structure(dataset, atoms, angstrom, -1))
    for name in COMPARED:
        assert torch.equal(getattr(graph, name), getattr(training, name)), name
    assert graph.molecular_charge.tolist() == [-1]


def _coordinate_dataset():
    return next(
        name
        for name in DATASETS
        if registry.get(name).get_molecule_creation_strategy() == "coordinate_based"
        and registry.get(name).get_coordinate_units() == "angstrom"
    )


def _identifier_dataset():
    return next(
        name
        for name in DATASETS
        if registry.get(name).get_molecule_creation_strategy() == "identifier_coordinate_based"
    )


def test_coordinate_based_creation_needs_the_stated_charge():
    dataset = _coordinate_dataset()
    with pytest.raises(StructureInputError, match="needs the molecular charge") as info:
        _builder(dataset).build_one(_structure(dataset, *WATER, 0, state_charge=False))
    assert info.value.refusals[0][0] == 0


def test_a_charge_the_model_was_not_trained_on_is_refused():
    """OECD (Q)SAR principle 3: a defined domain of applicability."""
    for dataset in (_coordinate_dataset(), _identifier_dataset()):
        with pytest.raises(StructureInputError, match=r"outside the charges .* \[0\]"):
            _builder("QDPi" if dataset == "QDPi" else dataset).build_one(
                _structure(dataset, *HYDROXIDE, -1)
            )


def test_identifier_creation_takes_the_identifier_charge_and_checks_a_stated_one():
    dataset = _identifier_dataset()
    builder = _builder(dataset, charges=(-1, 0))
    _, kind, identifier = _identifier(dataset, 0)
    neutral = StructureInput.create(*WATER, identifier=identifier, identifier_type=kind)
    assert builder.build_one(neutral).molecular_charge.tolist() == [0]
    stated = StructureInput.create(*WATER, charge=-1, identifier=identifier, identifier_type=kind)
    with pytest.raises(StructureInputError, match="differs from the identifier's charge 0"):
        builder.build_one(stated)
    with pytest.raises(StructureInputError, match="reads bonds from an identifier"):
        builder.build_one(StructureInput.create(*WATER, charge=0))


def test_an_identifier_of_another_kind_is_converted_to_the_dataset_kind():
    dataset = _identifier_dataset()  # its first identifier kind is InChI
    builder = _builder(dataset)
    from_smiles = builder.build_one(
        StructureInput.create(*WATER, identifier="O", identifier_type="smiles")
    )
    from_inchi = builder.build_one(_structure(dataset, *WATER, 0))
    for name in COMPARED:
        assert torch.equal(getattr(from_smiles, name), getattr(from_inchi, name)), name
    assert from_smiles.inchi == "InChI=1S/H2O/h1H2"


def test_every_refused_structure_of_a_request_is_reported():
    dataset = _coordinate_dataset()
    builder = _builder(dataset)
    structures = [
        _structure(dataset, *WATER, 0),
        _structure(dataset, *WATER, 0, state_charge=False),
        _structure(dataset, *HYDROXIDE, -1),
    ]
    with pytest.raises(StructureInputError) as info:
        builder.build(structures)
    assert [index for index, _ in info.value.refusals] == [1, 2]
    assert len(builder.build(structures[:1])) == 1


def test_structure_input_validation():
    with pytest.raises(ValueError, match="coordinate triples"):
        StructureInput(atomic_numbers=(8, 1), coordinates=((0.0, 0.0, 0.0),))
    with pytest.raises(ValueError, match="1-118"):
        StructureInput.create([0], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        StructureInput.create([8], [[np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match="together"):
        StructureInput.create([8], [[0.0, 0.0, 0.0]], identifier="O")
    with pytest.raises(ValueError, match="shape"):
        StructureInput.create([8], [0.0, 0.0, 0.0])


# ----------------------------------------------------------------------------- readers


def test_xyz_frames_with_extended_xyz_keys(tmp_path):
    path = tmp_path / "two.xyz"
    path.write_text(
        "3\ncharge=0 inchi=InChI=1S/H2O/h1H2 name=water\n"
        "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n"
        '2\ncharge=-1 smiles="[OH-]" Lattice="1 0 0 0 1 0 0 0 1"\n'
        "8 0.0 0.0 0.0\n1 0.0 0.0 0.97\n"
    )
    water, hydroxide = read_xyz(path)
    assert water.atomic_numbers == (8, 1, 1) and water.charge == 0
    assert (water.identifier, water.identifier_type, water.name) == (
        "InChI=1S/H2O/h1H2",
        "inchi",
        "water",
    )
    assert hydroxide.charge == -1 and hydroxide.identifier == "[OH-]"
    assert hydroxide.name == "two.xyz:1" and hydroxide.coordinates[1] == (0.0, 0.0, 0.97)


def test_a_plain_xyz_frame_states_nothing(tmp_path):
    path = tmp_path / "plain.xyz"
    path.write_text("1\nanything here\nHe 0 0 0\n")
    (structure,) = read_structures(path)
    assert structure.charge is None and structure.identifier is None


@pytest.mark.parametrize(
    "text", ["3\n\nO 0 0 0\nH 0 0 1\n", "1\n\nXx 0 0 0\n", "2\n\nO 0 0\nH 0 0 1\n"]
)
def test_a_malformed_xyz_frame_is_refused_not_skipped(tmp_path, text):
    path = tmp_path / "bad.xyz"
    path.write_text(text)
    with pytest.raises(ValueError, match="frame 0"):
        read_xyz(path)


def _water_block(dimensions="3D"):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    if dimensions == "2D":
        AllChem.Compute2DCoords(mol)
    mol.SetProp("_Name", "acetate")
    return mol, Chem.MolToMolBlock(mol)


def test_sdf_keeps_every_hydrogen_its_coordinates_and_charge(tmp_path):
    """RDKit removes hydrogens by default (``removeHs=True``); a structure keeps them."""
    from rdkit import Chem

    mol, block = _water_block()
    path = tmp_path / "acetate.sdf"
    path.write_text(block + "$$$$\n" + block + "$$$$\n")
    first, second = read_sdf(path)
    assert first.atomic_numbers == tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
    assert first.atomic_numbers.count(1) == 3 and first.charge == -1 and first.name == "acetate"
    assert first.identifier == Chem.MolToInchi(mol) and first.identifier_type == "inchi"
    np.testing.assert_allclose(first.coordinates, mol.GetConformer().GetPositions(), atol=1e-4)
    assert second == first
    assert Chem.SDMolSupplier(str(path))[0].GetNumAtoms() < len(first.atomic_numbers)
    single = tmp_path / "acetate.mol"
    single.write_text(block)
    assert read_structures(single) == [first]


def test_a_2d_record_or_an_unreadable_one_is_refused(tmp_path):
    _, flat = _water_block("2D")
    path = tmp_path / "flat.sdf"
    path.write_text(flat + "$$$$\n")
    with pytest.raises(ValueError, match="record 0: the molecule has no 3D coordinates"):
        read_sdf(path)
    broken = tmp_path / "broken.sdf"
    broken.write_text("not a molfile\n$$$$\n")
    with pytest.raises(ValueError, match="record 0 cannot be read"):
        read_sdf(broken)
    with pytest.raises(ValueError, match="not a structure file"):
        read_structures(tmp_path / "molecules.csv")


def test_an_sdf_record_builds_as_its_identifier_does(tmp_path):
    """An SDF record carries its InChI, so identifier-based creation needs nothing more."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    path = tmp_path / "water.sdf"
    path.write_text(Chem.MolToMolBlock(mol) + "$$$$\n")
    (structure,) = read_sdf(path)
    graph = _builder(_identifier_dataset()).build_one(structure)
    assert graph.z.tolist() == [8, 1, 1] and graph.molecular_charge.tolist() == [0]


# ----------------------------------------------------------------------------- contract


def _recorded(dataset="QDPi", charges=(0,), **changes):
    creation = molecule_creation_record(dataset)
    creation["charges"] = list(charges)
    return {
        "explicit_hydrogens": True,
        "qm_input_features": [],
        "qm_input_sources": {},
        "input_transforms": None,
        "input_filters": {"max_atoms": None, "min_atoms": None, "heavy_atom_filter": None},
        "structural_features_config": FEATURES,
        "molecule_creation": creation,
        **changes,
    }


def test_a_fully_recorded_model_accepts_structures():
    contract = input_contract(_recorded(), requires_positions=True)
    assert contract.accepted == [STRUCTURE, DATASET]  # positions: identifiers refused
    contract.require(STRUCTURE)
    assert input_contract(_recorded(), requires_positions=False).accepted == [
        IDENTIFIER,
        STRUCTURE,
        DATASET,
    ]


@pytest.mark.parametrize("record", STRUCTURE_RECORDS)
def test_a_missing_record_refuses_structures(record):
    data_info = {k: v for k, v in _recorded().items() if k != record}
    reason = input_contract(data_info, requires_positions=True).refused[STRUCTURE]
    assert f"does not record {record}" in reason


@pytest.mark.parametrize("name", ["charges", "featurization"])
def test_unrecorded_charges_or_switches_refuse_structures(name):
    data_info = _recorded()
    data_info["molecule_creation"][name] = None
    reason = input_contract(data_info, requires_positions=True).refused[STRUCTURE]
    assert f"molecule_creation.{name}" in reason and "re-process" in reason


def test_inputs_from_the_dataset_refuse_structures_naming_them():
    data_info = _recorded(
        qm_input_sources={
            "bond.bond_length": "coordinates",
            "atom.atomic_charge": "dataset_property",
        }
    )
    reason = input_contract(data_info, requires_positions=False).refused[STRUCTURE]
    assert "atom.atomic_charge from its dataset's QM data" in reason
    assert "bond.bond_length" not in reason  # a structure carries its coordinates


def test_the_record_holds_the_switches_training_read():
    creation = molecule_creation_record(_identifier_dataset())
    assert creation["featurization"] == creation_switches()
    assert set(creation_switches()) == {
        "structural_features",
        "pass_coordinates",
        "stereochemistry",
        "stereochemistry_cleanup",
    }


def _predictor(data_info, model=None):
    from milia_pipeline.models.post_training.inference.predictor import Predictor

    return Predictor(
        model=model or nn.Identity(),
        working_root_dir=".",
        device=torch.device("cpu"),
        model_info={"data_info": data_info},
    )


def test_a_predictor_prepares_structures_through_the_training_transforms():
    from milia_pipeline.transformations.pre_transform_record import (
        RECORD_VERSION,
        load_input_transform,
    )

    record = {
        "version": RECORD_VERSION,
        "transforms": [{"name": "NormalizeFeatures", "kwargs": {"attrs": ["x"]}}],
        "repr": "Compose([\n  NormalizeFeatures()\n])",
    }
    dataset = _coordinate_dataset()
    predictor = _predictor(_recorded(dataset, input_transforms=record))
    (graph,) = predictor.prepare_structures([_structure(dataset, *WATER, 0)])
    plain = _builder(dataset).build_one(_structure(dataset, *WATER, 0))
    expected = load_input_transform(record)(plain).x
    assert torch.equal(graph.x, expected) and not torch.equal(graph.x, plain.x)
    with pytest.raises(StructureInputError, match="needs the molecular charge"):
        predictor.prepare_structures([_structure(dataset, *WATER, 0, state_charge=False)])


def test_a_predictor_refuses_structures_its_model_does_not_accept():
    predictor = _predictor(_recorded(qm_input_sources={"x.Vesp": "dataset_property"}))
    with pytest.raises(ConfigurationError, match="does not accept structure inputs"):
        predictor.prepare_structures([_structure("QDPi", *WATER, 0)])
