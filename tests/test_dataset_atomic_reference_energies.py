"""Atomic reference energies at each dataset's level of theory, with their provenance.

An atomization energy subtracts the atoms' energies from the molecular energy, so both must come
from the same level of theory. References are declared per dataset in
``<dataset>_config.atomic_references`` (author-published, milia-computed or user-supplied); a dataset
without them does not compute its atomization energy. See docs/atomic_reference_energies.md.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from milia_pipeline.config import config_constants, get_dataset_atomic_references
from milia_pipeline.config.config_loader import load_config
from milia_pipeline.datasets.milia_dataset import miliaDataset
from milia_pipeline.exceptions import ConfigurationError
from milia_pipeline.handlers.implementations.xxmd import XXMDDatasetHandler
from milia_pipeline.molecules.property_enrichment import calculate_atomization_energy

REPO = Path(__file__).resolve().parents[1]

# Psi4 1.11, UKS M06/6-31G, defaults; docs/atomic_references/xxmd/results.json rounded to 1e-10 Eh.
XXMD_M06_631G = {
    1: -0.4979196097,
    6: -37.8125682417,
    7: -54.5569858687,
    8: -75.0254051530,
    16: -398.0661265660,
}

# QM9 atomref.txt (Ramakrishnan et al. 2014), verbatim.
QM9_ATOMREF = {
    "U0": {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579, 9: -99.718730},
    "U": {1: -0.498857, 6: -37.845355, 7: -54.582445, 8: -75.063163, 9: -99.717314},
    "H": {1: -0.497912, 6: -37.844411, 7: -54.581501, 8: -75.062219, 9: -99.716370},
    "G": {1: -0.510927, 6: -37.861317, 7: -54.598897, 8: -75.079532, 9: -99.733544},
}

VALID = {
    "source": "user-supplied",
    "level_of_theory": "PBE/def2-SVP",
    "program": "ORCA 5",
    "citation": "user",
    "energies_hartree": {"energy": {1: -0.5}},
}


def test_qm9_uses_the_published_atomref_table():
    references = get_dataset_atomic_references("QM9")
    assert references.source == "author-published"
    assert {key: dict(table) for key, table in references.energies_hartree.items()} == QM9_ATOMREF


def test_xxmd_uses_its_computed_level_of_theory_references():
    references = get_dataset_atomic_references("XXMD")
    assert references.source == "milia-computed"
    assert dict(references.for_property("energy")) == XXMD_M06_631G


def test_xxmd_references_match_the_committed_psi4_results():
    results = json.loads(
        (REPO / "docs/atomic_references/xxmd/results.json").read_text(encoding="utf-8")
    )
    computed = {
        atom["Z"]: round(atom["selected_energy_hartree"], 10) for atom in results["atoms"].values()
    }
    assert computed == XXMD_M06_631G


def test_a_dataset_without_references_has_none():
    assert get_dataset_atomic_references("ANI1X") is None


def test_references_apply_only_to_their_energy_property():
    assert get_dataset_atomic_references("XXMD").for_property("forces") is None


@pytest.mark.parametrize(
    "invalid",
    [
        [1],
        {**VALID, "source": "estimated"},
        {**VALID, "level_of_theory": ""},
        {key: value for key, value in VALID.items() if key != "citation"},
        {**VALID, "energies_hartree": {}},
        {**VALID, "energies_hartree": {"energy": {}}},
        {**VALID, "energies_hartree": {"energy": {"H": -0.5}}},
        {**VALID, "energies_hartree": {"energy": {1: "x"}}},
        {**VALID, "energies_hartree": {"energy": {True: -0.5}}},
        {**VALID, "energies_hartree": {"energy": {0: -0.5}}},
    ],
)
def test_invalid_declarations_raise(monkeypatch, invalid):
    def load_config(*args, **kwargs):
        return {"xxmd_config": {"atomic_references": invalid}}

    monkeypatch.setattr("milia_pipeline.config.config_loader.load_config", load_config)
    with pytest.raises(ConfigurationError, match=r"xxmd_config\.atomic_references"):
        get_dataset_atomic_references("XXMD")


def _handler(energy_key="energy", attribute="energy_ATOM"):
    handler = XXMDDatasetHandler.__new__(XXMDDatasetHandler)
    handler.logger = Mock(spec=logging.Logger)
    handler.processing_config = Mock(
        calculate_atomization_energy_from=energy_key, atomization_energy_key_name=attribute
    )
    return handler


def test_without_references_the_atomization_energy_is_not_computed(monkeypatch):
    monkeypatch.setattr(
        "milia_pipeline.handlers.base_handler.get_dataset_atomic_references", lambda _: None
    )
    handler = _handler()
    data = Mock(z=torch.tensor([8, 1, 1]))
    for index in range(3):
        assert (
            handler._calculate_atomization_energy_internal({"energy": -76.0}, data, index, "m")
            is None
        )
    handler.logger.warning.assert_called_once()
    assert "not computed" in handler.logger.warning.call_args.args[0]


def test_provenance_records_the_references_used():
    record = _handler().atomic_references_provenance()
    assert record["atomization_energy"] == "computed"
    assert record["source"] == "milia-computed"
    assert record["energy_property"] == "energy"
    assert record["energies_hartree"] == {"energy": {str(z): e for z, e in XXMD_M06_631G.items()}}
    json.dumps(record)


def test_provenance_records_why_the_atomization_energy_is_absent(monkeypatch):
    monkeypatch.setattr(
        "milia_pipeline.handlers.base_handler.get_dataset_atomic_references", lambda _: None
    )
    record = _handler().atomic_references_provenance()
    assert record["atomization_energy"] == "not computed"
    assert "xxmd_config.atomic_references" in record["reason"]
    assert _handler(energy_key=None).atomic_references_provenance()["atomization_energy"] == (
        "not configured"
    )


def test_provenance_sidecar_is_written_next_to_the_processed_file(tmp_path):
    processed = tmp_path / "xxmd.pt"
    dataset = miliaDataset.__new__(miliaDataset)
    dataset._dataset_handler = _handler()
    dataset.logger = Mock(spec=logging.Logger)
    dataset._write_atomic_references_provenance(processed)
    sidecar = miliaDataset.atomic_references_provenance_path(processed)
    assert sidecar == tmp_path / "xxmd.atomic_references.json"
    assert json.loads(sidecar.read_text(encoding="utf-8"))["source"] == "milia-computed"


def test_the_global_atomic_energies_are_deprecated():
    with pytest.warns(DeprecationWarning, match="atomic_energies_hartree"):
        config_constants.ATOMIC_ENERGIES_HARTREE  # noqa: B018


def test_atomization_without_references_uses_the_deprecated_global_ones():
    z = torch.tensor([1, 1])
    with pytest.warns(DeprecationWarning, match="atomic_energies_hartree"):
        calculate_atomization_energy(-1.2, z, 0, logging.getLogger(__name__))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        energy = calculate_atomization_energy(
            -1.2, z, 0, logging.getLogger(__name__), atomic_energies_hartree={1: -0.5}
        )
    assert energy == pytest.approx((-1.2 + 1.0) * config_constants.get_har2ev())


def test_vqm24_dft_uses_its_published_atomization_energy():
    """VQM24 publishes Eatomization per molecule, so MILIA computes no atomization energy for it."""
    config = load_config()
    assert (
        config["data_config"]["property_selection"]["DFT"]["calculate_atomization_energy_from"]
        is None
    )
    assert "Eatomization" in config["property_availability"]["DFT"]["scalar_graph_targets"]
    assert get_dataset_atomic_references("DFT") is None


# Ground-state spin multiplicities of the isolated atoms.
MULTIPLICITY = {"H": 2, "C": 3, "N": 4, "O": 3, "F": 2, "S": 3, "Cl": 2}
REFERENCE_INPUTS = {
    "qm40": ("gjf", {"H", "C", "N", "O", "F", "S", "Cl"}, "UB3LYP/6-31G(2df,p) Int=UltraFine"),
    "ani1x": ("gjf", {"H", "C", "N", "O"}, "UwB97X/6-31G* Int=FineGrid"),
    "ani2x": ("gjf", {"H", "C", "N", "O", "F", "S", "Cl"}, "UwB97X/6-31G* Int=FineGrid"),
    "rmd17": (
        "inp",
        {"H", "C", "N", "O"},
        "UKS PBE def2-SVP def2/J RI VeryTightSCF Grid7 NoFinalGrid",
    ),
}


@pytest.mark.parametrize("dataset", sorted(REFERENCE_INPUTS))
def test_reference_inputs_cover_each_element_at_its_ground_state(dataset):
    suffix, elements, method = REFERENCE_INPUTS[dataset]
    found = set()
    for path in (REPO / "docs/atomic_references" / dataset / "inputs").glob(f"*.{suffix}"):
        text = path.read_text(encoding="utf-8")
        assert method in text, path
        charge_multiplicity = re.search(r"^(?:\* xyz )?0 (\d)$", text, re.MULTILINE)
        element = path.stem.split("_")[0]
        assert int(charge_multiplicity.group(1)) == MULTIPLICITY[element], path
        found.add(element)
    assert found == elements
    assert (REPO / "docs/atomic_references" / dataset / "README.md").is_file()


def _parse(tmp_path, program, files):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for name, text in files.items():
        (outputs / name).write_text(text, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPO / "docs/atomic_references/parse_outputs.py"),
            str(tmp_path),
            "--program",
            program,
        ],
        capture_output=True,
        text=True,
    )


def test_parse_outputs_reads_gaussian_energies_and_thermochemistry(tmp_path):
    log = (
        " Initial command:\n SCF Done:  E(UB3LYP) =  -37.8467720  A.U.\n Normal termination of Gaussian 16\n"
        " Initial command:\n SCF Done:  E(UB3LYP) =  -37.8467720  A.U.\n"
        " Sum of electronic and zero-point Energies=  -37.846772\n"
        " Sum of electronic and thermal Energies=  -37.845355\n"
        " Sum of electronic and thermal Enthalpies=  -37.844411\n"
        " Sum of electronic and thermal Free Energies=  -37.861317\n Normal termination of Gaussian 16\n"
    )
    assert _parse(tmp_path, "gaussian", {"C.log": log}).returncode == 0
    energies = json.loads((tmp_path / "results.json").read_text())["atoms"]["C"]["energies_hartree"]
    assert energies == {
        "scf": -37.846772,
        "U0": -37.846772,
        "U298": -37.845355,
        "H298": -37.844411,
        "G298": -37.861317,
    }


def test_parse_outputs_keeps_the_lowest_orca_guess(tmp_path):
    done = "****ORCA TERMINATED NORMALLY****\n"
    files = {
        "C_pmodel.out": "FINAL SINGLE POINT ENERGY   -37.801\n" + done,
        "C_hcore.out": "FINAL SINGLE POINT ENERGY   -37.802\n" + done,
    }
    assert _parse(tmp_path, "orca", files).returncode == 0
    atom = json.loads((tmp_path / "results.json").read_text())["atoms"]["C"]
    assert atom["selected_guess"] == "hcore"
    assert atom["selected_energy_hartree"] == -37.802


def test_parse_outputs_rejects_an_abnormal_termination(tmp_path):
    log = " Initial command:\n Initial command:\n SCF Done:  E(U) = -1.0\n Normal termination of Gaussian 16\n"
    result = _parse(tmp_path, "gaussian", {"C.log": log})
    assert result.returncode != 0
    assert "terminated normally" in result.stderr
