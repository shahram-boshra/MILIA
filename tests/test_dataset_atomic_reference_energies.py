"""Atomic reference energies at each dataset's level of theory, with their provenance.

An atomization energy subtracts the atoms' energies from the molecular energy, so both must come
from the same level of theory. References are declared per dataset in
``<dataset>_config.atomic_references`` (author-published, milia-computed or user-supplied); a dataset
without them does not compute its atomization energy. See docs/atomic_reference_energies.md.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from milia_pipeline.config import config_constants, get_dataset_atomic_references
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
