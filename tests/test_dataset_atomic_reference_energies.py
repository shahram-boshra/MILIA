"""Atomic reference energies at each dataset's level of theory.

An atomization energy subtracts the atoms' energies from the molecular energy, so both must come
from the same level of theory. xxMD-DFT (UKS M06/6-31G) sets its own references in
``configs/datasets/xxmd.yaml``; a dataset without its own uses the global ``atomic_energies_hartree``.
"""

from __future__ import annotations

import pytest

from milia_pipeline.config import config_constants
from milia_pipeline.exceptions import ConfigurationError

# Psi4 1.11, UKS M06/6-31G, default grid; see the provenance comment in configs/datasets/xxmd.yaml.
XXMD_M06_631G = {
    1: -0.4979196097,
    6: -37.8125682417,
    7: -54.5569858687,
    8: -75.0254051528,
    16: -398.0661265642,
}


def test_xxmd_uses_its_level_of_theory_references():
    assert config_constants.get_dataset_atomic_energies_hartree("XXMD") == XXMD_M06_631G


def test_a_dataset_without_references_uses_the_global_ones():
    assert (
        config_constants.get_dataset_atomic_energies_hartree("QM9")
        is config_constants.get_atomic_energies_hartree()
    )


@pytest.mark.parametrize("invalid", [{}, [1], {"H": -0.5}, {1: "x"}, {True: -0.5}, {1: True}])
def test_invalid_dataset_references_raise(monkeypatch, invalid):
    def load_config(*args, **kwargs):
        return {"xxmd_config": {"atomic_energies_hartree": invalid}}

    monkeypatch.setattr("milia_pipeline.config.config_loader.load_config", load_config)
    with pytest.raises(ConfigurationError, match=r"xxmd_config\.atomic_energies_hartree"):
        config_constants.get_dataset_atomic_energies_hartree("XXMD")
