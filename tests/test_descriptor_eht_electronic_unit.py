#!/usr/bin/env python3
"""
Unit + oracle tests for the ``eht_electronic`` descriptor plugin
(descriptor programme, Pace 10 / v1.13.0).

Validation protocol (Blueprint §6, §9; Pace-9 frozen-conformer pattern):
    EHT values are conformer- and RDKit-version-sensitive (``rdEHTTools`` is flagged
    experimental), so the frozen snapshot stores the EXACT conformer (MolBlock topology +
    full-precision coordinates) + the 25 EHT values (rdkit==2025.3.6). The test RECONSTRUCTS
    that conformer (no re-embedding) and checks MILIA against the frozen values -> bit-exact,
    version-pinned. A live re-derivation cross-check validates the conceptual-DFT formulas
    from the raw ``rdEHTTools`` primitives independently.

Also: determinism, robustness (None / missing-conformer -> NaN), contract (25 unique;
name==function_name; category electronic; requires_3d), and analytical identities
(IP=-HOMO, EA=-LUMO, gap=LUMO-HOMO, chi=(IP+EA)/2, eta=gap/2, mu=-chi).
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

_PLUGIN_DIR = (
    Path(__file__).parent.parent / "milia_pipeline" / "plugins" / "descriptors" / "eht_electronic"
)
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_eht_electronic_oracle.json"

_spec = importlib.util.spec_from_file_location("eht_descriptors", _PLUGIN_DIR / "descriptors.py")
eht = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eht)

_yaml = pytest.importorskip("yaml")
_DECLS = _yaml.safe_load(_YAML.read_text())["descriptors"]
_NAMES = [d["name"] for d in _DECLS]

_RTOL = 1e-6
_ATOL = 1e-9
_SNAP = json.loads(_ORACLE_JSON.read_text())
_PANEL = list(_SNAP["values"].keys())


def _reconstruct(name):
    mol = Chem.MolFromMolBlock(_SNAP["molblocks"][name], removeHs=False)
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(_SNAP["coords"][name]):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return mol


def _eq(got, expected, rtol=_RTOL):
    if expected == "nan":
        return isinstance(got, float) and math.isnan(got)
    return math.isclose(got, float(expected), rel_tol=rtol, abs_tol=_ATOL)


_SNAPSHOT_CASES = [
    pytest.param(mol, name, _SNAP["values"][mol][name], id=f"{mol}-{name}")
    for mol in _PANEL
    for name in _NAMES
]
_NAME_IDS = [pytest.param(n, id=n) for n in _NAMES]
_MOL_IDS = [pytest.param(m, id=m) for m in _PANEL]


# 1. frozen-conformer snapshot (HARD GATE, version-pinned)
@pytest.mark.parametrize("mol,name,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(mol, name, expected):
    assert _eq(getattr(eht, name)(_reconstruct(mol)), expected), name


# 1b. live re-derivation of conceptual-DFT formulas from raw rdEHTTools primitives
@pytest.mark.parametrize("mol", _MOL_IDS)
def test_conceptual_dft_formulas(mol):
    import numpy as np
    from rdkit.Chem import rdEHTTools

    m = _reconstruct(mol)
    ok, res = rdEHTTools.RunMol(m)
    assert ok
    oe = np.array(res.GetOrbitalEnergies())
    nocc = res.numElectrons // 2
    homo, lumo = float(oe[nocc - 1]), float(oe[nocc])
    ip, ea = -homo, -lumo
    chi, eta = (ip + ea) / 2.0, (ip - ea) / 2.0
    assert _eq(eht.E_HOMO(m), homo)
    assert _eq(eht.IonizationPotential(m), ip)
    assert _eq(eht.ElectronAffinity(m), ea)
    assert _eq(eht.Electronegativity(m), chi)
    assert _eq(eht.ChemicalHardness(m), eta)
    assert _eq(eht.ElectrophilicityIndex(m), chi * chi / (2 * eta))


# 2. determinism + robustness
@pytest.mark.parametrize("name", _NAME_IDS)
def test_determinism(name):
    mol = _reconstruct(_PANEL[0])
    first = getattr(eht, name)(mol)
    for _ in range(3):
        again = getattr(eht, name)(mol)
        if isinstance(first, float) and math.isnan(first):
            assert math.isnan(again)
        else:
            assert again == first, f"{name} non-deterministic"


@pytest.mark.parametrize("name", _NAME_IDS)
def test_none_returns_nan(name):
    assert math.isnan(getattr(eht, name)(None))


@pytest.mark.parametrize("name", _NAME_IDS)
def test_missing_conformer_returns_nan(name):
    """EHT needs a conformer; a flat molecule -> NaN, no raise."""
    assert math.isnan(getattr(eht, name)(Chem.MolFromSmiles("CCO")))


# 3. contract / invariants
def test_total_count():
    assert len(_DECLS) == 25
    assert len(set(eht.ALL_DESCRIPTOR_NAMES)) == 25


def test_name_equals_function_name():
    for d in _DECLS:
        assert d["name"] == d["function_name"], d["name"]
        assert callable(getattr(eht, d["function_name"], None))
        assert d["category"] == "electronic"
        assert d["block"] == "EHT"
        assert d["requires_3d"] is True


# 4. analytical identities (Koopmans + conceptual DFT)
@pytest.mark.parametrize("mol", _MOL_IDS)
def test_identities(mol):
    m = _reconstruct(mol)
    homo, lumo = eht.E_HOMO(m), eht.E_LUMO(m)
    assert math.isclose(eht.HOMO_LUMO_Gap(m), lumo - homo, rel_tol=1e-9)
    assert math.isclose(eht.IonizationPotential(m), -homo, rel_tol=1e-9)
    assert math.isclose(eht.ElectronAffinity(m), -lumo, rel_tol=1e-9)
    assert math.isclose(eht.ChemicalHardness(m), eht.HOMO_LUMO_Gap(m) / 2.0, rel_tol=1e-9)
    assert math.isclose(eht.ChemicalPotential(m), -eht.Electronegativity(m), rel_tol=1e-9)
    assert math.isclose(
        eht.Electronegativity(m),
        (eht.IonizationPotential(m) + eht.ElectronAffinity(m)) / 2.0,
        rel_tol=1e-9,
    )
