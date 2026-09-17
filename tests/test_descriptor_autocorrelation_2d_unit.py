#!/usr/bin/env python3
"""
Unit + oracle tests for the ``autocorrelation_2d`` descriptor plugin
(descriptor programme, Pace 6 / v1.9.0).

Production-grade protocol (Blueprint §6, §9):
    1. Ground-truth — frozen mordredcommunity snapshot, one node per (mol, descriptor).
       Live cross-check via importorskip.
    2. Determinism; robustness (None -> NaN, degenerate never raise).
    3. Contract/invariants — 606 unique; name==function_name; per-family composition;
       no HAVE/addcore_3d collision.
    4. Assertive guards — AATS0 denominator (atoms not half-diagonal); MATS/GATS lag=0
       returns NaN (no lag-0 formula); de-dup (mordred ATS != RDKit AUTOCORR2D).
    5. Analytical boundary values — ATS0 = sum(w^2) (verified on small graph).
"""

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

_PLUGIN_DIR = (
    Path(__file__).parent.parent
    / "milia_pipeline"
    / "plugins"
    / "descriptors"
    / "autocorrelation_2d"
)
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_autocorrelation_2d_oracle.json"

_spec = importlib.util.spec_from_file_location(
    "autocorrelation_2d_descriptors", _PLUGIN_DIR / "descriptors.py"
)
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

_yaml = pytest.importorskip("yaml")
_DECLS = _yaml.safe_load(_YAML.read_text())["descriptors"]
_NAMES = [d["name"] for d in _DECLS]

_RTOL = 1e-4
_ATOL = 1e-9
_SNAPSHOT = json.loads(_ORACLE_JSON.read_text())
_PANEL = _SNAPSHOT["panel"]
_SNAP = _SNAPSHOT["values"]

_SNAPSHOT_CASES = [
    pytest.param(smiles, name, _SNAP[mol][name], id=f"{mol}-{name}")
    for mol, smiles in _PANEL.items()
    for name in _NAMES
]
_MOL_CASES = [pytest.param(smiles, id=mol) for mol, smiles in _PANEL.items()]
_NAME_IDS = [pytest.param(n, id=n) for n in _NAMES]


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def _nan_eq(got, expected, rtol=_RTOL):
    if expected == "nan":
        return isinstance(got, float) and math.isnan(got)
    return math.isclose(got, float(expected), rel_tol=rtol, abs_tol=_ATOL)


# 1. frozen snapshot (HARD GATE)
@pytest.mark.parametrize("smiles,name,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(smiles, name, expected):
    got = getattr(ac, name)(_mol(smiles))
    assert _nan_eq(got, expected), f"{name}: got {got!r} expected {expected!r}"


# 1b. live mordred cross-check
@pytest.mark.parametrize("smiles", _MOL_CASES)
def test_live_mordred_oracle(smiles):
    pytest.importorskip("mordred")
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    calc.descriptors = [d for d in calc.descriptors if str(d) in set(_NAMES)]
    mol = _mol(smiles)
    oracle = {str(k): v for k, v in zip(calc.descriptors, calc(mol), strict=False)}
    for name in _NAMES:
        raw = oracle[name]
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = float("nan")
        expected = "nan" if math.isnan(raw) else raw
        assert _nan_eq(getattr(ac, name)(mol), expected), f"{name}: live-oracle mismatch"


# 2. determinism + robustness
@pytest.mark.parametrize("name", _NAME_IDS)
def test_determinism(name):
    mol = _mol("CC(=O)Oc1ccccc1C(=O)O")
    first = getattr(ac, name)(mol)
    for _ in range(3):
        again = getattr(ac, name)(mol)
        if isinstance(first, float) and math.isnan(first):
            assert math.isnan(again)
        else:
            assert again == first, f"{name} non-deterministic"


@pytest.mark.parametrize("name", _NAME_IDS)
def test_none_returns_nan(name):
    val = getattr(ac, name)(None)
    assert isinstance(val, float) and math.isnan(val)


@pytest.mark.parametrize("smiles", ["C", "CC", "[H][H]"])
def test_no_raise_on_degenerate(smiles):
    mol = _mol(smiles)
    for name in _NAMES:
        assert isinstance(getattr(ac, name)(mol), float)


# 3. contract / invariants
def test_total_count():
    assert len(_DECLS) == 606
    assert len(set(ac.ALL_DESCRIPTOR_NAMES)) == 606


def test_family_composition():
    counts = Counter(d["block"] for d in _DECLS)
    assert counts == {"BrotoMoreau": 414, "MoranGeary": 192}


def test_name_equals_function_name():
    for d in _DECLS:
        assert d["name"] == d["function_name"], d["name"]
        assert callable(getattr(ac, d["function_name"], None))
        assert d["requires_3d"] is False
        assert d["requires_charges"] is False


def test_no_collision_with_have_or_addcore():
    """mordred ATS != AUTOCORR3D_* (addcore_3d ships 3D, we ship 2D); no name collision."""
    autocorr3d = {f"AUTOCORR3D_{i}" for i in range(1, 81)}
    autocorr2d_rdkit = {f"AUTOCORR2D_{i}" for i in range(1, 193)}
    assert autocorr3d.isdisjoint(set(ac.ALL_DESCRIPTOR_NAMES))
    assert autocorr2d_rdkit.isdisjoint(set(ac.ALL_DESCRIPTOR_NAMES))


# 4. assertive guards
def test_mats_gats_lag0_returns_nan():
    """MATS and GATS have no lag=0 formula (Moran/Geary are correlation functions
    between pairs at lag k>=1); the engine returns NaN for lag 0."""
    mol = _mol("c1ccccc1")
    # call the internal engine directly — lag=0 is not exposed as a named descriptor
    # but the engine must handle it gracefully
    assert math.isnan(ac._autocorr(mol, "MATS", 0, "c"))
    assert math.isnan(ac._autocorr(mol, "GATS", 0, "c"))


def test_aats0_denominator_is_atom_count():
    """AATS0 = ATS0 / n_atoms (self-pairs Delta_0 = n, NOT 0.5*n).
    Regression guard for the 2× normalization bug caught during the build."""
    mol = _mol("CCO")
    mh = Chem.AddHs(mol)
    n = mh.GetNumAtoms()
    ats0_m = ac.ATS0m(mol)
    aats0_m = ac.AATS0m(mol)
    assert math.isclose(aats0_m, ats0_m / n, rel_tol=1e-9)


def test_ats_not_rdkit_autocorr2d():
    """mordred ATS and RDKit AUTOCORR2D are distinct descriptor sets (different
    weighting and normalization); shipping ATS does NOT duplicate the RDKit baseline.
    Verified structurally: ATS names (ATS0Z, MATS1c …) != AUTOCORR2D_N names."""
    autocorr2d_names = {f"AUTOCORR2D_{i}" for i in range(1, 193)}
    assert autocorr2d_names.isdisjoint(set(ac.ALL_DESCRIPTOR_NAMES))


# 5. analytical boundary values
@pytest.mark.parametrize(
    "smiles,name,expected_fn",
    [
        pytest.param("CC", "ATS0Z", lambda: 6.0**2 + 6.0**2 + 1.0**2 * 6, id="ethane-ATS0Z"),
        pytest.param("C", "ATS0Z", lambda: 6.0**2 + 1.0**2 * 4, id="methane-ATS0Z"),
    ],
)
def test_ats0_equals_sum_of_squared_props(smiles, name, expected_fn):
    """ATS0 = sum(w_i^2) over H-included atoms (Moreau-Broto formula at lag=0)."""
    assert math.isclose(getattr(ac, name)(_mol(smiles)), expected_fn(), rel_tol=1e-9)


@pytest.mark.parametrize("name", _NAME_IDS)
def test_returns_float(name):
    assert isinstance(getattr(ac, name)(_mol("Cc1ccccc1")), float)
