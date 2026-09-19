#!/usr/bin/env python3
"""
Unit + oracle tests for the ``eta`` descriptor plugin
(descriptor programme, Pace 8 / v1.11.0).

Production-grade protocol (Blueprint §6, §9):
    1. Ground-truth — frozen mordredcommunity snapshot (mordred.ExtendedTopochemicalAtom),
       one node per (mol, descriptor); live cross-check via importorskip.
    2. Determinism; robustness (None -> NaN; degenerate never raise).
    3. Contract/invariants — 45 unique; name==function_name; block/category; name order ==
       oracle order; no collision with HAVE / prior paces.
    4. Assertive guards — disconnected & empty (0-atom) molecules -> NaN for all (oracle
       require_connected); averaged descriptor == plain / heavy-atom-count.
    5. Analytical boundary — every atom of the all-carbon reference graph has core-count 0.5,
       so ETA_alpha of an acyclic alkane equals 0.5 * (heavy-atom count).
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

_PLUGIN_DIR = Path(__file__).parent.parent / "milia_pipeline" / "plugins" / "descriptors" / "eta"
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_eta_oracle.json"

_spec = importlib.util.spec_from_file_location("eta_descriptors", _PLUGIN_DIR / "descriptors.py")
eta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eta)

_yaml = pytest.importorskip("yaml")
_DECLS = _yaml.safe_load(_YAML.read_text())["descriptors"]
_NAMES = [d["name"] for d in _DECLS]

_RTOL = 1e-6
_ATOL = 1e-9
_SNAPSHOT = json.loads(_ORACLE_JSON.read_text())
_PANEL = _SNAPSHOT["panel"]
_SNAP = _SNAPSHOT["values"]

_SNAPSHOT_CASES = [
    pytest.param(smiles, name, _SNAP[mol][name], id=f"{mol}-{name}")
    for mol, smiles in _PANEL.items()
    for name in _NAMES
]
_MOL_CASES = [pytest.param(s, id=m) for m, s in _PANEL.items()]
_NAME_IDS = [pytest.param(n, id=n) for n in _NAMES]


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def _eq(got, expected, rtol=_RTOL):
    if expected == "nan":
        return isinstance(got, float) and math.isnan(got)
    return math.isclose(got, float(expected), rel_tol=rtol, abs_tol=_ATOL)


# 1. frozen snapshot (HARD GATE)
@pytest.mark.parametrize("smiles,name,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(smiles, name, expected):
    assert _eq(getattr(eta, name)(_mol(smiles)), expected), name


# 1b. live mordred cross-check
@pytest.mark.parametrize("smiles", _MOL_CASES)
def test_live_mordred_oracle(smiles):
    pytest.importorskip("mordred")
    from mordred import Calculator
    from mordred import ExtendedTopochemicalAtom as _ME

    calc = Calculator(_ME)
    mol = _mol(smiles)
    oracle = {str(k): v for k, v in zip(calc.descriptors, calc(mol), strict=False)}
    for name in _NAMES:
        raw = oracle[name]
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = float("nan")
        expected = "nan" if math.isnan(raw) else raw
        assert _eq(getattr(eta, name)(mol), expected), name


# 2. determinism + robustness
@pytest.mark.parametrize("name", _NAME_IDS)
def test_determinism(name):
    mol = _mol("CC(=O)Oc1ccccc1C(=O)O")
    first = getattr(eta, name)(mol)
    for _ in range(3):
        again = getattr(eta, name)(mol)
        if isinstance(first, float) and math.isnan(first):
            assert math.isnan(again)
        else:
            assert again == first, f"{name} non-deterministic"


@pytest.mark.parametrize("name", _NAME_IDS)
def test_none_returns_nan(name):
    assert math.isnan(getattr(eta, name)(None))


@pytest.mark.parametrize("smiles", ["C", "CC", "[H][H]", "[Na+]", "O", "c1ccncc1", "CC#N"])
def test_no_raise_on_degenerate(smiles):
    mol = _mol(smiles)
    for name in _NAMES:
        assert isinstance(getattr(eta, name)(mol), float)


def test_disconnected_all_nan():
    """Oracle require_connected: multi-fragment molecule -> NaN for all."""
    mol = _mol("CCO.CCO")
    for name in _NAMES:
        assert math.isnan(getattr(eta, name)(mol)), name


def test_empty_molecule_all_nan():
    mol = _mol("")
    for name in _NAMES:
        assert math.isnan(getattr(eta, name)(mol)), name


# 3. contract / invariants
def test_total_count_and_order():
    assert len(_DECLS) == 45
    assert len(set(eta.ALL_DESCRIPTOR_NAMES)) == 45
    assert list(eta.ALL_DESCRIPTOR_NAMES) == _NAMES  # == oracle order


def test_name_equals_function_name():
    for d in _DECLS:
        assert d["name"] == d["function_name"], d["name"]
        assert callable(getattr(eta, d["function_name"], None))
        assert d["category"] == "topological"
        assert d["block"] == "ExtendedTopochemicalAtom"
        assert d["requires_3d"] is False
        assert d["requires_charges"] is False


def test_no_collision_with_have_or_prior_paces():
    names = set(eta.ALL_DESCRIPTOR_NAMES)
    assert all(n.startswith("ETA_") or n.startswith("AETA_") for n in names)
    # distinct from the Pace-7 families and BCUT (matrix_spectral)
    assert not any(n.startswith(("CATS_", "BCUT")) for n in names)


# 4. assertive guards
@pytest.mark.parametrize("smiles", ["CCCC", "CC(C)C", "CCO", "Cc1ccccc1"])
def test_averaged_equals_plain_over_heavy_count(smiles):
    mol = _mol(smiles)
    heavy = Chem.RemoveHs(mol).GetNumAtoms()
    for plain, avg in (
        ("ETA_alpha", "AETA_alpha"),
        ("ETA_beta", "AETA_beta"),
        ("ETA_dBeta", "AETA_dBeta"),
    ):
        p, a = getattr(eta, plain)(mol), getattr(eta, avg)(mol)
        assert math.isclose(a, p / heavy, rel_tol=1e-9), (smiles, plain)


# 5. analytical boundary value
@pytest.mark.parametrize(
    "smiles,heavy", [("C", 1), ("CC", 2), ("CCC", 3), ("CCCC", 4), ("CCCCCC", 6)]
)
def test_alkane_alpha_is_half_heavy_count(smiles, heavy):
    """All-carbon acyclic: every atom core-count = (6-4)/(4*(2-1)) = 0.5."""
    assert math.isclose(eta.ETA_alpha(_mol(smiles)), 0.5 * heavy, rel_tol=1e-9)
