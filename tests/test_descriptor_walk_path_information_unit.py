#!/usr/bin/env python3
"""
Unit + oracle tests for the ``walk_path_information`` descriptor plugin
(descriptor programme, Pace 4 / v1.7.0).

Protocol (Blueprint §6, §9): frozen oracle snapshot hard-gate (``mordredcommunity`` for all
86) with a live cross-check when installed; determinism; robustness (invalid -> NaN, never
raise); contract (86 unique registrations; ``name == function_name``; per-family
composition; boundary values). Module loaded by path (no torch/package import).
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
    / "walk_path_information"
)
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_walk_path_information_oracle.json"

_spec = importlib.util.spec_from_file_location(
    "walk_path_information_descriptors", _PLUGIN_DIR / "descriptors.py"
)
wpi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wpi)

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
    got = getattr(wpi, name)(_mol(smiles))
    assert _nan_eq(got, expected), f"{name}: got {got!r} expected {expected!r}"


# 1b. live mordred cross-check (skips if mordredcommunity absent)
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
        assert _nan_eq(getattr(wpi, name)(mol), expected), f"{name}: live-oracle mismatch"


# 2. determinism + robustness
@pytest.mark.parametrize("name", _NAME_IDS)
def test_determinism(name):
    mol = _mol("CC(=O)Oc1ccccc1C(=O)O")
    first = getattr(wpi, name)(mol)
    for _ in range(3):
        again = getattr(wpi, name)(mol)
        if isinstance(first, float) and math.isnan(first):
            assert math.isnan(again)
        else:
            assert again == first, f"{name} non-deterministic"


@pytest.mark.parametrize("name", _NAME_IDS)
def test_none_returns_nan(name):
    val = getattr(wpi, name)(None)
    assert isinstance(val, float) and math.isnan(val)


@pytest.mark.parametrize("smiles", ["C", "O", "[H][H]", "CC"])
def test_no_raise_on_degenerate(smiles):
    mol = _mol(smiles)
    for name in _NAMES:
        assert isinstance(getattr(wpi, name)(mol), float)


# 3. contract / invariants
def test_total_count():
    assert len(_DECLS) == 86
    assert len(set(wpi.ALL_DESCRIPTOR_NAMES)) == 86


def test_family_composition():
    counts = Counter(d["block"] for d in _DECLS)
    assert counts == {
        "WalkCount": 21,
        "PathCount": 21,
        "DetourIndex": 1,
        "InformationContent": 42,
        "VertexAdjacencyInformation": 1,
    }


def test_name_equals_function_name():
    for d in _DECLS:
        assert d["name"] == d["function_name"], d["name"]
        assert callable(getattr(wpi, d["function_name"], None)), d["function_name"]
        assert d["requires_3d"] is False
        assert d["requires_charges"] is False


def test_no_collision_with_have_baseline():
    have = {"Ipc", "AvgIpc", "Chi0n", "Kappa1", "BalabanJ", "BertzCT", "HallKierAlpha"}
    assert have.isdisjoint(set(wpi.ALL_DESCRIPTOR_NAMES))


@pytest.mark.parametrize(
    "smiles,expected",
    [
        pytest.param("CC", {"MWC01": 1.0, "MPC2": 0.0, "DetourIndex": 1}, id="ethane"),
        pytest.param("c1ccccc1", {"DetourIndex": 63, "SRW02": math.log(13)}, id="benzene"),
    ],
)
def test_boundary_values(smiles, expected):
    mol = _mol(smiles)
    for name, exp in expected.items():
        assert math.isclose(getattr(wpi, name)(mol), exp, rel_tol=1e-9, abs_tol=1e-9), name


def test_detour_step_budget_fails_soft(monkeypatch):
    """The NP-hard longest-simple-path DFS returns NaN (never hangs or raises) once the step
    budget is exceeded — exercises the fail-soft guard path, not just the happy path."""
    naphthalene = _mol("c1ccc2ccccc2c1")
    assert math.isfinite(wpi.DetourIndex(naphthalene))  # default budget -> finite (= 345)
    monkeypatch.setattr(wpi, "_DETOUR_STEP_BUDGET", 5)
    assert math.isnan(wpi.DetourIndex(naphthalene))  # tiny budget -> graceful NaN


def test_information_content_uses_h_included_graph():
    """Regression guard for the H-included convention (the bug caught during the build):
    IC0 of ethane is the entropy of the H-included atom set (2 C + 6 H); a heavy-only
    implementation would collapse to a single class and return 0.0."""
    n = 8  # C2H6
    expected = -(2 / n * math.log2(2 / n) + 6 / n * math.log2(6 / n))
    assert math.isclose(wpi.IC0(_mol("CC")), expected, abs_tol=1e-9)
    assert wpi.IC0(_mol("CC")) > 0.0  # heavy-only (both carbons identical) would be exactly 0.0


def test_pipc_accounts_for_bond_order():
    """piPC weights each path by the product of its bond orders: piPC1(benzene) =
    log(Σ aromatic bond orders + 1) = log(6·1.5 + 1) = log(10), not the unweighted log(7)."""
    assert math.isclose(wpi.piPC1(_mol("c1ccccc1")), math.log(10.0), abs_tol=1e-9)


@pytest.mark.parametrize("name", _NAME_IDS)
def test_returns_float(name):
    assert isinstance(getattr(wpi, name)(_mol("Cc1ccccc1")), float)
