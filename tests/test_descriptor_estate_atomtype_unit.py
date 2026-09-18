#!/usr/bin/env python3
"""
Unit + oracle tests for the ``estate_atomtype`` descriptor plugin
(descriptor programme, Pace 7 / v1.10.0).

Production-grade protocol (Blueprint §6, §9):
    1. Ground-truth — frozen mordredcommunity snapshot, one node per (mol, descriptor);
       live cross-check via importorskip.
    2. Determinism; robustness (None -> NaN, degenerate never raise).
    3. Contract/invariants — 316 unique; name==function_name; 79 types x {N,S,MAX,MIN};
       no HAVE collision (aggregate {Max,Min}EStateIndex are distinct names).
    4. Assertive guards — absent atom type -> N=0/S=0/MAX=NaN/MIN=NaN; empty (0-atom)
       molecule -> NaN for all (MILIA purity contract; oracle is internally inconsistent
       there); S<type> == sum of per-atom E-State over that type.
    5. Analytical boundary values — benzene has exactly 6 ``aaCH`` atoms.
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
    Path(__file__).parent.parent / "milia_pipeline" / "plugins" / "descriptors" / "estate_atomtype"
)
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_estate_atomtype_oracle.json"

_spec = importlib.util.spec_from_file_location(
    "estate_atomtype_descriptors", _PLUGIN_DIR / "descriptors.py"
)
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)

_yaml = pytest.importorskip("yaml")
_DECLS = _yaml.safe_load(_YAML.read_text())["descriptors"]
_NAMES = [d["name"] for d in _DECLS]

_RTOL = 1e-4
_ATOL = 1e-9
_SNAPSHOT = json.loads(_ORACLE_JSON.read_text())
_PANEL = _SNAPSHOT["panel"]
_SNAP = _SNAPSHOT["values"]

# The empty (0-atom) molecule is treated as invalid -> NaN by MILIA; the mordred oracle
# is internally inconsistent there (NaN for counts, 0 for sums), so it is excluded from
# exact snapshot comparison and covered by a dedicated all-NaN test instead.
_SNAPSHOT_CASES = [
    pytest.param(smiles, name, _SNAP[mol][name], id=f"{mol}-{name}")
    for mol, smiles in _PANEL.items()
    if mol != "empty"
    for name in _NAMES
]
_MOL_CASES = [pytest.param(smiles, id=mol) for mol, smiles in _PANEL.items() if mol != "empty"]
_NAME_IDS = [pytest.param(n, id=n) for n in _NAMES]

_ES_TYPES = list(es._ES_TYPES)
_AGGR = ("N", "S", "MAX", "MIN")


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def _nan_eq(got, expected, rtol=_RTOL):
    if expected == "nan":
        return isinstance(got, float) and math.isnan(got)
    return math.isclose(got, float(expected), rel_tol=rtol, abs_tol=_ATOL)


# 1. frozen snapshot (HARD GATE)
@pytest.mark.parametrize("smiles,name,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(smiles, name, expected):
    got = getattr(es, name)(_mol(smiles))
    assert _nan_eq(got, expected), f"{name}: got {got!r} expected {expected!r}"


# 1b. live mordred cross-check
@pytest.mark.parametrize("smiles", _MOL_CASES)
def test_live_mordred_oracle(smiles):
    pytest.importorskip("mordred")
    from mordred import Calculator
    from mordred import EState as _ME

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
        assert _nan_eq(getattr(es, name)(mol), expected), f"{name}: live-oracle mismatch"


# 2. determinism + robustness
@pytest.mark.parametrize("name", _NAME_IDS)
def test_determinism(name):
    mol = _mol("CC(=O)Oc1ccccc1C(=O)O")
    first = getattr(es, name)(mol)
    for _ in range(3):
        again = getattr(es, name)(mol)
        if isinstance(first, float) and math.isnan(first):
            assert math.isnan(again)
        else:
            assert again == first, f"{name} non-deterministic"


@pytest.mark.parametrize("name", _NAME_IDS)
def test_none_returns_nan(name):
    val = getattr(es, name)(None)
    assert isinstance(val, float) and math.isnan(val)


@pytest.mark.parametrize("smiles", ["C", "CC", "[H][H]", "[Na+]", "O"])
def test_no_raise_on_degenerate(smiles):
    mol = _mol(smiles)
    for name in _NAMES:
        assert isinstance(getattr(es, name)(mol), float)


def test_empty_molecule_all_nan():
    """0-atom molecule is invalid -> NaN for every aggregation (MILIA purity contract)."""
    mol = _mol("")
    for name in _NAMES:
        val = getattr(es, name)(mol)
        assert isinstance(val, float) and math.isnan(val), name


# 3. contract / invariants
def test_total_count():
    assert len(_DECLS) == 316
    assert len(set(es.ALL_DESCRIPTOR_NAMES)) == 316
    assert len(_ES_TYPES) == 79


def test_family_composition():
    counts = Counter(d["block"] for d in _DECLS)
    assert counts == {"AtomTypeEState": 316}
    prefixes = Counter()
    for d in _DECLS:
        for pfx in _AGGR:
            if d["name"].startswith(pfx):
                prefixes[pfx] += 1
                break
    assert prefixes == {"N": 79, "S": 79, "MAX": 79, "MIN": 79}


def test_name_equals_function_name():
    for d in _DECLS:
        assert d["name"] == d["function_name"], d["name"]
        assert callable(getattr(es, d["function_name"], None))
        assert d["category"] == "electronic"
        assert d["requires_3d"] is False
        assert d["requires_charges"] is False


def test_no_collision_with_have():
    """Atom-type E-State names are distinct from the HAVE aggregate E-State indices."""
    have_estate = {
        "MaxEStateIndex",
        "MinEStateIndex",
        "MaxAbsEStateIndex",
        "MinAbsEStateIndex",
    }
    assert have_estate.isdisjoint(set(es.ALL_DESCRIPTOR_NAMES))


# 4. assertive guards
def test_absent_type_convention():
    """A type with no atoms -> N=0, S=0, MAX=NaN, MIN=NaN (matches oracle)."""
    mol = _mol("c1ccccc1")  # benzene: no sCH3 atoms
    assert es.NsCH3(mol) == 0.0
    assert es.SsCH3(mol) == 0.0
    assert math.isnan(es.MAXsCH3(mol))
    assert math.isnan(es.MINsCH3(mol))


def test_sum_equals_atom_estate_sum():
    """S<type> == sum of per-atom E-State values over atoms of that type."""
    from rdkit.Chem.EState import AtomTypes, EStateIndices

    mol = _mol("CCO")
    labels = AtomTypes.TypeAtoms(mol)
    vals = EStateIndices(mol)
    manual = {}
    for lab_tuple, v in zip(labels, vals, strict=False):
        for lab in lab_tuple:
            manual.setdefault(lab, []).append(v)
    for lab, vs in manual.items():
        assert math.isclose(getattr(es, "S" + lab)(mol), sum(vs), rel_tol=1e-9)
        assert getattr(es, "N" + lab)(mol) == float(len(vs))


# 5. analytical boundary values
def test_benzene_has_six_aromatic_ch():
    assert es.NaaCH(_mol("c1ccccc1")) == 6.0


@pytest.mark.parametrize("name", _NAME_IDS)
def test_returns_float(name):
    assert isinstance(getattr(es, name)(_mol("Cc1ccccc1")), float)
