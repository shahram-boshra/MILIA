#!/usr/bin/env python3
"""
Unit + oracle tests for the ``cats2d`` descriptor plugin
(descriptor programme, Pace 7 / v1.10.0).

Production-grade protocol (Blueprint §6, §9):
    1. Ground-truth — frozen PyBioMed snapshot (scale=1, raw counts), one node per
       (mol, descriptor). PyBioMed is the peer-reviewed reference implementation (Dong
       et al. 2018); it requires a legacy scipy shim, so it is not a runtime/test
       dependency and the frozen snapshot is the authoritative gate.
    2. Determinism; robustness (None -> NaN, degenerate never raise).
    3. Contract/invariants — 150 unique; name==function_name; 15 pairs x 10 distances;
       no collision with HAVE / prior paces (USR/USRCAT/AUTOCORR*).
    4. Assertive guards — benzene is purely lipophilic (only LL* nonzero); a single
       carbon (methane) is an L self-pair (LL0=1); phenol O is both donor and acceptor
       (DA0 > 0); empty (0-atom) molecule -> NaN for all (MILIA purity contract).
    5. Analytical boundary values — benzene LL0 == 6 (6 carbons self-paired at dist 0).
"""

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

_PLUGIN_DIR = Path(__file__).parent.parent / "milia_pipeline" / "plugins" / "descriptors" / "cats2d"
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_cats2d_oracle.json"

_spec = importlib.util.spec_from_file_location("cats2d_descriptors", _PLUGIN_DIR / "descriptors.py")
cat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cat)

_yaml = pytest.importorskip("yaml")
_DECLS = _yaml.safe_load(_YAML.read_text())["descriptors"]
_NAMES = [d["name"] for d in _DECLS]

_ATOL = 1e-9
_SNAPSHOT = json.loads(_ORACLE_JSON.read_text())
_PANEL = _SNAPSHOT["panel"]
_SNAP = _SNAPSHOT["values"]

_SNAPSHOT_CASES = [
    pytest.param(smiles, name, _SNAP[mol][name], id=f"{mol}-{name}")
    for mol, smiles in _PANEL.items()
    if mol != "empty"
    for name in _NAMES
]
_NAME_IDS = [pytest.param(n, id=n) for n in _NAMES]


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def _eq(got, expected):
    if expected == "nan":
        return isinstance(got, float) and math.isnan(got)
    return math.isclose(got, float(expected), rel_tol=0.0, abs_tol=_ATOL)


# 1. frozen snapshot (HARD GATE)
@pytest.mark.parametrize("smiles,name,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(smiles, name, expected):
    got = getattr(cat, name)(_mol(smiles))
    assert _eq(got, expected), f"{name}: got {got!r} expected {expected!r}"


# 2. determinism + robustness
@pytest.mark.parametrize("name", _NAME_IDS)
def test_determinism(name):
    mol = _mol("CC(=O)Oc1ccccc1C(=O)O")
    first = getattr(cat, name)(mol)
    for _ in range(3):
        again = getattr(cat, name)(mol)
        if isinstance(first, float) and math.isnan(first):
            assert math.isnan(again)
        else:
            assert again == first, f"{name} non-deterministic"


@pytest.mark.parametrize("name", _NAME_IDS)
def test_none_returns_nan(name):
    val = getattr(cat, name)(None)
    assert isinstance(val, float) and math.isnan(val)


@pytest.mark.parametrize("smiles", ["C", "CC", "[H][H]", "[Na+]", "O", "Clc1ccccc1"])
def test_no_raise_on_degenerate(smiles):
    mol = _mol(smiles)
    for name in _NAMES:
        assert isinstance(getattr(cat, name)(mol), float)


def test_empty_molecule_all_nan():
    mol = _mol("")
    for name in _NAMES:
        val = getattr(cat, name)(mol)
        assert isinstance(val, float) and math.isnan(val), name


# 3. contract / invariants
def test_total_count():
    assert len(_DECLS) == 150
    assert len(set(cat.ALL_DESCRIPTOR_NAMES)) == 150


def test_family_composition():
    counts = Counter(d["block"] for d in _DECLS)
    assert counts == {"CATS2D": 150}
    pairs = {n[len("CATS_") : -1] for n in _NAMES}
    assert pairs == {
        "DD",
        "DA",
        "DP",
        "DN",
        "DL",
        "AA",
        "AP",
        "AN",
        "AL",
        "PP",
        "PN",
        "PL",
        "NN",
        "NL",
        "LL",
    }
    for p in pairs:
        assert sum(1 for n in _NAMES if n[len("CATS_") : -1] == p) == 10


def test_name_equals_function_name():
    for d in _DECLS:
        assert d["name"] == d["function_name"], d["name"]
        assert callable(getattr(cat, d["function_name"], None))
        assert d["category"] == "fragments"
        assert d["requires_3d"] is False
        assert d["requires_charges"] is False


def test_no_collision_with_have_or_prior_paces():
    names = set(cat.ALL_DESCRIPTOR_NAMES)
    usr = {f"USR_{i}" for i in range(1, 13)} | {f"USRCAT_{i}" for i in range(1, 61)}
    autocorr3d = {f"AUTOCORR3D_{i}" for i in range(1, 81)}
    assert usr.isdisjoint(names)
    assert autocorr3d.isdisjoint(names)
    assert all(n.startswith("CATS_") for n in names)


# 4. assertive guards
def test_benzene_is_purely_lipophilic():
    """Every benzene carbon is L (carbon bonded only to carbons); no D/A/P/N features."""
    mol = _mol("c1ccccc1")
    for name in _NAMES:
        if name.startswith("CATS_LL"):
            continue
        assert getattr(cat, name)(mol) == 0.0, name


def test_single_carbon_is_lipophilic_self_pair():
    """Methane's lone carbon is an isolated-carbon L -> one LL self-pair at distance 0."""
    assert cat.CATS_LL0(_mol("C")) == 1.0


def test_phenol_oxygen_is_donor_and_acceptor():
    """Phenol O matches both [OH] (donor) and [O] (acceptor) -> DA self-pair at dist 0."""
    assert cat.CATS_DA0(_mol("Oc1ccccc1")) > 0.0


# 5. analytical boundary values
def test_benzene_ll0_is_six():
    assert cat.CATS_LL0(_mol("c1ccccc1")) == 6.0


@pytest.mark.parametrize("name", _NAME_IDS)
def test_returns_float(name):
    assert isinstance(getattr(cat, name)(_mol("Cc1ccccc1")), float)
