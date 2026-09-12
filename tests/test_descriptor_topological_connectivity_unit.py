#!/usr/bin/env python3
"""
Unit + oracle tests for the ``topological_connectivity`` descriptor plugin
(descriptor programme, Pace 3 / v1.6.0).

Validation-oracle protocol (Blueprint §6, §9):
    1. Ground-truth correctness — native output equals a FROZEN oracle snapshot
       (``mordredcommunity`` for the 104 Mordred-defined descriptors; native impl for the
       3 Schultz indices) to ``rtol=1e-4``, NaN-equality for invalids; one parametrized
       node per (molecule, descriptor). A live ``mordredcommunity`` cross-check runs
       additionally when the package is importable (``importorskip``).
    2. Schultz published values — MTI/SchultzIndex/ModSchultzIndex against hand-computed
       reference values (benzene, propane) + the identity ``MTI = Zagreb1 + 2*SchultzIndex``.
    3. Determinism, robustness (invalid / degenerate -> NaN, never raise).
    4. Contract — 107 unique registrations; ``plugin.yaml`` reconciles with the module;
       function_name resolves; no collision with the HAVE baseline.

The plugin module is loaded directly from its file path (no torch/package import).
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

_PLUGIN_DIR = (
    Path(__file__).parent.parent
    / "milia_pipeline"
    / "plugins"
    / "descriptors"
    / "topological_connectivity"
)
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_topological_connectivity_oracle.json"

_spec = importlib.util.spec_from_file_location(
    "topological_connectivity_descriptors", _PLUGIN_DIR / "descriptors.py"
)
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

_yaml = pytest.importorskip("yaml")
_DECLS = _yaml.safe_load(_YAML.read_text())["descriptors"]
_FN_OF = {d["name"]: d["function_name"] for d in _DECLS}
_SCHULTZ = ("MTI", "SchultzIndex", "ModSchultzIndex")
_ORACLE_NAMES = tuple(d["name"] for d in _DECLS if d["name"] not in _SCHULTZ)

_RTOL = 1e-4
_ATOL = 1e-9
_SNAPSHOT = json.loads(_ORACLE_JSON.read_text())
_PANEL = _SNAPSHOT["panel"]
_SNAP = _SNAPSHOT["values"]

_SNAPSHOT_CASES = [
    pytest.param(smiles, d["name"], _SNAP[mol][d["name"]], id=f"{mol}-{d['name']}")
    for mol, smiles in _PANEL.items()
    for d in _DECLS
]
_MOL_CASES = [pytest.param(smiles, id=mol) for mol, smiles in _PANEL.items()]
_NAME_IDS = [pytest.param(d["name"], id=d["name"]) for d in _DECLS]


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def _call(name, mol):
    return getattr(tc, _FN_OF[name])(mol)


def _nan_eq(got, expected, rtol=_RTOL):
    if expected == "nan":
        return isinstance(got, float) and math.isnan(got)
    return math.isclose(got, float(expected), rel_tol=rtol, abs_tol=_ATOL)


# 1. frozen snapshot (HARD GATE) — one node per (molecule, descriptor)
@pytest.mark.parametrize("smiles,descriptor,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(smiles, descriptor, expected):
    got = _call(descriptor, _mol(smiles))
    assert _nan_eq(got, expected), f"{descriptor}: got {got!r} expected {expected!r}"


# 1b. live mordred cross-check (skips if mordredcommunity absent)
@pytest.mark.parametrize("smiles", _MOL_CASES)
def test_live_mordred_oracle(smiles):
    pytest.importorskip("mordred")
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    mord = {n: n.replace("_", "-") for n in _ORACLE_NAMES}
    calc.descriptors = [d for d in calc.descriptors if str(d) in set(mord.values())]
    mol = _mol(smiles)
    oracle = {str(k): v for k, v in zip(calc.descriptors, calc(mol), strict=False)}
    for name in _ORACLE_NAMES:
        raw = oracle[mord[name]]
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = float("nan")
        expected = "nan" if math.isnan(raw) else raw
        assert _nan_eq(_call(name, mol), expected), f"{name}: live-oracle mismatch"


# 2. Schultz published reference values + identity
def test_schultz_published_values():
    bz = _mol("c1ccccc1")
    assert math.isclose(tc.MTI(bz), 132.0, abs_tol=1e-9)
    assert math.isclose(tc.SchultzIndex(bz), 54.0, abs_tol=1e-9)
    assert math.isclose(tc.ModSchultzIndex(bz), 54.0, abs_tol=1e-9)
    pr = _mol("CCC")
    assert math.isclose(tc.SchultzIndex(pr), 5.0, abs_tol=1e-9)
    assert math.isclose(tc.ModSchultzIndex(pr), 3.0, abs_tol=1e-9)


@pytest.mark.parametrize("smiles", _MOL_CASES)
def test_schultz_identity(smiles):
    mol = _mol(smiles)
    assert math.isclose(tc.MTI(mol), tc.Zagreb1(mol) + 2 * tc.SchultzIndex(mol), rel_tol=1e-9)


# 3. determinism + robustness
@pytest.mark.parametrize("descriptor", _NAME_IDS)
def test_determinism(descriptor):
    mol = _mol("CC(=O)Oc1ccccc1C(=O)O")
    first = _call(descriptor, mol)
    for _ in range(4):
        again = _call(descriptor, mol)
        if isinstance(first, float) and math.isnan(first):
            assert math.isnan(again)
        else:
            assert again == first, f"{descriptor} non-deterministic"


@pytest.mark.parametrize("descriptor", _NAME_IDS)
def test_none_returns_nan(descriptor):
    val = _call(descriptor, None)
    assert isinstance(val, float) and math.isnan(val)


@pytest.mark.parametrize("smiles", ["C", "O", "[H][H]", "CC"])
def test_no_raise_on_degenerate(smiles):
    mol = _mol(smiles)
    for d in _DECLS:
        assert isinstance(_call(d["name"], mol), float)


# 4. contract
def test_total_count():
    assert len(_DECLS) == 107
    assert len(tc.ALL_DESCRIPTOR_NAMES) == 107
    assert len(set(tc.ALL_DESCRIPTOR_NAMES)) == 107


def test_yaml_reconciles_with_module():
    assert {d["name"] for d in _DECLS} == set(tc.ALL_DESCRIPTOR_NAMES)
    for d in _DECLS:
        assert callable(getattr(tc, d["function_name"], None)), d["function_name"]
        assert d["category"] == "topological"
        assert d["block"]
        assert d["requires_3d"] is False
        assert d["requires_charges"] is False


def test_no_collision_with_have_baseline():
    have = {"Chi0n", "Chi1n", "Chi2v", "Kappa1", "BalabanJ", "BertzCT", "Ipc", "HallKierAlpha"}
    assert have.isdisjoint(set(tc.ALL_DESCRIPTOR_NAMES))


def test_dedup_drops_have_valence_path_chi():
    """Scope invariant: the 5 valence-path low orders that equal HAVE RDKit Chi0v..Chi4v are NOT
    shipped, while their simple-path counterparts (different normalisation) and the averaged
    valence-path variants ARE."""
    names = set(tc.ALL_DESCRIPTOR_NAMES)
    for o in range(5):
        assert f"Xp_{o}dv" not in names, f"Xp_{o}dv duplicates HAVE Chi{o}v and must be de-duped"
        assert f"Xp_{o}d" in names, f"Xp_{o}d (distinct from HAVE Chi{o}n) must be shipped"
        assert f"AXp_{o}dv" in names, f"AXp_{o}dv (averaged, distinct) must be shipped"


def test_family_block_composition():
    """Spec invariant: exact per-block counts (guards against family-count regressions that a bare
    total of 107 would not catch)."""
    from collections import Counter

    counts = Counter(d["block"] for d in _DECLS)
    assert counts == {
        "WienerIndex": 2,
        "ZagrebIndex": 4,
        "ABCIndex": 2,
        "EccentricConnectivity": 1,
        "TopologicalIndex": 4,
        "Schultz": 3,
        "TopologicalCharge": 21,
        "MolecularDistanceEdge": 19,
        "Chi": 51,
    }


def test_function_name_is_sanitized_canonical_name():
    """Naming contract: function_name is the hyphen-sanitized canonical Mordred name."""
    for d in _DECLS:
        assert d["function_name"] == d["name"].replace("-", "_"), d["name"]


@pytest.mark.parametrize(
    "smiles,expected",
    [
        pytest.param(
            "C", {"WPath": 0.0, "Radius": 0.0, "Diameter": 0.0, "Zagreb1": 0.0}, id="methane"
        ),
        pytest.param(
            "CC", {"WPath": 1.0, "Radius": 1.0, "Diameter": 1.0, "Zagreb1": 2.0}, id="ethane"
        ),
        pytest.param(
            "CCC", {"WPath": 4.0, "Radius": 1.0, "Diameter": 2.0, "Zagreb1": 6.0}, id="propane"
        ),
    ],
)
def test_boundary_values(smiles, expected):
    """Exact known values on small graphs (boundary conditions, not just 'returns a float')."""
    mol = _mol(smiles)
    for name, exp in expected.items():
        assert math.isclose(_call(name, mol), exp, abs_tol=1e-9), f"{name}({smiles})"


@pytest.mark.parametrize("descriptor", _NAME_IDS)
def test_returns_float(descriptor):
    assert isinstance(_call(descriptor, _mol("Cc1ccccc1")), float)
