#!/usr/bin/env python3
"""
Unit + oracle tests for the ``matrix_spectral`` descriptor plugin
(descriptor programme, Pace 5 / v1.8.0).

Production-grade protocol (Blueprint §6, §9; industry best practice — spec/invariant + regression
guards, edge/error paths, one behaviour per parametrized node):
    1. Ground-truth — native output equals a FROZEN ``mordredcommunity`` snapshot to ``rtol=1e-4``,
       one node per (molecule, descriptor); + a live cross-check when mordred is importable.
    2. Determinism — identical across repeated calls (deterministic eigendecomposition).
    3. Robustness / error paths — invalid -> NaN; disconnected graph -> NaN (`require_connected`);
       degenerate inputs never raise.
    4. Contract / invariants — 177 unique registrations; ``name == function_name`` (BCUT names
       hyphen-sanitized); per-family composition; no collision with the HAVE baseline.
    5. De-dup guards — BCUT is NOT RDKit's ``BCUT2D`` (distinct Burden convention); ``DetourIndex``
       is not shipped here (it lives in ``walk_path_information``, Pace 4).
    6. Assertive boundary values — trace-based ``SM1`` identities.
"""

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

_PLUGIN_DIR = (
    Path(__file__).parent.parent / "milia_pipeline" / "plugins" / "descriptors" / "matrix_spectral"
)
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_matrix_spectral_oracle.json"

_spec = importlib.util.spec_from_file_location(
    "matrix_spectral_descriptors", _PLUGIN_DIR / "descriptors.py"
)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)

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


def _mord(name):
    """MILIA sanitized name -> Mordred canonical (BCUT hyphens)."""
    return name.replace("_1h", "-1h").replace("_1l", "-1l") if name.startswith("BCUT") else name


def _nan_eq(got, expected, rtol=_RTOL):
    if expected == "nan":
        return isinstance(got, float) and math.isnan(got)
    return math.isclose(got, float(expected), rel_tol=rtol, abs_tol=_ATOL)


# 1. frozen snapshot (HARD GATE) — one node per (molecule, descriptor)
@pytest.mark.parametrize("smiles,name,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(smiles, name, expected):
    got = getattr(ms, name)(_mol(smiles))
    assert _nan_eq(got, expected), f"{name}: got {got!r} expected {expected!r}"


# 1b. live mordred cross-check
@pytest.mark.parametrize("smiles", _MOL_CASES)
def test_live_mordred_oracle(smiles):
    pytest.importorskip("mordred")
    from mordred import Calculator, descriptors

    mord = {n: _mord(n) for n in _NAMES}
    calc = Calculator(descriptors, ignore_3D=True)
    calc.descriptors = [d for d in calc.descriptors if str(d) in set(mord.values())]
    mol = _mol(smiles)
    oracle = {str(k): v for k, v in zip(calc.descriptors, calc(mol), strict=False)}
    for name in _NAMES:
        raw = oracle[mord[name]]
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = float("nan")
        expected = "nan" if math.isnan(raw) else raw
        assert _nan_eq(getattr(ms, name)(mol), expected), f"{name}: live-oracle mismatch"


# 2. determinism
@pytest.mark.parametrize("name", _NAME_IDS)
def test_determinism(name):
    mol = _mol("CC(=O)Oc1ccccc1C(=O)O")
    first = getattr(ms, name)(mol)
    for _ in range(3):
        again = getattr(ms, name)(mol)
        if isinstance(first, float) and math.isnan(first):
            assert math.isnan(again)
        else:
            assert again == first, f"{name} non-deterministic"


# 3. robustness / error paths
@pytest.mark.parametrize("name", _NAME_IDS)
def test_none_returns_nan(name):
    val = getattr(ms, name)(None)
    assert isinstance(val, float) and math.isnan(val)


@pytest.mark.parametrize("name", _NAME_IDS)
def test_disconnected_returns_nan(name):
    """`require_connected`: every descriptor is NaN on a disconnected graph."""
    val = getattr(ms, name)(_mol("CCO.CCO"))
    assert isinstance(val, float) and math.isnan(val)


@pytest.mark.parametrize("smiles", ["C", "CC", "O=C=O", "c1ccccc1"])
def test_no_raise_on_small_inputs(smiles):
    mol = _mol(smiles)
    for name in _NAMES:
        assert isinstance(getattr(ms, name)(mol), float)


# 4. contract / invariants
def test_total_count():
    assert len(_DECLS) == 177
    assert len(set(ms.ALL_DESCRIPTOR_NAMES)) == 177


def test_family_composition():
    counts = Counter(d["block"] for d in _DECLS)
    assert counts == {
        "AdjacencyMatrix": 12,
        "DistanceMatrix": 12,
        "DetourMatrix": 13,
        "BaryszMatrix": 104,
        "BCUT": 24,
        "MolecularId": 12,
    }


def test_name_equals_function_name():
    for d in _DECLS:
        assert d["name"] == d["function_name"], d["name"]
        assert callable(getattr(ms, d["function_name"], None)), d["function_name"]
        assert d["requires_3d"] is False
        assert d["requires_charges"] is False


def test_bcut_names_are_sanitized_identifiers():
    """The 24 BCUT names are hyphen-sanitized to valid identifiers (single-registration invariant)."""
    bcut = [d["name"] for d in _DECLS if d["block"] == "BCUT"]
    assert len(bcut) == 24
    assert all("-" not in n and (n.endswith("_1h") or n.endswith("_1l")) for n in bcut)


def test_no_collision_with_have_baseline():
    have = {"BCUT2D_MWHI", "BCUT2D_CHGLO", "Ipc", "AvgIpc", "BalabanJ", "Chi0n", "BertzCT"}
    assert have.isdisjoint(set(ms.ALL_DESCRIPTOR_NAMES))


# 5. de-dup guards
def test_detour_index_not_shipped_here():
    """DetourIndex belongs to walk_path_information (Pace 4), not this plugin (Pace-4/5 split)."""
    assert "DetourIndex" not in set(ms.ALL_DESCRIPTOR_NAMES)
    assert not hasattr(ms, "DetourIndex")


def test_bcut_is_not_rdkit_bcut2d():
    """mordred BCUT uses a different Burden convention than RDKit BCUT2D — distinct values,
    so shipping mordred BCUT does not duplicate the HAVE RDKit BCUT2D baseline."""
    mol = _mol("CC(=O)Oc1ccccc1C(=O)O")
    rd = rdMolDescriptors.BCUT2D(mol)  # [MWHI, MWLOW, CHGHI, CHGLO, LOGPHI, LOGPLOW, MRHI, MRLOW]
    assert not math.isclose(ms.BCUTm_1h(mol), rd[0], rel_tol=1e-6)  # mass-hi vs MWHI
    assert not math.isclose(ms.BCUTc_1h(mol), rd[2], rel_tol=1e-6)  # charge-hi vs CHGHI


# 6. assertive boundary values (trace-based SM1 identities)
def test_sm1_detour_trace_is_zero():
    """SM1_Dt = trace of the detour matrix = 0 (zero diagonal) for any molecule."""
    for smiles in ("c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1"):
        assert abs(ms.SM1_Dt(_mol(smiles))) < 1e-9


def test_sm1_barysz_zero_for_all_carbon():
    """SM1_DzZ = Σ(1 − C/Z_i); for an all-carbon graph (benzene) every term is 0."""
    assert abs(ms.SM1_DzZ(_mol("c1ccccc1"))) < 1e-9


@pytest.mark.parametrize("name", _NAME_IDS)
def test_returns_float(name):
    assert isinstance(getattr(ms, name)(_mol("Cc1ccccc1")), float)
