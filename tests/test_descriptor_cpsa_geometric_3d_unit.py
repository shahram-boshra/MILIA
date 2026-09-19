#!/usr/bin/env python3
"""
Unit + oracle tests for the ``cpsa_geometric_3d`` descriptor plugin
(descriptor programme, Pace 9 / v1.12.0).

3D validation protocol (Blueprint §6, §9; addcore_3d precedent):
    3D descriptor values depend on the conformer, which is RDKit-version-sensitive. To stay
    reproducible without pinning the oracle at test time, the frozen snapshot stores, per
    molecule, the EXACT conformer (MolBlock topology + full-precision coordinates) plus the
    50 mordred values computed on it. The test RECONSTRUCTS that exact conformer (no
    re-embedding) and checks MILIA against the frozen values — verified bit-exact under both
    rdkit 2025.3.6 (repo pin) and 2026.03.6. A live mordred cross-check (importorskip)
    re-embeds fresh and compares self-consistently as defence-in-depth.

Also: determinism, robustness (None / missing-conformer / degenerate → NaN), contract
(50 unique; name==function_name; category geometric; requires_3d), de-dup guards (no TPSA /
PBF / MOMI), and analytical identities (DPSA == PPSA − PNSA).
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

_PLUGIN_DIR = (
    Path(__file__).parent.parent
    / "milia_pipeline"
    / "plugins"
    / "descriptors"
    / "cpsa_geometric_3d"
)
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_cpsa_geometric_3d_oracle.json"

_spec = importlib.util.spec_from_file_location("cg_descriptors", _PLUGIN_DIR / "descriptors.py")
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)

_yaml = pytest.importorskip("yaml")
_DECLS = _yaml.safe_load(_YAML.read_text())["descriptors"]
_NAMES = [d["name"] for d in _DECLS]

_RTOL = 1e-6
_ATOL = 1e-9
_SNAP = json.loads(_ORACLE_JSON.read_text())
_PANEL = list(_SNAP["values"].keys())


def _reconstruct(name):
    """Rebuild the EXACT frozen conformer: MolBlock topology + full-precision coordinates."""
    mol = Chem.MolFromMolBlock(_SNAP["molblocks"][name], removeHs=False)
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(_SNAP["coords"][name]):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return mol


def _embed(smiles, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
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


# 1. frozen-conformer snapshot (HARD GATE, version-robust)
@pytest.mark.parametrize("mol,name,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(mol, name, expected):
    assert _eq(getattr(cg, name)(_reconstruct(mol)), expected), name


# 1b. live mordred cross-check (defence-in-depth; re-embeds fresh, self-consistent)
@pytest.mark.parametrize("mol", [pytest.param(m, id=m) for m in _PANEL])
def test_live_mordred_oracle(mol):
    pytest.importorskip("mordred")
    from mordred import CPSA, Calculator, GeometricalIndex, GravitationalIndex

    smiles = Chem.MolToSmiles(Chem.MolFromMolBlock(_SNAP["molblocks"][mol], removeHs=True))
    m = _embed(smiles)
    if m is None:
        pytest.skip("embed failed")
    calc = Calculator([CPSA, GravitationalIndex, GeometricalIndex])
    oracle = {str(k): v for k, v in zip(calc.descriptors, calc(m), strict=False)}
    for name in _NAMES:
        raw = float(oracle[name])
        expected = "nan" if math.isnan(raw) else raw
        assert _eq(getattr(cg, name)(m), expected), name


# 2. determinism + robustness
@pytest.mark.parametrize("name", _NAME_IDS)
def test_determinism(name):
    mol = _reconstruct(_PANEL[0])
    first = getattr(cg, name)(mol)
    for _ in range(3):
        again = getattr(cg, name)(mol)
        if isinstance(first, float) and math.isnan(first):
            assert math.isnan(again)
        else:
            assert again == first, f"{name} non-deterministic"


@pytest.mark.parametrize("name", _NAME_IDS)
def test_none_returns_nan(name):
    assert math.isnan(getattr(cg, name)(None))


@pytest.mark.parametrize("name", _NAME_IDS)
def test_missing_conformer_returns_nan(name):
    """3D descriptors on a conformer-free (flat) molecule → NaN, no raise."""
    flat = Chem.MolFromSmiles("CCO")
    assert math.isnan(getattr(cg, name)(flat))


# 3. contract / invariants
def test_total_count():
    assert len(_DECLS) == 50
    assert len(set(cg.ALL_DESCRIPTOR_NAMES)) == 50


def test_name_equals_function_name():
    from collections import Counter

    blocks = Counter(d["block"] for d in _DECLS)
    assert blocks == {"CPSA": 42, "GravitationalIndex": 4, "GeometricalIndex": 4}
    for d in _DECLS:
        assert d["name"] == d["function_name"], d["name"]
        assert callable(getattr(cg, d["function_name"], None))
        assert d["category"] == "geometric"
        assert d["requires_3d"] is True


def test_dedup_guards():
    """Excluded duplicates must NOT appear: mordred TPSA (name-clash), PBF, MOMI (=PMI)."""
    names = set(cg.ALL_DESCRIPTOR_NAMES)
    assert "TPSA" not in names  # HAVE = RDKit topological TPSA
    assert "PBF" not in names  # HAVE
    assert not any(n.startswith("MOMI") for n in names)  # = HAVE PMI1/2/3


# 4. analytical identities
@pytest.mark.parametrize("mol", [pytest.param(m, id=m) for m in _PANEL])
def test_dpsa_identity(mol):
    """DPSA_v == PPSA_v − PNSA_v for every version."""
    m = _reconstruct(mol)
    for v in range(1, 6):
        dpsa = getattr(cg, f"DPSA{v}")(m)
        ppsa = getattr(cg, f"PPSA{v}")(m)
        pnsa = getattr(cg, f"PNSA{v}")(m)
        if not any(math.isnan(x) for x in (dpsa, ppsa, pnsa)):
            assert math.isclose(dpsa, ppsa - pnsa, rel_tol=1e-9, abs_tol=1e-9)


def test_geom_shape_and_petitjean_consistency():
    """GeomShapeIndex=(D−R)/R and GeomPetitjeanIndex=(D−R)/D from the same D,R."""
    m = _reconstruct(_PANEL[0])
    d = cg.GeomDiameter(m)
    r = cg.GeomRadius(m)
    assert math.isclose(cg.GeomShapeIndex(m), (d - r) / r, rel_tol=1e-9)
    assert math.isclose(cg.GeomPetitjeanIndex(m), (d - r) / d, rel_tol=1e-9)
