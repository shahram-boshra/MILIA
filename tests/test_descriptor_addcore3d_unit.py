#!/usr/bin/env python3
"""
Unit + discovery tests for the ``addcore_3d`` descriptor plugin
(descriptor programme, Pace 1 / v1.4.0).

Validation-oracle protocol (Blueprint §6, §9):
    1. Ground-truth correctness — plugin indexers equal a direct ``rdMolDescriptors.Calc*``
       call on a fixed molecule panel (RDKit is its own oracle for ADD-CORE blocks).
    2. Determinism — identical output across repeated calls and a fresh embed with seed=42.
    3. Robustness — invalid molecule / RDKit precondition (USR < 3 atoms) → NaN, never raise.
    4. Contract — every declared function returns ``float``; block-cache computes once per
       molecule; declared/registered counts reconcile; names are unique.

The plugin module is loaded directly from its file path so this test has no torch / package
import dependency (mirrors the descriptor loader's own ``spec_from_file_location`` mechanism).
"""

import importlib.util
import math
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem, rdMolDescriptors  # noqa: E402

# ---------------------------------------------------------------------------
# Load the plugin module by path (no package import side effects)
# ---------------------------------------------------------------------------
_PLUGIN_DIR = (
    Path(__file__).parent.parent / "milia_pipeline" / "plugins" / "descriptors" / "addcore_3d"
)
_DESC_PY = _PLUGIN_DIR / "descriptors.py"
_YAML = _PLUGIN_DIR / "plugin.yaml"

_spec = importlib.util.spec_from_file_location("addcore_3d_descriptors", _DESC_PY)
addcore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(addcore)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_SEED = 42

# Vector blocks: (prefix, count, direct RDKit callable, requires_3d)
_VECTOR_BLOCKS = [
    ("WHIM", 114, rdMolDescriptors.CalcWHIM, True),
    ("GETAWAY", 273, rdMolDescriptors.CalcGETAWAY, True),
    ("RDF", 210, rdMolDescriptors.CalcRDF, True),
    ("MoRSE", 224, rdMolDescriptors.CalcMORSE, True),
    ("AUTOCORR3D", 80, rdMolDescriptors.CalcAUTOCORR3D, True),
    ("USR", 12, rdMolDescriptors.GetUSR, True),
    ("USRCAT", 60, rdMolDescriptors.GetUSRCAT, True),
    ("MQN", 42, rdMolDescriptors.MQNs_, False),
]

_PANEL_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "benzene": "c1ccccc1",
    "ethanol": "CCO",
    "charged": "C(=O)([O-])[NH3+]",
    "multiring": "c1ccc2ccccc2c1",
    "stereo": "C[C@H](O)C(=O)O",
}


def _embed(smiles: str, seed: int = _SEED) -> Chem.Mol:
    """Deterministic 3D embed (mirrors the MILIA calculator: AddHs + seed=42 + MMFF)."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


@pytest.fixture(scope="module")
def panel() -> dict[str, Chem.Mol]:
    return {name: _embed(smi) for name, smi in _PANEL_SMILES.items()}


def _nan_eq(a: float, b: float, rtol: float = 1e-6) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 1. Ground-truth correctness — oracle = direct RDKit call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prefix,count,rdkit_fn,_r3d", _VECTOR_BLOCKS)
def test_vector_block_matches_rdkit_oracle(panel, prefix, count, rdkit_fn, _r3d):
    """Every indexer equals the corresponding element of the direct RDKit block call."""
    for mol in panel.values():
        expected = list(rdkit_fn(mol))
        assert len(expected) == count
        for i in range(count):
            fn = getattr(addcore, f"{prefix}_{i + 1}")
            got = fn(mol)
            assert isinstance(got, float)
            assert _nan_eq(got, float(expected[i])), f"{prefix}_{i + 1} mismatch"


def test_oxidation_aggregates_match_manual_oracle(panel):
    """Oxidation aggregates equal min/max/mean/sum over heavy-atom Pauling states."""
    for mol in panel.values():
        work = Chem.RemoveHs(Chem.Mol(mol))
        rdMolDescriptors.CalcOxidationNumbers(work)
        states = [
            a.GetIntProp("OxidationNumber")
            for a in work.GetAtoms()
            if a.GetAtomicNum() > 1 and a.HasProp("OxidationNumber")
        ]
        assert states  # panel molecules all have heavy atoms
        assert _nan_eq(addcore.OxidationNumber_min(mol), float(min(states)))
        assert _nan_eq(addcore.OxidationNumber_max(mol), float(max(states)))
        assert _nan_eq(addcore.OxidationNumber_mean(mol), sum(states) / len(states))
        assert _nan_eq(addcore.OxidationNumber_sum(mol), float(sum(states)))


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------
def test_determinism_repeated_calls(panel):
    """Same molecule → identical values across repeated indexer calls."""
    mol = panel["aspirin"]
    for name in ("WHIM_1", "GETAWAY_5", "RDF_10", "MoRSE_3", "MQN_1"):
        fn = getattr(addcore, name)
        assert fn(mol) == fn(mol)


def test_determinism_across_fresh_embed():
    """Re-embedding with the same seed reproduces identical 3D descriptor values."""
    a = _embed("CC(=O)Oc1ccccc1C(=O)O", seed=_SEED)
    b = _embed("CC(=O)Oc1ccccc1C(=O)O", seed=_SEED)
    for name in ("WHIM_1", "GETAWAY_1", "RDF_1", "AUTOCORR3D_1"):
        fn = getattr(addcore, name)
        assert _nan_eq(fn(a), fn(b))


# ---------------------------------------------------------------------------
# 3. Robustness — NaN, never raise
# ---------------------------------------------------------------------------
def test_none_molecule_returns_nan():
    for name in addcore.ALL_DESCRIPTOR_NAMES:
        assert math.isnan(getattr(addcore, name)(None))


def test_missing_conformer_returns_nan():
    """3D indexers on a molecule without a conformer → NaN (no raise)."""
    flat = Chem.MolFromSmiles("CCO")  # no conformer embedded
    assert math.isnan(addcore.WHIM_1(flat))
    assert math.isnan(addcore.GETAWAY_1(flat))


def test_usr_too_few_atoms_returns_nan():
    """GetUSR requires ≥ 3 atoms; a 1-atom molecule → NaN, not ValueError."""
    ne = Chem.AddHs(Chem.MolFromSmiles("[Ne]"))
    AllChem.EmbedMolecule(ne, randomSeed=_SEED)
    assert math.isnan(addcore.USR_1(ne))


def test_mqn_is_2d_no_conformer_needed():
    """MQN is a 2D block: values on a conformer-free molecule are finite."""
    flat = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    assert not math.isnan(addcore.MQN_1(flat))


# ---------------------------------------------------------------------------
# 4. Contract — block-cache, counts, uniqueness, dtype
# ---------------------------------------------------------------------------
def test_total_descriptor_count():
    """1,015 vector/scalar + 4 oxidation aggregates = 1,019."""
    assert len(addcore.ALL_DESCRIPTOR_NAMES) == 1019
    assert len(set(addcore.ALL_DESCRIPTOR_NAMES)) == 1019  # unique


def test_block_cache_computes_once_per_molecule(panel, monkeypatch):
    """All 42 MQN indexers on one molecule trigger exactly one RDKit MQNs_ call."""
    calls = {"n": 0}
    real = rdMolDescriptors.MQNs_

    def counting(mol, *a, **k):
        calls["n"] += 1
        return real(mol, *a, **k)

    addcore._compute_mqn.cache_clear()
    monkeypatch.setattr(rdMolDescriptors, "MQNs_", counting)
    mol = panel["caffeine"]
    for i in range(1, 43):
        getattr(addcore, f"MQN_{i}")(mol)
    assert calls["n"] == 1


def test_every_declared_function_exists_and_is_callable():
    for name in addcore.ALL_DESCRIPTOR_NAMES:
        fn = getattr(addcore, name, None)
        assert callable(fn), f"missing indexer {name}"


def test_yaml_declarations_reconcile_with_module():
    """plugin.yaml declares exactly the module's descriptors; name == function_name."""
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(_YAML.read_text())
    decls = data["descriptors"]
    assert len(decls) == 1019
    declared = {d["name"] for d in decls}
    assert declared == set(addcore.ALL_DESCRIPTOR_NAMES)
    for d in decls:
        assert d["name"] == d["function_name"]  # single-registration invariant
        assert d["module_path"] == "descriptors"
        assert "block" in d
        assert d["version"] == "1.4.0"


def test_category_and_requires_3d_assignment():
    """Geometric 3D blocks vs 2D constitutional blocks per Blueprint §5/§9."""
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(_YAML.read_text())
    by_name = {d["name"]: d for d in data["descriptors"]}
    # 3D geometric
    assert by_name["WHIM_1"]["category"] == "geometric"
    assert by_name["WHIM_1"]["requires_3d"] is True
    assert by_name["GETAWAY_1"]["requires_3d"] is True
    # 2D constitutional
    assert by_name["MQN_1"]["category"] == "constitutional"
    assert by_name["MQN_1"]["requires_3d"] is False
    assert by_name["OxidationNumber_sum"]["category"] == "constitutional"
    assert by_name["OxidationNumber_sum"]["requires_3d"] is False
