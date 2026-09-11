#!/usr/bin/env python3
"""
Unit + oracle tests for the ``constitutional_property`` descriptor plugin
(descriptor programme, Pace 2 / v1.5.0).

Validation-oracle protocol (Blueprint §6, §9):
    1. Ground-truth correctness — native output equals a FROZEN oracle snapshot
       (``mordredcommunity`` for the 18 Mordred-defined descriptors; published
       thresholds for ``VeberFilter``/``EganFilter``) to ``rtol=1e-4``, NaN-equality
       for invalids. A live ``mordredcommunity`` cross-check runs additionally when the
       package is importable (``importorskip``), guarding against snapshot rot.
    2. Determinism — identical output across repeated calls (2D, no RNG, no conformer).
    3. Robustness — invalid molecule / out-of-parameter element / undefined ratio → NaN,
       never raise.
    4. Contract — every declared function returns ``float``; ``plugin.yaml`` reconciles
       with the module (``name == function_name``); names unique; counts fixed at 20;
       categories/blocks valid; boolean flags ∈ {0.0, 1.0}.

The plugin module is loaded directly from its file path so this test has no torch /
package import dependency (mirrors the descriptor loader's ``spec_from_file_location``).
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

# ---------------------------------------------------------------------------
# Load the plugin module by path (no package import side effects)
# ---------------------------------------------------------------------------
_PLUGIN_DIR = (
    Path(__file__).parent.parent
    / "milia_pipeline"
    / "plugins"
    / "descriptors"
    / "constitutional_property"
)
_DESC_PY = _PLUGIN_DIR / "descriptors.py"
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_constitutional_property_oracle.json"

_spec = importlib.util.spec_from_file_location("constitutional_property_descriptors", _DESC_PY)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

_ORACLE_18 = (
    "VMcGowan",
    "Vabc",
    "apol",
    "bpol",
    "C1SP1",
    "C2SP1",
    "C1SP2",
    "C2SP2",
    "C3SP2",
    "C1SP3",
    "C2SP3",
    "C3SP3",
    "C4SP3",
    "HybRatio",
    "fMF",
    "fragCpx",
    "Lipinski",
    "GhoseFilter",
)
_RULE_FLAGS = ("Lipinski", "GhoseFilter", "VeberFilter", "EganFilter")
_RTOL = 1e-4
_ATOL = 1e-9

# The frozen oracle snapshot is loaded at COLLECTION time (not via a fixture) so that
# each (molecule, descriptor) pair becomes an independently-reported, -k-targetable test
# node. A single mismatch then localises precisely and never masks the remaining cases —
# the failure mode of a monolithic for-loop-with-one-assert (pytest parametrization guide;
# pytest docs "How to parametrize test functions").
_SNAPSHOT = json.loads(_ORACLE_JSON.read_text())
_PANEL: dict[str, str] = _SNAPSHOT["panel"]
_SNAP_VALUES: dict[str, dict] = _SNAPSHOT["values"]

_SNAPSHOT_CASES = [
    pytest.param(smiles, desc, _SNAP_VALUES[mol][desc], id=f"{mol}-{desc}")
    for mol, smiles in _PANEL.items()
    for desc in cp.ALL_DESCRIPTOR_NAMES
]
_MOL_CASES = [pytest.param(smiles, id=mol) for mol, smiles in _PANEL.items()]
_DESCRIPTOR_IDS = [pytest.param(name, id=name) for name in cp.ALL_DESCRIPTOR_NAMES]


def _mol(smiles: str) -> Chem.Mol:
    return Chem.MolFromSmiles(smiles)


def _nan_eq(got: float, expected, rtol: float = _RTOL) -> bool:
    """NaN-aware closeness: exact NaN match for the ``"nan"`` sentinel, else math.isclose."""
    if expected == "nan":
        return isinstance(got, float) and math.isnan(got)
    return math.isclose(got, float(expected), rel_tol=rtol, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# 1. Ground-truth correctness — frozen snapshot (HARD GATE), one node per pair
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,descriptor,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(smiles, descriptor, expected):
    """Native output equals the frozen oracle snapshot for this (molecule, descriptor)."""
    got = getattr(cp, descriptor)(_mol(smiles))
    assert _nan_eq(got, expected), f"{descriptor}: got {got!r} expected {expected!r}"


# ---------------------------------------------------------------------------
# 1b. Live mordred cross-check (runs only where mordredcommunity is installed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", _MOL_CASES)
def test_live_mordred_oracle(smiles):
    """The 18 Mordred-defined descriptors equal a live mordredcommunity calculation."""
    pytest.importorskip("mordred")
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    calc.descriptors = [d for d in calc.descriptors if str(d) in set(_ORACLE_18)]
    mol = _mol(smiles)
    oracle = {str(k): v for k, v in zip(calc.descriptors, calc(mol), strict=False)}
    for desc in _ORACLE_18:
        raw = oracle[desc]
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = float("nan")
        expected = "nan" if math.isnan(raw) else raw
        assert _nan_eq(getattr(cp, desc)(mol), expected), f"{desc}: live-oracle mismatch"


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("descriptor", _DESCRIPTOR_IDS)
def test_determinism_repeated_calls(descriptor):
    fn = getattr(cp, descriptor)
    mol = _mol("CC(=O)Oc1ccccc1C(=O)O")
    first = fn(mol)
    for _ in range(5):
        again = fn(mol)
        if math.isnan(first):
            assert math.isnan(again), f"{descriptor} non-deterministic (NaN drift)"
        else:
            assert again == first, f"{descriptor} non-deterministic"


# ---------------------------------------------------------------------------
# 3. Robustness — NaN, never raise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("descriptor", _DESCRIPTOR_IDS)
def test_none_molecule_returns_nan(descriptor):
    assert math.isnan(getattr(cp, descriptor)(None)), f"{descriptor}(None) should be NaN"


@pytest.mark.parametrize("descriptor", ["VMcGowan", "Vabc", "apol", "bpol"])
def test_out_of_parameter_element_skips_to_nan(descriptor):
    """Elements outside the parameterized table skip to NaN (Blueprint §4), not raise."""
    mol = _mol("[Na+].[Cl-]")  # sodium is outside the McGowan/vdW/polarizability tables
    assert math.isnan(getattr(cp, descriptor)(mol)), f"{descriptor} should NaN on Na"


@pytest.mark.parametrize("smiles", ["O", "O=C=O"], ids=["water_no_carbon", "co2_sp_only"])
def test_hybratio_nan_without_sp2_or_sp3_carbon(smiles):
    assert math.isnan(cp.HybRatio(_mol(smiles)))


@pytest.mark.parametrize("smiles", ["C", "[H][H]", "O=C=O", "[Ne]"])
def test_no_raise_on_degenerate_inputs(smiles):
    """Every descriptor returns a float (NaN or finite) — never raises — on degenerate input."""
    mol = _mol(smiles)
    for descriptor in cp.ALL_DESCRIPTOR_NAMES:
        assert isinstance(getattr(cp, descriptor)(mol), float)


# ---------------------------------------------------------------------------
# 4. Contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("descriptor", _DESCRIPTOR_IDS)
def test_descriptor_returns_float(descriptor):
    assert isinstance(getattr(cp, descriptor)(_mol("Cc1ccccc1")), float)


@pytest.mark.parametrize("smiles", ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO"])
@pytest.mark.parametrize("flag", _RULE_FLAGS)
def test_boolean_flags_are_zero_or_one(flag, smiles):
    assert getattr(cp, flag)(_mol(smiles)) in (0.0, 1.0), f"{flag} not boolean-valued"


@pytest.mark.parametrize(
    "smiles,expected",
    [
        pytest.param("CCCCCCCCCCCCCCCC", 0.0, id="hexadecane_gt10_rotbonds"),
        pytest.param("CC(=O)Oc1ccccc1C(=O)O", 1.0, id="aspirin_pass"),
    ],
)
def test_veber_filter_thresholds(smiles, expected):
    assert cp.VeberFilter(_mol(smiles)) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        pytest.param(
            "OCC1OC(O)(COC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O", 0.0, id="sucrose_tpsa_gt_131_6"
        ),
        pytest.param("c1ccccc1", 1.0, id="benzene_pass"),
    ],
)
def test_egan_filter_thresholds(smiles, expected):
    assert cp.EganFilter(_mol(smiles)) == expected


def test_total_descriptor_count():
    assert len(cp.ALL_DESCRIPTOR_NAMES) == 20
    assert len(set(cp.ALL_DESCRIPTOR_NAMES)) == 20


def test_no_collision_with_known_have_baseline():
    """Guard the FCSP3-drop / dedup decisions: none of our names is a known built-in."""
    have = {"FractionCSP3", "qed", "MolWt", "TPSA", "MolLogP", "MolMR", "FCSP3"}
    assert have.isdisjoint(set(cp.ALL_DESCRIPTOR_NAMES))


def test_yaml_declarations_reconcile_with_module():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(_YAML.read_text())
    decls = data["descriptors"]
    assert len(decls) == 20
    declared = {d["name"] for d in decls}
    assert declared == set(cp.ALL_DESCRIPTOR_NAMES)
    for d in decls:
        assert d["name"] == d["function_name"], f"{d['name']}: name != function_name"
        assert d["category"] in ("constitutional", "drug_likeness")
        assert d["block"]
        assert d["requires_3d"] is False
        assert d["requires_charges"] is False


@pytest.mark.parametrize("descriptor", _DESCRIPTOR_IDS)
def test_every_declared_function_exists_and_is_callable(descriptor):
    assert callable(getattr(cp, descriptor, None)), f"{descriptor} missing/not callable"
