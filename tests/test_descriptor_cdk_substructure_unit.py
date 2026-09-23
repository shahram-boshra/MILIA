#!/usr/bin/env python3
"""
Unit + oracle tests for the ``cdk_substructure`` descriptor plugin
(descriptor programme, Pace 11 / v1.14.0).

DEP-BOUND engine plugin: the descriptors are the CANONICAL CDK values (not an RDKit
reimplementation), so the "oracle" is CDK itself and the test is an integration/reproducibility
check — the plugin's output must equal the frozen CDK snapshot (CDK via the ``CDK-pywrapper``
jar). Requires ``CDK_pywrapper`` + ``jpype`` + a JRE; skipped cleanly where absent.

Covered: frozen-snapshot equality (307 integer counts), determinism, robustness (None -> NaN),
and the plugin contract (307 unique; name==function_name; category fragments; block
SubstructureCount; requires_3d false; declared CDK-pywrapper dependency).
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
# The plugin module imports jpype + CDK_pywrapper at module level (opt-in gate); skip the whole
# module cleanly where the [descriptors-cdk] extra / Python >= 3.11 aren't present.
pytest.importorskip("jpype")
pytest.importorskip("CDK_pywrapper")
from rdkit import Chem  # noqa: E402

_PLUGIN_DIR = (
    Path(__file__).parent.parent / "milia_pipeline" / "plugins" / "descriptors" / "cdk_substructure"
)
_YAML = _PLUGIN_DIR / "plugin.yaml"
_ORACLE_JSON = Path(__file__).parent / "data_descriptor_cdk_substructure_oracle.json"

_spec = importlib.util.spec_from_file_location("cdk_descriptors", _PLUGIN_DIR / "descriptors.py")
cdk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdk)

_yaml = pytest.importorskip("yaml")
_DECLS = _yaml.safe_load(_YAML.read_text())["descriptors"]
_NAMES = [d["name"] for d in _DECLS]

_SNAP = json.loads(_ORACLE_JSON.read_text())
_PANEL = _SNAP["panel"]
_VALUES = _SNAP["values"]


def _cdk_available():
    return cdk._init_cdk()


_needs_cdk = pytest.mark.skipif(
    not _cdk_available(), reason="CDK engine (CDK-pywrapper + jpype + JRE) not available"
)

_SNAPSHOT_CASES = [
    pytest.param(smiles, mol, name, _VALUES[mol][name], id=f"{mol}-{name}")
    for mol, smiles in _PANEL.items()
    for name in _NAMES
]
_NAME_IDS = [pytest.param(n, id=n) for n in _NAMES]


# 1. frozen CDK snapshot (HARD GATE when CDK present)
@_needs_cdk
@pytest.mark.parametrize("smiles,mol,name,expected", _SNAPSHOT_CASES)
def test_oracle_snapshot(smiles, mol, name, expected):
    got = getattr(cdk, name)(Chem.MolFromSmiles(smiles))
    assert not math.isnan(got) and int(got) == int(expected), f"{mol}/{name}"


# 2. determinism + robustness
@_needs_cdk
def test_determinism():
    m = Chem.MolFromSmiles(_PANEL["aspirin"])
    first = {n: getattr(cdk, n)(m) for n in _NAMES}
    for _ in range(3):
        assert {n: getattr(cdk, n)(m) for n in _NAMES} == first


@pytest.mark.parametrize("name", _NAME_IDS)
def test_none_returns_nan(name):
    assert math.isnan(getattr(cdk, name)(None))


# 3. contract / invariants (do NOT require the CDK engine)
def test_total_count():
    assert len(_DECLS) == 307
    assert len(set(cdk.ALL_DESCRIPTOR_NAMES)) == 307


def test_name_equals_function_name():
    for d in _DECLS:
        assert d["name"] == d["function_name"], d["name"]
        assert callable(getattr(cdk, d["function_name"], None))
        assert d["category"] == "fragments"
        assert d["block"] == "SubstructureCount"
        assert d["requires_3d"] is False


def test_declares_cdk_dependency():
    doc = _yaml.safe_load(_YAML.read_text())
    assert any("CDK-pywrapper" in str(dep) for dep in (doc.get("dependencies") or []))
    assert all(n.startswith("SubFPC") for n in cdk.ALL_DESCRIPTOR_NAMES)


# 4. counts are non-negative integers (when CDK present)
@_needs_cdk
@pytest.mark.parametrize("smiles", [pytest.param(s, id=m) for m, s in _PANEL.items()])
def test_counts_are_nonneg_integers(smiles):
    m = Chem.MolFromSmiles(smiles)
    for n in _NAMES:
        v = getattr(cdk, n)(m)
        assert not math.isnan(v) and v >= 0 and float(v).is_integer()
