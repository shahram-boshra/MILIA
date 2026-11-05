"""Plugin version requirements state lower bounds only.

Every first-party plugin manifest, every user template, and the transformation
plugin loader's defaults declare the MILIA, PyG and Python versions they need as
PEP 440 specifiers with lower bounds only (``>=``, ``>``), optionally excluding
known-bad releases (``!=``). Upper bounds (``<``, ``<=``, ``==``, ``~=``, ``===``)
are a guess about future releases: a plugin host that checks them rejects
compatible plugins at the next release they exclude, and users cannot override
the check.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet

import milia_pipeline
from milia_pipeline.transformations.plugin_system import PluginMetadata

PLUGINS_ROOT = Path(milia_pipeline.__file__).resolve().parent / "plugins"

# Manifest keys whose values are version *specifiers* (PEP 440).
SPECIFIER_KEYS = ("milia_version", "pyg_version", "python_version")
ALLOWED_OPERATORS = {">=", ">", "!="}


def _manifest_files() -> list[Path]:
    return sorted([*PLUGINS_ROOT.rglob("plugin.yaml"), *PLUGINS_ROOT.rglob("plugin.yaml.template")])


def _metadata_modules() -> list[Path]:
    return sorted([*PLUGINS_ROOT.rglob("__init__.py"), *PLUGINS_ROOT.rglob("__init__.py.template")])


def _yaml_specifiers(path: Path) -> list[tuple[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    found = [(key, data[key]) for key in SPECIFIER_KEYS if key in data]
    requirements = data.get("requirements")
    if isinstance(requirements, dict):
        found += [(f"requirements.{key}", value) for key, value in requirements.items()]
    return found


def _plugin_metadata_specifiers(path: Path) -> list[tuple[str, str]]:
    """String values of version keys in a module-level ``PLUGIN_METADATA = {...}``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "PLUGIN_METADATA" for t in node.targets):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value in SPECIFIER_KEYS
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                found.append((f"PLUGIN_METADATA[{key.value!r}]", value.value))
    return found


def _collect() -> list[tuple[str, str]]:
    cases = []
    for path in _manifest_files():
        rel = path.relative_to(PLUGINS_ROOT.parent)
        cases += [(f"{rel}:{key}", value) for key, value in _yaml_specifiers(path)]
    for path in _metadata_modules():
        rel = path.relative_to(PLUGINS_ROOT.parent)
        cases += [(f"{rel}:{key}", value) for key, value in _plugin_metadata_specifiers(path)]
    for field in SPECIFIER_KEYS:
        cases.append(
            (f"PluginMetadata.{field} default", PluginMetadata.model_fields[field].default)
        )
    return cases


CASES = _collect()


def test_every_plugin_kind_is_covered():
    """The scan reaches manifests of all three plugin kinds and every template."""
    locations = {case_id.split(":")[0] for case_id, _ in CASES}
    for kind in ("descriptors", "models", "transformations"):
        assert any(f"plugins/{kind}/" in loc for loc in locations), kind
    templates = {p.relative_to(PLUGINS_ROOT.parent) for p in PLUGINS_ROOT.rglob("*.template")}
    scanned_templates = {loc for loc in locations if loc.endswith(".template")}
    expected = {
        str(t) for t in templates if t.name in ("plugin.yaml.template", "__init__.py.template")
    }
    assert scanned_templates == expected


@pytest.mark.parametrize(("case_id", "specifier"), CASES, ids=[case_id for case_id, _ in CASES])
def test_version_requirement_has_lower_bounds_only(case_id, specifier):
    try:
        specifiers = SpecifierSet(specifier)
    except InvalidSpecifier as exc:
        pytest.fail(f"{case_id}: {specifier!r} is not a PEP 440 specifier ({exc})")

    operators = {spec.operator for spec in specifiers}
    assert operators, f"{case_id}: empty specifier"
    assert operators <= ALLOWED_OPERATORS, (
        f"{case_id}: {specifier!r} uses {sorted(operators - ALLOWED_OPERATORS)}; "
        "state a lower bound only (>=, >), optionally excluding releases with !="
    )
    assert operators & {">=", ">"}, f"{case_id}: {specifier!r} has no lower bound"
