"""The torch / PyG platform is declared once and repeated consistently.

The accelerator extras pin torch and its compiled PyG companions; each extra resolves them from its own
PyTorch index and PyG wheel page; Dependabot must reach the same PyG pages; docker-publish builds only
declared variants. (The Dockerfile's ``ACCEL`` default is not read here: ``.dockerignore`` keeps the
Dockerfile out of the test image, and ``uv sync --locked --extra ${ACCEL}`` plus ``verify_build.py --accel``
already fail the build for an undeclared or mismatched variant.) A torch upgrade that misses one of
these places locks or builds against the wrong wheels, so every copy is checked against the extras.

PyG names its wheel page after the minor's ``.0`` release for every patch of it ("PyTorch 2.12.*:
``${TORCH}=2.12.0``", PyG installation docs).
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())
COMPILED_STACK = {"torch", "pyg-lib", "torch-scatter", "torch-sparse"}


def _accelerators() -> list[str]:
    (group,) = PYPROJECT["tool"]["uv"]["conflicts"]
    return [entry["extra"] for entry in group]


def _pins(extra: str) -> dict[str, str]:
    pins = {}
    for line in PYPROJECT["project"]["optional-dependencies"][extra]:
        requirement = Requirement(line)
        (specifier,) = requirement.specifier
        assert specifier.operator == "==", line
        pins[requirement.name] = specifier.version
    return pins


def _torch_minor() -> str:
    major, minor, _ = _pins("cpu")["torch"].split(".")
    return f"{major}.{minor}"


def _index_urls() -> dict[str, str]:
    return {index["name"]: index["url"] for index in PYPROJECT["tool"]["uv"]["index"]}


ACCELERATORS = _accelerators()


def test_accelerators_are_cpu_and_cuda_variants():
    assert ACCELERATORS[0] == "cpu"
    assert all(re.fullmatch(r"cu\d{3}", accel) for accel in ACCELERATORS[1:])


@pytest.mark.parametrize("accel", ACCELERATORS)
def test_every_accelerator_pins_the_same_compiled_stack(accel):
    assert _pins(accel) == _pins("cpu")
    assert set(_pins(accel)) == COMPILED_STACK


def test_lock_wide_torch_constraint_matches_the_extras():
    # A sync without an accelerator extra resolves torch from PyPI through torchmetrics; the
    # constraint keeps that resolution on the same release as the extras.
    assert PYPROJECT["tool"]["uv"]["constraint-dependencies"] == [f"torch=={_pins('cpu')['torch']}"]


def test_index_urls_follow_the_torch_pin():
    expected = {}
    for accel in ACCELERATORS:
        expected[f"pytorch-{accel}"] = f"https://download.pytorch.org/whl/{accel}"
        expected[f"pyg-{accel}"] = f"https://data.pyg.org/whl/torch-{_torch_minor()}.0+{accel}.html"
    assert _index_urls() == expected


def test_each_compiled_package_comes_from_its_accelerator_index():
    sources = PYPROJECT["tool"]["uv"]["sources"]
    for package in COMPILED_STACK:
        prefix = "pytorch" if package == "torch" else "pyg"
        assert sources[package] == [
            {"index": f"{prefix}-{accel}", "extra": accel} for accel in ACCELERATORS
        ], package


def test_dependabot_reaches_exactly_the_pyg_indexes():
    dependabot = yaml.safe_load((REPO / ".github" / "dependabot.yml").read_text())
    pyg_urls = {name: url for name, url in _index_urls().items() if name.startswith("pyg-")}
    registries = {name: entry["url"] for name, entry in dependabot["registries"].items()}
    assert registries == pyg_urls
    (uv_update,) = [u for u in dependabot["updates"] if u["package-ecosystem"] == "uv"]
    assert sorted(uv_update["registries"]) == sorted(pyg_urls)
    ignored = {entry["dependency-name"] for entry in uv_update["ignore"]}
    assert COMPILED_STACK | {"torch-geometric"} <= ignored


def test_docker_publish_builds_only_declared_variants():
    workflow = (REPO / ".github" / "workflows" / "docker-publish.yml").read_text()
    lists = re.findall(r"echo '(?:accel|gpu)=(\[[^']*\])'", workflow)
    assert lists, "variant lists not found in docker-publish.yml"
    built = {accel for found in lists for accel in json.loads(found)}
    assert built <= set(ACCELERATORS)
    assert set(ACCELERATORS) <= built
