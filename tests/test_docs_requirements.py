"""docs/requirements.txt (Read the Docs installs it) states the same requirements as the `docs` extra.

Read the Docs builds from ``docs/requirements.txt`` without installing MILIA, while uv.lock and
``pip install milia-py[docs]`` use the ``docs`` extra in pyproject.toml. Two lists drift apart unless a
test compares them; this one does, by parsed PEP 508 requirement (name, specifier, marker).
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent


def _key(requirement: Requirement) -> tuple[str, str, str]:
    name = requirement.name.lower().replace("_", "-")
    return name, str(requirement.specifier), str(requirement.marker or "")


def _docs_requirements_file() -> set[tuple[str, str, str]]:
    lines = (ROOT / "docs" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {
        _key(Requirement(line.split("#", 1)[0].strip()))
        for line in lines
        if line.split("#", 1)[0].strip()
    }


def _docs_extra() -> set[tuple[str, str, str]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {_key(Requirement(spec)) for spec in project["optional-dependencies"]["docs"]}


def test_docs_requirements_file_equals_the_docs_extra():
    assert _docs_requirements_file() == _docs_extra()


def test_docs_requirements_are_not_empty():
    assert {name for name, _, _ in _docs_extra()} >= {
        "sphinx",
        "pydata-sphinx-theme",
        "myst-parser",
    }
