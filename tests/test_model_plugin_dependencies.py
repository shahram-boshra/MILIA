"""Model plugin dependencies are checked from installed package metadata.

Each entry is a PEP 508 requirement on a distribution name. It is parsed with
``packaging``, its marker is evaluated, and the installed version is read with
``importlib.metadata``; nothing is imported.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy
import pytest

from milia_pipeline.models.model_plugin_system import ModelPluginLoader, ModelPluginMetadata


def _problems(*dependencies: str) -> list[str]:
    metadata = ModelPluginMetadata(
        plugin_name="dependency_probe",
        version="1.0.0",
        author="probe",
        description="probe",
        plugin_type="user_experimental",
        milia_version=">=1.0.0",
        pyg_version=">=2.0.0",
        python_version=">=3.10",
        license="MIT",
        model_declarations=[],
        dependencies=list(dependencies),
    )
    result = ModelPluginLoader()._validate_dependencies(metadata)
    assert result["satisfied"] is (not result["missing"])
    return result["missing"]


def test_installed_distribution_with_matching_version_is_satisfied():
    assert _problems("torch>=2.0") == []


@pytest.mark.parametrize("name", ["torch_geometric", "torch-geometric", "Torch_Geometric"])
def test_distribution_names_are_matched_after_normalization(name):
    assert _problems(name) == []


def test_distribution_that_is_not_installed_is_missing():
    assert _problems("nonexistent-dist-xyz") == [
        "Missing required dependency: nonexistent-dist-xyz"
    ]


def test_installed_version_outside_the_specifier_is_unsatisfied():
    assert _problems("numpy<1") == [
        f"Unsatisfied dependency: numpy<1 (installed: numpy {numpy.__version__})"
    ]


def test_invalid_requirement_is_reported():
    (problem,) = _problems("bad spec !!")

    assert problem.startswith("Invalid dependency specifier: 'bad spec !!' (")
    assert "\n" not in problem


def test_requirement_whose_marker_excludes_this_environment_is_skipped():
    assert _problems("nonexistent-dist-xyz; python_version < '3.0'") == []


def test_dependencies_are_not_imported():
    with patch.object(importlib, "import_module") as import_module:
        problems = _problems("torch>=2.0", "nonexistent-dist-xyz", "numpy<1")

    import_module.assert_not_called()
    assert len(problems) == 2
