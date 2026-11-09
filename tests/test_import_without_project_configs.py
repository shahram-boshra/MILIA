"""MILIA imports outside a project directory: importing loads no configuration.

An installed package must import from any working directory. Before, ``import milia_pipeline``
failed with ``ConfigurationError: Configuration directory not found: ./configs/`` unless the
working directory held ``configs/``: the module ``__getattr__`` of ``config_constants`` loaded the
configuration for every missing attribute, including the import system's ``__path__`` probe.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from milia_pipeline.config import config_constants

REPO = Path(__file__).resolve().parents[1]

# Modules that still read configuration constants at import (they import the lazily loaded
# constants by name): converted in a follow-up change.
IMPORTS_CONSTANTS_BY_NAME = (
    "S8b: modules importing the lazy configuration constants by name "
    "load the configuration at import"
)


@pytest.mark.parametrize(
    "module",
    [
        "milia_pipeline",
        "milia_pipeline.config",
        "milia_pipeline.handlers",
        "milia_pipeline.transformations",
        "milia_pipeline.models",
        pytest.param(
            "milia_pipeline.datasets",
            marks=pytest.mark.xfail(strict=True, reason=IMPORTS_CONSTANTS_BY_NAME),
        ),
        pytest.param(
            "milia_pipeline.molecules",
            marks=pytest.mark.xfail(strict=True, reason=IMPORTS_CONSTANTS_BY_NAME),
        ),
    ],
)
def test_imports_from_a_directory_without_configs(module, tmp_path):
    """A fresh interpreter imports ``module`` with the working directory holding no ``configs/``."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]),
    }
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_unknown_attributes_raise_without_loading_the_configuration(monkeypatch):
    """PEP 562: a module ``__getattr__`` raises AttributeError for names it does not provide."""

    def fail(*args, **kwargs):
        raise AssertionError("load_config called for an attribute that is not a lazy constant")

    monkeypatch.setattr("milia_pipeline.config.config_loader.load_config", fail)
    for name in ("__path__", "__wrapped__", "NOT_A_CONSTANT"):
        with pytest.raises(AttributeError, match=name):
            getattr(config_constants, name)
        assert not hasattr(config_constants, name)
