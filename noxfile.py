# =============================================================================
# noxfile.py — MILIA
# =============================================================================
# Reproducible multi-Python-version test automation per Scientific Python
# Development Guide (learn.scientific-python.org/development/guides/tasks/).
#
# Nox is the recommended task runner for the Scientific Python ecosystem.
# It creates isolated virtual environments per session, ensuring reproducible
# results across Python versions without polluting the developer's environment.
#
# Source evidence:
#   - pyproject.toml: Python 3.10/3.11/3.12 classifiers, [tool.pytest],
#     [tool.ruff], [project.optional-dependencies].dev, [build-system]
#   - conftest.py: 8 markers (smoke, contract, e2e, regression,
#     thread_safety, slow, integration, gpu)
#   - ci.yml: smoke tests as first CI gate, Ruff lint + format check
#   - .pre-commit-config.yaml: pre-commit-hooks v6.0.0, ruff v0.15.0,
#     detect-secrets v1.5.0
#   - Makefile: 20 targets (test, lint, format, build, etc.)
#   - CONTRIBUTING.md: conda install for heavy deps, pip install -e ".[dev]"
#
# Sessions:
#   lint        — Ruff linter + formatter check (Python 3.12 only)
#   tests       — Full test suite across Python 3.10/3.11/3.12
#   tests_smoke — Smoke tests only (fast first gate, no heavy deps)
#   build       — Verify sdist + wheel build (Python 3.12 only)
#
# Usage:
#   pipx install nox                  # install nox (or: brew install nox)
#   nox                               # run default sessions (lint, tests)
#   nox -l                            # list all available sessions
#   nox -s lint                       # run lint session only
#   nox -s tests                      # run tests across all Python versions
#   nox -s tests -- -m smoke          # pass extra args to pytest
#   nox -s tests -p 3.12              # run tests on Python 3.12 only
#   nox -R                            # reuse existing virtual environments
#   nox -s tests_conda                # run tests in conda environment
#
# GitHub Actions integration:
#   - uses: wntrblm/nox@2026.2.9
#     with:
#       python-versions: "3.10, 3.11, 3.12"
#   - run: pipx run nox -s lint
#   - run: pipx run nox -s tests
#
# Note on daily development: Nox is for reproducible multi-version testing
# and new contributor onboarding. Daily developers should use direct commands
# (pytest, ruff) or Makefile targets — per Scientific Python Development
# Guide: "A daily developer is not expected to use nox for simple tasks."
# =============================================================================

from __future__ import annotations

import shutil
from pathlib import Path

import nox

# ---------------------------------------------------------------------------
# Nox global configuration
# ---------------------------------------------------------------------------
# Minimum nox version required (Scientific Python Dev Guide recommends
# specifying this for reproducibility).
nox.needs_version = ">=2025.11.12"

# Use uv as the default venv backend when available; fall back to virtualenv.
# Source: Scientific Python Dev Guide NOX102.
nox.options.default_venv_backend = "uv|virtualenv"

# Default sessions to run when `nox` is invoked without `-s`.
# Lint + tests across all supported Python versions.
nox.options.sessions = ["lint", "tests"]

# ---------------------------------------------------------------------------
# Constants — sourced from pyproject.toml
# ---------------------------------------------------------------------------
# Python versions from pyproject.toml classifiers:
#   "Programming Language :: Python :: 3.10"
#   "Programming Language :: Python :: 3.11"
#   "Programming Language :: Python :: 3.12"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]

# Default Python for single-version sessions (latest supported).
PYTHON_DEFAULT = "3.12"

# Project root directory.
DIR = Path(__file__).parent.resolve()


# =============================================================================
# Session: lint — Ruff linter + formatter check
# =============================================================================
# Runs Ruff linter and formatter check in an isolated environment.
# Source: pyproject.toml [tool.ruff] (py310, line-length 100,
#   select E/W/F/I/UP/B/SIM, ignore E501, isort known-first-party).
# Mirrors: Makefile `lint` + `format-check` targets, ci.yml lint job.
@nox.session(python=PYTHON_DEFAULT)
def lint(session: nox.Session) -> None:
    """Run Ruff linter and formatter check."""
    session.install("ruff")
    session.run("ruff", "check", ".", *session.posargs)
    session.run("ruff", "format", "--check", ".")


# =============================================================================
# Session: tests — Full test suite across Python versions
# =============================================================================
# Parametrized across Python 3.10/3.11/3.12 (pyproject.toml classifiers).
# Installs the package in editable mode with dev extras.
# Source: pyproject.toml [tool.pytest.ini_options] (testpaths = ["tests"],
#   addopts = ["-v", "--tb=short", "--strict-markers"]).
# Note: Full test suite requires conda-managed scientific dependencies
# (PyTorch, PyG, RDKit). This session works when those are available in
# the system environment or when running smoke tests only.
@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite across Python versions."""
    session.install("-e", ".[dev]")
    session.run("pytest", *session.posargs)


# =============================================================================
# Session: tests_smoke — Smoke tests only (no heavy dependencies)
# =============================================================================
# Smoke tests are the "first gate in CI" (CONTRIBUTING.md, ci.yml).
# These are designed to run without conda-managed scientific packages.
# Source: conftest.py line 524 marker registration, ci.yml line 85.
@nox.session(python=PYTHON_VERSIONS)
def tests_smoke(session: nox.Session) -> None:
    """Run smoke tests only (fast, no heavy dependencies required)."""
    session.install("-e", ".[dev]")
    session.run("pytest", "-m", "smoke", *session.posargs)


# =============================================================================
# Session: tests_conda — Full test suite in a conda environment
# =============================================================================
# Uses nox's built-in conda backend for environments requiring heavy
# scientific dependencies (PyTorch, PyG, RDKit, etc.).
# Source: ci.yml commented conda job (lines 89–104), CONTRIBUTING.md
#   "Development Setup" (conda env create).
# Note: Requires conda/mamba/micromamba to be installed on the system.
#   Not a default session — run explicitly with: nox -s tests_conda
@nox.session(
    python=PYTHON_DEFAULT,
    venv_backend="conda",
    default=False,
)
def tests_conda(session: nox.Session) -> None:
    """Run the full test suite in a conda environment with scientific deps."""
    # Install heavy scientific dependencies via conda.
    # Package list sourced from ci.yml commented conda job (lines 93–96).
    session.conda_install(
        "numpy",
        "scipy",
        "pyyaml",
        "h5py",
        "pandas",
        "rdkit",
        "matplotlib",
        "pydantic-settings",
        "ase",
        "torchmetrics",
        "hydra-core",
        "optuna",
        "plotly",
        "scikit-learn",
        channel="conda-forge",
    )
    session.conda_install("pytorch", "cpuonly", channel="pytorch")
    session.conda_install("torch-geometric", channel="pyg")

    # Install MILIA + dev deps via pip (--no-deps to avoid breaking conda env).
    # Source: nox tutorial — "best practice only install pip packages with
    # the --no-deps option" in conda environments.
    session.install("-e", ".[dev]", "--no-deps")

    session.run("pytest", *session.posargs)


# =============================================================================
# Session: build — Verify sdist and wheel build
# =============================================================================
# Ensures the package builds correctly. Not a default session.
# Source: pyproject.toml [build-system] (setuptools>=77, build_meta).
# Mirrors: Makefile `build` target.
@nox.session(python=PYTHON_DEFAULT, default=False)
def build(session: nox.Session) -> None:
    """Build sdist and wheel, verify the package builds correctly."""
    build_path = DIR / "build"
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")
