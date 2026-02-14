# =============================================================================
# Makefile — MILIA
# =============================================================================
# Developer convenience targets for the MILIA molecular sciences framework.
#
# Source: MILIA Production Release File Audit §3.2. Targets sourced from
# pyproject.toml, CONTRIBUTING.md, .pre-commit-config.yaml, and .gitignore.
#
# Common in scientific Python projects (NumPy, SciPy, scikit-learn).
# See: pyOpenSci Python Package Guide — Task Runners; KDnuggets — The Case
# for Makefiles in Python Projects; Scientific Python Development Guide.
#
# Usage:
#   make              Show available targets (default)
#   make help         Same as above
#   make install-dev  Install in editable mode with dev extras
#   make test         Run the full test suite
#   make lint         Run Ruff linter
#   make format       Auto-format code with Ruff
#   make clean        Remove build/cache artifacts
#
# Requirements:
#   - Python 3.10+ (pyproject.toml requires-python = ">=3.10")
#   - conda environment with scientific dependencies (see CONTRIBUTING.md)
# =============================================================================

# ---- Variables ----
# Overridable from the command line: make test PYTHON=python3.12
# Source: pyproject.toml requires-python = ">=3.10"
PYTHON       ?= python3
PIP          ?= pip
PACKAGE_NAME := milia_pipeline
TEST_DIR     := tests
SRC_DIR      := milia_pipeline

# ---- Shell configuration ----
# Use bash with strict error handling for reliable recipe execution.
# Source: Makefile best practices (ricardoanderegg.com, GNU Make manual).
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# ---- Default target ----
.DEFAULT_GOAL := help

# =============================================================================
# Help
# =============================================================================
# Self-documenting help: parses '## ' comments after target names.
# Source: KDnuggets, bargsten.org, ricardoanderegg.com — standard pattern.

.PHONY: help
help:  ## Show available targets with descriptions
	@echo ""
	@echo "MILIA — Development Targets"
	@echo "==========================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Installation
# =============================================================================
# Source: pyproject.toml [project.optional-dependencies].dev,
#         pyproject.toml [project.scripts] milia = "main:main",
#         CONTRIBUTING.md "Development Setup" section.

.PHONY: install
install:  ## Install MILIA in editable mode (runtime only)
	$(PIP) install -e .

.PHONY: install-dev
install-dev:  ## Install MILIA in editable mode with dev extras (pytest, ruff)
	$(PIP) install -e ".[dev]"

# =============================================================================
# Testing
# =============================================================================
# Source: pyproject.toml [tool.pytest.ini_options]
#   testpaths = ["tests"], addopts = ["-v", "--tb=short", "--strict-markers"]
#   markers: slow, integration, gpu, smoke, contract, e2e, regression, thread_safety
# Source: CONTRIBUTING.md "Running Tests" section.
# Note: addopts in pyproject.toml already supplies -v --tb=short --strict-markers,
# so they do not need to be repeated here.

.PHONY: test
test:  ## Run the full test suite
	$(PYTHON) -m pytest

.PHONY: test-smoke
test-smoke:  ## Run smoke tests only (fast first gate)
	$(PYTHON) -m pytest -m smoke

.PHONY: test-integration
test-integration:  ## Run integration tests only
	$(PYTHON) -m pytest -m integration

.PHONY: test-fast
test-fast:  ## Run tests excluding slow tests
	$(PYTHON) -m pytest -m "not slow"

# =============================================================================
# Code Quality
# =============================================================================
# Source: pyproject.toml [tool.ruff] (py310, line-length 100,
#   select E/W/F/I/UP/B/SIM, ignore E501, isort known-first-party milia_pipeline)
# Source: CONTRIBUTING.md "Running Ruff" section.
# Source: .pre-commit-config.yaml — ruff-check args: --fix --exit-non-zero-on-fix --show-fixes

.PHONY: lint
lint:  ## Check code for linting issues (Ruff)
	$(PYTHON) -m ruff check .

.PHONY: lint-fix
lint-fix:  ## Auto-fix linting issues where possible (Ruff)
	$(PYTHON) -m ruff check --fix --show-fixes .

.PHONY: format
format:  ## Auto-format code (Ruff)
	$(PYTHON) -m ruff format .

.PHONY: format-check
format-check:  ## Check formatting without modifying files (Ruff)
	$(PYTHON) -m ruff format --check .

.PHONY: check
check: lint format-check test  ## Run all checks: lint + format-check + test

# =============================================================================
# Pre-commit
# =============================================================================
# Source: .pre-commit-config.yaml — 3 repos (pre-commit-hooks v6.0.0,
#   ruff-pre-commit v0.15.0, detect-secrets v1.5.0).
# Activation requires git init + pre-commit install (documented in config header).

.PHONY: pre-commit
pre-commit:  ## Run all pre-commit hooks on the entire codebase
	pre-commit run --all-files

.PHONY: pre-commit-install
pre-commit-install:  ## Install pre-commit hooks into .git/hooks/
	pre-commit install

.PHONY: pre-commit-update
pre-commit-update:  ## Update pre-commit hook versions
	pre-commit autoupdate

# =============================================================================
# Build & Distribution
# =============================================================================
# Source: pyproject.toml [build-system] requires = ["setuptools>=77"],
#   build-backend = "setuptools.build_meta".
# Uses PEP 517 build frontend (python -m build).

.PHONY: build
build:  ## Build sdist and wheel distributions
	$(PYTHON) -m build

# =============================================================================
# Documentation
# =============================================================================
# Source: docs/conf.py (Sphinx 8.1.x), docs/requirements.txt,
#   Scientific Python Dev Guide — sphinx-build invocation.
# Install doc dependencies first: pip install -r docs/requirements.txt

.PHONY: docs
docs:  ## Build HTML documentation (Sphinx)
	sphinx-build -b html docs docs/_build

.PHONY: docs-serve
docs-serve: docs  ## Build docs and open in browser
	$(PYTHON) -m webbrowser -t "docs/_build/index.html"

.PHONY: docs-clean
docs-clean:  ## Remove documentation build output
	rm -rf docs/_build docs/api/generated

# =============================================================================
# Cleanup
# =============================================================================
# Patterns sourced from .gitignore (GitHub Python template + MILIA-specific):
#   __pycache__/ (L8), *.py[codz] (L9), *.egg-info/ (L30),
#   build/ (L17), dist/ (L19), .pytest_cache/ (L57),
#   .ruff_cache/ (L211), .coverage (L49-50), htmlcov/ (L47),
#   *.egg (L32), MANIFEST (L33).
# Does NOT delete: .egg-info in project root (needed for editable install,
#   per pypa/setuptools #3348 — see audit §4.1).
# Does NOT delete: test_data/, scripts/, docs/, archive/ (local-only directories).

.PHONY: clean
clean:  ## Remove build artifacts, caches, and compiled files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[codz]" -delete 2>/dev/null || true
	find . -type f -name "*$$py.class" -delete 2>/dev/null || true
	rm -rf build/ dist/ sdist/ wheels/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage .coverage.*
	rm -f coverage.xml
	rm -f MANIFEST

.PHONY: clean-all
clean-all: clean  ## Remove all artifacts including .egg-info (full reset)
	rm -rf $(PACKAGE_NAME).egg-info/
	rm -rf docs/_build docs/api/generated

# =============================================================================
# Information
# =============================================================================
# Source: milia_pipeline/__init__.py L252: __version__ = "1.1.0"
#   get_package_info() returns version, author, license, status, python_requires.
#   check_dependencies() returns dependency availability status.

.PHONY: version
version:  ## Display the current MILIA version
	@$(PYTHON) -c "from $(PACKAGE_NAME) import __version__; print(__version__)"

.PHONY: info
info:  ## Display MILIA package info and dependency status
	@$(PYTHON) -c "\
		from $(PACKAGE_NAME) import get_package_info, check_dependencies; \
		info = get_package_info(); \
		deps = check_dependencies(); \
		print('Package Info:'); \
		[print(f'  {k}: {v}') for k, v in info.items()]; \
		print('Dependencies:'); \
		[print(f'  {k}: {v}') for k, v in deps.items()]"
