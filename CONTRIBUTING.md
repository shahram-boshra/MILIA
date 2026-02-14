# Contributing to MILIA

Thank you for your interest in contributing to MILIA! We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code changes. Whether this is your first open-source contribution or your hundredth, we appreciate your time and effort.

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please [open an issue](https://github.com/shahram-boshra/MILIA/issues) with:

1. A clear, descriptive title.
2. Steps to reproduce the problem.
3. What you expected to happen versus what actually happened.
4. Your environment details: Python version, OS, and versions of key dependencies (PyTorch, PyTorch Geometric, RDKit). You can retrieve MILIA's dependency status programmatically:
   ```python
   from milia_pipeline import get_package_info, check_dependencies
   print(get_package_info())
   print(check_dependencies())
   ```
5. The full error traceback, if applicable.

Before opening a new issue, please search [existing issues](https://github.com/shahram-boshra/MILIA/issues) to check whether it has already been reported.

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/shahram-boshra/MILIA/issues). When suggesting an enhancement, please include:

1. A clear description of the proposed feature and the problem it solves.
2. An explanation of why this enhancement would be useful to MILIA users.
3. If possible, a sketch of the proposed API or configuration (MILIA is configuration-driven — YAML examples are especially helpful).

### Submitting Changes

We use the **fork and pull request** workflow:

1. [Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the [MILIA repository](https://github.com/shahram-boshra/MILIA).
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/MILIA.git
   cd MILIA
   ```
3. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes (see [Development Setup](#development-setup) and [Code Style](#code-style) below).
5. Commit with a clear, descriptive message:
   ```bash
   git add .
   git commit -m "Add concise description of the change"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. [Open a pull request](https://github.com/shahram-boshra/MILIA/compare) against the `main` branch.

For significant changes (new modules, architectural modifications, new dataset handlers), please **open an issue first** to discuss the approach before investing substantial effort.

## Development Setup

### Prerequisites

MILIA requires **Python 3.10 or later** and uses conda for managing heavy scientific dependencies (PyTorch, PyTorch Geometric, RDKit). See the [README](README.md#installation) for full installation details.

### Setting Up for Development

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/MILIA.git
cd MILIA

# 2. Create and activate conda environment
conda create -n milia python=3.10
conda activate milia

# 3. Install core dependencies via conda-forge
conda install -c conda-forge numpy scipy pyyaml h5py pandas rdkit \
    matplotlib pydantic-settings ase torchmetrics hydra-core optuna \
    plotly scikit-learn pytorch cpuonly -c pytorch

# 4. Install PyTorch Geometric and extensions
conda install -c pyg torch-geometric
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv

# 5. Install MILIA in editable mode with development dependencies
pip install -e ".[dev]"
```

This installs the development extras defined in `pyproject.toml`: `pytest>=8.0`, `pytest-mock>=3.14`, and `ruff`.

### Verifying Your Setup

```bash
# Confirm the package is installed and importable
python -c "from milia_pipeline import get_version; print(get_version())"

# Confirm the CLI entry point works
milia --help
```

## Running Tests

MILIA's test suite lives in the `tests/` directory (127 tests) and is configured in `pyproject.toml` under `[tool.pytest.ini_options]`. The project uses `--strict-markers` mode, so all custom markers must be registered.

```bash
# Run the full test suite
pytest

# Run with verbose output (default via pyproject.toml addopts)
pytest -v --tb=short

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only GPU-specific tests
pytest -m gpu

# Run a specific test file
pytest tests/test_example.py

# Run a specific test by name
pytest -k "test_function_name"
```

### Available Test Markers

The following markers are registered in `conftest.py` and `pyproject.toml`:

| Marker | Description |
|--------|-------------|
| `slow` | Tests taking more than 30 seconds |
| `integration` | Integration tests across components |
| `gpu` | Tests requiring GPU hardware |
| `smoke` | Fast smoke tests — first gate in CI |
| `contract` | Interface contract validation tests |
| `e2e` | End-to-end workflow tests |
| `regression` | Regression protection tests |
| `thread_safety` | Concurrent access tests |

### Writing Tests

When adding new functionality, include tests that cover it. Follow these conventions:

- Place test files in `tests/` with the naming pattern `test_*.py`.
- Name test classes `Test*` and test functions `test_*`.
- Use the shared fixtures from `tests/conftest.py` (e.g., `synthetic_pyg_dataset`, `minimal_config`, `mutable_config`, `mock_checkpoint`, `isolated_dataset_registry`, `tmp_working_dir`, `sample_mol_data`) rather than duplicating setup logic.
- For tests that mutate configuration, use the `mutable_config` fixture (which deep-copies the session-scoped `minimal_config`).
- Defer heavy imports (`torch`, `torch_geometric`, `numpy`) inside test functions or fixtures to avoid collection-time failures.
- Apply the appropriate marker(s) to your tests (e.g., `@pytest.mark.slow`, `@pytest.mark.integration`).

## Code Style

MILIA uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting, configured in `pyproject.toml` under `[tool.ruff]`.

### Configuration Summary

- **Target Python version**: 3.10
- **Line length**: 100 characters
- **Enabled rule sets**: `E` (pycodestyle errors), `W` (pycodestyle warnings), `F` (pyflakes), `I` (isort), `UP` (pyupgrade), `B` (flake8-bugbear), `SIM` (flake8-simplify)
- **Ignored**: `E501` (line-too-long — handled by the formatter)
- **Import sorting**: `milia_pipeline` is recognized as first-party via `[tool.ruff.lint.isort]`

### Running Ruff

```bash
# Check for linting issues
ruff check .

# Auto-fix linting issues where possible
ruff check --fix .

# Format code
ruff format .

# Check formatting without modifying files
ruff format --check .
```

Please ensure `ruff check .` and `ruff format --check .` pass before submitting a pull request.

## Documentation

MILIA uses [Sphinx](https://www.sphinx-doc.org/) with the [MyST Markdown](https://myst-parser.readthedocs.io/) parser and the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/). Documentation source files live in the `docs/` directory.

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt
# or: pip install -e ".[docs]"

# Build HTML documentation
make docs

# Build and open in browser
make docs-serve

# Clean build output
make docs-clean
```

### Writing Documentation

- Documentation pages are written in Markdown (`.md`) using MyST syntax.
- API documentation is auto-generated from docstrings via `sphinx.ext.autodoc` and `sphinx.ext.autosummary`.
- Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for all public functions, classes, and methods.
- Cross-reference other modules using MyST syntax: `` {mod}`milia_pipeline.config` ``.

## Changelog

MILIA follows the [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) format and adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

If your contribution adds, changes, deprecates, removes, or fixes functionality, add a brief entry under the `## [Unreleased]` section at the top of [CHANGELOG.md](CHANGELOG.md). Use the appropriate subsection (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`). Use past-tense, concise descriptions (e.g., "Added support for XYZ dataset handler").

## Pull Request Guidelines

When submitting a pull request:

1. **Keep it focused.** Each PR should address a single concern — one bug fix, one feature, or one refactoring. Avoid mixing unrelated changes.
2. **Include tests.** New features should include tests; bug fixes should include a test that reproduces the original issue.
3. **Pass all checks.** Ensure `pytest` passes and `ruff check .` / `ruff format --check .` report no issues.
4. **Update the changelog.** Add an entry to the `[Unreleased]` section of `CHANGELOG.md` if applicable.
5. **Write a clear description.** Explain what the PR does, why the change is needed, and any design decisions you made.
6. **Reference related issues.** Use GitHub keywords (e.g., "Closes #42") to link the PR to the issue it addresses.

## Project Structure

For an overview of the codebase architecture, module organization, and configuration system, see the [README](README.md#architecture).

Key directories for contributors:

| Directory | Contents |
|-----------|----------|
| `milia_pipeline/` | Core package (11 submodules) |
| `tests/` | Test suite (127 tests, `conftest.py` with shared fixtures) |
| `configs/` | Split YAML configuration files (7 root + 10 dataset-specific) |

## Questions?

If you have questions that are not answered by this guide, please [open an issue](https://github.com/shahram-boshra/MILIA/issues) with the details of your question. We are happy to help.

## License

By contributing to MILIA, you agree that your contributions will be licensed under the [MIT License](LICENSE).
