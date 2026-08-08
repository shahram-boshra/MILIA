# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.2] - 2026-08-08

### Added

- Bundled sample inputs under `test_data/` (`molecules.csv`, `molecules_inchi.csv`, `molecules.sdf`, `molecules.xyz`) covering every `--predict` input format; re-tracked via `.gitignore` (text fixtures only — heavy binaries remain ignored, and the samples stay excluded from the PyPI sdist/wheel).

### Changed

- `--predict` walkthrough in `README.md`, `QUICKSTART.md`, and `docs/getting-started.md` now points at the bundled `test_data/molecules.csv`.
- Enabled the models master switch in `configs/models.yaml`.

### Fixed

- Corrected stale documentation: checkpoint filename `best_model.pt` → `best.pt`; test-suite counts `127` → 163 test files / 1,965 smoke; removed the inaccurate "pytest runs inside the container" claim (the runtime image has no test tooling).

## [1.2.1] - 2026-08-06

### Changed

- PyPI distribution name is now `milia-py` (installed via `pip install milia-py`).
  The bare name `milia` was unavailable on PyPI (blocked by its name-similarity
  check), so the distribution name is decoupled from the import name — a standard
  practice (cf. `scikit-learn`/`sklearn`, `Pillow`/`PIL`). The import package
  (`import milia_pipeline`), the `milia` CLI entry point, the GitHub repository
  (`MILIA`), and the Zenodo/Mendeley DOIs are unchanged.

### Added

- First automated PyPI release via GitHub Actions **Trusted Publishing** (OIDC,
  no API tokens), gated through a `pypi` deployment environment.
- `PyPI` project URL in `[project.urls]`.

## [1.2.0] - 2026-06-26

### Added

- QM40 dataset support: 162,954 neutral drug-like ZINC molecules (10-40 heavy
  atoms) at B3LYP/6-31G(2df,p), with optimized geometries, Mulliken charges,
  16 scalar quantum-mechanical properties, and per-bond local vibrational mode
  force constants. Reference: Madushanka, Moura Jr. & Kraka, *Scientific Data*
  11, 1376 (2024). Includes dataset implementation, handler, preprocessor (ZIP
  of three CSV files joined by Zinc_id), CSV parser, colocated YAML
  configuration, and `config_constants` registry entries. Uses the
  `coordinate_based` molecule-creation strategy (SMILES present, no InChI) and
  is neutral-only (`supports_charged_molecules()` is `False`).

## [1.1.0] - 2026-02-12

### Added

- Multi-format molecular conversion with RDKit integration (`molecules/`).
- Automated structural and chemical feature extraction and enrichment.
- PyTorch Geometric compatible dataset implementation (`miliaDataset`).
- Modular wavefunction data preprocessing for MOLDEN and FCHK formats.
- Extensible graph transformation system with experimental setup support.
- Three-tier plugin architecture for descriptors, transformations, and general extensions.
- Unified dataset handler pattern with DFT and DMC support (`create_handler`).
- Schema-validated YAML configuration system (`config/`).
- Comprehensive CLI with interactive mode and `milia` entry point.
- GNN model training with hyperparameter optimization support (`models/`).
- Post-training prediction and inference workflow with checkpoint support (`models/post_training/`).
- Transfer learning via `FineTuner` and `FreezeStrategy`.
- Multi-format input support via `DataConverterRegistry`.
- Molecular descriptor calculation with plugin system (`descriptors/`).
- Three-tier exception hierarchy with 50+ specialized exception classes.
- Registry integration for dataset type validation and CLI diagnostics.
- Full test suite with pytest configuration (`tests/`).
- MIT license.
- Production `pyproject.toml` with PEP 517/518/621/639 compliance.
- Comprehensive `README.md` with installation, quick start, and API reference.

[unreleased]: https://github.com/shahram-boshra/MILIA/compare/v1.2.2...HEAD
[1.2.2]: https://github.com/shahram-boshra/MILIA/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/shahram-boshra/MILIA/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/shahram-boshra/MILIA/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/shahram-boshra/MILIA/releases/tag/v1.1.0
