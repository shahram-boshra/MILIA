# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.0] - 2026-09-02

### Added

- **Molecular Descriptors — Pace 1: 1,019 ADD-CORE RDKit 3D descriptors** shipped as the first-party `addcore_3d` descriptor plugin (`plugins/descriptors/addcore_3d/`), extending the descriptor universe beyond the ~492-descriptor RDKit `HAVE` baseline. Blocks (RDKit-native, verified against the pinned runtime): WHIM (114), GETAWAY (273), RDF (210), 3D-MoRSE (224), Autocorr3D (80), USR (12), USRCAT (60), MQN (42), and heavy-atom oxidation-number aggregates (min/max/mean/sum). Vector blocks use a compute-once-per-molecule block-cache (`functools.lru_cache(maxsize=1)` keyed on the molecule), exposing each element as an individually-named scalar descriptor while issuing a single RDKit call per block. All descriptors are deterministic under the pinned ETKDG seed (42) and return `NaN` (never raise) on invalid input or unmet preconditions (e.g. USR's ≥3-atom requirement). One-per-family test module `tests/test_descriptor_addcore3d_unit.py` gates ground-truth-vs-RDKit-oracle equality, determinism, robustness, and contract (1,019 unique registrations, `block`/`category`/`requires_3d` metadata).
- Descriptor-plugin discovery is now config-driven for the descriptor kind-container: `configs/descriptors.yaml` gains a `molecular_descriptors.plugins` block (`enabled`, `plugin_paths`, `auto_discover`, `auto_validate`) pointing at `milia_pipeline/plugins/descriptors/` — adding a descriptor plugin requires no core-file edits.

### Changed

- `DescriptorMetadata` (`descriptors/descriptor_categories.py`) gains a non-breaking optional `block` field carrying the Dragon/Todeschini block-provenance label (e.g. `WHIM`, `GETAWAY`), so descriptors record fine-grained provenance without expanding the six-value `DescriptorCategory` enum per release. `DescriptorDeclaration` (`descriptors/descriptor_plugin_system.py`) gains matching optional `block` and `requires_extra` fields (the latter reserved for optional-dependency skip-not-fail wiring in later paces); both are threaded from `plugin.yaml` through registration into the registry metadata. All additions default to `None`/absent — existing plugins and metadata are unaffected.
- Pinned `rdkit==2025.3.5` → `rdkit==2025.3.6`. Release 2025.03.6 fixes GETAWAY nondeterminism ([rdkit#7264](https://github.com/rdkit/rdkit/issues/7264)); the previous pin was the last release exhibiting call-to-call GETAWAY variance, which would otherwise violate the descriptor determinism gate. Baseline descriptor values are bit-identical across the two patch releases (no regression).


## [1.3.0] - 2026-08-26

### Added

- **55 novel graph transforms across 9 first-party plugin families**, all configuration-driven and composable: graph products (Cartesian, Tensor, Strong, Lexicographic, Rooted, Corona), combinatorial & dual operators (line/Levi graphs, quotient, transitive closure/reduction, bipartite projection, graph dual), curvature-based rewiring (Forman–Ricci, effective-resistance, spectral-gap, resistance-curvature), topological lifts (hypergraph clique/star expansion, persistent-homology features), spectral & signal transforms, graph reduction/coarsening, structural padding for static batching, non-spatial augmentation, and latent-structural encodings (motif compression, structural-role, anonymous-walk). Brings the discoverable transform total to 130+.

### Changed

- Reorganised `milia_pipeline/plugins/` into per-kind containers: transformation plugins now live under `plugins/transformations/` (symmetric with `plugins/descriptors/`), and a `plugins/models/` container was added. Introduced `user_template/` scaffolds for the transformation and model kinds and `get_transform_plugins_directory()` / `get_model_plugins_directory()` helpers; model plugin discovery now resolves package-relative (CWD-independent) instead of relative to the working directory. The `pyg_augmentation` plugin moved into `plugins/transformations/`; the empty `plugins/myplugins/` scaffold was removed (users add plugins via each kind's `user_template/` or an external `plugins.plugin_paths` entry).
- Made the plugin compatibility smoke-test precondition-aware: a new `TransformPreconditionError` distinguishes transforms that require caller-supplied inputs (a second operand graph, a partition, or a graph property such as planarity/acyclicity/bipartiteness) from genuine defects. Such transforms are now reported as skipped (informational) rather than failed; constructor parameters are synthesised from declared constraints where possible, and paramless-constructor defects are still surfaced as failures.

### Fixed

- `RandomNodeSample` now defaults to `ratio=0.5` when neither `num` nor `ratio` is given, making it usable bare and consistent with the sibling augmentation transforms (which default `p=0.5`); with `num` set, `ratio` remains `None`.
- Logging setup tests: removed a stale `inspect` mock (the code no longer uses `inspect`) and corrected an obsolete file-handler-`OSError` expectation — `setup_logging` intentionally falls back to console-only logging on file errors rather than raising. Aligned the `Raises:` docstring accordingly.

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

[unreleased]: https://github.com/shahram-boshra/MILIA/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/shahram-boshra/MILIA/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/shahram-boshra/MILIA/compare/v1.2.2...v1.3.0
[1.2.2]: https://github.com/shahram-boshra/MILIA/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/shahram-boshra/MILIA/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/shahram-boshra/MILIA/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/shahram-boshra/MILIA/releases/tag/v1.1.0
