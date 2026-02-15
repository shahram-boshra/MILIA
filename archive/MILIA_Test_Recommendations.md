# MILIA Pipeline — Recommended Tests Beyond Unit Tests

**Purpose**: Identify the essential test files to add to the MILIA project's existing 127-file test suite, targeting CI/CD readiness on a private GitHub repository and demonstrating software engineering maturity for postdoc/industry job applications.

**Context**: All unit tests are complete. The project already has 9 integration tests and 1 performance test. This document focuses on the **missing test categories** that industry best practices and CI/CD workflows demand.

---

## 1. Smoke Tests

Smoke tests are rapid, lightweight checks that verify the pipeline's core mechanics run without crashing. They are the **first gate in any CI/CD pipeline** — if smoke tests fail, no further (more expensive) tests are triggered. They do not validate correctness; they confirm the system is "not on fire."

**Why essential for your goals**: Smoke tests are the single most impactful test category for CI/CD pipelines. GitHub Actions workflows typically run smoke tests first, gating all subsequent stages. They demonstrate you understand production ML pipeline practices.

### 1.1 `test_smoke_pipeline_end_to_end.py`

**What it tests**: The entire pipeline can execute from configuration loading through data processing to model training output — on a tiny synthetic dataset — without crashing.

**Modules exercised**:
- `milia_pipeline/config/config_loader.py` — Configuration loading
- `milia_pipeline/config/config_containers.py` — Config container creation
- `milia_pipeline/config/config_accessors.py` — Config access patterns
- `milia_pipeline/datasets/registry.py` — Dataset registry discovery
- `milia_pipeline/handlers/base_handler.py` — Handler factory (`create_dataset_handler`)
- `milia_pipeline/handlers/handler_registry.py` — Handler registry lookup
- `milia_pipeline/molecules/molecule_converter_core.py` — `MoleculeDataConverter`
- `milia_pipeline/transformations/graph_transforms.py` — Transform composition
- `milia_pipeline/models/registry/model_registry.py` — Model discovery
- `milia_pipeline/models/factory/model_factory.py` — Model creation
- `milia_pipeline/models/training/trainer.py` — Trainer instantiation
- `main.py` — Main entry point orchestration

**Scope**: Uses minimal synthetic data (5–10 molecules). Trains for 1–2 epochs. Asserts no exceptions raised and outputs are produced.

---

### 1.2 `test_smoke_imports.py`

**What it tests**: All top-level package imports succeed without errors (catches circular imports, missing dependencies, broken `__init__.py` files).

**Modules exercised**:
- `milia_pipeline/__init__.py`
- `milia_pipeline/config/__init__.py`
- `milia_pipeline/molecules/__init__.py`
- `milia_pipeline/transformations/__init__.py`
- `milia_pipeline/datasets/__init__.py`
- `milia_pipeline/handlers/__init__.py`
- `milia_pipeline/preprocessing/__init__.py`
- `milia_pipeline/descriptors/__init__.py`
- `milia_pipeline/models/__init__.py`
- `milia_pipeline/models/hpo/__init__.py`
- `milia_pipeline/models/post_training/__init__.py`
- `milia_pipeline/models/builders/__init__.py`
- `milia_pipeline/models/acceleration/__init__.py`
- `milia_pipeline/models/deployment/__init__.py`
- `milia_pipeline/plugins/__init__.py`

**Scope**: Each import is a separate test case. Asserts no `ImportError`, no `CircularImportError`. Fast (< 5 seconds total).

**Relationship to `test__init__*.py` files (15 files)**:
- `test_smoke_imports.py` is the **thin CI/CD gate** — "can we import each package at all?" (5 tests × 15 packages = 75 parametrized cases, < 5 s).
- The 15 individual `test__init__*.py` files (e.g. `test__init__config.py`, `test__init__datasets.py`, ...) provide **deep smoke + contract testing** — `__all__` completeness, return-type contracts, conditional availability flags, exception hierarchy validation, Pydantic model serialization, registry consistency, etc. These files run in the **contract** and **unit** CI stages, not the fast smoke gate.
- Both layers coexist by design: `test_smoke_imports.py` catches hard import failures in < 5 s (CI Stage 1), while the `test__init__*.py` files catch subtle API surface regressions in the later CI stages.

---

### 1.3 `test_smoke_cli.py`

**What it tests**: CLI entry points respond correctly to `--help`, `--list-transforms`, `--list-handlers`, `--validate-config` without crashing.

**Modules exercised**:
- `milia_pipeline/cli_manager.py` — `CLIManager`, `parse_cli_args`
- `milia_pipeline/config/config_loader.py` — Config loading via CLI
- `milia_pipeline/transformations/__init__.py` — Transform listing
- `milia_pipeline/handlers/__init__.py` — Handler listing

**Scope**: Invokes CLI with safe read-only flags. Asserts exit code 0 and expected output patterns.

---

### 1.4 `test_smoke_prediction_pipeline.py`

**What it tests**: The post-training prediction pathway can load a mock checkpoint and run prediction without crashing.

**Modules exercised**:
- `milia_pipeline/models/post_training/checkpoint/checkpoint_manager.py` — `CheckpointManager`
- `milia_pipeline/models/post_training/inference/model_loader.py` — `ModelLoader`
- `milia_pipeline/models/post_training/inference/predictor.py` — `Predictor`
- `milia_pipeline/models/post_training/data_preparation/data_converter.py` — `DataConverterRegistry`

**Scope**: Creates a minimal mock checkpoint, loads it, runs prediction on 1–2 synthetic PyG Data objects. Asserts predictions are tensors of correct shape.

---

## 2. Contract Tests

Contract tests verify that interfaces between modules (protocols, ABCs, registries) conform to their documented contracts. They catch breaking changes when internal APIs evolve — critical for a project with 11 core modules and 3 plugin systems.

**Why essential for your goals**: Contract tests demonstrate deep understanding of interface-driven design and the Open/Closed Principle — key architectural patterns in MILIA. They are especially valuable for demonstrating the registry-based, protocol-driven architecture to reviewers.

### 2.1 `test_contract_dataset_handler_protocol.py`

**What it tests**: Every registered handler implementation satisfies all 11 methods of `DatasetHandlerProtocol` with correct return types.

**Modules exercised**:
- `milia_pipeline/datasets/protocols.py` — `DatasetHandlerProtocol` (11 methods)
- `milia_pipeline/handlers/handler_registry.py` — `HandlerRegistry.list_all()`
- `milia_pipeline/handlers/implementations/dft.py` — `DFTDatasetHandler`
- `milia_pipeline/handlers/implementations/dmc.py` — `DMCDatasetHandler`
- `milia_pipeline/handlers/implementations/wavefunction.py` — `WavefunctionDatasetHandler`
- `milia_pipeline/handlers/implementations/qm9.py` — `QM9DatasetHandler`
- `milia_pipeline/handlers/implementations/ani1x.py` — `ANI1xDatasetHandler`
- `milia_pipeline/handlers/implementations/ani1ccx.py` — `ANI1ccxDatasetHandler`
- `milia_pipeline/handlers/implementations/ani2x.py` — `ANI2xDatasetHandler`
- `milia_pipeline/handlers/implementations/rmd17.py` — `RMD17DatasetHandler`
- `milia_pipeline/handlers/implementations/xxmd.py` — `XXMDDatasetHandler`
- `milia_pipeline/handlers/implementations/qdpi.py` — `QDPiDatasetHandler`

**Scope**: Iterates over all registered handlers. For each, verifies: method existence, return type correctness (e.g., `get_dataset_type()` returns `str`, `get_required_properties()` returns `List[str]`), and `runtime_checkable` protocol conformance.

---

### 2.2 `test_contract_dataset_base.py`

**What it tests**: Every registered dataset implementation (via `@register` decorator) satisfies the `BaseDataset` ABC contract and has valid `DatasetMetadata`, `DatasetSchema`, and `DatasetFeatures`.

**Modules exercised**:
- `milia_pipeline/datasets/base.py` — `BaseDataset`, `DatasetMetadata`, `DatasetSchema`, `DatasetFeatures`
- `milia_pipeline/datasets/registry.py` — `DatasetRegistry`, `list_all()`, `get()`
- `milia_pipeline/datasets/implementations/dft.py` — `DFTDataset`
- `milia_pipeline/datasets/implementations/dmc.py` — `DMCDataset`
- `milia_pipeline/datasets/implementations/wavefunction.py` — `WavefunctionDataset`
- `milia_pipeline/datasets/implementations/qm9.py` — `QM9Dataset`
- `milia_pipeline/datasets/implementations/ani1x.py` — `ANI1xDataset`
- `milia_pipeline/datasets/implementations/ani1ccx.py` — `ANI1ccxDataset`
- `milia_pipeline/datasets/implementations/rmd17.py` — `RMD17Dataset`
- `milia_pipeline/datasets/implementations/ani2x.py` — `ANI2xDataset`
- `milia_pipeline/datasets/implementations/xxmd.py` — `XXMDDataset`
- `milia_pipeline/datasets/implementations/qdpi.py` — `QDPiDataset`

**Scope**: For each registered dataset: verifies `metadata` is a `DatasetMetadata` instance, `schema` is a `DatasetSchema` instance, `features` is a `DatasetFeatures` instance. Validates Pydantic V2 model serialization (`model_dump()`).

---

### 2.3 `test_contract_registry_consistency.py`

**What it tests**: Cross-registry consistency — every dataset type registered in `DatasetRegistry` has a corresponding handler in `HandlerRegistry`, and vice versa.

**Modules exercised**:
- `milia_pipeline/datasets/registry.py` — `DatasetRegistry`, `list_all()`
- `milia_pipeline/handlers/handler_registry.py` — `HandlerRegistry`, `list_all()`
- `milia_pipeline/handlers/base_handler.py` — `create_dataset_handler()`

**Scope**: Asserts bijection between dataset registry and handler registry entries. Verifies `create_dataset_handler(dataset_type)` succeeds for every registered dataset type.

---

### 2.4 `test_contract_model_registries.py`

**What it tests**: Model-related registries (LossRegistry, OptimizerRegistry, SchedulerRegistry, MetricsRegistry) expose consistent APIs and all registered entries can be instantiated.

**Modules exercised**:
- `milia_pipeline/models/training/loss_functions.py` — `LossRegistry`, `get_loss()`, `list_losses()`
- `milia_pipeline/models/training/optimizers.py` — `OptimizerRegistry`, `get_optimizer()`, `list_optimizers()`
- `milia_pipeline/models/training/schedulers.py` — `SchedulerRegistry`, `get_scheduler()`, `list_schedulers()`
- `milia_pipeline/models/training/metrics.py` — `MetricsRegistry`, `get_metric()`, `list_available()`

**Scope**: For each registry: list all entries, instantiate each with default/minimal parameters, verify the returned object is the expected type.

---

## 3. End-to-End (E2E) Tests

E2E tests validate complete user-facing workflows from input to output, exercising the full stack. They are the final validation gate before deployment.

**Why essential for your goals**: E2E tests prove the system works as a user would use it. They are the tests a CV reviewer or hiring committee will look for to confirm the software actually functions as documented.

### 3.1 `test_e2e_training_workflow.py`

**What it tests**: A complete training workflow: config loading → dataset creation → data splitting → model creation → training (few epochs) → checkpoint saving → checkpoint loading → prediction.

**Modules exercised**:
- `milia_pipeline/config/config_loader.py` — `load_config()`
- `milia_pipeline/models/factory/model_factory.py` — `ModelFactory.create_model()`
- `milia_pipeline/models/factory/target_selection_config.py` — `TargetSelectionConfig`
- `milia_pipeline/models/training/data_splitting.py` — `DataSplitter`
- `milia_pipeline/models/training/data_preparation.py` — `TaskDataPreparer`
- `milia_pipeline/models/training/trainer.py` — `Trainer.fit()`
- `milia_pipeline/models/training/callbacks.py` — `ModelCheckpoint`, `EarlyStopping`
- `milia_pipeline/models/training/loss_functions.py` — `get_loss()`
- `milia_pipeline/models/training/optimizers.py` — `get_optimizer()`
- `milia_pipeline/models/training/schedulers.py` — `get_scheduler()`
- `milia_pipeline/models/training/metrics.py` — `get_metrics_for_task()`
- `milia_pipeline/models/post_training/checkpoint/checkpoint_manager.py` — `CheckpointManager`
- `milia_pipeline/models/post_training/inference/model_loader.py` — `load_model()`
- `milia_pipeline/models/post_training/inference/predictor.py` — `Predictor`

**Scope**: Uses a small synthetic PyG dataset (50–100 graphs). Trains for 3–5 epochs. Validates: loss decreases, checkpoint files exist on disk, loaded model produces predictions, predictions have correct tensor shape.

---

### 3.2 `test_e2e_preprocessing_workflow.py`

**What it tests**: A complete preprocessing workflow: raw data → handler selection → molecule conversion → feature enrichment → transform application → PyG dataset creation.

**Modules exercised**:
- `milia_pipeline/config/config_loader.py` — Config loading
- `milia_pipeline/handlers/base_handler.py` — `create_dataset_handler()`
- `milia_pipeline/molecules/molecule_converter_core.py` — `MoleculeDataConverter.convert()`
- `milia_pipeline/molecules/mol_conversion_utils.py` — `create_rdkit_mol()`, `mol_to_pyg_data()`
- `milia_pipeline/molecules/molecule_validator.py` — Molecular validation
- `milia_pipeline/molecules/mol_structural_features.py` — `add_structural_features()`
- `milia_pipeline/molecules/molecule_feature_enricher.py` — Feature enrichment
- `milia_pipeline/molecules/property_enrichment.py` — `enrich_pyg_data_with_properties()`
- `milia_pipeline/transformations/graph_transforms.py` — Transform composition and application
- `milia_pipeline/datasets/milia_dataset.py` — `miliaDataset`

**Scope**: Uses minimal test NPZ or synthetic data. Asserts: PyG Data objects have expected attributes (`x`, `edge_index`, `y`, `pos`), correct tensor dtypes and shapes.

---

### 3.3 `test_e2e_hpo_workflow.py`

**What it tests**: A complete HPO workflow: config → HPOManager creation → search space building → optimization (2–3 trials) → best parameters retrieval → study analysis.

**Modules exercised**:
- `milia_pipeline/models/hpo/hpo_config.py` — `HPOConfig`
- `milia_pipeline/models/hpo/hpo_manager.py` — `HPOManager.optimize()`
- `milia_pipeline/models/hpo/backends/optuna_backend.py` — `OptunaBackend`
- `milia_pipeline/models/hpo/search_spaces/search_space_builder.py` — `SearchSpaceBuilder`
- `milia_pipeline/models/hpo/search_spaces/param_types.py` — `ParamType`
- `milia_pipeline/models/hpo/callbacks/optuna_callback.py` — `OptunaPruningCallback`
- `milia_pipeline/models/hpo/analysis/study_analyzer.py` — `StudyAnalyzer`
- `milia_pipeline/models/training/trainer.py` — Training within trials
- `milia_pipeline/models/training/data_splitting.py` — Data splitting within trials

**Scope**: Runs 2–3 trials with a tiny model and small dataset. Asserts: best parameters are returned, `StudyAnalyzer` produces analysis without errors, trial count matches expected.

---

### 3.4 `test_e2e_transfer_learning_workflow.py`

**What it tests**: Complete transfer learning workflow: train base model → save checkpoint → load via `FineTuner` → apply freeze strategy → fine-tune on new data → predict.

**Modules exercised**:
- `milia_pipeline/models/training/trainer.py` — Initial training
- `milia_pipeline/models/training/callbacks.py` — `ModelCheckpoint`
- `milia_pipeline/models/post_training/checkpoint/checkpoint_manager.py` — Checkpoint save/load
- `milia_pipeline/models/post_training/transfer_learning/fine_tuner.py` — `FineTuner.from_checkpoint()`, `FreezeStrategy`
- `milia_pipeline/models/post_training/inference/predictor.py` — Post-fine-tuning prediction

**Scope**: Trains a small model for 2 epochs, saves checkpoint, loads via FineTuner with `FreezeStrategy.ENCODER`, fine-tunes for 2 more epochs. Asserts: frozen layers have `requires_grad=False`, fine-tuned model produces predictions.

---

## 4. Regression Tests

Regression tests verify that known bugs stay fixed and that critical outputs remain stable across code changes. They protect against silent regressions.

**Why essential for your goals**: Regression tests are a CI/CD staple — they are what `git push` triggers on every commit. They demonstrate that the codebase is maintained and stable.

### 4.1 `test_regression_config_migration.py` — ⚠️ RETIRED (migrate_config.py DEPRECATED)

**Status**: **RETIRED** — `migrate_config.py` has been deprecated (renamed to `migrate_config_py.DEPRECATED`) and is scheduled for removal after full pipeline testing with the new YAML-Splitting system. This one-time migration utility is not part of the runtime pipeline.

**Rationale for retirement**:
- `migrate_config.py` was a one-time utility script that split the monolithic `config.yaml` into the `configs/` directory structure. It is not invoked at runtime.
- The actual regression concern — ensuring single-file (`config.yaml`) and split-file (`configs/`) modes produce identical effective configurations — is already covered by the runtime `config_loader.py` infrastructure: `_discover_config_files()`, `_collect_yaml_files()`, `_deep_merge_configs()`, `_load_and_merge_yaml_files()`, and `load_config()` (lines 398–906).
- This test responsibility is **absorbed by Section 5.2** (`test_config_split_mode_parity.py`), which tests the same core concern (single-file vs split-file parity) against the live `config_loader.py` code path rather than against a deprecated migration script.

**Original modules exercised** (now covered by Section 5.2):
- `milia_pipeline/config/config_loader.py` — `load_config()` (both modes) → **covered in 5.2**
- ~~`migrate_config.py`~~ — **DEPRECATED, removed from test scope**
- `milia_pipeline/config/config_containers.py` — Container equality verification → **covered in 5.2**

**Action**: Do not create this test file. See Section 5.2 (`test_config_split_mode_parity.py`) which now carries **High** priority to compensate.

---

### 4.2 `test_regression_checkpoint_compatibility.py`

**What it tests**: Checkpoints saved with the current code can be loaded correctly, and the v2.0 format is maintained. Also tests backward compatibility with v1.0 format if applicable.

**Modules exercised**:
- `milia_pipeline/models/post_training/checkpoint/checkpoint_manager.py` — `CheckpointManager`, `CHECKPOINT_FORMAT_VERSION`
- `milia_pipeline/models/post_training/inference/model_loader.py` — `ModelLoader.load_from_checkpoint()`

**Scope**: Creates a checkpoint with a known model, saves it, reloads it. Asserts: model weights are identical (bitwise), `data_info` including `structural_features_config` is preserved, format version is `'2.0'`.

---

### 4.3 `test_regression_featurization_consistency.py`

**What it tests**: The structural features configuration persistence fix (v1.6.0) — training-time featurization config is saved in checkpoints and correctly applied during prediction to avoid dimension mismatches.

**Modules exercised**:
- `milia_pipeline/models/training/callbacks.py` — `ModelCheckpoint._save_checkpoint()` (saves `structural_features_config` in `data_info`)
- `milia_pipeline/models/post_training/inference/predictor.py` — `Predictor.structural_features_config` property
- `milia_pipeline/models/post_training/data_preparation/data_converter.py` — `convert_to_pyg(..., structural_features_config=)`
- `milia_pipeline/molecules/mol_structural_features.py` — `add_structural_features()`

**Scope**: Trains with specific structural features config, saves checkpoint, loads checkpoint, verifies `structural_features_config` is identical, converts new data with that config, asserts feature dimensions match training-time dimensions.

---

## 5. Configuration Validation Tests

Configuration tests verify that the YAML configuration system handles valid, invalid, edge-case, and migration scenarios correctly.

**Why essential for your goals**: Configuration is the primary user interface for scientific software. Robust config validation tests prevent "silent misconfiguration" — a common source of incorrect scientific results.

### 5.1 `test_config_validation_comprehensive.py`

**What it tests**: End-to-end configuration validation including schema validation, Pydantic V2 model validation, cross-field consistency checks, and meaningful error messages for invalid configs.

**Modules exercised**:
- `milia_pipeline/config/config_loader.py` — `load_config()`
- `milia_pipeline/config/config_schemas.py` — `YAMLSchemaValidator`, `ValidationConfig`
- `milia_pipeline/config/validators.py` — `validate_with_pydantic_model()`, `ValidationResult`
- `milia_pipeline/config/config_containers.py` — `DatasetConfig`, `FilterConfig`, `ProcessingConfig`
- `milia_pipeline/datasets/base.py` — `DatasetSchema` (Pydantic V2)
- `milia_pipeline/models/utils/config_bridge.py` — `ConfigBridge` (31 Pydantic BaseModel classes)

**Scope**: Tests with valid configs, configs with missing required fields, configs with invalid types, configs with out-of-range values. Asserts: valid configs load successfully, invalid configs raise `ConfigurationError` or `ValidationError` with informative messages.

---

### 5.2 `test_config_split_mode_parity.py` — ⬆️ EXPANDED (absorbs retired Section 4.1)

**What it tests**: Split-mode configuration (`configs/` directory with multiple YAML files) produces the same effective configuration as single-file mode (`config.yaml`). This test now also carries the regression protection responsibility originally assigned to Section 4.1 (`test_regression_config_migration.py`), which was retired due to `migrate_config.py` deprecation.

**Modules exercised**:
- `milia_pipeline/config/config_loader.py` — Both loading modes (`load_config()`, `_discover_config_files()`, `_collect_yaml_files()`, `_deep_merge_configs()`, `_load_and_merge_yaml_files()`)
- `milia_pipeline/config/config_accessors.py` — Accessor consistency across modes
- `milia_pipeline/config/config_containers.py` — Container equality verification (absorbed from Section 4.1)

**Scope**: Provides identical configurations in both formats. Asserts: (1) deep equality of resulting config dictionaries, (2) accessor results are identical across both modes, (3) `DatasetConfig` and other Pydantic containers created from both modes are equivalent. Also tests edge cases: empty split files, missing `main.yaml` in split directory, `property_availability` colocation merging from `datasets/` subdirectory.

**Priority**: **High** (upgraded from Medium to compensate for Section 4.1 retirement)

---

## 6. Thread Safety Tests

Thread safety tests verify that concurrent access to thread-safe components (registries, singletons) does not cause race conditions or data corruption.

**Why essential for your goals**: MILIA documents extensive thread safety guarantees. Tests that verify these guarantees demonstrate rigorous engineering — highly valued in both research and industry.

### 6.1 `test_thread_safety_registries.py`

**What it tests**: Concurrent registration, lookup, and unregistration in all registry types do not cause race conditions or inconsistent state.

**Modules exercised**:
- `milia_pipeline/datasets/registry.py` — `DatasetRegistry` (non-singleton, RLock)
- `milia_pipeline/handlers/handler_registry.py` — `HandlerRegistry` (RLock)
- `milia_pipeline/descriptors/descriptor_registry.py` — `DescriptorRegistry` (singleton)
- `milia_pipeline/models/registry/model_registry.py` — `ModelRegistry` (singleton)
- `milia_pipeline/models/builders/layer_registry.py` — `LayerRegistry` (singleton, RLock)

**Scope**: Spawns 10–20 threads performing simultaneous `register()`, `get()`, `list_all()`, `is_registered()` operations. Asserts: no exceptions raised, final state is consistent, no data corruption.

---

## 7. Fixture and Conftest Infrastructure

While not test files themselves, shared fixtures are essential CI/CD infrastructure that enables all the tests above to run efficiently.

### 7.1 `conftest.py` (Root-level test fixtures)   <─── ONGOING

**What it provides**: Shared pytest fixtures used across multiple test categories.

**Key fixtures to define**:
- `synthetic_pyg_dataset` — A small (10–50 graph) synthetic PyG dataset for training/prediction tests
- `minimal_config` — A valid minimal `config.yaml` dict for config tests
- `mock_checkpoint` — A pre-built checkpoint file for prediction and regression tests
- `isolated_dataset_registry` — A fresh `DatasetRegistry()` instance (non-singleton) for contract tests
- `tmp_working_dir` — A temporary directory as `working_root_dir` for DI-pattern tests
- `sample_mol_data` — Synthetic molecular data dicts for preprocessing E2E tests

**Modules required**:
- `milia_pipeline/datasets/registry.py` — `DatasetRegistry()`
- `milia_pipeline/models/factory/model_factory.py` — Model creation for mock checkpoints
- `torch_geometric.data` — `Data`, `InMemoryDataset` for synthetic data
- `torch` — Tensor creation
- `pathlib.Path`, `tempfile` — Temporary directories

---

## Summary: Recommended Test Files

| # | Test File | Category | Priority | Est. CI Time |
|---|-----------|----------|----------|-------------|
| 1 | `test_smoke_pipeline_end_to_end.py` | Smoke | **Critical** | ~30s |
| 2 | `test_smoke_imports.py` | Smoke | **Critical** | ~5s |
| 3 | `test_smoke_cli.py` | Smoke | **Critical** | ~10s |
| 4 | `test_smoke_prediction_pipeline.py` | Smoke | **Critical** | ~15s |
| 5 | `test_contract_dataset_handler_protocol.py` | Contract | **High** | ~10s |
| 6 | `test_contract_dataset_base.py` | Contract | **High** | ~10s |
| 7 | `test_contract_registry_consistency.py` | Contract | **High** | ~5s |
| 8 | `test_contract_model_registries.py` | Contract | **High** | ~15s |
| 9 | `test_e2e_training_workflow.py` | E2E | **High** | ~60s |
| 10 | `test_e2e_preprocessing_workflow.py` | E2E | **High** | ~45s |
| 11 | `test_e2e_hpo_workflow.py` | E2E | **Medium** | ~90s |
| 12 | `test_e2e_transfer_learning_workflow.py` | E2E | **Medium** | ~60s |
| 13 | ~~`test_regression_config_migration.py`~~ | ~~Regression~~ | ~~**High**~~ | ~~RETIRED — `migrate_config.py` deprecated; absorbed by #17~~ |
| 14 | `test_regression_checkpoint_compatibility.py` | Regression | **High** | ~15s |
| 15 | `test_regression_featurization_consistency.py` | Regression | **High** | ~20s |
| 16 | `test_config_validation_comprehensive.py` | Config | **High** | ~10s |
| 17 | `test_config_split_mode_parity.py` | Config + Regression | **High** ⬆️ | ~10s |
| 18 | `test_thread_safety_registries.py` | Thread Safety | **Medium** | ~15s |
| 19 | `conftest.py` | Infrastructure | **Critical** | N/A |

**Total: 18 new test files + 1 conftest** (bringing the project from 127 to 146 test files)
**Note**: Section 4.1 (`test_regression_config_migration.py`) retired due to `migrate_config.py` deprecation — its regression coverage absorbed by Section 5.2 (`test_config_split_mode_parity.py`, priority upgraded to **High**).
**Note**: The 15 `test__init__*.py` files (deep smoke + contract tests for each `__init__.py`) are **separate from** `test_smoke_imports.py` (thin import gate) and are counted as part of the existing 127-file base.

---

## Recommended GitHub Actions CI/CD Pipeline Structure

The tests above are designed to integrate into a staged CI pipeline:

```yaml
# .github/workflows/ci.yml
name: MILIA CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # Stage 1: Fast gate (< 30s)
  smoke-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"
      - run: pytest tests/test_smoke_*.py -v --tb=short

  # Stage 2: Contract + Config (< 60s)
  contract-tests:
    needs: smoke-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"
      - run: pytest tests/test_contract_*.py tests/test_config_validation_*.py tests/test_config_split_*.py -v

  # Stage 3: Unit tests (bulk)
  unit-tests:
    needs: smoke-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"
      - run: pytest tests/test_*_unit.py -v --tb=short

  # Stage 4: Integration + E2E + Regression (longer)
  integration-tests:
    needs: [contract-tests, unit-tests]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"
      - run: pytest tests/test_e2e_*.py tests/test_regression_*.py tests/test_*_integration*.py tests/test_thread_safety_*.py -v

  # Stage 5: Quality gate
  quality-gate:
    needs: [integration-tests]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]" coverage
      - run: coverage run -m pytest tests/ --tb=short
      - run: coverage report --fail-under=70
```

---

## Pytest Markers for Test Organization

Add to `pyproject.toml` or `pytest.ini`:

```ini
[tool:pytest]
markers =
    smoke: Quick health checks (< 30s total)
    contract: Interface contract validation
    e2e: End-to-end workflow tests
    regression: Regression protection tests
    thread_safety: Concurrent access tests
    slow: Tests taking > 30 seconds
```

This enables selective execution:
```bash
pytest -m smoke          # CI Stage 1
pytest -m contract       # CI Stage 2
pytest -m "not slow"     # Quick feedback loop
pytest -m e2e            # Full validation
```

---

**Document Version**: 1.2.0
**Created**: February 2026
**Updated**: February 2026 — Section 4.1 retired (`migrate_config.py` deprecated); Section 5.2 expanded and upgraded to **High** priority; Section 1.2 updated to document `test_smoke_imports.py` ↔ `test__init__*.py` relationship
**Based on**: MILIA Pipeline Project Structure v1.1.0 (127 existing test files)
**References**: Industry CI/CD best practices (GitHub Actions, pytest), ML testing strategies (smoke → unit → integration → E2E), Python testing patterns (contract tests, regression tests, thread safety verification)
