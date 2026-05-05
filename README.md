# MILIA (Machine Intelligent Learning Interface Assistant)

**Molecular graph processing and machine learning framework for computational chemistry**

[![CI](https://github.com/shahram-boshra/MILIA/actions/workflows/ci.yml/badge.svg)](https://github.com/shahram-boshra/MILIA/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

---

MILIA (**M**achine **I**ntelligent **L**earning **I**nference **A**ssistant) is a production-ready, research-oriented Python framework for molecular data processing and graph-based machine learning. It provides a complete, no-code ML/DL workflow — from dataset curation, molecular graph transformation, and molecular descriptor computation through graph neural network training, hyperparameter optimization, and model deployment — requiring only YAML configuration to run. Every level of the pipeline (datasets, transformations, descriptors, models, training, deployment) is fully configurable without writing code, while remaining extensible via plugins for untouched areas of research. Built on PyTorch, PyTorch Geometric, and RDKit, MILIA is designed for researchers, educators, and teams who need a flexible, extensible platform across the full ML/DL stack.

MILIA fills a gap in the molecular ML ecosystem by unifying dataset handling, feature engineering, model training, and deployment into a single, extensible framework with plugin support. Where tools like PyTorch Geometric provide GNN building blocks and DeepChem offers pre-built models, MILIA provides a configuration-driven research workflow — supporting any PyG model architecture through dynamic introspection, any hardware from CPU to TPU, and any dataset through its zero-modification extension architecture. The framework currently ships with 10 dataset implementations spanning the VQM24 family (DFT, DMC, Wavefunction), the QM9 benchmark, the ANI family (ANI-1x, ANI-1ccx, ANI-2x) including coupled-cluster reference data, the rMD17 and xxMD reactive dynamics datasets, and the drug-discovery-oriented QDπ dataset.

> **🚀 New here?** If you have authenticated access to this repository and want the shortest reproducible path from zero to a running MILIA install — Docker pull, smoke test, one full walkthrough — see **[QUICKSTART.md](QUICKSTART.md)**. It is designed to be executable end-to-end in ≤30 minutes on a CPU-only laptop without contacting the authors.

## Key Features

### No-Code ML/DL Workflow
Run the entire machine learning pipeline — dataset curation, molecular graph transformations, molecular descriptor computation, model training, hyperparameter optimization, and prediction — through YAML configuration and CLI commands. No code required at any level. For research requiring capabilities beyond the built-in functionality, the plugin architecture allows code-level extension of datasets, transformations, descriptors, and models without modifying the core framework.

### Unlimited Model Flexibility
Access **every PyTorch Geometric model** (SchNet, DimeNet, GIN, GAT, and all others) simply by naming them in configuration — no model-level code. Define custom architectures from 10 built-in templates, compose multi-model ensembles with parallel, sequential, or hierarchical strategies — all through YAML configuration, no code required. For research beyond built-in capabilities, bring your own models through the model plugin system.

### Hardware Agnostic
Train on any device — CPU, CUDA GPU, Apple MPS, or TPU — with automatic device detection or explicit selection through configuration. Scale from single-device to distributed training with 4 strategies (DataParallel, DistributedDataParallel, FSDP, Horovod). Includes memory optimization (AMP, gradient checkpointing) and computation optimization.

### Molecular Descriptors
Select from 400+ molecular descriptors across 6 categories — Constitutional (35), Topological (350+), Electronic (8), Geometric (10), Drug-likeness (4), and Fragments (85) — entirely through YAML configuration. MILIA handles the full computation pipeline: multi-format molecular conversion via RDKit, atom-level feature extraction (degree, hybridization, chirality, Mulliken charges), bond-level properties (type, conjugation, stereo, length), automatic conformer generation for 3D descriptors, and result caching. Extend with custom descriptors through the plugin system for advanced research needs.

### Advanced Hyperparameter Optimization
Optuna-based optimization with 5 search algorithms (TPE, CMA-ES, Random, Grid, NSGA-II for multi-objective), 5 pruning strategies (Median, Hyperband, Percentile, Patient, Threshold), cross-validation integration, study persistence, and resume support. Includes Neural Architecture Search for GNNs (7 layer types, heterogeneous architectures) and HPO transfer learning with meta-feature extraction and warm-starting across studies.

### Production-Ready Deployment
Edge, cloud, and federated deployment strategies with model quantization and pruning. Production monitoring for drift detection and performance tracking. Checkpoint management with training state persistence, model loading, and fine-tuning via transfer learning.

### Extensible Graph Transformation System
30+ pre-registered PyG transforms with a 7-layer architecture: dynamic discovery, registry, validation (semantic + dataset-aware), composition with intelligent caching, configuration bridge, error recovery, and production metrics (Prometheus/DataDog export). Three validation levels (Strict/Standard/Lenient) and five validation scopes. Edge-attr-aware parameter injection prevents shape mismatch errors.

### Three-Tier Plugin Architecture
Extend descriptors, transformations, and models independently without modifying core code. Plugin discovery with YAML manifests, validation, and security controls. Ships with example plugins and user templates.

### Flexible Configuration System
Schema-validated YAML with Pydantic V2 (10 frozen BaseModel containers, 60+ accessor functions). Split-file `configs/` directory as the sole configuration source with deep merge, CLI override, and configuration migration. Explicit single-file paths are supported via `--config`. Each dataset type has colocated configuration files for self-contained setup.

### Wavefunction Preprocessing
Modular preprocessing for quantum chemistry data formats (MOLDEN, FCHK) with structural feature filtering per dataset type, wavefunction data extraction, and VQM24 support.

### Research-Ready
Experiment configuration with research API for transformation ablation studies, hyperparameter sweeps, and model comparison. Research-grade recommendations per dataset type (DFT precision, DMC uncertainty preservation, wavefunction orbital analysis). Benchmarking and validation reporting in text, JSON, and markdown formats.

### Educational Use
The no-code design, comprehensive CLI with interactive mode, and 12+ processing modes make MILIA suitable for teaching molecular ML concepts without requiring students to write model code. Configuration-driven workflows allow focusing on chemistry and ML concepts rather than software engineering.

## Installation

MILIA relies on heavy scientific packages (PyTorch, PyTorch Geometric, RDKit) that are best managed through conda to avoid binary dependency conflicts.

### Method 1: Docker (Recommended)

The fastest way to get MILIA running. A pre-built image is available on GitHub Container Registry:

> **Note (Private Repository):** MILIA's GHCR image is private. Before pulling, authenticate
> to GHCR using either [GitHub CLI](https://cli.github.com/) or a
> [Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
> with `read:packages` scope:
>
> ```bash
> # Option A: GitHub CLI (recommended — no PAT needed)
> gh auth login
> echo $(gh auth token) | docker login ghcr.io -u USERNAME --password-stdin
>
> # Option B: Personal Access Token with read:packages scope
> echo YOUR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
> ```

```bash
# Pull and run the pre-built image
docker pull ghcr.io/shahram-boshra/milia:latest
docker run -it ghcr.io/shahram-boshra/milia:latest
# → (shah_env) root@...:/app/milia#

# Verify MILIA works (inside the container)
pytest -m smoke --tb=short
milia --help
```

Or build locally from the Dockerfile:

```bash
git clone https://github.com/shahram-boshra/MILIA.git
cd MILIA
docker build -t milia .
docker run -it milia
# → (shah_env) root@...:/app/milia#
```

### Method 2: Conda (Without Docker)

```bash
# 1. Create and activate conda environment
conda create -n milia python=3.10
conda activate milia

# 2. Install core dependencies via conda-forge
conda install -c conda-forge numpy scipy pyyaml h5py pandas rdkit \
    matplotlib pydantic-settings ase torchmetrics hydra-core optuna \
    plotly scikit-learn pytorch cpuonly -c pytorch

# 3. Install PyTorch Geometric and extensions
conda install -c pyg torch-geometric
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv

# 4. Install MILIA in editable mode
git clone https://github.com/shahram-boshra/MILIA.git
cd MILIA
pip install -e .
```

For GPU support, replace `cpuonly` with the appropriate CUDA toolkit version (e.g., `pytorch-cuda=12.1 -c nvidia`).

For development (includes pytest and ruff):

```bash
pip install -e ".[dev]"
```

## Quick Start

### Command Line

```bash
# Process molecular data (auto-detects configs/ directory)
milia --process

# Train a GNN model
milia --train

# Train with hyperparameter optimization (via CLI flag)
milia --train --hpo

# Run predictions on new molecules
milia --predict --model-path ./checkpoints/best.pt \
      --test-path test_data/molecules.csv --preds-path ./predictions.csv

# Validate configuration without processing
milia --dry-run

# List available transforms and experimental setups
milia --list-transforms
milia --list-experimental-setups

# Generate statistics from existing data
milia --stats-only

# Explicit config path (equivalent to auto-detection when configs/ exists)
milia --config configs/ --process
```

### Programmatic API

```python
from milia_pipeline import create_cli_manager, setup_logging

# Setup
logger = setup_logging(log_level="INFO")
cli = create_cli_manager(logger=logger)
args = cli.parse_args(['--config', 'configs/', '--process'])

# Load and validate configuration
config = cli.load_and_merge_config(args)
cli.validate_args(args, config)
```

```python
from milia_pipeline.config import load_config
from milia_pipeline.handlers import create_handler
from milia_pipeline.datasets import miliaDataset

# Load configuration (split-file mode: merges all YAML files in configs/)
config = load_config('configs/')
handler = create_handler(
    dataset_type='DFT',
    config=config,
    logger=logger
)

# Build a PyTorch Geometric dataset
dataset = miliaDataset(
    root='./data',
    handler=handler,
    transform=my_transforms
)
```

## Trying MILIA — Reproducible Walkthrough

This section walks a reviewer through the shortest path from a fresh clone to a trained model and a prediction. Every command below has been validated end-to-end. Paths are relative throughout, so the walkthrough works identically on any machine — Linux, macOS, WSL, or inside the Docker container.

### Prerequisites

1. **MILIA installed** — see [Installation](#installation) above. Method 1 (Docker) is the fastest route for a one-shot evaluation; Method 2 (Conda + `pip install -e .`) is recommended if you intend to inspect or modify source.

2. **`working_root_dir` set in your configuration** — this is the only path you must configure. It tells MILIA where to download datasets, write processed graphs, and save checkpoints. There is **no implicit default** — the framework asks you to choose deliberately.

   Open `configs/main.yaml` and set:

   ```yaml
   global_paths:
     working_root_dir: ~/Chem_Data/Milia_PyG_Dataset    # or any directory you prefer
   ```

   The `~` is expanded to your home directory at runtime, so the same line works for any user. Inside the Docker image the value is preset to `/root/Chem_Data/Milia_PyG_Dataset` and you can leave it as-is.

3. **A laptop-friendly configuration** — the shipped `configs/models.yaml` is preconfigured for low-resource execution (small batch size, few epochs, small ensembles). A reviewer with a CPU-only laptop can run the walkthrough below without a GPU.

### Step-by-step

Run each command from the repository root (or, in Docker, from `/app/milia`). Every command auto-detects the `configs/` directory.

```bash
# 1. Sanity-check the install (under two minutes on CPU)
pytest -m smoke --tb=short

# 2. Validate the configuration without doing any work
milia --dry-run

# 3. Process the dataset (downloads if needed, writes PyG graphs under working_root_dir)
milia --process

# 4. Inspect the processed dataset
milia --stats-only

# 5. Train a model
#    Best checkpoint written to {working_root_dir}/checkpoints/best.pt
milia --train

# 6. (Optional) Train with hyperparameter optimization instead of step 5
milia --train --hpo

# 7. Run prediction on the sample molecules shipped with the repo
milia --predict \
    --model-path ./checkpoints/best.pt \
    --test-path test_data/molecules.csv \
    --preds-path ./predictions.csv
```

After step 5 or 6, your `{working_root_dir}/checkpoints/` directory contains a `best.pt` checkpoint. Step 7 reads that checkpoint and writes per-molecule predictions to `./predictions.csv` for the five sample molecules in `test_data/molecules.csv` (ethanol, acetic acid, benzene, isopropanol, triethylamine — all common organic molecules in SMILES format).

### Where the outputs live

Every artifact MILIA produces lands under `working_root_dir`. With the example value `~/Chem_Data/Milia_PyG_Dataset` this means:

| Artifact | Location |
|---|---|
| Processed PyG graphs | `~/Chem_Data/Milia_PyG_Dataset/processed/` |
| Best model checkpoint | `~/Chem_Data/Milia_PyG_Dataset/checkpoints/best.pt` |
| Per-epoch checkpoints | `~/Chem_Data/Milia_PyG_Dataset/checkpoints/epoch=*.pt` |
| HPO best parameters | `~/Chem_Data/Milia_PyG_Dataset/hpo_output/best_params.json` |
| Training plots | `~/Chem_Data/Milia_PyG_Dataset/training_plots/` |

The CLI resolves checkpoint paths against `working_root_dir` automatically, so commands that reference `./checkpoints/best.pt` work regardless of which directory you run them from.

### Introspection commands

These are read-only and useful for orientation:

```bash
milia --list-transforms              # 30+ pre-registered PyG transforms + plugin transforms
milia --list-experimental-setups     # Available research/experiment configurations
milia --help                         # Full CLI reference
```

### Sample data shipped with the repo

The `test_data/` directory contains `molecules.csv` — a small CSV with five common organic molecules in SMILES format, ready for use with `milia --predict` (step 7 above). This is the minimal sample needed to verify end-to-end inference; for production use, supply your own input file in the same format (`smiles,molecule_id` header, one molecule per row).

## Architecture

MILIA is organized into 11 core modules and a split configuration system:

| Module | Files | Purpose |
|--------|-------|---------|
| `config/` | 7 files, ~22K lines | Multi-layered configuration management with Pydantic V2 validation, YAML splitting (`configs/` directory with deep merge), thread-safe caching, schema migration, and 60+ accessor functions |
| `molecules/` | 7 files, ~14K lines | Molecular conversion (RDKit → PyG), structural feature extraction, property enrichment, filtering with transform compatibility, and registry-integrated validation |
| `transformations/` | 4 files, ~16K lines | 7-layer graph transformation system: dynamic discovery, registry, semantic validation, composition with caching, configuration bridge, error recovery, and production metrics |
| `datasets/` | 10 files | Registry-based PyTorch Geometric datasets with Protocol contracts, compile-time validation, and 10 concrete implementations spanning the VQM24 family (DFT, DMC, Wavefunction), QM9, the ANI family (ANI-1x, ANI-1ccx, ANI-2x), rMD17, xxMD, and QDπ — see the [Datasets](#datasets) section |
| `handlers/` | 15+ files | 10 dataset handler types (DFT, DMC, Wavefunction, QM9, ANI-1x, ANI-1ccx, rMD17, ANI-2x, XXMD, QDPi) with transform integration and lazy loading |
| `preprocessing/` | 8+ files | Modular wavefunction preprocessing (MOLDEN, FCHK), data refinement, and VQM24 support |
| `descriptors/` | 6+ files | 400+ molecular descriptors across 6 categories with thread-safe singleton registry, caching, conformer generation, and plugin support |
| `models/` | 25+ files | Full ML lifecycle: registry with dynamic PyG introspection, factory, trainer with callbacks, post-training inference, architecture builder (10 templates), model composer, acceleration (CPU/GPU/MPS/TPU + DP/DDP/FSDP), deployment (edge/cloud/federated), monitoring, and model plugins |
| `models/hpo/` | 12 files | Hyperparameter optimization: Optuna backend, 5 search algorithms, 5 pruners, neural architecture search, transfer learning with warm-starting, and study analysis |
| `cli_manager` | 1 file, ~3.8K lines | 12 argument groups, 12+ processing modes, interactive mode, and post-training prediction arguments |
| `exceptions` | 1 file | Comprehensive exception hierarchy with registry-based dataset-specific errors |

**Supporting directories:**

| Directory | Purpose |
|-----------|---------|
| `configs/` | Split YAML configuration with per-dataset files (10 datasets) and deep-merge architecture |
| `plugins/` | Plugin storage: descriptor plugins, transformation plugins, model plugins — with YAML manifests and user templates |
| `tests/` | 127 unit and integration tests across all modules |

## Datasets

MILIA ships with **10 production-ready dataset implementations** covering the major quantum-chemistry and molecular-dynamics benchmarks used in modern molecular machine learning. Selecting a dataset is a one-line change in `configs/main.yaml` (`dataset_type: <Registry key>`) — no code, no rebuild, no glue.

### Shipped datasets

| Registry key | Class | Source / level of theory | Coordinates | Energy | Strategy |
|---|---|---|---|---|---|
| `DFT` | `DFTDataset` | VQM24 — ωB97X-D3/cc-pVDZ DFT properties for ~785k conformers (Khan et al., *Sci. Data* 12, 1551, 2025) | Å | Hartree | identifier + coordinate (InChI/SMILES) |
| `DMC` | `DMCDataset` | VQM24 — DMC@PBE0/ccECP-cc-pVQZ energies with statistical uncertainties for 10,793 constitutional isomers (same reference) | Å | Hartree | identifier + coordinate (InChI/SMILES) |
| `Wavefunction` | `WavefunctionDataset` | VQM24 wavefunction files (MOLDEN/FCHK) — molecular orbitals, HOMO–LUMO gap, MO energies | Bohr → Å | eV | coordinate-based (charge inferred from `n_electrons`) |
| `QM9` | `QM9Dataset` | QM9 — 133,885 small organic molecules (CHONF) at B3LYP/6-31G(2df,p) (Ramakrishnan et al., *Sci. Data* 1, 140022, 2014) | Å | Hartree | identifier + coordinate (InChI → SMILES) |
| `ANI1x` | `ANI1xDataset` | ANI-1x — ~5M ωB97x/6-31G* DFT conformations of CHNO molecules from active learning (Smith et al., *Sci. Data* 7, 134, 2020) | Å | Hartree | coordinate-based |
| `ANI1ccx` | `ANI1ccxDataset` | ANI-1ccx — ~500k CCSD(T)/CBS energies on a curated subset of ANI-1x (same reference) | Å | Hartree | coordinate-based |
| `ANI2x` | `ANI2xDataset` | ANI-2x — DFT conformations at ωB97X/6-31G(d) extended to S, F, Cl (Devereux et al., *J. Chem. Theory Comput.* 16, 4192, 2020) | Å | Hartree | coordinate-based |
| `RMD17` | `RMD17Dataset` | revised MD17 — ~100k PBE/def2-SVP conformations for each of 10 small molecules with very tight SCF and dense grids (Christensen & von Lilienfeld, *MLST* 1, 045018, 2020) | Å | Hartree (converted from kcal/mol) | coordinate-based |
| `XXMD` | `XXMDDataset` | xxMD-DFT — non-adiabatic dynamics trajectories for 4 photo-active molecules at the M06 level, including transition states and conical-intersection regions (Pengmei et al., *Sci. Data* 11, 222, 2024) | Å | Hartree (converted from eV) | coordinate-based |
| `QDPi` | `QDPiDataset` | QDπ — ~1.6M ωB97M-D3(BJ)/def2-TZVPPD structures for drug-like neutral *and* charged species across 13 elements (Zeng et al., *Sci. Data* 12, 693, 2025) | Å | Hartree (converted from eV) | coordinate-based (charge-aware) |

The `DFT`, `DMC`, and `Wavefunction` entries are the three deliverables of the **Vector-QM24 (VQM24)** dataset (Zenodo: [10.5281/zenodo.11164951](https://doi.org/10.5281/zenodo.11164951)) — DFT geometries and properties, DMC reference energies, and quantum-mechanical wavefunctions respectively — for which MILIA provides VQM24-aware vibrational refinement and MOLDEN/FCHK readers out of the box.

#### Per-dataset capability matrix

Each shipped dataset declares an immutable `DatasetFeatures` record (a Pydantic V2 frozen dataclass) that drives feature-aware code paths in MILIA — descriptor selection, transform compatibility validation, atomization-energy bookkeeping, and orbital-property extraction — without any consumer needing to hard-code dataset names. The five flags below are the most consequential for downstream model targets and are queried at runtime via `_get_dataset_feature(dataset_type, feature_name)`:

| Dataset | Vibrational analysis | Uncertainty handling | Atomization energy | Orbital analysis | HOMO–LUMO gap |
|---|:---:|:---:|:---:|:---:|:---:|
| `DFT` | ✓ | ✗ | ✓ | ✗ | ✗ |
| `DMC` | ✗ | ✓ | ✗ | ✗ | ✗ |
| `Wavefunction` | ✗ | ✗ | ✗ | ✓ | ✓ |
| `QM9` | ✓ | ✗ | ✓ | ✗ | ✓ |
| `ANI1x` | ✗ | ✗ | ✓ | ✗ | ✗ |
| `ANI1ccx` | ✗ | ✗ | ✓ | ✗ | ✗ |
| `ANI2x` | ✗ | ✗ | ✓ | ✗ | ✗ |
| `RMD17` | ✗ | ✗ | ✓ | ✗ | ✗ |
| `XXMD` | ✗ | ✗ | ✓ | ✗ | ✗ |
| `QDPi` | ✗ | ✗ | ✓ | ✗ | ✗ |

These five columns are a subset of the eight flags carried by `DatasetFeatures`; the other three (`rotational_constants`, `frequency_analysis`, `mo_energies`) gate finer-grained behaviour and are queried programmatically rather than displayed here. The matrix is not a documentation artefact — it is the same registry data that MILIA consults to decide, for example, whether a transform that requires vibrational modes is admissible against a given dataset, or whether atomization-energy targets are derivable. Setting `vibrational_analysis: True` on a new dataset's `DatasetFeatures` is therefore sufficient to opt into vibrational-aware processing across the pipeline; no other module needs editing.

### Adding a dataset

Adding a new dataset to MILIA is a three-file operation against existing extension points — no core file is modified, and no registration list, switch statement, or import is touched anywhere else in the codebase:

| Touch-point | What you create | How MILIA picks it up |
|---|---|---|
| `milia_pipeline/datasets/implementations/<name>.py` | A `BaseDataset` subclass decorated with `@register` | Dynamic discovery in `datasets/implementations/__init__.py` auto-imports every `.py` file in the directory and triggers the decorator at import time |
| `milia_pipeline/handlers/implementations/<name>.py` | A `DatasetHandler` subclass decorated with `@register_handler` | Dynamic discovery in `handlers/implementations/__init__.py` auto-imports the file, finds classes ending in `DatasetHandler` or `Handler`, and triggers the decorator |
| `configs/datasets/<name>.yaml` | Colocated dataset config containing `<name>_config`, `data_config.property_selection.<KEY>`, and `property_availability.<KEY>` | YAML splitting merges `datasets/*.yaml` automatically — no edit to `main.yaml` beyond setting `dataset_type` to the new key |

The architecture enforcing this is **Protocol + ABC + explicit registry**, applied identically to datasets and handlers:

- **`@register` / `@register_handler` decorators** wire the new class into thread-safe (`RLock`-protected) registries at import time, with no central registration list to update.
- **`DatasetHandlerProtocol`** (11 runtime-checkable methods) and the `BaseDataset` ABC (with `__init_subclass__` validation) catch contract violations at *import time*, not at runtime — a missing required method fails fast before any data is loaded.
- **`DatasetMetadata`, `DatasetSchema`, `DatasetFeatures`** are immutable Pydantic V2 frozen dataclasses, so the new dataset's metadata is validated structurally before registration succeeds.
- **Feature flags** (`vibrational_analysis`, `uncertainty_handling`, `atomization_energy`, `orbital_analysis`, `homo_lumo_gap`, and three others) declared in the `BaseDataset` subclass make the new dataset visible to feature-aware code paths in `milia_dataset.py` and the transform validator without any of those modules learning the new dataset's name.

Once the three files are in place, the new dataset is a first-class citizen of every part of the pipeline — CLI (`milia --process`, `--train`, `--predict`), YAML configuration, hyperparameter optimization, transform validation, descriptors, and prediction — selectable by setting `dataset_type: <key>` in `configs/main.yaml`. The 10 shipped datasets follow this exact pattern; the registry has no knowledge that they are "built-in" rather than user-added.

## Testing

```bash
# Run the full test suite
pytest

# Run specific test categories
pytest -m "not slow"         # Skip slow tests
pytest -m integration        # Integration tests only
pytest -m gpu                # GPU-specific tests only
```

The test suite includes 127 unit and integration tests covering all core modules.

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started, coding standards, and the pull request process.

## Citation

If you use MILIA in your research, please cite it. See [CITATION.cff](CITATION.cff) for the preferred citation format.

## License

MILIA is released under the [MIT License](LICENSE).

## Authors

- **Asadollah (Shahram) Boshra** — [a.boshra@gmail.com](mailto:a.boshra@gmail.com)
- **Ilia Boshra** — [ilia.boshra@gmail.com](mailto:ilia.boshra@gmail.com)

## Links

- **Repository**: [github.com/shahram-boshra/MILIA](https://github.com/shahram-boshra/MILIA)
- **Issues**: [github.com/shahram-boshra/MILIA/issues](https://github.com/shahram-boshra/MILIA/issues)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
