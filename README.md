# MILIA (Machine Intelligent Learning Inference Assistant)

**Molecular graph processing and machine learning framework for computational chemistry**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

---

MILIA (**M**achine **I**ntelligent **L**earning **I**nference **A**ssistant) is a production-ready, research-oriented Python framework for molecular data processing and graph-based machine learning. It provides a complete, no-code ML/DL workflow — from dataset curation, molecular graph transformation, and molecular descriptor computation through graph neural network training, hyperparameter optimization, and model deployment — requiring only YAML configuration to run. Every level of the pipeline (datasets, transformations, descriptors, models, training, deployment) is fully configurable without writing code, while remaining extensible via plugins for untouched areas of research. Built on PyTorch, PyTorch Geometric, and RDKit, MILIA is designed for researchers, educators, and teams who need a flexible, extensible platform across the full ML/DL stack.

MILIA fills a gap in the molecular ML ecosystem by unifying dataset handling, feature engineering, model training, and deployment into a single, extensible framework with plugin support. Where tools like PyTorch Geometric provide GNN building blocks and DeepChem offers pre-built models, MILIA provides a configuration-driven research workflow — supporting any PyG model architecture through dynamic introspection, any hardware from CPU to TPU, and any dataset through its zero-modification extension architecture. The framework currently ships with 10 dataset handlers spanning quantum chemistry (DFT, QM9, ANI), quantum Monte Carlo (DMC), wavefunction analysis, molecular dynamics (rMD17), coupled cluster methods (ANI-1ccx), and more — and is readily extensible to new domains including pharmaceutical chemistry, materials science, and beyond.

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
Schema-validated YAML with Pydantic V2 (10 frozen BaseModel containers, 60+ accessor functions). Supports single-file (`config.yaml`) or split-file (`configs/`) modes with deep merge, CLI override, and configuration migration. Each dataset type has colocated configuration files for self-contained setup.

### Wavefunction Preprocessing
Modular preprocessing for quantum chemistry data formats (MOLDEN, FCHK) with structural feature filtering per dataset type, wavefunction data extraction, and VQM24 support.

### Research-Ready
Experiment configuration with research API for transformation ablation studies, hyperparameter sweeps, and model comparison. Research-grade recommendations per dataset type (DFT precision, DMC uncertainty preservation, wavefunction orbital analysis). Benchmarking and validation reporting in text, JSON, and markdown formats.

### Educational Use
The no-code design, comprehensive CLI with interactive mode, and 12+ processing modes make MILIA suitable for teaching molecular ML concepts without requiring students to write model code. Configuration-driven workflows allow focusing on chemistry and ML concepts rather than software engineering.

## Installation

MILIA relies on heavy scientific packages (PyTorch, PyTorch Geometric, RDKit) that are best managed through conda to avoid binary dependency conflicts.

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
# Process molecular data
milia --config config.yaml --process

# Train a GNN model
milia --config config.yaml --train

# Train with hyperparameter optimization
milia --config config.yaml --train  # (set models.hpo.enabled: true in config)

# Run predictions on new molecules
milia --predict --model-path ./checkpoints/best_model.pt \
      --test-path ./molecules.csv --preds-path ./predictions.csv

# Validate configuration without processing
milia --config config.yaml --dry-run

# List available transforms and experimental setups
milia --config config.yaml --list-transforms
milia --config config.yaml --list-experimental-setups

# Generate statistics from existing data
milia --config config.yaml --stats-only
```

### Programmatic API

```python
from milia_pipeline import create_cli_manager, setup_logging

# Setup
logger = setup_logging(log_level="INFO")
cli = create_cli_manager(logger=logger)
args = cli.parse_args(['--config', 'config.yaml', '--process'])

# Load and validate configuration
config = cli.load_and_merge_config(args)
cli.validate_args(args, config)
```

```python
from milia_pipeline.config import load_config
from milia_pipeline.handlers import create_handler
from milia_pipeline.datasets import miliaDataset

# Load configuration and create a dataset handler
config = load_config('config.yaml')
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

## Architecture

MILIA is organized into 11 core modules and a split configuration system:

| Module | Files | Purpose |
|--------|-------|---------|
| `config/` | 7 files, ~22K lines | Multi-layered configuration management with Pydantic V2 validation, YAML splitting (single-file or `configs/` directory), thread-safe caching, schema migration, and 60+ accessor functions |
| `molecules/` | 7 files, ~14K lines | Molecular conversion (RDKit → PyG), structural feature extraction, property enrichment, filtering with transform compatibility, and registry-integrated validation |
| `transformations/` | 4 files, ~16K lines | 7-layer graph transformation system: dynamic discovery, registry, semantic validation, composition with caching, configuration bridge, error recovery, and production metrics |
| `datasets/` | 10 files | Registry-based PyTorch Geometric datasets with Protocol contracts, compile-time validation, and 5 concrete implementations (DFT, DMC, Wavefunction, XXMD, QDPi) |
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

## Links

- **Repository**: [github.com/shahram-boshra/MILIA](https://github.com/shahram-boshra/MILIA)
- **Issues**: [github.com/shahram-boshra/MILIA/issues](https://github.com/shahram-boshra/MILIA/issues)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
