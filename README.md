# MILIA (Machine Intelligent Learning Interface Assistant)

**Molecular graph processing and machine learning framework for computational chemistry**

[![CI](https://github.com/shahram-boshra/MILIA/actions/workflows/ci.yml/badge.svg)](https://github.com/shahram-boshra/MILIA/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Software Heritage](https://archive.softwareheritage.org/badge/origin/https://github.com/shahram-boshra/MILIA/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/shahram-boshra/MILIA)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21854751.svg)](https://doi.org/10.5281/zenodo.21854751)

---

MILIA (**M**achine **I**ntelligent **L**earning **I**nterface **A**ssistant) is a production-ready, research-oriented Python framework for molecular data processing and graph-based machine learning. It provides a complete, no-code ML/DL workflow — from dataset curation, molecular graph transformation, and molecular descriptor computation through graph neural network training, hyperparameter optimization, and model deployment — requiring only YAML configuration to run. Every level of the pipeline (datasets, transformations, descriptors, models, training, deployment) is fully configurable without writing code, while remaining extensible via plugins for untouched areas of research. Built on PyTorch, PyTorch Geometric, and RDKit, MILIA is designed for researchers, educators, and teams who need a flexible, extensible platform across the full ML/DL stack.

MILIA fills a gap in the molecular ML ecosystem by unifying dataset handling, feature engineering, model training, and deployment into a single, extensible framework with plugin support. Where tools like PyTorch Geometric provide GNN building blocks and DeepChem offers pre-built models, MILIA provides a configuration-driven research workflow — supporting any PyG model architecture through dynamic introspection, any hardware from CPU to TPU, and any dataset through its zero-modification extension architecture. The framework currently ships with 11 dataset implementations spanning the VQM24 family (DFT, DMC, Wavefunction), the QM9 benchmark, the ANI family (ANI-1x, ANI-1ccx, ANI-2x) including coupled-cluster reference data, the rMD17 and xxMD reactive dynamics datasets, the drug-discovery-oriented QDπ dataset, and the QM40 dataset of neutral drug-like ZINC molecules.

> **🚀 New here?** For the shortest reproducible path from zero to a running MILIA install — Docker pull, smoke test, one full walkthrough — see **[QUICKSTART.md](QUICKSTART.md)**. It is designed to be executable end-to-end in ≤30 minutes on a CPU-only laptop without contacting the author.

## Key Features

### No-Code ML/DL Workflow
Run the entire machine learning pipeline — dataset curation, molecular graph transformations, molecular descriptor computation, model training, hyperparameter optimization, and prediction — through YAML configuration and CLI commands. No code required at any level. For research requiring capabilities beyond the built-in functionality, the plugin architecture allows code-level extension of datasets, transformations, descriptors, and models without modifying the core framework.

### Unlimited Model Flexibility
Access **every PyTorch Geometric model** (SchNet, DimeNet, GIN, GAT, and all others) simply by naming them in configuration — no model-level code. Use one model, or compose multi-model ensembles of PyG and plugin models with parallel, sequential, or hierarchical strategies, for every task type — all through YAML configuration, no code required. For research beyond built-in capabilities, bring your own models through the model plugin system.

### Hardware Agnostic
Train on any device — CPU, CUDA GPU, Apple MPS, or TPU — with automatic device detection or explicit selection through configuration. One `models.acceleration` section turns on mixed precision (bf16/fp16 AMP), `torch.compile` with dynamic shapes, gradient checkpointing, and multi-process data loading for every training and HPO run (see [Acceleration](#acceleration)). Distributed strategies (DataParallel, DistributedDataParallel, FSDP) are available through the Python API; the CLI and HPO paths train single-process.

### Molecular Descriptors
Select from 3,000+ molecular descriptors across 6 categories — a ~492-descriptor RDKit baseline (Constitutional, Topological, Electronic, Geometric, Drug-likeness, Fragments) plus the first-party `addcore_3d` plugin adding **1,019 ADD-CORE RDKit 3D descriptors** (WHIM, GETAWAY, RDF, 3D-MoRSE, Autocorr3D, USR, USRCAT, MQN, and oxidation-number aggregates) the `constitutional_property` plugin adding **20 constitutional-ext + property/physicochemical descriptors** (McGowan/vdW-ABC volume, polarizability, CarbonTypes, framework/complexity, and Lipinski/Ghose/Veber/Egan rule filters), the `topological_connectivity` plugin adding **107 topological/connectivity indices** (Wiener, Zagreb, ABC, eccentric-connectivity, Gálvez topological charge, molecular-distance-edge, Schultz, and the Kier-Hall Chi family), and the `walk_path_information` plugin adding **86 walk/path + information-content descriptors** (molecular walk/self-returning-walk counts, path counts, detour index, neighborhood information-content indices, and vertex-adjacency information), the `matrix_spectral` plugin adding **177 eigenvalue/matrix-spectral descriptors** (adjacency/distance/detour/Barysz matrix spectra, Burden BCUT eigenvalues, and Randić molecular ID), and the `autocorrelation_2d` plugin adding **606 2D topological autocorrelation descriptors** (Moreau-Broto ATS/AATS/ATSC/AATSC, Moran MATS, and Geary GATS over 11-12 atomic-property weightings × lags 0-8), the `estate_atomtype` plugin adding **316 Kier-Hall E-State atom-type descriptors** (count/sum/max/min of E-State values over 79 atom types), and the `cats2d` plugin adding **150 CATS2D topological pharmacophore-pair descriptors** (15 potential-pharmacophore-point pairs × 10 topological-distance bins), and the `eta` plugin adding **45 Extended Topochemical Atom (ETA) indices** (Roy & Ghosh core-count/shape/VEM/composite/epsilon/psi descriptors), and the `cpsa_geometric_3d` plugin adding **50 charged-partial-surface-area, gravitational and geometrical 3D descriptors** (CPSA/GRAV*/Geom* on the ETKDG conformer), and the `eht_electronic` plugin adding **25 extended-Hückel conceptual-DFT reactivity descriptors** (HOMO/LUMO/gap, electronegativity/hardness/electrophilicity, Fukui — EHT-level) — entirely through YAML configuration. MILIA handles the full computation pipeline: multi-format molecular conversion via RDKit, atom-level feature extraction (degree, hybridization, chirality, the dataset's atomic charges: Mulliken, Hirshfeld, …), bond-level properties (type, conjugation, stereo, length), automatic conformer generation for 3D descriptors (deterministic ETKDG seed), a compute-once-per-molecule block-cache for vector descriptors, and result caching. Extend with custom descriptors through the plugin system for advanced research needs. Descriptors are rolled out in versioned paces (see `CHANGELOG.md`).

### Advanced Hyperparameter Optimization
Optuna-based optimization with 7 search algorithms (TPE, Gaussian process, CMA-ES, Random, Quasi-Monte Carlo, Grid, NSGA-II), 7 pruning strategies (Median, Percentile, Hyperband, Successive Halving, Threshold, Patient, and Wilcoxon for cross-validation), single- or multi-objective studies (for example accuracy against model size, with the Pareto front saved), cross-validation integration, study persistence, and resume support. Parallel HPO: worker processes or containers share one study on PostgreSQL (or a journal file on one host), with trial heartbeats, a study-wide trial budget, per-worker seeds and a Docker Compose profile. Includes Neural Architecture Search for GNNs (7 layer types, heterogeneous architectures) and HPO transfer learning with meta-feature extraction and warm-starting across studies.

### Production-Ready Deployment
Cloud, container, local and federated deployment strategies with model quantization and pruning; trained checkpoints are served over HTTP (see [Serving a trained model](#serving-a-trained-model)). The `mobile` and `iot` edge targets are deprecated and will be removed in 2.0.0. Production monitoring for drift detection and performance tracking. Checkpoint management with training state persistence, model loading, and fine-tuning via transfer learning.

### Extensible Graph Transformation System
130+ discoverable transforms — the full PyG transform set plus **55 novel graph transforms across 9 first-party plugin families** that ship ready-to-deploy and configurable from YAML: **graph products** (Cartesian, Tensor, Strong, Lexicographic, Rooted, Corona), **combinatorial & dual operators** (line/Levi graphs, quotient, transitive closure/reduction, bipartite projection, graph dual), **curvature-based rewiring** (Forman–Ricci, effective-resistance, spectral-gap, resistance-curvature), **topological lifts** (hypergraph clique/star expansion, persistent-homology features), **spectral & signal** transforms, **graph reduction/coarsening**, **structural padding** for static batching, **non-spatial augmentation**, and **latent-structural encodings** (motif compression, structural-role, anonymous-walk). Built on a 7-layer architecture: dynamic discovery, registry, validation (semantic + dataset-aware, precondition-aware), composition with intelligent caching, configuration bridge, error recovery, and production metrics (Prometheus/DataDog export). Three validation levels (Strict/Standard/Lenient) and five validation scopes. Edge-attr-aware parameter injection prevents shape mismatch errors.

### Kind-Container Plugin Architecture
Extend descriptors, transformations, and models independently without modifying core code. Plugins are organised into per-kind containers (`plugins/transformations/`, `plugins/descriptors/`, `plugins/models/`), each with a `user_template/` scaffold and config-driven discovery via `plugin_paths` (in-tree or external, upgrade-safe). YAML manifests, with validation of manifests, versions and dependencies. Plugins are trusted code: loading a plugin runs it, so install only plugins you trust. Ships with example plugins and user templates.

### Flexible Configuration System
Schema-validated YAML with Pydantic V2 (10 frozen BaseModel containers, 60+ accessor functions). Split-file `configs/` directory as the sole configuration source with deep merge, CLI override, and configuration migration. Explicit single-file paths are supported via `--config`. Each dataset type has colocated configuration files for self-contained setup.

### Wavefunction Preprocessing
Modular preprocessing for quantum chemistry data formats (MOLDEN, FCHK) with structural feature filtering per dataset type, wavefunction data extraction, and VQM24 support.

### Research-Ready
Experiment configuration with research API for transformation ablation studies, hyperparameter sweeps, and model comparison. Research-grade recommendations per dataset type (DFT precision, DMC uncertainty preservation, wavefunction orbital analysis). Benchmarking and validation reporting in text, JSON, and markdown formats.

### Educational Use
The no-code design, comprehensive CLI with interactive mode, and 12+ processing modes make MILIA suitable for teaching molecular ML concepts without requiring students to write model code. Configuration-driven workflows allow focusing on chemistry and ML concepts rather than software engineering.

## Installation

MILIA's dependencies (PyTorch, PyTorch Geometric + compiled companions, RDKit, …) are managed with [uv](https://docs.astral.sh/uv/) from a committed lockfile (`uv.lock`) for fully reproducible installs. A pre-built Docker image is also published.

### Method 1: Docker (Recommended)

The fastest way to get MILIA running. A pre-built image is publicly available on GitHub Container Registry — no authentication required:

```bash
# Pull the pre-built CPU image (no authentication required)
docker pull ghcr.io/shahram-boshra/milia:latest

# The image's entry point IS the `milia` CLI — pass CLI args directly:
docker run --rm ghcr.io/shahram-boshra/milia:latest --help

# Process a dataset (runs non-root; mount your data read-write at /data):
docker run --rm --user "$(id -u):$(id -g)" \
  -v ~/Chem_Data/Milia_PyG_Dataset:/data:rw \
  ghcr.io/shahram-boshra/milia:latest --root-dir /data
```

Or build locally from the Dockerfile:

```bash
git clone https://github.com/shahram-boshra/MILIA.git
cd MILIA
docker build -t milia:cpu .                    # CPU (default); GPU: see "GPU images" below
docker run --rm milia:cpu --help
```

> The published image is a lean production runtime (no test tools). The smoke suite runs in CI (and locally via `uv run --extra cpu --extra dev pytest -m smoke`), not inside the runtime image.

#### GPU images (CUDA 13.0 and 12.6)

Host prerequisites: an NVIDIA GPU, an NVIDIA driver — **≥ 580** for `cu130`, **≥ 525.60.13** for `cu126` (the CUDA 13.x and 12.x minimums for minor-version compatibility; a current driver is recommended) — and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The CUDA runtime ships inside the image's PyTorch/PyG wheels — no CUDA toolkit is needed on the host.

```bash
docker pull ghcr.io/shahram-boshra/milia:cu130
docker run --rm --gpus all ghcr.io/shahram-boshra/milia:cu130 --help

# Confirm the GPU is visible inside the container:
docker run --rm --gpus all --entrypoint python ghcr.io/shahram-boshra/milia:cu130 \
  -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
```

| Tag | Variant |
|---|---|
| `latest`, `cpu`, `X.Y.Z`, `X.Y` | CPU (`latest` is always the CPU image) |
| `cu130`, `X.Y.Z-cu130`, `X.Y-cu130` | CUDA 13.0 — Turing and newer GPUs (CUDA 13.0 dropped Maxwell, Pascal and Volta) |
| `cu126`, `X.Y.Z-cu126`, `X.Y-cu126` | CUDA 12.6 — for Maxwell, Pascal and Volta GPUs, or drivers older than 580 |
| `cpu-<sha>`, `cu130-<sha>`, `cu126-<sha>` | Immutable per-commit builds |

`cu124` tags (CUDA 12.4, torch 2.4) are no longer updated; their last build is from the release before the move to torch 2.12.

Images are `linux/amd64`. **Validation level:** every published GPU image is built and verified on a CPU CI runner (PyTorch reports the variant's CUDA version; MILIA and the PyG kernels import; `torch.compile` runs) and smoke-tested through the CLI; execution on a GPU is verified in CI only when a GPU runner is configured — run the check above on your host.

To build a GPU image locally: `docker build --build-arg ACCEL=cu130 --build-arg NVIDIA_VISIBLE_DEVICES=all -t milia:cu130 .`

#### Parallel HPO with Docker Compose (PostgreSQL)

`compose.yaml` (profile `hpo`) runs one HPO study shared by several worker containers, stored in PostgreSQL 18,
with [optuna-dashboard](https://github.com/optuna/optuna-dashboard) on `http://127.0.0.1:8080`. One command runs the
whole lifecycle, each step starting when the previous one succeeded: dataset processing → study creation
(`--hpo-init`) → two workers (`--hpo-worker 0/1`, trials only) → results and final model, written once
(`--hpo-finalize`).

```bash
# 1. Secrets (files, never environment variables; ./secrets/ is git- and docker-ignored)
(
  umask 077 && mkdir -p secrets
  openssl rand -hex 24 > secrets/postgres_password      # hex: safe inside a URL
  pw="$(cat secrets/postgres_password)"
  printf 'postgresql+psycopg://milia:%s@postgres:5432/milia_hpo' "$pw"  > secrets/hpo_storage_url  # pragma: allowlist secret
  printf 'postgresql+psycopg2://milia:%s@postgres:5432/milia_hpo' "$pw" > secrets/dashboard_storage_url  # pragma: allowlist secret
)

# 2. A configs directory for the containers: a copy of configs/ with
#      main.yaml    global_paths.working_root_dir: /data
#      models.yaml  models.hpo.study.storage: null
#                   models.hpo.study.storage_options: {kind: rdb, url_file: /run/secrets/hpo_storage_url,
#                                                      heartbeat_interval: 60}
mkdir -p ~/milia-hpo/data ~/milia-hpo/db && cp -r configs ~/milia-hpo/configs   # then edit the two files

# 3. Run (all containers run as your user, so they can read the 0600 secrets and write the directories)
cat > .env <<EOF
MILIA_UID=$(id -u)
MILIA_GID=$(id -g)
MILIA_DATA_DIR=$HOME/milia-hpo/data
MILIA_CONFIG_DIR=$HOME/milia-hpo/configs
MILIA_DB_DIR=$HOME/milia-hpo/db
EOF
docker compose --profile hpo up -d
docker compose --profile hpo logs -f hpo-finalize    # results + final model under ~/milia-hpo/data

# 4. Stop (the study stays in ~/milia-hpo/db)
docker compose --profile hpo down
```

GPU host: `docker compose -f compose.yaml -f compose.gpu.yaml --profile hpo up -d` gives each worker its own GPU
(`cu130` image). PostgreSQL is reachable only on the project's network (no published port); a database on another
host should use `sslmode=verify-full` in the URL. More workers: add `hpo-worker-N` services (see `compose.yaml`).
Each worker uses 1 PyTorch CPU thread by default (`OMP_NUM_THREADS`, as `torchrun` does for several processes); set
`MILIA_WORKER_THREADS` in `.env` to at most `floor(CPUs / workers)`. A worker logs its threads, CPUs, GPUs and memory,
and stops before loading a processed dataset larger than the memory available to it.

**Heartbeat and grace period (`rdb` storage).** A background thread in each worker writes the running trial's heartbeat
to the database every `heartbeat_interval` seconds. When another worker asks for a trial, Optuna marks as `FAIL` every
running trial whose last heartbeat is older than `grace_period` (default `2 × heartbeat_interval`). Both timestamps are
the database server's clock (`now()`), so clock differences between hosts do not matter. A worker that is alive but
cannot write a heartbeat for longer than `grace_period` loses its trial: it is failed, and the worker's own result
for it is then rejected ("Cannot tell a FAIL trial") and the worker stops. Such pauses include a suspended process,
container or VM, a host under memory pressure, and a database that is slow or unreachable. Set `grace_period` to
`heartbeat_interval` plus the longest pause a live worker can have. Optuna's own example uses `heartbeat_interval: 60`
and `grace_period: 120`; MILIA requires `grace_period` to be greater than `heartbeat_interval`. A shorter interval
detects a dead worker sooner but writes to the database more often.

### Method 2: uv (Without Docker)

```bash
# 1. Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone MILIA
git clone https://github.com/shahram-boshra/MILIA.git
cd MILIA

# 3. Install the full stack from the committed lockfile.
#    Choose exactly ONE accelerator extra: cpu | cu126 | cu130.
uv sync --locked --extra cpu
```

For GPU support, choose a CUDA accelerator extra instead of `cpu` — e.g. `uv sync --locked --extra cu130` (the CUDA runtime comes with the PyTorch/PyG wheels; the host needs an NVIDIA driver supporting that CUDA version — ≥ 580 for `cu130`, ≥ 525.60.13 for `cu126`; Maxwell, Pascal and Volta GPUs need `cu126`). The NVIDIA Container Toolkit is needed only for Docker.

Supported platform: **Linux x86_64** — natively via the lockfile, or through the `linux/amd64` Docker image. The lock is resolved for Linux x86_64 only (`[tool.uv] environments` in `pyproject.toml`).

For development (adds pytest and ruff):

```bash
uv sync --locked --extra cpu --extra dev
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

# Parallel HPO: several processes on one study. Needs a shared storage in
# models.hpo.study.storage_options (journal_file on one host, or rdb with heartbeat_interval).
milia --process                                               # process the dataset once
milia --hpo-init                                              # create the study once
# one GPU per worker; CPU threads per worker at most floor(CPUs / workers); workers only run trials
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 milia --train --hpo --hpo-worker 0 &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 milia --train --hpo --hpo-worker 1 &
wait
milia --train --hpo --hpo-finalize                            # results + final model, once

# Run predictions (bundled sample; or supply your own CSV: header smiles,molecule_id)
milia --predict --model-path ./checkpoints/best.pt \
      --test-path test_data/molecules.csv --preds-path ./predictions.csv

# Validate configuration without processing
milia --dry-run

# List available transforms and experimental setups
milia --list-transforms
milia --list-experimental-setups

# List all molecular descriptors (built-in + plugin, e.g. addcore_3d)
milia --list-descriptors

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

### Acceleration

Disabled by default. Enable it in `configs/models.yaml`; it then applies to `train`, HPO trials, cross-validation folds and the final HPO model:

```yaml
models:
  acceleration:
    enabled: true
    device: {type: auto}               # auto | cpu | cuda | mps
    memory:
      mixed_precision: bf16            # no | bf16 | fp16 (fp16 uses a GradScaler on CUDA)
    computation:
      compile_model: true              # torch.compile (Inductor; needs a C/C++ compiler — included in the images)
      compile_dynamic: true            # shape-generic kernels for varying graph sizes
      dataloader: {num_workers: 4, pin_memory: true}   # pinning applies only to CUDA targets
```

Validation and test always run in fp32, so HPO objectives stay comparable across trials. Settings the training paths cannot apply (`distributed.enabled`, `deepspeed`, `fp8`, …) are rejected before training starts, with a message naming each one.

## Trying MILIA — Reproducible Walkthrough

This section walks a reviewer through the shortest path from a fresh clone to a trained model and a prediction. Every command below has been validated end-to-end. Paths are relative throughout, so the walkthrough works identically on any machine — Linux, macOS, WSL, or inside the Docker container.

### Prerequisites

1. **MILIA installed** — see [Installation](#installation) above. Method 1 (Docker) is the fastest route for a one-shot evaluation; Method 2 (uv) is recommended if you intend to inspect or modify source.

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

# 7. Run prediction (bundled sample; swap in your own CSV: header smiles,molecule_id)
milia --predict \
    --model-path ./checkpoints/best.pt \
    --test-path test_data/molecules.csv \
    --preds-path ./predictions.csv
```

After step 5 or 6, your `{working_root_dir}/checkpoints/` directory contains a `best.pt` checkpoint. Step 7 reads that checkpoint and writes per-molecule predictions to `{working_root_dir}/predictions.csv` for the molecules in your `--test-path` input — a CSV with a `smiles,molecule_id` header and one molecule (SMILES) per row. Its columns are named by the targets the model was trained on (`run_info.targets.names`, e.g. `Etot`, `gap`; `json` output lists them as `output_names`); a checkpoint that records no names writes `prediction_<i>`. A node-level model (`task_type` `node_*`, per-atom targets from `node_targets_to_include`) writes one CSV row per atom (`molecule`, `atom`, `atomic_number`, then its outputs) and one JSON entry per molecule (its atoms' rows, with `output_level` and `atomic_numbers`); its conformal intervals are calibrated per molecule and cover all atoms of a molecule at once.

### Where the outputs live

Every artifact MILIA produces lands under `working_root_dir`. With the example value `~/Chem_Data/Milia_PyG_Dataset` this means:

| Artifact | Location |
|---|---|
| Processed PyG graphs | `~/Chem_Data/Milia_PyG_Dataset/processed/` |
| Best model checkpoint | `~/Chem_Data/Milia_PyG_Dataset/checkpoints/best.pt` |
| Per-epoch checkpoints | `~/Chem_Data/Milia_PyG_Dataset/checkpoints/epoch=*.pt` |
| HPO best parameters | `~/Chem_Data/Milia_PyG_Dataset/hpo_output/best_params.json` |
| Training plots | `~/Chem_Data/Milia_PyG_Dataset/training_plots/` |
| Predictions (`--preds-path ./predictions.csv`) | `~/Chem_Data/Milia_PyG_Dataset/predictions.csv` |

Input paths (`--model-path`, `--test-path`) are looked up first as given, relative to the current directory, then under `working_root_dir`; `--model-path` is also looked up by file name in `{working_root_dir}/checkpoints/`. So `--test-path test_data/molecules.csv` reads the bundled sample from the checkout (or `/app` in the Docker image), and `--model-path ./checkpoints/best.pt` finds the trained checkpoint from any directory. A path that is not found lists every location checked. Output paths such as `--preds-path` are always written under `working_root_dir`.

### Serving a trained model

A trained checkpoint becomes a self-contained serving image. The `aws`, `gcp`, `azure` and `container` deployment targets write a package (`model.pt`; `deployment.json` with its SHA-256, the model's signature, an input example, its validation and test metrics and its lineage; and `MODEL_CARD.md`, the same for people) and a `Dockerfile` built on the published MILIA image, which carries the `serve` extra (FastAPI, Uvicorn):

```python
from milia_pipeline.models.deployment import DeploymentManager

DeploymentManager(target="container").prepare_model(
    "~/Chem_Data/Milia_PyG_Dataset/checkpoints/best.pt", "serving_image"
)
```

```bash
docker build -t milia-serve serving_image
docker run -p 8080:8080 milia-serve
curl -s localhost:8080/predict -H 'Content-Type: application/json' -d '{"instances": ["CCO", "c1ccccc1"]}'
```

One app answers every target on port 8080:
- **Amazon SageMaker:** `GET /ping` and `POST /invocations`.
- **Vertex AI:** the routes and port it sets in `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE` and `AIP_HTTP_PORT`.
- **Azure ML, Kubernetes and plain Docker:** `GET /health` and `POST /predict`.

Every predict route takes `{"instances": [...]}`, each instance a SMILES string or a structure object (`{"atomic_numbers": [...], "coordinates": [[x, y, z], ...], "charge": 0, "identifier": "InChI=...", "identifier_type": "inchi"}`, ångström; unknown fields rejected), and returns `{"predictions": [[...], ...], "output_names": [...]}`, one row per molecule. A package of a regression model trained with this version also returns `"intervals"`: split conformal `lower` and `upper` rows at `1 - alpha` (default 90%), calibrated on the validation set after training and, for an ensemble, scaled by the spread across its members; `milia --predict --predict-uncertainty` writes the same as `lower_<name>`/`upper_<name>` columns (`lower_<i>` when the checkpoint records no target names). When the calibration set is too small for 1 − α, an output's interval has no finite bound: JSON outputs (the response, `--predict` JSON) write its bounds as `null` and name it in `intervals.unbounded` (JSON has no infinity); CSV writes `inf`. The coverage holds per target, on average over molecules like the validation set. At startup the server featurizes the package's examples (its SMILES input example, and its structure example when it serves structures) and refuses to serve a model whose node-feature width or output count differs from its signature. The SMILES are featurized exactly as `milia --predict` featurizes them. `MILIA_SERVE_MAX_INSTANCES` caps the batch size per request (default 1000). A graph-level model (`task_type` `graph_*`) answers one row of outputs per instance; a node-level model (`node_*`) answers, per instance, one row per atom with `"output_level": "node"` and the instance's `atomic_numbers` in that order (intervals nested alike, covering all atoms of a molecule at once). The server refuses to start for an edge-level model, or a task type that names no level, with the reason.

**Input contract.** A package records which inputs its model accepts (`signature.inputs.kinds`, derived from the checkpoint): `identifier` (a SMILES or InChI, or a structure file whose coordinates the model does not use), `structure` (an XYZ, SDF or MOL file with given coordinates, built as the training dataset built its molecules) and `dataset` (a processed dataset of the same type, always accepted). Identifiers are refused, with the reason, for a model that takes values from its dataset's QM data (`qm_input_features`: atomic charges, per-atom properties, the given geometry), that reads atomic positions (SchNet, DimeNet, ViSNet, …), or whose checkpoint predates a record identifier featurization must reproduce (hydrogens, QM inputs, pre-transforms). For such a model the server does not start, no serving image or ONNX export is written (deploy it with the `local` target), and `milia --predict` refuses SMILES, InChI and structure files before featurizing them; predict it from a dataset directory (`milia --predict --test-path <dataset directory>`). Structures are accepted when the checkpoint records how its molecules were built (`molecule_creation`: creation strategy, coordinate units, the charges of its training molecules, the converter switches) and no input came from a per-atom value only the dataset holds (`qm_input_sources` `dataset_property`, such as its atomic charges); `milia --predict --test-path molecules.xyz` (or `.sdf`, `.mol`) then builds each molecule with its given coordinates through the training creation, featurization and pre-transforms, so a molecule from the training data gives its training graph. An SDF/MOL record keeps every hydrogen and carries its charge and InChI; an XYZ frame states them in extended-XYZ comment pairs (`charge=-1 inchi=... smiles=... name=...`, coordinates in ångström). A dataset that builds bonds from coordinates (`coordinate_based`: ANI, QDπ, QM40, rMD17, xxMD, Wavefunction) needs the charge; one that reads bonds from an identifier (DFT, DMC, QM9) needs an InChI or SMILES and takes the charge from it. A structure whose charge is not among the training charges, or that cannot be built, is refused with its reason. The server takes structure objects when the model accepts structures and its package records a structure example (up to five training molecules, recorded by training and verified at packaging; a package written without one serves no structures), so a model that reads atomic positions (SchNet, DimeNet, ViSNet, …) is served from structures. Every input, identifier or structure, must also pass the structural filters its training molecules passed (`data_info.input_filters`: `max_atoms`, `min_atoms`, `heavy_atom_filter`, checked with the training filter code; label filters such as the DMC uncertainty filter do not apply to inputs). A refused input fails the request (HTTP 422, `milia --predict` exits with the error) and every refused input is named with its index and reason. A checkpoint that does not record a record (trained before MILIA recorded it, or by a path that does not compute it, such as the HPO final model) is predicted from a dataset only.

`GET /monitoring/drift` compares the served molecules with the training inputs: training records a histogram of each node-feature column and of atoms per molecule, and the server keeps the same histograms over the last `MILIA_DRIFT_WINDOW` molecules (default 1000). Once the window is full it reports each feature's Jensen-Shannon distance to the training histogram and `"status": "drift"` with the features at or above `MILIA_DRIFT_THRESHOLD` (default 0.1); each change of status is logged as one JSON line. A package without a training reference reports `"not_recorded"`.

Every package is verified before it is kept: its checksums always, and, reloaded as the server loads it, it must predict its input example (a model that accepts identifiers) and its structure example (a model that accepts structures) as the source checkpoint does. To keep every packaging as its own version, `publish_package` writes `<output_dir>/<model_name>/<N>/` (1, 2, …) and never overwrites an existing version:

```python
from milia_pipeline.models.deployment.package import publish_package

publish_package("~/Chem_Data/Milia_PyG_Dataset/checkpoints/best.pt", "models")  # models/<name>/1
```

Versions are numbered per model family: `<output_dir>/<family>/<N>/`, where a family is what the model predicts (dataset type, task type, target names and count, source units: `<dataset>-<task>-<digest>`, recorded in the folder's `family.json`), so a GCN and a GAT on the same targets are versions of one family and a GCN on QM9 and one on DFT are not. From the command line, `milia --deploy` publishes a checkpoint (or an existing package) as the next version, with the target's `Dockerfile` and, if asked, `model.onnx`; `milia --predict` accepts the version folder and checks its SHA-256 first:

```bash
milia --deploy --model-path best.pt --deploy-target container   # {working_root_dir}/deployments/<family>/1
milia --predict --model-path deployments/<family>/1 --test-path molecules.csv
```

The `models.deployment` section of `configs/models.yaml` holds the deployment settings (target, base image, output folder, ONNX export, a molecules file to predict, and a gate on a checkpoint metric). With `enabled: true`, standard training gates its best checkpoint on `gate.metric` (threshold, and optionally the newest published version of its model family, compared only when scored on the same data split and, for `val_loss`, the same target scaling; otherwise `gate.json` records why it was not compared), records the decision in `training_output/gate.json`, and when the gate passes publishes the next version as `milia --deploy` does and predicts `predict.input_path`; a failed gate deploys nothing and the run still succeeds. The section is validated before training: an unknown key is an error, and the keys of the earlier section (`strategy`, `edge`, `cloud`, `federated`, `optimization`, `monitoring`) warn and are ignored until their removal in 2.0.0.

MILIA prepares the image but creates no endpoint, so `predict()` on these four targets raises `DeploymentError` naming the route and request body to call. Predict through the endpoint, or in-process with the `local` target.

The `mobile` and `iot` targets are deprecated (`DeprecationWarning`) and will be removed in 2.0.0. MILIA builds no on-device model, because a prediction starts from a SMILES that MILIA featurizes with RDKit in Python, so the model alone cannot predict on a device. These targets write the same package, without a `Dockerfile`, and do not predict. Serve the package instead.

### Exporting a model to ONNX

With the `onnx` extra (`pip install 'milia-py[onnx]'`), a trained model exports to ONNX with PyTorch's `torch.export`-based exporter. Pass a PyG `Data` or `Batch` as the example: the ONNX inputs become `x`, `edge_index`, `edge_attr` (when the data has it) and `batch`, with dynamic node and edge counts, and the output has one row per graph.

```python
from milia_pipeline.models.deployment.onnx_export import export_onnx

export_onnx(model, "model.onnx", example_batch)  # verified against PyTorch with ONNX Runtime
```

MILIA captures the model with `torch.export` in strict mode, which keeps the number of graphs in a batch symbolic, and the exported file is run with ONNX Runtime on the example and compared with PyTorch before it is kept. A model `torch.export` cannot capture raises `ExportError` naming the cause.

Coverage for graph-level tasks (PyG 2.8.1, PyTorch 2.12; `tests/test_onnx_export_coverage.py`):

| Exports | Blocked upstream (`ExportError`) |
|---|---|
| GCN, GAT, GIN, GraphSAGE, EdgeCNN, PNA, MLP, PMLP (`norm=False`), Polynormer, AttentiveFP, NeuralFingerprint | GraphUNet (`augment_adj` branches on tensor values in `coalesce()` and builds sparse CSR tensors; [pyg-team/pytorch_geometric#10914](https://github.com/pyg-team/pytorch_geometric/issues/10914)), SGFormer (ONNX translation fails on PyTorch 2.12, and `aten::sort.stable` has no ONNX function), SchNet, DimeNet, DimeNetPlusPlus, ViSNet (`radius_graph` infers the graph count with `int(batch.max())`, [pyg-team/pytorch_geometric#10915](https://github.com/pyg-team/pytorch_geometric/issues/10915); the `pyg.radius` operator has no fake kernel, [pyg-team/pyg-lib#740](https://github.com/pyg-team/pyg-lib/issues/740)) |

Only tensors are graph inputs; Python numbers such as PyG's `batch_size` would be frozen into the file and are not exported. PMLP's default `norm=True` normalizes with the statistics of the batch it is given, also at inference, so its exported predictions depend on batch composition.

### Exporting a model to `.pt2` (`torch.export`)

`export_pt2` writes the same captured program (inputs, dynamic node and edge counts, coverage as above) as a PyTorch `.pt2` file, PyTorch's replacement for TorchScript. No extra is needed.

```python
from milia_pipeline.models.deployment.pt2_export import export_pt2

export_pt2(model, "model.pt2", example_batch)                    # ExportedProgram
export_pt2(model, "model_aoti.pt2", example_batch, aot_compile=True)  # AOTInductor package (needs a C++ compiler)
```

| Artifact | Load with | Runs |
|---|---|---|
| `ExportedProgram` (default) | `torch.export.load("model.pt2").module()` | in Python with PyTorch, without MILIA's model code |
| AOTInductor package (`aot_compile=True`) | `torch._inductor.aoti_load_package("model_aoti.pt2")` | compiled ahead of time to native code |

Each file is loaded back and compared with PyTorch on the example before it is kept. An `ExportedProgram` stores MILIA's metadata (`milia.json`: MILIA and PyTorch versions, input names), read with `torch.export.load(path, extra_files={"milia.json": ""})`. Two PyTorch rules apply: `torch.export.load` uses pickle, so load only files from a trusted source; and the format is under active development, so a file may not load with a newer PyTorch than the one recorded.

### Introspection commands

These are read-only and useful for orientation:

```bash
milia --list-transforms              # 130+ transforms: full PyG set + 55 novel across 9 plugin families
milia --list-descriptors             # 3,026+ descriptors: 418 built-in + 1,019 via addcore_3d + 20 via constitutional_property + 107 via topological_connectivity + 86 via walk_path_information + 177 via matrix_spectral + 606 via autocorrelation_2d + 316 via estate_atomtype + 150 via cats2d + 45 via eta + 50 via cpsa_geometric_3d + 25 via eht_electronic + 310 via cdk_substructure (opt-in, [descriptors-cdk])
milia --list-experimental-setups     # Available research/experiment configurations
milia --help                         # Full CLI reference
```

### Prediction input format

`milia --predict --test-path <file>` expects a CSV with a `smiles,molecule_id` header and one molecule per row, for example:

```csv
smiles,molecule_id
CCO,ethanol
CC(=O)O,acetic_acid
c1ccccc1,benzene
```

A ready-to-run sample is bundled at [`test_data/molecules.csv`](test_data/molecules.csv) (10 common small molecules); substitute your own file to predict on different molecules.

## Architecture

MILIA is organized into 11 core modules and a split configuration system:

| Module | Files | Purpose |
|--------|-------|---------|
| `config/` | 7 files, ~22K lines | Multi-layered configuration management with Pydantic V2 validation, YAML splitting (`configs/` directory with deep merge), thread-safe caching, schema migration, and 60+ accessor functions |
| `molecules/` | 7 files, ~14K lines | Molecular conversion (RDKit → PyG), structural feature extraction, property enrichment, filtering with transform compatibility, and registry-integrated validation |
| `transformations/` | 4 files, ~16K lines | 7-layer graph transformation system: dynamic discovery, registry, semantic validation, composition with caching, configuration bridge, error recovery, and production metrics |
| `datasets/` | 11 files | Registry-based PyTorch Geometric datasets with Protocol contracts, compile-time validation, and 11 concrete implementations spanning the VQM24 family (DFT, DMC, Wavefunction), QM9, the ANI family (ANI-1x, ANI-1ccx, ANI-2x), rMD17, xxMD, QDπ, and QM40 — see the [Datasets](#datasets) section |
| `handlers/` | 15+ files | 11 dataset handler types (DFT, DMC, Wavefunction, QM9, ANI-1x, ANI-1ccx, rMD17, ANI-2x, XXMD, QDPi, QM40) with transform integration and lazy loading |
| `preprocessing/` | 8+ files | Modular wavefunction preprocessing (MOLDEN, FCHK), data refinement, and VQM24 support |
| `descriptors/` | 6+ files | 3,000+ molecular descriptors across 6 categories (~492 RDKit baseline + 1,019 via the `addcore_3d` plugin + 20 via the `constitutional_property` plugin + 107 via the `topological_connectivity` plugin + 86 via the `walk_path_information` plugin + 177 via the `matrix_spectral` plugin + 606 via the `autocorrelation_2d` plugin + 316 via the `estate_atomtype` plugin + 150 via the `cats2d` plugin + 45 via the `eta` plugin + 50 via the `cpsa_geometric_3d` plugin + 25 via the `eht_electronic` plugin; + 310 via the opt-in `cdk_substructure` plugin) with thread-safe singleton registry, caching, conformer generation, block-cache for vector blocks, and plugin support |
| `models/` | 25+ files | Full ML lifecycle: registry with dynamic PyG introspection, factory, trainer with callbacks, post-training inference, architecture builder (10 templates), model composer, acceleration (CPU/GPU/MPS/TPU; AMP, `torch.compile`, loader workers from config; DP/DDP/FSDP via the Python API), deployment (cloud/container/local serving, federated; edge deprecated), monitoring, and model plugins |
| `models/hpo/` | 17 files | Hyperparameter optimization: Optuna backend, 7 search algorithms, 7 pruners, multi-objective studies, parallel workers on shared storage, neural architecture search, transfer learning with warm-starting, and study analysis |
| `cli_manager` | 1 file, ~3.8K lines | 12 argument groups, 12+ processing modes, interactive mode, and post-training prediction arguments |
| `exceptions` | 1 file | Comprehensive exception hierarchy with registry-based dataset-specific errors |

**Supporting directories:**

| Directory | Purpose |
|-----------|---------|
| `configs/` | Split YAML configuration with per-dataset files (11 datasets) and deep-merge architecture |
| `plugins/` | Plugin kind-containers — `transformations/`, `descriptors/`, `models/` — each holding its plugin categories, a `user_template/` scaffold, and YAML manifests (discovered via config-driven `plugin_paths`) |
| `tests/` | 229 test files spanning all core modules (the `smoke` subset is 1,971 tests) |

## Datasets

MILIA ships with **11 production-ready dataset implementations** covering the major quantum-chemistry and molecular-dynamics benchmarks used in modern molecular machine learning. Selecting a dataset is a one-line change in `configs/main.yaml` (`dataset_type: <Registry key>`) — no code, no rebuild, no glue.

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
| `QM40` | `QM40Dataset` | QM40 — 162,954 neutral drug-like ZINC molecules (10–40 heavy atoms; H, C, N, O, F, S, Cl) at B3LYP/6-31G(2df,p) with Mulliken charges, 16 scalar QM properties, and per-bond local vibrational mode force constants (Madushanka, Moura Jr. & Kraka, *Sci. Data* 11, 1376, 2024) | Å | Hartree | coordinate-based (neutral-only; SMILES, no InChI) |

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
| `QM40` | ✗ | ✗ | ✓ | ✗ | ✓ |

The atomization-energy column says a dataset's data supports one. MILIA computes it only from isolated-atom references at that dataset's level of theory, declared in `<dataset>_config.atomic_references` (`docs/atomic_reference_energies.md`): QM9 uses its authors' published `atomref.txt`, xxMD and QDπ (neutral molecules) use references MILIA computed in their own programs, and the DFT dataset (VQM24) publishes `Eatomization` itself; QM40, ANI-1x, ANI-1ccx, ANI-2x and rMD17 compute none until references at their level of theory are available.

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

**Graph-level properties follow one contract for every dataset.** `DatasetHandler.enrich_pyg_data` is concrete and may not be overridden (a handler that does fails when its class is created). It runs the same shared steps (`milia_pipeline/molecules/property_enrichment.py`) for every dataset, in a fixed order:

1. the handler hook `prepare_graph_properties` — values the dataset normalises together (DFT refines `freqs` and `vibmodes` as pairs); single values are normalised in `process_property_value` (DFT pads a linear molecule's two rotational constants to three);
2. the atomization energy from `calculate_atomization_energy_from`, stored as `atomization_energy_key_name` in the dataset's energy unit (Hartree), where atomic references are declared (`docs/atomic_reference_energies.md`);
3. `vector_graph_properties_to_include` and `variable_len_graph_properties_to_include`, one attribute each, checked against `property_availability.<KEY>.shapes` (`null` is a dimension that varies between molecules; a property without a shape takes the first molecule's, and every molecule must match);
4. `scalar_graph_targets_to_include` → `y`, with its column names in `y_property_names` (recorded in the checkpoint's `run_info.targets.names`); a derived value is a target only when its configured name is listed;
5. the handler hook `add_dataset_extras` — attributes outside the configuration (DMC uncertainty, QDπ charge, formula and subset); it may not change `y`.

The selection is the configured properties that the data provides: a handler may answer `provided_properties` (Wavefunction: the file's `feature_tier`), and the result is fixed by the first molecule for the whole run and logged once. A configured property missing or malformed for one molecule refuses that molecule, so every graph carries the same attributes. When processing starts, every configured name must be listed in `property_availability.<KEY>` and have one role (input, target or attribute); violations are reported together as a `ConfigurationError`. A new handler therefore contains no graph-building code. `tests/test_contract_dataset_graph_properties.py` checks this contract for every registered dataset, read from the registry, so a new dataset is covered without editing the tests.

Once the three files are in place, the new dataset is a first-class citizen of every part of the pipeline — CLI (`milia --process`, `--train`, `--predict`), YAML configuration, hyperparameter optimization, transform validation, descriptors, and prediction — selectable by setting `dataset_type: <key>` in `configs/main.yaml`. The 11 shipped datasets follow this exact pattern; the registry has no knowledge that they are "built-in" rather than user-added.

## Testing

```bash
# Run the full test suite
pytest

# Run specific test categories
pytest -m "not slow"         # Skip slow tests
pytest -m integration        # Integration tests only
pytest -m gpu                # GPU-specific tests only
```

The test suite spans 229 test files across all core modules; the `smoke` subset is 1,971 tests. Run `pytest --collect-only -q | tail -1` for the exact collected count in your checkout.

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started, coding standards, and the pull request process.

## Citation

If you use MILIA in your research, please cite it. See [CITATION.cff](CITATION.cff) for the preferred citation format.

Archived releases are citable via Zenodo. Cite the **concept DOI** [10.5281/zenodo.21854751](https://doi.org/10.5281/zenodo.21854751) to always reference the latest release, or a **version DOI** (e.g. [10.5281/zenodo.23212466](https://doi.org/10.5281/zenodo.23212466) for v1.16.0) to pin a specific one. An earlier manual deposit of v1.1.0 has its own record, [10.5281/zenodo.20151553](https://doi.org/10.5281/zenodo.20151553), which releases do not update.

## License

MILIA is released under the [MIT License](LICENSE).

## Author

- **Asadollah (Shahram) Boshra** — [a.boshra@gmail.com](mailto:a.boshra@gmail.com)

## Links

- **Repository**: [github.com/shahram-boshra/MILIA](https://github.com/shahram-boshra/MILIA)
- **Issues**: [github.com/shahram-boshra/MILIA/issues](https://github.com/shahram-boshra/MILIA/issues)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
