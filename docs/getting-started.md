# Getting Started

## Installation

MILIA uses conda for managing heavy scientific dependencies (PyTorch, PyTorch
Geometric, RDKit). See the [README](https://github.com/shahram-boshra/MILIA#installation)
for complete installation instructions.

### Quick Install

```bash
# 1. Create and activate conda environment
conda create -n milia python=3.10
conda activate milia

# 2. Install core scientific dependencies via conda-forge
conda install -c conda-forge numpy scipy pyyaml h5py pandas rdkit \
    matplotlib pydantic-settings ase torchmetrics hydra-core optuna \
    plotly scikit-learn pytorch cpuonly -c pytorch

# 3. Install PyTorch Geometric and extensions
conda install -c pyg torch-geometric
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv

# 4. Install MILIA
pip install -e .
```

### Verify Installation

```bash
# Confirm the package is installed
python -c "from milia_pipeline import get_version; print(get_version())"

# Confirm the CLI entry point works
milia --help
```

## Quick Start

### CLI Usage

```bash
# Process a dataset
milia --config configs/main.yaml --process

# Run inference with a trained model
milia --predict \
    --model-path ./checkpoints/best_model.pt \
    --test-path ./molecules.csv \
    --preds-path ./predictions.csv
```

### Programmatic Usage

```python
from milia_pipeline import create_cli_manager, setup_logging

# Setup logging
logger = setup_logging(log_level="INFO")

# Create CLI manager and parse arguments
cli = create_cli_manager(logger=logger)
args = cli.parse_args(["--config", "configs/main.yaml", "--process"])

# Load and validate configuration
config = cli.load_and_merge_config(args)
cli.validate_args(args, config)
```

## Next Steps

- Browse the {doc}`api/index` for detailed module documentation.
- Read the {doc}`contributing` guide to get involved.
