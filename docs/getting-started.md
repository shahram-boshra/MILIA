# Getting Started

## Installation

MILIA uses [uv](https://docs.astral.sh/uv/) to manage all dependencies (PyTorch, PyTorch
Geometric, RDKit, …) from a committed lockfile. See the [README](https://github.com/shahram-boshra/MILIA#installation)
for complete installation instructions.

### Quick Install

```bash
# 1. Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone MILIA
git clone https://github.com/shahram-boshra/MILIA.git
cd MILIA

# 3. Install the full stack (PyTorch, PyG + compiled companions, RDKit, …) from the
#    committed lockfile. Choose exactly ONE accelerator extra: cpu | cu118 | cu121 | cu124.
uv sync --locked --extra cpu

# 4. Run MILIA (inside the locked environment)
uv run --extra cpu milia --help
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
