# PyTorch Geometric Models Documentation Generator

## Overview

`generate_pyg_models_docs.py` is a production-ready script that discovers and documents all available ML/DL models in PyTorch Geometric (PyG) library. The script generates comprehensive markdown documentation organized by model categories, including usage examples, configuration guides, and installation instructions.

## Features

✅ **Comprehensive Model Catalog**: Documents 100+ PyG models across 12 categories  
✅ **Organized by Category**: GNN architectures, pooling, aggregation, transformers, and more  
✅ **Rich Documentation**: Includes descriptions, import paths, and paper references  
✅ **Usage Examples**: Ready-to-use code snippets for common models  
✅ **Configuration Guide**: Task-based model selection and hyperparameter recommendations  
✅ **Installation Instructions**: Complete setup guide with dependencies  
✅ **Additional Resources**: Links to documentation, tutorials, and papers  

## Requirements

- Python 3.7+
- No PyTorch Geometric installation required to run the script (but needed to use the models)
- The script works standalone and generates documentation based on the built-in model catalog

## Installation

No installation required! The script is self-contained and runs immediately.

## Usage

### Basic Usage

```bash
python generate_pyg_models_docs.py
```

This generates `PYG_MODELS_REFERENCE.md` in the current directory.

### Custom Output Path

```bash
python generate_pyg_models_docs.py --output /path/to/custom_output.md
```

### Verbose Mode

```bash
python generate_pyg_models_docs.py --verbose
```

Shows detailed progress and warnings during generation.

### Combined Options

```bash
python generate_pyg_models_docs.py -o ./docs/pyg_models.md -v
```

## Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output file path | `./PYG_MODELS_REFERENCE.md` |
| `--verbose` | `-v` | Enable verbose output | `False` |
| `--help` | `-h` | Show help message | - |

## Output Structure

The generated markdown documentation includes:

1. **Header**
   - Generation timestamp
   - Total model count
   - PyG documentation links

2. **Table of Contents**
   - All categories with model counts
   - Clickable links to sections

3. **Model Categories**
   - Basic GNN (GCN, GraphSAGE, GAT, GIN, etc.)
   - Convolutional Layers (50+ conv layers)
   - Attention Mechanisms (GAT, Transformer, etc.)
   - Pooling Layers (TopK, SAG, Global, etc.)
   - Aggregation Functions (Mean, Max, LSTM, etc.)
   - Encoders (Node2Vec, DeepGraphInfomax, etc.)
   - Autoencoders (GAE, VGAE, ARGA, etc.)
   - Transformers (GPSConv, HGT, etc.)
   - Temporal Models (TGN, TGAT, etc.)
   - Meta-Learning
   - Explainability (GNNExplainer, PGExplainer)
   - Utility Models (MLP, SchNet, DimeNet, etc.)

4. **Usage Examples**
   - GCN basic example
   - GAT with attention heads
   - GraphSAGE for inductive learning
   - Custom message passing layer

5. **Configuration Guide**
   - Task-based model selection
   - Hyperparameter recommendations
   - Best practices per model type

6. **Installation Guide**
   - Basic PyG installation
   - Optional dependencies
   - Verification steps

7. **Additional Resources**
   - Official documentation
   - Tutorials and courses
   - Research papers
   - Community links

## Model Categories Explained

### Basic GNN
Core graph neural network architectures (GCN, GraphSAGE, GAT, GIN, EdgeCNN, PNA)

### Convolutional
50+ specialized convolutional layers for different graph types and tasks

### Attention
Attention-based models that learn importance weights for neighbors

### Pooling
Graph pooling operations for graph-level tasks and hierarchical learning

### Aggregation
Various aggregation functions for neighborhood information

### Encoder
Unsupervised learning models for node embeddings

### Autoencoder
Variational and adversarial autoencoders for graphs

### Transformer
Transformer-style architectures adapted for graphs

### Temporal
Models for dynamic graphs and temporal networks

### Meta-Learning
Meta-learning approaches for graphs

### Explainability
Models and methods for explaining GNN predictions

### Utility
Helper models and specialized architectures (MLP, SchNet, DimeNet, etc.)

## Example Output

```markdown
# PyTorch Geometric Models Reference

Complete list of available GNN/ML/DL models in PyTorch Geometric library.

**Generated:** 2025-11-19 15:30:00
**Total Models:** 120
**PyG Documentation:** https://pytorch-geometric.readthedocs.io/

---

## Table of Contents

- [Basic GNN](#basic-gnn) (6 models)
- [Convolutional](#convolutional) (52 models)
...

## Basic GNN

**Count:** 6 models

| Model | Import Path | Description |
|-------|-------------|-------------|
| `GCN` | `torch_geometric.nn.models.GCN` | Graph Convolutional Network - Semi-supervised node classification |
| `GraphSAGE` | `torch_geometric.nn.models.GraphSAGE` | Inductive representation learning on large graphs |
...
```

## Customization

### Adding New Models

To add new models to the catalog, edit the `MODEL_CATALOG` dictionary in the script:

```python
MODEL_CATALOG: Dict[ModelCategory, List[Tuple[str, str, str]]] = {
    ModelCategory.BASIC_GNN: [
        ("ModelName", "import.path.ModelName", "Description"),
        # Add more models here
    ],
}
```

### Adding New Categories

1. Add to `ModelCategory` enum:
```python
class ModelCategory(Enum):
    NEW_CATEGORY = "new_category"
```

2. Add models to `MODEL_CATALOG`:
```python
ModelCategory.NEW_CATEGORY: [
    ("Model1", "path.Model1", "Description"),
],
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Update PyG Models Documentation

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  update-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Generate documentation
        run: python generate_pyg_models_docs.py -o docs/PYG_MODELS.md -v
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add docs/PYG_MODELS.md
          git commit -m "Update PyG models documentation" || echo "No changes"
          git push
```

## Comparison with Descriptor Script

This script is designed similarly to `generate_descriptor_docs.py` but for PyG models:

| Feature | Descriptor Script | PyG Models Script |
|---------|------------------|-------------------|
| **Purpose** | Document molecular descriptors | Document PyG ML/DL models |
| **Data Source** | Internal registry | Built-in catalog |
| **Categories** | Chemical properties | Model architectures |
| **Count** | ~200 descriptors | ~120 models |
| **Output** | DESCRIPTOR_REFERENCE.md | PYG_MODELS_REFERENCE.md |

## Maintenance

### Keeping Models Up-to-Date

PyG frequently adds new models. To keep the documentation current:

1. **Check PyG Releases**: Monitor [PyG releases](https://github.com/pyg-team/pytorch_geometric/releases)
2. **Update MODEL_CATALOG**: Add new models to appropriate categories
3. **Regenerate Documentation**: Run the script after updates
4. **Version Control**: Track changes in your repository

### Recommended Update Schedule

- **Minor Updates**: After each PyG minor release
- **Major Review**: Quarterly comprehensive audit
- **Automated**: Set up CI/CD for weekly regeneration

## Troubleshooting

### Script Won't Run

**Issue**: `ModuleNotFoundError` or import errors

**Solution**: The script doesn't require PyG to be installed. It generates documentation from the built-in catalog. If you see import errors, check Python version (3.7+ required).

### Missing Models

**Issue**: New PyG models not appearing in documentation

**Solution**: Models must be manually added to `MODEL_CATALOG`. The script doesn't dynamically introspect PyG (by design, so it can run without PyG installed).

### Output File Not Created

**Issue**: No output file generated

**Solution**: Check write permissions on the output directory. Use `-v` flag to see detailed error messages.

## Contributing

To contribute improvements to this script:

1. Add new models to `MODEL_CATALOG`
2. Enhance documentation sections
3. Add new usage examples
4. Improve categorization logic
5. Add new configuration recommendations

## License

MIT License - Feel free to use and modify for your projects.

## Credits

- Script inspired by the VQM24 descriptor documentation generator
- Model information from [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/)
- Created for the VQM24 Pipeline Project

## Support

For issues or questions:
- Check [PyG Documentation](https://pytorch-geometric.readthedocs.io/)
- Review [PyG Examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples)
- Join [PyG Slack Community](https://pytorch-geometric.slack.com/)

---

**Last Updated**: November 2025  
**Script Version**: 1.0.0  
**Compatible with**: PyTorch Geometric 2.0+
