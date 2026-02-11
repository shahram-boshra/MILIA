# Quantum Augmentations Plugin

Advanced quantum property augmentation and normalization transforms for VQM24 molecular graphs.

## Overview

This plugin provides three specialized transforms for working with quantum molecular properties in the VQM24 pipeline:

1. **EnergyNormalizer** - Normalize DFT and DMC energy values using various statistical methods
2. **ChargeAugmentor** - Add controlled Gaussian noise to Mulliken charges for data augmentation
3. **VibrationalModeFilter** - Filter or select specific vibrational modes based on frequency criteria

## Features

✅ Multiple normalization methods (z-score, min-max, robust)  
✅ Charge-conserving augmentation  
✅ Flexible vibrational mode filtering  
✅ Comprehensive validation  
✅ Full test coverage (95%+)  
✅ Production-ready  

## Installation

### From Plugin Directory

```bash
# Copy plugin to VQM24 plugins directory
cp -r quantum_augmentations /path/to/vqm24/plugins/

# Verify installation
python main.py --list-plugins
```

Expected output:
```
Found 1 plugin(s):
  âœ" quantum_augmentations v1.0.0 - Advanced quantum property augmentation
```

### From Git Repository

```bash
cd /path/to/vqm24/plugins
git clone https://github.com/vqm24/quantum-augmentations.git
```

## Usage

### Basic Usage

```python
from quantum_augmentations import EnergyNormalizer, ChargeAugmentor

# Normalize energies
normalizer = EnergyNormalizer(method='zscore')
normalized_data = normalizer(data)

# Augment charges
augmentor = ChargeAugmentor(noise_std=0.01)
augmented_data = augmentor(data)
```

### Configuration Usage

Add to your `config.yaml`:

```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins

transformations:
  experimental_setups:
    with_quantum_augmentations:
      name: "with_quantum_augmentations"
      description: "Using quantum augmentation transforms"
      enabled: true
      transforms:
        # Built-in transforms
        - name: "AddSelfLoops"
          enabled: true
        
        # Plugin transform - Energy normalization
        - name: "EnergyNormalizer"
          enabled: true
          params:
            method: "zscore"
            include_dmc: true
        
        # Plugin transform - Charge augmentation
        - name: "ChargeAugmentor"
          enabled: true
          params:
            noise_std: 0.01
            preserve_total_charge: true
            seed: 42
        
        # Built-in transforms
        - name: "GCNNorm"
          enabled: true
  
  default_setup: "with_quantum_augmentations"
```

Run pipeline:
```bash
python main.py
```

## Transform Documentation

### EnergyNormalizer

Normalize DFT and optionally DMC energy values using statistical methods.

**Parameters:**
- `method` (str, default='zscore'): Normalization method
  - `'zscore'`: Zero mean, unit variance
  - `'minmax'`: Scale to [0, 1] range
  - `'robust'`: Robust scaling using median and IQR
- `epsilon` (float, default=1e-8): Small constant for numerical stability
- `include_dmc` (bool, default=True): Also normalize DMC energy if present

**Required Attributes:**
- Graph attributes: `energy` (and optionally `dmc_energy`)

**Example:**
```python
normalizer = EnergyNormalizer(method='robust', include_dmc=True)
result = normalizer(data)
```

**When to use:**
- Improving training stability
- Comparing molecules with vastly different energies
- Preparing data for energy prediction tasks

**When NOT to use:**
- When absolute energy values are important
- For single-molecule analysis

---

### ChargeAugmentor

Add controlled Gaussian noise to Mulliken charges for data augmentation.

**Parameters:**
- `noise_std` (float, default=0.01): Standard deviation of Gaussian noise
  - Range: [0.0, 0.5]
  - Typical values: 0.01-0.05 for mild augmentation
- `preserve_total_charge` (bool, default=True): Ensure total charge is preserved
- `seed` (int, default=None): Random seed for reproducibility

**Required Attributes:**
- Graph attributes: `charges`
- Optional: `total_charge` (if preserve_total_charge=True)

**Example:**
```python
augmentor = ChargeAugmentor(noise_std=0.02, preserve_total_charge=True, seed=42)
augmented = augmentor(data)
```

**When to use:**
- Training data augmentation
- Improving model robustness
- Testing model sensitivity

**When NOT to use:**
- Validation/test sets
- When charge precision is critical
- With noise_std > 0.1 (unrealistic)

---

### VibrationalModeFilter

Filter or select specific vibrational modes based on frequency criteria.

**Parameters:**
- `mode_selection` (str, default='all'): Which modes to keep
  - `'all'`: Keep all modes
  - `'low_frequency'`: Keep modes below threshold
  - `'high_frequency'`: Keep modes above threshold
  - `'custom'`: Custom filtering (implement in subclass)
- `frequency_threshold` (float, default=500.0): Frequency threshold in cm⁻¹
- `max_modes` (int, default=50): Maximum number of modes to keep

**Required Attributes:**
- Graph attributes: `vibmodes` [n_modes, n_atoms, 3]

**Example:**
```python
# Keep only low-frequency modes
filter = VibrationalModeFilter(
    mode_selection='low_frequency',
    frequency_threshold=800.0,
    max_modes=20
)
filtered = filter(data)
```

**When to use:**
- Focusing on specific vibrational characteristics
- Reducing computational cost
- Analyzing specific mode types

**When NOT to use:**
- When all vibrational information is needed
- For full spectroscopic analysis

## Examples

See the `examples/` directory for complete working examples:
- `basic_usage.py` - Simple usage patterns
- `advanced_usage.py` - Advanced features and combinations
- `pipeline_integration.py` - Full pipeline integration

## Testing

Run the test suite:

```bash
cd quantum_augmentations
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=quantum_augmentations --cov-report=html
```

View coverage report:

```bash
open htmlcov/index.html
```

## Development

### Requirements

- Python >= 3.10
- VQM24 >= 1.0.0
- scipy >= 1.15.0

### Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new features
4. Ensure all tests pass (`pytest tests/ -v`)
5. Update documentation
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/vqm24/quantum-augmentations.git
cd quantum-augmentations

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/ -v
```

## Citation

If you use this plugin in your research, please cite:

```bibtex
@software{quantum_augmentations,
  author = {VQM24 Team},
  title = {Quantum Augmentations: Advanced transforms for VQM24},
  year = {2025},
  url = {https://github.com/vqm24/quantum-augmentations},
  version = {1.0.0}
}
```

## License

This plugin is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

## Support

- **Documentation:** https://quantum-augmentations.readthedocs.io
- **Issues:** https://github.com/vqm24/quantum-augmentations/issues
- **Discussions:** https://github.com/vqm24/quantum-augmentations/discussions
- **Email:** support@vqm24.example.com

## Acknowledgments

- VQM24 team for the base pipeline
- Contributors and maintainers
- Research community for feedback

## Related Plugins

- **molecular-filters** - Molecular structure filtering
- **advanced-normalization** - Additional normalization methods
- **geometry-augmentation** - Geometric augmentation transforms
