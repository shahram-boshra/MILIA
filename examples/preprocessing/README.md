# Preprocessing System - Configuration Examples

This directory contains example configurations for the VQM24 Pipeline preprocessing subsystem.

## Overview

The preprocessing subsystem handles **one-time transformation** of raw dataset files into the `.npz` format expected by `VQM24Dataset`. This is an **offline operation** that happens before dataset creation.

### Current Support

- ✅ **Wavefunction Dataset**: `.tar.gz` archive → `.npz` file
- 🚧 **DFT Dataset**: Planned (Phase 4+)
- 🚧 **DMC Dataset**: Planned (Phase 4+)

## Configuration Files

### 1. wavefunction_preprocess.yaml

**Purpose**: Complete production configuration for wavefunction dataset preprocessing.

**Key Features**:
- Full configuration with all options documented
- Production-ready defaults
- Performance tuning parameters
- Validation settings

**Usage**:
```bash
python main.py --preprocess --preprocess-config examples/preprocessing/wavefunction_preprocess.yaml
```

**Processing Time**: 13 hours (standard tier, 937k molecules)

---

### 2. quick_preprocess.yaml

**Purpose**: Quick testing configuration for validation.

**Key Features**:
- Processes only 100 molecules
- Basic feature tier (fastest)
- Keeps temp files for inspection
- Debug logging enabled

**Usage**:
```bash
python main.py --preprocess --preprocess-config examples/preprocessing/quick_preprocess.yaml
```

**Processing Time**: 1-5 seconds

---

## Configuration Structure

All preprocessing configurations follow this structure:
```yaml
preprocessing:
  dataset_type: "Wavefunction"  # Required
  input_path: "path/to/input"   # Required
  output_path: "path/to/output" # Required
  num_molecules: 1000           # Optional (null = all)
  feature_tier: "standard"      # Optional (basic/standard/full)
  force_overwrite: false        # Optional
  cleanup_temp: true            # Optional
  show_progress: true           # Optional

wavefunction_config:          # Dataset-specific settings
  file_pattern: "*.molden"
  extraction_method: "streaming"
  # ... more options

logging:                        # Logging configuration
  log_level: "INFO"
  log_file: "logs/preprocessing.log"
```

---

## Feature Tiers

The preprocessing system supports three feature extraction tiers:

### Basic Tier (Fastest)
- Atomic numbers
- Coordinates
- Charge
- Multiplicity

**Use Case**: Quick testing, initial validation  
**Processing Time**: ~0.01 sec/molecule

### Standard Tier (Recommended)
- All basic features
- MO coefficients
- MO energies
- Total energy
- HOMO-LUMO gap

**Use Case**: Production machine learning workflows  
**Processing Time**: ~0.05 sec/molecule

### Full Tier (Most Complete)
- All standard features
- Density matrix
- Dipole moment
- Quadrupole moment
- Orbital symmetries
- Vibrational frequencies

**Use Case**: Advanced research, complete dataset archival  
**Processing Time**: ~0.1 sec/molecule

---

## CLI Usage

### Basic Preprocessing
```bash
# Using configuration file
python main.py --preprocess --preprocess-config examples/preprocessing/wavefunction_preprocess.yaml

# Using CLI overrides
python main.py --preprocess \
  --preprocess-dataset Wavefunction \
  --preprocess-input raw/wavefunctions.tar.gz \
  --preprocess-output processed/wavefunctions.npz \
  --preprocess-feature-tier standard
```

### Validation Modes
```bash
# Validate configuration only
python main.py --validate-preprocessing-only \
  --preprocess-config examples/preprocessing/wavefunction_preprocess.yaml

# Test preprocessor with small dataset
python main.py --test-preprocessor-only \
  --preprocess-dataset Wavefunction \
  --preprocess-num-molecules 10

# List available preprocessors
python main.py --list-preprocessors
```

### CLI Overrides

All configuration options can be overridden via CLI:
```bash
python main.py --preprocess \
  --preprocess-config examples/preprocessing/wavefunction_preprocess.yaml \
  --preprocess-num-molecules 1000 \        # Override: process 1000 instead of all
  --preprocess-feature-tier basic \         # Override: use basic tier
  --preprocess-force \                      # Override: force overwrite
  --preprocess-progress                     # Override: show progress
```

---

## Recommended Workflow

### 1. Quick Test (1 minute)
```bash
# Test with 10 molecules
python main.py --preprocess \
  --preprocess-dataset Wavefunction \
  --preprocess-input raw/wavefunctions.tar.gz \
  --preprocess-output processed/test_10.npz \
  --preprocess-num-molecules 10 \
  --preprocess-feature-tier basic
```

### 2. Validation Test (5 minutes)
```bash
# Test with 100 molecules using config
python main.py --preprocess \
  --preprocess-config examples/preprocessing/quick_preprocess.yaml
```

### 3. Small Dataset (1 hour)
```bash
# Process 10,000 molecules with standard features
python main.py --preprocess \
  --preprocess-config examples/preprocessing/wavefunction_preprocess.yaml \
  --preprocess-num-molecules 10000
```

### 4. Full Production (overnight)
```bash
# Process all molecules (937k+)
python main.py --preprocess \
  --preprocess-config examples/preprocessing/wavefunction_preprocess.yaml
```

---

## Troubleshooting

### Issue: "Preprocessing config file not found"

**Solution**: Check that the config file path is correct:
```bash
ls -la examples/preprocessing/wavefunction_preprocess.yaml
```

### Issue: "Input file not found"

**Solution**: Verify the input path in the config matches your file location:
```bash
ls -la raw/wavefunctions.tar.gz
```

### Issue: "Insufficient disk space"

**Solution**: Preprocessing needs ~2x the tar.gz size for temporary extraction:
- Check available space: `df -h`
- Reduce `num_molecules` for testing
- Use `cleanup_temp: true` to remove temp files immediately

### Issue: "Memory error during processing"

**Solution**: Reduce batch size and enable streaming:
```yaml
preprocessing:
  batch_size: 50  # Reduce from 100
  
wavefunction_config:
  extraction_method: "streaming"  # Memory-efficient
```

---

## Performance Tips

1. **Use SSD for temp directory**: Significantly faster extraction
```yaml
   wavefunction_config:
     temp_dir: "/mnt/ssd/temp"
```

2. **Adjust batch size**: Balance memory vs. speed
   - Large batch (500): Faster, more memory
   - Small batch (50): Slower, less memory

3. **Use appropriate feature tier**:
   - Testing → `basic`
   - Production ML → `standard`
   - Complete archive → `full`

4. **Enable cleanup**: Save disk space
```yaml
   preprocessing:
     cleanup_temp: true
```

---

## Output Structure

The preprocessing system generates a `.npz` file with this structure:
```python
data = np.load('wavefunctions.npz', allow_pickle=True)

# Keys in the .npz file
data.files  # ['compounds', 'metadata', 'features']

# Compounds: molecule identifiers
compounds = data['compounds']  # Array of compound IDs

# Metadata: per-molecule metadata
metadata = data['metadata']  # Dict with charge, multiplicity, etc.

# Features: extracted quantum mechanical features
features = data['features']  # Dict organized by feature type
```

---

## Integration with VQM24Dataset

After preprocessing, use the `.npz` file with `VQM24Dataset`:
```python
from vqm24_pipeline.datasets import VQM24Dataset

# Load preprocessed wavefunction dataset
dataset = VQM24Dataset(
    root='data/',
    raw_file='processed/wavefunctions.npz',  # Preprocessed file
    # ... other options
)
```

---

## Additional Resources

- **Phase 1 Documentation**: Foundation and base infrastructure
- **Phase 2 Documentation**: Wavefunction implementation details
- **Phase 3 Documentation**: CLI integration (this phase)
- **Main Pipeline Documentation**: `README.md` in project root

---

## Support

For issues, questions, or contributions:
1. Check troubleshooting section above
2. Review phase documentation
3. Check project logs in `logs/` directory
4. Open an issue with detailed error messages

---

**Version**: 1.0  
**Last Updated**: November 2025  
**Author**: VQM24 Pipeline Team
