# MILIA Pipeline: Edge Index Data Corruption Bug Analysis

**Document Version**: 3.0.0  
**Date**: December 2025  
**Status**: ✅ ALL BUGS FIXED & VERIFIED  
**Priority**: CRITICAL (Blocks all training)

---

## 1. Executive Summary

### Problem Statement
HPO training fails with error `BasicGNN.forward() missing 1 required positional argument: 'edge_index'`. This error message is **misleading** — the actual root causes are:

1. **Corrupted `edge_index` data** where node indices are out of bounds
2. **Hardcoded `data.pt` filename** instead of dynamic filename based on config
3. **Path with `~` not expanded** causing file not found errors

### Root Causes (ALL VERIFIED & FIXED)

| Bug | File | Line(s) | Issue |
|-----|------|---------|-------|
| **Bug #1** | `milia_dataset.py` | 3891-3895 | Missing `increment=False` in `collate()` |
| **Bug #2** | `config_constants.py` | 2348-2350 | Hardcoded `'data.pt'` fallback |
| **Bug #3** | `milia_dataset.py` | 512-515 | Path `~` not expanded |

### Verification Results
```
✅ torch.load now looks for: DFT_all_sliced.pt (dynamic)
✅ Dataset loaded with 100 samples
✅ Graph 1: nodes=2, edge_max=1 (VALID - edge_max < nodes)
```

---

## 2. Bug #1: Missing `increment=False` in collate()

### Location
**File:** `milia_pipeline/datasets/milia_dataset.py`  
**Lines:** 3891-3895

### The Problem
The `collate()` method calls `torch_geometric.data.collate.collate()` **without passing `increment=False`**.

When `increment=True` (the default), PyG adds cumulative node offsets to `edge_index` — this is designed for creating Batch objects for DataLoader, NOT for InMemoryDataset storage.

### BEFORE (BUGGY):
```python
data_batch, slices, _ = torch_geometric.data.collate.collate(
    data_list[0].__class__,
    data_list,
    exclude_keys=all_excluded_attrs
    # MISSING: increment=False, add_batch=False
)
```

### AFTER (FIXED):
```python
data_batch, slices, _ = torch_geometric.data.collate.collate(
    data_list[0].__class__,
    data_list,
    increment=False,      # CRITICAL: Don't add cumulative offsets to edge_index
    add_batch=False,      # CRITICAL: Don't add batch attribute for storage
    exclude_keys=all_excluded_attrs
)
```

### Why This Bug Causes Corruption

| Graph | Local edge_index | With increment=True | Problem |
|-------|------------------|---------------------|---------|
| Graph 0 | `[0,1,2]` | `[0,1,2]` (offset=0) | ✓ Works (coincidentally) |
| Graph 1 | `[0,1]` | `[3,4]` (offset=3) | ✗ Invalid for 2-node graph |
| Graph 2 | `[0,1]` | `[5,6]` (offset=5) | ✗ Invalid for 2-node graph |

---

## 3. Bug #2: Hardcoded `'data.pt'` Filename

### Location
**File:** `milia_pipeline/config/config_constants.py`  
**Lines:** 2338-2351

### The Problem
When `PROCESSED_DATA_FILENAME` was accessed before config was fully loaded, it would cache `'data.pt'` as the filename instead of the dynamic name like `'DFT_all_sliced.pt'`.

### BEFORE (BUGGY):
```python
elif name == 'PROCESSED_DATA_FILENAME':
    if name not in _CONSTANTS_CACHE:
        # Get RAW_NPZ_FILENAME_CACHED
        if 'RAW_NPZ_FILENAME_CACHED' not in _CONSTANTS_CACHE:
            constants = get_dataset_constants()
            _CONSTANTS_CACHE['RAW_NPZ_FILENAME_CACHED'] = constants[0]
        
        raw_filename = _CONSTANTS_CACHE['RAW_NPZ_FILENAME_CACHED']
        if raw_filename:
            _CONSTANTS_CACHE[name] = Path(raw_filename).stem + '.pt'
        else:
            logger.warning("The raw NPZ filename is not specified in config.yaml. Falling back to default 'data.pt'.")
            _CONSTANTS_CACHE[name] = 'data.pt'
    return _CONSTANTS_CACHE[name]
```

### AFTER (FIXED):
```python
elif name == 'PROCESSED_DATA_FILENAME':
    if name not in _CONSTANTS_CACHE:
        # Always fetch fresh to ensure config is loaded
        constants = get_dataset_constants()
        raw_filename = constants[0]
        
        if raw_filename:
            _CONSTANTS_CACHE[name] = Path(raw_filename).stem + '.pt'
            # Also update RAW_NPZ_FILENAME_CACHED for consistency
            _CONSTANTS_CACHE['RAW_NPZ_FILENAME_CACHED'] = raw_filename
        else:
            # Retry fetching config - it may not have been loaded on first attempt
            from milia_pipeline.config.config_loader import load_config
            config = load_config()
            dataset_config = _get_dataset_config_local()
            raw_filename = _get_config_value(dataset_config, 'raw_npz_filename', str)
            if raw_filename:
                _CONSTANTS_CACHE[name] = Path(raw_filename).stem + '.pt'
            else:
                logger.warning("The raw NPZ filename is not specified in config.yaml. Falling back to default 'data.pt'.")
                _CONSTANTS_CACHE[name] = 'data.pt'
    return _CONSTANTS_CACHE[name]
```

### Impact
- `torch.load` was looking for `data.pt` but file was saved as `DFT_all_sliced.pt`
- PyG thought file didn't exist and triggered unnecessary reprocessing/download

---

## 4. Bug #3: Path `~` Not Expanded

### Location
**File:** `milia_pipeline/datasets/milia_dataset.py`  
**Lines:** 512-515

### The Problem
When `root` path contained `~` (e.g., `~/Chem_Data/...`), it wasn't expanded to the actual home directory path.

- Config has: `~/Chem_Data/Milia_PyG_Dataset`
- Code looked at: `/app/milia/Chem_Data/Milia_PyG_Dataset` (wrong)
- File was at: `/root/Chem_Data/Milia_PyG_Dataset` (correct)

### BEFORE (BUGGY):
```python
# Initialize root SECOND - always ensure we have a valid root directory
if root is None:
    root = tempfile.mkdtemp()
```

### AFTER (FIXED):
```python
# Initialize root SECOND - always ensure we have a valid root directory
if root is None:
    root = tempfile.mkdtemp()
else:
    # Normalize path: expand ~, environment variables, and resolve to absolute path
    root = str(Path(os.path.expandvars(root)).expanduser().resolve())
```

### Handles All Path Variants
| Input | After Fix |
|-------|-----------|
| `~/Chem_Data/...` | `/root/Chem_Data/...` |
| `$HOME/Chem_Data/...` | `/root/Chem_Data/...` |
| `./Chem_Data/...` | `/app/milia/Chem_Data/...` |
| `../Chem_Data/...` | `/app/Chem_Data/...` |
| `/absolute/path` | `/absolute/path` (unchanged) |

---

## 5. Implementation Guide

### Step 1: Apply All Three Fixes

1. **Fix #1:** `milia_dataset.py` line 3891-3895 — Add `increment=False, add_batch=False`
2. **Fix #2:** `config_constants.py` lines 2338-2351 — Dynamic filename retry logic
3. **Fix #3:** `milia_dataset.py` lines 512-515 — Path expansion

### Step 2: Reprocess the Dataset

```bash
# Remove corrupted processed data
rm -rf ~/Chem_Data/Milia_PyG_Dataset/processed/

# Rerun processing (will regenerate with correct edge_index)
python -c "from milia_pipeline.datasets import miliaDataset; ds = miliaDataset(root='~/Chem_Data/Milia_PyG_Dataset')"
```

### Step 3: Verify All Fixes

```python
from milia_pipeline.datasets import miliaDataset
from torch_geometric.loader import DataLoader
import torch
from pathlib import Path

# Test 1: Dataset loads with correct filename
ds = miliaDataset(root='~/Chem_Data/Milia_PyG_Dataset')
print(f"✅ Dataset loaded with {len(ds)} samples")

# Test 2: Edge index validation
pt_file = Path('~/Chem_Data/Milia_PyG_Dataset/processed/DFT_all_sliced.pt').expanduser()
data, slices = torch.load(pt_file, weights_only=False)
e_s, e_e = slices['edge_index'][1].item(), slices['edge_index'][2].item()
x_s, x_e = slices['x'][1].item(), slices['x'][2].item()
num_nodes = x_e - x_s
edge_max = data.edge_index[:, e_s:e_e].max().item()
print(f"Graph 1: nodes={num_nodes}, edge_max={edge_max}")
assert edge_max < num_nodes, "BROKEN: still global indices"
print("✅ Edge index validation passed")

# Test 3: Individual graph validity
for i in range(min(10, len(ds))):
    g = ds[i]
    assert g.edge_index.max() < g.x.shape[0], f"Graph {i}: edge_index out of bounds"
    print(f"Graph {i}: ✓ Valid (nodes={g.x.shape[0]}, edge_max={g.edge_index.max().item()})")

# Test 4: Batch validity
loader = DataLoader(ds, batch_size=8)
batch = next(iter(loader))
assert batch.edge_index.max() < batch.x.shape[0], "Batch: edge_index out of bounds"
print(f"✅ Batch valid (nodes={batch.x.shape[0]}, edge_max={batch.edge_index.max().item()})")

print("\n🎉 ALL TESTS PASSED - All bugs fixed!")
```

---

## 6. Root Cause Chain (Complete)

```
Bug #2 & #3: File Loading Issues
================================
1. config.yaml has: working_root_dir: ~/Chem_Data/...
   ↓
2. ~ not expanded → looks in wrong directory
   ↓
3. File not found at ./Chem_Data/... (looking for /app/milia/Chem_Data)
   ↓
4. config_constants.py caches 'data.pt' as fallback
   ↓
5. torch.load looks for data.pt instead of DFT_all_sliced.pt
   ↓
6. PyG triggers unnecessary download/reprocess

Bug #1: Edge Index Corruption
=============================
1. milia_dataset.py:collate() called without increment=False
   ↓
2. torch_geometric.data.collate.collate() uses increment=True (default)
   ↓
3. edge_index values get cumulative node offsets added
   ↓
4. Data saved with global indices instead of local indices
   ↓
5. dataset[i] returns graph with wrong edge_index values
   ↓
6. edge_index.max() > num_nodes for all graphs except Graph 0
   ↓
7. Forward pass fails: "index X out of bounds for dimension 0 with size Y"
   ↓
8. Trainer tries fallback strategies, all fail
   ↓
9. Misleading error: "missing 1 required positional argument: 'edge_index'"
```

---

## 7. Files Modified

| File | Lines Changed | Fix |
|------|---------------|-----|
| `milia_pipeline/datasets/milia_dataset.py` | 3891-3895 | Add `increment=False, add_batch=False` |
| `milia_pipeline/datasets/milia_dataset.py` | 512-519 | Path expansion with `expanduser()`, `expandvars()`, `resolve()` |
| `milia_pipeline/config/config_constants.py` | 2338-2351 | Dynamic filename retry logic |

---

## 8. Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2025 | Initial analysis document |
| 1.1.0 | Dec 2025 | Added diagnostic results confirming global vs local index issue |
| 1.2.0 | Dec 2025 | Identified `from_rdmol` as edge_index source, narrowed bug to dataset assembly |
| 2.0.0 | Dec 2025 | ROOT CAUSE #1 FOUND: Missing `increment=False` in collate() |
| **3.0.0** | **Dec 2025** | **ALL BUGS FIXED: Added Bug #2 (hardcoded data.pt) and Bug #3 (path expansion)** |

---

## 9. Quick Reference

### All Three Fixes at a Glance

**Fix #1 — `milia_dataset.py:3891-3895`**
```python
data_batch, slices, _ = torch_geometric.data.collate.collate(
    data_list[0].__class__, data_list, 
    increment=False, add_batch=False,  # ADD THESE
    exclude_keys=all_excluded_attrs
)
```

**Fix #2 — `config_constants.py:2348-2350`**
```python
# Replace hardcoded fallback with dynamic retry logic
# (see full code in Section 3)
```

**Fix #3 — `milia_dataset.py:512-515`**
```python
else:
    root = str(Path(os.path.expandvars(root)).expanduser().resolve())
```

### Quick Validation
```bash
python3 -c "
from milia_pipeline.datasets import miliaDataset
ds = miliaDataset(root='~/Chem_Data/Milia_PyG_Dataset')
print(f'SUCCESS! {len(ds)} samples, edge_max={ds[1].edge_index.max().item()}, nodes={ds[1].x.shape[0]}')
"
```

Expected output:
```
SUCCESS! 100 samples, edge_max=1, nodes=2
```
