# Models Module Lifecycle Guide
**Complete Output Reference for VQM24 Pipeline Models Module**

---

## Document Information

**Document Title:** Models Module Lifecycle Guide  
**Version:** 1.0.0  
**Last Updated:** 2024-11-23  
**Module Version:** v1.1.0 (Phase 7 Extended)  
**Purpose:** Comprehensive reference for all outputs, artifacts, and behaviors of the models module throughout its complete execution lifecycle

**Intended Audience:**
- Developers integrating with the models module
- QA engineers validating model training outputs
- Data scientists analyzing training results
- DevOps engineers deploying models
- New team members learning the system

---

## Table of Contents

1. [Overview](#overview)
2. [Module Configuration](#module-configuration)
3. [Complete Lifecycle Phases](#complete-lifecycle-phases)
4. [Output Directory Structure](#output-directory-structure)
5. [File Formats Reference](#file-formats-reference)
6. [Console Output Examples](#console-output-examples)
7. [Python API Return Objects](#python-api-return-objects)
8. [Integration Examples](#integration-examples)
9. [Troubleshooting](#troubleshooting)
10. [Appendix](#appendix)

---

## Overview

### What This Guide Covers

This guide documents **every output, artifact, log, and file** produced by the models module during a complete training lifecycle, from initialization through deployment. It serves as a definitive reference for:

- Understanding what the models module produces
- Locating specific outputs for debugging
- Validating correct execution
- Integrating with downstream systems
- Monitoring production deployments

### Module Capabilities

The models module (`vqm24_pipeline/models/`) provides:
- **120+ PyTorch Geometric models** across 12 categories
- **Custom architecture building** (Phase 7)
- **Multi-model ensembles** (Phase 7)
- **Complete training infrastructure** with callbacks
- **Hardware acceleration** (GPU/MPS/distributed)
- **Model deployment** and optimization
- **Production monitoring** and drift detection

### Execution Modes

The models module can run in several modes:
1. **Standalone mode**: Independent of data processing pipeline
2. **Integrated mode**: As part of full VQM24 pipeline
3. **Training mode**: Complete training lifecycle
4. **Evaluation mode**: Evaluate existing checkpoints
5. **Deployment mode**: Optimize and deploy models

---

## Module Configuration

### Primary Configuration File

**Location:** `config.yaml` (project root)

**Key Section:**
```yaml
models:
  enabled: true                    # Master switch (line 830)
  
  selection:
    mode: "single"                 # "single", "custom", or "ensemble"
    task_type: "graph_regression"  # Task type
    model_name: "GCN"              # Model name from registry
```

### Configuration Hierarchy

```
models: (line 829)
├── enabled: true                  # Master enable/disable
├── selection: (lines 836-851)     # Model selection
├── custom_architecture: (lines 859-924)  # Phase 7: Custom builds
├── ensemble: (lines 930-965)      # Phase 7: Ensembles
├── hyperparameters: (lines 967-999)  # Model hyperparameters
├── training: (lines 1000-1078)    # Training configuration
├── evaluation: (lines 1079-1096)  # Evaluation settings
├── acceleration: (lines 1102-1162)  # Hardware acceleration
├── deployment: (lines 1168-1239)  # Deployment settings
└── plugins: (lines 1245-1251)     # Plugin system
```

### Enabling the Models Module

**Requirement:** `enabled: true` at line 830 of `config.yaml`

**Verification:**
```python
from vqm24_pipeline.models.utils.config_bridge import is_models_enabled

if is_models_enabled():
    print("✓ Models module is enabled")
else:
    print("✗ Models module is disabled")
```

---

## Complete Lifecycle Phases

### Phase 1: Module Initialization

**Trigger:** First import of `vqm24_pipeline.models`

**Location:** `vqm24_pipeline/models/__init__.py` (lines 584-607)

#### Console Output
```
INFO: VQM24 Models Module v1.1.0 initialized (Phase 7 Extended)
INFO: All components loaded successfully, including builders module
INFO: Model registry: 120 models available
DEBUG: Category breakdown: {'BASIC_GNN': 8, 'CONVOLUTIONAL': 15, 'ATTENTION': 12, ...}
```

#### In-Memory Outputs
- **ModelRegistry singleton**: 120+ models registered and indexed
- **LayerRegistry**: Catalog of available GNN layers (Phase 7)
- **LossRegistry**: Available loss functions
- **OptimizerRegistry**: Available optimizers
- **SchedulerRegistry**: Available LR schedulers

#### Verification
```python
from vqm24_pipeline.models import get_module_info, print_module_summary

# Get module information
info = get_module_info()
print(f"Total models: {info['total_models']}")
print(f"Builders available: {info['builders_available']}")

# Print formatted summary
print_module_summary()
```

#### Expected Output
```
======================================================================
VQM24 Models Module v1.1.0
======================================================================

Total Models Available: 120
Model Categories: 12

Category Breakdown:
  BASIC_GNN                     :   8 models
  CONVOLUTIONAL                 :  15 models
  ATTENTION                     :  12 models
  MESSAGE_PASSING               :  10 models
  ...

Training Infrastructure:
  Loss Functions: 12
  Optimizers: 8
  Schedulers: 6

Phase 7 Features:
  Builders Module: ✓ Enabled
    - Custom Architectures: ✓
    - Ensemble Models: ✓
    - Architecture Templates: ✓
    - Layer Registry: ✓
======================================================================
```

---

### Phase 2: Model Discovery & Selection

**Trigger:** User queries available models or selects a model

**Location:** `vqm24_pipeline/models/registry/model_registry.py`

#### API Calls

**List Models by Task:**
```python
from vqm24_pipeline.models import list_models

models = list_models(task_type="graph_regression")
# Returns: ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'TransformerConv', ...]
```

**Get Model Information:**
```python
from vqm24_pipeline.models import get_model_info

info = get_model_info("GCN")
```

#### Output Object (Model Info)
```python
{
    'name': 'GCN',
    'category': 'BASIC_GNN',
    'import_path': 'torch_geometric.nn.models.GCN',
    'description': 'Graph Convolutional Network',
    'supported_tasks': [
        'node_classification',
        'node_regression', 
        'graph_regression',
        'graph_classification'
    ],
    'hyperparameters': {
        'in_channels': {
            'type': 'integer',
            'required': True,
            'min': 1,
            'description': 'Input feature dimensions'
        },
        'hidden_channels': {
            'type': 'integer',
            'required': True,
            'min': 1,
            'description': 'Hidden layer dimensions'
        },
        'num_layers': {
            'type': 'integer',
            'required': False,
            'default': 2,
            'min': 1,
            'max': 10,
            'description': 'Number of GCN layers'
        },
        'dropout': {
            'type': 'float',
            'required': False,
            'default': 0.0,
            'min': 0.0,
            'max': 1.0,
            'description': 'Dropout probability'
        }
    },
    'tags': ['basic', 'convolutional', 'spectral'],
    'requires_edge_index': True,
    'requires_edge_features': False,
    'requires_edge_weights': False,
    'supports_heterogeneous': False,
    'supports_directed': True,
    'is_builtin': True,
    'plugin_name': None
}
```

#### Search Models
```python
from vqm24_pipeline.models import search_models

# Search by tags
attention_models = search_models(tags=["attention"])

# Search by category
basic_models = search_models(category="BASIC_GNN")
```

---

### Phase 3: Model Creation

**Trigger:** User calls `create_model()` or factory creates model

**Location:** `vqm24_pipeline/models/factory/model_factory.py` (lines 468-493)

#### Input Requirements
```python
from vqm24_pipeline.models import create_model
import torch
from torch_geometric.data import Data

# Minimal sample data for channel inference
sample_data = Data(
    x=torch.randn(10, 16),              # 10 nodes, 16 features
    edge_index=torch.randint(0, 10, (2, 20))  # 20 edges
)

# Create model
model = create_model(
    name="GCN",
    hyperparameters={
        "hidden_channels": 64,
        "num_layers": 3,
        "dropout": 0.5
    },
    task_type="graph_regression",
    sample_data=sample_data,
    device=torch.device('cuda:0')
)
```

#### Console Output
```
DEBUG: Inferred in_channels=16 from sample data
DEBUG: Inferred out_channels=1 from task type (graph_regression)
DEBUG: Processing hyperparameters for model 'GCN'
DEBUG: Applying defaults: act=relu, jk=None
INFO: Model created successfully: GCN (45,632 parameters) in 0.23s
DEBUG: Moved model to device: cuda:0
```

#### Output Object
**Type:** `torch.nn.Module` (PyTorch Geometric model instance)

**Attributes:**
```python
# Model configuration
model.in_channels      # 16
model.hidden_channels  # 64
model.out_channels     # 1
model.num_layers       # 3
model.dropout          # 0.5

# Model state
type(model)            # <class 'torch_geometric.nn.models.GCN'>
next(model.parameters()).device  # cuda:0
sum(p.numel() for p in model.parameters())  # 45632 (total parameters)
```

#### Parameter Count Breakdown
```
Layer-wise parameter count:
  conv1 (GCNConv): 1,088 parameters (16 → 64)
  conv2 (GCNConv): 4,160 parameters (64 → 64)
  conv3 (GCNConv): 4,160 parameters (64 → 64)
  lin (Linear): 65 parameters (64 → 1)
  Total: 45,632 parameters
```

---

### Phase 4: Data Splitting

**Trigger:** User splits dataset before training

**Location:** `vqm24_pipeline/models/training/data_splitting.py`

#### API Call
```python
from vqm24_pipeline.models import DataSplitter

train_data, val_data, test_data = DataSplitter.random_split(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
)
```

#### Console Output
```
INFO: Performing random split with seed=42
INFO: Dataset size: 1000 samples
INFO: Split ratios: train=0.8, val=0.1, test=0.1
INFO: Data split complete: 800 train, 100 val, 100 test samples
```

#### Output Objects
```python
# Returns tuple of three Subset objects
train_data  # torch.utils.data.Subset (800 samples)
val_data    # torch.utils.data.Subset (100 samples)
test_data   # torch.utils.data.Subset (100 samples)

# Each subset contains:
len(train_data)         # 800
train_data.dataset      # Original dataset reference
train_data.indices      # array([234, 12, 789, ...])
```

#### Splitting Strategies
```python
# 1. Random split (default)
DataSplitter.random_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

# 2. Stratified split (for classification)
DataSplitter.stratified_split(dataset, y_values=labels, train_ratio=0.8)

# 3. Temporal split (time-series data)
DataSplitter.temporal_split(dataset, timestamps=timestamps, train_ratio=0.8)

# 4. Scaffold split (molecular data)
DataSplitter.scaffold_split(dataset, smiles_list=smiles, train_ratio=0.8)
```

---

### Phase 5: Training Loop

**Trigger:** User calls `trainer.fit()`

**Location:** `vqm24_pipeline/models/training/trainer.py`

**Configuration:** `config.yaml` lines 1000-1078

#### Training Setup
```python
from vqm24_pipeline.models import Trainer, EarlyStopping, ModelCheckpoint
import torch.nn as nn
import torch.optim as optim

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=nn.MSELoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    device=torch.device('cuda:0'),
    callbacks=[
        EarlyStopping(patience=20, min_delta=0.0001),
        ModelCheckpoint(save_top_k=3, save_last=True)
    ]
)

# Start training
results = trainer.fit(epochs=100)
```

#### Console Output (Epoch-by-Epoch)
```
================================================================================
Starting Training: GCN (45,632 parameters)
================================================================================
Task: graph_regression
Device: cuda:0
Loss Function: MSELoss()
Optimizer: Adam (lr=0.001, weight_decay=0.0001)

Training Set: 800 samples (10 batches)
Validation Set: 100 samples (2 batches)
Test Set: 100 samples (2 batches)

Callbacks:
  ✓ EarlyStopping (patience=20, min_delta=0.0001)
  ✓ ModelCheckpoint (save_top_k=3, save_last=True)
  ✓ TensorBoard (log_dir=./logs)
  ✓ LearningRateMonitor
  ✓ ProgressBar

================================================================================

Epoch 1/100:
  ├─ Train Loss: 2.3456 │ Train MAE: 1.234 │ Train R²: -0.156
  ├─ Val Loss:   2.1234 │ Val MAE:   1.123 │ Val R²:   -0.089
  ├─ LR: 0.001000
  ├─ Time: 12.5s
  └─ ✓ New best model (val_loss=2.1234)

Epoch 2/100:
  ├─ Train Loss: 1.8765 │ Train MAE: 0.987 │ Train R²: 0.234
  ├─ Val Loss:   1.7654 │ Val MAE:   0.876 │ Val R²:   0.312
  ├─ LR: 0.001000
  ├─ Time: 11.8s
  └─ ✓ New best model (val_loss=1.7654)

Epoch 3/100:
  ├─ Train Loss: 1.5432 │ Train MAE: 0.823 │ Train R²: 0.456
  ├─ Val Loss:   1.4567 │ Val MAE:   0.734 │ Val R²:   0.478
  ├─ LR: 0.001000
  ├─ Time: 11.6s
  └─ ✓ New best model (val_loss=1.4567)

...

Epoch 67/100:
  ├─ Train Loss: 0.2145 │ Train MAE: 0.312 │ Train R²: 0.892
  ├─ Val Loss:   0.2187 │ Val MAE:   0.315 │ Val R²:   0.876
  ├─ LR: 0.000543
  ├─ Time: 10.2s
  └─ ✓ New best model (val_loss=0.2187)

Epoch 68/100:
  ├─ Train Loss: 0.2123 │ Train MAE: 0.308 │ Train R²: 0.894
  ├─ Val Loss:   0.2201 │ Val MAE:   0.318 │ Val R²:   0.874
  ├─ LR: 0.000532
  └─ Time: 10.1s

...

Epoch 87/100:
  ├─ Train Loss: 0.2089 │ Train MAE: 0.304 │ Train R²: 0.897
  ├─ Val Loss:   0.2245 │ Val MAE:   0.322 │ Val R²:   0.871
  ├─ LR: 0.000421
  └─ Time: 9.8s
  ⚠ EarlyStopping: No improvement for 20 epochs

================================================================================
Training Complete!
================================================================================
Total Time: 20m 35s (1235.6 seconds)
Total Epochs: 87 (stopped early)
Best Epoch: 67
Best Val Loss: 0.2187

Final Metrics:
  Train Loss: 0.2089 │ Train MAE: 0.304 │ Train R²: 0.897
  Val Loss:   0.2245 │ Val MAE: 0.322 │ Val R²: 0.871

Checkpoints Saved:
  ✓ best_model_epoch=67_val_loss=0.219.pth
  ✓ best_model_epoch=52_val_loss=0.223.pth
  ✓ best_model_epoch=45_val_loss=0.234.pth
  ✓ last_model.pth

TensorBoard Logs: ./logs/GCN_graph_regression_2024-11-23_10-15-23
================================================================================
```

#### Progress Bar Output
```
Training: 100%|██████████████████████████| 87/100 [20:35<00:00, 14.19s/epoch]
```

---

### Phase 6: File Outputs (Checkpoints & Logs)

#### A. Model Checkpoints

**Configuration:** `config.yaml` lines 1043-1050
```yaml
model_checkpoint:
  enabled: true
  params:
    monitor: "val_loss"
    save_top_k: 3          # Save best 3 checkpoints
    mode: "min"
    save_last: true
    dirpath: null          # Auto-generate path
```

**Directory:** `~/Chem_Data/VQM24_PyG_Dataset/checkpoints/GCN_graph_regression_2024-11-23_10-15-23/`

**Files Created:**
```
checkpoints/
└── GCN_graph_regression_2024-11-23_10-15-23/
    ├── best_model_epoch=67_val_loss=0.219.pth    # Best checkpoint
    ├── best_model_epoch=52_val_loss=0.223.pth    # 2nd best
    ├── best_model_epoch=45_val_loss=0.234.pth    # 3rd best
    ├── last_model.pth                             # Last epoch
    └── checkpoint_metadata.json                   # Metadata
```

**Checkpoint File Structure (`.pth`):**
```python
# Load checkpoint
checkpoint = torch.load('best_model_epoch=67_val_loss=0.219.pth')

# Checkpoint contents
{
    # Training state
    'epoch': 67,
    'global_step': 6700,
    
    # Model weights
    'model_state_dict': OrderedDict([
        ('conv1.lin.weight', tensor([...])),
        ('conv1.bias', tensor([...])),
        ('conv2.lin.weight', tensor([...])),
        # ... all model parameters
    ]),
    
    # Optimizer state
    'optimizer_state_dict': {
        'state': {...},
        'param_groups': [{
            'lr': 0.000543,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0.0001,
            'params': [...]
        }]
    },
    
    # Scheduler state
    'scheduler_state_dict': {
        'last_epoch': 67,
        'best': 0.2187,
        'num_bad_epochs': 0,
        '_last_lr': [0.000543]
    },
    
    # Metrics
    'train_loss': 0.2145,
    'val_loss': 0.2187,
    'train_mae': 0.312,
    'val_mae': 0.315,
    'train_r2': 0.892,
    'val_r2': 0.876,
    'best_val_loss': 0.2187,
    
    # Metadata
    'model_name': 'GCN',
    'task_type': 'graph_regression',
    'hyperparameters': {
        'in_channels': 16,
        'hidden_channels': 64,
        'out_channels': 1,
        'num_layers': 3,
        'dropout': 0.5
    },
    'total_parameters': 45632,
    'timestamp': '2024-11-23T10:28:45.123456',
    'config_path': 'config.yaml',
    'pytorch_version': '2.1.0',
    'cuda_version': '12.1'
}
```

**Metadata File (`checkpoint_metadata.json`):**
```json
{
    "training_session": {
        "session_id": "GCN_graph_regression_2024-11-23_10-15-23",
        "start_time": "2024-11-23T10:15:23.456789",
        "end_time": "2024-11-23T10:35:58.789012",
        "total_duration_seconds": 1235.332,
        "early_stopped": true,
        "stop_reason": "EarlyStopping (patience=20)"
    },
    "model": {
        "name": "GCN",
        "task_type": "graph_regression",
        "total_parameters": 45632,
        "trainable_parameters": 45632,
        "non_trainable_parameters": 0
    },
    "training": {
        "total_epochs": 87,
        "best_epoch": 67,
        "best_val_loss": 0.2187,
        "final_train_loss": 0.2089,
        "final_val_loss": 0.2245
    },
    "data": {
        "train_samples": 800,
        "val_samples": 100,
        "test_samples": 100,
        "batch_size": 80
    },
    "checkpoints": {
        "best": "best_model_epoch=67_val_loss=0.219.pth",
        "last": "last_model.pth",
        "top_k": [
            {"epoch": 67, "val_loss": 0.2187, "file": "best_model_epoch=67_val_loss=0.219.pth"},
            {"epoch": 52, "val_loss": 0.2234, "file": "best_model_epoch=52_val_loss=0.223.pth"},
            {"epoch": 45, "val_loss": 0.2345, "file": "best_model_epoch=45_val_loss=0.234.pth"}
        ]
    },
    "hardware": {
        "device": "cuda:0",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "cuda_version": "12.1",
        "pytorch_version": "2.1.0"
    }
}
```

#### B. TensorBoard Logs

**Configuration:** `config.yaml` lines 1052-1055
```yaml
tensorboard:
  enabled: true
  params:
    log_dir: null  # Auto-generate
```

**Directory:** `~/Chem_Data/VQM24_PyG_Dataset/logs/GCN_graph_regression_2024-11-23_10-15-23/`

**Files Created:**
```
logs/
└── GCN_graph_regression_2024-11-23_10-15-23/
    └── events.out.tfevents.1700740523.hostname.12345.0
```

**Viewing TensorBoard Logs:**
```bash
# Start TensorBoard server
tensorboard --logdir ~/Chem_Data/VQM24_PyG_Dataset/logs

# Navigate to http://localhost:6006
```

**Logged Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `train/loss` | Scalar | Training loss per epoch |
| `train/mae` | Scalar | Training MAE per epoch |
| `train/r2` | Scalar | Training R² score per epoch |
| `val/loss` | Scalar | Validation loss per epoch |
| `val/mae` | Scalar | Validation MAE per epoch |
| `val/r2` | Scalar | Validation R² score per epoch |
| `learning_rate` | Scalar | LR per epoch |
| `epoch_time` | Scalar | Time per epoch (seconds) |
| `gpu_memory` | Scalar | GPU memory usage (MB) |

**TensorBoard Dashboard Sections:**
1. **SCALARS**: All numeric metrics over time
2. **GRAPHS**: Model architecture visualization
3. **DISTRIBUTIONS**: Parameter distributions (if enabled)
4. **HISTOGRAMS**: Gradient histograms (if enabled)
5. **IMAGES**: Sample predictions (if enabled)

#### C. Training Logs

**Configuration:** `config.yaml` lines 1072-1077
```yaml
logging:
  log_every_n_steps: 50
  log_metrics: true
  log_gradients: false
  log_weights: false
```

**File:** `~/Chem_Data/VQM24_PyG_Dataset/logs/training_2024-11-23.log`

**Log Format:**
```
2024-11-23 10:15:23.456 INFO [models.factory] Model created successfully: GCN (45,632 parameters) in 0.23s
2024-11-23 10:15:23.567 INFO [models.training] Starting training: GCN
2024-11-23 10:15:23.568 INFO [models.training] Task: graph_regression | Device: cuda:0
2024-11-23 10:15:23.569 INFO [models.training] Training set: 800 samples | Validation set: 100 samples
2024-11-23 10:15:23.570 INFO [models.training] Optimizer: Adam(lr=0.001, weight_decay=0.0001)
2024-11-23 10:15:23.571 INFO [models.training] Loss function: MSELoss()
2024-11-23 10:15:35.789 INFO [models.training] Epoch 1/100 - Train Loss: 2.3456, Val Loss: 2.1234
2024-11-23 10:15:35.790 INFO [models.callbacks] ModelCheckpoint: Saved best model (val_loss=2.1234)
2024-11-23 10:15:47.123 INFO [models.training] Epoch 2/100 - Train Loss: 1.8765, Val Loss: 1.7654
2024-11-23 10:15:47.124 INFO [models.callbacks] ModelCheckpoint: Saved best model (val_loss=1.7654)
...
2024-11-23 10:35:45.234 INFO [models.training] Epoch 87/100 - Train Loss: 0.2089, Val Loss: 0.2245
2024-11-23 10:35:45.235 WARN [models.callbacks] EarlyStopping: No improvement for 20 epochs, stopping training
2024-11-23 10:35:58.456 INFO [models.training] Training completed in 20m 35s
2024-11-23 10:35:58.457 INFO [models.training] Best epoch: 67 (val_loss=0.2187)
2024-11-23 10:35:58.458 INFO [models.training] Final metrics - Train Loss: 0.2089, Val Loss: 0.2245
2024-11-23 10:35:58.459 INFO [models.callbacks] Saved checkpoint metadata to: checkpoints/.../checkpoint_metadata.json
```

**Log Levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages (non-critical)
- `ERROR`: Error messages (critical failures)

---

### Phase 7: Model Evaluation

**Trigger:** User calls `trainer.test()` or evaluation after training

**Configuration:** `config.yaml` lines 1079-1096

#### API Call
```python
# Evaluate on test set
test_results = trainer.test(test_loader=test_loader)
```

#### Console Output
```
================================================================================
Evaluating Model on Test Set
================================================================================
Loading best model from: checkpoints/.../best_model_epoch=67_val_loss=0.219.pth
Model loaded successfully (epoch 67)

Test Set: 100 samples (2 batches)
Device: cuda:0

Evaluating: 100%|████████████████████████████| 2/2 [00:02<00:00, 1.23batch/s]

================================================================================
Test Results
================================================================================
Metrics:
  MSE:  0.2234
  MAE:  0.3123
  R²:   0.8764
  RMSE: 0.4727

Evaluation Time: 2.3 seconds
Samples/second: 43.5

Predictions saved to: predictions/GCN_graph_regression_2024-11-23_10-35-58/
================================================================================
```

#### Evaluation Metrics by Task Type

**Regression Tasks:**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- RMSE (Root Mean Squared Error)

**Classification Tasks:**
- Accuracy
- Precision
- Recall
- F1-Score
- AUROC (Area Under ROC Curve)
- Confusion Matrix

---

### Phase 8: Predictions Output

**Configuration:** `config.yaml` lines 1093-1096
```yaml
test_after_training: true
save_predictions: true
predictions_dir: null  # Auto-generate
```

**Directory:** `~/Chem_Data/VQM24_PyG_Dataset/predictions/GCN_graph_regression_2024-11-23_10-35-58/`

**Files Created:**
```
predictions/
└── GCN_graph_regression_2024-11-23_10-35-58/
    ├── test_predictions.csv            # CSV format
    ├── test_predictions.npz            # NumPy format
    ├── prediction_metadata.json        # Metadata
    └── visualization/                   # Optional visualizations
        ├── predictions_scatter.png
        ├── residuals_plot.png
        └── error_distribution.png
```

#### A. CSV Predictions (`test_predictions.csv`)

**Format:**
```csv
sample_id,compound_id,true_value,predicted_value,absolute_error,squared_error,relative_error
0,mol_001,-0.2345,-0.2187,0.0158,0.00024964,6.74%
1,mol_002,1.4567,1.4234,0.0333,0.00110889,2.29%
2,mol_003,-1.2345,-1.2678,0.0333,0.00110889,2.70%
3,mol_004,0.8765,0.8912,0.0147,0.00021609,1.68%
4,mol_005,2.1234,2.0987,0.0247,0.00061009,1.16%
...
```

**Columns:**
- `sample_id`: Index in test set (0-based)
- `compound_id`: Original compound identifier (if available)
- `true_value`: Ground truth target value
- `predicted_value`: Model prediction
- `absolute_error`: |true - predicted|
- `squared_error`: (true - predicted)²
- `relative_error`: (|true - predicted| / |true|) × 100%

**Statistics Footer:**
```csv
# Summary Statistics
# Total Samples: 100
# MSE: 0.2234
# MAE: 0.3123
# R²: 0.8764
# RMSE: 0.4727
```

#### B. NumPy Predictions (`test_predictions.npz`)

**Format:**
```python
import numpy as np

# Load predictions
data = np.load('test_predictions.npz')

# Available arrays
print(data.files)
# ['sample_ids', 'compound_ids', 'true_values', 'predictions', 
#  'absolute_errors', 'squared_errors', 'relative_errors']

# Access data
sample_ids = data['sample_ids']        # shape: (100,)
true_values = data['true_values']      # shape: (100,)
predictions = data['predictions']      # shape: (100,)
errors = data['absolute_errors']       # shape: (100,)
```

#### C. Prediction Metadata (`prediction_metadata.json`)

```json
{
    "evaluation_session": {
        "session_id": "GCN_graph_regression_2024-11-23_10-35-58",
        "timestamp": "2024-11-23T10:35:58.789012",
        "duration_seconds": 2.3
    },
    "model": {
        "name": "GCN",
        "task_type": "graph_regression",
        "checkpoint_path": "checkpoints/.../best_model_epoch=67_val_loss=0.219.pth",
        "checkpoint_epoch": 67,
        "total_parameters": 45632
    },
    "test_set": {
        "total_samples": 100,
        "batch_size": 50,
        "num_batches": 2
    },
    "metrics": {
        "mse": 0.2234,
        "mae": 0.3123,
        "r2": 0.8764,
        "rmse": 0.4727
    },
    "statistics": {
        "true_values": {
            "mean": 0.4567,
            "std": 0.8912,
            "min": -1.2345,
            "max": 2.1234
        },
        "predictions": {
            "mean": 0.4534,
            "std": 0.8734,
            "min": -1.2678,
            "max": 2.0987
        },
        "errors": {
            "mean": 0.3123,
            "std": 0.2456,
            "median": 0.2789,
            "min": 0.0012,
            "max": 0.9876,
            "percentile_95": 0.7654
        }
    },
    "output_files": {
        "csv": "test_predictions.csv",
        "npz": "test_predictions.npz",
        "visualizations": [
            "visualization/predictions_scatter.png",
            "visualization/residuals_plot.png",
            "visualization/error_distribution.png"
        ]
    },
    "hardware": {
        "device": "cuda:0",
        "inference_time_per_sample_ms": 23.0
    }
}
```

#### D. Visualizations (Optional)

**Configuration:**
```yaml
evaluation:
  save_visualizations: true  # Optional feature
```

**Generated Plots:**

1. **Predictions Scatter (`predictions_scatter.png`)**
   - X-axis: True values
   - Y-axis: Predicted values
   - Diagonal line: Perfect predictions
   - Points: Individual predictions
   - Color: Absolute error magnitude

2. **Residuals Plot (`residuals_plot.png`)**
   - X-axis: Predicted values
   - Y-axis: Residuals (true - predicted)
   - Horizontal line at y=0
   - Shows prediction bias

3. **Error Distribution (`error_distribution.png`)**
   - Histogram of absolute errors
   - Shows error distribution pattern
   - Includes mean, median, 95th percentile

---

### Phase 9: Training Results Object

**Output:** Returned by `trainer.fit()`

**Type:** Python dictionary

```python
results = trainer.fit(epochs=100)

# Complete results structure
{
    # Training metadata
    'training_time': 1235.6,          # Total seconds
    'total_epochs': 87,               # Actual epochs (may stop early)
    'requested_epochs': 100,          # Originally requested
    'early_stopped': True,
    'stop_reason': 'EarlyStopping: patience=20',
    
    # Best model information
    'best_epoch': 67,
    'best_val_loss': 0.2187,
    'best_val_mae': 0.315,
    'best_val_r2': 0.876,
    
    # Final metrics
    'final_train_loss': 0.2089,
    'final_train_mae': 0.304,
    'final_train_r2': 0.897,
    'final_val_loss': 0.2245,
    'final_val_mae': 0.322,
    'final_val_r2': 0.871,
    
    # Test metrics (if test_after_training=true)
    'test_metrics': {
        'mse': 0.2234,
        'mae': 0.3123,
        'r2': 0.8764,
        'rmse': 0.4727
    },
    
    # File paths
    'checkpoint_path': 'checkpoints/.../best_model_epoch=67_val_loss=0.219.pth',
    'checkpoint_dir': 'checkpoints/GCN_graph_regression_2024-11-23_10-15-23/',
    'logs_dir': 'logs/GCN_graph_regression_2024-11-23_10-15-23/',
    'tensorboard_log': 'logs/.../events.out.tfevents...',
    'predictions_path': 'predictions/.../test_predictions.csv',
    
    # Training history (per epoch)
    'history': {
        'train_loss': [2.3456, 1.8765, 1.5432, ..., 0.2089],
        'train_mae': [1.234, 0.987, 0.823, ..., 0.304],
        'train_r2': [-0.156, 0.234, 0.456, ..., 0.897],
        'val_loss': [2.1234, 1.7654, 1.4567, ..., 0.2245],
        'val_mae': [1.123, 0.876, 0.734, ..., 0.322],
        'val_r2': [-0.089, 0.312, 0.478, ..., 0.871],
        'learning_rate': [0.001, 0.001, 0.00098, ..., 0.000421],
        'epoch_time': [12.5, 11.8, 11.6, ..., 9.8]
    },
    
    # Callbacks information
    'callbacks_executed': [
        'EarlyStopping',
        'ModelCheckpoint',
        'TensorBoardLogger',
        'LearningRateMonitor',
        'ProgressBar'
    ],
    
    # Hardware information
    'device': 'cuda:0',
    'gpu_memory_peak_mb': 3456.78,
    'gpu_utilization_avg': 0.87,
    
    # Dataset information
    'train_samples': 800,
    'val_samples': 100,
    'test_samples': 100,
    'batch_size': 80,
    'num_train_batches': 10,
    'num_val_batches': 2,
    
    # Model information
    'model_name': 'GCN',
    'task_type': 'graph_regression',
    'total_parameters': 45632,
    'trainable_parameters': 45632
}
```

---

### Phase 10: Deployment (Optional)

**Configuration:** `config.yaml` lines 1168-1239

**Enabled:** Set `deployment.enabled: true`

#### A. Model Optimization

**Quantization Output:**
```
deployment/
└── optimized_models/
    ├── model_quantized_int8.pth          # Quantized model
    ├── model_quantized_fp16.pth          # FP16 model
    └── quantization_report.json          # Optimization report
```

**Quantization Report (`quantization_report.json`):**
```json
{
    "optimization": "quantization",
    "method": "dynamic",
    "dtype": "qint8",
    "original_model": {
        "size_mb": 0.178,
        "parameters": 45632
    },
    "quantized_model": {
        "size_mb": 0.045,
        "parameters": 45632,
        "compression_ratio": 3.96
    },
    "performance": {
        "accuracy_before": 0.876,
        "accuracy_after": 0.873,
        "accuracy_drop": 0.003,
        "inference_speedup": 2.34
    },
    "timestamp": "2024-11-23T10:40:12.345678"
}
```

**Pruning Output:**
```
deployment/
└── optimized_models/
    ├── model_pruned_30pct.pth            # Pruned model (30% sparsity)
    └── pruning_report.json               # Pruning report
```

**Pruning Report (`pruning_report.json`):**
```json
{
    "optimization": "pruning",
    "method": "magnitude",
    "sparsity": 0.30,
    "original_model": {
        "parameters": 45632,
        "non_zero_parameters": 45632
    },
    "pruned_model": {
        "parameters": 45632,
        "non_zero_parameters": 31942,
        "zero_parameters": 13690,
        "actual_sparsity": 0.30
    },
    "performance": {
        "accuracy_before": 0.876,
        "accuracy_after": 0.869,
        "accuracy_drop": 0.007,
        "size_reduction_mb": 0.053
    },
    "timestamp": "2024-11-23T10:42:34.567890"
}
```

#### B. Deployment Monitoring

**Configuration:** `deployment.monitoring.enabled: true` (line 1221)

**Monitoring Outputs:**
```
deployment/
└── monitoring/
    ├── inference_latency.log             # Latency tracking
    ├── drift_detection.log               # Distribution drift
    ├── performance_metrics.json          # Real-time metrics
    └── alerts.log                        # System alerts
```

**Inference Latency Log (`inference_latency.log`):**
```
2024-11-23 11:00:00.123 INFO Batch inference: 32 samples in 45.6ms (1.425ms/sample)
2024-11-23 11:00:05.456 INFO Batch inference: 64 samples in 87.3ms (1.364ms/sample)
2024-11-23 11:00:10.789 INFO Batch inference: 128 samples in 172.5ms (1.348ms/sample)
2024-11-23 11:00:15.012 INFO Single inference: 1 sample in 2.1ms
2024-11-23 11:00:20.345 WARN Slow inference: 1 sample in 15.6ms (threshold: 5ms)
```

**Performance Metrics (`performance_metrics.json`):**
```json
{
    "last_updated": "2024-11-23T11:15:45.678901",
    "inference_stats": {
        "total_inferences": 15234,
        "avg_latency_ms": 1.42,
        "p50_latency_ms": 1.35,
        "p95_latency_ms": 2.87,
        "p99_latency_ms": 4.23,
        "throughput_samples_per_sec": 704.2
    },
    "accuracy_stats": {
        "recent_mae": 0.315,
        "recent_r2": 0.874,
        "baseline_mae": 0.312,
        "baseline_r2": 0.876,
        "mae_drift": 0.003,
        "r2_drift": -0.002
    },
    "resource_usage": {
        "avg_gpu_memory_mb": 1234.5,
        "peak_gpu_memory_mb": 1567.8,
        "avg_gpu_utilization": 0.45,
        "avg_cpu_utilization": 0.23
    },
    "error_rate": {
        "total_errors": 12,
        "error_rate": 0.00079,
        "recent_errors": [
            {
                "timestamp": "2024-11-23T11:10:23.456",
                "error_type": "TimeoutError",
                "message": "Inference timeout after 30s"
            }
        ]
    }
}
```

**Drift Detection Log (`drift_detection.log`):**
```
2024-11-23 10:00:00.000 INFO Drift detection initialized (method=statistical, threshold=0.1)
2024-11-23 11:00:00.123 INFO Feature drift check: drift_score=0.023 (PASS)
2024-11-23 12:00:00.234 WARN Feature drift detected: drift_score=0.156 (threshold=0.1)
2024-11-23 12:00:00.235 INFO Drift details: features=['x_5', 'x_12', 'x_18'] exceeded threshold
2024-11-23 12:00:00.236 WARN Recommendation: Consider model retraining
2024-11-23 13:00:00.345 INFO Prediction drift check: mae_drift=0.045 (PASS)
```

---

## Output Directory Structure

### Complete Directory Tree

```
~/Chem_Data/VQM24_PyG_Dataset/
│
├── checkpoints/                                 # Model checkpoints
│   └── GCN_graph_regression_2024-11-23_10-15-23/
│       ├── best_model_epoch=67_val_loss=0.219.pth
│       ├── best_model_epoch=52_val_loss=0.223.pth
│       ├── best_model_epoch=45_val_loss=0.234.pth
│       ├── last_model.pth
│       └── checkpoint_metadata.json
│
├── logs/                                        # Training logs
│   ├── GCN_graph_regression_2024-11-23_10-15-23/
│   │   └── events.out.tfevents.1700740523.hostname.12345.0
│   └── training_2024-11-23.log
│
├── predictions/                                 # Model predictions
│   └── GCN_graph_regression_2024-11-23_10-35-58/
│       ├── test_predictions.csv
│       ├── test_predictions.npz
│       ├── prediction_metadata.json
│       └── visualization/
│           ├── predictions_scatter.png
│           ├── residuals_plot.png
│           └── error_distribution.png
│
└── deployment/                                  # Deployment artifacts (optional)
    ├── optimized_models/
    │   ├── model_quantized_int8.pth
    │   ├── model_quantized_fp16.pth
    │   ├── model_pruned_30pct.pth
    │   ├── quantization_report.json
    │   └── pruning_report.json
    │
    └── monitoring/
        ├── inference_latency.log
        ├── drift_detection.log
        ├── performance_metrics.json
        └── alerts.log
```

### Directory Organization Patterns

**By Timestamp:**
```
checkpoints/
├── GCN_graph_regression_2024-11-23_10-15-23/
├── GCN_graph_regression_2024-11-23_14-30-45/
└── GAT_graph_regression_2024-11-24_09-00-12/
```

**By Model Name:**
```
checkpoints/
├── GCN/
│   ├── 2024-11-23_10-15-23/
│   └── 2024-11-23_14-30-45/
└── GAT/
    └── 2024-11-24_09-00-12/
```

**Configuration:** Set in `config.yaml` or via `ModelCheckpoint` callback parameters.

---

## File Formats Reference

### 1. PyTorch Checkpoint Files (`.pth`)

**Format:** PyTorch serialized dictionary

**Loading:**
```python
import torch

# Load checkpoint
checkpoint = torch.load('best_model_epoch=67_val_loss=0.219.pth')

# Restore model
model = create_model("GCN", hyperparameters, task_type, sample_data)
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Restore scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Get epoch
start_epoch = checkpoint['epoch'] + 1
```

**Best Practices:**
- Always load to same device used for saving
- Verify checkpoint integrity before loading
- Handle version compatibility issues

### 2. JSON Metadata Files (`.json`)

**Format:** JSON (JavaScript Object Notation)

**Loading:**
```python
import json

with open('checkpoint_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Best epoch: {metadata['training']['best_epoch']}")
print(f"Best val loss: {metadata['training']['best_val_loss']}")
```

**Best Practices:**
- Human-readable format
- Easy to parse in any language
- Good for configuration and metadata

### 3. NumPy Archives (`.npz`)

**Format:** NumPy compressed archive

**Loading:**
```python
import numpy as np

# Load predictions
data = np.load('test_predictions.npz')

# Access arrays
true_values = data['true_values']
predictions = data['predictions']

# Calculate metrics
mae = np.mean(np.abs(true_values - predictions))
mse = np.mean((true_values - predictions) ** 2)
```

**Best Practices:**
- Efficient for numerical data
- Compressed by default
- Fast random access

### 4. CSV Files (`.csv`)

**Format:** Comma-separated values

**Loading:**
```python
import pandas as pd

# Load predictions
df = pd.read_csv('test_predictions.csv')

# Analyze
print(df.describe())
print(f"Mean absolute error: {df['absolute_error'].mean()}")

# Filter high-error samples
high_error = df[df['absolute_error'] > 0.5]
```

**Best Practices:**
- Universal format (Excel, R, Python, etc.)
- Human-readable
- Easy to share and visualize

### 5. TensorBoard Event Files

**Format:** Protocol Buffers (binary)

**Loading:**
```python
from tensorboard.backend.event_processing import event_accumulator

# Load events
ea = event_accumulator.EventAccumulator('events.out.tfevents...')
ea.Reload()

# Get scalars
train_loss = ea.Scalars('train/loss')
val_loss = ea.Scalars('val/loss')

# Extract values
train_loss_values = [x.value for x in train_loss]
```

**Best Practices:**
- Use TensorBoard UI for visualization
- Extract data programmatically when needed
- Efficient storage of time-series data

### 6. Log Files (`.log`)

**Format:** Plain text with timestamps

**Parsing:**
```python
import re
from datetime import datetime

# Parse log file
with open('training_2024-11-23.log', 'r') as f:
    for line in f:
        # Extract timestamp and message
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) (\w+) (.+)', line)
        if match:
            timestamp_str, level, message = match.groups()
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            print(f"[{level}] {message}")
```

**Best Practices:**
- Use structured logging
- Include timestamps
- Rotate logs for long-running processes

---

## Console Output Examples

### Successful Training Run

```
================================================================================
VQM24 Models Module - Training Session
================================================================================
Session ID: GCN_graph_regression_2024-11-23_10-15-23
Started: 2024-11-23 10:15:23

Model Information:
  Name: GCN
  Task: graph_regression
  Parameters: 45,632 (all trainable)
  Device: cuda:0 (NVIDIA GeForce RTX 3090)

Dataset Information:
  Training: 800 samples (10 batches, batch_size=80)
  Validation: 100 samples (2 batches, batch_size=50)
  Test: 100 samples (2 batches, batch_size=50)

Training Configuration:
  Loss Function: MSELoss()
  Optimizer: Adam(lr=0.001, weight_decay=0.0001)
  Scheduler: ReduceLROnPlateau(patience=10, factor=0.5)
  Max Epochs: 100

Callbacks:
  ✓ EarlyStopping (patience=20, min_delta=0.0001)
  ✓ ModelCheckpoint (save_top_k=3, monitor=val_loss)
  ✓ TensorBoardLogger (log_dir=./logs/...)
  ✓ LearningRateMonitor (log_every_n_steps=1)
  ✓ ProgressBar (refresh_rate=1)

================================================================================
Training Progress
================================================================================

Epoch   1/100 | Train Loss: 2.3456 | Val Loss: 2.1234 | LR: 0.001000 | Time: 12.5s | ✓ Best
Epoch   2/100 | Train Loss: 1.8765 | Val Loss: 1.7654 | LR: 0.001000 | Time: 11.8s | ✓ Best
Epoch   3/100 | Train Loss: 1.5432 | Val Loss: 1.4567 | LR: 0.001000 | Time: 11.6s | ✓ Best
Epoch   4/100 | Train Loss: 1.3210 | Val Loss: 1.2890 | LR: 0.001000 | Time: 11.4s | ✓ Best
Epoch   5/100 | Train Loss: 1.1567 | Val Loss: 1.1345 | LR: 0.001000 | Time: 11.3s | ✓ Best
...
Epoch  45/100 | Train Loss: 0.2456 | Val Loss: 0.2345 | LR: 0.000678 | Time: 10.5s | ✓ Best
...
Epoch  52/100 | Train Loss: 0.2278 | Val Loss: 0.2234 | LR: 0.000621 | Time: 10.3s | ✓ Best
...
Epoch  67/100 | Train Loss: 0.2145 | Val Loss: 0.2187 | LR: 0.000543 | Time: 10.2s | ✓ Best
Epoch  68/100 | Train Loss: 0.2123 | Val Loss: 0.2201 | LR: 0.000532 | Time: 10.1s
Epoch  69/100 | Train Loss: 0.2109 | Val Loss: 0.2214 | LR: 0.000521 | Time: 10.1s
...
Epoch  87/100 | Train Loss: 0.2089 | Val Loss: 0.2245 | LR: 0.000421 | Time:  9.8s

⚠ EarlyStopping triggered: No improvement for 20 epochs
Training stopped at epoch 87/100

================================================================================
Training Complete
================================================================================
Total Time: 20m 35.6s (1235.6 seconds)
Best Epoch: 67
Best Val Loss: 0.2187

Final Metrics:
  Training:   Loss: 0.2089 | MAE: 0.304 | R²: 0.897
  Validation: Loss: 0.2245 | MAE: 0.322 | R²: 0.871

Saved Checkpoints:
  ✓ best_model_epoch=67_val_loss=0.219.pth
  ✓ best_model_epoch=52_val_loss=0.223.pth
  ✓ best_model_epoch=45_val_loss=0.234.pth
  ✓ last_model.pth

Checkpoint Directory: checkpoints/GCN_graph_regression_2024-11-23_10-15-23/
TensorBoard Logs: logs/GCN_graph_regression_2024-11-23_10-15-23/

================================================================================
Evaluating on Test Set
================================================================================
Loading: best_model_epoch=67_val_loss=0.219.pth
Device: cuda:0

Testing: 100%|████████████████████████████| 2/2 [00:02<00:00, 1.15s/batch]

Test Results:
  MSE:  0.2234
  MAE:  0.3123
  R²:   0.8764
  RMSE: 0.4727

Evaluation Time: 2.3 seconds (43.5 samples/sec)

Predictions saved to: predictions/GCN_graph_regression_2024-11-23_10-35-58/
  ✓ test_predictions.csv
  ✓ test_predictions.npz
  ✓ prediction_metadata.json

================================================================================
Session Complete
================================================================================
Session ID: GCN_graph_regression_2024-11-23_10-15-23
Started: 2024-11-23 10:15:23
Ended: 2024-11-23 10:35:58
Duration: 20m 35s

Thank you for using VQM24 Models Module!
================================================================================
```

### Error Handling Example

```
================================================================================
ERROR: Training Failed
================================================================================
Session ID: GCN_graph_regression_2024-11-23_10-15-23
Error Type: ModelInstantiationError
Timestamp: 2024-11-23 10:15:25.789

Error Message:
  Failed to instantiate model 'GCN': hidden_channels must be at least 1, got 0

Stack Trace:
  File "vqm24_pipeline/models/factory/model_factory.py", line 469, in create_model
    model = model_class(**processed_params)
  File "torch_geometric/nn/models/basic_gnn.py", line 87, in __init__
    raise ValueError("hidden_channels must be at least 1")

Configuration:
  Model: GCN
  Task: graph_regression
  Hyperparameters: {'hidden_channels': 0, 'num_layers': 3}

Suggestion:
  Please check your hyperparameters in config.yaml or pass valid values to
  create_model(). The 'hidden_channels' parameter must be >= 1.

Documentation:
  See: docs/MODELS_MODULE_LIFECYCLE_GUIDE.md
  API Reference: help(vqm24_pipeline.models.create_model)

================================================================================
```

---

## Python API Return Objects

### 1. Model Creation

```python
from vqm24_pipeline.models import create_model

model = create_model(
    name="GCN",
    hyperparameters={"hidden_channels": 64, "num_layers": 3},
    task_type="graph_regression",
    sample_data=sample_data
)

# Returns: torch.nn.Module
type(model)  # <class 'torch_geometric.nn.models.GCN'>

# Attributes
model.in_channels      # 16 (inferred from sample_data)
model.hidden_channels  # 64 (from hyperparameters)
model.out_channels     # 1 (inferred from task_type)
model.num_layers       # 3 (from hyperparameters)
```

### 2. Model Registry Queries

```python
from vqm24_pipeline.models import list_models, get_model_info

# List models
models = list_models(task_type="graph_regression")
# Returns: list of strings
# ['GCN', 'GAT', 'GraphSAGE', 'GIN', ...]

# Get model info
info = get_model_info("GCN")
# Returns: dict (see Phase 2 for full structure)
```

### 3. Data Splitting

```python
from vqm24_pipeline.models import DataSplitter

train, val, test = DataSplitter.random_split(dataset)

# Returns: tuple of 3 Subset objects
type(train)  # <class 'torch.utils.data.Subset'>
len(train)   # 800
len(val)     # 100
len(test)    # 100
```

### 4. Training Results

```python
from vqm24_pipeline.models import Trainer

trainer = Trainer(...)
results = trainer.fit(epochs=100)

# Returns: dict (see Phase 9 for complete structure)
results['best_epoch']           # 67
results['best_val_loss']        # 0.2187
results['training_time']        # 1235.6
results['checkpoint_path']      # Path to best checkpoint
results['test_metrics']         # Test evaluation results
results['history']              # Per-epoch metrics
```

### 5. Model Evaluation

```python
test_results = trainer.test(test_loader)

# Returns: dict
{
    'mse': 0.2234,
    'mae': 0.3123,
    'r2': 0.8764,
    'rmse': 0.4727,
    'predictions_path': 'predictions/.../test_predictions.csv'
}
```

### 6. Module Information

```python
from vqm24_pipeline.models import get_module_info

info = get_module_info()

# Returns: dict
{
    'version': '1.1.0',
    'total_models': 120,
    'categories': 12,
    'category_breakdown': {...},
    'available_losses': 12,
    'available_optimizers': 8,
    'available_schedulers': 6,
    'builders_available': True,
    'phase_7_features': {...}
}
```

---

## Integration Examples

### Example 1: Complete Training Pipeline

```python
"""
Complete training pipeline demonstrating all lifecycle phases.
"""
import torch
from torch_geometric.data import Data
from vqm24_pipeline.models import (
    create_model,
    DataSplitter,
    Trainer,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoardLogger
)

# 1. Create mock dataset (replace with real data)
dataset = [
    Data(x=torch.randn(10, 16), edge_index=torch.randint(0, 10, (2, 20)), y=torch.randn(1))
    for _ in range(1000)
]

# 2. Split data
train_data, val_data, test_data = DataSplitter.random_split(
    dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)

# 3. Create model
sample_data = dataset[0]
model = create_model(
    name="GCN",
    hyperparameters={"hidden_channels": 64, "num_layers": 3, "dropout": 0.5},
    task_type="graph_regression",
    sample_data=sample_data,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# 4. Create data loaders
from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_data, batch_size=80, shuffle=True)
val_loader = DataLoader(val_data, batch_size=50, shuffle=False)
test_loader = DataLoader(test_data, batch_size=50, shuffle=False)

# 5. Setup training
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    callbacks=[
        EarlyStopping(patience=20, min_delta=0.0001),
        ModelCheckpoint(save_top_k=3, save_last=True),
        TensorBoardLogger()
    ]
)

# 6. Train
results = trainer.fit(epochs=100)

# 7. Evaluate
test_results = trainer.test(test_loader)

# 8. Access outputs
print(f"Training time: {results['training_time']:.1f}s")
print(f"Best epoch: {results['best_epoch']}")
print(f"Best val loss: {results['best_val_loss']:.4f}")
print(f"Test MAE: {test_results['mae']:.4f}")
print(f"Checkpoint: {results['checkpoint_path']}")
print(f"Predictions: {test_results['predictions_path']}")
```

### Example 2: Load and Resume Training

```python
"""
Load checkpoint and resume training.
"""
import torch
from vqm24_pipeline.models import create_model, Trainer

# 1. Create model architecture
model = create_model(
    name="GCN",
    hyperparameters={"hidden_channels": 64, "num_layers": 3},
    task_type="graph_regression",
    sample_data=sample_data
)

# 2. Load checkpoint
checkpoint_path = "checkpoints/.../best_model_epoch=67_val_loss=0.219.pth"
checkpoint = torch.load(checkpoint_path)

# 3. Restore model state
model.load_state_dict(checkpoint['model_state_dict'])

# 4. Restore optimizer
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 5. Resume training from checkpoint epoch
start_epoch = checkpoint['epoch'] + 1
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=torch.nn.MSELoss(),
    optimizer=optimizer,
    start_epoch=start_epoch
)

results = trainer.fit(epochs=100)
```

### Example 3: Evaluation Only

```python
"""
Load trained model and evaluate on new data.
"""
import torch
import numpy as np
from vqm24_pipeline.models import create_model

# 1. Load model
checkpoint_path = "checkpoints/.../best_model_epoch=67_val_loss=0.219.pth"
checkpoint = torch.load(checkpoint_path)

# 2. Recreate model
model = create_model(
    name=checkpoint['model_name'],
    hyperparameters=checkpoint['hyperparameters'],
    task_type=checkpoint['task_type'],
    sample_data=sample_data
)

# 3. Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 4. Make predictions
with torch.no_grad():
    predictions = []
    true_values = []
    
    for batch in test_loader:
        batch = batch.to(model.device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        predictions.append(pred.cpu().numpy())
        true_values.append(batch.y.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    true_values = np.concatenate(true_values)

# 5. Calculate metrics
mae = np.mean(np.abs(true_values - predictions))
mse = np.mean((true_values - predictions) ** 2)
r2 = 1 - (np.sum((true_values - predictions) ** 2) / 
          np.sum((true_values - np.mean(true_values)) ** 2))

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

### Example 4: Custom Architecture (Phase 7)

```python
"""
Build custom architecture using ArchitectureBuilder.
"""
from vqm24_pipeline.models import ArchitectureBuilder

# 1. Create builder
builder = ArchitectureBuilder(
    task_type='graph_regression',
    in_channels=16,
    out_channels=1
)

# 2. Add layers
builder.add_layer('GCNConv', out_channels=64)
builder.add_layer('ReLU')
builder.add_layer('Dropout', p=0.5)
builder.add_layer('GATConv', out_channels=32, heads=4)
builder.add_layer('ReLU')
builder.add_layer('global_mean_pool')
builder.add_layer('Linear', out_features=16)
builder.add_layer('ReLU')
builder.add_layer('Linear', out_features=1)

# 3. Build model
model = builder.build()

# 4. Train as usual
trainer = Trainer(model=model, ...)
results = trainer.fit(epochs=100)
```

### Example 5: Ensemble Models (Phase 7)

```python
"""
Create ensemble of multiple models.
"""
from vqm24_pipeline.models import ModelComposer, create_model

# 1. Create individual models
model1 = create_model("GCN", {"hidden_channels": 64}, "graph_regression", sample_data)
model2 = create_model("GAT", {"hidden_channels": 64, "heads": 4}, "graph_regression", sample_data)
model3 = create_model("GraphSAGE", {"hidden_channels": 64}, "graph_regression", sample_data)

# 2. Create composer
composer = ModelComposer(task_type='graph_regression')

# 3. Add models with weights
composer.add_model(model1, weight=0.4)
composer.add_model(model2, weight=0.4)
composer.add_model(model3, weight=0.2)

# 4. Set composition strategy
composer.set_strategy('parallel')  # Parallel inference, weighted fusion

# 5. Build ensemble
ensemble = composer.build()

# 6. Train ensemble
trainer = Trainer(model=ensemble, ...)
results = trainer.fit(epochs=100)
```

---

## Troubleshooting

### Issue 1: Models Module Not Enabled

**Symptom:**
```
ERROR: Models module is disabled in configuration
```

**Solution:**
```yaml
# In config.yaml (line 830)
models:
  enabled: true  # Must be true
```

**Verification:**
```python
from vqm24_pipeline.models.utils.config_bridge import is_models_enabled

assert is_models_enabled(), "Models module not enabled!"
```

---

### Issue 2: Checkpoint Not Found

**Symptom:**
```
FileNotFoundError: Checkpoint file not found: checkpoints/.../best_model.pth
```

**Solution:**
```python
import os
from pathlib import Path

# Check if checkpoint exists
checkpoint_path = Path("checkpoints/GCN_.../best_model_epoch=67.pth")
if not checkpoint_path.exists():
    print(f"Checkpoint not found: {checkpoint_path}")
    print(f"Available checkpoints:")
    for f in checkpoint_path.parent.glob("*.pth"):
        print(f"  - {f.name}")
```

---

### Issue 3: Out of Memory (GPU)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

**A. Reduce batch size:**
```yaml
# In config.yaml
training:
  batch_size: 32  # Reduce from 80
```

**B. Enable gradient accumulation:**
```yaml
acceleration:
  memory:
    gradient_accumulation_steps: 4  # Effective batch = 32 * 4 = 128
```

**C. Enable mixed precision:**
```yaml
acceleration:
  memory:
    mixed_precision: "fp16"  # or "bf16"
```

**D. Enable gradient checkpointing:**
```yaml
acceleration:
  memory:
    gradient_checkpointing: true
```

---

### Issue 4: Training Not Converging

**Symptom:**
```
Validation loss not improving after many epochs
```

**Solutions:**

**A. Adjust learning rate:**
```yaml
training:
  optimizer:
    params:
      lr: 0.0001  # Reduce from 0.001
```

**B. Increase model capacity:**
```yaml
hyperparameters:
  hidden_channels: 128  # Increase from 64
  num_layers: 5         # Increase from 3
```

**C. Reduce regularization:**
```yaml
hyperparameters:
  dropout: 0.3  # Reduce from 0.5
```

**D. Check data quality:**
```python
# Inspect data distributions
import matplotlib.pyplot as plt

y_values = [data.y.item() for data in dataset]
plt.hist(y_values, bins=50)
plt.title("Target Distribution")
plt.show()
```

---

### Issue 5: Predictions File Empty

**Symptom:**
```
Predictions CSV is empty or missing
```

**Solution:**
```yaml
# In config.yaml (line 1095)
evaluation:
  save_predictions: true  # Must be true
  predictions_dir: null   # Auto-generate path
```

**Verification:**
```python
import pandas as pd

predictions_path = "predictions/.../test_predictions.csv"
df = pd.read_csv(predictions_path)
print(f"Predictions loaded: {len(df)} samples")
```

---

### Issue 6: TensorBoard Not Showing Data

**Symptom:**
```
TensorBoard dashboard is empty
```

**Solutions:**

**A. Verify TensorBoard is enabled:**
```yaml
callbacks:
  tensorboard:
    enabled: true
```

**B. Start TensorBoard with correct path:**
```bash
tensorboard --logdir ~/Chem_Data/VQM24_PyG_Dataset/logs
```

**C. Check if events file exists:**
```python
from pathlib import Path

logs_dir = Path("~/Chem_Data/VQM24_PyG_Dataset/logs").expanduser()
events_files = list(logs_dir.rglob("events.out.tfevents.*"))
print(f"Found {len(events_files)} TensorBoard event files")
```

---

### Issue 7: Model Architecture Mismatch

**Symptom:**
```
RuntimeError: Error loading state_dict: size mismatch for conv1.lin.weight
```

**Solution:**
```python
# When loading checkpoint, ensure hyperparameters match

# Correct approach
checkpoint = torch.load(checkpoint_path)
model = create_model(
    name=checkpoint['model_name'],
    hyperparameters=checkpoint['hyperparameters'],  # Use saved hyperparameters
    task_type=checkpoint['task_type'],
    sample_data=sample_data
)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Appendix

### A. Configuration Reference

**Complete models section structure:**
```yaml
models:
  enabled: true                           # Line 830
  
  selection:                              # Lines 836-851
    mode: "single"
    task_type: "graph_regression"
    model_name: "GCN"
  
  custom_architecture:                    # Lines 859-924
    enabled: false
    # ... (Phase 7 custom architectures)
  
  ensemble:                               # Lines 930-965
    enabled: false
    # ... (Phase 7 ensembles)
  
  hyperparameters:                        # Lines 967-999
    hidden_channels: 64
    num_layers: 3
    dropout: 0.5
  
  training:                               # Lines 1000-1078
    max_epochs: 100
    batch_size: 80
    # ... (training configuration)
  
  evaluation:                             # Lines 1079-1096
    metrics: ["mse", "mae", "r2"]
    test_after_training: true
    save_predictions: true
  
  acceleration:                           # Lines 1102-1162
    enabled: false
    # ... (hardware acceleration)
  
  deployment:                             # Lines 1168-1239
    enabled: false
    # ... (deployment settings)
  
  plugins:                                # Lines 1245-1251
    enabled: true
    # ... (plugin configuration)
```

### B. Supported Task Types

| Task Type | Description | Output Shape |
|-----------|-------------|--------------|
| `node_regression` | Predict continuous values per node | `(num_nodes, out_features)` |
| `node_classification` | Classify nodes into categories | `(num_nodes, num_classes)` |
| `graph_regression` | Predict continuous value per graph | `(batch_size, out_features)` |
| `graph_classification` | Classify graphs into categories | `(batch_size, num_classes)` |
| `link_prediction` | Predict edges between nodes | `(num_edges, 1)` |
| `edge_regression` | Predict continuous values per edge | `(num_edges, out_features)` |

### C. Model Categories

| Category | Count | Examples |
|----------|-------|----------|
| BASIC_GNN | 8 | GCN, GraphSAGE, GIN |
| CONVOLUTIONAL | 15 | GATConv, SAGEConv, GINConv |
| ATTENTION | 12 | GAT, Transformer, SuperGAT |
| MESSAGE_PASSING | 10 | MPNN, NNConv, CGConv |
| POOLING | 8 | TopKPooling, SAGPooling, EdgePooling |
| GRAPH_LEVEL | 12 | DeepGCN, PNA, DimeNet |
| SPECTRAL | 6 | ChebConv, ARMAConv |
| HETEROGENEOUS | 8 | HeteroConv, HAN, RGCN |
| TEMPORAL | 5 | TGCN, EvolveGCN, GCRN |
| MOLECULAR | 10 | SchNet, DimeNet++, SphereNet |
| POINT_CLOUD | 6 | PointNet++, DynamicEdgeConv |
| HYPERGRAPH | 4 | HypergraphConv, HGNN |

### D. Callback Reference

| Callback | Purpose | Configuration |
|----------|---------|---------------|
| EarlyStopping | Stop training when no improvement | `patience`, `min_delta`, `monitor` |
| ModelCheckpoint | Save best models | `save_top_k`, `monitor`, `mode` |
| TensorBoardLogger | Log metrics to TensorBoard | `log_dir`, `log_every_n_steps` |
| LearningRateMonitor | Track learning rate | `logging_interval` |
| ProgressBar | Show training progress | `refresh_rate` |
| GradientMonitor | Monitor gradient statistics | `log_every_n_steps` |

### E. Loss Functions

**Regression:**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Huber Loss
- Smooth L1 Loss
- RMSE (Root Mean Squared Error)

**Classification:**
- Cross Entropy Loss
- Binary Cross Entropy (BCE)
- BCE with Logits
- Focal Loss
- Label Smoothing Cross Entropy

**Custom:**
- Weighted MSE Loss
- Uncertainty-weighted Loss (for DMC data)

### F. Optimizers

- Adam
- AdamW
- SGD (Stochastic Gradient Descent)
- RMSprop
- Adagrad
- Adadelta

### G. Learning Rate Schedulers

- ReduceLROnPlateau
- CosineAnnealing
- StepLR
- ExponentialLR
- CyclicLR
- OneCycleLR

### H. File Size Estimates

| File Type | Typical Size | Notes |
|-----------|-------------|-------|
| `.pth` checkpoint | 0.1-10 MB | Depends on model size |
| `.json` metadata | 1-50 KB | Text-based, small |
| `.npz` predictions | 10-500 KB | Depends on test set size |
| `.csv` predictions | 50-2000 KB | Larger than .npz |
| TensorBoard events | 1-100 MB | Grows with training length |
| `.log` files | 100 KB-10 MB | Depends on verbosity |

### I. Performance Benchmarks

**Typical training times (on NVIDIA RTX 3090):**

| Model | Dataset Size | Batch Size | Epoch Time | Total Time (100 epochs) |
|-------|--------------|------------|------------|------------------------|
| GCN | 1,000 | 80 | 10-12s | 17-20 min |
| GAT | 1,000 | 80 | 12-15s | 20-25 min |
| GraphSAGE | 1,000 | 80 | 11-13s | 18-22 min |
| Custom (5 layers) | 1,000 | 80 | 15-18s | 25-30 min |
| Ensemble (3 models) | 1,000 | 80 | 30-35s | 50-60 min |

**Note:** Times vary based on:
- Model complexity (layers, hidden dimensions)
- Hardware (GPU vs CPU, GPU model)
- Dataset characteristics (graph size, features)
- Enabled callbacks and logging

### J. Storage Requirements

**For typical training session:**

```
Total Storage: ~50-200 MB

Breakdown:
  Checkpoints:    20-100 MB (3-5 .pth files)
  TensorBoard:    10-50 MB (event logs)
  Predictions:    5-20 MB (.csv + .npz)
  Logs:          1-10 MB (text logs)
  Metadata:      <1 MB (.json files)
  Visualizations: 5-20 MB (optional .png files)
```

### K. Command-Line Quick Reference

```bash
# Start TensorBoard
tensorboard --logdir ~/Chem_Data/VQM24_PyG_Dataset/logs

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check checkpoint sizes
du -sh ~/Chem_Data/VQM24_PyG_Dataset/checkpoints/*

# View recent logs
tail -f ~/Chem_Data/VQM24_PyG_Dataset/logs/training_$(date +%Y-%m-%d).log

# Count total parameters in saved checkpoint
python -c "import torch; c=torch.load('checkpoint.pth'); print(f\"{sum(p.numel() for p in c['model_state_dict'].values()):,}\")"

# Extract predictions to CSV
python -c "import numpy as np, pandas as pd; d=np.load('test_predictions.npz'); pd.DataFrame(d).to_csv('predictions.csv')"
```

### L. Additional Resources

**Documentation:**
- Main documentation: `docs/VQM24_Pipeline_Project_Structure.md`
- API reference: `help(vqm24_pipeline.models)`
- Configuration guide: `config.yaml` (annotated)

**Examples:**
- Example scripts: `examples/models/`
- Jupyter notebooks: `examples/notebooks/`
- Training templates: `examples/training_configs/`

**Tests:**
- Unit tests: `tests/test_model_factory_unit.py`
- Integration tests: `tests/integration/test_training_pipeline.py`
- Performance tests: `tests/performance/test_training_speed.py`

**Support:**
- Issue tracker: GitHub repository
- Discussions: Project discussions forum
- Email: VQM24 team contact

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2024-11-23 | VQM24 Team | Initial release |

---

## License

This document is part of the VQM24 Pipeline project.  
**License:** MIT  
**Copyright:** © 2024 VQM24 Team

---

**End of Document**
