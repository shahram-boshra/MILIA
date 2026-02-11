# MILIA Pipeline HPO Module Implementation Blueprint    <--- DONE!

**Document Version**: 1.0.0
**Created**: November 27, 2025
**Target Module**: `milia_pipeline/models/hpo/`
**Primary Backend**: Optuna (with Ray Tune as optional scale-out)

---

**NOTE**

## Some Considerations on the Blueprint Implementation

# Consideration 1
**Required files for new context window to fully implement the blueprint:**
1. `MILIA_HPO_Implementation_Blueprint.md` 
2. `trainer.py`
3. `callbacks.py`
4. `model_factory.py`
5. `config_bridge.py`
6. `config.yaml`
7. `exceptions.py`
8. `data_splitting.py`
9. `loss_functions.py`
10. `schedulers.py`
11. `main.py`
12. `milia_pipeline_models__init__.py` 

# Consideration 2
**Completely Shown codes (ready to copy-paste):**
- `exceptions.py`                                           <--- DONE!  <--- TESTED
- `hpo_config.py`                                           <--- DONE! 
- `backends/base.py`                                        <--- DONE! 
- `backends/optuna_backend.py`                              <--- DONE! 
- `callbacks/optuna_callback.py`                            <--- DONE! 

**Not Completely Shown codes(need implementation):**
- `hpo_manager.py` (90% shown, helper functions incomplete) <--- DONE! 
- `search_spaces/search_space_builder.py`                   <--- DONE!
- `search_spaces/param_types.py`                            <--- DONE!
- `analysis/study_analyzer.py`                              <--- DONE!
- All `__init__.py` files (6 total)                         <--- DONE!
- `backends/ray_tune_backend.py` (stub only)                <--- DONE!
- `callbacks/ray_tune_callback.py` (stub only)              <--- DONE!

# Consideration 3
**Incomplete Sections:**

**8.2.2: Integration tests** skeleton only, needs full test cases
**9.3: Future enhancements** conceptual, not implementation-ready

All other sections are complete.


## Part 1: Rationale for Choosing Optuna as Primary Backend

### Evidence-Based Decision Summary

After systematic line-by-line analysis of 14 project files totaling ~20,000 lines of code, **Optuna** emerges as the optimal primary choice for the following evidence-backed reasons:

#### 1. Callback System Alignment (callbacks.py:35-96, 103-227)

**Evidence**: Your `Callback` ABC defines 4 hooks: `set_trainer()`, `on_train_begin()`, `on_epoch_end()`, `on_train_end()`. The `EarlyStopping` callback (lines 103-227) implements `should_stop()` method with patience-based stopping.

**Optuna Alignment**: Optuna's `trial.report(metric, epoch)` + `trial.should_prune()` maps **1:1** to your existing callback pattern. A single new callback class (`OptunaPruningCallback`) integrating with your ABC is sufficient.

**Ray Tune Impact**: Would require restructuring the training loop to use `tune.report()`, breaking the existing callback abstraction.

#### 2. Configuration System Compatibility (config_bridge.py:244-264, 967-993)

**Evidence**: Your `OptimizerConfig` dataclass (lines 244-264) uses `name: str` + `params: Dict[str, Any]` structure. Your 30+ frozen dataclasses follow consistent patterns with validation via `ConfigurationError`.

**Optuna Alignment**: `trial.suggest_float('lr', 1e-4, 1e-2)` directly populates the `params` dict. No transformation layer needed.

**Ray Tune Impact**: Requires conversion to `tune.loguniform(1e-4, 1e-2)` format with additional mapping code.

#### 3. Thread-Safe Registry Compatibility (model_factory.py:851-865)

**Evidence**: `ModelRegistry` uses singleton pattern with `RLock` for thread safety. `get_factory()` function (lines 851-865) returns cached singleton instance.

**Optuna Alignment**: Runs trials in-process by default. Your existing `RLock` synchronization works without modification.

**Ray Tune Impact**: Spawns separate processes for each trial. Would require process-safe registry synchronization (e.g., Redis backend or shared memory).

#### 4. Trainer Return Structure (trainer.py:74-628)

**Evidence**: `Trainer.fit()` returns structured dict (line ~600):
```python
return {
    'train_metrics': dict,
    'test_metrics': dict, 
    'best_val_loss': float,
    'best_epoch': int
}
```

**Optuna Alignment**: Direct metric extraction: `return trial_result['best_val_loss']` as objective value.

**Ray Tune Impact**: Similar, but requires wrapping in `tune.report()` calls throughout the training loop.

#### 5. Research Experiments Style (research_experiments.yaml:20-467)

**Evidence**: Your experiments use define-by-run style:
```yaml
experiments:
  transform_ablation:
    num_runs: 5
    random_seed: 42
    metadata:
      primary_metric: "validation_mae"
```

**Optuna Alignment**: Optuna's define-by-run API matches this pattern exactly.

#### 6. Loss/Scheduler Registry Integration (loss_functions.py, schedulers.py)

**Evidence**: `LossRegistry.get_loss(name, params)` (lines 183-230) and `SchedulerRegistry.get_scheduler(name, optimizer, params)` (lines 108-168) both accept `params: Dict[str, Any]`.

**Optuna Alignment**: Search spaces can directly feed these registries:
```python
loss_params = {'alpha': trial.suggest_float('focal_alpha', 0.1, 0.9)}
loss_fn = LossRegistry.get_loss('focal', loss_params)
```

#### 7. Exception Hierarchy Alignment (exceptions.py:2644-2676)

**Evidence**: `HyperparameterError` class (lines 2644-2676) already exists with attributes: `model_name`, `parameter_name`, `parameter_value`, `expected_type`.

**Optuna Alignment**: HPO exceptions can inherit from existing `ModelError` hierarchy. No new base exception classes needed.

#### 8. Data Splitting Integration (data_splitting.py:38-527)

**Evidence**: `DataSplitter` class provides 5 strategies: `random_split`, `stratified_split`, `temporal_split`, `scaffold_split`, `k_fold_split`. All return `Tuple[Subset, Subset, Subset]` or `List[Tuple[Subset, Subset]]`.

**Optuna Alignment**: K-fold cross-validation for HPO can directly use `k_fold_split()`. No wrapper needed.

### Performance Considerations

| Factor | Optuna | Ray Tune |
|--------|--------|----------|
| Single-node overhead | ~5ms/trial | ~50-100ms/trial (Ray actor system) |
| Memory footprint | ~5MB | ~100MB+ |
| Startup time | <1s | 3-10s (Ray cluster init) |
| Search algorithms | Native TPE, CMA-ES | Uses OptunaSearch wrapper |

### When to Use Ray Tune (Future)

Ray Tune becomes advantageous when:
- Running 100+ concurrent trials across GPU cluster
- Existing Ray infrastructure in production
- Need for Population Based Training (PBT)
- 1000+ trial experiments requiring distributed coordination

---

## Confirmed: No Existing HPO/Tuning Directory

Per `MILIA_Pipeline_Project_Structure.md` analysis, the `models/` module contains:
- `models/registry/`
- `models/factory/`
- `models/training/`
- `models/builders/`
- `models/acceleration/`
- `models/deployment/`
- `models/utils/`
- `models/plugins/`

**No `hpo/` or `tuning/` directory exists.** This implementation creates a new submodule.


---

## Part 2: Module Architecture and File Structure

### 2.1 Directory Structure

```
milia_pipeline/models/hpo/
├── __init__.py                      # Public API exports                           <--- DONE! 
├── hpo_config.py                    # HPOConfig dataclass (frozen, validated)      <--- DONE! <--- TESTED
├── hpo_manager.py                   # HPOManager orchestrator class                <--- DONE! <--- TESTED 
├── backends/
│   ├── __init__.py                  # Backend exports                              <--- DONE!
│   ├── base.py                      # HPOBackendProtocol (abstract interface)      <--- DONE! <--- TESTED
│   ├── optuna_backend.py            # OptunaBackend (primary implementation)       <--- DONE! <--- TESTED
│   └── ray_tune_backend.py          # RayTuneBackend (complete, inactive)          <--- DONE! <--- DEFERRED
├── callbacks/
│   ├── __init__.py                  # Callback exports                             <--- DONE!
│   ├── optuna_callback.py           # OptunaPruningCallback                        <--- DONE! <--- TESTED 
│   └── ray_tune_callback.py         # RayTuneReportCallback (complete, inactive)   <--- DONE! <--- DEFERRED
├── search_spaces/
│   ├── __init__.py                  # Search space exports                         <--- DONE!
│   ├── search_space_builder.py      # SearchSpaceBuilder class                     <--- DONE! <--- TESTED
│   └── param_types.py               # Parameter type definitions                   <--- DONE! <--- TESTED
├── transfer/
│   ├── __init__.py                  # Transfer learning exports                    <--- DONE!  
│   ├── transfer_manager.py          # HPOTransferManager class                     <--- DONE! <--- TESTED
│   ├── meta_features.py             # MetaFeatureExtractor class                   <--- DONE! <--- TESTED
│   └── warm_start.py                # WarmStartStrategy class                      <--- DONE! <--- TESTED 
├── nas/
│   ├── __init__.py                  # NAS exports                                  <--- DONE!
│   ├── search_space.py              # GNNArchitectureSpace class                   <--- DONE! <--- TESTED 
│   └── nas_manager.py               # NASManager class                             <--- DONE! <--- TESTED 
└── analysis/
    ├── __init__.py                  # Analysis exports                             <--- DONE!
    └── study_analyzer.py            # StudyAnalyzer for results analysis           <--- DONE! <--- TESTED
```

**Total: 22 new files** (exceptions added to centralized `exceptions.py`)

### 2.2 Dependency on Existing Modules

| HPO Component | Depends On | File Reference | Integration Point |
|---------------|------------|----------------|-------------------|
| `HPOConfig` | `config_bridge.py` | Lines 167-649 | Inherits frozen dataclass pattern |
| `OptunaPruningCallback` | `callbacks.py` | Lines 35-96 | Extends `Callback` ABC |
| `HPOManager` | `trainer.py` | Lines 74-628 | Calls `Trainer.fit()` |
| `HPOManager` | `model_factory.py` | Lines 315-953 | Calls `create_model()` |
| `SearchSpaceBuilder` | `loss_functions.py` | Lines 183-230 | Maps to `LossRegistry.get_loss()` |
| `SearchSpaceBuilder` | `schedulers.py` | Lines 108-168 | Maps to `SchedulerRegistry.get_scheduler()` |
| `HPOExceptions` | `exceptions.py` | Lines 2810+ | Add HPO exceptions to centralized module |
| `HPOManager` | `data_splitting.py` | Lines 454-527 | Uses `k_fold_split()` for CV |
| `HPOManager` | `device_manager.py` | Lines 126-232 | Uses `DeviceManager` for device placement |

### 2.3 File Descriptions

#### Core Files

| File | Lines (Est.) | Purpose |
|------|--------------|---------|
| `__init__.py` | ~100 | Public API: `HPOManager`, `HPOConfig`, `is_hpo_enabled()`, `get_best_params()` |
| `hpo_config.py` | ~350 | Frozen dataclasses: `HPOConfig`, `SearchSpaceConfig`, `PrunerConfig`, `SamplerConfig`, `StudyConfig`, `MultiObjectiveStudyConfig` |
| `hpo_manager.py` | ~500 | Main orchestrator: `optimize()`, `resume_study()`, `get_best_trial()`, `get_pareto_front()`, `select_by_preference()` |

**Note**: HPO exceptions (`HPOError`, `HPOConfigurationError`, `TrialFailedError`, etc.) are added to centralized `milia_pipeline/exceptions.py`

#### Transfer Learning Files

| File | Lines (Est.) | Purpose |
|------|--------------|---------|
| `transfer/__init__.py` | ~20 | Exports: `HPOTransferManager`, `MetaFeatureExtractor`, `WarmStartStrategy` |
| `transfer/transfer_manager.py` | ~200 | Transfer knowledge between studies, warm-start optimization |
| `transfer/meta_features.py` | ~150 | Extract dataset meta-features for similarity computation |
| `transfer/warm_start.py` | ~100 | Strategies for warm-starting from source studies |

#### NAS (Neural Architecture Search) Files

| File | Lines (Est.) | Purpose |
|------|--------------|---------|
| `nas/__init__.py` | ~15 | Exports: `GNNArchitectureSpace`, `NASManager` |
| `nas/search_space.py` | ~200 | Define GNN architecture search space (layers, pooling, aggregation) |
| `nas/nas_manager.py` | ~250 | Orchestrate architecture search, build models from configs |

#### Backend Files

| File | Lines (Est.) | Purpose |
|------|--------------|---------|
| `backends/__init__.py` | ~20 | Exports: `HPOBackendProtocol`, `OptunaBackend`, `get_backend()` |
| `backends/base.py` | ~100 | Protocol definition with 6 abstract methods |
| `backends/optuna_backend.py` | ~350 | Optuna implementation: study creation, trial execution, pruning |
| `backends/ray_tune_backend.py` | ~350 | Complete Ray Tune implementation (inactive until activated) |

#### Callback Files

| File | Lines (Est.) | Purpose |
|------|--------------|---------|
| `callbacks/__init__.py` | ~15 | Exports: `OptunaPruningCallback`, `create_hpo_callback()` |
| `callbacks/optuna_callback.py` | ~120 | Implements `Callback` ABC with Optuna integration |
| `callbacks/ray_tune_callback.py` | ~80 | Complete Ray Tune callback (inactive, in ray_tune_backend.py) |

#### Search Space Files

| File | Lines (Est.) | Purpose |
|------|--------------|---------|
| `search_spaces/__init__.py` | ~15 | Exports: `SearchSpaceBuilder`, `ParamType` |
| `search_spaces/search_space_builder.py` | ~200 | Converts YAML config to backend-specific search spaces |
| `search_spaces/param_types.py` | ~80 | Enum and dataclasses for parameter types |

#### Analysis Files

| File | Lines (Est.) | Purpose |
|------|--------------|---------|
| `analysis/__init__.py` | ~10 | Exports: `StudyAnalyzer` |
| `analysis/study_analyzer.py` | ~150 | Results analysis: importance, visualization prep |

**Total Estimated Lines: ~2,020**

### 2.4 Public API Design

```python
# milia_pipeline/models/hpo/__init__.py

# Primary exports (user-facing)
from .hpo_manager import HPOManager
from .hpo_config import (
    HPOConfig,
    SearchSpaceConfig,
    PrunerConfig,
    SamplerConfig,
    StudyConfig,
)

# Exceptions from centralized module
from milia_pipeline.exceptions import (
    HPOError,
    HPOConfigurationError,
    TrialFailedError,
    StudyNotFoundError,
)

# Convenience functions
from .hpo_manager import (
    is_hpo_enabled,
    get_best_params,
    create_hpo_manager,
)

# Backend access (advanced users)
from .backends import (
    HPOBackendProtocol,
    OptunaBackend,
    get_backend,
)

# Callback access
from .callbacks import (
    OptunaPruningCallback,
    create_hpo_callback,
)

# Search space utilities
from .search_spaces import (
    SearchSpaceBuilder,
    ParamType,
)

# Analysis utilities
from .analysis import StudyAnalyzer

__all__ = [
    # Core
    'HPOManager',
    'HPOConfig',
    'SearchSpaceConfig',
    'PrunerConfig',
    'SamplerConfig',
    'StudyConfig',
    # Exceptions
    'HPOError',
    'HPOConfigurationError',
    'TrialFailedError',
    'StudyNotFoundError',
    # Convenience
    'is_hpo_enabled',
    'get_best_params',
    'create_hpo_manager',
    # Backends
    'HPOBackendProtocol',
    'OptunaBackend',
    'get_backend',
    # Callbacks
    'OptunaPruningCallback',
    'create_hpo_callback',
    # Search spaces
    'SearchSpaceBuilder',
    'ParamType',
    # Analysis
    'StudyAnalyzer',
]
```


---

## Part 3: Detailed Implementation Specifications

### 3.1 Configuration Dataclasses (`hpo_config.py`)

#### 3.1.1 Pattern Alignment with config_bridge.py

Following the pattern from `config_bridge.py` (lines 167-649), all dataclasses must be:
- **Frozen** (`frozen=True`)
- **Validated** in `__post_init__`
- **Type-hinted** with Optional defaults
- **Documented** with docstrings

#### 3.1.2 SearchSpaceConfig

```python
# Location: milia_pipeline/models/hpo/hpo_config.py
# Pattern: config_bridge.py lines 244-264 (OptimizerConfig)

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from enum import Enum

class ParamType(Enum):
    """Parameter types for search space definition."""
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOGUNIFORM = "loguniform"
    UNIFORM = "uniform"
    INT_UNIFORM = "int_uniform"
    DISCRETE_UNIFORM = "discrete_uniform"


@dataclass(frozen=True)
class SearchSpaceParamConfig:
    """
    Configuration for a single hyperparameter in search space.
    
    Pattern: Follows OptimizerConfig.params structure (config_bridge.py:244-264)
    
    Attributes:
        type: Parameter type (int, float, categorical, loguniform)
        low: Lower bound for numeric types
        high: Upper bound for numeric types
        step: Step size for int types (optional)
        choices: List of choices for categorical type
        log: Whether to use log scale (for float type)
    """
    type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[int] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    
    def __post_init__(self):
        """Validate configuration based on type."""
        # Import here to avoid circular imports
        # Pattern: exceptions.py lines 271-308 (_init_registry)
        from milia_pipeline.exceptions import ConfigurationError
        
        if self.type in (ParamType.INT, ParamType.FLOAT, ParamType.LOGUNIFORM, 
                         ParamType.UNIFORM, ParamType.INT_UNIFORM):
            if self.low is None or self.high is None:
                raise ConfigurationError(
                    f"Parameter type '{self.type.value}' requires 'low' and 'high'",
                    config_key="search_space",
                    details=f"low={self.low}, high={self.high}"
                )
            if self.low >= self.high:
                raise ConfigurationError(
                    f"'low' must be less than 'high'",
                    config_key="search_space",
                    actual_value=f"low={self.low}, high={self.high}"
                )
        
        if self.type == ParamType.CATEGORICAL:
            if not self.choices or len(self.choices) == 0:
                raise ConfigurationError(
                    "Categorical parameter requires non-empty 'choices' list",
                    config_key="search_space"
                )
```

#### 3.1.3 PrunerConfig

```python
# Location: milia_pipeline/models/hpo/hpo_config.py

class PrunerType(Enum):
    """Supported pruner types."""
    MEDIAN = "median"
    PERCENTILE = "percentile"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    THRESHOLD = "threshold"
    PATIENT = "patient"
    NONE = "none"


@dataclass(frozen=True)
class PrunerConfig:
    """
    Pruner configuration for early trial termination.
    
    Pattern: Follows CallbacksConfig structure (config_bridge.py)
    
    Attributes:
        type: Pruner type (median, hyperband, percentile, etc.)
        n_startup_trials: Trials before pruning begins
        n_warmup_steps: Epochs before pruning within a trial
        interval_steps: Check pruning every N steps
        percentile: For percentile pruner (default: 25.0)
        n_brackets: For Hyperband pruner (default: 4)
    """
    type: PrunerType = PrunerType.MEDIAN
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    interval_steps: int = 1
    percentile: float = 25.0
    n_brackets: int = 4
    
    def __post_init__(self):
        """Validate pruner configuration."""
        from milia_pipeline.exceptions import ConfigurationError
        
        if self.n_startup_trials < 0:
            raise ConfigurationError(
                "n_startup_trials must be non-negative",
                config_key="pruner.n_startup_trials",
                actual_value=self.n_startup_trials
            )
        
        if self.n_warmup_steps < 0:
            raise ConfigurationError(
                "n_warmup_steps must be non-negative",
                config_key="pruner.n_warmup_steps",
                actual_value=self.n_warmup_steps
            )
        
        if self.type == PrunerType.PERCENTILE and not (0 < self.percentile < 100):
            raise ConfigurationError(
                "percentile must be between 0 and 100",
                config_key="pruner.percentile",
                actual_value=self.percentile
            )
```

#### 3.1.4 SamplerConfig

```python
# Location: milia_pipeline/models/hpo/hpo_config.py

class SamplerType(Enum):
    """Supported sampler types."""
    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"
    GRID = "grid"
    NSGAII = "nsgaii"  # Multi-objective
    MOTPE = "motpe"    # Multi-objective TPE
    QMCSAMPLER = "qmc" # Quasi-Monte Carlo


@dataclass(frozen=True)
class SamplerConfig:
    """
    Sampler configuration for hyperparameter suggestion.
    
    Attributes:
        type: Sampler type (tpe, random, cmaes, grid)
        n_startup_trials: Random trials before Bayesian optimization
        seed: Random seed for reproducibility
        multivariate: Whether to use multivariate TPE
        constant_liar: For parallel optimization
    """
    type: SamplerType = SamplerType.TPE
    n_startup_trials: int = 10
    seed: Optional[int] = None
    multivariate: bool = True
    constant_liar: bool = False
    
    def __post_init__(self):
        """Validate sampler configuration."""
        from milia_pipeline.exceptions import ConfigurationError
        
        if self.n_startup_trials < 0:
            raise ConfigurationError(
                "n_startup_trials must be non-negative",
                config_key="sampler.n_startup_trials",
                actual_value=self.n_startup_trials
            )
```

#### 3.1.5 StudyConfig

```python
# Location: milia_pipeline/models/hpo/hpo_config.py

class OptimizationDirection(Enum):
    """Optimization direction."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class StudyConfig:
    """
    Optuna study configuration (single-objective).
    
    Attributes:
        direction: Optimization direction (minimize/maximize)
        metric: Metric name to optimize (must match Trainer output)
        study_name: Name for the study (for persistence)
        storage: Storage URL (None for in-memory, "sqlite:///file.db" for persistence)
        load_if_exists: Whether to resume existing study
    """
    direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    metric: str = "val_loss"
    study_name: str = "milia_hpo"
    storage: Optional[str] = None
    load_if_exists: bool = True
    
    def __post_init__(self):
        """Validate study configuration."""
        from milia_pipeline.exceptions import ConfigurationError
        
        if not self.metric:
            raise ConfigurationError(
                "metric cannot be empty",
                config_key="study.metric"
            )
        
        if not self.study_name:
            raise ConfigurationError(
                "study_name cannot be empty",
                config_key="study.study_name"
            )
    
    @property
    def is_multi_objective(self) -> bool:
        """Check if this is multi-objective optimization."""
        return False
```

#### 3.1.6 MultiObjectiveStudyConfig

```python
# Location: milia_pipeline/models/hpo/hpo_config.py

@dataclass(frozen=True)
class MultiObjectiveStudyConfig:
    """
    Configuration for multi-objective optimization.
    
    Supports Pareto optimization for competing objectives
    (e.g., accuracy vs speed, MAE vs model size).
    
    Attributes:
        directions: Optimization direction per objective
        metrics: Metric names to optimize
        study_name: Name for the study
        storage: Storage URL
        load_if_exists: Whether to resume existing study
        reference_point: Reference point for hypervolume calculation
    """
    directions: Tuple[str, ...] = ("minimize",)
    metrics: Tuple[str, ...] = ("val_loss",)
    study_name: str = "milia_hpo_multi"
    storage: Optional[str] = None
    load_if_exists: bool = True
    reference_point: Optional[Tuple[float, ...]] = None
    
    def __post_init__(self):
        """Validate multi-objective configuration."""
        from milia_pipeline.exceptions import ConfigurationError
        
        if len(self.directions) != len(self.metrics):
            raise ConfigurationError(
                "directions and metrics must have same length",
                config_key="study.directions",
                expected_value=f"length {len(self.metrics)}",
                actual_value=f"length {len(self.directions)}"
            )
        
        for d in self.directions:
            if d not in ("minimize", "maximize"):
                raise ConfigurationError(
                    f"Invalid direction: {d}",
                    config_key="study.directions",
                    expected_value="'minimize' or 'maximize'",
                    actual_value=d
                )
        
        if len(self.metrics) < 2:
            raise ConfigurationError(
                "Multi-objective requires at least 2 metrics",
                config_key="study.metrics",
                expected_value="2+ metrics",
                actual_value=f"{len(self.metrics)} metrics"
            )
        
        if self.reference_point and len(self.reference_point) != len(self.metrics):
            raise ConfigurationError(
                "reference_point must match number of metrics",
                config_key="study.reference_point"
            )
    
    @property
    def is_multi_objective(self) -> bool:
        """Check if this is multi-objective optimization."""
        return True
```

#### 3.1.7 Main HPOConfig

```python
# Location: milia_pipeline/models/hpo/hpo_config.py

@dataclass(frozen=True)
class HPOConfig:
    """
    Master HPO configuration.
    
    Pattern: Follows TrainingConfig structure (config_bridge.py)
    
    This is the MASTER SWITCH configuration that enables/disables HPO
    and configures all aspects of hyperparameter optimization.
    
    Attributes:
        enabled: MASTER SWITCH - enables HPO when True
        backend: HPO backend ("optuna" or "ray_tune")
        n_trials: Number of trials to run
        timeout: Maximum time in seconds (None for no limit)
        n_jobs: Number of parallel jobs (1 for sequential)
        search_space: Hyperparameter search space configuration
        pruner: Pruner configuration
        sampler: Sampler configuration
        study: Study configuration
        cv_folds: Number of cross-validation folds (0 for no CV)
        cv_metric_aggregation: How to aggregate CV metrics ("mean", "median", "min")
    """
    enabled: bool = False
    backend: str = "optuna"
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    search_space: Dict[str, Dict[str, SearchSpaceParamConfig]] = field(default_factory=dict)
    pruner: PrunerConfig = field(default_factory=PrunerConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    study: StudyConfig = field(default_factory=StudyConfig)
    cv_folds: int = 0
    cv_metric_aggregation: str = "mean"
    
    def __post_init__(self):
        """Validate HPO configuration."""
        from milia_pipeline.exceptions import ConfigurationError
        
        if self.backend not in ("optuna", "ray_tune"):
            raise ConfigurationError(
                f"Unknown HPO backend: '{self.backend}'",
                config_key="hpo.backend",
                expected_value="'optuna' or 'ray_tune'",
                actual_value=self.backend
            )
        
        if self.n_trials < 1:
            raise ConfigurationError(
                "n_trials must be at least 1",
                config_key="hpo.n_trials",
                actual_value=self.n_trials
            )
        
        if self.timeout is not None and self.timeout < 1:
            raise ConfigurationError(
                "timeout must be positive or None",
                config_key="hpo.timeout",
                actual_value=self.timeout
            )
        
        if self.n_jobs < 1:
            raise ConfigurationError(
                "n_jobs must be at least 1",
                config_key="hpo.n_jobs",
                actual_value=self.n_jobs
            )
        
        if self.cv_folds < 0:
            raise ConfigurationError(
                "cv_folds must be non-negative",
                config_key="hpo.cv_folds",
                actual_value=self.cv_folds
            )
        
        if self.cv_metric_aggregation not in ("mean", "median", "min", "max"):
            raise ConfigurationError(
                "Invalid cv_metric_aggregation",
                config_key="hpo.cv_metric_aggregation",
                expected_value="'mean', 'median', 'min', or 'max'",
                actual_value=self.cv_metric_aggregation
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HPOConfig':
        """
        Create HPOConfig from dictionary.
        
        Pattern: Follows ModelConfig.from_yaml() (config_bridge.py:967-993)
        """
        # Parse nested configs
        pruner_dict = config_dict.get('pruner', {})
        sampler_dict = config_dict.get('sampler', {})
        study_dict = config_dict.get('study', {})
        
        # Convert type strings to enums
        if 'type' in pruner_dict:
            pruner_dict['type'] = PrunerType(pruner_dict['type'])
        if 'type' in sampler_dict:
            sampler_dict['type'] = SamplerType(sampler_dict['type'])
        if 'direction' in study_dict:
            study_dict['direction'] = OptimizationDirection(study_dict['direction'])
        
        # Parse search space
        search_space = {}
        raw_search_space = config_dict.get('search_space', {})
        for category, params in raw_search_space.items():
            search_space[category] = {}
            for param_name, param_config in params.items():
                param_config['type'] = ParamType(param_config['type'])
                search_space[category][param_name] = SearchSpaceParamConfig(**param_config)
        
        return cls(
            enabled=config_dict.get('enabled', False),
            backend=config_dict.get('backend', 'optuna'),
            n_trials=config_dict.get('n_trials', 100),
            timeout=config_dict.get('timeout'),
            n_jobs=config_dict.get('n_jobs', 1),
            search_space=search_space,
            pruner=PrunerConfig(**pruner_dict) if pruner_dict else PrunerConfig(),
            sampler=SamplerConfig(**sampler_dict) if sampler_dict else SamplerConfig(),
            study=StudyConfig(**study_dict) if study_dict else StudyConfig(),
            cv_folds=config_dict.get('cv_folds', 0),
            cv_metric_aggregation=config_dict.get('cv_metric_aggregation', 'mean'),
        )
```


---

## Part 4: Exception Hierarchy (Centralized) and Backend Protocol

### 4.1 HPO Exception Classes (Add to `exceptions.py` after line 2810)

#### 4.1.1 Pattern Alignment with exceptions.py

Following the pattern from `exceptions.py` (lines 2544-2810), HPO exceptions must:
- Inherit from `ModelError` (not `BaseProjectError`) for consistency with models module
- Include contextual attributes for debugging
- Override `__str__` for formatted messages
- Support `**kwargs` for extensibility

**Location**: Add to `milia_pipeline/exceptions.py` after line 2810             <--- DONE!

```python
# =============================================================================
# HPO SYSTEM EXCEPTIONS (Phase 8)
# =============================================================================

class HPOError(ModelError):
    """
    Base exception for all HPO-related errors.
    
    Pattern: Follows ModelError (exceptions.py:2547-2564)
    
    Attributes:
        message: Description of the error
        study_name: Name of the HPO study (if applicable)
        trial_number: Trial number (if applicable)
        details: Additional technical details
    """
    
    def __init__(
        self,
        message: str,
        study_name: Optional[str] = None,
        trial_number: Optional[int] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, details=details, **kwargs)
        self.study_name = study_name
        self.trial_number = trial_number
    
    def __str__(self) -> str:
        msg = self.message
        if self.study_name:
            msg += f". Study: '{self.study_name}'"
        if self.trial_number is not None:
            msg += f", Trial: {self.trial_number}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class HPOConfigurationError(HPOError):
    """
    Exception raised for HPO configuration errors.
    
    Pattern: Follows ConfigurationError (exceptions.py:489-532)
    
    Attributes:
        message: Description of the configuration error
        config_key: The specific configuration key that caused the issue
        actual_value: The actual value that was found
        expected_value: The expected value or type
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        actual_value: Any = None,
        expected_value: Any = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.actual_value = actual_value
        self.expected_value = expected_value
    
    def __str__(self) -> str:
        parts = [self.message]
        
        if self.config_key:
            parts.append(f"Key: '{self.config_key}'")
        
        if self.expected_value is not None:
            parts.append(f"Expected: {self.expected_value}")
        
        if self.actual_value is not None:
            parts.append(f"Actual: {self.actual_value}")
        
        if self.details:
            parts.append(f"Details: {self.details}")
        
        return " | ".join(parts)


class TrialFailedError(HPOError):
    """
    Exception raised when a trial fails during execution.
    
    Pattern: Follows TrainingError (exceptions.py:2709-2741)
    
    Attributes:
        message: Description of the failure
        trial_number: The trial that failed
        trial_params: Hyperparameters used in the failed trial
        original_error: The original exception that caused the failure
        epoch: Epoch at which failure occurred (if applicable)
    """
    
    def __init__(
        self,
        message: str,
        trial_number: Optional[int] = None,
        trial_params: Optional[Dict[str, Any]] = None,
        original_error: Optional[str] = None,
        epoch: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, trial_number=trial_number, **kwargs)
        self.trial_params = trial_params or {}
        self.original_error = original_error
        self.epoch = epoch
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.epoch is not None:
            msg += f", Epoch: {self.epoch}"
        if self.original_error:
            msg += f". Original error: {self.original_error}"
        return msg


class StudyNotFoundError(HPOError):
    """
    Exception raised when a requested study is not found.
    
    Pattern: Follows ModelNotFoundError (exceptions.py:2566-2587)
    
    Attributes:
        message: Description of the error
        study_name: Name of the study that was not found
        available_studies: List of available study names
        storage_url: Storage URL that was searched
    """
    
    def __init__(
        self,
        message: str,
        study_name: str,
        available_studies: Optional[List[str]] = None,
        storage_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, study_name=study_name, **kwargs)
        self.available_studies = available_studies or []
        self.storage_url = storage_url
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.available_studies:
            msg += f". Available studies: {', '.join(self.available_studies[:5])}"
            if len(self.available_studies) > 5:
                msg += f" (and {len(self.available_studies) - 5} more)"
        if self.storage_url:
            msg += f". Storage: {self.storage_url}"
        return msg


class BackendError(HPOError):
    """
    Exception raised for backend-specific errors.
    
    Attributes:
        message: Description of the error
        backend_name: Name of the backend (optuna, ray_tune)
        operation: Operation that failed
    """
    
    def __init__(
        self,
        message: str,
        backend_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.backend_name = backend_name
        self.operation = operation
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.backend_name:
            msg += f". Backend: {self.backend_name}"
        if self.operation:
            msg += f", Operation: {self.operation}"
        return msg


class SearchSpaceError(HPOError):
    """
    Exception raised for search space definition errors.
    
    Attributes:
        message: Description of the error
        parameter_name: Name of the problematic parameter
        parameter_config: Configuration that caused the error
    """
    
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.parameter_name = parameter_name
        self.parameter_config = parameter_config
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.parameter_name:
            msg += f". Parameter: '{self.parameter_name}'"
        return msg


class PruningError(HPOError):
    """
    Exception raised for pruning-related errors.
    
    Attributes:
        message: Description of the error
        trial_number: Trial that was being pruned
        pruner_type: Type of pruner being used
        intermediate_value: Value that triggered pruning decision
    """
    
    def __init__(
        self,
        message: str,
        trial_number: Optional[int] = None,
        pruner_type: Optional[str] = None,
        intermediate_value: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, trial_number=trial_number, **kwargs)
        self.pruner_type = pruner_type
        self.intermediate_value = intermediate_value
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.pruner_type:
            msg += f". Pruner: {self.pruner_type}"
        if self.intermediate_value is not None:
            msg += f", Value: {self.intermediate_value}"
        return msg
```

### 4.2 Backend Protocol (`backends/base.py`)

```python
# Location: milia_pipeline/models/hpo/backends/base.py
# Pattern: Follows DatasetHandlerProtocol (datasets/protocols.py)

"""
HPO Backend Protocol

Defines the abstract interface for HPO backends.
Enables swapping between Optuna and Ray Tune without code changes.
"""

from typing import Protocol, Dict, Any, Optional, Callable, List, runtime_checkable
from abc import abstractmethod

# Import types for type hints
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@runtime_checkable
class HPOBackendProtocol(Protocol):
    """
    Protocol defining the interface for HPO backends.
    
    Pattern: Follows DatasetHandlerProtocol (datasets/protocols.py)
    
    All HPO backends must implement these 6 methods:
    1. create_study() - Initialize optimization study
    2. optimize() - Run optimization loop
    3. get_best_params() - Retrieve best hyperparameters
    4. get_best_value() - Retrieve best objective value
    5. get_all_trials() - Retrieve all trial information
    6. create_pruner() - Create pruner instance
    7. create_sampler() - Create sampler instance
    """
    
    @abstractmethod
    def create_study(
        self,
        study_name: str,
        direction: str,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        sampler: Optional[Any] = None,
        pruner: Optional[Any] = None,
    ) -> Any:
        """
        Create or load an HPO study.
        
        Args:
            study_name: Name for the study
            direction: "minimize" or "maximize"
            storage: Storage URL (None for in-memory)
            load_if_exists: Whether to resume existing study
            sampler: Sampler instance
            pruner: Pruner instance
            
        Returns:
            Study object (backend-specific type)
        """
        ...
    
    @abstractmethod
    def optimize(
        self,
        study: Any,
        objective_fn: Callable[[Any], float],
        n_trials: int,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        catch: tuple = (),
        callbacks: Optional[List[Callable]] = None,
    ) -> None:
        """
        Run optimization on the study.
        
        Args:
            study: Study object from create_study()
            objective_fn: Function that takes trial and returns metric
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            n_jobs: Number of parallel jobs
            catch: Exceptions to catch and mark as failed trials
            callbacks: Optuna-style callbacks
        """
        ...
    
    @abstractmethod
    def get_best_params(self, study: Any) -> Dict[str, Any]:
        """
        Get best hyperparameters from completed study.
        
        Args:
            study: Completed study object
            
        Returns:
            Dict of parameter name to best value
        """
        ...
    
    @abstractmethod
    def get_best_value(self, study: Any) -> float:
        """
        Get best objective value from completed study.
        
        Args:
            study: Completed study object
            
        Returns:
            Best objective value
        """
        ...
    
    @abstractmethod
    def get_all_trials(self, study: Any) -> List[Dict[str, Any]]:
        """
        Get information about all trials.
        
        Args:
            study: Study object
            
        Returns:
            List of trial info dicts with keys:
            - number: Trial number
            - params: Hyperparameters
            - value: Objective value (None if not completed)
            - state: Trial state (COMPLETE, PRUNED, FAIL, etc.)
            - duration: Trial duration in seconds
        """
        ...
    
    @abstractmethod
    def create_pruner(
        self,
        pruner_type: str,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 10,
        **kwargs
    ) -> Any:
        """
        Create a pruner instance.
        
        Args:
            pruner_type: Type of pruner (median, hyperband, etc.)
            n_startup_trials: Trials before pruning begins
            n_warmup_steps: Steps before pruning within trial
            **kwargs: Additional pruner-specific arguments
            
        Returns:
            Pruner instance (backend-specific type)
        """
        ...
    
    @abstractmethod
    def create_sampler(
        self,
        sampler_type: str,
        seed: Optional[int] = None,
        n_startup_trials: int = 10,
        **kwargs
    ) -> Any:
        """
        Create a sampler instance.
        
        Args:
            sampler_type: Type of sampler (tpe, random, cmaes, etc.)
            seed: Random seed for reproducibility
            n_startup_trials: Random trials before Bayesian optimization
            **kwargs: Additional sampler-specific arguments
            
        Returns:
            Sampler instance (backend-specific type)
        """
        ...


def get_backend(backend_name: str) -> HPOBackendProtocol:
    """
    Factory function to get HPO backend by name.
    
    Pattern: Follows create_dataset_handler() (handlers/__init__.py)
    
    Args:
        backend_name: Name of backend ("optuna" or "ray_tune")
        
    Returns:
        Backend instance implementing HPOBackendProtocol
        
    Raises:
        BackendError: If backend not found or not available
    """
    from .optuna_backend import OptunaBackend
    from milia_pipeline.exceptions import BackendError
    
    backends = {
        'optuna': OptunaBackend,
        # 'ray_tune': RayTuneBackend,  # Future
    }
    
    if backend_name not in backends:
        available = ', '.join(backends.keys())
        raise BackendError(
            f"Unknown HPO backend: '{backend_name}'",
            backend_name=backend_name,
            details=f"Available backends: {available}"
        )
    
    backend_cls = backends[backend_name]
    
    try:
        return backend_cls()
    except ImportError as e:
        raise BackendError(
            f"Backend '{backend_name}' dependencies not installed",
            backend_name=backend_name,
            details=str(e)
        )
```
---

## Part 5: Optuna Backend and Callback Implementations

### 5.1 Optuna Backend (`backends/optuna_backend.py`)               <--- optuna_backend

```python
# Location: milia_pipeline/models/hpo/backends/optuna_backend.py

"""
Optuna Backend Implementation

Primary HPO backend using Optuna's Tree-Parzen Estimators (TPE).
"""

import logging
from typing import Dict, Any, Optional, Callable, List

try:
    import optuna
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TrialState = None

from .base import HPOBackendProtocol
from milia_pipeline.exceptions import BackendError, HPOError

logger = logging.getLogger(__name__)


class OptunaBackend:
    """
    Optuna HPO backend implementation.
    
    Implements HPOBackendProtocol using Optuna library.
    
    Features:
        - TPE, CMA-ES, Random samplers
        - Median, Hyperband, Successive Halving pruners
        - SQLite/PostgreSQL storage for persistence
        - Parallel optimization support
        - Multi-objective optimization
    
    Usage:
        >>> backend = OptunaBackend()
        >>> study = backend.create_study("my_study", "minimize")
        >>> backend.optimize(study, objective_fn, n_trials=100)
        >>> best_params = backend.get_best_params(study)
    """
    
    def __init__(self):
        """Initialize Optuna backend."""
        if not OPTUNA_AVAILABLE:
            raise BackendError(
                "Optuna is not installed",
                backend_name="optuna",
                details="Install with: pip install optuna"
            )
        
        logger.info("OptunaBackend initialized")
    
    def create_study(
        self,
        study_name: str,
        direction: str,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        sampler: Optional[Any] = None,
        pruner: Optional[Any] = None,
    ) -> 'optuna.Study':
        """
        Create or load an Optuna study.
        
        Args:
            study_name: Name for the study
            direction: "minimize" or "maximize"
            storage: Storage URL (e.g., "sqlite:///optuna.db")
            load_if_exists: Whether to resume existing study
            sampler: Optuna sampler instance
            pruner: Optuna pruner instance
            
        Returns:
            Optuna Study object
        """
        try:
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                storage=storage,
                load_if_exists=load_if_exists,
                sampler=sampler,
                pruner=pruner,
            )
            
            n_existing = len(study.trials)
            if n_existing > 0:
                logger.info(
                    f"Resumed study '{study_name}' with {n_existing} existing trials"
                )
            else:
                logger.info(f"Created new study '{study_name}'")
            
            return study
            
        except optuna.exceptions.DuplicatedStudyError:
            # This shouldn't happen with load_if_exists=True, but handle it
            logger.warning(
                f"Study '{study_name}' exists, loading..."
            )
            return optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
            )
        except Exception as e:
            raise BackendError(
                f"Failed to create study: {e}",
                backend_name="optuna",
                operation="create_study",
                details=str(e)
            )
    
    def optimize(
        self,
        study: 'optuna.Study',
        objective_fn: Callable[['optuna.Trial'], float],
        n_trials: int,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        catch: tuple = (Exception,),
        callbacks: Optional[List[Callable]] = None,
    ) -> None:
        """
        Run optimization on the study.
        
        Args:
            study: Optuna Study object
            objective_fn: Objective function taking trial, returning metric
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            n_jobs: Number of parallel jobs
            catch: Exceptions to catch and mark as failed
            callbacks: Optuna callbacks
        """
        logger.info(
            f"Starting optimization: {n_trials} trials, "
            f"{n_jobs} jobs, timeout={timeout}s"
        )
        
        try:
            study.optimize(
                objective_fn,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                catch=catch,
                callbacks=callbacks or [],
                show_progress_bar=True,
            )
            
            # Log summary
            completed = len([
                t for t in study.trials 
                if t.state == TrialState.COMPLETE
            ])
            pruned = len([
                t for t in study.trials 
                if t.state == TrialState.PRUNED
            ])
            failed = len([
                t for t in study.trials 
                if t.state == TrialState.FAIL
            ])
            
            logger.info(
                f"Optimization complete: {completed} completed, "
                f"{pruned} pruned, {failed} failed"
            )
            
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
            raise
        except Exception as e:
            raise BackendError(
                f"Optimization failed: {e}",
                backend_name="optuna",
                operation="optimize",
                details=str(e)
            )
    
    def get_best_params(self, study: 'optuna.Study') -> Dict[str, Any]:
        """Get best hyperparameters from study."""
        try:
            return study.best_params
        except ValueError as e:
            # No completed trials
            raise HPOError(
                "No completed trials in study",
                study_name=study.study_name,
                details=str(e)
            )
    
    def get_best_value(self, study: 'optuna.Study') -> float:
        """Get best objective value from study."""
        try:
            return study.best_value
        except ValueError as e:
            raise HPOError(
                "No completed trials in study",
                study_name=study.study_name,
                details=str(e)
            )
    
    def get_all_trials(self, study: 'optuna.Study') -> List[Dict[str, Any]]:
        """Get information about all trials."""
        trials_info = []
        
        for trial in study.trials:
            trial_info = {
                'number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name,
                'duration': (
                    (trial.datetime_complete - trial.datetime_start).total_seconds()
                    if trial.datetime_complete and trial.datetime_start
                    else None
                ),
                'user_attrs': trial.user_attrs,
                'intermediate_values': trial.intermediate_values,
            }
            trials_info.append(trial_info)
        
        return trials_info
    
    def create_pruner(
        self,
        pruner_type: str,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 10,
        **kwargs
    ) -> 'optuna.pruners.BasePruner':
        """
        Create an Optuna pruner instance.
        
        Args:
            pruner_type: Type of pruner
            n_startup_trials: Trials before pruning begins
            n_warmup_steps: Steps before pruning within trial
            
        Returns:
            Optuna pruner instance
        """
        pruner_map = {
            'median': optuna.pruners.MedianPruner,
            'percentile': optuna.pruners.PercentilePruner,
            'hyperband': optuna.pruners.HyperbandPruner,
            'successive_halving': optuna.pruners.SuccessiveHalvingPruner,
            'threshold': optuna.pruners.ThresholdPruner,
            'patient': optuna.pruners.PatientPruner,
            'none': optuna.pruners.NopPruner,
        }
        
        if pruner_type not in pruner_map:
            available = ', '.join(pruner_map.keys())
            raise BackendError(
                f"Unknown pruner type: '{pruner_type}'",
                backend_name="optuna",
                operation="create_pruner",
                details=f"Available pruners: {available}"
            )
        
        pruner_cls = pruner_map[pruner_type]
        
        # Build pruner kwargs based on type
        pruner_kwargs = {}
        
        if pruner_type in ('median', 'percentile'):
            pruner_kwargs = {
                'n_startup_trials': n_startup_trials,
                'n_warmup_steps': n_warmup_steps,
                'interval_steps': kwargs.get('interval_steps', 1),
            }
            if pruner_type == 'percentile':
                pruner_kwargs['percentile'] = kwargs.get('percentile', 25.0)
        
        elif pruner_type == 'hyperband':
            pruner_kwargs = {
                'min_resource': kwargs.get('min_resource', 1),
                'max_resource': kwargs.get('max_resource', n_warmup_steps + 100),
                'reduction_factor': kwargs.get('reduction_factor', 3),
            }
        
        elif pruner_type == 'successive_halving':
            pruner_kwargs = {
                'min_resource': kwargs.get('min_resource', 1),
                'reduction_factor': kwargs.get('reduction_factor', 4),
                'min_early_stopping_rate': kwargs.get('min_early_stopping_rate', 0),
            }
        
        elif pruner_type == 'threshold':
            pruner_kwargs = {
                'lower': kwargs.get('lower'),
                'upper': kwargs.get('upper'),
                'n_warmup_steps': n_warmup_steps,
            }
        
        elif pruner_type == 'patient':
            # PatientPruner wraps another pruner
            wrapped_pruner = self.create_pruner(
                kwargs.get('wrapped_pruner_type', 'median'),
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
            )
            pruner_kwargs = {
                'wrapped_pruner': wrapped_pruner,
                'patience': kwargs.get('patience', 5),
            }
        
        logger.debug(f"Creating {pruner_type} pruner with kwargs: {pruner_kwargs}")
        return pruner_cls(**pruner_kwargs)
    
    def create_sampler(
        self,
        sampler_type: str,
        seed: Optional[int] = None,
        n_startup_trials: int = 10,
        **kwargs
    ) -> 'optuna.samplers.BaseSampler':
        """
        Create an Optuna sampler instance.
        
        Args:
            sampler_type: Type of sampler
            seed: Random seed
            n_startup_trials: Random trials before Bayesian optimization
            
        Returns:
            Optuna sampler instance
        """
        sampler_map = {
            'tpe': optuna.samplers.TPESampler,
            'random': optuna.samplers.RandomSampler,
            'cmaes': optuna.samplers.CmaEsSampler,
            'grid': optuna.samplers.GridSampler,
            'nsgaii': optuna.samplers.NSGAIISampler,
            'motpe': optuna.samplers.MOTPESampler,
            'qmc': optuna.samplers.QMCSampler,
        }
        
        if sampler_type not in sampler_map:
            available = ', '.join(sampler_map.keys())
            raise BackendError(
                f"Unknown sampler type: '{sampler_type}'",
                backend_name="optuna",
                operation="create_sampler",
                details=f"Available samplers: {available}"
            )
        
        sampler_cls = sampler_map[sampler_type]
        
        # Build sampler kwargs based on type
        sampler_kwargs = {'seed': seed}
        
        if sampler_type == 'tpe':
            sampler_kwargs.update({
                'n_startup_trials': n_startup_trials,
                'multivariate': kwargs.get('multivariate', True),
                'constant_liar': kwargs.get('constant_liar', False),
            })
        
        elif sampler_type == 'cmaes':
            sampler_kwargs.update({
                'n_startup_trials': n_startup_trials,
                'restart_strategy': kwargs.get('restart_strategy', 'ipop'),
            })
        
        elif sampler_type == 'grid':
            # Grid sampler requires search_space
            if 'search_space' not in kwargs:
                raise BackendError(
                    "GridSampler requires 'search_space' kwarg",
                    backend_name="optuna",
                    operation="create_sampler"
                )
            sampler_kwargs = {'search_space': kwargs['search_space']}
        
        logger.debug(f"Creating {sampler_type} sampler with seed={seed}")
        return sampler_cls(**sampler_kwargs)
    
    def suggest_params(
        self,
        trial: 'optuna.Trial',
        search_space: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters from search space.
        
        Utility method that maps search space config to Optuna suggest calls.
        
        Args:
            trial: Optuna Trial object
            search_space: Search space configuration
            
        Returns:
            Dict of suggested parameter values
        """
        params = {}
        
        for category, category_params in search_space.items():
            for param_name, config in category_params.items():
                full_name = f"{category}.{param_name}"
                
                param_type = config.type if hasattr(config, 'type') else config.get('type')
                
                if isinstance(param_type, str):
                    param_type_str = param_type
                else:
                    param_type_str = param_type.value if hasattr(param_type, 'value') else str(param_type)
                
                if param_type_str == 'int':
                    low = config.low if hasattr(config, 'low') else config.get('low')
                    high = config.high if hasattr(config, 'high') else config.get('high')
                    step = config.step if hasattr(config, 'step') else config.get('step', 1)
                    params[full_name] = trial.suggest_int(full_name, int(low), int(high), step=step)
                
                elif param_type_str == 'float':
                    low = config.low if hasattr(config, 'low') else config.get('low')
                    high = config.high if hasattr(config, 'high') else config.get('high')
                    log = config.log if hasattr(config, 'log') else config.get('log', False)
                    params[full_name] = trial.suggest_float(full_name, low, high, log=log)
                
                elif param_type_str == 'loguniform':
                    low = config.low if hasattr(config, 'low') else config.get('low')
                    high = config.high if hasattr(config, 'high') else config.get('high')
                    params[full_name] = trial.suggest_float(full_name, low, high, log=True)
                
                elif param_type_str == 'categorical':
                    choices = config.choices if hasattr(config, 'choices') else config.get('choices')
                    params[full_name] = trial.suggest_categorical(full_name, choices)
                
                elif param_type_str == 'uniform':
                    low = config.low if hasattr(config, 'low') else config.get('low')
                    high = config.high if hasattr(config, 'high') else config.get('high')
                    params[full_name] = trial.suggest_float(full_name, low, high)
                
                else:
                    raise BackendError(
                        f"Unknown parameter type: '{param_type_str}'",
                        backend_name="optuna",
                        operation="suggest_params",
                        details=f"Parameter: {full_name}"
                    )
        
        return params
```

### 5.2 Optuna Pruning Callback (`callbacks/optuna_callback.py`)            <--- DONE!

```python
# Location: milia_pipeline/models/hpo/callbacks/optuna_callback.py
# Pattern: callbacks.py lines 35-96 (Callback ABC), 103-227 (EarlyStopping)

"""
Optuna Pruning Callback

Integrates Optuna's pruning mechanism with MILIA's callback system.
"""

import logging
from typing import Dict, Any, Optional

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

# Import from existing callback system
try:
    from milia_pipeline.models.training.callbacks import Callback
except ImportError:
    # Fallback for testing
    class Callback:
        def set_trainer(self, trainer): pass
        def on_train_begin(self, **kwargs): pass
        def on_epoch_end(self, trainer, epoch, metrics): pass
        def on_train_end(self, trainer, metrics): pass

from milia_pipeline.exceptions import PruningError

logger = logging.getLogger(__name__)


class OptunaPruningCallback(Callback):
    """
    Callback for Optuna trial pruning during training.
    
    Extends the MILIA Callback ABC to integrate with Optuna's pruning system.
    Reports intermediate metrics to the trial and checks for pruning decisions.
    
    Pattern: Extends Callback ABC (callbacks.py:35-96)
    Inspired by: EarlyStopping (callbacks.py:103-227)
    
    Attributes:
        trial: Optuna Trial object
        monitor: Metric name to monitor for pruning
        report_every: Report metric every N epochs (default: 1)
    
    Usage:
        >>> trial = ...  # From Optuna objective function
        >>> callback = OptunaPruningCallback(
        ...     trial=trial,
        ...     monitor="val_loss",
        ...     report_every=1
        ... )
        >>> trainer.fit(callbacks=[callback])
    
    Integration Point:
        This callback is automatically created by HPOManager and injected
        into the Trainer's callback list during HPO trials.
    """
    
    def __init__(
        self,
        trial: 'optuna.Trial',
        monitor: str = "val_loss",
        report_every: int = 1,
    ):
        """
        Initialize OptunaPruningCallback.
        
        Args:
            trial: Optuna Trial object from objective function
            monitor: Metric name to report (must be in Trainer metrics)
            report_every: Frequency of metric reporting
        """
        super().__init__()
        
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for OptunaPruningCallback. "
                "Install with: pip install optuna"
            )
        
        self.trial = trial
        self.monitor = monitor
        self.report_every = report_every
        self._trainer = None
        self._last_reported_value: Optional[float] = None
        
        logger.debug(
            f"OptunaPruningCallback initialized: "
            f"monitor='{monitor}', report_every={report_every}"
        )
    
    def set_trainer(self, trainer) -> None:
        """
        Store reference to trainer.
        
        Pattern: Follows Callback.set_trainer() (callbacks.py:61-65)
        """
        self._trainer = trainer
    
    def on_train_begin(self, **kwargs) -> None:
        """
        Called at the beginning of training.
        
        Pattern: Follows Callback.on_train_begin() (callbacks.py:67-72)
        """
        logger.debug(
            f"Training started for trial {self.trial.number}, "
            f"monitoring '{self.monitor}'"
        )
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Report metric to Optuna and check for pruning.
        
        Pattern: Follows EarlyStopping.on_epoch_end() (callbacks.py:160-202)
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Dict of metric names to values
            
        Raises:
            optuna.TrialPruned: If trial should be pruned
            PruningError: If monitor metric not found in metrics
        """
        # Check if we should report this epoch
        if self.report_every > 1 and epoch % self.report_every != 0:
            return
        
        # Get monitored metric value
        value = metrics.get(self.monitor)
        
        if value is None:
            # Check alternative metric names
            alt_names = [
                f"val_{self.monitor}",
                f"validation_{self.monitor}",
                self.monitor.replace("val_", ""),
            ]
            for alt in alt_names:
                if alt in metrics:
                    value = metrics[alt]
                    break
        
        if value is None:
            available = ', '.join(sorted(metrics.keys()))
            logger.warning(
                f"Metric '{self.monitor}' not found in epoch {epoch} metrics. "
                f"Available: {available}"
            )
            # Don't raise - allow training to continue
            # This handles cases where metric is computed less frequently
            return
        
        # Store for reference
        self._last_reported_value = value
        
        # Report to Optuna
        self.trial.report(value, epoch)
        
        logger.debug(
            f"Trial {self.trial.number}, Epoch {epoch}: "
            f"{self.monitor}={value:.6f}"
        )
        
        # Check for pruning
        if self.trial.should_prune():
            logger.info(
                f"Trial {self.trial.number} pruned at epoch {epoch} "
                f"({self.monitor}={value:.6f})"
            )
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} with {self.monitor}={value}"
            )
    
    def on_train_end(self, trainer, metrics: Dict[str, float]) -> None:
        """
        Called at the end of training.
        
        Pattern: Follows Callback.on_train_end() (callbacks.py:90-96)
        """
        final_value = metrics.get(self.monitor, self._last_reported_value)
        logger.debug(
            f"Training ended for trial {self.trial.number}, "
            f"final {self.monitor}={final_value}"
        )
    
    @property
    def should_stop(self) -> bool:
        """
        Check if training should stop due to pruning.
        
        This property allows integration with existing early stopping logic.
        
        Returns:
            True if trial is pruned, False otherwise
        """
        return self.trial.should_prune()


def create_hpo_callback(
    trial: 'optuna.Trial',
    monitor: str = "val_loss",
    report_every: int = 1,
    backend: str = "optuna"
) -> Callback:
    """
    Factory function to create HPO callback for the specified backend.
    
    Pattern: Follows create_dataset_handler() (handlers/__init__.py)
    
    Args:
        trial: Trial object from HPO backend
        monitor: Metric name to monitor
        report_every: Reporting frequency
        backend: Backend name ("optuna" or "ray_tune")
        
    Returns:
        Callback instance appropriate for the backend
    """
    if backend == "optuna":
        return OptunaPruningCallback(
            trial=trial,
            monitor=monitor,
            report_every=report_every
        )
    elif backend == "ray_tune":
        # Future: RayTuneReportCallback
        raise NotImplementedError(
            "Ray Tune callback not yet implemented. Use 'optuna' backend."
        )
    else:
        raise ValueError(f"Unknown backend: '{backend}'")
```


---

## Part 6: HPO Manager Implementation

### 6.1 HPOManager Class (`hpo_manager.py`)

```python
# Location: milia_pipeline/models/hpo/hpo_manager.py
# Pattern: Follows ModelFactory (model_factory.py:315-953)

"""
HPO Manager

Main orchestrator for hyperparameter optimization.
Coordinates between backend, callbacks, search spaces, and training.
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from dataclasses import asdict
import time

# Import existing modules
try:
    from milia_pipeline.models.training.trainer import Trainer
    from milia_pipeline.models.training.data_splitting import DataSplitter
    from milia_pipeline.models.factory.model_factory import ModelFactory, get_factory
    from milia_pipeline.models.training.loss_functions import LossRegistry
    from milia_pipeline.models.training.schedulers import SchedulerRegistry
    from milia_pipeline.models.acceleration.device_manager import DeviceManager
except ImportError as e:
    # Allow partial imports for testing
    Trainer = None
    DataSplitter = None
    ModelFactory = None
    get_factory = None
    LossRegistry = None
    SchedulerRegistry = None
    DeviceManager = None

from .hpo_config import (
    HPOConfig,
    SearchSpaceParamConfig,
    PrunerConfig,
    SamplerConfig,
    StudyConfig,
    ParamType,
)
from milia_pipeline.exceptions import (
    HPOError,
    HPOConfigurationError,
    TrialFailedError,
    StudyNotFoundError,
)
from .backends import get_backend, HPOBackendProtocol
from .callbacks import create_hpo_callback

logger = logging.getLogger(__name__)


class HPOManager:
    """
    Main HPO orchestrator class.
    
    Coordinates hyperparameter optimization by:
    1. Setting up the HPO backend (Optuna/Ray Tune)
    2. Creating search space from configuration
    3. Managing trial execution via objective function
    4. Integrating with Trainer for model training
    5. Handling cross-validation if configured
    
    Pattern: Follows ModelFactory (model_factory.py:315-953)
    
    Attributes:
        config: HPOConfig instance
        backend: HPO backend instance
        study: Current study object (set after optimize())
        best_params: Best hyperparameters found (set after optimize())
    
    Usage:
        >>> # From configuration
        >>> manager = HPOManager.from_config(hpo_config)
        >>> 
        >>> # Run optimization
        >>> best_params = manager.optimize(
        ...     model_name="GCN",
        ...     dataset=dataset,
        ...     base_hyperparameters={"num_layers": 3}
        ... )
        >>> 
        >>> # Get results
        >>> print(f"Best params: {best_params}")
        >>> print(f"Best value: {manager.get_best_value()}")
    """
    
    def __init__(self, config: HPOConfig):
        """
        Initialize HPOManager.
        
        Args:
            config: HPOConfig instance with all HPO settings
        """
        if not config.enabled:
            logger.warning(
                "HPO is disabled in config. "
                "Set hpo.enabled=True to enable optimization."
            )
        
        self.config = config
        self.backend: Optional[HPOBackendProtocol] = None
        self.study: Optional[Any] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self._model_factory: Optional[ModelFactory] = None
        
        # Initialize backend if enabled
        if config.enabled:
            self.backend = get_backend(config.backend)
            logger.info(
                f"HPOManager initialized with {config.backend} backend, "
                f"n_trials={config.n_trials}"
            )
    
    @classmethod
    def from_config(cls, config: Union[HPOConfig, Dict[str, Any]]) -> 'HPOManager':
        """
        Create HPOManager from configuration.
        
        Pattern: Follows ModelConfig.from_yaml() (config_bridge.py:967-993)
        
        Args:
            config: HPOConfig instance or dict
            
        Returns:
            HPOManager instance
        """
        if isinstance(config, dict):
            config = HPOConfig.from_dict(config)
        
        return cls(config)
    
    @classmethod
    def from_yaml(cls, config_path: str, section: str = "models.hpo") -> 'HPOManager':
        """
        Create HPOManager from YAML configuration file.
        
        Args:
            config_path: Path to config.yaml
            section: Config section path (default: "models.hpo")
            
        Returns:
            HPOManager instance
        """
        import yaml
        
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Navigate to section
        hpo_config = full_config
        for key in section.split('.'):
            hpo_config = hpo_config.get(key, {})
        
        return cls.from_config(hpo_config)
    
    def optimize(
        self,
        model_name: str,
        dataset,
        base_hyperparameters: Optional[Dict[str, Any]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            model_name: Name of model to optimize (from ModelRegistry)
            dataset: PyG dataset or DataLoader
            base_hyperparameters: Fixed hyperparameters not being optimized
            trainer_kwargs: Additional kwargs for Trainer
            callbacks: Additional callbacks (HPO callback added automatically)
            
        Returns:
            Dict of best hyperparameters found
            
        Raises:
            HPOError: If HPO is disabled or optimization fails
            HPOConfigurationError: If configuration is invalid
        """
        if not self.config.enabled:
            raise HPOError(
                "HPO is disabled. Set hpo.enabled=True in config.",
                details="Cannot run optimization when disabled"
            )
        
        if self.backend is None:
            raise HPOError(
                "Backend not initialized",
                details="Call from_config() or ensure enabled=True"
            )
        
        base_hyperparameters = base_hyperparameters or {}
        trainer_kwargs = trainer_kwargs or {}
        callbacks = callbacks or []
        
        logger.info(
            f"Starting HPO for model '{model_name}' with "
            f"{self.config.n_trials} trials"
        )
        
        # Get model factory
        self._model_factory = get_factory() if get_factory else None
        
        # Create pruner and sampler
        pruner = self.backend.create_pruner(
            pruner_type=self.config.pruner.type.value,
            n_startup_trials=self.config.pruner.n_startup_trials,
            n_warmup_steps=self.config.pruner.n_warmup_steps,
            interval_steps=self.config.pruner.interval_steps,
            percentile=self.config.pruner.percentile,
        )
        
        sampler = self.backend.create_sampler(
            sampler_type=self.config.sampler.type.value,
            seed=self.config.sampler.seed,
            n_startup_trials=self.config.sampler.n_startup_trials,
            multivariate=self.config.sampler.multivariate,
            constant_liar=self.config.sampler.constant_liar,
        )
        
        # Create study
        self.study = self.backend.create_study(
            study_name=self.config.study.study_name,
            direction=self.config.study.direction.value,
            storage=self.config.study.storage,
            load_if_exists=self.config.study.load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )
        
        # Create objective function
        objective_fn = self._create_objective(
            model_name=model_name,
            dataset=dataset,
            base_hyperparameters=base_hyperparameters,
            trainer_kwargs=trainer_kwargs,
            additional_callbacks=callbacks,
        )
        
        # Run optimization
        start_time = time.time()
        
        self.backend.optimize(
            study=self.study,
            objective_fn=objective_fn,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            catch=(Exception,),  # Catch all, mark as failed
        )
        
        elapsed = time.time() - start_time
        
        # Get results
        self.best_params = self.backend.get_best_params(self.study)
        best_value = self.backend.get_best_value(self.study)
        
        logger.info(
            f"HPO completed in {elapsed:.1f}s. "
            f"Best {self.config.study.metric}: {best_value:.6f}"
        )
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def _create_objective(
        self,
        model_name: str,
        dataset,
        base_hyperparameters: Dict[str, Any],
        trainer_kwargs: Dict[str, Any],
        additional_callbacks: List,
    ) -> Callable:
        """
        Create objective function for optimization.
        
        The objective function:
        1. Suggests hyperparameters from search space
        2. Merges with base hyperparameters
        3. Creates model with suggested params
        4. Trains model with pruning callback
        5. Returns metric value
        
        Args:
            model_name: Model name
            dataset: Dataset for training
            base_hyperparameters: Fixed hyperparameters
            trainer_kwargs: Trainer configuration
            additional_callbacks: User-provided callbacks
            
        Returns:
            Objective function for the backend
        """
        config = self.config
        backend = self.backend
        factory = self._model_factory
        
        def objective(trial) -> float:
            """Objective function for single trial."""
            trial_number = trial.number
            logger.info(f"Starting trial {trial_number}")
            
            try:
                # 1. Suggest hyperparameters
                suggested_params = backend.suggest_params(
                    trial, config.search_space
                )
                
                # 2. Flatten and merge parameters
                flat_params = _flatten_params(suggested_params)
                hyperparameters = {**base_hyperparameters, **flat_params}
                
                logger.debug(f"Trial {trial_number} params: {flat_params}")
                
                # 3. Extract special parameters
                model_params, optimizer_params, scheduler_params, loss_params = \
                    _extract_param_categories(hyperparameters)
                
                # 4. Create model
                if factory:
                    model = factory.create_model(
                        model_name=model_name,
                        hyperparameters=model_params,
                        sample_data=dataset[0] if hasattr(dataset, '__getitem__') else None,
                    )
                else:
                    raise HPOError(
                        "ModelFactory not available",
                        trial_number=trial_number
                    )
                
                # 5. Create HPO callback
                hpo_callback = create_hpo_callback(
                    trial=trial,
                    monitor=config.study.metric,
                    report_every=1,
                    backend=config.backend,
                )
                
                all_callbacks = [hpo_callback] + additional_callbacks
                
                # 6. Handle cross-validation if configured
                if config.cv_folds > 0:
                    metric_value = _run_cross_validation(
                        model_name=model_name,
                        dataset=dataset,
                        model_params=model_params,
                        optimizer_params=optimizer_params,
                        scheduler_params=scheduler_params,
                        loss_params=loss_params,
                        trainer_kwargs=trainer_kwargs,
                        callbacks=all_callbacks,
                        n_folds=config.cv_folds,
                        metric=config.study.metric,
                        aggregation=config.cv_metric_aggregation,
                        factory=factory,
                    )
                else:
                    # 7. Standard training
                    trainer = Trainer(
                        model=model,
                        **trainer_kwargs,
                    )
                    
                    # Merge callbacks
                    existing_callbacks = trainer_kwargs.get('callbacks', [])
                    trainer.callbacks = existing_callbacks + all_callbacks
                    
                    # Train
                    results = trainer.fit(dataset)
                    
                    # Get metric
                    metric_value = results.get(
                        config.study.metric,
                        results.get('best_val_loss')
                    )
                
                if metric_value is None:
                    raise TrialFailedError(
                        f"Metric '{config.study.metric}' not found in results",
                        trial_number=trial_number,
                        trial_params=flat_params,
                    )
                
                logger.info(
                    f"Trial {trial_number} completed: "
                    f"{config.study.metric}={metric_value:.6f}"
                )
                
                return metric_value
                
            except Exception as e:
                # Check if it's a pruning exception
                import optuna
                if isinstance(e, optuna.TrialPruned):
                    raise  # Re-raise pruning
                
                logger.error(f"Trial {trial_number} failed: {e}")
                raise TrialFailedError(
                    f"Trial failed: {e}",
                    trial_number=trial_number,
                    original_error=str(e),
                )
        
        return objective
    
    def get_best_value(self) -> float:
        """Get best objective value from completed study."""
        if self.study is None:
            raise HPOError("No study available. Run optimize() first.")
        return self.backend.get_best_value(self.study)
    
    def get_best_trial(self) -> Dict[str, Any]:
        """Get information about the best trial."""
        if self.study is None:
            raise HPOError("No study available. Run optimize() first.")
        
        all_trials = self.backend.get_all_trials(self.study)
        best_value = self.get_best_value()
        
        for trial in all_trials:
            if trial['value'] == best_value:
                return trial
        
        raise HPOError("Could not find best trial")
    
    def get_all_trials(self) -> List[Dict[str, Any]]:
        """Get information about all trials."""
        if self.study is None:
            raise HPOError("No study available. Run optimize() first.")
        return self.backend.get_all_trials(self.study)
    
    def get_study_statistics(self) -> Dict[str, Any]:
        """Get study statistics summary."""
        if self.study is None:
            raise HPOError("No study available. Run optimize() first.")
        
        trials = self.get_all_trials()
        
        completed = [t for t in trials if t['state'] == 'COMPLETE']
        pruned = [t for t in trials if t['state'] == 'PRUNED']
        failed = [t for t in trials if t['state'] == 'FAIL']
        
        values = [t['value'] for t in completed if t['value'] is not None]
        durations = [t['duration'] for t in completed if t['duration'] is not None]
        
        return {
            'n_trials': len(trials),
            'n_completed': len(completed),
            'n_pruned': len(pruned),
            'n_failed': len(failed),
            'best_value': min(values) if values else None,
            'worst_value': max(values) if values else None,
            'mean_value': sum(values) / len(values) if values else None,
            'mean_duration': sum(durations) / len(durations) if durations else None,
            'total_duration': sum(durations) if durations else None,
            'pruning_rate': len(pruned) / len(trials) if trials else 0,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _flatten_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested parameter dict.
    
    Converts "hyperparameters.hidden_channels" to "hidden_channels".
    """
    flat = {}
    for key, value in params.items():
        # Remove category prefix
        if '.' in key:
            flat_key = key.split('.')[-1]
        else:
            flat_key = key
        flat[flat_key] = value
    return flat


def _extract_param_categories(
    params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Separate parameters into categories.
    
    Returns:
        (model_params, optimizer_params, scheduler_params, loss_params)
    """
    optimizer_keys = {'lr', 'learning_rate', 'weight_decay', 'momentum', 'betas'}
    scheduler_keys = {'factor', 'patience', 'step_size', 'gamma', 'T_max'}
    loss_keys = {'alpha', 'gamma', 'reduction', 'weight'}
    
    model_params = {}
    optimizer_params = {}
    scheduler_params = {}
    loss_params = {}
    
    for key, value in params.items():
        if key in optimizer_keys:
            optimizer_params[key] = value
        elif key in scheduler_keys:
            scheduler_params[key] = value
        elif key in loss_keys:
            loss_params[key] = value
        else:
            model_params[key] = value
    
    return model_params, optimizer_params, scheduler_params, loss_params


def _run_cross_validation(
    model_name: str,
    dataset,
    model_params: Dict[str, Any],
    optimizer_params: Dict[str, Any],
    scheduler_params: Dict[str, Any],
    loss_params: Dict[str, Any],
    trainer_kwargs: Dict[str, Any],
    callbacks: List,
    n_folds: int,
    metric: str,
    aggregation: str,
    factory,
) -> float:
    """
    Run k-fold cross-validation for a trial.
    
    Uses DataSplitter.k_fold_split() from data_splitting.py.
    
    Returns:
        Aggregated metric value across folds
    """
    from statistics import mean, median
    
    # Get folds
    folds = DataSplitter.k_fold_split(
        dataset=dataset,
        n_splits=n_folds,
        random_seed=42,
    )
    
    fold_metrics = []
    
    for fold_idx, (train_subset, val_subset) in enumerate(folds):
        logger.debug(f"  Fold {fold_idx + 1}/{n_folds}")
        
        # Create fresh model for each fold
        model = factory.create_model(
            model_name=model_name,
            hyperparameters=model_params,
            sample_data=train_subset[0] if hasattr(train_subset, '__getitem__') else None,
        )
        
        # Train
        trainer = Trainer(
            model=model,
            **trainer_kwargs,
        )
        trainer.callbacks = callbacks.copy()
        
        results = trainer.fit(
            train_dataset=train_subset,
            val_dataset=val_subset,
        )
        
        fold_value = results.get(metric, results.get('best_val_loss'))
        if fold_value is not None:
            fold_metrics.append(fold_value)
    
    # Aggregate
    if not fold_metrics:
        raise HPOError(f"No valid fold metrics for {metric}")
    
    if aggregation == 'mean':
        return mean(fold_metrics)
    elif aggregation == 'median':
        return median(fold_metrics)
    elif aggregation == 'min':
        return min(fold_metrics)
    elif aggregation == 'max':
        return max(fold_metrics)
    else:
        return mean(fold_metrics)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_hpo_enabled(config: Optional[HPOConfig] = None) -> bool:
    """
    Check if HPO is enabled in configuration.
    
    Args:
        config: HPOConfig or None to check default
        
    Returns:
        True if HPO is enabled
    """
    if config is None:
        return False
    return config.enabled


def get_best_params(manager: HPOManager) -> Dict[str, Any]:
    """
    Get best parameters from completed HPO.
    
    Args:
        manager: HPOManager instance after optimize()
        
    Returns:
        Dict of best hyperparameters
    """
    if manager.best_params is None:
        raise HPOError("No optimization completed yet")
    return manager.best_params


def create_hpo_manager(
    enabled: bool = True,
    n_trials: int = 100,
    **kwargs
) -> HPOManager:
    """
    Convenience function to create HPOManager.
    
    Args:
        enabled: Enable HPO
        n_trials: Number of trials
        **kwargs: Additional HPOConfig parameters
        
    Returns:
        HPOManager instance
    """
    config = HPOConfig(
        enabled=enabled,
        n_trials=n_trials,
        **kwargs
    )
    return HPOManager(config)
```


---

## Part 7: Integration Points and Required Modifications

### 7.1 config.yaml Addition

Add the following section after line 1252 in `config.yaml`:

```yaml
# =============================================================================
# HYPERPARAMETER OPTIMIZATION (HPO)
# =============================================================================
# MASTER SWITCH for enabling/disabling HPO
# When enabled, the training loop will perform hyperparameter search
# instead of single training run.

models:
  # ... existing model configuration ...

  hpo:
    # =========================================================================
    # MASTER SWITCH
    # =========================================================================
    enabled: false                    # Set to true to enable HPO
    
    # =========================================================================
    # BACKEND CONFIGURATION
    # =========================================================================
    backend: "optuna"                 # "optuna" (primary) or "ray_tune" (future)
    n_trials: 100                     # Number of trials to run
    timeout: null                     # Max time in seconds (null = no limit)
    n_jobs: 1                         # Parallel trials (1 = sequential)
    
    # =========================================================================
    # SEARCH SPACE DEFINITION
    # =========================================================================
    # Define hyperparameters to optimize and their ranges
    # Categories: hyperparameters, optimizer, scheduler, loss
    
    search_space:
      # Model hyperparameters
      hyperparameters:
        hidden_channels:
          type: "int"
          low: 32
          high: 256
          step: 32
        
        num_layers:
          type: "int"
          low: 2
          high: 6
        
        dropout:
          type: "float"
          low: 0.0
          high: 0.7
        
        heads:                        # For attention models (GAT, etc.)
          type: "int"
          low: 1
          high: 8
      
      # Optimizer hyperparameters
      optimizer:
        lr:
          type: "loguniform"
          low: 1.0e-5
          high: 1.0e-2
        
        weight_decay:
          type: "loguniform"
          low: 1.0e-6
          high: 1.0e-3
      
      # Scheduler hyperparameters
      scheduler:
        factor:
          type: "float"
          low: 0.1
          high: 0.9
        
        patience:
          type: "int"
          low: 5
          high: 20
      
      # Loss function hyperparameters (for focal loss, etc.)
      loss:
        alpha:
          type: "float"
          low: 0.1
          high: 0.9
    
    # =========================================================================
    # PRUNER CONFIGURATION
    # =========================================================================
    # Controls early stopping of unpromising trials
    
    pruner:
      type: "median"                  # median, hyperband, percentile, none
      n_startup_trials: 5             # Trials before pruning begins
      n_warmup_steps: 10              # Epochs before pruning within trial
      interval_steps: 1               # Check pruning every N steps
      percentile: 25.0                # For percentile pruner
    
    # =========================================================================
    # SAMPLER CONFIGURATION
    # =========================================================================
    # Controls hyperparameter suggestion strategy
    
    sampler:
      type: "tpe"                     # tpe, random, cmaes, grid
      n_startup_trials: 10            # Random trials before Bayesian
      seed: null                      # Random seed (null = random)
      multivariate: true              # Use multivariate TPE
      constant_liar: false            # For parallel optimization
    
    # =========================================================================
    # STUDY CONFIGURATION
    # =========================================================================
    # Optuna study settings
    
    study:
      direction: "minimize"           # minimize or maximize
      metric: "val_loss"              # Metric to optimize
      study_name: "milia_hpo"         # Study name for persistence
      storage: null                   # null = in-memory
                                      # "sqlite:///hpo_results.db" for persistence
      load_if_exists: true            # Resume existing study
    
    # =========================================================================
    # CROSS-VALIDATION
    # =========================================================================
    # Optional k-fold CV for more robust evaluation
    
    cv_folds: 0                       # 0 = no CV, >0 = k-fold CV
    cv_metric_aggregation: "mean"     # mean, median, min, max
```

### 7.2 config_bridge.py Modifications

Add the following after line 649 in `config_bridge.py`:

```python
# =============================================================================
# HPO CONFIGURATION CLASSES
# =============================================================================
# Note: Full implementation in milia_pipeline/models/hpo/hpo_config.py
# These are bridge classes for integration with config system

from enum import Enum
from typing import Dict, Any, Optional, List


class HPOParamType(Enum):
    """HPO parameter types for config bridge."""
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOGUNIFORM = "loguniform"


class HPOPrunerType(Enum):
    """HPO pruner types for config bridge."""
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    PERCENTILE = "percentile"
    NONE = "none"


class HPOSamplerType(Enum):
    """HPO sampler types for config bridge."""
    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"
    GRID = "grid"


class HPODirection(Enum):
    """HPO optimization direction."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class HPOSearchSpaceParamBridge:
    """Bridge class for HPO search space parameter."""
    type: HPOParamType
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[int] = None
    choices: Optional[List[Any]] = None
    log: bool = False


@dataclass(frozen=True)
class HPOPrunerConfigBridge:
    """Bridge class for HPO pruner configuration."""
    type: HPOPrunerType = HPOPrunerType.MEDIAN
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    interval_steps: int = 1
    percentile: float = 25.0


@dataclass(frozen=True)
class HPOSamplerConfigBridge:
    """Bridge class for HPO sampler configuration."""
    type: HPOSamplerType = HPOSamplerType.TPE
    n_startup_trials: int = 10
    seed: Optional[int] = None
    multivariate: bool = True
    constant_liar: bool = False


@dataclass(frozen=True)
class HPOStudyConfigBridge:
    """Bridge class for HPO study configuration."""
    direction: HPODirection = HPODirection.MINIMIZE
    metric: str = "val_loss"
    study_name: str = "milia_hpo"
    storage: Optional[str] = None
    load_if_exists: bool = True


@dataclass(frozen=True)
class HPOConfigBridge:
    """
    Bridge class for HPO configuration.
    
    Integrates HPO settings with the config bridge system.
    Full implementation in milia_pipeline/models/hpo/hpo_config.py
    """
    enabled: bool = False
    backend: str = "optuna"
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    search_space: Dict[str, Dict[str, HPOSearchSpaceParamBridge]] = field(default_factory=dict)
    pruner: HPOPrunerConfigBridge = field(default_factory=HPOPrunerConfigBridge)
    sampler: HPOSamplerConfigBridge = field(default_factory=HPOSamplerConfigBridge)
    study: HPOStudyConfigBridge = field(default_factory=HPOStudyConfigBridge)
    cv_folds: int = 0
    cv_metric_aggregation: str = "mean"
    
    def __post_init__(self):
        """Validate HPO configuration."""
        if self.backend not in ("optuna", "ray_tune"):
            raise ConfigurationError(
                f"Unknown HPO backend: '{self.backend}'",
                config_key="hpo.backend",
                expected_value="'optuna' or 'ray_tune'",
                actual_value=self.backend
            )
```

### 7.3 trainer.py Modifications

**Minimal modification** - Add HPO callback support after line 117:

```python
# In Trainer.__init__ (after line 117):

def __init__(
    self,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    loss_fn: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    callbacks: Optional[List['Callback']] = None,
    logger: Optional[logging.Logger] = None,
    # NEW: HPO-specific parameters
    hpo_callback: Optional['Callback'] = None,
):
    # ... existing initialization ...
    
    # NEW: Store HPO callback reference (set by HPOManager)
    self.hpo_callback = hpo_callback
    
    # If HPO callback provided, add to callbacks list
    if self.hpo_callback is not None:
        if self.callbacks is None:
            self.callbacks = []
        self.callbacks.append(self.hpo_callback)
```

**Note**: This is the ONLY required modification to trainer.py. The existing callback system handles all HPO integration automatically.

### 7.4 main.py Integration

**DEFERRED** to models/ module full integration phase.

The HPO module will be self-contained and testable without main.py integration. Integration with main.py will occur alongside the full models/ module integration after HPO implementation and testing is complete.

**Usage until then:** Direct programmatic usage via `HPOManager` (see Part 9.1.2).

### 7.5 models/__init__.py Export Addition

Add to `milia_pipeline/models/__init__.py`:

```python
# HPO exports (conditional import for optional dependency)
try:
    from .hpo import (
        HPOManager,
        HPOConfig,
        is_hpo_enabled,
        get_best_params,
        create_hpo_manager,
        OptunaPruningCallback,
    )
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False
    HPOManager = None
    HPOConfig = None
    is_hpo_enabled = None
    get_best_params = None
    create_hpo_manager = None
    OptunaPruningCallback = None

# Add to __all__
__all__ = [
    # ... existing exports ...
    
    # HPO
    'HPOManager',
    'HPOConfig',
    'is_hpo_enabled',
    'get_best_params',
    'create_hpo_manager',
    'OptunaPruningCallback',
    'HPO_AVAILABLE',
]
```

### 7.6 exceptions.py Addition

Add HPO exceptions after line 2810 in `exceptions.py` (see Part 4.1 for complete implementation).

**Summary of classes to add:**
- `HPOError(ModelError)` - Base HPO exception
- `HPOConfigurationError(HPOError)` - Config validation failures
- `TrialFailedError(HPOError)` - Trial execution failures
- `StudyNotFoundError(HPOError)` - Study lookup failures
- `BackendError(HPOError)` - Backend-specific errors
- `SearchSpaceError(HPOError)` - Search space definition errors
- `PruningError(HPOError)` - Pruning-related errors


---

## Part 8: Implementation Steps and Testing Plan

### 8.1 Implementation Order

The implementation should proceed in the following order to ensure proper dependency resolution:

#### Phase 1: Foundation (Files without dependencies on other new files)

| Step | File | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 1.1 | Update `exceptions.py` (add HPO exceptions) | 30 min | None (add after line 2810) |
| 1.2 | `hpo/hpo_config.py` | 45 min | `config_bridge.py` (existing) |
| 1.3 | `hpo/search_spaces/param_types.py` | 20 min | None |

#### Phase 2: Backend Infrastructure

| Step | File | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 2.1 | `hpo/backends/base.py` | 30 min | `exceptions.py` |
| 2.2 | `hpo/backends/optuna_backend.py` | 60 min | `base.py`, `exceptions.py` |
| 2.3 | `hpo/backends/ray_tune_backend.py` | 60 min | `base.py` (complete, inactive) |
| 2.4 | `hpo/backends/__init__.py` | 10 min | All backends |

#### Phase 3: Callbacks and Search Spaces

| Step | File | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 3.1 | `hpo/callbacks/optuna_callback.py` | 45 min | `callbacks.py`, `exceptions.py` |
| 3.2 | `hpo/callbacks/ray_tune_callback.py` | 20 min | Complete, inactive (imports from ray_tune_backend) |
| 3.3 | `hpo/callbacks/__init__.py` | 10 min | All callbacks |
| 3.4 | `hpo/search_spaces/search_space_builder.py` | 40 min | `param_types.py`, `hpo_config.py` |
| 3.5 | `hpo/search_spaces/__init__.py` | 10 min | All search space files |

#### Phase 4: Manager, Transfer, NAS, and Analysis

| Step | File | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 4.1 | `hpo/hpo_manager.py` | 90 min | All previous files, `trainer.py`, `model_factory.py` |
| 4.2 | `hpo/transfer/meta_features.py` | 30 min | None |
| 4.3 | `hpo/transfer/warm_start.py` | 20 min | None |
| 4.4 | `hpo/transfer/transfer_manager.py` | 45 min | `meta_features.py`, `warm_start.py` |
| 4.5 | `hpo/transfer/__init__.py` | 10 min | All transfer files |
| 4.6 | `hpo/nas/search_space.py` | 40 min | `hpo_config.py` |
| 4.7 | `hpo/nas/nas_manager.py` | 50 min | `search_space.py`, `hpo_manager.py` |
| 4.8 | `hpo/nas/__init__.py` | 10 min | All NAS files |
| 4.9 | `hpo/analysis/study_analyzer.py` | 45 min | `hpo_manager.py`, `optuna_backend.py` |
| 4.10 | `hpo/analysis/__init__.py` | 10 min | `study_analyzer.py` |

#### Phase 5: Module Initialization and Integration

| Step | File | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 5.1 | `hpo/__init__.py` | 20 min | All HPO files |
| 5.2 | Update `config.yaml` | 15 min | None |
| 5.3 | Update `config_bridge.py` | 20 min | `hpo_config.py` |
| 5.4 | Update `models/__init__.py` | 10 min | `hpo/__init__.py` |

#### Phase 6: Testing

| Step | File | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 6.1 | `tests/models/hpo/test_hpo_config.py` | 30 min | `hpo_config.py` |
| 6.2 | `tests/models/hpo/test_optuna_backend.py` | 45 min | `optuna_backend.py` |
| 6.3 | `tests/models/hpo/test_optuna_callback.py` | 30 min | `optuna_callback.py` |
| 6.4 | `tests/models/hpo/test_hpo_manager.py` | 60 min | `hpo_manager.py` |
| 6.5 | `tests/models/hpo/test_integration.py` | 60 min | All HPO files |

**Total Estimated Time: ~16 hours**

### 8.2 Testing Strategy

#### 8.2.1 Unit Tests

```python
# tests/models/hpo/test_hpo_config.py

import pytest
from milia_pipeline.models.hpo import (
    HPOConfig,
    SearchSpaceParamConfig,
    PrunerConfig,
    SamplerConfig,
    ParamType,
    PrunerType,
    SamplerType,
)
from milia_pipeline.exceptions import ConfigurationError


class TestSearchSpaceParamConfig:
    """Test SearchSpaceParamConfig validation."""
    
    def test_valid_int_param(self):
        """Test valid integer parameter."""
        config = SearchSpaceParamConfig(
            type=ParamType.INT,
            low=1,
            high=10,
            step=1
        )
        assert config.type == ParamType.INT
        assert config.low == 1
        assert config.high == 10
    
    def test_invalid_int_missing_bounds(self):
        """Test that int type requires low and high."""
        with pytest.raises(ConfigurationError):
            SearchSpaceParamConfig(
                type=ParamType.INT,
                low=1,
                # high is missing
            )
    
    def test_invalid_bounds_order(self):
        """Test that low must be less than high."""
        with pytest.raises(ConfigurationError):
            SearchSpaceParamConfig(
                type=ParamType.INT,
                low=10,
                high=5,
            )
    
    def test_valid_categorical_param(self):
        """Test valid categorical parameter."""
        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL,
            choices=['relu', 'gelu', 'elu']
        )
        assert config.choices == ['relu', 'gelu', 'elu']
    
    def test_invalid_categorical_empty_choices(self):
        """Test that categorical requires non-empty choices."""
        with pytest.raises(ConfigurationError):
            SearchSpaceParamConfig(
                type=ParamType.CATEGORICAL,
                choices=[]
            )


class TestHPOConfig:
    """Test HPOConfig validation."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = HPOConfig()
        assert config.enabled == False
        assert config.backend == "optuna"
        assert config.n_trials == 100
    
    def test_invalid_backend(self):
        """Test invalid backend raises error."""
        with pytest.raises(ConfigurationError):
            HPOConfig(backend="unknown_backend")
    
    def test_invalid_n_trials(self):
        """Test n_trials must be positive."""
        with pytest.raises(ConfigurationError):
            HPOConfig(n_trials=0)
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'enabled': True,
            'n_trials': 50,
            'search_space': {
                'hyperparameters': {
                    'hidden_channels': {
                        'type': 'int',
                        'low': 32,
                        'high': 128,
                    }
                }
            }
        }
        config = HPOConfig.from_dict(config_dict)
        assert config.enabled == True
        assert config.n_trials == 50
        assert 'hyperparameters' in config.search_space
```

```python
# tests/models/hpo/test_optuna_backend.py

import pytest
from unittest.mock import Mock, patch

# Skip tests if optuna not installed
optuna = pytest.importorskip("optuna")

from milia_pipeline.models.hpo.backends import OptunaBackend
from milia_pipeline.exceptions import BackendError


class TestOptunaBackend:
    """Test OptunaBackend implementation."""
    
    def test_initialization(self):
        """Test backend initializes correctly."""
        backend = OptunaBackend()
        assert backend is not None
    
    def test_create_study_in_memory(self):
        """Test creating in-memory study."""
        backend = OptunaBackend()
        study = backend.create_study(
            study_name="test_study",
            direction="minimize",
            storage=None,
        )
        assert study.study_name == "test_study"
        assert study.direction.name == "MINIMIZE"
    
    def test_create_pruner_median(self):
        """Test creating median pruner."""
        backend = OptunaBackend()
        pruner = backend.create_pruner(
            pruner_type="median",
            n_startup_trials=5,
            n_warmup_steps=10,
        )
        assert isinstance(pruner, optuna.pruners.MedianPruner)
    
    def test_create_pruner_invalid_type(self):
        """Test invalid pruner type raises error."""
        backend = OptunaBackend()
        with pytest.raises(BackendError):
            backend.create_pruner(pruner_type="invalid_pruner")
    
    def test_create_sampler_tpe(self):
        """Test creating TPE sampler."""
        backend = OptunaBackend()
        sampler = backend.create_sampler(
            sampler_type="tpe",
            seed=42,
        )
        assert isinstance(sampler, optuna.samplers.TPESampler)
    
    def test_suggest_params(self):
        """Test parameter suggestion."""
        backend = OptunaBackend()
        study = backend.create_study("test", "minimize")
        
        search_space = {
            'hyperparameters': {
                'hidden': {'type': 'int', 'low': 32, 'high': 128},
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
            }
        }
        
        def objective(trial):
            params = backend.suggest_params(trial, search_space)
            assert 'hyperparameters.hidden' in params
            assert 'hyperparameters.dropout' in params
            assert 32 <= params['hyperparameters.hidden'] <= 128
            assert 0.0 <= params['hyperparameters.dropout'] <= 0.5
            return 0.5
        
        study.optimize(objective, n_trials=1)
```

```python
# tests/models/hpo/test_optuna_callback.py

import pytest
from unittest.mock import Mock, MagicMock

optuna = pytest.importorskip("optuna")

from milia_pipeline.models.hpo.callbacks import OptunaPruningCallback


class TestOptunaPruningCallback:
    """Test OptunaPruningCallback implementation."""
    
    def test_initialization(self):
        """Test callback initializes correctly."""
        trial = Mock()
        callback = OptunaPruningCallback(
            trial=trial,
            monitor="val_loss",
        )
        assert callback.monitor == "val_loss"
        assert callback.trial == trial
    
    def test_on_epoch_end_reports_metric(self):
        """Test that on_epoch_end reports metric to trial."""
        trial = Mock()
        trial.should_prune.return_value = False
        
        callback = OptunaPruningCallback(trial=trial, monitor="val_loss")
        
        metrics = {"val_loss": 0.5, "train_loss": 0.3}
        callback.on_epoch_end(trainer=Mock(), epoch=5, metrics=metrics)
        
        trial.report.assert_called_once_with(0.5, 5)
    
    def test_on_epoch_end_raises_pruned(self):
        """Test that pruning raises TrialPruned."""
        trial = Mock()
        trial.should_prune.return_value = True
        
        callback = OptunaPruningCallback(trial=trial, monitor="val_loss")
        
        metrics = {"val_loss": 1.5}
        
        with pytest.raises(optuna.TrialPruned):
            callback.on_epoch_end(trainer=Mock(), epoch=5, metrics=metrics)
    
    def test_on_epoch_end_alternative_metric_names(self):
        """Test that callback finds alternative metric names."""
        trial = Mock()
        trial.should_prune.return_value = False
        
        callback = OptunaPruningCallback(trial=trial, monitor="loss")
        
        # Metric is named "val_loss" but we're monitoring "loss"
        metrics = {"val_loss": 0.5}
        callback.on_epoch_end(trainer=Mock(), epoch=5, metrics=metrics)
        
        # Should find val_loss as alternative
        trial.report.assert_called_once_with(0.5, 5)
```

#### 8.2.2 Integration Tests

```python
# tests/models/hpo/test_integration.py

import pytest
import tempfile
import os

optuna = pytest.importorskip("optuna")
torch = pytest.importorskip("torch")

from milia_pipeline.models.hpo import (
    HPOManager,
    HPOConfig,
    is_hpo_enabled,
)


class TestHPOIntegration:
    """Integration tests for HPO module."""
    
    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset for testing."""
        from torch_geometric.data import Data, InMemoryDataset
        
        class SimpleDataset(InMemoryDataset):
            def __init__(self):
                super().__init__('.')
                data_list = []
                for i in range(100):
                    x = torch.randn(10, 16)
                    edge_index = torch.randint(0, 10, (2, 20))
                    y = torch.randn(1)
                    data_list.append(Data(x=x, edge_index=edge_index, y=y))
                self.data, self.slices = self.collate(data_list)
        
        return SimpleDataset()
    
    @pytest.fixture
    def hpo_config(self):
        """Create HPO configuration for testing."""
        return HPOConfig(
            enabled=True,
            backend="optuna",
            n_trials=5,  # Small for testing
            search_space={
                'hyperparameters': {
                    'hidden_channels': {
                        'type': 'int',
                        'low': 16,
                        'high': 64,
                    }
                }
            },
            pruner={'type': 'median', 'n_startup_trials': 2},
            sampler={'type': 'tpe', 'seed': 42},
            study={'direction': 'minimize', 'metric': 'val_loss'},
        )
    
    def test_hpo_manager_creation(self, hpo_config):
        """Test HPOManager creation from config."""
        manager = HPOManager(hpo_config)
        assert manager.config.enabled == True
        assert manager.backend is not None
    
    def test_is_hpo_enabled(self, hpo_config):
        """Test is_hpo_enabled function."""
        assert is_hpo_enabled(hpo_config) == True
        
        disabled_config = HPOConfig(enabled=False)
        assert is_hpo_enabled(disabled_config) == False
    
    @pytest.mark.slow
    def test_full_optimization_run(self, simple_dataset, hpo_config):
        """Test complete HPO optimization run."""
        # This test requires full model training infrastructure
        # Mark as slow and skip if infrastructure not available
        
        try:
            from milia_pipeline.models import get_factory
            factory = get_factory()
        except ImportError:
            pytest.skip("Model factory not available")
        
        manager = HPOManager(hpo_config)
        
        best_params = manager.optimize(
            model_name="GCN",
            dataset=simple_dataset,
            base_hyperparameters={"num_layers": 2},
        )
        
        assert 'hidden_channels' in best_params
        assert 16 <= best_params['hidden_channels'] <= 64
        assert manager.study is not None
        assert len(manager.get_all_trials()) == 5
    
    def test_study_persistence(self, hpo_config):
        """Test study persistence to SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_study.db")
            
            # Modify config to use SQLite storage
            config_dict = {
                'enabled': True,
                'n_trials': 3,
                'study': {
                    'storage': f"sqlite:///{db_path}",
                    'study_name': "test_persistence",
                },
            }
            config = HPOConfig.from_dict(config_dict)
            
            # Create manager and verify study can be created
            manager = HPOManager(config)
            
            # Verify database file would be created
            # (actual persistence tested with full optimization)
            assert config.study.storage is not None
```

### 8.3 Validation Checklist

Before considering implementation complete, verify:

#### Configuration Validation
- [ ] HPOConfig validates all fields on creation
- [ ] Invalid backend raises ConfigurationError
- [ ] Invalid n_trials (< 1) raises ConfigurationError
- [ ] Invalid search space types raise ConfigurationError
- [ ] from_dict() correctly parses all nested configs

#### Backend Validation
- [ ] OptunaBackend creates studies successfully
- [ ] Pruner creation works for all supported types
- [ ] Sampler creation works for all supported types
- [ ] suggest_params() returns correct parameter types
- [ ] get_best_params() returns best trial parameters
- [ ] get_all_trials() returns complete trial information

#### Callback Validation
- [ ] OptunaPruningCallback integrates with Trainer callback system
- [ ] Metric reporting works correctly
- [ ] Pruning raises optuna.TrialPruned
- [ ] Alternative metric names are found

#### Manager Validation
- [ ] HPOManager initializes with disabled config (warning only)
- [ ] optimize() runs correct number of trials
- [ ] Cross-validation works when cv_folds > 0
- [ ] Best parameters are accessible after optimization
- [ ] Study statistics are calculated correctly

#### Integration Validation
- [ ] config.yaml HPO section parses correctly
- [ ] models/__init__.py exports work
- [ ] Exception hierarchy is correct
- [ ] Trainer integration is seamless


---

## Part 9: Usage Examples and Future Enhancements

### 9.1 Usage Examples

#### 9.1.1 Basic Usage - Enable HPO in config.yaml

```yaml
# config.yaml
models:
  selection:
    name: "GCN"
    hyperparameters:
      num_layers: 3           # Fixed, not optimized
      aggregation: "mean"     # Fixed, not optimized
  
  hpo:
    enabled: true             # MASTER SWITCH ON
    n_trials: 100
    
    search_space:
      hyperparameters:
        hidden_channels:
          type: "int"
          low: 32
          high: 256
          step: 32
        dropout:
          type: "float"
          low: 0.0
          high: 0.5
      
      optimizer:
        lr:
          type: "loguniform"
          low: 1.0e-5
          high: 1.0e-2
    
    study:
      direction: "minimize"
      metric: "val_loss"
```

#### 9.1.2 Programmatic Usage

```python
from milia_pipeline.models.hpo import (
    HPOManager,
    HPOConfig,
    SearchSpaceParamConfig,
    ParamType,
)
from milia_pipeline.datasets import MiliaDataset

# Load dataset
dataset = MiliaDataset(root='./data', dataset_type='DFT')

# Define search space programmatically
search_space = {
    'hyperparameters': {
        'hidden_channels': SearchSpaceParamConfig(
            type=ParamType.INT,
            low=32,
            high=256,
            step=32,
        ),
        'num_layers': SearchSpaceParamConfig(
            type=ParamType.INT,
            low=2,
            high=6,
        ),
        'dropout': SearchSpaceParamConfig(
            type=ParamType.FLOAT,
            low=0.0,
            high=0.5,
        ),
    },
    'optimizer': {
        'lr': SearchSpaceParamConfig(
            type=ParamType.LOGUNIFORM,
            low=1e-5,
            high=1e-2,
        ),
    },
}

# Create HPO configuration
config = HPOConfig(
    enabled=True,
    backend="optuna",
    n_trials=100,
    search_space=search_space,
    study={'direction': 'minimize', 'metric': 'val_mae'},
)

# Create manager and run optimization
manager = HPOManager(config)
best_params = manager.optimize(
    model_name="GCN",
    dataset=dataset,
    base_hyperparameters={'aggregation': 'mean'},
    trainer_kwargs={'epochs': 100, 'batch_size': 32},
)

# Print results
print(f"Best parameters: {best_params}")
print(f"Best MAE: {manager.get_best_value():.4f}")

# Get detailed statistics
stats = manager.get_study_statistics()
print(f"Completed trials: {stats['n_completed']}")
print(f"Pruned trials: {stats['n_pruned']}")
print(f"Pruning rate: {stats['pruning_rate']:.1%}")
```

#### 9.1.3 Using Cross-Validation

```python
config = HPOConfig(
    enabled=True,
    n_trials=50,
    cv_folds=5,                    # 5-fold cross-validation
    cv_metric_aggregation="mean",  # Average across folds
    search_space={...},
)

manager = HPOManager(config)
best_params = manager.optimize(
    model_name="GAT",
    dataset=dataset,
)

# Each trial runs 5-fold CV internally
# More robust but 5x slower per trial
```

#### 9.1.4 Resuming a Study

```python
# First run - save to SQLite
config = HPOConfig(
    enabled=True,
    n_trials=50,
    study={
        'study_name': 'my_experiment',
        'storage': 'sqlite:///hpo_results.db',
        'load_if_exists': True,
    },
)

manager = HPOManager(config)
manager.optimize(model_name="GCN", dataset=dataset)

# Later - resume the study
config_resume = HPOConfig(
    enabled=True,
    n_trials=50,  # 50 more trials
    study={
        'study_name': 'my_experiment',
        'storage': 'sqlite:///hpo_results.db',
        'load_if_exists': True,  # Resume existing
    },
)

manager_resume = HPOManager(config_resume)
manager_resume.optimize(model_name="GCN", dataset=dataset)
# Now has 100 total trials
```

#### 9.1.5 Multi-Objective Optimization

```python
from milia_pipeline.models.hpo import (
    HPOManager,
    HPOConfig,
    MultiObjectiveStudyConfig,
)

# Multi-objective configuration
config = HPOConfig(
    enabled=True,
    n_trials=100,
    search_space={...},
    study=MultiObjectiveStudyConfig(
        directions=['minimize', 'minimize'],  # Pareto optimization
        metrics=['val_mae', 'inference_time'],
        reference_point=[1.0, 100.0],  # For hypervolume calculation
    ),
    sampler={'type': 'nsgaii'},  # NSGA-II for multi-objective
)

manager = HPOManager(config)
best_trials = manager.optimize(
    model_name="GCN",
    dataset=dataset,
)

# Get Pareto front
pareto_front = manager.get_pareto_front()
for trial in pareto_front:
    print(f"MAE: {trial['values'][0]:.4f}, Time: {trial['values'][1]:.2f}ms")

# Select best trade-off
selected = manager.select_by_preference(
    weights=[0.7, 0.3]  # 70% accuracy, 30% speed
)
```

### 9.2 Common Patterns

#### 9.2.1 Architecture-Specific Search Spaces

```python
# For GAT models
gat_search_space = {
    'hyperparameters': {
        'hidden_channels': {'type': 'int', 'low': 32, 'high': 256},
        'num_layers': {'type': 'int', 'low': 2, 'high': 5},
        'heads': {'type': 'int', 'low': 1, 'high': 8},  # GAT-specific
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.6},
        'attention_dropout': {'type': 'float', 'low': 0.0, 'high': 0.6},
    },
}

# For SchNet (quantum chemistry)
schnet_search_space = {
    'hyperparameters': {
        'hidden_channels': {'type': 'int', 'low': 64, 'high': 256},
        'num_filters': {'type': 'int', 'low': 64, 'high': 256},
        'num_interactions': {'type': 'int', 'low': 3, 'high': 6},
        'num_gaussians': {'type': 'int', 'low': 25, 'high': 100},
        'cutoff': {'type': 'float', 'low': 5.0, 'high': 10.0},
    },
}
```

#### 9.2.2 Scheduler Hyperparameter Search

```python
search_space = {
    'scheduler': {
        'type': {'type': 'categorical', 'choices': ['cosine', 'step', 'plateau']},
        'factor': {'type': 'float', 'low': 0.1, 'high': 0.9},
        'patience': {'type': 'int', 'low': 5, 'high': 20},
    },
}
```

#### 9.2.3 Loss Function Search

```python
search_space = {
    'loss': {
        'type': {'type': 'categorical', 'choices': ['mse', 'mae', 'huber']},
    },
}

# For focal loss (classification)
focal_search_space = {
    'loss': {
        'alpha': {'type': 'float', 'low': 0.1, 'high': 0.9},
        'gamma': {'type': 'float', 'low': 0.5, 'high': 5.0},
    },
}
```

### 9.3 Advanced Features (Core Implementation)

#### 9.3.1 Multi-Objective Optimization (`hpo_config.py`, `hpo_manager.py`)

```python
# Location: milia_pipeline/models/hpo/hpo_config.py

@dataclass(frozen=True)
class MultiObjectiveStudyConfig:
    """
    Configuration for multi-objective optimization.
    
    Supports Pareto optimization for competing objectives
    (e.g., accuracy vs speed, MAE vs model size).
    """
    directions: Tuple[str, ...] = ("minimize",)  # Per-objective direction
    metrics: Tuple[str, ...] = ("val_loss",)     # Metric names
    reference_point: Optional[Tuple[float, ...]] = None  # For hypervolume
    
    def __post_init__(self):
        if len(self.directions) != len(self.metrics):
            raise ConfigurationError(
                "directions and metrics must have same length",
                config_key="study.directions",
                expected_value=f"length {len(self.metrics)}",
                actual_value=f"length {len(self.directions)}"
            )
        for d in self.directions:
            if d not in ("minimize", "maximize"):
                raise ConfigurationError(
                    f"Invalid direction: {d}",
                    config_key="study.directions",
                    expected_value="'minimize' or 'maximize'",
                    actual_value=d
                )
    
    @property
    def is_multi_objective(self) -> bool:
        return len(self.metrics) > 1


# Location: milia_pipeline/models/hpo/hpo_manager.py

class HPOManager:
    # ... existing methods ...
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """
        Get Pareto-optimal trials from multi-objective study.
        
        Returns:
            List of trial dicts on the Pareto front
        """
        if self.study is None:
            raise HPOError("No study available. Run optimize() first.")
        
        if not self.config.study.is_multi_objective:
            raise HPOError("get_pareto_front() requires multi-objective study")
        
        # Optuna provides best_trials for multi-objective
        return [
            {
                'number': t.number,
                'values': t.values,
                'params': t.params,
            }
            for t in self.study.best_trials
        ]
    
    def select_by_preference(
        self,
        weights: List[float],
    ) -> Dict[str, Any]:
        """
        Select trial from Pareto front by weighted preference.
        
        Args:
            weights: Importance weights per objective (sum to 1)
            
        Returns:
            Selected trial dict
        """
        pareto = self.get_pareto_front()
        
        # Normalize and compute weighted scores
        best_score = float('inf')
        best_trial = None
        
        for trial in pareto:
            score = sum(w * v for w, v in zip(weights, trial['values']))
            if score < best_score:
                best_score = score
                best_trial = trial
        
        return best_trial
    
    def get_hypervolume(self) -> float:
        """
        Calculate hypervolume indicator for multi-objective study.
        
        Requires reference_point in config.
        """
        if self.config.study.reference_point is None:
            raise HPOError("Hypervolume requires reference_point in config")
        
        from optuna.multi_objective._hypervolume import WFG
        
        pareto = self.get_pareto_front()
        points = [t['values'] for t in pareto]
        
        wfg = WFG()
        return wfg.compute(points, self.config.study.reference_point)
```

#### 9.3.2 Hyperparameter Transfer Learning (`hpo/transfer/`)

```python
# Location: milia_pipeline/models/hpo/transfer/__init__.py

from .transfer_manager import HPOTransferManager
from .meta_features import MetaFeatureExtractor
from .warm_start import WarmStartStrategy

__all__ = ['HPOTransferManager', 'MetaFeatureExtractor', 'WarmStartStrategy']
```

```python
# Location: milia_pipeline/models/hpo/transfer/transfer_manager.py

"""
Hyperparameter Transfer Learning

Transfers HPO knowledge between related tasks/datasets using meta-learning.
High research value for few-shot optimization and domain adaptation.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

from milia_pipeline.exceptions import HPOError


@dataclass(frozen=True)
class TransferConfig:
    """Configuration for HPO transfer learning."""
    n_warm_start_trials: int = 10          # Trials to transfer
    similarity_threshold: float = 0.7       # Min dataset similarity
    meta_feature_method: str = "statistical" # statistical, learned, landmark
    adaptation_method: str = "weighted"     # weighted, filtered, full


class HPOTransferManager:
    """
    Manages hyperparameter transfer between studies.
    
    Use cases:
    - Transfer from small dataset to large dataset
    - Transfer between related molecular properties
    - Transfer from DFT to experimental data
    - Few-shot optimization on new tasks
    
    Research basis:
    - Meta-learning for hyperparameter optimization
    - Dataset similarity measures
    - Warm-starting Bayesian optimization
    """
    
    def __init__(self, config: TransferConfig):
        self.config = config
        self._meta_db: Dict[str, Dict] = {}  # study_name -> meta_info
    
    def register_study(
        self,
        study_name: str,
        study,
        dataset,
        meta_features: Optional[Dict[str, float]] = None,
    ):
        """
        Register completed study for future transfer.
        
        Args:
            study_name: Unique identifier
            study: Completed Optuna study
            dataset: Dataset used for study
            meta_features: Pre-computed meta-features (optional)
        """
        if meta_features is None:
            meta_features = MetaFeatureExtractor.extract(dataset)
        
        self._meta_db[study_name] = {
            'study': study,
            'meta_features': meta_features,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
        }
    
    def find_similar_studies(
        self,
        target_dataset,
        top_k: int = 3,
    ) -> List[str]:
        """
        Find most similar registered studies to target dataset.
        
        Args:
            target_dataset: New dataset to optimize for
            top_k: Number of similar studies to return
            
        Returns:
            List of study names, most similar first
        """
        target_features = MetaFeatureExtractor.extract(target_dataset)
        
        similarities = []
        for name, info in self._meta_db.items():
            sim = self._compute_similarity(
                target_features,
                info['meta_features']
            )
            if sim >= self.config.similarity_threshold:
                similarities.append((name, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similarities[:top_k]]
    
    def warm_start_study(
        self,
        target_study,
        source_studies: List[str],
    ) -> int:
        """
        Warm-start target study with trials from source studies.
        
        Args:
            target_study: New Optuna study to warm-start
            source_studies: List of source study names
            
        Returns:
            Number of trials transferred
        """
        n_transferred = 0
        trials_per_source = self.config.n_warm_start_trials // len(source_studies)
        
        for source_name in source_studies:
            source_info = self._meta_db.get(source_name)
            if source_info is None:
                continue
            
            source_study = source_info['study']
            
            # Get best trials from source
            sorted_trials = sorted(
                source_study.trials,
                key=lambda t: t.value if t.value else float('inf')
            )[:trials_per_source]
            
            # Add to target study
            for trial in sorted_trials:
                try:
                    target_study.enqueue_trial(trial.params)
                    n_transferred += 1
                except Exception:
                    pass  # Skip incompatible trials
        
        return n_transferred
    
    def _compute_similarity(
        self,
        features_a: Dict[str, float],
        features_b: Dict[str, float],
    ) -> float:
        """Compute cosine similarity between meta-feature vectors."""
        common_keys = set(features_a.keys()) & set(features_b.keys())
        if not common_keys:
            return 0.0
        
        vec_a = np.array([features_a[k] for k in common_keys])
        vec_b = np.array([features_b[k] for k in common_keys])
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class MetaFeatureExtractor:
    """
    Extracts meta-features from datasets for similarity computation.
    
    Meta-features capture dataset characteristics without training:
    - Statistical: size, dimensionality, class balance
    - Graph-specific: density, degree distribution, clustering
    - Molecular: atom types, bond types, molecular weight distribution
    """
    
    @staticmethod
    def extract(dataset) -> Dict[str, float]:
        """Extract meta-features from PyG dataset."""
        features = {}
        
        # Size features
        features['n_samples'] = len(dataset)
        features['n_features'] = dataset[0].x.shape[1] if hasattr(dataset[0], 'x') else 0
        
        # Graph features
        if hasattr(dataset[0], 'edge_index'):
            edge_counts = [d.edge_index.shape[1] for d in dataset]
            node_counts = [d.x.shape[0] for d in dataset]
            
            features['mean_nodes'] = np.mean(node_counts)
            features['mean_edges'] = np.mean(edge_counts)
            features['mean_density'] = np.mean([
                e / (n * (n - 1)) if n > 1 else 0
                for e, n in zip(edge_counts, node_counts)
            ])
        
        # Target features
        if hasattr(dataset[0], 'y'):
            targets = [d.y.item() if d.y.numel() == 1 else d.y.mean().item() 
                      for d in dataset]
            features['target_mean'] = np.mean(targets)
            features['target_std'] = np.std(targets)
            features['target_range'] = np.max(targets) - np.min(targets)
        
        return features


class WarmStartStrategy:
    """Strategies for warm-starting optimization."""
    
    @staticmethod
    def weighted_transfer(
        source_trials: List,
        similarities: List[float],
    ) -> List:
        """Weight trials by source study similarity."""
        pass  # Implementation
    
    @staticmethod
    def filtered_transfer(
        source_trials: List,
        target_search_space: Dict,
    ) -> List:
        """Filter trials compatible with target search space."""
        pass  # Implementation
```

#### 9.3.3 Neural Architecture Search (`hpo/nas/`)

```python
# Location: milia_pipeline/models/hpo/nas/__init__.py

from .search_space import GNNArchitectureSpace
from .nas_manager import NASManager

__all__ = ['GNNArchitectureSpace', 'NASManager']
```

```python
# Location: milia_pipeline/models/hpo/nas/search_space.py

"""
GNN Architecture Search Space

Defines searchable architecture components for graph neural networks.
Enables automated discovery of optimal GNN architectures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class LayerType(Enum):
    """Available GNN layer types."""
    GCN = "gcn"
    GAT = "gat"
    SAGE = "sage"
    GIN = "gin"
    GATV2 = "gatv2"
    TRANSFORMER = "transformer"
    PNA = "pna"


class PoolingType(Enum):
    """Available graph pooling types."""
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    ATTENTION = "attention"
    SET2SET = "set2set"
    TOPK = "topk"


class AggregationType(Enum):
    """Available aggregation types."""
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    LSTM = "lstm"
    MULTI = "multi"  # PNA-style multi-aggregator


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for a single GNN layer."""
    type: LayerType
    hidden_channels: int
    heads: int = 1  # For attention layers
    dropout: float = 0.0
    activation: str = "relu"
    batch_norm: bool = True
    residual: bool = False


@dataclass
class GNNArchitectureSpace:
    """
    Defines the search space for GNN architectures.
    
    Searchable components:
    - Number of layers (depth)
    - Layer types (can vary per layer)
    - Hidden dimensions
    - Attention heads
    - Skip connections
    - Pooling strategy
    - Aggregation functions
    """
    
    # Layer search space
    min_layers: int = 2
    max_layers: int = 8
    layer_types: List[LayerType] = field(
        default_factory=lambda: [LayerType.GCN, LayerType.GAT, LayerType.SAGE]
    )
    
    # Dimension search space
    hidden_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    
    # Attention search space (for GAT, Transformer)
    heads: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    # Regularization search space
    dropout_range: tuple = (0.0, 0.6)
    
    # Architecture options
    allow_skip_connections: bool = True
    allow_dense_connections: bool = False  # DenseGCN style
    allow_mixed_layers: bool = True  # Different layer types
    
    # Pooling search space
    pooling_types: List[PoolingType] = field(
        default_factory=lambda: [PoolingType.MEAN, PoolingType.ATTENTION]
    )
    
    # Aggregation search space
    aggregation_types: List[AggregationType] = field(
        default_factory=lambda: [AggregationType.MEAN, AggregationType.SUM]
    )
    
    def to_optuna_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Convert to Optuna search space format."""
        space = {
            'architecture': {
                'num_layers': {
                    'type': 'int',
                    'low': self.min_layers,
                    'high': self.max_layers,
                },
                'hidden_channels': {
                    'type': 'categorical',
                    'choices': self.hidden_channels,
                },
                'pooling': {
                    'type': 'categorical',
                    'choices': [p.value for p in self.pooling_types],
                },
                'dropout': {
                    'type': 'float',
                    'low': self.dropout_range[0],
                    'high': self.dropout_range[1],
                },
            }
        }
        
        # Add per-layer choices if mixed layers allowed
        if self.allow_mixed_layers:
            for i in range(self.max_layers):
                space['architecture'][f'layer_{i}_type'] = {
                    'type': 'categorical',
                    'choices': [lt.value for lt in self.layer_types],
                }
        
        return space
```

```python
# Location: milia_pipeline/models/hpo/nas/nas_manager.py

"""
Neural Architecture Search Manager

Orchestrates architecture search using HPO infrastructure.
"""

from typing import Dict, Any, Optional
import torch.nn as nn

from ..hpo_manager import HPOManager
from ..hpo_config import HPOConfig
from .search_space import GNNArchitectureSpace, LayerType, LayerConfig


class NASManager:
    """
    Neural Architecture Search for GNNs.
    
    Uses HPO infrastructure to search over architectures.
    Builds models dynamically from architecture configurations.
    """
    
    def __init__(
        self,
        arch_space: GNNArchitectureSpace,
        hpo_config: Optional[HPOConfig] = None,
    ):
        self.arch_space = arch_space
        
        # Create HPO config with architecture search space
        if hpo_config is None:
            hpo_config = HPOConfig(
                enabled=True,
                n_trials=100,
                search_space=arch_space.to_optuna_search_space(),
            )
        else:
            # Merge architecture space into existing config
            merged_space = {
                **hpo_config.search_space,
                **arch_space.to_optuna_search_space(),
            }
            hpo_config = HPOConfig(
                **{**hpo_config.__dict__, 'search_space': merged_space}
            )
        
        self.hpo_manager = HPOManager(hpo_config)
    
    def search(
        self,
        dataset,
        base_hyperparameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run architecture search.
        
        Returns:
            Best architecture configuration
        """
        # Override model creation to use dynamic architecture
        # This hooks into HPOManager's objective function
        
        best_params = self.hpo_manager.optimize(
            model_name="DynamicGNN",  # Special model that reads arch params
            dataset=dataset,
            base_hyperparameters=base_hyperparameters,
            **kwargs,
        )
        
        return self._extract_architecture(best_params)
    
    def _extract_architecture(
        self,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract architecture config from HPO params."""
        arch = {
            'num_layers': params.get('num_layers', 3),
            'hidden_channels': params.get('hidden_channels', 64),
            'pooling': params.get('pooling', 'mean'),
            'dropout': params.get('dropout', 0.0),
            'layers': [],
        }
        
        # Extract per-layer configs if present
        for i in range(arch['num_layers']):
            layer_type = params.get(f'layer_{i}_type', 'gcn')
            arch['layers'].append({
                'type': layer_type,
                'hidden_channels': arch['hidden_channels'],
            })
        
        return arch
    
    def build_model(
        self,
        architecture: Dict[str, Any],
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """
        Build model from architecture configuration.
        
        Args:
            architecture: Architecture config from search
            in_channels: Input feature dimension
            out_channels: Output dimension
            
        Returns:
            PyTorch model
        """
        from milia_pipeline.models.factory import ModelFactory
        
        # Convert architecture to model hyperparameters
        hyperparameters = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'hidden_channels': architecture['hidden_channels'],
            'num_layers': architecture['num_layers'],
            'dropout': architecture['dropout'],
        }
        
        # For homogeneous architectures, use standard model
        layer_types = [l['type'] for l in architecture['layers']]
        if len(set(layer_types)) == 1:
            model_name = layer_types[0].upper()  # e.g., "GCN", "GAT"
            return ModelFactory().create_model(model_name, hyperparameters)
        
        # For heterogeneous architectures, build custom model
        return self._build_heterogeneous_model(architecture, in_channels, out_channels)
    
    def _build_heterogeneous_model(
        self,
        architecture: Dict[str, Any],
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """Build model with mixed layer types."""
        # Implementation for heterogeneous GNN
        # Uses dynamic layer construction based on architecture config
        pass  # Full implementation
```

### 9.4 Inactive Backends (Complete, Not Integrated)

#### 9.4.1 Ray Tune Backend (`backends/ray_tune_backend.py`)            <--- DONE!

```python
# Location: milia_pipeline/models/hpo/backends/ray_tune_backend.py

"""
Ray Tune HPO Backend (INACTIVE)

Status: Complete implementation, not integrated into main HPO flow.
Activation: Set backend="ray_tune" in config when Ray infrastructure available.

This backend is provided for future use cases requiring:
- Distributed trials across GPU cluster (100+ concurrent)
- Population Based Training (PBT)
- AsyncHyperBand scheduler
- Multi-node parallelization

Prerequisites for activation:
1. Install ray[tune]: pip install "ray[tune]"
2. Initialize Ray cluster (or use local: ray.init())
3. Set backend="ray_tune" in HPOConfig
4. Configure num_gpus, num_cpus per trial

Performance characteristics:
- Overhead: ~50-100ms per trial (Ray actor system)
- Memory: ~100MB+ baseline
- Startup: 3-10s (Ray cluster initialization)
- Best for: Large-scale experiments (1000+ trials)

When to use Ray Tune instead of Optuna:
- 100+ concurrent trials needed
- Existing Ray infrastructure
- Need for PBT or advanced schedulers
- Multi-node GPU cluster available
- Experiment requires 1000+ total trials
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
import logging

from .base import HPOBackendProtocol
from milia_pipeline.exceptions import BackendError, HPOError

logger = logging.getLogger(__name__)

# Lazy import to avoid dependency if not used
ray = None
tune = None


def _ensure_ray_installed():
    """Lazy import Ray Tune."""
    global ray, tune
    if ray is None:
        try:
            import ray as _ray
            from ray import tune as _tune
            ray = _ray
            tune = _tune
        except ImportError:
            raise BackendError(
                "Ray Tune not installed. Install with: pip install 'ray[tune]'",
                backend_name="ray_tune",
                operation="import"
            )


class RayTuneBackend(HPOBackendProtocol):
    """
    Ray Tune HPO backend for distributed optimization.
    
    Status: INACTIVE - Complete implementation for future activation.
    
    Features:
    - Distributed trials across GPU cluster
    - Population Based Training (PBT)
    - AsyncHyperBand scheduler
    - Fault tolerance and checkpointing
    - Integration with Ray cluster
    
    Usage (when activated):
        >>> config = HPOConfig(
        ...     enabled=True,
        ...     backend="ray_tune",
        ...     n_trials=1000,
        ...     n_jobs=8,  # 8 parallel trials
        ... )
        >>> manager = HPOManager(config)
        >>> best_params = manager.optimize(...)
    """
    
    def __init__(
        self,
        num_gpus: float = 0,
        num_cpus: float = 1,
        resources_per_trial: Optional[Dict[str, float]] = None,
        local_dir: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        Initialize Ray Tune backend.
        
        Args:
            num_gpus: GPUs per trial (can be fractional, e.g., 0.5)
            num_cpus: CPUs per trial
            resources_per_trial: Custom resource dict (overrides num_gpus/num_cpus)
            local_dir: Directory for Ray Tune logs/checkpoints
            verbose: Verbosity level (0=silent, 1=status, 2=debug)
        """
        _ensure_ray_installed()
        
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.resources_per_trial = resources_per_trial or {
            "cpu": num_cpus,
            "gpu": num_gpus,
        }
        self.local_dir = local_dir
        self.verbose = verbose
        self._analysis = None  # Stores tune.ExperimentAnalysis after run
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Ray initialized for HPO")
    
    def create_study(
        self,
        study_name: str,
        direction: str,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        sampler: Optional[Any] = None,
        pruner: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Create Ray Tune experiment configuration.
        
        Note: Ray Tune doesn't have a "study" concept like Optuna.
        This returns a config dict used in optimize().
        
        Args:
            study_name: Experiment name
            direction: "minimize" or "maximize"
            storage: Not used (Ray uses local_dir)
            load_if_exists: Whether to resume from checkpoint
            sampler: Search algorithm (converted from Optuna sampler)
            pruner: Scheduler (converted from Optuna pruner)
            
        Returns:
            Experiment configuration dict
        """
        mode = "min" if direction == "minimize" else "max"
        
        return {
            "name": study_name,
            "mode": mode,
            "resume": "AUTO" if load_if_exists else False,
            "search_alg": sampler,
            "scheduler": pruner,
        }
    
    def optimize(
        self,
        study: Dict[str, Any],
        objective_fn: Callable,
        n_trials: int,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        catch: Tuple[type, ...] = (Exception,),
    ) -> None:
        """
        Run Ray Tune optimization.
        
        Args:
            study: Experiment config from create_study()
            objective_fn: Training function (must use tune.report())
            n_trials: Number of trials
            timeout: Max time in seconds
            n_jobs: Concurrent trials
            catch: Exception types to catch (trial marked as failed)
        """
        _ensure_ray_installed()
        
        # Wrap objective to use Ray Tune API
        def trainable(config):
            """Ray Tune trainable wrapper."""
            try:
                # Create a mock trial object for compatibility
                trial = RayTuneTrial(config)
                result = objective_fn(trial)
                tune.report(objective=result)
            except Exception as e:
                if not any(isinstance(e, exc) for exc in catch):
                    raise
                tune.report(objective=float('inf'), error=str(e))
        
        # Configure search algorithm
        search_alg = study.get("search_alg")
        if search_alg is None:
            # Default: Use Optuna search via Ray Tune
            from ray.tune.search.optuna import OptunaSearch
            search_alg = OptunaSearch(metric="objective", mode=study["mode"])
        
        # Configure scheduler
        scheduler = study.get("scheduler")
        if scheduler is None:
            # Default: AsyncHyperBand
            from ray.tune.schedulers import ASHAScheduler
            scheduler = ASHAScheduler(
                metric="objective",
                mode=study["mode"],
                max_t=100,
                grace_period=10,
                reduction_factor=3,
            )
        
        # Run optimization
        self._analysis = tune.run(
            trainable,
            name=study["name"],
            num_samples=n_trials,
            config={},  # Search space added via search_alg
            search_alg=search_alg,
            scheduler=scheduler,
            resources_per_trial=self.resources_per_trial,
            local_dir=self.local_dir,
            verbose=self.verbose,
            resume=study["resume"],
            time_budget_s=timeout,
            max_concurrent_trials=n_jobs,
            raise_on_failed_trial=False,
        )
        
        logger.info(f"Ray Tune optimization completed: {n_trials} trials")
    
    def get_best_params(self, study: Dict[str, Any]) -> Dict[str, Any]:
        """Get best hyperparameters from completed experiment."""
        if self._analysis is None:
            raise HPOError("No experiment completed. Run optimize() first.")
        
        return self._analysis.best_config
    
    def get_best_value(self, study: Dict[str, Any]) -> float:
        """Get best objective value from completed experiment."""
        if self._analysis is None:
            raise HPOError("No experiment completed. Run optimize() first.")
        
        best_trial = self._analysis.best_trial
        return best_trial.last_result.get("objective", float('inf'))
    
    def get_all_trials(self, study: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get information about all trials."""
        if self._analysis is None:
            raise HPOError("No experiment completed. Run optimize() first.")
        
        trials = []
        for trial in self._analysis.trials:
            trials.append({
                'number': trial.trial_id,
                'state': trial.status,
                'value': trial.last_result.get("objective"),
                'params': trial.config,
                'duration': trial.last_result.get("time_total_s"),
                'intermediate_values': {},  # Ray doesn't track same way
            })
        
        return trials
    
    def create_pruner(
        self,
        pruner_type: str,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 10,
        interval_steps: int = 1,
        **kwargs,
    ) -> Any:
        """
        Create Ray Tune scheduler (equivalent to Optuna pruner).
        
        Maps Optuna pruner types to Ray Tune schedulers:
        - median -> MedianStoppingRule
        - hyperband -> HyperBandScheduler
        - percentile -> ASHAScheduler with custom percentile
        - none -> FIFOScheduler (no early stopping)
        """
        _ensure_ray_installed()
        from ray.tune.schedulers import (
            FIFOScheduler,
            ASHAScheduler,
            HyperBandScheduler,
            MedianStoppingRule,
        )
        
        if pruner_type == "none":
            return FIFOScheduler()
        
        elif pruner_type == "median":
            return MedianStoppingRule(
                metric="objective",
                mode="min",
                grace_period=n_warmup_steps,
                min_samples_required=n_startup_trials,
            )
        
        elif pruner_type == "hyperband":
            return HyperBandScheduler(
                metric="objective",
                mode="min",
                max_t=100,
            )
        
        elif pruner_type in ("percentile", "successive_halving"):
            return ASHAScheduler(
                metric="objective",
                mode="min",
                max_t=100,
                grace_period=n_warmup_steps,
                reduction_factor=kwargs.get("reduction_factor", 4),
            )
        
        else:
            logger.warning(f"Unknown pruner type '{pruner_type}', using ASHA")
            return ASHAScheduler(metric="objective", mode="min")
    
    def create_sampler(
        self,
        sampler_type: str,
        seed: Optional[int] = None,
        n_startup_trials: int = 10,
        **kwargs,
    ) -> Any:
        """
        Create Ray Tune search algorithm (equivalent to Optuna sampler).
        
        Maps Optuna sampler types to Ray Tune search algorithms:
        - tpe -> OptunaSearch (uses Optuna's TPE internally)
        - random -> BasicVariantGenerator
        - cmaes -> OptunaSearch with CmaEsSampler
        - grid -> GridSearch
        """
        _ensure_ray_installed()
        from ray.tune.search import BasicVariantGenerator
        from ray.tune.search.optuna import OptunaSearch
        
        if sampler_type == "random":
            return BasicVariantGenerator(random_state=seed)
        
        elif sampler_type == "tpe":
            import optuna
            sampler = optuna.samplers.TPESampler(
                seed=seed,
                n_startup_trials=n_startup_trials,
                multivariate=kwargs.get("multivariate", True),
            )
            return OptunaSearch(sampler=sampler, metric="objective", mode="min")
        
        elif sampler_type == "cmaes":
            import optuna
            sampler = optuna.samplers.CmaEsSampler(seed=seed)
            return OptunaSearch(sampler=sampler, metric="objective", mode="min")
        
        elif sampler_type == "grid":
            # Grid search requires predefined search space
            from ray.tune.search.basic_variant import GridSearch
            return GridSearch()
        
        else:
            logger.warning(f"Unknown sampler type '{sampler_type}', using TPE")
            return OptunaSearch(metric="objective", mode="min")
    
    def suggest_params(
        self,
        trial: 'RayTuneTrial',
        search_space: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get suggested parameters from Ray Tune trial config.
        
        Note: Ray Tune pre-samples all parameters in config,
        unlike Optuna's suggest_* API.
        """
        return trial.config
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if ray is not None and ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")


class RayTuneTrial:
    """
    Mock trial object for Ray Tune compatibility.
    
    Provides Optuna-like interface for objective functions
    that were written for Optuna but need to run on Ray Tune.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._number = hash(str(config)) % 100000
        self._reported_values = []
    
    @property
    def number(self) -> int:
        """Trial number (approximated from config hash)."""
        return self._number
    
    def suggest_int(self, name: str, low: int, high: int, **kwargs) -> int:
        """Get int parameter (pre-sampled in config)."""
        return self.config.get(name, low)
    
    def suggest_float(self, name: str, low: float, high: float, **kwargs) -> float:
        """Get float parameter (pre-sampled in config)."""
        return self.config.get(name, low)
    
    def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
        """Get categorical parameter (pre-sampled in config)."""
        return self.config.get(name, choices[0])
    
    def report(self, value: float, step: int):
        """Report intermediate value (forwarded to tune.report)."""
        self._reported_values.append((step, value))
        tune.report(intermediate_value=value, step=step)
    
    def should_prune(self) -> bool:
        """Check if trial should be pruned (handled by scheduler)."""
        # Ray Tune handles pruning via scheduler, not explicit check
        return False


class RayTuneReportCallback:
    """
    Callback for Ray Tune integration with Trainer.
    
    Status: INACTIVE - For use when Ray Tune backend is activated.
    
    Reports metrics to Ray Tune for scheduler-based early stopping.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        report_every: int = 1,
    ):
        """
        Initialize Ray Tune callback.
        
        Args:
            monitor: Metric to report
            report_every: Report frequency (epochs)
        """
        _ensure_ray_installed()
        
        self.monitor = monitor
        self.report_every = report_every
        self._trainer = None
    
    def set_trainer(self, trainer):
        """Store trainer reference."""
        self._trainer = trainer
    
    def on_train_begin(self, trainer):
        """Called at start of training."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Report metrics to Ray Tune."""
        if epoch % self.report_every != 0:
            return
        
        value = metrics.get(self.monitor)
        if value is None:
            # Try alternative names
            for alt in [f"val_{self.monitor}", f"validation_{self.monitor}"]:
                if alt in metrics:
                    value = metrics[alt]
                    break
        
        if value is not None:
            tune.report(**{self.monitor: value, "epoch": epoch})
    
    def on_train_end(self, trainer, metrics: Dict[str, float]):
        """Called at end of training."""
        pass
```

#### 9.4.2 Ray Tune Activation Guide

**Prerequisites:**
```bash
# Install Ray Tune
pip install "ray[tune]"

# Optional: Install additional search algorithms
pip install "ray[tune]" hyperopt bayesian-optimization
```

**Configuration:**
```yaml
# config.yaml
models:
  hpo:
    enabled: true
    backend: "ray_tune"  # Activate Ray Tune
    n_trials: 1000
    n_jobs: 8            # 8 concurrent trials
    
    # Ray-specific settings (future config extension)
    ray_tune:
      num_gpus_per_trial: 0.5  # Fractional GPU
      num_cpus_per_trial: 2
      local_dir: "./ray_results"
      cluster_address: "auto"  # Or specific address
```

**Programmatic Usage:**
```python
from milia_pipeline.models.hpo import HPOManager, HPOConfig
from milia_pipeline.models.hpo.backends import RayTuneBackend

# Initialize Ray cluster (if not auto)
import ray
ray.init(address="auto")  # Connect to existing cluster

# Create HPO config with Ray Tune
config = HPOConfig(
    enabled=True,
    backend="ray_tune",
    n_trials=1000,
    n_jobs=8,
)

# Run distributed optimization
manager = HPOManager(config)
best_params = manager.optimize(
    model_name="GCN",
    dataset=dataset,
)

# Shutdown Ray when done
ray.shutdown()
```

**When to Activate:**
| Scenario | Use Optuna | Use Ray Tune |
|----------|------------|--------------|
| < 100 trials | ✓ | |
| Single machine | ✓ | |
| Quick experiments | ✓ | |
| 100+ concurrent trials | | ✓ |
| Multi-node cluster | | ✓ |
| Need PBT scheduler | | ✓ |
| 1000+ total trials | | ✓ |

### 9.5 Future Enhancements (Post-Release)

#### 9.5.1 AutoML Pipeline (Priority: Low)

Full AutoML combining NAS + HPO + feature engineering:

```python
# Future: milia_pipeline/automl/
# Requires stable HPO, NAS, and feature engineering modules
```

### 9.4 Dependencies

#### Required Dependencies

```
# In setup.py or requirements.txt
optuna>=3.0.0           # Primary HPO backend
```

#### Optional Dependencies

```
# Optional - for advanced features
ray[tune]>=2.0.0        # Ray Tune backend (future)
optuna-dashboard        # Web-based study visualization
plotly>=5.0.0           # Visualization
kaleido                 # Static image export for plots
scikit-learn>=1.0.0     # For cross-validation utilities
```

### 9.5 Performance Considerations

#### Memory Usage

| Trials | In-Memory Storage | SQLite Storage |
|--------|-------------------|----------------|
| 100    | ~10 MB            | ~5 MB disk     |
| 1,000  | ~100 MB           | ~50 MB disk    |
| 10,000 | ~1 GB             | ~500 MB disk   |

**Recommendation**: Use SQLite storage for >500 trials.

#### Trial Execution Time

| Dataset Size | Model Complexity | Est. Time/Trial |
|--------------|------------------|-----------------|
| 1K samples   | Simple GCN       | ~30 seconds     |
| 10K samples  | GAT with 4 heads | ~2 minutes      |
| 100K samples | SchNet           | ~10 minutes     |

**Recommendation**: Use aggressive pruning for large datasets.

#### Parallel Execution

| n_jobs | Memory Multiplier | Speedup |
|--------|-------------------|---------|
| 1      | 1x                | 1x      |
| 2      | ~1.8x             | ~1.7x   |
| 4      | ~3.2x             | ~2.5x   |
| 8      | ~6x               | ~3.5x   |

**Recommendation**: Set n_jobs based on available GPU memory.

---

## Part 10: Summary

### Key Design Decisions

1. **Optuna as Primary**: Based on code alignment analysis (callback system, config structure, registry patterns)

2. **MASTER SWITCH**: `hpo.enabled: true/false` in config.yaml

3. **Minimal Trainer Modification**: Only 1 new parameter (`hpo_callback`)

4. **Backend Abstraction**: Protocol-based design allows future Ray Tune integration

5. **Exception Hierarchy**: Inherits from `ModelError` in centralized `exceptions.py`

6. **Cross-Validation Support**: Built-in k-fold CV using existing `DataSplitter`

7. **Multi-Objective Optimization**: Pareto front, hypervolume, preference-based selection

8. **Transfer Learning**: Meta-feature similarity, warm-starting, cross-dataset knowledge transfer

9. **Neural Architecture Search**: GNN-specific search space, layer-type mixing, dynamic model building

### Files to Create (22 total)

| Directory | File | Purpose |
|-----------|------|---------|
| `hpo/` | `__init__.py` | Public API |
| `hpo/` | `hpo_config.py` | Configuration classes |
| `hpo/` | `hpo_manager.py` | Main orchestrator |
| `hpo/backends/` | `__init__.py` | Backend exports |
| `hpo/backends/` | `base.py` | Protocol definition |
| `hpo/backends/` | `optuna_backend.py` | Optuna implementation |
| `hpo/backends/` | `ray_tune_backend.py` | Complete, inactive |
| `hpo/callbacks/` | `__init__.py` | Callback exports |
| `hpo/callbacks/` | `optuna_callback.py` | Pruning callback |
| `hpo/callbacks/` | `ray_tune_callback.py` | Complete, inactive |
| `hpo/search_spaces/` | `__init__.py` | Search space exports |
| `hpo/search_spaces/` | `search_space_builder.py` | Space builder |
| `hpo/search_spaces/` | `param_types.py` | Type definitions |
| `hpo/transfer/` | `__init__.py` | Transfer learning exports |
| `hpo/transfer/` | `transfer_manager.py` | HPO transfer manager |
| `hpo/transfer/` | `meta_features.py` | Meta-feature extraction |
| `hpo/transfer/` | `warm_start.py` | Warm-start strategies |
| `hpo/nas/` | `__init__.py` | NAS exports |
| `hpo/nas/` | `search_space.py` | GNN architecture space |
| `hpo/nas/` | `nas_manager.py` | NAS orchestrator |
| `hpo/analysis/` | `__init__.py` | Analysis exports |
| `hpo/analysis/` | `study_analyzer.py` | Results analysis |

### Files to Modify (5 total)

| File | Modification |
|------|--------------|
| `config.yaml` | Add HPO section (~80 lines) |
| `config_bridge.py` | Add HPO bridge classes (~100 lines) |
| `exceptions.py` | Add HPO exceptions (~150 lines after line 2810) |
| `trainer.py` | Add `hpo_callback` parameter (~5 lines) |
| `models/__init__.py` | Add HPO exports (~15 lines) |

**Deferred to models/ full integration:**
- `main.py` - HPO orchestration from CLI/config entry point

### Estimated Implementation Time

| Phase | Time |
|-------|------|
| Foundation | 1.5 hours |
| Backend Infrastructure | 3 hours |
| Callbacks and Search Spaces | 2.5 hours |
| Manager, Transfer, NAS, Analysis | 6 hours |
| Integration | 1.5 hours |
| Testing | 4.5 hours |
| **Total** | **~19 hours** |

---

**End of HPO Implementation Blueprint**

