# models/__init__
"""
milia Pipeline - Models Module

Production-ready ML/DL model lifecycle management with:
- 120+ PyTorch Geometric models organized in 12 categories
- Config-driven training and evaluation
- Comprehensive callbacks (EarlyStopping, ModelCheckpoint, TensorBoard)
- Data splitting strategies (random, stratified, temporal, scaffold)
- Hardware acceleration support (CPU, CUDA, MPS, distributed)
- Plugin architecture for custom models
- Complete training infrastructure
- **PHASE 7**: Custom architecture building and multi-model ensembles
- **PHASE 8**: Hyperparameter Optimization (HPO) with Optuna backend

Quick Start
-----------
1. Select a model:
    >>> from milia_pipeline.models import list_models, get_model_info
    >>> models = list_models(task_type="graph_regression")
    >>> info = get_model_info("GCN")

2. Create and train:
    >>> from milia_pipeline.models import create_model, Trainer
    >>> from milia_pipeline.models import DataSplitter, EarlyStopping
    >>>
    >>> # Create model
    >>> model = create_model(
    ...     name="GCN",
    ...     hyperparameters={"hidden_channels": 64, "num_layers": 3},
    ...     task_type="graph_regression",
    ...     sample_data=dataset[0]
    ... )
    >>>
    >>> # Split data
    >>> train_data, val_data, test_data = DataSplitter.random_split(dataset)
    >>>
    >>> # Train
    >>> trainer = Trainer(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     loss_fn=nn.MSELoss(),
    ...     optimizer=torch.optim.Adam(model.parameters()),
    ...     callbacks=[EarlyStopping(patience=10)]
    ... )
    >>> results = trainer.fit()

3. **PHASE 7: Build custom architectures**:
    >>> from milia_pipeline.models import ArchitectureBuilder
    >>>
    >>> builder = ArchitectureBuilder('graph_regression', in_channels=16, out_channels=1)
    >>> builder.add_layer('GCNConv', out_channels=64)
    >>> builder.add_layer('ReLU')
    >>> builder.add_layer('GATConv', out_channels=32, heads=4)
    >>> builder.add_layer('global_mean_pool')
    >>> builder.add_layer('Linear', out_features=1)
    >>> model = builder.build()

4. **PHASE 7: Create ensembles**:
    >>> from milia_pipeline.models import ModelComposer
    >>>
    >>> composer = ModelComposer('graph_regression')
    >>> composer.add_model(create_model("GCN", {...}, "graph_regression"))
    >>> composer.add_model(create_model("GAT", {...}, "graph_regression"))
    >>> composer.set_strategy('parallel')
    >>> ensemble = composer.build()

5. **PHASE 8: Hyperparameter Optimization**:
    >>> from milia_pipeline.models import HPOManager, HPOConfig, HPO_AVAILABLE
    >>>
    >>> if HPO_AVAILABLE:
    ...     config = HPOConfig(
    ...         enabled=True,
    ...         n_trials=100,
    ...         metric="val_loss",
    ...         direction="minimize"
    ...     )
    ...     manager = HPOManager(config)
    ...     best_params = manager.optimize(
    ...         model_name="GCN",
    ...         dataset=dataset,
    ...         search_space={"hidden_channels": (32, 256), "num_layers": (2, 6)}
    ...     )

Components
----------
- **Registry**: ModelRegistry, get_model, list_models, search_models
- **Factory**: ModelFactory, create_model, ModelValidator
- **Training**: Trainer, callbacks, DataSplitter
- **Infrastructure**: LossRegistry, OptimizerRegistry, SchedulerRegistry
- **Categories**: 120+ models in 12 categories (BASIC_GNN, CONVOLUTIONAL, etc.)
- **PHASE 7: Builders**: LayerRegistry, ArchitectureBuilder, ModelComposer, Templates
- **PHASE 8: HPO**: HPOManager, HPOConfig, OptunaPruningCallback, SearchSpaceBuilder

Version: 1.2.0 (Phase 8 HPO)
Author: milia Team
"""

__version__ = "1.2.0"

import logging

# ============================================================================
# LOGGING SETUP (MUST BE BEFORE IMPORTS)
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS - CORE COMPONENTS
# ============================================================================

# Initialize empty lists for missing imports
_MISSING_COMPONENTS = []

# Categories and Metadata - Now uses dynamic introspection (Phase 2.3 migration)
try:
    # MODELS_BY_CATEGORY is deprecated but kept for backward compatibility
    # It's now dynamically generated from the introspector
    from .registry import MODELS_BY_CATEGORY
    from .registry.pyg_introspector import (
        ALL_MODELS,
        ModelCategory,
        ModelMetadata,
        get_all_model_names,
        get_category_statistics,
        get_model_metadata,
        get_models_by_category,
        get_models_by_tag,
        get_models_by_task,
        search_models,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("registry.pyg_introspector")
    logger.warning(f"Could not import pyg_introspector: {e}")

    # Create dummy classes/functions to prevent import errors
    class ModelCategory:
        pass

    class ModelMetadata:
        pass

    def get_model_metadata(*args, **kwargs):
        raise NotImplementedError("pyg_introspector module not available")

    def get_all_model_names(*args, **kwargs):
        raise NotImplementedError("pyg_introspector module not available")

    def get_models_by_category(*args, **kwargs):
        raise NotImplementedError("pyg_introspector module not available")

    def get_models_by_task(*args, **kwargs):
        raise NotImplementedError("pyg_introspector module not available")

    def get_models_by_tag(*args, **kwargs):
        raise NotImplementedError("pyg_introspector module not available")

    def search_models(*args, **kwargs):
        raise NotImplementedError("pyg_introspector module not available")

    def get_category_statistics(*args, **kwargs):
        return {}

    ALL_MODELS = []
    MODELS_BY_CATEGORY = {}

# Registry System
try:
    from .registry.model_registry import (
        ModelRegistration,
        ModelRegistry,
        get_model,
        get_model_info,
        has_model,
        list_models,
        registry,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("registry.model_registry")
    logger.warning(f"Could not import model_registry: {e}")

    class ModelRegistry:
        def list_available_models(self):
            return []

    class ModelRegistration:
        pass

    registry = ModelRegistry()

    def get_model(*args, **kwargs):
        raise NotImplementedError("model_registry module not available")

    def has_model(*args, **kwargs):
        return False

    def list_models(*args, **kwargs):
        return []

    def get_model_info(*args, **kwargs):
        raise NotImplementedError("model_registry module not available")


# Factory and Validator
try:
    from .factory.model_factory import (
        EdgeLevelModelWrapper,
        GraphLevelModelWrapper,
        ModelFactory,
        ModelValidator,
        create_model,
        get_factory,
    )
    from .factory.model_factory import (
        get_model_info as get_factory_model_info,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("factory.model_factory")
    logger.warning(f"Could not import model_factory: {e}")

    class ModelFactory:
        pass

    class ModelValidator:
        pass

    class GraphLevelModelWrapper:
        pass

    class EdgeLevelModelWrapper:
        pass

    def create_model(*args, **kwargs):
        raise NotImplementedError("model_factory module not available")

    def get_factory(*args, **kwargs):
        raise NotImplementedError("model_factory module not available")

    def get_factory_model_info(*args, **kwargs):
        raise NotImplementedError("model_factory module not available")

# ============================================================================
# PHASE 7: BUILDERS MODULE
# ============================================================================

try:
    from .builders.layer_registry import (
        LayerCategory,
        LayerMetadata,
        LayerRegistry,
    )
    from .builders.layer_registry import (
        registry as layer_registry,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("builders.layer_registry")
    logger.warning(f"Could not import layer_registry: {e}")

    class LayerRegistry:
        pass

    class LayerCategory:
        pass

    class LayerMetadata:
        pass

    layer_registry = None

try:
    from .builders.architecture_builder import (
        ArchitectureBuilder,
        ArchitectureConfig,
        LayerConfig,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("builders.architecture_builder")
    logger.warning(f"Could not import architecture_builder: {e}")

    class ArchitectureBuilder:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("builders module not available")

    class LayerConfig:
        pass

    class ArchitectureConfig:
        pass


try:
    from .builders.model_composer import (
        EnsembleConfig,
        ModelComposer,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("builders.model_composer")
    logger.warning(f"Could not import model_composer: {e}")

    class ModelComposer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("builders module not available")

    class EnsembleConfig:
        pass


try:
    from .builders.validation import (
        ArchitectureValidator,
        validate_architecture,
        validate_data_compatibility,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("builders.validation")
    logger.warning(f"Could not import validation: {e}")

    class ArchitectureValidator:
        pass

    def validate_architecture(*args, **kwargs):
        raise NotImplementedError("builders module not available")

    def validate_data_compatibility(*args, **kwargs):
        raise NotImplementedError("builders module not available")


try:
    from .builders.templates import (
        ArchitectureTemplates,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("builders.templates")
    logger.warning(f"Could not import templates: {e}")

    class ArchitectureTemplates:
        pass


try:
    from .builders.config_parser import (
        load_config,
        parse_custom_architecture,
        parse_ensemble,
        validate_config,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("builders.config_parser")
    logger.warning(f"Could not import config_parser: {e}")

    def parse_custom_architecture(*args, **kwargs):
        raise NotImplementedError("builders module not available")

    def parse_ensemble(*args, **kwargs):
        raise NotImplementedError("builders module not available")

    def load_config(*args, **kwargs):
        raise NotImplementedError("builders module not available")

    def validate_config(*args, **kwargs):
        raise NotImplementedError("builders module not available")

# ============================================================================
# PHASE 8: HPO MODULE (Hyperparameter Optimization)
# ============================================================================

# HPO exports (conditional import for optional dependency)
try:
    from .hpo import (
        HPOConfig,
        HPOManager,
        OptunaPruningCallback,
        create_hpo_manager,
        get_best_params,
        infer_task_type,
        is_hpo_enabled,
    )

    HPO_AVAILABLE = True
except ImportError as e:
    _MISSING_COMPONENTS.append("hpo")
    logger.warning(f"Could not import HPO module: {e}")
    HPO_AVAILABLE = False
    HPOManager = None
    HPOConfig = None
    is_hpo_enabled = None
    get_best_params = None
    create_hpo_manager = None
    infer_task_type = None
    OptunaPruningCallback = None

# ============================================================================
# TRAINING INFRASTRUCTURE
# ============================================================================

try:
    from .training.trainer import (
        Trainer,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("training.trainer")
    logger.warning(f"Could not import trainer: {e}")

    class Trainer:
        pass


try:
    from .training.callbacks import (
        Callback,
        # Config-based callback creation (v2.12.0)
        CallbackFactory,
        EarlyStopping,
        GradientMonitor,
        LearningRateMonitor,
        ModelCheckpoint,
        ProgressBar,
        TensorBoardLogger,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("training.callbacks")
    logger.warning(f"Could not import callbacks: {e}")

    class Callback:
        pass

    class EarlyStopping:
        pass

    class ModelCheckpoint:
        pass

    class TensorBoardLogger:
        pass

    class LearningRateMonitor:
        pass

    class ProgressBar:
        pass

    class GradientMonitor:
        pass

    class CallbackFactory:
        pass


try:
    from .training.data_splitting import (
        DataSplitter,
        k_fold_split,
        random_split,
        scaffold_split,
        stratified_split,
        temporal_split,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("training.data_splitting")
    logger.warning(f"Could not import data_splitting: {e}")

    class DataSplitter:
        pass

    def random_split(*args, **kwargs):
        raise NotImplementedError("data_splitting module not available")

    def stratified_split(*args, **kwargs):
        raise NotImplementedError("data_splitting module not available")

    def temporal_split(*args, **kwargs):
        raise NotImplementedError("data_splitting module not available")

    def scaffold_split(*args, **kwargs):
        raise NotImplementedError("data_splitting module not available")

    def k_fold_split(*args, **kwargs):
        raise NotImplementedError("data_splitting module not available")


try:
    from .training.loss_functions import (
        FocalLoss,
        LossRegistry,
        RMSELoss,
        WeightedMSELoss,
        get_loss,
        list_losses,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("training.loss_functions")
    logger.warning(f"Could not import loss_functions: {e}")

    class LossRegistry:
        @staticmethod
        def list_available():
            return []

    class FocalLoss:
        pass

    class WeightedMSELoss:
        pass

    class RMSELoss:
        pass

    def get_loss(*args, **kwargs):
        raise NotImplementedError("loss_functions module not available")

    def list_losses(*args, **kwargs):
        return []


try:
    from .training.optimizers import (
        OptimizerRegistry,
        get_optimizer,
        list_optimizers,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("training.optimizers")
    logger.warning(f"Could not import optimizers: {e}")

    class OptimizerRegistry:
        @staticmethod
        def list_available():
            return []

    def get_optimizer(*args, **kwargs):
        raise NotImplementedError("optimizers module not available")

    def list_optimizers(*args, **kwargs):
        return []


try:
    from .training.schedulers import (
        SchedulerRegistry,
        create_warmup_scheduler,
        get_scheduler,
        list_schedulers,
    )
except ImportError as e:
    _MISSING_COMPONENTS.append("training.schedulers")
    logger.warning(f"Could not import schedulers: {e}")

    class SchedulerRegistry:
        @staticmethod
        def list_available():
            return []

    def get_scheduler(*args, **kwargs):
        raise NotImplementedError("schedulers module not available")

    def list_schedulers(*args, **kwargs):
        return []

    def create_warmup_scheduler(*args, **kwargs):
        raise NotImplementedError("schedulers module not available")


# Exceptions (from main exceptions module)
try:
    from milia_pipeline.exceptions import (
        CheckpointError,
        DataCompatibilityError,
        DataError,
        HyperparameterError,
        ModelError,
        ModelInstantiationError,
        ModelNotFoundError,
        ModelValidationError,
        PluginModelError,
        TrainingError,
    )
except ImportError:
    # Fallback if exceptions module structure is different
    from .factory.model_factory import (
        DataCompatibilityError,
        HyperparameterError,
        ModelError,
        ModelInstantiationError,
        ModelValidationError,
    )
    from .training.data_splitting import (
        DataError,
    )
    from .training.trainer import (
        CheckpointError,
        TrainingError,
    )

    # Define missing ones
    class ModelNotFoundError(ModelError):
        pass

    class PluginModelError(ModelError):
        pass


# ============================================================================
# PUBLIC API DEFINITION
# ============================================================================

__all__ = [
    # ========================================================================
    # VERSION
    # ========================================================================
    "__version__",
    # ========================================================================
    # CATEGORIES & METADATA
    # ========================================================================
    "ModelCategory",
    "ModelMetadata",
    "get_model_metadata",
    "get_all_model_names",
    "get_models_by_category",
    "get_models_by_task",
    "get_models_by_tag",
    "search_models",
    "get_category_statistics",
    "ALL_MODELS",
    "MODELS_BY_CATEGORY",
    # ========================================================================
    # REGISTRY SYSTEM
    # ========================================================================
    "ModelRegistry",
    "ModelRegistration",
    "registry",
    "get_model",
    "has_model",
    "list_models",
    "get_model_info",
    # ========================================================================
    # FACTORY & VALIDATOR
    # ========================================================================
    "ModelFactory",
    "ModelValidator",
    "GraphLevelModelWrapper",
    "EdgeLevelModelWrapper",
    "create_model",
    "get_factory",
    # ========================================================================
    # PHASE 7: BUILDERS MODULE
    # ========================================================================
    # Layer Registry
    "LayerRegistry",
    "LayerCategory",
    "LayerMetadata",
    "layer_registry",
    # Architecture Builder
    "ArchitectureBuilder",
    "LayerConfig",
    "ArchitectureConfig",
    # Model Composer
    "ModelComposer",
    "EnsembleConfig",
    # Validation
    "ArchitectureValidator",
    "validate_architecture",
    "validate_data_compatibility",
    # Templates
    "ArchitectureTemplates",
    # Config Parser
    "parse_custom_architecture",
    "parse_ensemble",
    "load_config",
    "validate_config",
    # ========================================================================
    # PHASE 8: HPO MODULE (Hyperparameter Optimization)
    # ========================================================================
    "HPOManager",
    "HPOConfig",
    "is_hpo_enabled",
    "get_best_params",
    "create_hpo_manager",
    "infer_task_type",
    "OptunaPruningCallback",
    "HPO_AVAILABLE",
    # ========================================================================
    # TRAINING INFRASTRUCTURE
    # ========================================================================
    "Trainer",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "TensorBoardLogger",
    "LearningRateMonitor",
    "ProgressBar",
    "GradientMonitor",
    "CallbackFactory",
    # Data Splitting
    "DataSplitter",
    "random_split",
    "stratified_split",
    "temporal_split",
    "scaffold_split",
    "k_fold_split",
    # Loss Functions
    "LossRegistry",
    "FocalLoss",
    "WeightedMSELoss",
    "RMSELoss",
    "get_loss",
    "list_losses",
    # Optimizers
    "OptimizerRegistry",
    "get_optimizer",
    "list_optimizers",
    # Schedulers
    "SchedulerRegistry",
    "get_scheduler",
    "list_schedulers",
    "create_warmup_scheduler",
    # ========================================================================
    # EXCEPTIONS
    # ========================================================================
    "ModelError",
    "ModelNotFoundError",
    "ModelValidationError",
    "ModelInstantiationError",
    "HyperparameterError",
    "DataCompatibilityError",
    "TrainingError",
    "CheckpointError",
    "DataError",
    "PluginModelError",
]


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Log initialization
logger.info(f"milia Models Module v{__version__} initialized (Phase 8 HPO)")

# Warn about missing components
if _MISSING_COMPONENTS:
    logger.warning(
        f"Some components could not be imported: {', '.join(_MISSING_COMPONENTS)}. "
        "Acceleration module and other available components will still work."
    )
else:
    logger.info("All components loaded successfully, including builders and HPO modules")

# Log registry statistics
try:
    available_models = registry.list_available_models()
    logger.info(f"Model registry: {len(available_models)} models available")

    # Log category breakdown
    stats = get_category_statistics()
    if stats:
        logger.debug(f"Category breakdown: {stats}")

except Exception as e:
    logger.warning(f"Could not load registry statistics: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_module_info() -> dict:
    """
    Get comprehensive information about the models module.

    Returns:
        Dictionary with module information including:
        - version: Module version
        - total_models: Total number of available models
        - categories: Number of model categories
        - category_breakdown: Models per category
        - available_losses: Number of loss functions
        - available_optimizers: Number of optimizers
        - available_schedulers: Number of schedulers
        - builders_available: Whether builders module is loaded
        - hpo_available: Whether HPO module is loaded

    Example:
        >>> from milia_pipeline.models import get_module_info
        >>> info = get_module_info()
        >>> print(f"Models available: {info['total_models']}")
        >>> print(f"Builders module: {info['builders_available']}")
        >>> print(f"HPO module: {info['hpo_available']}")
    """
    try:
        builders_available = "builders.layer_registry" not in _MISSING_COMPONENTS
        hpo_available = HPO_AVAILABLE

        return {
            "version": __version__,
            "total_models": len(registry.list_available_models()),
            "categories": len(ModelCategory),
            "category_breakdown": get_category_statistics(),
            "available_losses": len(LossRegistry.list_available()),
            "available_optimizers": len(OptimizerRegistry.list_available()),
            "available_schedulers": len(SchedulerRegistry.list_available()),
            "builders_available": builders_available,
            "hpo_available": hpo_available,
            "phase_7_features": {
                "custom_architectures": builders_available,
                "ensemble_models": builders_available,
                "architecture_templates": builders_available,
                "layer_registry": builders_available,
            },
            "phase_8_features": {
                "hyperparameter_optimization": hpo_available,
                "optuna_backend": hpo_available,
                "pruning_callbacks": hpo_available,
                "search_space_builder": hpo_available,
                "study_analyzer": hpo_available,
            },
        }
    except Exception as e:
        logger.error(f"Error getting module info: {e}")
        return {"version": __version__, "error": str(e)}


def print_module_summary():
    """
    Print a formatted summary of the models module.

    Example:
        >>> from milia_pipeline.models import print_module_summary
        >>> print_module_summary()
    """
    info = get_module_info()

    print("=" * 70)
    print(f"milia Models Module v{info['version']}")
    print("=" * 70)

    if "error" in info:
        print(f"Error: {info['error']}")
        return

    print(f"\nTotal Models Available: {info['total_models']}")
    print(f"Model Categories: {info['categories']}")

    print("\nCategory Breakdown:")
    for category, count in sorted(info["category_breakdown"].items(), key=lambda x: -x[1]):
        print(f"  {category:30s}: {count:3d} models")

    print("\nTraining Infrastructure:")
    print(f"  Loss Functions: {info['available_losses']}")
    print(f"  Optimizers: {info['available_optimizers']}")
    print(f"  Schedulers: {info['available_schedulers']}")

    print("\nPhase 7 Features:")
    builders = "✓ Enabled" if info["builders_available"] else "✗ Not Available"
    print(f"  Builders Module: {builders}")
    if info["builders_available"]:
        print("    - Custom Architectures: ✓")
        print("    - Ensemble Models: ✓")
        print("    - Architecture Templates: ✓")
        print("    - Layer Registry: ✓")

    print("\nPhase 8 Features:")
    hpo = "✓ Enabled" if info["hpo_available"] else "✗ Not Available"
    print(f"  HPO Module: {hpo}")
    if info["hpo_available"]:
        print("    - Hyperparameter Optimization: ✓")
        print("    - Optuna Backend: ✓")
        print("    - Pruning Callbacks: ✓")
        print("    - Search Space Builder: ✓")
        print("    - Study Analyzer: ✓")

    print("=" * 70)


# ============================================================================
# MODULE METADATA
# ============================================================================

__author__ = "milia Team"
__license__ = "MIT"
__maintainer__ = "milia Team"
__status__ = "Production"

# Module description for help()
__doc__ = """
milia Models Module - Production-Ready ML/DL Model Lifecycle Management

This module provides a complete infrastructure for training and evaluating
graph neural networks with PyTorch Geometric, featuring:

Core Features:
- 120+ pre-configured PyTorch Geometric models
- Zero-code model selection via configuration
- Comprehensive training infrastructure
- Hardware acceleration support
- Plugin architecture for custom models
- **PHASE 7**: Custom architecture building and multi-model ensembles
- **PHASE 8**: Hyperparameter Optimization (HPO) with Optuna backend

Main Components:
- ModelRegistry: Discover and manage 120+ PyG models
- ModelFactory: Create models with automatic validation
- Trainer: Full-featured training loop with callbacks
- DataSplitter: Multiple splitting strategies
- Registries: Loss functions, optimizers, schedulers
- **LayerRegistry**: Catalog of GNN layers
- **ArchitectureBuilder**: Dynamic layer composition
- **ModelComposer**: Multi-model ensembles
- **ArchitectureTemplates**: Pre-built architectures
- **HPOManager**: Hyperparameter optimization orchestrator
- **HPOConfig**: HPO configuration management
- **OptunaPruningCallback**: Trial pruning integration

Quick Start:
    from milia_pipeline.models import create_model, Trainer

    # Standard model
    model = create_model("GCN", {...}, "graph_regression", data[0])
    trainer = Trainer(model, train_loader, val_loader, ...)
    results = trainer.fit()

    # Custom architecture (Phase 7)
    from milia_pipeline.models import ArchitectureBuilder
    builder = ArchitectureBuilder('graph_regression', 16, 1)
    builder.add_layer('GCNConv', out_channels=64)
    builder.add_layer('ReLU')
    model = builder.build()

    # Ensemble (Phase 7)
    from milia_pipeline.models import ModelComposer
    composer = ModelComposer('graph_regression')
    composer.add_model(model1)
    composer.add_model(model2)
    ensemble = composer.build()

    # Hyperparameter Optimization (Phase 8)
    from milia_pipeline.models import HPOManager, HPOConfig, HPO_AVAILABLE
    if HPO_AVAILABLE:
        config = HPOConfig(enabled=True, n_trials=100)
        manager = HPOManager(config)
        best_params = manager.optimize(model_name="GCN", dataset=dataset)

For more information:
    - help(milia_pipeline.models.ModelRegistry)
    - help(milia_pipeline.models.Trainer)
    - help(milia_pipeline.models.create_model)
    - help(milia_pipeline.models.ArchitectureBuilder)  # Phase 7
    - help(milia_pipeline.models.ModelComposer)  # Phase 7
    - help(milia_pipeline.models.HPOManager)  # Phase 8
    - help(milia_pipeline.models.HPOConfig)  # Phase 8
"""
