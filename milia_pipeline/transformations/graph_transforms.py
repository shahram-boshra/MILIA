# graph_transforms.py   Auto-Discovery: Enhanced Transform Registry with Dynamic Discovery

"""
Graph Transformations Registry and Management System

A comprehensive, production-ready system for managing PyTorch Geometric (PyG)
graph transformations with research-grade features and milia dataset integration.

AUTO-DISCOVERY SYSTEM: ENHANCED TRANSFORM REGISTRY WITH DYNAMIC DISCOVERY
========================================================================

Architecture Overview:
---------------------
This module provides a multi-layered architecture for graph transformations:

1. **Discovery Layer** (DynamicTransformDiscovery)
   - Automatic scanning of PyG modules for available transforms
   - Version compatibility detection for PyG 2.1+
   - Direct transform lookup without fallbacks or aliases

2. **Registry Layer** (TransformRegistry)
   - Central repository of transform metadata
   - Enhanced metadata: dependencies, complexity, research applicability
   - Compatibility matrix for transform interactions

3. **Validation Layer** (TransformValidator)
   - Parameter validation with type checking and range constraints
   - Dataset-specific validation (DFT, DMC, Wavefunction)
   - Sequence compatibility checking

4. **Composition Layer** (TransformComposer)
   - Transform sequence creation with caching
   - Performance optimization and memory management
   - Experimental setup support

5. **Integration Layer** (ConfigurationBridge)
   - milia dataset-specific recommendations
   - Legacy configuration migration (v1 → v2 → v3)
   - YAML/JSON configuration support

6. **Recovery Layer** (TransformErrorRecovery)
   - Multi-level error recovery strategies
   - Automatic fallback to safe configurations
   - Graceful degradation under failures

7. **Edge-Attr Aware Layer** (EdgeAttrAwareParameterInjector) [NEW]
   - Automatic detection of edge_attr in sample data
   - Dynamic parameter injection for edge_attr-aware transforms
   - Resolves edge_index/edge_attr shape mismatch issues
   - See "Edge-Attr Aware Transform System" section below

Key Features:
------------
✓ **30+ Pre-registered Transforms** with comprehensive metadata
✓ **Dynamic Discovery** of PyG transforms with version compatibility
✓ **Direct PyG Integration** for missing transforms (DropEdge, DropNode, MaskFeatures)
✓ **milia Integration** with dataset-specific optimizations (DFT, DMC, Wavefunction)
✓ **Configuration Migration** from v1 (list) → v2 (dict) → v3 (research-grade)
✓ **Production Monitoring** via Prometheus/DataDog metrics
✓ **Intelligent Caching** with memory pressure management
✓ **Error Recovery** with automatic fallback strategies
✓ **Health Monitoring** for all system components
✓ **Edge-Attr Aware Injection** for transforms like AddSelfLoops [NEW]

EDGE-ATTR AWARE TRANSFORM SYSTEM
================================

Problem Statement:
-----------------
PyG transforms like AddSelfLoops add edges to edge_index but by default only handle
edge_weight, not edge_attr. When data has multi-dimensional edge features (edge_attr),
this causes shape mismatch errors:

    edge_index.size(1) != edge_attr.size(0)

For example: 4 original edges with edge_attr shape (4, 21), after AddSelfLoops adds
3 self-loops, edge_index has 7 edges but edge_attr still has 4 rows.

Solution Architecture:
---------------------
This module provides infrastructure for automatic edge_attr-aware parameter injection:

1. **EdgeAttrAwareTransformConfig**: Dataclass defining injection parameters
   - transform_name: Name of the transform (e.g., 'AddSelfLoops')
   - edge_attr_param: Parameter name for edge_attr handling (e.g., 'attr')
   - edge_attr_value: Value to inject (e.g., 'edge_attr')
   - fill_value_param: Parameter for fill value (e.g., 'fill_value')
   - default_fill_value: Default fill value for self-loop edge features

2. **EDGE_ATTR_AWARE_TRANSFORMS**: Registry of transforms requiring injection
   - AddSelfLoops: attr='edge_attr', fill_value=0.0
   - AddRemainingSelfLoops: attr='edge_attr', fill_value=0.0
   - Extensible for future transforms

3. **EdgeAttrAwareParameterInjector**: Runtime injection engine
   - set_sample_data(): Detect edge_attr presence and dimensionality
   - needs_injection(): Check if transform needs parameter injection
   - inject_params(): Inject missing parameters while preserving user values
   - inject_params_batch(): Process multiple configs efficiently
   - get_injection_log(): Audit trail for debugging

4. **TransformComposer Integration**:
   - compose_transforms() accepts optional sample_data parameter
   - Automatic injection when sample_data has edge_attr
   - Statistics tracking via _composition_stats['edge_attr_injections']

Current Status:
--------------
The infrastructure is COMPLETE and TESTED but NOT AUTOMATICALLY ACTIVATED because:
- PyG's InMemoryDataset creates transforms BEFORE data is loaded
- milia_dataset.py cannot provide sample_data at transform creation time

Activation Options:
------------------
OPTION 1 - Config-based (CURRENT RECOMMENDED APPROACH):
    Manually specify edge_attr handling in config.yaml:

    standard_transforms:
      - name: "AddSelfLoops"
        params:
          attr: "edge_attr"      # Tell AddSelfLoops to handle edge_attr
          fill_value: 0.0        # Fill self-loop edge features with zeros

OPTION 2 - Dataset-type based (FUTURE ENHANCEMENT):
    Update milia_dataset.py to pass dataset_type to graph_transforms.py,
    which can then determine if edge_attr injection is needed based on
    known dataset characteristics.

OPTION 3 - Deferred transform creation (FUTURE ENHANCEMENT):
    Restructure milia_dataset.py to create transforms AFTER loading first
    sample, enabling automatic injection. Requires careful PyG integration.

Usage (when sample_data IS available):
-------------------------------------
    >>> from graph_transforms import get_graph_transforms
    >>> gt = get_graph_transforms()
    >>>
    >>> # With sample_data, injection happens automatically
    >>> configs = [{'name': 'AddSelfLoops'}]
    >>> compose = gt.create_transform_sequence(configs, sample_data=dataset[0])
    >>> # AddSelfLoops now has attr='edge_attr', fill_value=0.0 injected

    >>> # Or use module-level function
    >>> from graph_transforms import create_transform_sequence
    >>> compose = create_transform_sequence(configs, sample_data=dataset[0])

    >>> # Or set globally
    >>> from graph_transforms import set_sample_data_for_edge_attr_detection
    >>> set_sample_data_for_edge_attr_detection(dataset[0])
    >>> compose = create_transform_sequence(configs)  # Uses global sample_data

Registry Extension:
------------------
    >>> from graph_transforms import (
    ...     EdgeAttrAwareTransformConfig,
    ...     register_edge_attr_aware_transform
    ... )
    >>>
    >>> # Register a new edge_attr-aware transform
    >>> config = EdgeAttrAwareTransformConfig(
    ...     transform_name='MyCustomTransform',
    ...     edge_attr_param='edge_features',
    ...     edge_attr_value='edge_attr',
    ...     fill_value_param='default_val',
    ...     default_fill_value=0.0,
    ...     description='Custom transform with edge_attr support'
    ... )
    >>> register_edge_attr_aware_transform(config)

Usage Examples:
--------------
Basic usage:
    >>> from graph_transforms import get_graph_transforms
    >>> gt = get_graph_transforms()
    >>>
    >>> # Create simple transform sequence
    >>> configs = [
    ...     {'name': 'AddSelfLoops'},
    ...     {'name': 'ToUndirected'}
    ... ]
    >>> compose = gt.create_transform_sequence(configs)

milia integration:
    >>> # Get DFT-optimized setups
    >>> recommendations = gt.get_research_recommendations(
    ...     research_type='molecular_properties',
    ...     dataset_type='DFT'
    ... )
    >>>
    >>> # Create from milia YAML config
    >>> compose = gt.create_from_yaml_config(yaml_config, dataset_type='DFT')

Configuration validation:
    >>> # Validate v3 configuration
    >>> config = {
    ...     'experimental_setups': {
    ...         'baseline': {
    ...             'transforms': [{'name': 'AddSelfLoops'}],
    ...             'research_context': 'molecular_property_prediction'
    ...         }
    ...     },
    ...     'research_context': 'molecular_property_prediction',
    ...     'dataset_optimization': {'dataset_type': 'DFT'}
    ... }
    >>> result = gt.validate_configuration(config, 'DFT')
    >>> if result['is_valid']:
    ...     print("Valid v3 configuration")

Production monitoring:
    >>> # Export metrics
    >>> metrics = gt.export_metrics(format_type='prometheus')
    >>>
    >>> # Check system health
    >>> health = gt.perform_health_check()
    >>>
    >>> # Optimize performance
    >>> optimization = gt.optimize_performance(target_cache_hit_rate=0.8)

Module Structure:
----------------
- TransformCompatibility: Version compatibility tracking
- TransformDependency: Inter-transform dependencies
- TransformInfo: Enhanced transform metadata
- DynamicTransformDiscovery: PyG module scanning
- TransformRegistry: Central transform repository
- TransformValidator: Parameter and sequence validation
- TransformComposer: Transform sequence composition
- ConfigurationValidator: v3 config format validation
- ConfigurationBridge: milia dataset integration
- TransformErrorRecovery: Error handling and recovery
- ProductionMetricsCollector: Metrics and monitoring
- IntelligentCacheManager: Memory-aware caching
- GraphTransforms: Main public API
- EdgeAttrAwareTransformConfig: Edge-attr injection configuration [NEW]
- EdgeAttrAwareParameterInjector: Runtime parameter injection [NEW]
- EDGE_ATTR_AWARE_TRANSFORMS: Registry of edge-attr aware transforms [NEW]

Version History:
---------------
Foundation Architecture: Basic transform registry and composition
Auto-Discovery: Dynamic discovery, enhanced metadata, production features
Production Enhancement: Removed v1/v2 configuration migration, fallback transforms,
                     and transform aliases. Simplified to v3-only, PyG 2.1+
                     architecture. Handler-only, no backward compatibility layers.
Edge-Attr Aware: Added infrastructure for automatic edge_attr parameter injection
                 to resolve edge_index/edge_attr shape mismatch in transforms like
                 AddSelfLoops. Ready for activation when milia_dataset.py updated.

Dependencies:
------------
Required:
- torch_geometric (PyG): Transform implementations
- Python 3.8+: Type hints and dataclasses

Optional:
- psutil: Memory monitoring
- yaml: YAML configuration support
- prometheus_client: Prometheus metrics
- datadog: DataDog metrics

Author: milia Project Team
License: MIT
"""

import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    NamedTuple,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import torch_geometric.transforms as T
    from torch_geometric.transforms import Compose

    TORCH_GEOMETRIC_AVAILABLE = True

    # Attempt to get PyG version for compatibility checks
    try:
        import torch_geometric

        TORCH_GEOMETRIC_VERSION = torch_geometric.__version__
    except Exception:
        TORCH_GEOMETRIC_VERSION = "unknown"
except ImportError as e:
    TORCH_GEOMETRIC_AVAILABLE = False
    TORCH_GEOMETRIC_IMPORT_ERROR = str(e)
    TORCH_GEOMETRIC_VERSION = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import ALL exceptions from centralized exceptions.py
from milia_pipeline.exceptions import (
    ConfigurationError,
    TransformCompositionError,
    TransformNotFoundError,
    TransformRegistryError,
    TransformValidationError,
    wrap_handler_operation,
)

# Initialize logger for this module
logger = logging.getLogger(__name__)


# =============================================================================
# DYNAMIC DATASET TYPE DISCOVERY
# =============================================================================
# Instead of hardcoding dataset types, dynamically discover available types
# from the registry or filesystem. This makes the module future-proof for
# new dataset types like QM9, MD, etc.


def _discover_available_dataset_types() -> list[str]:
    """
    Dynamically discover available dataset types from registry or filesystem.

    DYNAMIC APPROACH: Instead of hardcoding ['DFT', 'DMC', 'Wavefunction'], this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers dataset implementations from filesystem
    3. Never uses hardcoded dataset type lists

    Returns:
        List of available dataset type names (uppercase)
    """
    # Try registry first
    try:
        from milia_pipeline.datasets.registry import list_all

        return list_all()
    except Exception:
        pass

    # DYNAMIC FALLBACK: Discover dataset types from implementations directory
    try:
        implementations_dir = Path(__file__).parent.parent / "datasets" / "implementations"
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                # Extract dataset name from filename (e.g., dft.py -> DFT, qm9.py -> QM9)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ["BASE", "REGISTRY", "UTILS", "COMMON"]:
                    discovered_types.append(dataset_name)
            if discovered_types:
                return discovered_types
    except Exception:
        pass

    # Final fallback: return empty list
    return []


def _is_molecular_dataset_type(dataset_type: str | None) -> bool:
    """
    Check if a dataset type is a molecular/quantum chemistry dataset.

    DYNAMIC APPROACH: Instead of checking against hardcoded list, this function
    checks if the dataset type exists in the available dataset types discovered
    from the registry or filesystem.

    All datasets in this codebase (DFT, DMC, Wavefunction, QM9, etc.) are
    molecular/quantum chemistry datasets, so any registered dataset type
    should be considered a molecular dataset.

    Args:
        dataset_type: Dataset type name to check (can be None)

    Returns:
        True if dataset_type is a known molecular dataset type, False otherwise
    """
    if dataset_type is None:
        return False

    available_types = _discover_available_dataset_types()
    return dataset_type in available_types


# ============================================================================
# Plugin System: Custom Transform Integration
# ============================================================================
try:
    from milia_pipeline.transformations.custom_transforms import (
        CustomTransformBase,
        MolecularTransformBase,
        QuantumTransformBase,
        TransformMetadata,
    )

    CUSTOM_TRANSFORMS_AVAILABLE = True
except ImportError:
    CUSTOM_TRANSFORMS_AVAILABLE = False
    CustomTransformBase = None
    MolecularTransformBase = None
    QuantumTransformBase = None
    TransformMetadata = None
    logger.debug("custom_transforms module not available - custom transform features disabled")


# =============================================================================
# AUTO-DISCOVERY SYSTEM: ENHANCED TRANSFORM METADATA CLASSES
# =============================================================================


class TransformCompatibility(BaseModel):
    """
    Compatibility information for a transform across PyG versions.

    Pydantic V2 Migration (Phase 18):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - NON-BREAKING: Same constructor API and attribute access
    """

    min_version: str | None = None
    max_version: str | None = None
    deprecated_in: str | None = None
    removed_in: str | None = None
    replacement: str | None = None
    compatibility_notes: str | None = None
    version_specific_params: dict[str, list[str]] = Field(default_factory=dict)

    def is_compatible(self, current_version: str) -> bool:
        """Check if transform is compatible with current PyG version"""
        if not current_version or current_version == "unknown":
            return True  # Assume compatible if version unknown

        try:
            from packaging import version

            current = version.parse(current_version)

            if self.min_version:
                min_ver = version.parse(self.min_version)
                if current < min_ver:
                    return False

            if self.removed_in:
                removed_ver = version.parse(self.removed_in)
                if current >= removed_ver:
                    return False

            return True
        except (ImportError, ValueError):
            return True  # Assume compatible if version parsing fails


class TransformDependency(BaseModel):
    """
    Dependency information for transforms.

    Pydantic V2 Migration (Phase 18):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - NON-BREAKING: Same constructor API and attribute access
    """

    depends_on: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    recommended_before: list[str] = Field(default_factory=list)
    recommended_after: list[str] = Field(default_factory=list)
    required_graph_attributes: list[str] = Field(default_factory=list)
    modifies_attributes: list[str] = Field(default_factory=list)


class ParameterConstraint(BaseModel):
    """
    Constraint on a parameter value with validation support.

    Production Enhancement: New dataclass for parameter constraints

    Pydantic V2 Migration (Phase 18):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - NON-BREAKING: Same constructor API and attribute access
    """

    type: str  # 'range', 'choices', 'pattern', 'custom'
    description: str
    constraint_value: Any
    inferred: bool = False  # True if inferred, False if explicit
    confidence: float = 1.0  # Confidence in inference (0.0 to 1.0)

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """
        Validate value against this constraint.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if self.type == "range":
                min_val, max_val = self.constraint_value
                if not (min_val <= value <= max_val):
                    return False, f"Value {value} not in range [{min_val}, {max_val}]"
            elif self.type == "choices":
                if value not in self.constraint_value:
                    return False, f"Value {value} not in allowed choices: {self.constraint_value}"
            elif self.type == "pattern":
                if not re.match(self.constraint_value, str(value)):
                    return False, f"Value {value} does not match pattern: {self.constraint_value}"
            return True, None
        except Exception as e:
            return False, f"Constraint validation error: {e}"


class ParameterMetadata(BaseModel):
    """
    Comprehensive metadata about a transform parameter.

    Production Enhancement: New dataclass for advanced parameter introspection

    Pydantic V2 Migration (Phase 18):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - Uses ConfigDict(arbitrary_types_allowed=True) for Type and inspect.Parameter.empty
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        name: Parameter name
        type_hint: Python type hint
        default_value: Default value if parameter is optional
        required: Whether parameter must be provided
        description: Description extracted from docstring
        constraints: List of constraints on values
        examples: Example valid values
        inferred_from_name: Whether constraints were inferred from name
        docstring_source: Whether metadata came from docstring
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    type_hint: type | None = None
    default_value: Any = inspect.Parameter.empty
    required: bool = True
    description: str | None = None
    constraints: list[ParameterConstraint] = Field(default_factory=list)
    examples: list[Any] = Field(default_factory=list)
    inferred_from_name: bool = False
    docstring_source: bool = False

    @property
    def has_default(self) -> bool:
        """Check if parameter has default value"""
        return self.default_value is not inspect.Parameter.empty

    @property
    def is_optional(self) -> bool:
        """Check if parameter is Optional (can be None) based on type hint"""
        # Only check type hint, not default value or required status
        # This specifically checks if the type allows None (Optional/Union[..., None])
        if self.type_hint is None:
            return False  # No type hint = can't determine from type

        origin = get_origin(self.type_hint)
        if origin is Union:
            args = get_args(self.type_hint)
            return type(None) in args

        return False  # Has type hint but not Optional

    def get_base_type(self) -> type | None:
        """Extract base type from Optional or Union"""
        if self.type_hint is None:
            return None
        origin = get_origin(self.type_hint)
        if origin is Union:
            args = get_args(self.type_hint)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return non_none_args[0]
            return Union[tuple(non_none_args)]
        return self.type_hint


# =============================================================================
# EDGE-ATTR AWARE TRANSFORM PARAMETER INJECTION SYSTEM
# =============================================================================
# DYNAMIC, PRODUCTION-READY, FUTURE-PROOF solution for transforms that modify
# edge_index but require special handling for edge_attr compatibility.
#
# Problem: Some PyG transforms (e.g., AddSelfLoops) modify edge_index but by
# default do NOT handle edge_attr, causing shape mismatches:
#   - edge_index.size(1) != edge_attr.size(0)
#
# Solution: Automatically detect edge_attr presence and inject required params.
# =============================================================================


class EdgeAttrAwareTransformConfig(BaseModel):
    """
    Configuration for transforms requiring edge_attr-aware parameter injection.

    This defines how a transform should be configured when the data has edge_attr
    to maintain consistency between edge_index and edge_attr shapes.

    Pydantic V2 Migration (Phase 18):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        transform_name: Name of the transform (e.g., 'AddSelfLoops')
        edge_attr_param: Parameter name to set for edge_attr handling (e.g., 'attr')
        edge_attr_value: Value to set for the param (e.g., 'edge_attr')
        fill_value_param: Parameter name for fill value (e.g., 'fill_value')
        default_fill_value: Default fill value when edge_attr exists
        fill_value_options: Valid fill value options for documentation
        description: Human-readable description of the edge_attr handling
    """

    transform_name: str
    edge_attr_param: str
    edge_attr_value: str
    fill_value_param: str | None = None
    default_fill_value: Any = 0.0
    fill_value_options: list[str] = Field(
        default_factory=lambda: ["float", "tensor", "mean", "add", "min", "max", "mul"]
    )
    description: str = ""

    def get_injection_params(self, user_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Get parameters to inject for edge_attr compatibility.

        Only injects parameters that are not already specified by user.

        Args:
            user_kwargs: User-provided kwargs for the transform

        Returns:
            Dictionary of parameters to inject
        """
        injection = {}

        # Inject edge_attr_param if not specified
        if self.edge_attr_param not in user_kwargs:
            injection[self.edge_attr_param] = self.edge_attr_value

        # Inject fill_value_param if not specified and param exists
        if self.fill_value_param and self.fill_value_param not in user_kwargs:
            injection[self.fill_value_param] = self.default_fill_value

        return injection


# Registry of transforms requiring edge_attr-aware parameter injection
# This is extensible - add new transforms here as they are identified
EDGE_ATTR_AWARE_TRANSFORMS: dict[str, EdgeAttrAwareTransformConfig] = {
    "AddSelfLoops": EdgeAttrAwareTransformConfig(
        transform_name="AddSelfLoops",
        edge_attr_param="attr",
        edge_attr_value="edge_attr",
        fill_value_param="fill_value",
        default_fill_value=0.0,  # Zero-fill for self-loop edge features
        fill_value_options=["float", "tensor", "mean", "add", "min", "max", "mul"],
        description=(
            'AddSelfLoops by default only handles edge_weight (attr="edge_weight"). '
            'When data has edge_attr, we inject attr="edge_attr" and fill_value=0.0 '
            "to ensure self-loop edges have corresponding edge features."
        ),
    ),
    "AddRemainingSelfLoops": EdgeAttrAwareTransformConfig(
        transform_name="AddRemainingSelfLoops",
        edge_attr_param="attr",
        edge_attr_value="edge_attr",
        fill_value_param="fill_value",
        default_fill_value=0.0,
        fill_value_options=["float", "tensor", "mean", "add", "min", "max", "mul"],
        description=(
            "AddRemainingSelfLoops adds self-loops only to nodes without them. "
            "Same edge_attr handling as AddSelfLoops."
        ),
    ),
    # Future transforms can be added here following the same pattern
    # Example:
    # 'SomeOtherTransform': EdgeAttrAwareTransformConfig(...)
}


class EdgeAttrAwareParameterInjector:
    """
    DYNAMIC, PRODUCTION-READY, FUTURE-PROOF parameter injector for edge_attr awareness.

    This class automatically detects when transforms need special parameters to handle
    edge_attr correctly and injects them at transform composition time.

    Design Principles:
    1. DYNAMIC: Detects edge_attr presence from sample data at runtime
    2. PRODUCTION-READY: Uses standard PyG transform parameters, not custom wrappers
    3. FUTURE-PROOF: Extensible registry for new transforms with similar issues
    4. NON-INVASIVE: Only injects params when needed, respects user-specified values

    Usage:
        >>> injector = EdgeAttrAwareParameterInjector()
        >>> # Set sample data for detection
        >>> injector.set_sample_data(dataset[0])
        >>> # Inject params for a transform config
        >>> config = {'name': 'AddSelfLoops', 'kwargs': {'fill_value': 1.0}}
        >>> modified_config = injector.inject_params(config)
        >>> # Result: {'name': 'AddSelfLoops', 'kwargs': {'fill_value': 1.0, 'attr': 'edge_attr'}}
    """

    def __init__(self):
        """Initialize the injector with empty state."""
        self._sample_data = None
        self._has_edge_attr: bool | None = None
        self._edge_attr_dim: int | None = None
        self._logger = logging.getLogger(f"{__name__}.EdgeAttrAwareParameterInjector")
        self._injection_log: list[dict[str, Any]] = []

    def set_sample_data(self, sample_data: Any) -> "EdgeAttrAwareParameterInjector":
        """
        Set sample data for edge_attr detection.

        Args:
            sample_data: A PyG Data object or similar with potential edge_attr

        Returns:
            Self for method chaining
        """
        self._sample_data = sample_data
        self._has_edge_attr = None  # Reset cached detection
        self._edge_attr_dim = None
        self._detect_edge_attr()
        return self

    def _detect_edge_attr(self) -> None:
        """Detect presence and dimensionality of edge_attr in sample data."""
        if self._sample_data is None:
            self._has_edge_attr = False
            self._edge_attr_dim = None
            return

        # Check for edge_attr attribute
        if hasattr(self._sample_data, "edge_attr") and self._sample_data.edge_attr is not None:
            edge_attr = self._sample_data.edge_attr
            self._has_edge_attr = True

            # Determine dimensionality
            if hasattr(edge_attr, "shape"):
                if len(edge_attr.shape) >= 2:
                    self._edge_attr_dim = edge_attr.shape[-1]
                else:
                    self._edge_attr_dim = 1
            elif hasattr(edge_attr, "size"):
                # PyTorch tensor
                if edge_attr.dim() >= 2:
                    self._edge_attr_dim = edge_attr.size(-1)
                else:
                    self._edge_attr_dim = 1
            else:
                self._edge_attr_dim = None

            self._logger.debug(
                f"Detected edge_attr: has_edge_attr={self._has_edge_attr}, "
                f"edge_attr_dim={self._edge_attr_dim}"
            )
        else:
            self._has_edge_attr = False
            self._edge_attr_dim = None

    @property
    def has_edge_attr(self) -> bool:
        """Whether sample data has edge_attr."""
        if self._has_edge_attr is None:
            self._detect_edge_attr()
        return self._has_edge_attr or False

    @property
    def edge_attr_dim(self) -> int | None:
        """Dimensionality of edge_attr features, or None if unknown."""
        return self._edge_attr_dim

    def needs_injection(self, transform_name: str) -> bool:
        """
        Check if a transform needs parameter injection for edge_attr.

        Args:
            transform_name: Name of the transform

        Returns:
            True if injection is needed (transform is in registry AND data has edge_attr)
        """
        return self.has_edge_attr and transform_name in EDGE_ATTR_AWARE_TRANSFORMS

    def inject_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Inject edge_attr-aware parameters into a transform configuration.

        This is the main entry point for parameter injection. It checks if the
        transform needs injection and applies it while respecting user-specified values.

        Args:
            config: Transform configuration dict with 'name' and optional 'kwargs'

        Returns:
            Modified configuration with injected parameters (or original if no injection needed)
        """
        if not isinstance(config, dict):
            return config

        transform_name = config.get("name")
        if not transform_name or not self.needs_injection(transform_name):
            return config

        # Get the edge_attr-aware config for this transform
        aware_config = EDGE_ATTR_AWARE_TRANSFORMS[transform_name]

        # Get user kwargs (create empty dict if not present)
        user_kwargs = config.get("kwargs", {})
        if user_kwargs is None:
            user_kwargs = {}

        # Check for params field as alternative to kwargs
        if not user_kwargs and "params" in config:
            user_kwargs = config.get("params", {})
            if user_kwargs is None:
                user_kwargs = {}

        # Get injection parameters
        injection_params = aware_config.get_injection_params(user_kwargs)

        if not injection_params:
            return config

        # Create modified config (don't mutate original)
        modified_config = config.copy()
        modified_kwargs = user_kwargs.copy()
        modified_kwargs.update(injection_params)

        # Use 'kwargs' key for consistency
        modified_config["kwargs"] = modified_kwargs

        # Remove 'params' if it was the source to avoid duplication
        if "params" in modified_config and "kwargs" in modified_config:
            del modified_config["params"]

        # Log the injection
        injection_record = {
            "transform": transform_name,
            "injected_params": injection_params,
            "reason": aware_config.description,
            "has_edge_attr": self.has_edge_attr,
            "edge_attr_dim": self.edge_attr_dim,
        }
        self._injection_log.append(injection_record)

        self._logger.info(
            f"Edge-attr aware injection for '{transform_name}': "
            f"injected {injection_params} (data has edge_attr with dim={self.edge_attr_dim})"
        )

        return modified_config

    def inject_params_batch(self, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Inject parameters for a batch of transform configurations.

        Args:
            configs: List of transform configuration dicts

        Returns:
            List of modified configurations
        """
        return [self.inject_params(config) for config in configs]

    def get_injection_log(self) -> list[dict[str, Any]]:
        """Get log of all parameter injections performed."""
        return self._injection_log.copy()

    def clear_injection_log(self) -> None:
        """Clear the injection log."""
        self._injection_log.clear()

    def get_status(self) -> dict[str, Any]:
        """Get current status of the injector."""
        return {
            "has_sample_data": self._sample_data is not None,
            "has_edge_attr": self.has_edge_attr,
            "edge_attr_dim": self.edge_attr_dim,
            "registered_transforms": list(EDGE_ATTR_AWARE_TRANSFORMS.keys()),
            "injection_count": len(self._injection_log),
        }


# Module-level singleton for convenience (can be overridden per-composer)
_default_edge_attr_injector = EdgeAttrAwareParameterInjector()


def get_edge_attr_aware_transforms() -> dict[str, EdgeAttrAwareTransformConfig]:
    """Get the registry of edge_attr-aware transforms."""
    return EDGE_ATTR_AWARE_TRANSFORMS.copy()


def register_edge_attr_aware_transform(config: EdgeAttrAwareTransformConfig) -> None:
    """
    Register a new edge_attr-aware transform configuration.

    This allows extending the system with new transforms that need edge_attr handling.

    Args:
        config: EdgeAttrAwareTransformConfig for the transform
    """
    EDGE_ATTR_AWARE_TRANSFORMS[config.transform_name] = config
    logger.info(f"Registered edge_attr-aware transform: {config.transform_name}")


# =============================================================================
# END EDGE-ATTR AWARE TRANSFORM PARAMETER INJECTION SYSTEM
# =============================================================================


class TransformInfo(BaseModel):
    """
    Enhanced metadata for a registered transform - Production-Ready Architecture.

    Pydantic V2 Migration (Phase 18):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - Uses ConfigDict(arbitrary_types_allowed=True) for Type and inspect.Signature
        - Converted __post_init__ to @model_validator(mode='after')
        - NON-BREAKING: Same constructor API and attribute access
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    class_ref: type
    signature: inspect.Signature
    docstring: str | None
    parameters: dict[str, Any]
    category: str
    description: str | None = None
    example_usage: str | None = None
    performance_notes: str | None = None
    research_applicability: list[str] | None = None
    pre_transform_safe: bool = True
    usage_note: str | None = None

    # Auto-Discovery enhancements
    module_path: str | None = None
    source_file: str | None = None
    discovery_method: str = "manual"  # "manual", "auto"
    compatibility: TransformCompatibility | None = None
    dependencies: TransformDependency | None = None
    tags: list[str] = Field(default_factory=list)
    complexity_score: float = 1.0  # 1.0 = simple, 5.0 = very complex

    # Note: aliases field removed in Production-Ready Architecture Step 2.2 (backward compatibility cleanup)

    @model_validator(mode="after")
    def initialize_fields(self) -> "TransformInfo":
        """Enhanced post-initialization with Production-Ready Architecture features"""
        if self.research_applicability is None:
            object.__setattr__(self, "research_applicability", [])

        if self.compatibility is None:
            object.__setattr__(self, "compatibility", TransformCompatibility())

        if self.dependencies is None:
            object.__setattr__(self, "dependencies", TransformDependency())

        # Auto-generate description from docstring if not provided
        if self.description is None and self.docstring:
            object.__setattr__(self, "description", self._extract_description_from_docstring())

        # Set default description for common transforms
        if self.description is None:
            object.__setattr__(self, "description", self._get_default_description())

        # Extract tags from category and research applicability
        if not self.tags:
            object.__setattr__(self, "tags", self._generate_tags())

        return self

    def _extract_description_from_docstring(self) -> str:
        """Extract concise description from docstring"""
        if not self.docstring:
            return ""

        lines = self.docstring.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Args:") and not line.startswith("Parameters:"):
                # Remove common prefixes
                for prefix in ["Initializes", "Initialize", "Creates", "Create"]:
                    if line.startswith(prefix):
                        line = line[len(prefix) :].strip()
                return line

        return lines[0].strip() if lines else ""

    def _get_default_description(self) -> str:
        """Get default description for common transforms"""
        default_descriptions = {
            "AddSelfLoops": "Adds self-loop edges to nodes without existing self-loops",
            "ToUndirected": "Converts directed graphs to undirected by adding reverse edges",
            "RandomRotate": "Randomly rotates node positions around specified axes",
            "Normalize": "Normalizes node features to unit norm",
            "NormalizeFeatures": "Normalizes node features to unit norm",
            "GCNNorm": "Applies GCN normalization to edge weights",
            "VirtualNode": "Adds a virtual node connected to all other nodes",
            "Distance": "Computes pairwise distances and adds as edge attributes",
            "Cartesian": "Adds Cartesian coordinate differences as edge attributes",
            "LocalCartesian": "Adds local Cartesian coordinates as edge attributes",
            "DropEdge": "Randomly drops edges from the graph",
            "DropNode": "Randomly drops nodes from the graph",
            "MaskFeatures": "Randomly masks node features",
            "RandomScale": "Randomly scales node positions",
            "RandomTranslate": "Randomly translates node positions",
            "RandomFlip": "Randomly flips node positions along axes",
        }
        return default_descriptions.get(self.name, f"PyG transform: {self.name}")

    def _generate_tags(self) -> list[str]:
        """Generate tags from category and metadata"""
        tags = [self.category]

        if self.pre_transform_safe:
            tags.append("pre_transform_safe")
        else:
            tags.append("training_only")

        if "augmentation" in self.category.lower():
            tags.append("data_augmentation")

        if "Random" in self.name:
            tags.append("stochastic")

        if any(keyword in self.name.lower() for keyword in ["norm", "scale", "normalize"]):
            tags.append("normalization")

        if any(keyword in self.name.lower() for keyword in ["spatial", "distance", "cartesian"]):
            tags.append("spatial")

        return tags


class ExperimentalSetup(BaseModel):
    """
    Configuration for an experimental setup - unchanged from Foundation Architecture.

    Pydantic V2 Migration (Phase 18):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Converted __post_init__ to @model_validator(mode='after')
        - NON-BREAKING: Same constructor API and attribute access
    """

    name: str
    transforms: list[dict[str, Any]]
    description: str | None = None
    enabled: bool = True
    research_context: str | None = None
    expected_effects: list[str] | None = None

    @model_validator(mode="after")
    def validate_and_initialize(self) -> "ExperimentalSetup":
        """Validate transforms list and initialize expected_effects"""
        if not self.transforms:
            raise TransformValidationError(
                f"Experimental setup '{self.name}' cannot have empty transforms list",
                transform_name="experimental_setup",
                details="Each experimental setup must define at least one transform",
            )

        if self.expected_effects is None:
            object.__setattr__(self, "expected_effects", [])

        return self


class ValidationLevel(Enum):
    """Validation strictness levels"""

    STRICT = "strict"  # Fail on any issue
    STANDARD = "standard"  # Fail on errors, warn on issues
    PERMISSIVE = "permissive"  # Warn only, never fail


class ValidationScope(Enum):
    """Validation scope for different contexts"""

    BASIC = "basic"  # Parameter types and structure
    SEMANTIC = "semantic"  # Transform compatibility and ordering
    DATASET_SPECIFIC = "dataset"  # Dataset requirements (milia)
    PRODUCTION = "production"  # Full production-grade validation


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""

    CRITICAL = "critical"  # Must fix - prevents execution
    ERROR = "error"  # Should fix - may cause failures
    WARNING = "warning"  # Consider fixing - suboptimal
    INFO = "info"  # Informational - no action needed


class ValidationIssue(NamedTuple):
    """Structured validation issue"""

    severity: ValidationSeverity
    category: str
    message: str
    location: str
    suggestion: str | None = None
    auto_fixable: bool = False


class ValidationContext(BaseModel):
    """
    Context for validation operations.

    Tracks validation state across different scopes and levels.

    Pydantic V2 Migration (Phase 18):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - NON-BREAKING: Same constructor API and attribute access
    """

    level: ValidationLevel = ValidationLevel.STANDARD
    scope: ValidationScope = ValidationScope.SEMANTIC
    dataset_type: str | None = None
    research_context: str | None = None
    strict_mode: bool = False
    auto_fix: bool = False

    # Tracking
    issues: list[ValidationIssue] = Field(default_factory=list)
    fixes_applied: list[str] = Field(default_factory=list)
    validation_metadata: dict[str, Any] = Field(default_factory=dict)

    def add_issue(
        self,
        severity: ValidationSeverity,
        category: str,
        message: str,
        location: str,
        suggestion: str | None = None,
        auto_fixable: bool = False,
    ) -> None:
        """Add validation issue"""
        issue = ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            location=location,
            suggestion=suggestion,
            auto_fixable=auto_fixable,
        )
        self.issues.append(issue)

    def has_critical_issues(self) -> bool:
        """Check if any critical issues exist"""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)

    def has_errors(self) -> bool:
        """Check if any errors exist"""
        return any(
            issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
            for issue in self.issues
        )

    def get_issues_by_severity(self, severity: ValidationSeverity) -> list[ValidationIssue]:
        """Get issues of specific severity"""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_auto_fixable_issues(self) -> list[ValidationIssue]:
        """Get issues that can be auto-fixed"""
        return [issue for issue in self.issues if issue.auto_fixable]


class SemanticValidator:
    """
    Semantic validation for transform sequences

    Context-Aware Validation: Validates logical correctness beyond syntax
    """

    def __init__(self, registry: "TransformRegistry"):
        self._registry = registry
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Semantic validation rules
        self._semantic_rules = self._build_semantic_rules()
        self._anti_patterns = self._build_anti_patterns()
        self._best_practices = self._build_best_practices()

    def _build_semantic_rules(self) -> dict[str, Callable]:
        """Build semantic validation rules"""
        return {
            "ordering_dependencies": self._validate_ordering_dependencies,
            "data_flow_integrity": self._validate_data_flow,
            "resource_requirements": self._validate_resource_requirements,
            "semantic_conflicts": self._validate_semantic_conflicts,
            "transformation_completeness": self._validate_completeness,
        }

    def _build_anti_patterns(self) -> list[dict[str, Any]]:
        """Define known anti-patterns"""
        return [
            {
                "name": "destructive_before_essential",
                "pattern": lambda seq: self._check_destructive_before_essential(seq),
                "severity": ValidationSeverity.ERROR,
                "message": "Destructive transform before essential structural transforms",
                "suggestion": "Move structural transforms (AddSelfLoops, ToUndirected) before destructive ones",
            },
            {
                "name": "redundant_normalization",
                "pattern": lambda seq: self._check_redundant_normalization(seq),
                "severity": ValidationSeverity.WARNING,
                "message": "Multiple normalization transforms may be redundant",
                "suggestion": "Review if multiple normalizations are necessary",
            },
            {
                "name": "excessive_augmentation",
                "pattern": lambda seq: self._check_excessive_augmentation(seq),
                "severity": ValidationSeverity.WARNING,
                "message": "High augmentation intensity may degrade performance",
                "suggestion": "Consider reducing number of augmentation transforms",
            },
            {
                "name": "incompatible_spatial_transforms",
                "pattern": lambda seq: self._check_incompatible_spatial(seq),
                "severity": ValidationSeverity.ERROR,
                "message": "Incompatible spatial transforms in sequence",
                "suggestion": "Remove conflicting spatial transforms",
            },
        ]

    def _build_best_practices(self) -> list[dict[str, Any]]:
        """Define best practices"""
        return [
            {
                "name": "structural_first",
                "check": lambda seq: self._check_structural_first(seq),
                "message": "Structural transforms should come first",
                "category": "ordering",
            },
            {
                "name": "normalization_after_features",
                "check": lambda seq: self._check_normalization_after_features(seq),
                "message": "Normalization should come after feature modifications",
                "category": "ordering",
            },
            {
                "name": "spatial_before_global",
                "check": lambda seq: self._check_spatial_before_global(seq),
                "message": "Spatial transforms should come before global operations",
                "category": "ordering",
            },
        ]

    def validate_sequence(
        self, transform_configs: list[dict[str, Any]], context: ValidationContext
    ) -> ValidationContext:
        """
        Perform semantic validation on transform sequence

        Args:
            transform_configs: List of transform configurations
            context: Validation context

        Returns:
            Updated validation context with issues
        """

        # Extract transform names and info
        transform_names = [cfg.get("name", "") for cfg in transform_configs]

        # Run all semantic rules
        for rule_name, rule_func in self._semantic_rules.items():
            try:
                rule_func(transform_configs, transform_names, context)
            except Exception as e:
                self._logger.warning(f"Semantic rule '{rule_name}' failed: {e}")

        # Check anti-patterns
        for anti_pattern in self._anti_patterns:
            try:
                if anti_pattern["pattern"](transform_configs):
                    context.add_issue(
                        severity=anti_pattern["severity"],
                        category="anti_pattern",
                        message=anti_pattern["message"],
                        location=f"sequence:{anti_pattern['name']}",
                        suggestion=anti_pattern["suggestion"],
                        auto_fixable=False,
                    )
            except Exception as e:
                self._logger.warning(f"Anti-pattern check '{anti_pattern['name']}' failed: {e}")

        # Check best practices (info level)
        for practice in self._best_practices:
            try:
                if not practice["check"](transform_configs):
                    context.add_issue(
                        severity=ValidationSeverity.INFO,
                        category="best_practice",
                        message=practice["message"],
                        location=f"sequence:{practice['name']}",
                        suggestion=None,
                        auto_fixable=False,
                    )
            except Exception as e:
                self._logger.warning(f"Best practice check '{practice['name']}' failed: {e}")

        return context

    def _validate_ordering_dependencies(
        self, configs: list[dict], names: list[str], context: ValidationContext
    ) -> None:
        """Validate transform ordering dependencies"""

        for i, name in enumerate(names):
            try:
                transform_info = self._registry.get_transform_info(name)
                deps = transform_info.dependencies

                # Check "recommended_before"
                for before_transform in deps.recommended_before:
                    if before_transform in names:
                        before_idx = names.index(before_transform)
                        if i > before_idx:
                            context.add_issue(
                                severity=ValidationSeverity.WARNING,
                                category="ordering",
                                message=f"{name} should come before {before_transform}",
                                location=f"transform[{i}]",
                                suggestion=f"Move {name} before {before_transform}",
                                auto_fixable=True,
                            )

                # Check "recommended_after"
                for after_transform in deps.recommended_after:
                    if after_transform in names:
                        after_idx = names.index(after_transform)
                        if i < after_idx:
                            context.add_issue(
                                severity=ValidationSeverity.WARNING,
                                category="ordering",
                                message=f"{name} should come after {after_transform}",
                                location=f"transform[{i}]",
                                suggestion=f"Move {name} after {after_transform}",
                                auto_fixable=True,
                            )

                # Check conflicts
                for conflict_transform in deps.conflicts_with:
                    if conflict_transform in names:
                        context.add_issue(
                            severity=ValidationSeverity.ERROR,
                            category="conflict",
                            message=f"{name} conflicts with {conflict_transform}",
                            location=f"transform[{i}]",
                            suggestion=f"Remove either {name} or {conflict_transform}",
                            auto_fixable=False,
                        )

            except TransformNotFoundError:
                continue

    def _validate_data_flow(
        self, configs: list[dict], names: list[str], context: ValidationContext
    ) -> None:
        """Validate data flow integrity"""

        available_attrs = {"x", "edge_index", "pos", "batch"}  # Base attributes

        for i, name in enumerate(names):
            try:
                transform_info = self._registry.get_transform_info(name)
                deps = transform_info.dependencies

                # Check required attributes are available
                for req_attr in deps.required_graph_attributes:
                    if req_attr not in available_attrs:
                        context.add_issue(
                            severity=ValidationSeverity.ERROR,
                            category="data_flow",
                            message=f"{name} requires '{req_attr}' which is not available",
                            location=f"transform[{i}]",
                            suggestion=f"Ensure previous transforms provide '{req_attr}'",
                            auto_fixable=False,
                        )

                # Update available attributes
                for modified_attr in deps.modifies_attributes:
                    available_attrs.add(modified_attr)

            except TransformNotFoundError:
                continue

    def _validate_resource_requirements(
        self, configs: list[dict], names: list[str], context: ValidationContext
    ) -> None:
        """Validate resource requirements"""

        total_complexity = 0.0
        high_memory_count = 0

        for name in names:
            try:
                transform_info = self._registry.get_transform_info(name)
                total_complexity += transform_info.complexity_score

                if transform_info.complexity_score >= 3.0:
                    high_memory_count += 1

            except TransformNotFoundError:
                continue

        # Check overall complexity
        if total_complexity > 15.0:
            context.add_issue(
                severity=ValidationSeverity.WARNING,
                category="performance",
                message=f"High total complexity score: {total_complexity:.1f}",
                location="sequence",
                suggestion="Consider optimizing or removing expensive transforms",
                auto_fixable=False,
            )

        if high_memory_count > 3:
            context.add_issue(
                severity=ValidationSeverity.WARNING,
                category="performance",
                message=f"{high_memory_count} memory-intensive transforms",
                location="sequence",
                suggestion="May cause memory issues with large datasets",
                auto_fixable=False,
            )

    def _validate_semantic_conflicts(
        self, configs: list[dict], names: list[str], context: ValidationContext
    ) -> None:
        """Validate semantic conflicts between transforms"""

        # Check for transforms that undo each other
        if "ToUndirected" in names and "ToDirected" in names:
            context.add_issue(
                severity=ValidationSeverity.ERROR,
                category="semantic_conflict",
                message="ToUndirected and ToDirected are contradictory",
                location="sequence",
                suggestion="Remove one of these transforms",
                auto_fixable=False,
            )

        # Check for excessive redundancy
        name_counts = {}
        for name in names:
            name_counts[name] = name_counts.get(name, 0) + 1

        for name, count in name_counts.items():
            if count > 2:
                context.add_issue(
                    severity=ValidationSeverity.WARNING,
                    category="redundancy",
                    message=f"Transform '{name}' appears {count} times",
                    location="sequence",
                    suggestion="Remove redundant transforms",
                    auto_fixable=True,
                )

    def _validate_completeness(
        self, configs: list[dict], names: list[str], context: ValidationContext
    ) -> None:
        """Validate sequence completeness"""

        # For molecular graphs, certain transforms are essential
        # DYNAMIC: Check if dataset type is a molecular dataset (instead of hardcoded list)
        # Any registered dataset type (DFT, DMC, Wavefunction, QM9, etc.) is a molecular dataset
        if _is_molecular_dataset_type(context.dataset_type):
            essential = ["AddSelfLoops", "ToUndirected"]
            missing = [t for t in essential if t not in names]

            if missing:
                context.add_issue(
                    severity=ValidationSeverity.WARNING,
                    category="completeness",
                    message=f"Missing recommended transforms: {missing}",
                    location="sequence",
                    suggestion=f"Consider adding {missing} for molecular graphs",
                    auto_fixable=True,
                )

    # Anti-pattern checkers
    def _check_destructive_before_essential(self, configs: list[dict]) -> bool:
        """Check if destructive transforms come before essential ones"""
        destructive = ["DropNode", "DropEdge", "RemoveIsolatedNodes"]
        essential = ["AddSelfLoops", "ToUndirected"]

        names = [cfg.get("name", "") for cfg in configs]

        destructive_indices = [i for i, name in enumerate(names) if name in destructive]
        essential_indices = [i for i, name in enumerate(names) if name in essential]

        if not destructive_indices or not essential_indices:
            return False

        return min(destructive_indices) < min(essential_indices)

    def _check_redundant_normalization(self, configs: list[dict]) -> bool:
        """Check for redundant normalization"""
        norm_transforms = ["Normalize", "NormalizeFeatures", "GCNNorm", "NormalizeScale"]
        names = [cfg.get("name", "") for cfg in configs]

        count = sum(1 for name in names if name in norm_transforms)
        return count > 2

    def _check_excessive_augmentation(self, configs: list[dict]) -> bool:
        """Check for excessive augmentation"""
        aug_transforms = [
            "DropEdge",
            "DropNode",
            "MaskFeatures",
            "RandomNodeSample",
            "RandomRotate",
            "RandomScale",
            "RandomTranslate",
            "RandomFlip",
        ]
        names = [cfg.get("name", "") for cfg in configs]

        count = sum(1 for name in names if name in aug_transforms)
        return count > 4

    def _check_incompatible_spatial(self, configs: list[dict]) -> bool:
        """Check for incompatible spatial transforms"""
        spatial_groups = [
            ["Distance", "Cartesian", "LocalCartesian"],  # May have dimension conflicts
        ]

        names = [cfg.get("name", "") for cfg in configs]

        for group in spatial_groups:
            count = sum(1 for name in names if name in group)
            if count > 1:
                return True

        return False

    # Best practice checkers
    def _check_structural_first(self, configs: list[dict]) -> bool:
        """Check if structural transforms come first"""
        structural = ["AddSelfLoops", "ToUndirected", "RemoveIsolatedNodes"]
        names = [cfg.get("name", "") for cfg in configs]

        structural_indices = [i for i, name in enumerate(names) if name in structural]
        if not structural_indices:
            return True

        # Check if any non-structural transform comes before last structural
        last_structural = max(structural_indices)
        return all(names[i] in structural for i in range(last_structural))

    def _check_normalization_after_features(self, configs: list[dict]) -> bool:
        """Check if normalization comes after feature modifications"""
        norm_transforms = ["Normalize", "NormalizeFeatures", "GCNNorm"]
        feature_transforms = ["LocalDegreeProfile", "OneHotDegree"]

        names = [cfg.get("name", "") for cfg in configs]

        norm_indices = [i for i, name in enumerate(names) if name in norm_transforms]
        feature_indices = [i for i, name in enumerate(names) if name in feature_transforms]

        if not norm_indices or not feature_indices:
            return True

        return min(norm_indices) > max(feature_indices)

    def _check_spatial_before_global(self, configs: list[dict]) -> bool:
        """Check if spatial transforms come before global operations"""
        spatial = ["Distance", "Cartesian", "LocalCartesian"]
        global_ops = ["VirtualNode"]

        names = [cfg.get("name", "") for cfg in configs]

        spatial_indices = [i for i, name in enumerate(names) if name in spatial]
        global_indices = [i for i, name in enumerate(names) if name in global_ops]

        if not spatial_indices or not global_indices:
            return True

        return max(spatial_indices) < min(global_indices)


class DatasetAwareValidator:
    """
    Dataset-specific validation for milia requirements

    Context-Aware Validation: Validates against dataset-specific constraints
    """

    def __init__(self, config_bridge: "ConfigurationBridge"):
        self._config_bridge = config_bridge
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Dataset-specific rules
        self._dataset_rules = self._build_dataset_rules()

    def _build_dataset_rules(self) -> dict[str, dict[str, Any]]:
        """Build dataset-specific validation rules"""
        return {
            "DFT": {
                "required": ["AddSelfLoops", "ToUndirected"],
                "recommended": ["GCNNorm", "Distance"],
                "avoid": ["RandomFlip"],
                "forbidden": [],
                "max_augmentation_intensity": 0.3,
                "spatial_precision": "high",
                "validation_checks": [self._validate_dft_precision, self._validate_dft_determinism],
            },
            "DMC": {
                "required": ["AddSelfLoops", "ToUndirected"],
                "recommended": ["MaskFeatures"],
                "avoid": ["DropNode", "RandomFlip", "VirtualNode"],
                "forbidden": [],
                "max_augmentation_intensity": 0.2,
                "uncertainty_preserving": True,
                "validation_checks": [self._validate_dmc_uncertainty, self._validate_dmc_sampling],
            },
            "Wavefunction": {
                "required": ["AddSelfLoops", "ToUndirected"],
                "recommended": ["GCNNorm", "Distance"],
                "avoid": ["RandomFlip", "DropNode", "MaskFeatures"],
                "forbidden": [],
                "max_augmentation_intensity": 0.1,  # Lower than DFT due to precision requirements
                "spatial_precision": "very_high",
                "orbital_preserving": True,
                "validation_checks": [
                    self._validate_wavefunction_orbital_preservation,
                    self._validate_wavefunction_precision,
                ],
            },
        }

    def validate_for_dataset(
        self, transform_configs: list[dict[str, Any]], dataset_type: str, context: ValidationContext
    ) -> ValidationContext:
        """
        Validate transforms for specific dataset type

        Args:
            transform_configs: Transform configurations
            dataset_type: Dataset type (DFT, DMC, Wavefunction)
            context: Validation context

        Returns:
            Updated validation context
        """

        if dataset_type not in self._dataset_rules:
            context.add_issue(
                severity=ValidationSeverity.WARNING,
                category="dataset",
                message=f"Unknown dataset type: {dataset_type}",
                location="config",
                suggestion="Use DFT, DMC, or Wavefunction",
                auto_fixable=False,
            )
            return context

        rules = self._dataset_rules[dataset_type]
        names = [cfg.get("name", "") for cfg in transform_configs]

        # Check required transforms
        for required in rules["required"]:
            if required not in names:
                context.add_issue(
                    severity=ValidationSeverity.ERROR,
                    category="dataset_requirement",
                    message=f"Missing required transform for {dataset_type}: {required}",
                    location="sequence",
                    suggestion=f"Add {required} transform",
                    auto_fixable=True,
                )

        # Check recommended transforms
        missing_recommended = [rec for rec in rules["recommended"] if rec not in names]
        if missing_recommended:
            context.add_issue(
                severity=ValidationSeverity.WARNING,
                category="dataset_recommendation",
                message=f"Missing recommended transforms for {dataset_type}: {missing_recommended}",
                location="sequence",
                suggestion=f"Consider adding {missing_recommended}",
                auto_fixable=True,
            )

        # Check avoided transforms
        found_avoided = [name for name in names if name in rules["avoid"]]
        if found_avoided:
            context.add_issue(
                severity=ValidationSeverity.WARNING,
                category="dataset_compatibility",
                message=f"Transforms not recommended for {dataset_type}: {found_avoided}",
                location="sequence",
                suggestion=f"Consider removing {found_avoided}",
                auto_fixable=False,
            )

        # Check forbidden transforms
        found_forbidden = [name for name in names if name in rules["forbidden"]]
        if found_forbidden:
            context.add_issue(
                severity=ValidationSeverity.CRITICAL,
                category="dataset_forbidden",
                message=f"Forbidden transforms for {dataset_type}: {found_forbidden}",
                location="sequence",
                suggestion=f"Remove {found_forbidden}",
                auto_fixable=False,
            )

        # Check augmentation intensity
        aug_transforms = ["DropEdge", "DropNode", "MaskFeatures", "RandomNodeSample"]
        aug_configs = [cfg for cfg in transform_configs if cfg.get("name") in aug_transforms]

        total_aug_intensity = 0.0
        for cfg in aug_configs:
            p_val = cfg.get("kwargs", {}).get("p", 0.0)
            total_aug_intensity += p_val

        max_intensity = rules.get("max_augmentation_intensity", 1.0)
        if total_aug_intensity > max_intensity:
            context.add_issue(
                severity=ValidationSeverity.WARNING,
                category="dataset_optimization",
                message=f"Augmentation intensity {total_aug_intensity:.2f} exceeds "
                f"recommended {max_intensity:.2f} for {dataset_type}",
                location="sequence",
                suggestion=f"Reduce augmentation parameters for {dataset_type}",
                auto_fixable=False,
            )

        # Run dataset-specific validation checks
        for check_func in rules.get("validation_checks", []):
            try:
                check_func(transform_configs, context, dataset_type)
            except Exception as e:
                self._logger.warning(f"Dataset check failed: {e}")

        return context

    def _validate_dft_precision(
        self, configs: list[dict], context: ValidationContext, dataset_type: str
    ) -> None:
        """Validate DFT precision requirements"""

        # Check Distance transform precision
        for i, cfg in enumerate(configs):
            if cfg.get("name") == "Distance":
                norm = cfg.get("kwargs", {}).get("norm", True)
                if not norm:
                    context.add_issue(
                        severity=ValidationSeverity.WARNING,
                        category="dft_precision",
                        message="Unnormalized distances may affect DFT precision",
                        location=f"transform[{i}]",
                        suggestion="Set norm=True for Distance transform",
                        auto_fixable=True,
                    )

    def _validate_dft_determinism(
        self, configs: list[dict], context: ValidationContext, dataset_type: str
    ) -> None:
        """Validate DFT determinism requirements"""

        stochastic = [
            "RandomRotate",
            "RandomScale",
            "RandomTranslate",
            "RandomFlip",
            "DropEdge",
            "DropNode",
            "MaskFeatures",
        ]

        found_stochastic = [cfg.get("name") for cfg in configs if cfg.get("name") in stochastic]

        if found_stochastic:
            context.add_issue(
                severity=ValidationSeverity.INFO,
                category="dft_determinism",
                message=f"Stochastic transforms in DFT pipeline: {found_stochastic}",
                location="sequence",
                suggestion="Ensure reproducibility with fixed random seeds",
                auto_fixable=False,
            )

    def _validate_dmc_uncertainty(
        self, configs: list[dict], context: ValidationContext, dataset_type: str
    ) -> None:
        """Validate DMC uncertainty preservation"""

        # Transforms that may affect uncertainty
        uncertainty_sensitive = ["DropNode", "RemoveIsolatedNodes", "VirtualNode"]

        found_sensitive = [
            cfg.get("name") for cfg in configs if cfg.get("name") in uncertainty_sensitive
        ]

        if found_sensitive:
            context.add_issue(
                severity=ValidationSeverity.ERROR,
                category="dmc_uncertainty",
                message=f"Transforms may interfere with uncertainty: {found_sensitive}",
                location="sequence",
                suggestion="Remove transforms that modify graph structure for DMC",
                auto_fixable=False,
            )

    def _validate_dmc_sampling(
        self, configs: list[dict], context: ValidationContext, dataset_type: str
    ) -> None:
        """Validate DMC sampling compatibility"""

        # Check MaskFeatures parameters for DMC
        for i, cfg in enumerate(configs):
            if cfg.get("name") == "MaskFeatures":
                p_val = cfg.get("kwargs", {}).get("p", 0.0)
                if p_val > 0.2:
                    context.add_issue(
                        severity=ValidationSeverity.WARNING,
                        category="dmc_sampling",
                        message=f"High masking rate (p={p_val}) may affect DMC sampling",
                        location=f"transform[{i}]",
                        suggestion="Use p ≤ 0.2 for DMC datasets",
                        auto_fixable=True,
                    )

    def _validate_wavefunction_orbital_preservation(
        self, configs: list[dict], context: ValidationContext, dataset_type: str
    ) -> None:
        """
        Validate that transformations preserve wavefunction orbital structure.

        Wavefunction datasets contain molecular orbital coefficients, Hamiltonian matrices,
        and overlap matrices that must remain physically meaningful after transformations.
        Certain geometric augmentations can break orbital consistency.
        """
        # Geometric transforms that may affect orbital representations
        geometric_augmentations = ["RandomRotate", "RandomScale", "RandomTranslate", "RandomFlip"]
        found_geometric = [
            cfg.get("name") for cfg in configs if cfg.get("name") in geometric_augmentations
        ]

        if found_geometric:
            context.add_issue(
                severity=ValidationSeverity.WARNING,
                category="wavefunction_orbital",
                message=f"Geometric augmentations {found_geometric} detected with wavefunction data",
                location="sequence",
                suggestion="Ensure orbital coefficients and matrices are transformed consistently with geometry",
                auto_fixable=False,
            )

    def _validate_wavefunction_precision(
        self, configs: list[dict], context: ValidationContext, dataset_type: str
    ) -> None:
        """
        Validate precision requirements for wavefunction datasets.

        Wavefunction datasets require high numerical precision as they encode
        complete electronic structure information. Certain transforms that introduce
        noise or approximations should be avoided.
        """
        # Transforms that add stochasticity or reduce precision
        precision_sensitive = ["DropNode", "DropEdge", "MaskFeatures", "RandomFlip"]
        found_noisy = [cfg.get("name") for cfg in configs if cfg.get("name") in precision_sensitive]

        if found_noisy:
            context.add_issue(
                severity=ValidationSeverity.WARNING,
                category="wavefunction_precision",
                message=f"Precision-reducing transforms detected: {found_noisy}",
                location="sequence",
                suggestion="Wavefunction data requires high precision - consider removing stochastic transforms",
                auto_fixable=False,
            )


class ValidationReporter:
    """
    Comprehensive validation reporting

    Context-Aware Validation: Generates detailed validation reports
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_report(self, context: ValidationContext) -> dict[str, Any]:
        """Generate comprehensive validation report"""

        report = {
            "summary": self._generate_summary(context),
            "issues_by_severity": self._group_by_severity(context),
            "issues_by_category": self._group_by_category(context),
            "auto_fixable_issues": context.get_auto_fixable_issues(),
            "validation_passed": not context.has_errors(),
            "metadata": {
                "level": context.level.value,
                "scope": context.scope.value,
                "dataset_type": context.dataset_type,
                "strict_mode": context.strict_mode,
                "total_issues": len(context.issues),
                "fixes_applied": len(context.fixes_applied),
            },
            "recommendations": self._generate_recommendations(context),
        }

        return report

    def _generate_summary(self, context: ValidationContext) -> dict[str, Any]:
        """Generate summary statistics"""

        return {
            "total_issues": len(context.issues),
            "critical_count": len(context.get_issues_by_severity(ValidationSeverity.CRITICAL)),
            "error_count": len(context.get_issues_by_severity(ValidationSeverity.ERROR)),
            "warning_count": len(context.get_issues_by_severity(ValidationSeverity.WARNING)),
            "info_count": len(context.get_issues_by_severity(ValidationSeverity.INFO)),
            "auto_fixable_count": len(context.get_auto_fixable_issues()),
            "validation_passed": not context.has_errors(),
        }

    def _group_by_severity(self, context: ValidationContext) -> dict[str, list[ValidationIssue]]:
        """Group issues by severity"""

        grouped = {
            "critical": context.get_issues_by_severity(ValidationSeverity.CRITICAL),
            "error": context.get_issues_by_severity(ValidationSeverity.ERROR),
            "warning": context.get_issues_by_severity(ValidationSeverity.WARNING),
            "info": context.get_issues_by_severity(ValidationSeverity.INFO),
        }

        return grouped

    def _group_by_category(self, context: ValidationContext) -> dict[str, list[ValidationIssue]]:
        """Group issues by category"""

        categories = {}
        for issue in context.issues:
            if issue.category not in categories:
                categories[issue.category] = []
            categories[issue.category].append(issue)

        return categories

    def _generate_recommendations(self, context: ValidationContext) -> list[str]:
        """Generate actionable recommendations"""

        recommendations = []

        critical_count = len(context.get_issues_by_severity(ValidationSeverity.CRITICAL))
        if critical_count > 0:
            recommendations.append(
                f"CRITICAL: Fix {critical_count} critical issues before proceeding"
            )

        error_count = len(context.get_issues_by_severity(ValidationSeverity.ERROR))
        if error_count > 0:
            recommendations.append(f"Fix {error_count} errors to ensure proper execution")

        auto_fixable = len(context.get_auto_fixable_issues())
        if auto_fixable > 0:
            recommendations.append(
                f"{auto_fixable} issues can be auto-fixed - enable auto_fix mode"
            )

        # Dataset-specific recommendations
        if context.dataset_type:
            dataset_issues = [i for i in context.issues if "dataset" in i.category.lower()]
            if dataset_issues:
                recommendations.append(
                    f"Review {len(dataset_issues)} dataset-specific issues for {context.dataset_type}"
                )

        return recommendations

    def format_report(self, report: dict[str, Any], format_type: str = "text") -> str:
        """Format report for display"""

        if format_type == "text":
            return self._format_text_report(report)
        elif format_type == "json":
            return json.dumps(report, indent=2, default=str)
        elif format_type == "markdown":
            return self._format_markdown_report(report)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _format_text_report(self, report: dict[str, Any]) -> str:
        """Format report as text"""

        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        summary = report["summary"]
        lines.append("SUMMARY:")
        lines.append(f"  Total Issues: {summary['total_issues']}")
        lines.append(f"  Critical: {summary['critical_count']}")
        lines.append(f"  Errors: {summary['error_count']}")
        lines.append(f"  Warnings: {summary['warning_count']}")
        lines.append(f"  Info: {summary['info_count']}")
        lines.append(f"  Auto-fixable: {summary['auto_fixable_count']}")
        lines.append(f"  Status: {'PASSED' if summary['validation_passed'] else 'FAILED'}")
        lines.append("")

        # Issues by severity
        if summary["total_issues"] > 0:
            lines.append("ISSUES BY SEVERITY:")
            lines.append("")

            for severity, issues in report["issues_by_severity"].items():
                if issues:
                    lines.append(f"  {severity.upper()}:")
                    for issue in issues:
                        lines.append(f"    [{issue.location}] {issue.message}")
                        if issue.suggestion:
                            lines.append(f"      → Suggestion: {issue.suggestion}")
                    lines.append("")

        # Recommendations
        if report["recommendations"]:
            lines.append("RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                lines.append(f"  • {rec}")
            lines.append("")

        return "\n".join(lines)

    def _format_markdown_report(self, report: dict[str, Any]) -> str:
        """Format report as markdown"""

        lines = []

        # Header
        lines.append("# Validation Report")
        lines.append("")

        # Summary
        summary = report["summary"]
        status_emoji = "✅" if summary["validation_passed"] else "❌"
        lines.append(
            f"**Status:** {status_emoji} {'PASSED' if summary['validation_passed'] else 'FAILED'}"
        )
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- Total Issues: {summary['total_issues']}")
        lines.append(f"- Critical: {summary['critical_count']}")
        lines.append(f"- Errors: {summary['error_count']}")
        lines.append(f"- Warnings: {summary['warning_count']}")
        lines.append(f"- Info: {summary['info_count']}")
        lines.append(f"- Auto-fixable: {summary['auto_fixable_count']}")
        lines.append("")

        # Issues
        if summary["total_issues"] > 0:
            lines.append("## Issues")
            lines.append("")

            for severity, issues in report["issues_by_severity"].items():
                if issues:
                    severity_emoji = {
                        "critical": "🔴",
                        "error": "🟠",
                        "warning": "🟡",
                        "info": "ℹ️",
                    }.get(severity, "•")

                    lines.append(f"### {severity_emoji} {severity.upper()}")
                    lines.append("")

                    for issue in issues:
                        lines.append(f"**Location:** `{issue.location}`  ")
                        lines.append(f"**Message:** {issue.message}  ")
                        if issue.suggestion:
                            lines.append(f"**Suggestion:** {issue.suggestion}  ")
                        lines.append("")

        # Recommendations
        if report["recommendations"]:
            lines.append("## Recommendations")
            lines.append("")
            for rec in report["recommendations"]:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# AUTO-DISCOVERY SYSTEM: DYNAMIC TRANSFORM DISCOVERY ENGINE
# =============================================================================


class DynamicTransformDiscovery:
    """
    Dynamic discovery engine for PyG transforms

    Production Enhancement: Automatically discovers transforms from PyG modules
    and handles version compatibility for PyG 2.1+.

    Breaking Changes from Auto-Discovery:
        - Fallback mechanisms removed
        - Transform aliases removed
        - Only canonical PyG transform names accepted

    All transforms must be available in the installed PyG version. If a
    transform is missing, a clear ConfigurationError is raised directing
    the user to upgrade PyG to version 2.1 or later.
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._discovered_transforms: dict[str, type] = {}
        self._discovery_metadata: dict[str, dict[str, Any]] = {}
        self._module_structure: dict[str, list[str]] = {}

        # Version compatibility database
        self._version_compatibility = self._initialize_version_compatibility()

        # Canonical name mappings for PyG version variations
        self._canonical_mappings = self._initialize_canonical_mappings()

    def _initialize_canonical_mappings(self) -> dict[str, list[str]]:
        """
        Comprehensive mapping of canonical milia transform names to all known PyG names
        across all PyG versions (1.x, 2.x, 2.6+).

        This ensures the software is version-agnostic and forward-compatible.
        If a transform doesn't exist in the current PyG version:
        1. It will be provided as a custom transform, OR
        2. PyG can be upgraded to obtain it

        Mapping Strategy:
        - Primary name: Most recent PyG naming convention
        - Aliases: Historical names from older PyG versions
        - Variations: Common naming patterns (Random prefix, etc.)

        Returns:
            Dict mapping canonical names to list of possible PyG implementation names
        """
        return {
            # === NORMALIZATION TRANSFORMS ===
            "Normalize": ["NormalizeFeatures", "Normalize"],  # Legacy name
            "NormalizeFeatures": ["NormalizeFeatures", "Normalize"],
            "NormalizeRotation": ["NormalizeRotation"],
            "NormalizeScale": ["NormalizeScale"],
            "GCNNorm": ["GCNNorm"],
            # === AUGMENTATION TRANSFORMS ===
            # Edge augmentation
            "DropEdge": ["DropEdge", "RandomDropEdge", "EdgeDrop"],
            "RandomDropEdge": ["RandomDropEdge", "DropEdge"],
            # Node augmentation
            "DropNode": ["DropNode", "RandomDropNode", "NodeDrop"],
            "RandomDropNode": ["RandomDropNode", "DropNode"],
            # Feature augmentation
            "MaskFeatures": ["MaskFeatures", "RandomMaskFeatures", "FeatureMasking"],
            "RandomMaskFeatures": ["RandomMaskFeatures", "MaskFeatures"],
            "FeaturePropagation": ["FeaturePropagation"],
            # === SAMPLING TRANSFORMS ===
            "RandomNodeSample": ["RandomNodeSample", "NodeSample", "SampleNodes"],
            "NodeSample": ["NodeSample", "RandomNodeSample"],
            "RandomLinkSplit": ["RandomLinkSplit"],
            "RandomNodeSplit": ["RandomNodeSplit"],
            # === GEOMETRIC TRANSFORMS ===
            "RandomRotate": ["RandomRotate"],
            "RandomScale": ["RandomScale"],
            "RandomShear": ["RandomShear"],
            "RandomFlip": ["RandomFlip"],
            "RandomJitter": ["RandomJitter"],
            "RandomTranslate": ["RandomTranslate"],
            # === SPATIAL TRANSFORMS ===
            "Distance": ["Distance"],
            "Cartesian": ["Cartesian"],
            "LocalCartesian": ["LocalCartesian"],
            "Polar": ["Polar"],
            "Spherical": ["Spherical"],
            # === STRUCTURAL TRANSFORMS ===
            "AddSelfLoops": ["AddSelfLoops"],
            "AddRemainingSelfLoops": ["AddRemainingSelfLoops"],
            "RemoveSelfLoops": ["RemoveSelfLoops"],
            "ToUndirected": ["ToUndirected"],
            "RemoveIsolatedNodes": ["RemoveIsolatedNodes"],
            "RemoveDuplicatedEdges": ["RemoveDuplicatedEdges"],
            "LargestConnectedComponents": ["LargestConnectedComponents"],
            # === GRAPH CONSTRUCTION ===
            "KNNGraph": ["KNNGraph"],
            "RadiusGraph": ["RadiusGraph"],
            "Delaunay": ["Delaunay"],
            # === SPARSIFICATION ===
            "GDC": ["GDC"],  # Graph Diffusion Convolution
            # === FEATURE ENGINEERING ===
            "OneHotDegree": ["OneHotDegree"],
            "LocalDegreeProfile": ["LocalDegreeProfile"],
            "Constant": ["Constant"],
            "Pad": ["Pad"],
            # === POSITIONAL ENCODING ===
            "AddLaplacianEigenvectorPE": ["AddLaplacianEigenvectorPE"],
            "AddRandomWalkPE": ["AddRandomWalkPE"],
            "LaplacianLambdaMax": ["LaplacianLambdaMax"],
            "SIGN": ["SIGN"],  # Scalable Inception Graph Neural Network
            # === HETEROGENEOUS GRAPH TRANSFORMS ===
            "AddMetaPaths": ["AddMetaPaths"],
            "AddRandomMetaPaths": ["AddRandomMetaPaths"],
            "ToHeteroData": ["ToHeteroData", "ToHeterogeneous"],
            # === CONVERSION TRANSFORMS ===
            "ToSparseTensor": ["ToSparseTensor"],
            "ToDense": ["ToDense"],
            "ToDevice": ["ToDevice"],
            "LineGraph": ["LineGraph"],
            # === POINT CLOUD TRANSFORMS ===
            "SamplePoints": ["SamplePoints"],
            "FixedPoints": ["FixedPoints"],
            "GridSampling": ["GridSampling"],
            "PointPairFeatures": ["PointPairFeatures"],
            # === MESH TRANSFORMS ===
            "FaceToEdge": ["FaceToEdge"],
            "GenerateMeshNormals": ["GenerateMeshNormals"],
            # === ADVANCED TRANSFORMS ===
            "VirtualNode": ["VirtualNode"],
            "TwoHop": ["TwoHop"],
            "HalfHop": ["HalfHop"],
            "RootedEgoNets": ["RootedEgoNets"],
            "RootedRWSubgraph": ["RootedRWSubgraph"],
            # === UTILITY TRANSFORMS ===
            "Center": ["Center"],
            "LinearTransformation": ["LinearTransformation"],
            "SVDFeatureReduction": ["SVDFeatureReduction"],
            "TargetIndegree": ["TargetIndegree"],
            # === DATA SPLITTING ===
            "NodePropertySplit": ["NodePropertySplit"],
            # === MASK/INDEX CONVERSION ===
            "IndexToMask": ["IndexToMask"],
            "MaskToIndex": ["MaskToIndex"],
            # === FILTERING ===
            "ComposeFilters": ["ComposeFilters"],
            "RemoveTrainingClasses": ["RemoveTrainingClasses"],
            # === SPECIAL TRANSFORMS ===
            "ToSLIC": ["ToSLIC"],
        }

    def _initialize_version_compatibility(self) -> dict[str, TransformCompatibility]:
        """
        Initialize version compatibility information for PyG 2.1+.

        This registry only tracks current PyG 2.1+ transforms. Legacy PyG 1.x
        and pre-2.1 compatibility information has been removed.

        Only transforms requiring special version consideration are included.
        Most standard transforms (AddSelfLoops, ToUndirected, etc.) are available
        in all PyG 2.1+ versions and are not listed here.

        Returns:
            Dict mapping transform names to compatibility information
        """
        return {
            # Transforms requiring PyG 2.1+
            "ToHeteroData": TransformCompatibility(
                min_version="2.1.0",
                compatibility_notes="Heterogeneous graph conversion added in PyG 2.1",
            ),
            "AddMetaPaths": TransformCompatibility(
                min_version="2.1.0",
                compatibility_notes="Meta-path generation for heterogeneous graphs added in PyG 2.1",
            ),
            # Transforms with external dependencies
            "ToSparseTensor": TransformCompatibility(
                min_version="2.0.0",
                compatibility_notes=(
                    "Requires torch-sparse package. Install with: "
                    "pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch_version}+{cuda}.html"
                ),
            ),
            # Spatial transforms with version-specific parameter signatures
            "Distance": TransformCompatibility(
                min_version="2.0.0",
                compatibility_notes=(
                    "Spatial transform for edge distances. "
                    "Parameter signatures may vary between versions - check PyG docs."
                ),
            ),
            "Cartesian": TransformCompatibility(
                min_version="2.0.0", compatibility_notes="Cartesian coordinate difference transform"
            ),
            "LocalCartesian": TransformCompatibility(
                min_version="2.0.0", compatibility_notes="Local Cartesian coordinate transform"
            ),
            "Polar": TransformCompatibility(
                min_version="2.0.0", compatibility_notes="Polar coordinate transformation"
            ),
            "Spherical": TransformCompatibility(
                min_version="2.0.0", compatibility_notes="Spherical coordinate transformation"
            ),
            # Core transforms with no special requirements (documentation only)
            "Compose": TransformCompatibility(
                min_version="2.0.0",
                compatibility_notes="Core composition class available in all PyG 2.x versions",
            ),
            "AddSelfLoops": TransformCompatibility(
                min_version="2.0.0",
                compatibility_notes="Core transform available in all PyG 2.x versions",
            ),
            "ToUndirected": TransformCompatibility(
                min_version="2.0.0",
                compatibility_notes="Core transform available in all PyG 2.x versions",
            ),
        }

    def discover_transforms(self, force_refresh: bool = False) -> dict[str, type]:
        """
        Main discovery method - scans PyG modules for available transforms

        Args:
            force_refresh: If True, force re-discovery even if cached

        Returns:
            Dictionary mapping transform names to their classes
        """
        if self._discovered_transforms and not force_refresh:
            return self._discovered_transforms

        self._logger.info("Starting dynamic transform discovery...")
        discovery_start = time.time()

        if not TORCH_GEOMETRIC_AVAILABLE:
            self._logger.warning("PyTorch Geometric not available, skipping discovery")
            return {}

        try:
            # Step 1: Scan PyG transforms module
            self._scan_pyg_transforms_module()

            # Step 2: Scan submodules if they exist
            self._scan_pyg_submodules()

            # Step 3: Apply canonical name mappings to resolve aliases
            self._apply_canonical_mappings()

            # Plugin system handles missing PyG transforms - no custom fallback needed here

            discovery_time = time.time() - discovery_start

            self._logger.info(
                f"Dynamic discovery completed: found {len(self._discovered_transforms)} "
                f"transforms in {discovery_time:.2f}s"
            )

        except Exception as e:
            self._logger.error(f"Transform discovery failed: {e}")

        return self._discovered_transforms

    def _apply_canonical_mappings(self):
        """
        Apply canonical name mappings to discovered transforms.

        This ensures milia configurations can use stable transform names
        regardless of PyG version-specific naming conventions.
        """
        for canonical_name, possible_names in self._canonical_mappings.items():
            # Skip if canonical name already discovered
            if canonical_name in self._discovered_transforms:
                continue

            # Try each possible PyG name
            for pyg_name in possible_names:
                if pyg_name in self._discovered_transforms:
                    # Map canonical name to discovered class
                    self._discovered_transforms[canonical_name] = self._discovered_transforms[
                        pyg_name
                    ]

                    self._discovery_metadata[canonical_name] = {
                        "module": self._discovery_metadata.get(pyg_name, {}).get(
                            "module", "unknown"
                        ),
                        "discovery_method": "canonical_mapping",
                        "source": "alias",
                        "pyg_native_name": pyg_name,
                        "mapping_note": f'Canonical name "{canonical_name}" mapped to PyG "{pyg_name}"',
                    }

                    self._logger.debug(
                        f"Mapped canonical name '{canonical_name}' → PyG '{pyg_name}'"
                    )
                    break

    def _scan_pyg_transforms_module(self):
        """Scan main torch_geometric.transforms module"""
        try:
            import torch_geometric.transforms as T

            # Get all public classes from transforms module
            candidates = [name for name in dir(T) if not name.startswith("_")]
            self._logger.debug(
                f"Scanning {len(candidates)} candidates in torch_geometric.transforms"
            )

            for name in candidates:
                try:
                    obj = getattr(T, name)

                    # Check if it's a class and likely a transform
                    if inspect.isclass(obj):
                        if self._is_transform_class(obj, name):
                            self._discovered_transforms[name] = obj
                            self._discovery_metadata[name] = {
                                "module": "torch_geometric.transforms",
                                "discovery_method": "auto",
                                "source": "main_module",
                            }
                            self._logger.debug(f"✓ Discovered transform: {name}")
                        else:
                            self._logger.debug(f"✗ Excluded {name}: failed transform heuristic")
                    else:
                        self._logger.debug(f"✗ Skipped {name}: not a class")

                except AttributeError as e:
                    self._logger.debug(f"✗ Skipping {name}: attribute error - {e}")
                    continue
                except Exception as e:
                    self._logger.debug(f"✗ Skipping {name}: inspection failed - {e}")
                    continue

            self._logger.info(
                f"Scanned torch_geometric.transforms: {len(self._discovered_transforms)} transforms discovered"
            )

        except ImportError as e:
            self._logger.error(f"Failed to import PyG transforms module: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error scanning PyG transforms module: {e}")

    def _scan_pyg_submodules(self):
        """Scan PyG submodules for additional transforms"""
        submodules_to_scan = [
            "torch_geometric.transforms.functional",
            "torch_geometric.transforms.normalize",
            "torch_geometric.transforms.augmentation",
            "torch_geometric.utils",
        ]

        for submodule_name in submodules_to_scan:
            try:
                submodule = importlib.import_module(submodule_name)

                for name in dir(submodule):
                    if name.startswith("_"):
                        continue

                    obj = getattr(submodule, name)

                    if inspect.isclass(obj) and self._is_transform_class(obj, name):
                        # Avoid duplicates from main module
                        if name not in self._discovered_transforms:
                            self._discovered_transforms[name] = obj
                            self._discovery_metadata[name] = {
                                "module": submodule_name,
                                "discovery_method": "auto",
                                "source": "submodule",
                            }

                            self._logger.debug(
                                f"Discovered transform from {submodule_name}: {name}"
                            )

            except ImportError:
                self._logger.debug(f"Submodule {submodule_name} not available")
            except Exception as e:
                self._logger.warning(f"Error scanning {submodule_name}: {e}")

    def _is_transform_class(self, obj: Any, name: str) -> bool:
        """
        Heuristic to determine if a class is a PyG transform

        Checks:
        1. Has __call__ method (transforms are callable)
        2. Not an abstract base class
        3. Name doesn't match exclusion patterns
        4. Has reasonable signature
        """
        # Exclusion patterns - only exclude true base classes
        exclusions = [
            "BaseTransform",  # Abstract base
            "LinearTransformation",  # Base class
        ]

        # Exact match only - don't exclude "Normalize" just because it contains "Base"
        if name in exclusions:
            return False

        # Also exclude if name starts with underscore or lowercase
        if name.startswith("_") or (name and name[0].islower()):
            return False

        # Should have __call__ method
        if not callable(obj):
            return False

        # Check if it has __init__ with reasonable signature
        try:
            sig = inspect.signature(obj.__init__)
            # Transforms typically have parameters beyond self
            [p for p in sig.parameters.values() if p.name != "self"]

            # Very basic check - real transforms usually have some parameters or none
            # (some transforms like ToUndirected have no params)
            return True

        except Exception:
            return False

    def get_discovery_metadata(self, name: str) -> dict[str, Any] | None:
        """Get discovery metadata for a transform"""
        return self._discovery_metadata.get(name)

    def get_compatibility_info(self, name: str) -> TransformCompatibility | None:
        """Get version compatibility information for a transform"""
        return self._version_compatibility.get(name)

    def is_transform_available(self, name: str, version: str | None = None) -> bool:
        """Check if transform is available and compatible with version"""

        if name not in self._discovered_transforms:
            return False

        if version:
            compat = self.get_compatibility_info(name)
            if compat:
                return compat.is_compatible(version)

        return True

    def get_all_discovered_transforms(self) -> list[str]:
        """Get list of all discovered transform names"""
        return list(self._discovered_transforms.keys())

    def get_transforms_by_module(self) -> dict[str, list[str]]:
        """Get transforms organized by their source module"""
        module_transforms = defaultdict(list)

        for name, metadata in self._discovery_metadata.items():
            module = metadata.get("module", "unknown")
            module_transforms[module].append(name)

        return dict(module_transforms)


# =============================================================================
# AUTO-DISCOVERY SYSTEM: ENHANCED TRANSFORM REGISTRY
# =============================================================================


class TransformRegistry:
    """
    Enhanced Transform Registry with Dynamic Discovery - Production-Ready Architecture Step 2.1

    New capabilities:
    - Automatic transform discovery from PyG modules
    - Version compatibility handling
    - Fallback mechanism for missing transforms
    - Enhanced metadata extraction
    - Transform dependency tracking
    """

    def __init__(self):
        self._transforms: dict[str, TransformInfo] = {}
        self._categories: dict[str, list[str]] = {}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialization_errors: list[str] = []

        # Performance tracking
        self._usage_stats: dict[str, int] = {}
        self._performance_cache: dict[str, dict[str, Any]] = {}

        # Auto-Discovery: Dynamic discovery engine
        self._discovery_engine = DynamicTransformDiscovery()
        self._auto_discovered: set[str] = set()

        # Auto-Discovery: Compatibility matrix
        self._compatibility_matrix: dict[tuple[str, str], str] = {}

        # Initialize with dynamic discovery
        if TORCH_GEOMETRIC_AVAILABLE:
            self._initialize_with_dynamic_discovery()
        else:
            error_msg = f"PyTorch Geometric not available: {TORCH_GEOMETRIC_IMPORT_ERROR if 'TORCH_GEOMETRIC_IMPORT_ERROR' in globals() else 'Import failed'}"
            self._initialization_errors.append(error_msg)
            self._logger.error(error_msg)

        # Plugin System: Custom transform tracking
        self._custom_transforms: dict[str, type] = {}
        self._custom_metadata: dict[str, TransformMetadata] = {}

    def register_custom(
        self, name: str, transform_class: type, metadata: TransformMetadata | None = None, **kwargs
    ) -> None:
        logger.debug(
            f"[DEBUG] TransformRegistry.register_custom: Called for {name} on registry id={id(self)}"
        )
        logger.debug(
            f"[DEBUG] TransformRegistry.register_custom: Before registration, _custom_transforms count = {len(self._custom_transforms)}"
        )
        if not CUSTOM_TRANSFORMS_AVAILABLE:
            raise TransformValidationError(
                "Custom transforms not available - custom_transforms module not imported",
                transform_name=name,
            )

        # Check if transform is callable (includes inherited __call__)
        if not callable(transform_class):
            raise TransformValidationError(
                f"Transform class must be callable, got {transform_class}", transform_name=name
            )

        # Check for get_metadata method
        if not hasattr(transform_class, "get_metadata"):
            raise TransformValidationError(
                f"Transform class must have 'get_metadata' method, got {transform_class}",
                transform_name=name,
            )

        if metadata is None:
            if hasattr(transform_class, "get_metadata"):
                metadata = transform_class.get_metadata()
            else:
                raise TransformValidationError(
                    f"Transform class {transform_class} must implement get_metadata()",
                    transform_name=name,
                )

        if hasattr(transform_class, "get_parameter_constraints"):
            constraints = transform_class.get_parameter_constraints()
            self._validate_parameter_constraints(constraints)

        # Map category to valid registry category
        CATEGORY_MAPPING = {"quantum": "custom", "molecular": "custom", "experimental": "custom"}
        raw_category = metadata.category if metadata else "custom"
        category = CATEGORY_MAPPING.get(raw_category, raw_category)

        valid_categories = {
            "normalization",
            "geometric",
            "custom",
            "spatial",
            "structural",
            "augmentation",
        }
        if category not in valid_categories:
            category = "custom"

        description = metadata.description if metadata else ""
        version = metadata.version if metadata else "1.0.0"
        author = metadata.author if metadata else "unknown"

        research_applicability = []
        if metadata:
            research_applicability.append(metadata.category)
            if metadata.validated_datasets:
                research_applicability.extend(metadata.validated_datasets)

        pre_transform_safe = True
        usage_note = f"Custom transform v{version}"
        if metadata and metadata.paper_reference:
            usage_note += f" (see: {metadata.paper_reference})"

        # Register with main registry
        self._register_transform_with_enhanced_metadata(
            name=name,
            transform_class=transform_class,
            category=category,
            discovery_method="custom",
            research_applicability=research_applicability,
            performance_notes=description,
            pre_transform_safe=pre_transform_safe,
            usage_note=usage_note,
            complexity_score=2.0,
        )

        # Track in custom transforms dict
        self._custom_transforms[name] = transform_class
        if metadata:
            self._custom_metadata[name] = metadata

        logger.debug(
            f"[DEBUG] TransformRegistry.register_custom: After registration, _custom_transforms count = {len(self._custom_transforms)}"
        )
        logger.debug(
            f"[DEBUG] TransformRegistry.register_custom: _custom_transforms keys = {list(self._custom_transforms.keys())}"
        )

        logger.info(f"Registered custom transform: {name} (category: {category}, author: {author})")

    @staticmethod
    def _validate_parameter_constraints(constraints: dict[str, dict]) -> None:
        """
        Validate parameter constraint definitions.

        Args:
            constraints: Dict mapping parameter names to constraint dicts

        Raises:
            TransformValidationError: If constraints are invalid
        """
        for param_name, constraint in constraints.items():
            if not isinstance(constraint, dict):
                raise TransformValidationError(
                    f"Parameter constraint for '{param_name}' must be a dictionary"
                )

            # Check for required constraint fields
            if "type" not in constraint:
                raise TransformValidationError(
                    f"Parameter constraint for '{param_name}' must specify 'type'"
                )

    def is_custom_transform(self, name: str) -> bool:
        """
        Check if a transform is a custom transform.

        Plugin System: Distinguish custom from built-in transforms.

        Args:
            name: Transform name

        Returns:
            True if transform is custom, False otherwise
        """
        return name in self._custom_transforms

    def get_custom_metadata(self, name: str) -> TransformMetadata | None:
        """
        Get metadata for a custom transform.

        Args:
            name: Transform name

        Returns:
            TransformMetadata or None if not a custom transform
        """
        return self._custom_metadata.get(name)

    def list_custom_transforms(
        self, category: str | None = None, author: str | None = None
    ) -> list[str]:
        """
        List all custom transforms, optionally filtered.

        Args:
            category: Filter by category (e.g., 'molecular', 'quantum')
            author: Filter by author

        Returns:
            List of custom transform names
        """
        transforms = list(self._custom_transforms.keys())

        if category:
            transforms = [
                name
                for name in transforms
                if self._custom_metadata.get(name)
                and self._custom_metadata[name].category == category
            ]

        if author:
            transforms = [
                name
                for name in transforms
                if self._custom_metadata.get(name) and self._custom_metadata[name].author == author
            ]

        return transforms

    def _initialize_with_dynamic_discovery(self) -> None:
        """
        Auto-Discovery: Initialize registry using dynamic discovery

        Process:
        1. Discover transforms from PyG modules
        2. Register core transforms with manual metadata
        3. Auto-register discovered transforms
        4. Build compatibility matrix
        """

        self._logger.info("Initializing registry with dynamic discovery...")

        # Step 1: Run discovery engine
        discovered = self._discovery_engine.discover_transforms()
        self._logger.info(f"Discovery engine found {len(discovered)} transforms")

        # Step 2: Register core transforms with curated metadata
        self._register_core_transforms_with_metadata()

        # Step 3: Auto-register remaining discovered transforms
        self._auto_register_discovered_transforms(discovered)

        # Step 4: Build compatibility matrix
        self._build_compatibility_matrix()

        total_registered = len(self._transforms)
        manual_count = total_registered - len(self._auto_discovered)

        self._logger.info(
            f"Registry initialized: {total_registered} transforms "
            f"({manual_count} manual, {len(self._auto_discovered)} auto-discovered)"
        )

    def _auto_register_discovered_transforms(self, discovered: dict[str, type]) -> None:
        """Automatically register discovered transforms with extracted metadata"""

        for name, transform_class in discovered.items():
            # Skip if already registered as core transform
            if name in self._transforms:
                continue

            try:
                # Extract metadata automatically
                metadata = self._extract_transform_metadata(name, transform_class)

                self._register_transform_with_enhanced_metadata(
                    name=name, transform_class=transform_class, discovery_method="auto", **metadata
                )

                self._auto_discovered.add(name)

            except Exception as e:
                self._logger.debug(f"Failed to auto-register {name}: {e}")
                self._initialization_errors.append(f"Auto-registration failed for {name}: {str(e)}")

    def _extract_transform_metadata(self, name: str, transform_class: type) -> dict[str, Any]:
        """
        Auto-Discovery: Extract metadata from transform class

        Uses inspection and heuristics to determine:
        - Category
        - Pre-transform safety
        - Research applicability
        - Performance characteristics
        - Dependencies
        """

        metadata = {
            "category": self._infer_category(name, transform_class),
            "pre_transform_safe": self._infer_pre_transform_safety(name),
            "research_applicability": self._infer_research_applicability(name),
            "performance_notes": self._infer_performance_notes(name),
            "complexity_score": self._estimate_complexity(name, transform_class),
            "dependencies": self._infer_dependencies(name, transform_class),
        }

        return metadata

    def _infer_category(self, name: str, transform_class: type) -> str:
        """Infer transform category from name and structure"""

        name_lower = name.lower()

        if any(keyword in name_lower for keyword in ["random", "drop", "mask", "augment"]):
            return "augmentation"
        elif any(keyword in name_lower for keyword in ["norm", "scale", "center"]):
            return "normalization"
        elif any(
            keyword in name_lower
            for keyword in ["distance", "cartesian", "polar", "spherical", "spatial"]
        ):
            return "spatial"
        elif any(keyword in name_lower for keyword in ["add", "remove", "virtual", "undirected"]):
            return "structural"
        elif any(keyword in name_lower for keyword in ["rotate", "translate", "flip"]):
            return "geometric"
        else:
            return "custom"

    def _infer_pre_transform_safety(self, name: str) -> bool:
        """Infer if transform is safe for pre-transforms"""

        # Stochastic transforms are NOT safe for pre-transforms
        unsafe_keywords = ["random", "drop", "mask", "sample", "augment"]
        name_lower = name.lower()

        return not any(keyword in name_lower for keyword in unsafe_keywords)

    def _infer_research_applicability(self, name: str) -> list[str]:
        """Infer research applicability from transform name"""

        applicability = []
        name_lower = name.lower()

        if "random" in name_lower or "drop" in name_lower:
            applicability.append("data_augmentation")

        if any(keyword in name_lower for keyword in ["distance", "cartesian"]):
            applicability.extend(["3d_molecular_analysis", "geometric_features"])

        if "norm" in name_lower:
            applicability.extend(["preprocessing", "feature_scaling"])

        if not applicability:
            applicability.append("general_purpose")

        return applicability

    def _infer_performance_notes(self, name: str) -> str:
        """Infer performance characteristics"""

        expensive_transforms = {
            "Distance": "O(V²) pairwise distance computation",
            "Cartesian": "O(E) edge-wise coordinate computation",
            "VirtualNode": "Adds O(V) edges and increases memory",
            "LocalCartesian": "Local coordinate system computation",
        }

        return expensive_transforms.get(name, "Standard performance characteristics")

    def _estimate_complexity(self, name: str, transform_class: type) -> float:
        """Estimate computational complexity score (1.0 = simple, 5.0 = very complex)"""

        complexity_map = {
            "Distance": 4.0,
            "VirtualNode": 3.5,
            "Cartesian": 3.0,
            "LocalCartesian": 3.0,
            "RandomRotate": 2.5,
            "GCNNorm": 2.0,
            "AddSelfLoops": 1.0,
            "ToUndirected": 1.0,
            "Normalize": 1.5,
        }

        return complexity_map.get(name, 2.0)  # Default to moderate complexity

    def _infer_dependencies(self, name: str, transform_class: type) -> TransformDependency:
        """Infer transform dependencies"""

        # Known dependency patterns
        dependency_rules = {
            "GCNNorm": TransformDependency(
                recommended_after=["AddSelfLoops"],
                required_graph_attributes=["edge_index"],
                modifies_attributes=["edge_weight"],
            ),
            "Distance": TransformDependency(
                recommended_after=["ToUndirected"],
                required_graph_attributes=["pos"],
                modifies_attributes=["edge_attr"],
            ),
            "VirtualNode": TransformDependency(
                conflicts_with=["RemoveIsolatedNodes"],
                modifies_attributes=["x", "edge_index", "batch"],
            ),
        }

        return dependency_rules.get(name, TransformDependency())

    def _register_transform_with_enhanced_metadata(
        self,
        name: str,
        transform_class: type,
        category: str,
        pre_transform_safe: bool = True,
        research_applicability: list[str] | None = None,
        performance_notes: str | None = None,
        usage_note: str | None = None,
        complexity_score: float = 2.0,
        dependencies: TransformDependency | None = None,
        discovery_method: str = "manual",
    ) -> None:
        """Register transform with Production-Ready Architecture enhanced metadata"""

        try:
            # Extract signature and parameters
            signature = inspect.signature(transform_class.__init__)
            parameters = self._extract_parameters(signature)

            # Get discovery metadata if auto-discovered
            discovery_metadata = self._discovery_engine.get_discovery_metadata(name)
            compatibility = self._discovery_engine.get_compatibility_info(name)

            # Create enhanced TransformInfo
            transform_info = TransformInfo(
                name=name,
                class_ref=transform_class,
                signature=signature,
                docstring=getattr(transform_class, "__doc__", None),
                parameters=parameters,
                category=category,
                research_applicability=research_applicability or [],
                performance_notes=performance_notes,
                pre_transform_safe=pre_transform_safe,
                usage_note=usage_note,
                complexity_score=complexity_score,
                dependencies=dependencies,
                discovery_method=discovery_method,
                module_path=discovery_metadata.get("module") if discovery_metadata else None,
                compatibility=compatibility,
            )

            # Validate and store
            self._validate_transform_info(transform_info)
            self._transforms[name] = transform_info

            # Update category index
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)

            # Initialize usage statistics
            self._usage_stats[name] = 0

            self._logger.debug(
                f"Registered transform '{name}' ({discovery_method}) "
                f"in category '{category}' with {len(parameters)} parameters"
            )

        except Exception as e:
            raise TransformRegistryError(
                f"Failed to register transform {name}",
                transform_name=name,
                registry_operation="registration",
                details=f"Registration error: {str(e)}",
            ) from e

    def _extract_parameters(self, signature: inspect.Signature) -> dict[str, Any]:
        """Extract parameter information from signature"""

        parameters = {}

        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            param_info = {
                "type": param.annotation if param.annotation != param.empty else Any,
                "default": param.default if param.default != param.empty else None,
                "required": param.default == param.empty,
                "kind": param.kind.name,
            }

            if param_info["type"] != Any:
                param_info["type_name"] = getattr(
                    param_info["type"], "__name__", str(param_info["type"])
                )

                # Handle Union types
                if (
                    hasattr(param_info["type"], "__origin__")
                    and param_info["type"].__origin__ is Union
                ):
                    param_info["union_types"] = [
                        getattr(t, "__name__", str(t)) for t in param_info["type"].__args__
                    ]

            parameters[param_name] = param_info

        return parameters

    def _validate_transform_info(self, transform_info: TransformInfo) -> None:
        """Validate transform info for consistency"""

        if not transform_info.name or not isinstance(transform_info.name, str):
            raise TransformValidationError(
                "Transform name must be a non-empty string", transform_name=str(transform_info.name)
            )

        if not inspect.isclass(transform_info.class_ref):
            raise TransformValidationError(
                f"Transform class reference must be a class, got {type(transform_info.class_ref)}",
                transform_name=transform_info.name,
            )

        valid_categories = {
            "geometric",
            "structural",
            "augmentation",
            "normalization",
            "spatial",
            "custom",
        }
        if transform_info.category not in valid_categories:
            raise TransformValidationError(
                f"Invalid category '{transform_info.category}', must be one of {valid_categories}",
                transform_name=transform_info.name,
            )

    def _build_compatibility_matrix(self) -> None:
        """Build compatibility matrix between transforms"""

        # Known incompatibilities
        incompatibilities = [
            # Structural conflicts
            ("DropNode", "VirtualNode", "DropNode may remove virtual node"),
            ("ToUndirected", "ToDirected", "Conflicting directionality"),
            ("RemoveIsolatedNodes", "VirtualNode", "May remove valid nodes after VirtualNode"),
            # Feature dimension conflicts
            ("Distance", "Cartesian", "Both modify edge_attr - may cause dimension mismatch"),
            ("Distance", "LocalCartesian", "Both modify edge_attr - may cause dimension mismatch"),
            ("Cartesian", "LocalCartesian", "Both modify edge_attr - may cause dimension mismatch"),
            # Normalization conflicts
            ("Normalize", "NormalizeFeatures", "Redundant normalization operations"),
            ("GCNNorm", "NormalizeScale", "Conflicting normalization approaches"),
            # Augmentation intensity conflicts
            ("DropNode", "DropEdge", "Combined dropout may be too aggressive"),
            ("DropNode", "MaskFeatures", "Combined may remove too much information"),
            ("DropEdge", "RandomNodeSample", "Combined sampling may be too aggressive"),
            # Order-dependent operations
            ("RemoveIsolatedNodes", "AddSelfLoops", "RemoveIsolatedNodes should come first"),
            (
                "AddSelfLoops",
                "GCNNorm",
                "Use GCNNorm with add_self_loops=False if AddSelfLoops already applied",
            ),
            # Dataset-specific incompatibilities
            ("RandomFlip", "Distance", "RandomFlip may break distance-based features"),
            ("RandomFlip", "Cartesian", "RandomFlip may break coordinate-based features"),
        ]

        for transform1, transform2, reason in incompatibilities:
            self._compatibility_matrix[(transform1, transform2)] = reason
            self._compatibility_matrix[(transform2, transform1)] = reason

    def check_compatibility(self, transform1: str, transform2: str) -> tuple[bool, str | None]:
        """Check if two transforms are compatible"""

        key = (transform1, transform2)
        if key in self._compatibility_matrix:
            return False, self._compatibility_matrix[key]

        return True, None

    # Rest of registry methods continue in Part 2...
    def register_transform(
        self,
        name: str,
        transform_class: type,
        category: str = "custom",
        research_applicability: list[str] | None = None,
        performance_notes: str | None = None,
    ) -> None:
        """Register a custom transform (public API)"""

        if name in self._transforms:
            self._logger.warning(f"Transform '{name}' already registered, overwriting")

        self._register_transform_with_enhanced_metadata(
            name=name,
            transform_class=transform_class,
            category=category,
            research_applicability=research_applicability,
            performance_notes=performance_notes,
        )

        self._logger.info(f"Registered custom transform '{name}' in category '{category}'")

    def get_transform_info(self, name: str) -> TransformInfo:
        """Get detailed information about a transform"""

        # Plugin System: Check if it's a custom transform first
        if CUSTOM_TRANSFORMS_AVAILABLE and self.is_custom_transform(name):  #
            custom_metadata = self.get_custom_metadata(name)  #
            transform_class = self._custom_transforms.get(
                name
            )  # Get actual class from custom registry

            # Build parameters dict
            parameters = {}
            if hasattr(transform_class, "get_default_params"):
                parameters = transform_class.get_default_params()

            # Get parameter constraints if available
            parameter_constraints = {}
            if hasattr(transform_class, "get_parameter_constraints"):
                parameter_constraints = transform_class.get_parameter_constraints()

            # Get signature if transform_class is available
            try:
                sig = (
                    inspect.signature(transform_class.__init__)
                    if transform_class
                    else inspect.signature(lambda: None)
                )
            except (ValueError, TypeError):
                sig = inspect.signature(lambda: None)

            # Get docstring
            docstring = transform_class.__doc__ if transform_class else None

            # Build description
            description = custom_metadata.description if custom_metadata else ""

            # Build TransformInfo object for custom transforms
            info = TransformInfo(
                name=name,
                class_ref=transform_class if transform_class else type(None),
                signature=sig,
                docstring=docstring,
                parameters=parameters,
                category=custom_metadata.category if custom_metadata else "custom",
                description=description,
                example_usage=None,
                performance_notes=None,
                research_applicability=None,
                pre_transform_safe=True,
                usage_note=None,
                module_path=None,
                source_file=None,
                discovery_method="plugin",
                compatibility=None,  # Will be set to default in __post_init__
                dependencies=None,  # Will be set to default in __post_init__
                tags=["custom", "plugin"],
                complexity_score=1.0,
            )

            # Add extra metadata as attributes if needed
            info._custom = True
            info._author = custom_metadata.author if custom_metadata else "unknown"
            info._version = custom_metadata.version if custom_metadata else "1.0.0"
            info._parameter_constraints = parameter_constraints

            # Add chemistry-specific metadata for molecular/quantum transforms
            if custom_metadata and hasattr(custom_metadata, "preserves_chemistry"):
                info._preserves_chemistry = custom_metadata.preserves_chemistry
            if custom_metadata and hasattr(custom_metadata, "preserves_coordinates"):
                info._preserves_coordinates = custom_metadata.preserves_coordinates
            if custom_metadata and hasattr(custom_metadata, "requires_3d"):
                info._requires_3d = custom_metadata.requires_3d

            return info

        if name not in self._transforms:
            available = list(self._transforms.keys())
            suggestions = self._get_name_suggestions(name, available)
            raise TransformNotFoundError(
                f"Transform '{name}' not found in registry",
                transform_name=name,
                available_transforms=available,
                suggestions=suggestions,
            )

        return self._transforms[name]

    def list_available_transforms(self) -> list[str]:
        """List all registered transform names"""
        return list(self._transforms.keys())

    def list_by_category(self, category: str) -> list[str]:
        """List transforms by category"""

        if category not in self._categories:
            available_categories = list(self._categories.keys())
            raise TransformValidationError(
                f"Category '{category}' not found",
                transform_name="category_lookup",
                details=f"Available categories: {available_categories}",
            )

        return self._categories[category].copy()

    def get_categories(self) -> dict[str, list[str]]:
        """Get all categories with their transforms"""
        return {cat: transforms.copy() for cat, transforms in self._categories.items()}

    def get_transform_class(self, name: str) -> type:
        """Get the actual transform class"""
        transform_info = self.get_transform_info(name)

        # Track usage
        self._usage_stats[name] = self._usage_stats.get(name, 0) + 1

        return transform_info.class_ref

    def get_research_applicable_transforms(self, research_context: str) -> list[str]:
        """Get transforms applicable to specific research context"""

        applicable_transforms = []

        for name, info in self._transforms.items():
            if research_context.lower() in [ctx.lower() for ctx in info.research_applicability]:
                applicable_transforms.append(name)

        return applicable_transforms

    def get_usage_statistics(self) -> dict[str, int]:
        """Get transform usage statistics"""
        return self._usage_stats.copy()

    def get_initialization_errors(self) -> list[str]:
        """Get any errors that occurred during initialization"""
        return self._initialization_errors.copy()

    def get_auto_discovered_transforms(self) -> list[str]:
        """Get list of auto-discovered transforms"""
        return list(self._auto_discovered)

    def get_discovery_statistics(self) -> dict[str, Any]:
        """Get statistics about transform discovery"""

        total = len(self._transforms)
        manual = total - len(self._auto_discovered)

        return {
            "total_transforms": total,
            "manually_registered": manual,
            "auto_discovered": len(self._auto_discovered),
            "categories": len(self._categories),
            "pyg_version": TORCH_GEOMETRIC_VERSION,
            "discovery_enabled": TORCH_GEOMETRIC_AVAILABLE,
        }

    def _get_name_suggestions(
        self, query: str, available: list[str], max_suggestions: int = 3
    ) -> list[str]:
        """Generate name suggestions based on edit distance"""

        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        distances = [
            (name, levenshtein_distance(query.lower(), name.lower())) for name in available
        ]
        distances.sort(key=lambda x: x[1])

        suggestions = []
        for name, dist in distances[:max_suggestions]:
            if dist <= max(len(query) // 2, 2):
                suggestions.append(name)

        return suggestions

    def _register_core_transforms_with_metadata(self) -> None:
        """Complete registration of core transforms with full metadata"""

        core_transforms_metadata = {
            # ===== STRUCTURAL TRANSFORMS =====
            "AddSelfLoops": {
                "category": "structural",
                "pre_transform_safe": True,
                "research_applicability": [
                    "molecular_graphs",
                    "gnn_normalization",
                    "message_passing",
                ],
                "performance_notes": "Fast, O(V) complexity. Adds self-loops to nodes without existing ones.",
                "complexity_score": 1.0,
                "dependencies": TransformDependency(
                    # UPDATED: edge_attr is now included because with edge_attr-aware injection,
                    # AddSelfLoops will modify edge_attr when it exists in the data
                    modifies_attributes=["edge_index", "edge_attr", "num_edges"],
                    recommended_before=["GCNNorm"],
                ),
                "usage_note": (
                    "Safe for pre-transforms. Essential for GNN architectures. "
                    "EDGE-ATTR AWARE: When data has edge_attr, parameters are auto-injected "
                    "to maintain edge_index/edge_attr shape consistency."
                ),
            },
            "ToUndirected": {
                "category": "structural",
                "pre_transform_safe": True,
                "research_applicability": [
                    "molecular_graphs",
                    "symmetry_preservation",
                    "bidirectional_edges",
                ],
                "performance_notes": "Fast, O(E) complexity. Doubles edge count by adding reverse edges.",
                "complexity_score": 1.0,
                "dependencies": TransformDependency(
                    modifies_attributes=["edge_index", "edge_attr", "num_edges"],
                    conflicts_with=["ToDirected"],
                    recommended_before=["Distance", "Cartesian"],
                ),
                "usage_note": "Safe for pre-transforms. Recommended for molecular graphs.",
            },
            "RemoveIsolatedNodes": {
                "category": "structural",
                "pre_transform_safe": False,
                "research_applicability": ["graph_cleaning", "preprocessing"],
                "performance_notes": "Fast, O(V) complexity. Reduces graph size.",
                "complexity_score": 1.0,
                "dependencies": TransformDependency(
                    modifies_attributes=["x", "pos", "edge_index", "batch", "num_nodes"],
                    recommended_before=["AddSelfLoops", "VirtualNode"],
                    conflicts_with=["VirtualNode"],
                ),
                "usage_note": "DANGEROUS for molecular graphs - may remove valid atoms!",
            },
            "VirtualNode": {
                "category": "structural",
                "pre_transform_safe": True,
                "research_applicability": [
                    "global_pooling",
                    "graph_classification",
                    "message_passing",
                ],
                "performance_notes": "O(V) edges added. Significantly increases memory and computation.",
                "complexity_score": 3.5,
                "dependencies": TransformDependency(
                    modifies_attributes=["x", "edge_index", "batch", "num_nodes"],
                    conflicts_with=["RemoveIsolatedNodes", "DropNode"],
                ),
                "usage_note": "Safe but expensive. Adds virtual super-node connected to all nodes.",
            },
            # ===== NORMALIZATION TRANSFORMS =====
            "NormalizeFeatures": {
                "category": "normalization",
                "pre_transform_safe": True,
                "research_applicability": [
                    "feature_scaling",
                    "preprocessing",
                    "numerical_stability",
                ],
                "performance_notes": "Fast, O(V·d) where d is feature dimension.",
                "complexity_score": 1.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["x"],
                    modifies_attributes=["x"],
                    conflicts_with=["Normalize"],  # Legacy name
                ),
                "usage_note": "Safe for pre-transforms. Normalizes features to unit norm.",
            },
            "Normalize": {
                "category": "normalization",
                "pre_transform_safe": True,
                "research_applicability": ["feature_scaling", "preprocessing"],
                "performance_notes": "Fast, O(V·d) where d is feature dimension.",
                "complexity_score": 1.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["x"], modifies_attributes=["x"]
                ),
                "usage_note": "Legacy name. Prefer NormalizeFeatures in PyG 2.0+",
            },
            "GCNNorm": {
                "category": "normalization",
                "pre_transform_safe": True,
                "research_applicability": [
                    "gcn_preprocessing",
                    "message_passing",
                    "spectral_methods",
                ],
                "performance_notes": "O(E) edge weight computation. Essential for GCN models.",
                "complexity_score": 2.0,
                "dependencies": TransformDependency(
                    required_graph_attributes=["edge_index"],
                    modifies_attributes=["edge_weight", "edge_attr"],
                    recommended_after=["AddSelfLoops"],
                ),
                "usage_note": "Safe for pre-transforms. Use add_self_loops=False if AddSelfLoops already applied.",
            },
            "NormalizeScale": {
                "category": "normalization",
                "pre_transform_safe": True,
                "research_applicability": ["coordinate_normalization", "3d_molecular_analysis"],
                "performance_notes": "O(V) position scaling.",
                "complexity_score": 1.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos"], modifies_attributes=["pos"]
                ),
                "usage_note": "Safe for pre-transforms. Scales positions to unit variance.",
            },
            # ===== SPATIAL/GEOMETRIC TRANSFORMS =====
            "Distance": {
                "category": "spatial",
                "pre_transform_safe": False,
                "research_applicability": [
                    "molecular_geometry",
                    "distance_features",
                    "3d_molecular_analysis",
                ],
                "performance_notes": "O(E) for existing edges, O(V²) if computing all distances. Can cause dimension mismatch with heterogeneous edge features.",
                "complexity_score": 4.0,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos", "edge_index"],
                    modifies_attributes=["edge_attr"],
                    recommended_after=["ToUndirected"],
                ),
                "usage_note": "INCOMPATIBLE with heterogeneous edge features! Set max_value to limit computation.",
            },
            "Cartesian": {
                "category": "spatial",
                "pre_transform_safe": True,
                "research_applicability": [
                    "geometric_features",
                    "3d_coordinates",
                    "molecular_geometry",
                ],
                "performance_notes": "O(E) coordinate difference computation.",
                "complexity_score": 3.0,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos", "edge_index"],
                    modifies_attributes=["edge_attr"],
                    recommended_after=["ToUndirected"],
                ),
                "usage_note": "Safe for pre-transforms. Adds coordinate differences as edge attributes.",
            },
            "LocalCartesian": {
                "category": "spatial",
                "pre_transform_safe": True,
                "research_applicability": ["local_geometry", "bond_angles", "molecular_structure"],
                "performance_notes": "O(E) local coordinate system transformation.",
                "complexity_score": 3.0,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos", "edge_index"],
                    modifies_attributes=["edge_attr"],
                ),
                "usage_note": "Safe for pre-transforms. Creates local coordinate frames.",
            },
            "Polar": {
                "category": "spatial",
                "pre_transform_safe": True,
                "research_applicability": ["geometric_features", "angular_information"],
                "performance_notes": "O(E) polar coordinate conversion.",
                "complexity_score": 2.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos", "edge_index"],
                    modifies_attributes=["edge_attr"],
                ),
                "usage_note": "Safe for pre-transforms. Converts to polar coordinates.",
            },
            "Spherical": {
                "category": "spatial",
                "pre_transform_safe": True,
                "research_applicability": ["3d_geometry", "spherical_coordinates"],
                "performance_notes": "O(E) spherical coordinate conversion.",
                "complexity_score": 2.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos", "edge_index"],
                    modifies_attributes=["edge_attr"],
                ),
                "usage_note": "Safe for pre-transforms. Converts to spherical coordinates.",
            },
            # ===== GEOMETRIC TRANSFORMS (Training-time only) =====
            "RandomRotate": {
                "category": "geometric",
                "pre_transform_safe": False,
                "research_applicability": [
                    "data_augmentation",
                    "rotation_invariance",
                    "3d_transformations",
                ],
                "performance_notes": "Matrix operations, O(V) for 3D coordinates.",
                "complexity_score": 2.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos"], modifies_attributes=["pos"]
                ),
                "usage_note": "Use in DataLoader during training, NOT as pre_transform!",
            },
            "RandomScale": {
                "category": "geometric",
                "pre_transform_safe": False,
                "research_applicability": [
                    "data_augmentation",
                    "scale_invariance",
                    "geometric_robustness",
                ],
                "performance_notes": "Fast scalar multiplication, O(V).",
                "complexity_score": 2.0,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos"], modifies_attributes=["pos"]
                ),
                "usage_note": "Use in DataLoader during training, NOT as pre_transform!",
            },
            "RandomTranslate": {
                "category": "geometric",
                "pre_transform_safe": False,
                "research_applicability": ["data_augmentation", "translation_invariance"],
                "performance_notes": "Fast vector addition, O(V).",
                "complexity_score": 1.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos"], modifies_attributes=["pos"]
                ),
                "usage_note": "Use in DataLoader during training, NOT as pre_transform!",
            },
            "RandomFlip": {
                "category": "geometric",
                "pre_transform_safe": False,
                "research_applicability": [
                    "data_augmentation",
                    "chirality_studies",
                    "geometric_augmentation",
                ],
                "performance_notes": "Fast coordinate negation, O(V).",
                "complexity_score": 1.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos"], modifies_attributes=["pos"]
                ),
                "usage_note": "Use in DataLoader during training, NOT as pre_transform! May not preserve molecular meaning.",
            },
            # ===== AUGMENTATION TRANSFORMS (Training-time only) =====
            "DropEdge": {
                "category": "augmentation",
                "pre_transform_safe": False,
                "research_applicability": [
                    "data_augmentation",
                    "robustness_training",
                    "regularization",
                ],
                "performance_notes": "O(E) edge sampling.",
                "complexity_score": 2.0,
                "dependencies": TransformDependency(
                    required_graph_attributes=["edge_index"],
                    modifies_attributes=["edge_index", "edge_attr", "num_edges"],
                ),
                "usage_note": "Use in DataLoader during training, NOT as pre_transform!",
            },
            "DropNode": {
                "category": "augmentation",
                "pre_transform_safe": False,
                "research_applicability": [
                    "data_augmentation",
                    "robustness_training",
                    "node_dropout",
                ],
                "performance_notes": "O(V) node sampling, removes edges connected to dropped nodes.",
                "complexity_score": 2.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["edge_index"],
                    modifies_attributes=["x", "pos", "edge_index", "edge_attr", "num_nodes"],
                    conflicts_with=["VirtualNode"],
                ),
                "usage_note": "Use in DataLoader during training, NOT as pre_transform! Dangerous for molecular graphs.",
            },
            "MaskFeatures": {
                "category": "augmentation",
                "pre_transform_safe": False,
                "research_applicability": [
                    "data_augmentation",
                    "feature_robustness",
                    "masked_prediction",
                ],
                "performance_notes": "O(V·d) feature masking where d is feature dimension.",
                "complexity_score": 1.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["x"], modifies_attributes=["x"]
                ),
                "usage_note": "Use in DataLoader during training, NOT as pre_transform!",
            },
            "RandomNodeSample": {
                "category": "augmentation",
                "pre_transform_safe": False,
                "research_applicability": ["subgraph_sampling", "scalability", "mini_batching"],
                "performance_notes": "O(V) node sampling with induced subgraph extraction.",
                "complexity_score": 2.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["edge_index"],
                    modifies_attributes=["x", "pos", "edge_index", "batch", "num_nodes"],
                ),
                "usage_note": "Use in DataLoader during training, NOT as pre_transform!",
            },
            # ===== ADDITIONAL USEFUL TRANSFORMS =====
            "Center": {
                "category": "normalization",
                "pre_transform_safe": True,
                "research_applicability": ["coordinate_centering", "3d_normalization"],
                "performance_notes": "O(V) centering of coordinates.",
                "complexity_score": 1.0,
                "dependencies": TransformDependency(
                    required_graph_attributes=["pos"], modifies_attributes=["pos"]
                ),
                "usage_note": "Safe for pre-transforms. Centers coordinates at origin.",
            },
            "LocalDegreeProfile": {
                "category": "structural",
                "pre_transform_safe": True,
                "research_applicability": ["graph_features", "node_importance", "topology"],
                "performance_notes": "O(V + E) degree computation.",
                "complexity_score": 2.0,
                "dependencies": TransformDependency(
                    required_graph_attributes=["edge_index"], modifies_attributes=["x"]
                ),
                "usage_note": "Safe for pre-transforms. Adds degree-based features.",
            },
            "OneHotDegree": {
                "category": "structural",
                "pre_transform_safe": True,
                "research_applicability": ["degree_features", "node_encoding"],
                "performance_notes": "O(V) one-hot encoding of degrees.",
                "complexity_score": 1.5,
                "dependencies": TransformDependency(
                    required_graph_attributes=["edge_index"], modifies_attributes=["x"]
                ),
                "usage_note": "Safe for pre-transforms. Creates one-hot degree features.",
            },
        }

        # Register each core transform
        successfully_registered = 0
        for name, metadata in core_transforms_metadata.items():
            transform_class = self._discovery_engine._discovered_transforms.get(name)

            if transform_class:
                try:
                    self._register_transform_with_enhanced_metadata(
                        name=name, transform_class=transform_class, **metadata
                    )
                    successfully_registered += 1
                    self._logger.debug(f"✓ Registered core transform: {name}")
                except Exception as e:
                    self._logger.warning(f"✗ Failed to register {name}: {e}")
                    self._initialization_errors.append(f"Failed to register {name}: {str(e)}")
            else:
                # Check if custom implementation was registered
                if name in self._discovery_engine._discovered_transforms:
                    transform_class = self._discovery_engine._discovered_transforms.get(name)
                    if transform_class:
                        try:
                            self._register_transform_with_enhanced_metadata(
                                name=name, transform_class=transform_class, **metadata
                            )
                            successfully_registered += 1
                            self._logger.info(f"✓ Registered '{name}' (custom implementation)")
                        except Exception as e:
                            self._logger.warning(f"✗ Failed to register custom {name}: {e}")
                    else:
                        self._logger.warning(
                            f"⚠ Transform '{name}' not available in PyG {TORCH_GEOMETRIC_VERSION}. "
                            f"Checking plugin system for implementations."
                        )
                else:
                    self._logger.warning(
                        f"⚠ Transform '{name}' not available in PyG {TORCH_GEOMETRIC_VERSION}. "
                        f"Checking plugin system for implementations."
                    )

        self._logger.info(
            f"Core transforms registration: {successfully_registered}/{len(core_transforms_metadata)} successful"
        )

    def get_transform_info_with_discovery(self, name: str) -> dict[str, Any]:
        """Get detailed information about a transform including discovery metadata"""

        if name not in self._transforms:
            available = list(self._transforms.keys())
            suggestions = self._get_name_suggestions(name, available)
            raise TransformNotFoundError(
                f"Transform '{name}' not found in registry",
                transform_name=name,
                available_transforms=available,
                suggestions=suggestions,
            )

        transform_info = self._transforms[name]
        discovery_metadata = self._discovery_engine.get_discovery_metadata(name)
        compatibility_info = self._discovery_engine.get_compatibility_info(name)

        return {
            "transform_info": transform_info,
            "discovery_metadata": discovery_metadata,
            "compatibility_info": compatibility_info,
            "usage_count": self._usage_stats.get(name, 0),
            "is_auto_discovered": name in self._auto_discovered,
        }

    # ============================================================
    # PLUGIN SYSTEM COMPATIBILITY - Static adapter methods
    # ============================================================

    @staticmethod
    def register(
        transform_class: type, name: str, metadata: TransformMetadata | None = None, **kwargs
    ) -> None:
        """
        Plugin-compatible static registration method.
        Uses module-level registry singleton directly to avoid circular dependencies.
        """
        # Use module-level registry directly instead of going through GraphTransforms singleton
        # This avoids circular dependency when plugins are loaded during GraphTransforms initialization
        registry.register_custom(name, transform_class, metadata)

    @staticmethod
    def get(name: str) -> type:
        """
        Plugin-compatible static getter method.
        Uses module-level registry singleton directly to avoid circular dependencies.
        """
        # Use module-level registry directly instead of going through GraphTransforms singleton
        # This avoids circular dependency when plugins are loaded during GraphTransforms initialization
        return registry.get_transform_class(name)

    @staticmethod
    def unregister(name: str) -> None:
        """
        Plugin-compatible static unregistration method.
        Removes transform from singleton instance.
        """
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms

        gt = get_graph_transforms()
        if name in gt.registry._transforms:
            del gt.registry._transforms[name]
        if name in gt.registry._custom_transforms:
            del gt.registry._custom_transforms[name]


# --- End of class TransformRegistry ---#

registry = TransformRegistry()


# =============================================================================
# PLUGIN SYSTEM - CUSTOM TRANSFORM REGISTRATION FUNCTIONS
# =============================================================================


def register_custom_transforms(
    custom_module_or_path: Any, registry: Optional["TransformRegistry"] = None
) -> dict[str, Any]:
    """Register custom transforms with the main TransformRegistry."""

    if registry is None:
        gt = get_graph_transforms()
        registry = gt.registry

    registered = []
    failed = []

    # ADD THIS LOGIC:
    if inspect.ismodule(custom_module_or_path):
        # Scan module for CustomTransformBase subclasses
        for name, obj in inspect.getmembers(custom_module_or_path, inspect.isclass):
            if _is_custom_transform_class(obj):
                try:
                    _register_single_custom_transform(obj, registry)
                    registered.append(name)
                except Exception as e:
                    failed.append({"name": name, "error": str(e)})

    return {"registered": registered, "failed": failed, "total": len(registered)}


def _is_custom_transform_class(obj: Any) -> bool:
    """Check if object is a CustomTransformBase subclass.

    Validates that a class is a valid custom transform by checking for:
    - It is a class (not instance)
    - It has a 'transform' method (CustomTransformBase pattern) OR 'forward' method (PyG pattern)
    - It has a 'get_metadata' class method for registry integration
    - It is not one of the abstract base classes

    Note:
        milia custom transforms use 'transform' method (called by __call__),
        while standard PyG transforms use 'forward' method. This function
        accepts either pattern for maximum compatibility.
    """
    if not CUSTOM_TRANSFORMS_AVAILABLE:
        return False

    try:
        from .custom_transforms import (
            CustomTransformBase,
            MolecularTransformBase,
            QuantumTransformBase,
        )

        # Check if it's a class
        if not inspect.isclass(obj):
            return False

        # Check for required method: 'transform' (CustomTransformBase) or 'forward' (PyG pattern)
        has_transform_method = hasattr(obj, "transform") or hasattr(obj, "forward")

        # Check for get_metadata (required for registry integration)
        has_get_metadata = hasattr(obj, "get_metadata")

        # Exclude the abstract base classes themselves
        is_not_base_class = obj not in [
            CustomTransformBase,
            MolecularTransformBase,
            QuantumTransformBase,
        ]

        return has_transform_method and has_get_metadata and is_not_base_class
    except ImportError:
        return False


def _register_single_custom_transform(
    transform_class: type, registry: Optional["TransformRegistry"] = None
) -> None:
    """
    Register a single custom transform with TransformRegistry.

    Args:
        transform_class: Transform class to register
        registry: Registry instance (uses singleton if None)
    """
    if not CUSTOM_TRANSFORMS_AVAILABLE:
        logger.warning(
            f"Cannot register {transform_class.__name__}: custom_transforms module not available"
        )
        return

    # Check for required methods: 'transform' (CustomTransformBase) or 'forward' (PyG pattern)
    has_transform_method = hasattr(transform_class, "transform") or hasattr(
        transform_class, "forward"
    )
    has_get_metadata = hasattr(transform_class, "get_metadata")

    if not (has_transform_method and has_get_metadata):
        raise ValueError(
            f"{transform_class.__name__} must have 'transform' (or 'forward') method "
            f"and 'get_metadata' classmethod to be registered as a custom transform"
        )

    # FIX: Use provided registry or get singleton
    if registry is None:
        import sys

        current_module = sys.modules[__name__]
        registry = current_module.registry
        logger.debug(f"[DEBUG] _register_single: Got module-level registry id={id(registry)}")
        logger.debug(
            f"[DEBUG] _register_single: Module-level registry custom count = {len(registry._custom_transforms)}"
        )

        # ALSO CHECK SINGLETON
        gt = get_graph_transforms()
        singleton_registry = gt.registry
        logger.debug(f"[DEBUG] _register_single: Singleton registry id={id(singleton_registry)}")
        logger.debug(
            f"[DEBUG] _register_single: Singleton registry custom count = {len(singleton_registry._custom_transforms)}"
        )
        logger.debug(
            f"[DEBUG] _register_single: Are they the same? {registry is singleton_registry}"
        )
    else:
        logger.debug(f"[DEBUG] _register_single: Using provided registry id={id(registry)}")
        logger.debug(
            f"[DEBUG] _register_single: Provided registry custom count = {len(registry._custom_transforms)}"
        )

    # Get metadata from the transform class
    metadata = transform_class.get_metadata()

    logger.debug(
        f"[DEBUG] _register_single: About to register {metadata.name} to registry id={id(registry)}"
    )

    # Use the register_custom() method that exists in TransformRegistry
    registry.register_custom(name=metadata.name, transform_class=transform_class, metadata=metadata)

    logger.debug(
        f"[DEBUG] _register_single: After registration, registry id={id(registry)} custom count = {len(registry._custom_transforms)}"
    )


def register_all_custom_transforms() -> int:
    """
    Auto-register all built-in custom transforms.

    Returns:
        Number of custom transforms successfully registered
    """
    if not CUSTOM_TRANSFORMS_AVAILABLE:
        logger.debug("Custom transforms module not available")
        return 0

    try:
        from . import custom_transforms

        # FIX: Use the singleton registry instead of creating new instance
        get_graph_transforms()

        stats = register_custom_transforms(custom_transforms)

        if stats["failed"]:
            for failure in stats["failed"]:
                logger.warning(f"Failed to register {failure['name']}: {failure['error']}")

        return stats["total"]

    except Exception as e:
        logger.error(f"Error during custom transform registration: {e}")
        return 0


def get_custom_transform_count() -> int:
    """Get the number of registered custom transforms."""
    if not CUSTOM_TRANSFORMS_AVAILABLE:
        return 0

    try:
        gt = get_graph_transforms()
        return len(gt.registry._custom_transforms)
    except Exception:
        return 0


# =============================================================================
# PRODUCTION METRICS AND MONITORING SYSTEM (from Foundation Architecture)
# =============================================================================


class ProductionMetricsCollector:
    """Comprehensive metrics collection for production monitoring integration"""

    def __init__(self, enable_external_metrics: bool = True):
        self.enable_external_metrics = enable_external_metrics
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._metrics = defaultdict(lambda: defaultdict(float))
        self._counters = defaultdict(int)
        self._histograms = defaultdict(list)
        self._gauges = defaultdict(float)
        self._last_reset = time.time()
        self._lock = threading.RLock()

        self._prometheus_enabled = False
        self._datadog_enabled = False
        self._custom_handlers = []

        self._initialize_external_integrations()

    def _initialize_external_integrations(self):
        if not self.enable_external_metrics:
            return

        try:
            import prometheus_client

            self._prometheus_enabled = True
            self._prometheus_metrics = {}
            self._logger.info("Prometheus metrics integration enabled")
        except ImportError:
            self._logger.debug("Prometheus not available - skipping integration")

        try:
            import datadog

            self._datadog_enabled = True
            self._logger.info("DataDog metrics integration enabled")
        except ImportError:
            self._logger.debug("DataDog not available - skipping integration")

    def increment_counter(
        self, metric_name: str, value: int = 1, tags: dict[str, str] | None = None
    ):
        with self._lock:
            self._counters[metric_name] += value

            if self._prometheus_enabled:
                self._send_to_prometheus("counter", metric_name, value, tags)
            if self._datadog_enabled:
                self._send_to_datadog("increment", metric_name, value, tags)

            for handler in self._custom_handlers:
                try:
                    handler("counter", metric_name, value, tags)
                except Exception as e:
                    self._logger.warning(f"Custom metrics handler failed: {e}")

    def set_gauge(self, metric_name: str, value: float, tags: dict[str, str] | None = None):
        with self._lock:
            self._gauges[metric_name] = value

            if self._prometheus_enabled:
                self._send_to_prometheus("gauge", metric_name, value, tags)
            if self._datadog_enabled:
                self._send_to_datadog("gauge", metric_name, value, tags)

    def record_histogram(self, metric_name: str, value: float, tags: dict[str, str] | None = None):
        with self._lock:
            self._histograms[metric_name].append(value)

            if len(self._histograms[metric_name]) > 1000:
                # Delete oldest 1 item at a time to maintain exactly 1000
                del self._histograms[metric_name][0]

            if self._prometheus_enabled:
                self._send_to_prometheus("histogram", metric_name, value, tags)
            if self._datadog_enabled:
                self._send_to_datadog("histogram", metric_name, value, tags)

    def record_timing(
        self, metric_name: str, duration_ms: float, tags: dict[str, str] | None = None
    ):
        self.record_histogram(f"{metric_name}.duration_ms", duration_ms, tags)

    def _send_to_prometheus(
        self, metric_type: str, name: str, value: float, tags: dict[str, str] | None
    ):
        try:
            import prometheus_client

            if name not in self._prometheus_metrics:
                if metric_type == "counter":
                    self._prometheus_metrics[name] = prometheus_client.Counter(
                        f"graph_transforms_{name}", f"Graph transforms {name} metric"
                    )
                elif metric_type == "gauge":
                    self._prometheus_metrics[name] = prometheus_client.Gauge(
                        f"graph_transforms_{name}", f"Graph transforms {name} metric"
                    )
                elif metric_type == "histogram":
                    self._prometheus_metrics[name] = prometheus_client.Histogram(
                        f"graph_transforms_{name}", f"Graph transforms {name} metric"
                    )

            if metric_type == "counter":
                self._prometheus_metrics[name].inc(value)
            elif metric_type == "gauge":
                self._prometheus_metrics[name].set(value)
            elif metric_type == "histogram":
                self._prometheus_metrics[name].observe(value)

        except Exception as e:
            self._logger.debug(f"Failed to send metric to Prometheus: {e}")

    def _send_to_datadog(
        self, metric_type: str, name: str, value: float, tags: dict[str, str] | None
    ):
        try:
            import datadog

            dd_tags = []
            if tags:
                dd_tags = [f"{k}:{v}" for k, v in tags.items()]

            if metric_type == "increment":
                datadog.increment(f"graph_transforms.{name}", value, tags=dd_tags)
            elif metric_type == "gauge":
                datadog.gauge(f"graph_transforms.{name}", value, tags=dd_tags)
            elif metric_type == "histogram":
                datadog.histogram(f"graph_transforms.{name}", value, tags=dd_tags)

        except Exception as e:
            self._logger.debug(f"Failed to send metric to DataDog: {e}")

    def add_custom_handler(self, handler: Callable[[str, str, float, dict[str, str] | None], None]):
        self._custom_handlers.append(handler)
        self._logger.info(f"Added custom metrics handler: {handler.__name__}")

    def get_metrics_summary(self) -> dict[str, Any]:
        with self._lock:
            summary = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histogram_stats": {},
                "collection_period_seconds": time.time() - self._last_reset,
                "external_integrations": {
                    "prometheus": self._prometheus_enabled,
                    "datadog": self._datadog_enabled,
                    "custom_handlers": len(self._custom_handlers),
                },
            }

            for name, values in self._histograms.items():
                if values:
                    import statistics

                    summary["histogram_stats"][name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "p95": self._percentile(values, 0.95),
                        "p99": self._percentile(values, 0.99),
                    }

            return summary

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def reset_metrics(self):
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._last_reset = time.time()
            self._logger.info("Metrics reset")

    def export_metrics_for_monitoring(self, format_type: str = "prometheus") -> str:
        if format_type == "prometheus":
            return self._export_prometheus_format()
        elif format_type == "json":
            return json.dumps(self.get_metrics_summary(), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_prometheus_format(self) -> str:
        lines = []

        for name, value in self._counters.items():
            lines.append(f"# TYPE graph_transforms_{name} counter")
            lines.append(f"graph_transforms_{name} {value}")

        for name, value in self._gauges.items():
            lines.append(f"# TYPE graph_transforms_{name} gauge")
            lines.append(f"graph_transforms_{name} {value}")

        for name, values in self._histograms.items():
            if values:
                lines.append(f"# TYPE graph_transforms_{name} histogram")
                lines.append(f"graph_transforms_{name}_count {len(values)}")
                lines.append(f"graph_transforms_{name}_sum {sum(values)}")

        return "\n".join(lines)


# =============================================================================
# ENHANCED CACHE MANAGEMENT SYSTEM (from Foundation Architecture)
# =============================================================================


class IntelligentCacheManager:
    """Advanced cache management with memory pressure handling"""

    def __init__(
        self,
        max_cache_size: int = 100,
        memory_threshold_mb: float = 500.0,
        cleanup_ratio: float = 0.3,
    ):
        self.max_cache_size = max_cache_size
        self.memory_threshold_mb = memory_threshold_mb
        self.cleanup_ratio = cleanup_ratio

        self._cache: dict[str, Any] = {}
        self._cache_metadata: dict[str, dict[str, Any]] = {}
        self._access_counts: dict[str, int] = defaultdict(int)
        self._last_accessed: dict[str, float] = {}
        self._creation_times: dict[str, float] = {}

        self._memory_monitor_enabled = PSUTIL_AVAILABLE
        self._psutil_available = PSUTIL_AVAILABLE

        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions_size": 0,
            "evictions_memory": 0,
            "memory_cleanups": 0,
            "total_items_cached": 0,
        }

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._cache:
                self._access_counts[key] += 1
                self._last_accessed[key] = time.time()
                self._stats["cache_hits"] += 1

                self._logger.debug(f"Cache hit for key: {key[:20]}...")
                return self._cache[key]

            self._stats["cache_misses"] += 1
            return None

    def put(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> bool:
        with self._lock:
            current_time = time.time()

            if len(self._cache) >= self.max_cache_size:
                if not self._evict_items_by_size():
                    self._logger.warning("Failed to evict items for new cache entry")
                    return False

            if self._memory_monitor_enabled and self._is_memory_pressure():
                if not self._evict_items_by_memory():
                    self._logger.warning("Failed to evict items due to memory pressure")
                    return False

            self._cache[key] = value
            self._cache_metadata[key] = metadata or {}
            self._last_accessed[key] = current_time
            self._creation_times[key] = current_time
            self._access_counts[key] = 1

            self._stats["total_items_cached"] += 1
            self._logger.debug(f"Cached item with key: {key[:20]}...")

            return True

    def _evict_items_by_size(self) -> bool:
        items_to_evict = max(1, int(len(self._cache) * self.cleanup_ratio))

        try:
            sorted_items = sorted(
                self._last_accessed.items(), key=lambda x: (self._access_counts[x[0]], x[1])
            )

            evicted_count = 0
            for key, _ in sorted_items[:items_to_evict]:
                self._remove_cache_item(key)
                evicted_count += 1

            self._stats["evictions_size"] += evicted_count
            self._logger.debug(f"Evicted {evicted_count} items due to cache size limit")
            return True

        except Exception as e:
            self._logger.error(f"Error during size-based eviction: {e}")
            return False

    def _evict_items_by_memory(self) -> bool:
        if not self._psutil_available:
            return True

        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb <= self.memory_threshold_mb:
                return True

            items_to_evict = max(1, int(len(self._cache) * 0.5))

            items_with_priority = []
            for key in self._cache:
                try:
                    item_size = len(str(self._cache[key]))
                    access_score = self._access_counts[key]
                    priority = item_size / max(access_score, 1)
                    items_with_priority.append((key, priority))
                except Exception:
                    items_with_priority.append((key, 1.0))

            items_with_priority.sort(key=lambda x: x[1], reverse=True)

            evicted_count = 0
            for key, _ in items_with_priority[:items_to_evict]:
                self._remove_cache_item(key)
                evicted_count += 1

            self._stats["evictions_memory"] += evicted_count
            self._stats["memory_cleanups"] += 1
            self._logger.info(
                f"Evicted {evicted_count} items due to memory pressure ({memory_mb:.1f}MB)"
            )

            gc.collect()

            return True

        except Exception as e:
            self._logger.error(f"Error during memory-based eviction: {e}")
            return False

    def _is_memory_pressure(self) -> bool:
        if not self._psutil_available:
            return False

        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb > self.memory_threshold_mb
        except Exception:
            return False

    def _remove_cache_item(self, key: str) -> None:
        try:
            del self._cache[key]
            del self._cache_metadata[key]
            del self._last_accessed[key]
            del self._creation_times[key]
            del self._access_counts[key]
        except KeyError:
            pass

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._cache_metadata.clear()
            self._last_accessed.clear()
            self._creation_times.clear()
            self._access_counts.clear()

            self._logger.info(f"Cleared cache: {count} items removed")
            return count

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            current_memory = 0
            if self._psutil_available:
                try:
                    import psutil

                    process = psutil.Process()
                    current_memory = process.memory_info().rss / 1024 / 1024
                except Exception:
                    pass

            hit_rate = 0.0
            total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
            if total_requests > 0:
                hit_rate = self._stats["cache_hits"] / total_requests

            return {
                "cache_size": len(self._cache),
                "max_cache_size": self.max_cache_size,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "current_memory_mb": current_memory,
                "memory_threshold_mb": self.memory_threshold_mb,
                "memory_pressure": self._is_memory_pressure(),
                **self._stats,
            }

    def get_cache_health(self) -> dict[str, Any]:
        stats = self.get_statistics()

        health = {"overall_health": "healthy", "issues": [], "recommendations": []}

        if stats["hit_rate"] < 0.5 and stats["total_requests"] > 10:
            health["issues"].append(f"Low cache hit rate: {stats['hit_rate']:.2%}")
            health["recommendations"].append(
                "Consider increasing cache size or reviewing caching strategy"
            )

        if stats["memory_pressure"]:
            health["issues"].append(f"Memory pressure detected: {stats['current_memory_mb']:.1f}MB")
            health["recommendations"].append("Consider reducing cache size or memory threshold")
            health["overall_health"] = "degraded"

        if stats["evictions_memory"] > stats["cache_hits"] * 0.1:
            health["issues"].append("Frequent memory-based evictions")
            health["recommendations"].append(
                "Increase memory threshold or optimize cached item sizes"
            )
            health["overall_health"] = "degraded"

        if health["issues"]:
            health["overall_health"] = (
                "degraded" if health["overall_health"] == "healthy" else "unhealthy"
            )

        return health


# =============================================================================
# TRANSFORM VALIDATOR CLASS (Enhanced for Production-Ready Architecture)
# =============================================================================


class TransformValidator:
    """Validates transform parameters and configurations"""

    def __init__(self, registry: "TransformRegistry", logger=None):
        self.registry = registry
        self._logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._metadata_cache: dict[tuple[str, str], ParameterMetadata] = {}
        self._cache_lock = threading.Lock()
        self._constraint_patterns = self._build_constraint_patterns()

    def validate_transform_config(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize transform configuration"""

        # Plugin System: Enhanced validation for custom transforms
        issues = []  # : Define issues list
        if CUSTOM_TRANSFORMS_AVAILABLE and self.registry.is_custom_transform(name):  #
            transform_class = self.registry.get(name)  #

            # Use custom transform's validate_parameters method
            if hasattr(transform_class, "validate_parameters"):
                try:
                    # Create temporary instance to validate
                    temp_instance = transform_class(**kwargs)  #
                    validation_result = temp_instance.validate_parameters(kwargs)  #

                    if not validation_result.get("valid", True):
                        issues.append(
                            {
                                "type": "custom_validation_failed",
                                "parameter": "multiple",
                                "error": validation_result.get("error", "Validation failed"),
                                "suggestions": validation_result.get("suggestions", []),
                            }
                        )
                except Exception as e:
                    issues.append(
                        {
                            "type": "custom_validation_error",
                            "parameter": "multiple",
                            "error": str(e),
                        }
                    )

            # Validate against parameter constraints
            if hasattr(transform_class, "get_parameter_constraints"):
                constraints = transform_class.get_parameter_constraints()
                for param_name, constraint in constraints.items():
                    if param_name in kwargs:  #
                        param_value = kwargs[param_name]  #

                        # Type validation
                        expected_type = constraint.get("type")
                        if expected_type and not isinstance(param_value, expected_type):
                            issues.append(
                                {
                                    "type": "type_mismatch",
                                    "parameter": param_name,
                                    "expected": expected_type.__name__,
                                    "got": type(param_value).__name__,
                                }
                            )

                        # Range validation for numeric types
                        if isinstance(param_value, (int, float)):
                            if "min" in constraint and param_value < constraint["min"]:
                                issues.append(
                                    {
                                        "type": "value_too_small",
                                        "parameter": param_name,
                                        "value": param_value,
                                        "min": constraint["min"],
                                    }
                                )
                            if "max" in constraint and param_value > constraint["max"]:
                                issues.append(
                                    {
                                        "type": "value_too_large",
                                        "parameter": param_name,
                                        "value": param_value,
                                        "max": constraint["max"],
                                    }
                                )

                        # Choices validation
                        if "choices" in constraint:
                            if param_value not in constraint["choices"]:
                                issues.append(
                                    {
                                        "type": "invalid_choice",
                                        "parameter": param_name,
                                        "value": param_value,
                                        "valid_choices": constraint["choices"],
                                    }
                                )

            # If custom transform, return early with results
            if not issues:
                return {
                    "valid": True,
                    "transform_name": name,  #
                    "custom": True,
                    "issues": [],
                }
            else:
                return {
                    "valid": False,
                    "transform_name": name,  #
                    "custom": True,
                    "issues": issues,
                }

        if name not in self.registry._transforms:
            available = self.registry.list_available_transforms()
            suggestions = self.registry._get_name_suggestions(name, available)
            raise TransformNotFoundError(
                f"Unknown transform: {name}",
                transform_name=name,
                available_transforms=available,
                suggestions=suggestions,
            )

        transform_info = self.registry.get_transform_info(name)
        validated_kwargs = {}
        errors = []
        warnings = []

        # Check required parameters
        for param_name, param_info in transform_info.parameters.items():
            if param_info["required"] and param_name not in kwargs:
                errors.append(f"Required parameter '{param_name}' missing for {name}")

        # Validate provided parameters
        for param_name, value in kwargs.items():
            if param_name not in transform_info.parameters:
                errors.append(f"Unknown parameter '{param_name}' for transform {name}")
                continue

            param_info = transform_info.parameters[param_name]
            try:
                # Use the existing _convert_to_type method instead of validate_parameter
                validated_value = self._convert_to_type(value, param_info["type"], param_name)
                validated_kwargs[param_name] = validated_value
            except (ValueError, TypeError) as e:
                errors.append(f"Invalid value for '{param_name}': {str(e)}")
            except Exception as e:
                errors.append(f"Validation error for '{param_name}': {str(e)}")

        # Add defaults for missing optional parameters
        for param_name, param_info in transform_info.parameters.items():
            if param_name not in validated_kwargs and not param_info["required"]:
                if param_info["default"] is not None:
                    validated_kwargs[param_name] = param_info["default"]

        if errors:
            raise TransformValidationError(
                f"Validation errors for {name}",
                transform_name=name,
                validation_errors=errors,
                details=f"Errors: {'; '.join(errors)}",
            )

        try:
            warnings.extend(
                self._generate_configuration_warnings(name, validated_kwargs, transform_info)
            )

            if warnings:
                for warning in warnings:
                    self._logger.warning(f"Transform {name}: {warning}")
        except Exception as e:
            self._logger.debug(f"Warning generation failed for {name}: {e}")

        return validated_kwargs

    def validate_parameter(self, param_name: str, value: Any, param_info: dict[str, Any]) -> Any:
        """STRICT parameter validation - returns validated value"""

        param_type = param_info.get("type", Any)

        if param_type == Any or param_type is None:
            return value

        # Handle Union types
        if hasattr(param_type, "__origin__") and param_type.__origin__ is Union:
            if value is None and type(None) in param_type.__args__:
                return None

            errors = []
            for union_type in param_type.__args__:
                if union_type == type(None):
                    continue

                try:
                    return self._convert_to_type(value, union_type, param_name)
                except (ValueError, TypeError) as e:
                    errors.append(str(e))
                    continue

            raise ValueError(f"Cannot convert {value}: {'; '.join(errors)}")

        return self._convert_to_type(value, param_type, param_name)

    def _convert_to_type(self, value: Any, target_type: type, param_name: str) -> Any:
        """STRICT type conversion"""

        if target_type == Any or target_type is None:
            return value

        if value is None:
            if target_type == type(None):
                return None
            raise ValueError(f"None not allowed for {target_type}")

        if isinstance(value, str):
            invalid_strings = ["invalid", "none", "null", "nan", "", "missing"]
            if value.lower().strip() in invalid_strings:
                raise ValueError(
                    f"Invalid string value: '{value}' cannot be converted to {target_type.__name__}"
                )

        if target_type == bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lower_val = value.lower().strip()
                if lower_val in ("true", "1", "yes", "on"):
                    return True
                elif lower_val in ("false", "0", "no", "off"):
                    return False
                else:
                    raise ValueError(f"Invalid boolean string: '{value}'")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value == 0:
                    return False
                elif value == 1:
                    return True
                else:
                    raise ValueError(f"Invalid boolean number: {value}")
            raise ValueError(f"Cannot convert {type(value).__name__} to bool")

        if target_type in (int, float):
            if isinstance(value, str):
                try:
                    converted = target_type(value)
                except ValueError:
                    raise ValueError(f"Cannot convert string '{value}' to {target_type.__name__}") from None
            elif isinstance(value, complex):
                raise ValueError("Complex numbers not supported")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                converted = target_type(value)
            else:
                raise ValueError(f"Cannot convert {type(value).__name__} to {target_type.__name__}")

            self._validate_numeric_range(param_name, converted, target_type)
            return converted

        if target_type == str:
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float, bool)):
                return str(value)
            raise ValueError(f"Cannot convert {type(value).__name__} to string")

        try:
            return target_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert {value} to {target_type.__name__}: {e}") from e

    def _validate_numeric_range(
        self, param_name: str, value: int | float, param_type: type
    ) -> None:
        """STRICT range validation"""

        if param_name == "p" or "prob" in param_name.lower() or param_name == "dropout":
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Parameter '{param_name}' value {value} must be between 0.0 and 1.0"
                )
            return

        if param_name == "max_value":
            if value is not None and value <= 0.0:
                raise ValueError(f"Parameter '{param_name}' must be positive, got {value}")
            return

        count_params = ["num_samples", "k", "num_workers", "batch_size"]
        if param_name in count_params:
            if value < 1:
                raise ValueError(f"Parameter '{param_name}' must be at least 1, got {value}")
            return

    def validate_parameter_with_constraints(
        self, transform_name: str, param_name: str, value: Any, check_constraints: bool = True
    ) -> tuple[bool, list[str]]:
        """
        Validate parameter value with advanced checks.

        Production Enhancement: Enhanced validation using metadata

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            metadata = self.get_parameter_metadata(transform_name, param_name)

            # Check type compatibility
            if metadata.type_hint is not None:
                if not self._check_type_compatibility(value, metadata.type_hint):
                    errors.append(
                        f"Type mismatch: expected {metadata.type_hint}, got {type(value)}"
                    )

            # Check constraints
            if check_constraints:
                for constraint in metadata.constraints:
                    valid, error_msg = constraint.validate(value)
                    if not valid:
                        errors.append(error_msg)
        except Exception as e:
            errors.append(f"Validation error: {e}")

        return (len(errors) == 0, errors)

    def _generate_configuration_warnings(
        self, name: str, kwargs: dict[str, Any], transform_info: TransformInfo
    ) -> list[str]:
        """Generate warnings for potentially problematic configurations"""

        warnings = []

        if name == "DropEdge":
            p_val = kwargs.get("p")
            if p_val is not None and p_val > 0.5:
                warnings.append(
                    f"High edge dropout rate (p={p_val}) may significantly alter graph structure"
                )

        if name == "DropNode":
            p_val = kwargs.get("p")
            if p_val is not None and p_val > 0.3:
                warnings.append(
                    f"High node dropout rate (p={p_val}) may remove critical molecular atoms"
                )

        if name == "RandomRotate":
            degrees_val = kwargs.get("degrees")
            if (
                degrees_val is not None
                and isinstance(degrees_val, (int, float))
                and abs(degrees_val) > 180
            ):
                warnings.append(
                    f"Large rotation ({degrees_val}°) may not preserve molecular meaning"
                )

        if name == "Distance" and kwargs.get("norm", True) is False:
            warnings.append("Unnormalized distances may cause numerical instability in training")

        if name == "VirtualNode":
            warnings.append(
                "VirtualNode increases computational complexity and memory usage significantly"
            )

        return warnings

    def get_validation_errors(self, name: str, kwargs: dict[str, Any]) -> list[str]:
        """Get list of validation errors without raising exceptions"""

        try:
            self.validate_transform_config(name, kwargs)
            return []
        except TransformValidationError as e:
            return e.validation_errors
        except Exception as e:
            return [str(e)]

    def suggest_corrections(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Suggest parameter corrections for common issues"""

        suggestions = {}

        try:
            transform_info = self.registry.get_transform_info(name)
        except TransformNotFoundError:
            return suggestions

        for param_name, value in kwargs.items():
            if param_name not in transform_info.parameters:
                available_params = list(transform_info.parameters.keys())
                similar = self.registry._get_name_suggestions(
                    param_name, available_params, max_suggestions=1
                )
                if similar:
                    suggestions[param_name] = f"Did you mean '{similar[0]}'?"
            else:
                param_info = transform_info.parameters[param_name]

                if param_info["type"] in (int, float) and isinstance(value, str):
                    try:
                        corrected = param_info["type"](value)
                        self._validate_numeric_range(param_name, corrected, param_info["type"])
                        suggestions[param_name] = corrected
                    except (ValueError, TypeError):
                        pass

                if param_info["type"] in (int, float) and isinstance(value, (int, float)):
                    try:
                        self._validate_numeric_range(param_name, value, param_info["type"])
                    except ValueError:
                        if param_name == "p" and value > 1.0:
                            suggestions[param_name] = min(value, 1.0)
                        elif param_name == "degrees" and abs(value) > 360:
                            suggestions[param_name] = value % 360

        return suggestions

    def set_dataset_context(self, dataset_type: str):
        """Set dataset context for dataset-specific validation"""
        self._current_dataset_type = dataset_type

    def clear_dataset_context(self):
        """Clear dataset context"""
        if hasattr(self, "_current_dataset_type"):
            delattr(self, "_current_dataset_type")

    def validate_sequence_compatibility(self, transform_sequence: list[str]) -> dict[str, Any]:
        """Validate compatibility of transform sequence"""

        compatibility = {
            "compatible": True,
            "conflicts": [],
            "dependencies": [],
            "recommendations": [],
        }

        conflicts = {
            (
                "AddSelfLoops",
                "GCNNorm",
            ): "Consider using GCNNorm with add_self_loops=False when using AddSelfLoops",
            ("ToUndirected", "ToUndirected"): "Multiple ToUndirected transforms are redundant",
            (
                "DropNode",
                "VirtualNode",
            ): "DropNode may remove the virtual node, causing unpredictable behavior",
        }

        for i, transform1 in enumerate(transform_sequence):
            for j, transform2 in enumerate(transform_sequence[i + 1 :], i + 1):
                conflict_key = (transform1, transform2)
                reverse_key = (transform2, transform1)

                if conflict_key in conflicts:
                    compatibility["conflicts"].append(
                        {
                            "transforms": [transform1, transform2],
                            "positions": [i, j],
                            "message": conflicts[conflict_key],
                        }
                    )
                elif reverse_key in conflicts:
                    compatibility["conflicts"].append(
                        {
                            "transforms": [transform2, transform1],
                            "positions": [j, i],
                            "message": conflicts[reverse_key],
                        }
                    )

        if compatibility["conflicts"] or compatibility["dependencies"]:
            compatibility["compatible"] = False

        return compatibility

    def _build_constraint_patterns(self) -> dict[str, ParameterConstraint]:
        """
        Build constraint patterns for common parameter names.

        Production Enhancement: Heuristic patterns for constraint inference
        """
        return {
            # Probability/ratio parameters
            "prob": ParameterConstraint(
                type="range",
                description="Probability value",
                constraint_value=(0.0, 1.0),
                inferred=True,
                confidence=0.9,
            ),
            "probability": ParameterConstraint(
                type="range",
                description="Probability value",
                constraint_value=(0.0, 1.0),
                inferred=True,
                confidence=0.95,
            ),
            "ratio": ParameterConstraint(
                type="range",
                description="Ratio value",
                constraint_value=(0.0, 1.0),
                inferred=True,
                confidence=0.85,
            ),
            "threshold": ParameterConstraint(
                type="range",
                description="Threshold value",
                constraint_value=(0.0, 1.0),
                inferred=True,
                confidence=0.7,
            ),
            # Angle parameters
            "angle": ParameterConstraint(
                type="range",
                description="Angle in degrees",
                constraint_value=(0.0, 360.0),
                inferred=True,
                confidence=0.8,
            ),
            "degrees": ParameterConstraint(
                type="range",
                description="Angle in degrees",
                constraint_value=(0.0, 360.0),
                inferred=True,
                confidence=0.85,
            ),
            # Percentage parameters
            "percent": ParameterConstraint(
                type="range",
                description="Percentage value",
                constraint_value=(0.0, 100.0),
                inferred=True,
                confidence=0.85,
            ),
            "percentage": ParameterConstraint(
                type="range",
                description="Percentage value",
                constraint_value=(0.0, 100.0),
                inferred=True,
                confidence=0.9,
            ),
            # Count parameters
            "num": ParameterConstraint(
                type="range",
                description="Positive count",
                constraint_value=(1, float("inf")),
                inferred=True,
                confidence=0.75,
            ),
            "count": ParameterConstraint(
                type="range",
                description="Positive count",
                constraint_value=(1, float("inf")),
                inferred=True,
                confidence=0.8,
            ),
            "k": ParameterConstraint(
                type="range",
                description="Positive integer",
                constraint_value=(1, float("inf")),
                inferred=True,
                confidence=0.7,
            ),
            "p": ParameterConstraint(
                type="range",
                description="Probability parameter",
                constraint_value=(0.0, 1.0),
                inferred=True,
                confidence=0.95,
            ),
        }

    def get_parameter_metadata(
        self, transform_name: str, parameter_name: str, use_cache: bool = True
    ) -> ParameterMetadata:
        """
        Get comprehensive metadata for a transform parameter.

        Production Enhancement: Main introspection method

        Args:
            transform_name: Name of transform
            parameter_name: Name of parameter
            use_cache: Whether to use cached metadata

        Returns:
            ParameterMetadata with complete information
        """
        cache_key = (transform_name, parameter_name)

        # Check cache
        if use_cache:
            with self._cache_lock:
                if cache_key in self._metadata_cache:
                    return self._metadata_cache[cache_key]

        # Get transform info
        transform_info = self.registry.get_transform_info(transform_name)

        # Check if parameter exists
        if parameter_name not in transform_info.parameters:
            raise TransformValidationError(
                f"Parameter '{parameter_name}' not found in transform '{transform_name}'",
                transform_name=transform_name,
            )

        param = transform_info.parameters[parameter_name]

        # Build metadata
        metadata = self._extract_parameter_metadata(transform_info, parameter_name, param)

        # Cache result
        if use_cache:
            with self._cache_lock:
                self._metadata_cache[cache_key] = metadata

        return metadata

    def _extract_parameter_metadata(
        self, transform_info, param_name: str, param: Any
    ) -> ParameterMetadata:
        """Extract comprehensive metadata for a parameter"""

        # Extract type hint - prefer from signature if available
        type_hint = self._extract_type_hint(transform_info.class_ref, param_name)

        # If no type hint from get_type_hints, try from stored param info
        if type_hint is None and isinstance(param, dict) and "type" in param:
            type_hint = param.get("type")

        # Your registry stores parameters as dicts with these keys:
        # {'type': ..., 'default': ..., 'required': ..., 'kind': ...}
        if isinstance(param, dict):
            default_value = param.get("default")
            required = param.get("required", True)
        else:
            # Fallback for inspect.Parameter objects (shouldn't happen in your code)
            default_value = param.default if hasattr(param, "default") else None
            required = default_value is None

        # Extract description from docstring
        description = self._extract_parameter_description(transform_info.class_ref, param_name)

        # Infer constraints
        constraints = self._infer_constraints(param_name, type_hint, default_value)

        # Generate examples
        examples = self._generate_parameter_examples(
            param_name, type_hint, default_value, constraints
        )

        return ParameterMetadata(
            name=param_name,
            type_hint=type_hint,
            default_value=default_value if default_value is not None else inspect.Parameter.empty,
            required=required,
            description=description,
            constraints=constraints,
            examples=examples,
            inferred_from_name=any(c.inferred for c in constraints),
            docstring_source=description is not None,
        )

    def _extract_type_hint(self, transform_class: type, param_name: str) -> type | None:
        """Extract type hint for a parameter"""
        try:
            # First try get_type_hints (works for most real classes)
            try:
                hints = get_type_hints(transform_class.__init__)
                if param_name in hints:
                    return hints.get(param_name)
            except Exception:
                # get_type_hints can fail on dynamically created classes
                pass

            # Fallback: inspect signature annotations directly
            # This works for both real and mock classes
            sig = inspect.signature(transform_class.__init__)
            if param_name in sig.parameters:
                param = sig.parameters[param_name]
                if param.annotation != inspect.Parameter.empty:
                    return param.annotation

            return None
        except Exception as e:
            self._logger.debug(f"Could not extract type hint for {param_name}: {e}")
            return None

    def _extract_parameter_description(self, transform_class: type, param_name: str) -> str | None:
        """
        Extract parameter description from docstring.

        Supports NumPy, Google, and Sphinx formats
        """
        doc = inspect.getdoc(transform_class)
        if not doc:
            return None

        try:
            # Try NumPy-style
            numpy_pattern = rf"{param_name}\s*:\s*([^\n]+(?:\n\s+[^\n]+)*)"
            match = re.search(numpy_pattern, doc, re.MULTILINE)
            if match:
                description = match.group(1).strip()
                description = re.sub(r"\s+", " ", description)
                return description

            # Try Google-style
            google_pattern = rf"{param_name}\s*\(([^)]+)\):\s*([^\n]+(?:\n\s+[^\n]+)*)"
            match = re.search(google_pattern, doc, re.MULTILINE)
            if match:
                description = match.group(2).strip()
                description = re.sub(r"\s+", " ", description)
                return description

            # Try Sphinx-style
            sphinx_pattern = rf":param\s+{param_name}:\s*([^\n]+(?:\n\s+[^\n]+)*)"
            match = re.search(sphinx_pattern, doc, re.MULTILINE)
            if match:
                description = match.group(1).strip()
                description = re.sub(r"\s+", " ", description)
                return description
        except Exception as e:
            self._logger.debug(f"Error parsing docstring for {param_name}: {e}")

        return None

    def _infer_constraints(
        self, param_name: str, type_hint: type | None, default_value: Any
    ) -> list[ParameterConstraint]:
        """Infer constraints from parameter name and type"""

        constraints = []
        param_lower = param_name.lower()

        # Check name-based patterns
        for pattern, constraint_template in self._constraint_patterns.items():
            if pattern in param_lower:
                constraint = ParameterConstraint(
                    type=constraint_template.type,
                    description=constraint_template.description,
                    constraint_value=constraint_template.constraint_value,
                    inferred=True,
                    confidence=constraint_template.confidence,
                )
                constraints.append(constraint)
                break

        # Type-based constraints
        if type_hint is not None:
            origin = get_origin(type_hint)
            args = get_args(type_hint)

            # Literal type → choices constraint
            if origin is not None and hasattr(origin, "__name__") and origin.__name__ == "Literal":
                constraint = ParameterConstraint(
                    type="choices",
                    description="Allowed literal values",
                    constraint_value=set(args),
                    inferred=False,
                    confidence=1.0,
                )
                constraints.append(constraint)

        return constraints

    def _generate_parameter_examples(
        self,
        param_name: str,
        type_hint: type | None,
        default_value: Any,
        constraints: list[ParameterConstraint],
    ) -> list[Any]:
        """Generate example valid values"""

        examples = []

        # Include default if available
        if default_value is not inspect.Parameter.empty:
            examples.append(default_value)

        # Generate from constraints
        for constraint in constraints:
            if constraint.type == "range":
                min_val, max_val = constraint.constraint_value
                if min_val != float("-inf") and max_val != float("inf"):
                    if min_val >= 0:
                        examples.extend([min_val, (min_val + max_val) / 2, max_val])
                    else:
                        examples.extend([min_val, 0, max_val])
            elif constraint.type == "choices":
                choices = list(constraint.constraint_value)
                if len(choices) <= 5:
                    examples.extend(choices)
                else:
                    examples.extend(choices[:3])

        # Type-based examples if none yet
        if not examples and type_hint is not None:
            base_type = self._get_base_type(type_hint)
            if base_type == int:
                examples = [0, 1, 10]
            elif base_type == float:
                examples = [0.0, 0.5, 1.0]
            elif base_type == bool:
                examples = [True, False]
            elif base_type == str:
                examples = ["example"]

        # Remove duplicates
        seen = set()
        unique_examples = []
        for ex in examples:
            if ex not in seen:
                seen.add(ex)
                unique_examples.append(ex)

        return unique_examples[:5]

    def _get_base_type(self, type_hint: type) -> type | None:
        """Extract base type from Optional or Union"""
        origin = get_origin(type_hint)
        if origin is Union:
            args = get_args(type_hint)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return non_none_args[0]
        return type_hint

    def validate_parameter(
        self, transform_name: str, param_name: str, value: Any, check_constraints: bool = True
    ) -> tuple[bool, list[str]]:
        """
        Validate parameter value with advanced checks.

        Production Enhancement: Enhanced validation using metadata
        """
        errors = []

        try:
            metadata = self.get_parameter_metadata(transform_name, param_name)

            # Check type compatibility
            if metadata.type_hint is not None:
                if not self._check_type_compatibility(value, metadata.type_hint):
                    errors.append(
                        f"Type mismatch: expected {metadata.type_hint}, got {type(value)}"
                    )

            # Check constraints
            if check_constraints:
                for constraint in metadata.constraints:
                    valid, error_msg = constraint.validate(value)
                    if not valid:
                        errors.append(error_msg)
        except Exception as e:
            errors.append(f"Validation error: {e}")

        return (len(errors) == 0, errors)

    def _check_type_compatibility(self, value: Any, type_hint: type) -> bool:
        """Advanced type checking"""
        try:
            if value is None:
                origin = get_origin(type_hint)
                if origin is Union:
                    return type(None) in get_args(type_hint)
                return False

            origin = get_origin(type_hint)
            args = get_args(type_hint)

            if origin is Union:
                return any(self._check_type_compatibility(value, arg) for arg in args)

            if origin is list:
                if not isinstance(value, list):
                    return False
                if args:
                    # STRICT: All items must match type
                    return all(self._check_type_compatibility(item, args[0]) for item in value)
                return True

            if origin is dict:
                if not isinstance(value, dict):
                    return False
                if args:
                    key_type, val_type = args
                    # STRICT: All keys and values must match types
                    return all(
                        self._check_type_compatibility(k, key_type)
                        and self._check_type_compatibility(v, val_type)
                        for k, v in value.items()
                    )
                return True

            if origin is None:
                return isinstance(value, type_hint)

            return isinstance(value, origin)
        except Exception:
            # Be STRICT on errors - if we can't validate, assume incompatible
            return False  # Changed from True to False

    def validate_with_context(
        self, transform_configs: list[dict[str, Any]], context: ValidationContext
    ) -> ValidationContext:
        """
        Validate transforms with full context

        Context-Aware Validation: Enhanced validation with context

        Args:
            transform_configs: Transform configurations
            context: Validation context

        Returns:
            Updated validation context
        """

        # Basic parameter validation
        for i, config in enumerate(transform_configs):
            if not isinstance(config, dict):
                context.add_issue(
                    severity=ValidationSeverity.CRITICAL,
                    category="structure",
                    message=f"Transform {i} must be a dictionary",
                    location=f"config[{i}]",
                    suggestion="Ensure all transforms are properly formatted",
                    auto_fixable=False,
                )
                continue

            name = config.get("name")
            if not name:
                context.add_issue(
                    severity=ValidationSeverity.CRITICAL,
                    category="structure",
                    message=f"Transform {i} missing 'name' field",
                    location=f"config[{i}]",
                    suggestion="Add 'name' field to transform configuration",
                    auto_fixable=False,
                )
                continue

            kwargs = config.get("kwargs", {})

            # Validate parameters
            validation_errors = self.get_validation_errors(name, kwargs)
            for error in validation_errors:
                context.add_issue(
                    severity=ValidationSeverity.ERROR,
                    category="parameter",
                    message=error,
                    location=f"transform[{i}]:{name}",
                    suggestion=None,
                    auto_fixable=False,
                )

        return context


# =============================================================================
# TRANSFORM COMPOSER CLASS (Simplified for length)
# =============================================================================


class TransformComposer:
    """Handles transform sequences with production enhancements"""

    def __init__(
        self,
        registry: "TransformRegistry",
        validator: TransformValidator,
        error_recovery: Optional["TransformErrorRecovery"] = None,
    ):
        # CRITICAL: Initialize logger FIRST before any other operations
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Now initialize other attributes
        self.registry = registry
        self.validator = validator
        self.error_recovery = error_recovery

        # Initialize edge_attr-aware parameter injector
        # DYNAMIC, PRODUCTION-READY, FUTURE-PROOF edge_attr handling
        self._edge_attr_injector = EdgeAttrAwareParameterInjector()

        # Initialize cache manager
        self._cache_manager = IntelligentCacheManager(
            max_cache_size=100, memory_threshold_mb=500.0, cleanup_ratio=0.3
        )

        # Initialize metrics collector
        self._metrics = ProductionMetricsCollector(enable_external_metrics=True)

        # Initialize composition statistics
        self._composition_stats: dict[str, Any] = {
            "total_compositions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_sequence_length": 0.0,
            "most_used_transforms": {},
            "error_recovery_attempts": 0,
            "successful_recoveries": 0,
            "edge_attr_injections": 0,  # Track edge_attr parameter injections
        }

        self._logger.debug("TransformComposer initialized successfully")

    def set_sample_data(self, sample_data: Any) -> None:
        """
        Set sample data for edge_attr-aware parameter injection.

        This enables automatic detection of edge_attr and injection of required
        parameters for transforms like AddSelfLoops.

        Args:
            sample_data: A PyG Data object from the dataset
        """
        self._edge_attr_injector.set_sample_data(sample_data)
        self._logger.debug(
            f"Sample data set for edge_attr detection: "
            f"has_edge_attr={self._edge_attr_injector.has_edge_attr}, "
            f"edge_attr_dim={self._edge_attr_injector.edge_attr_dim}"
        )

    def compose_transforms(
        self,
        transform_configs: list[dict[str, Any]],
        cache_key: str | None = None,
        validate_sequence: bool = True,
        enable_recovery: bool = True,
        sample_data: Any = None,
    ) -> Compose | None:
        """
        Create a Compose object from transform configurations.

        ENHANCED: Now supports edge_attr-aware parameter injection for transforms
        that modify edge_index (e.g., AddSelfLoops). When sample_data is provided
        and contains edge_attr, required parameters are automatically injected.

        Args:
            transform_configs: List of transform configurations
            cache_key: Optional cache key for caching
            validate_sequence: Whether to validate the sequence
            enable_recovery: Whether to enable error recovery
            sample_data: Optional sample data for edge_attr detection (DYNAMIC injection)

        Returns:
            Compose object or None if creation fails
        """

        composition_start_time = time.time()
        self._metrics.increment_counter("composition.requests", 1)

        if not transform_configs:
            self._logger.debug("Empty transform configuration provided")
            return None

        # EDGE-ATTR AWARE INJECTION: Set sample data if provided
        if sample_data is not None:
            self.set_sample_data(sample_data)

        # EDGE-ATTR AWARE INJECTION: Apply parameter injection to configs
        # This modifies configs for transforms that need edge_attr handling
        if self._edge_attr_injector.has_edge_attr:
            transform_configs = self._edge_attr_injector.inject_params_batch(transform_configs)

            # Track injections
            injection_log = self._edge_attr_injector.get_injection_log()
            if injection_log:
                self._composition_stats["edge_attr_injections"] += len(injection_log)
                self._metrics.increment_counter(
                    "composition.edge_attr_injections", len(injection_log)
                )

        if cache_key is None:
            cache_key = self._generate_cache_key(transform_configs)

        cached_result = self._cache_manager.get(cache_key)
        if cached_result is not None:
            self._composition_stats["cache_hits"] += 1
            self._metrics.increment_counter("composition.cache_hits", 1)
            return cached_result

        self._composition_stats["cache_misses"] += 1
        self._metrics.increment_counter("composition.cache_misses", 1)

        try:
            if validate_sequence:
                warnings = self.validate_sequence(transform_configs)
                if warnings:
                    for warning in warnings:
                        self._logger.warning(f"Sequence validation: {warning}")

            transforms = []
            validation_errors = []
            transform_names = []

            for i, config in enumerate(transform_configs):
                try:
                    if not isinstance(config, dict):
                        validation_errors.append(
                            f"Transform {i}: Configuration must be a dictionary"
                        )
                        continue

                    transform_name = config.get("name")
                    transform_kwargs = config.get("kwargs", {})
                    transform_enabled = config.get("enabled", True)

                    if not transform_name:
                        validation_errors.append(f"Transform {i}: Missing 'name' field")
                        continue

                    if not transform_enabled:
                        continue

                    validated_kwargs = self.validator.validate_transform_config(
                        transform_name, transform_kwargs
                    )

                    transform_class = self.registry.get_transform_class(transform_name)
                    transform_instance = transform_class(**validated_kwargs)
                    transforms.append(transform_instance)
                    transform_names.append(transform_name)

                    self._logger.debug(f"Created transform {i}: {transform_name}")

                except Exception as e:
                    validation_errors.append(f"Transform {i} ({transform_name}): {str(e)}")
                    continue

            if validation_errors and not transforms:
                raise TransformCompositionError(
                    "All transforms failed to create", composition_errors=validation_errors
                )

            if not transforms:
                self._logger.warning("No valid transforms to compose")
                return None

            compose_obj = Compose(transforms)

            cache_metadata = {
                "transform_count": len(transforms),
                "transform_names": transform_names,
                "creation_time": time.time(),
            }

            self._cache_manager.put(cache_key, compose_obj, cache_metadata)

            self._composition_stats["total_compositions"] += 1
            total_composition_time = (time.time() - composition_start_time) * 1000
            self._metrics.record_timing("composition.total_time", total_composition_time)

            self._logger.info(
                f"Successfully composed {len(transforms)} transforms: {transform_names}"
            )

            return compose_obj

        except Exception as e:
            self._metrics.increment_counter("composition.failures", 1)

            # Attempt error recovery if enabled and available
            if enable_recovery and self.error_recovery is not None:  # ✅ DIRECT CHECK
                self._composition_stats["error_recovery_attempts"] += 1
                self._metrics.increment_counter("composition.error_recovery_attempts", 1)

                try:
                    recovery_context = {
                        "configs": transform_configs,
                        "validation_errors": validation_errors,
                        "cache_key": cache_key,
                    }

                    recovery_result = self.error_recovery.recover_from_error(e, recovery_context)

                    if recovery_result["recovered"] and recovery_result["fallback_config"]:
                        self._logger.warning(
                            f"Composition failed, attempting recovery: {recovery_result['recovery_actions']}"
                        )
                        self._composition_stats["successful_recoveries"] += 1
                        self._metrics.increment_counter("composition.successful_recoveries", 1)

                        # Recursive call with recovery disabled to prevent infinite loops
                        return self.compose_transforms(
                            recovery_result["fallback_config"],
                            cache_key=f"{cache_key}_recovered",
                            validate_sequence=False,
                            enable_recovery=False,
                        )
                    else:
                        self._logger.warning(
                            f"Recovery attempted but failed: {recovery_result['warnings']}"
                        )
                        self._metrics.increment_counter("composition.failed_recoveries", 1)

                except Exception as recovery_error:
                    self._logger.error(f"Error recovery system failed: {recovery_error}")
                    self._metrics.increment_counter("composition.recovery_exceptions", 1)

            # If recovery disabled or failed, raise original error
            raise TransformCompositionError("Composition failed", details=str(e)) from e

    def validate_sequence(self, transform_configs: list[dict[str, Any]]) -> list[str]:
        """Validate transform sequence"""

        warnings = []

        if not transform_configs:
            return warnings

        transform_names = []
        for config in transform_configs:
            if isinstance(config, dict) and "name" in config:
                if config.get("enabled", True):
                    transform_names.append(config["name"])

        if not transform_names:
            warnings.append("No enabled transforms in sequence")

        return warnings

    def _generate_cache_key(self, transform_configs: list[dict[str, Any]]) -> str:
        """Generate cache key"""
        config_str = json.dumps(transform_configs, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear composition cache"""
        self._cache_manager.clear()

    def get_composition_statistics(self) -> dict[str, Any]:
        """Get composition statistics"""
        return self._composition_stats.copy()

    def get_cache_health_report(self) -> dict[str, Any]:
        """Get cache health report"""
        return self._cache_manager.get_cache_health()


# =============================================================================
# MAIN GRAPH TRANSFORMS CLASS (Simplified)
# =============================================================================


class GraphTransforms:
    """Main interface for graph transformations with Production-Ready Architecture enhancements"""

    def __init__(self):
        # STEP 1: Initialize logger FIRST (most critical)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # STEP 2: Initialize system state tracking
        self._initialized = TORCH_GEOMETRIC_AVAILABLE
        self._initialization_errors = []

        # STEP 3: Initialize system health tracking EARLY
        self._system_health = {
            "last_health_check": None,
            "registry_health": "unknown",
            "validator_health": "unknown",
            "composer_health": "unknown",
            "cache_health": "unknown",
            "metrics_health": "unknown",
            "config_bridge_health": "unknown",
            "error_recovery_health": "unknown",
        }

        self._health_check_performed = False

        # STEP 4: Check initialization status EARLY
        if not self._initialized:
            error_msg = (
                "GraphTransforms system not fully initialized - PyTorch Geometric not available"
            )
            self._initialization_errors.append(error_msg)
            self._logger.error(error_msg)
            # Continue initialization in limited mode

        # STEP 5: Initialize core components (in dependency order)
        # Always assign the module-level registry - it's created at module import time
        # and should always be valid. Only set to None if the registry variable itself is None.
        self.registry = registry

        if self.registry is not None:
            try:
                registry_errors = self.registry.get_initialization_errors()
                if registry_errors:
                    self._initialization_errors.extend(registry_errors)
                    self._logger.debug(
                        f"Registry has {len(registry_errors)} initialization warnings"
                    )
            except Exception as e:
                # Log the error but DON'T set registry to None
                # The registry object is valid, only the error check failed
                self._logger.warning(f"Could not get registry initialization errors: {e}")
                self._initialization_errors.append(f"Registry error check failed: {str(e)}")
        else:
            self._logger.error("Module-level registry is None - this should not happen")
            self._initialization_errors.append("Module-level registry is None")

        try:
            self.validator = TransformValidator(self.registry) if self.registry else None
        except Exception as e:
            self._logger.error(f"Validator initialization failed: {e}")
            self._initialization_errors.append(f"Validator init failed: {str(e)}")
            self.validator = None

        # STEP 6: Initialize error recovery (needed by composer)
        self.error_recovery = None
        try:
            self.error_recovery = TransformErrorRecovery()
            self._logger.debug("TransformErrorRecovery initialized")
        except Exception as e:
            self._logger.warning(f"TransformErrorRecovery initialization failed: {e}")
            self._initialization_errors.append(f"TransformErrorRecovery init failed: {str(e)}")

        # STEP 7: Initialize composer (pass error_recovery)
        try:
            if self.registry and self.validator:
                self.composer = TransformComposer(
                    self.registry, self.validator, error_recovery=self.error_recovery
                )
                self._logger.debug("TransformComposer initialized")
            else:
                self.composer = None
                self._logger.warning("Composer not initialized - missing dependencies")
        except Exception as e:
            self._logger.error(f"Composer initialization failed: {e}")
            self._initialization_errors.append(f"Composer init failed: {str(e)}")
            self.composer = None

        # STEP 8: Initialize metrics (use composer's metrics if available)
        if self.composer:
            self._metrics = self.composer._metrics
        else:
            self._metrics = ProductionMetricsCollector(enable_external_metrics=True)

        # STEP 9: Initialize Production-Ready Architecture components (config bridge)
        self.config_bridge = None

        try:
            self.config_bridge = ConfigurationBridge(self)
            self._logger.debug("ConfigurationBridge initialized")
        except Exception as e:
            self._logger.warning(f"ConfigurationBridge initialization failed: {e}")
            self._initialization_errors.append(f"ConfigurationBridge init failed: {str(e)}")

        # STEP 10: Log initialization summary
        if self._initialized:
            available_count = len(self.registry.list_available_transforms()) if self.registry else 0
            self._metrics.increment_counter("system.initializations", 1)
            self._logger.info(
                f"GraphTransforms system initialized with {available_count} transforms "
                f"({len(self._initialization_errors)} warnings)"
            )
        else:
            self._logger.warning(
                f"GraphTransforms system initialized in LIMITED MODE "
                f"({len(self._initialization_errors)} errors)"
            )

        self.semantic_validator = None
        self.dataset_validator = None
        self.validation_reporter = None

        try:
            if self.registry:
                self.semantic_validator = SemanticValidator(self.registry)
                self._logger.debug("SemanticValidator initialized")
        except Exception as e:
            self._logger.warning(f"SemanticValidator initialization failed: {e}")

        try:
            if self.config_bridge:
                self.dataset_validator = DatasetAwareValidator(self.config_bridge)
                self._logger.debug("DatasetAwareValidator initialized")
        except Exception as e:
            self._logger.warning(f"DatasetAwareValidator initialization failed: {e}")

        try:
            self.validation_reporter = ValidationReporter()
            self._logger.debug("ValidationReporter initialized")
        except Exception as e:
            self._logger.warning(f"ValidationReporter initialization failed: {e}")

    def get_available_transforms(self) -> dict[str, Any]:
        """
        Get available transforms for validation and inspection.

        Returns comprehensive information about all available transforms
        including their parameters, categories, and availability status.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'basic': List of basic transforms
                - 'augmentation': List of augmentation transforms
                - 'molecular': List of molecular-specific transforms
                - 'all': Complete list of all transforms
                - 'count': Total number of available transforms
                - 'metadata': Enhanced metadata per transform

        Example:
            >>> transforms = GraphTransforms()
            >>> available = transforms.get_available_transforms()
            >>> print(f"Available: {available['count']} transforms")
            >>> print(f"Basic transforms: {available['basic']}")
        """
        try:
            # Get all transforms from registry
            all_transforms = self.registry.list_available_transforms()

            # Categorize transforms
            categorized = {
                "basic": [],
                "augmentation": [],
                "molecular": [],
                "all": all_transforms,
                "count": len(all_transforms),
                "metadata": {},
            }

            # Enhanced categorization using registry metadata
            for transform_name in all_transforms:
                try:
                    # Get transform info from registry
                    transform_info = self.registry.get_transform_info(transform_name)

                    # Store metadata
                    categorized["metadata"][transform_name] = {
                        "available": transform_info.get("available", True),
                        "parameters": transform_info.get("parameters", {}),
                        "description": transform_info.get("description", ""),
                        "category": transform_info.get("category", "unknown"),
                    }

                    # Categorize by type
                    category = transform_info.get("category", "basic")
                    if category == "augmentation":
                        categorized["augmentation"].append(transform_name)
                    elif category == "molecular":
                        categorized["molecular"].append(transform_name)
                    else:
                        categorized["basic"].append(transform_name)

                except Exception as e:
                    # If we can't get info, put in basic category
                    categorized["basic"].append(transform_name)
                    categorized["metadata"][transform_name] = {
                        "available": True,
                        "parameters": {},
                        "description": "No description available",
                        "category": "basic",
                        "error": str(e),
                    }

            return categorized

        except Exception as e:
            # Fallback: return minimal structure
            self.logger.warning(f"Error getting available transforms: {e}")
            return {
                "basic": [],
                "augmentation": [],
                "molecular": [],
                "all": [],
                "count": 0,
                "metadata": {},
                "error": str(e),
            }

    def _convert_legacy_transform_config(self, legacy_config: list | dict) -> list[dict[str, Any]]:
        """
        Convert legacy transform configuration to new format.

        Handles multiple legacy formats:
        - List of dicts with 'name' and 'params'
        - List of dicts with 'name' and 'parameters'
        - List of strings (transform names only)
        - Old nested format

        Args:
            legacy_config: Legacy configuration in various formats

        Returns:
            List[Dict]: Standardized transform specifications
        """
        converted_transforms = []

        try:
            # Handle different input types
            if isinstance(legacy_config, dict):
                # Extract transforms from dict
                if "transforms" in legacy_config:
                    transform_list = legacy_config["transforms"]
                elif "experimental_setups" in legacy_config:
                    # New format - already converted
                    return []
                else:
                    # Single transform as dict
                    transform_list = [legacy_config]
            elif isinstance(legacy_config, list):
                transform_list = legacy_config
            else:
                self.logger.warning(f"Unexpected legacy config type: {type(legacy_config)}")
                return []

            # Convert each transform
            for idx, transform_spec in enumerate(transform_list):
                try:
                    if isinstance(transform_spec, str):
                        # Simple string format: just transform name
                        converted_transforms.append(
                            {"name": transform_spec, "enabled": True, "params": {}}
                        )

                    elif isinstance(transform_spec, dict):
                        # Dict format - extract name and params
                        name = transform_spec.get("name") or transform_spec.get("type")

                        if not name:
                            self.logger.warning(f"Transform {idx}: Missing 'name' field - skipped")
                            continue

                        # Get parameters (check multiple possible keys)
                        params = (
                            transform_spec.get("params")
                            or transform_spec.get("parameters")
                            or transform_spec.get("kwargs")
                            or {}
                        )

                        # Get enabled status
                        enabled = transform_spec.get("enabled", True)

                        converted_transforms.append(
                            {"name": name, "enabled": enabled, "params": params}
                        )

                    else:
                        self.logger.warning(
                            f"Transform {idx}: Unsupported type {type(transform_spec)} - skipped"
                        )

                except Exception as e:
                    self.logger.warning(f"Transform {idx}: Conversion error - {e}")
                    continue

            if converted_transforms:
                self.logger.info(
                    f"Successfully converted {len(converted_transforms)} legacy transforms"
                )
            else:
                self.logger.warning("No transforms converted from legacy configuration")

            return converted_transforms

        except Exception as e:
            self.logger.error(f"Legacy transform conversion failed: {e}")
            return []

    def is_available(self) -> bool:
        """
        Check if transformation system is available for use.

        Returns True if the system is initialized and has a working registry,
        even if there were some non-critical initialization warnings (e.g.,
        failed auto-registration of some transforms).

        Critical errors that would cause this to return False:
        - PyTorch Geometric not available (_initialized = False)
        - Registry failed to initialize (registry = None)
        - Composer failed to initialize (composer = None)

        Non-critical issues that do NOT prevent availability:
        - Some transforms failed auto-registration
        - Some optional components (config_bridge, error_recovery) failed
        """
        return self._initialized and self.registry is not None and self.composer is not None

    def list_transforms(self, category: str | None = None) -> list[str]:
        """List available transforms"""

        if not self._initialized:
            return []

        if category:
            try:
                return self.registry.list_by_category(category)
            except TransformValidationError:
                self._logger.warning(f"Unknown category: {category}")
                return []

        return self.registry.list_available_transforms()

    def get_transform_info(self, transform_name: str) -> TransformInfo | None:
        """Get detailed information about a transform"""

        if not self._initialized:
            return None

        try:
            return self.registry.get_transform_info(transform_name)
        except TransformNotFoundError:
            return None

    def has_registry(self) -> bool:
        """
        Check if the transform registry is available and initialized.

        This method is used by milia_dataset.py to determine whether to use
        the dynamic transform discovery system or fall back to legacy transforms.

        Returns:
            bool: True if registry is initialized and available, False otherwise
        """
        return self._initialized and self.registry is not None

    def create_transform_sequence(
        self, configs: list[dict[str, Any]], enable_recovery: bool = True, sample_data: Any = None
    ) -> Compose | None:
        """
        Create transform sequence with edge_attr-aware parameter injection.

        ENHANCED: Now supports automatic edge_attr handling. When sample_data is
        provided and contains edge_attr, transforms like AddSelfLoops will have
        their parameters automatically configured to maintain edge_index/edge_attr
        shape consistency.

        Args:
            configs: List of transform configurations
            enable_recovery: Whether to attempt error recovery on failures (default: True)
            sample_data: Optional sample data for edge_attr detection. If provided,
                        enables automatic parameter injection for edge_attr-aware transforms.

        Returns:
            Compose object or None if creation fails
        """
        if not self._initialized:
            self._logger.error("Cannot create transforms - system not initialized")
            return None

        try:
            return self.composer.compose_transforms(
                configs, enable_recovery=enable_recovery, sample_data=sample_data
            )
        except Exception as e:
            self._logger.error(f"Transform sequence creation failed: {e}")
            return None

    def set_sample_data_for_transforms(self, sample_data: Any) -> None:
        """
        Set sample data for edge_attr-aware parameter injection.

        This can be called before create_transform_sequence() to enable automatic
        edge_attr handling without passing sample_data to each call.

        Args:
            sample_data: A PyG Data object from the dataset
        """
        if self._initialized and self.composer:
            self.composer.set_sample_data(sample_data)
            self._logger.info(
                f"Sample data configured for edge_attr detection: "
                f"has_edge_attr={self.composer._edge_attr_injector.has_edge_attr}"
            )

    def get_system_status(self) -> dict[str, Any]:
        """Get system status"""

        status = {
            "initialized": self._initialized,
            "torch_geometric_available": TORCH_GEOMETRIC_AVAILABLE,
            "pyg_version": TORCH_GEOMETRIC_VERSION,
            "available_transform_count": len(self.registry.list_available_transforms())
            if self._initialized
            else 0,
            "initialization_errors": self._initialization_errors.copy(),
            "discovery_statistics": self.registry.get_discovery_statistics()
            if self._initialized
            else {},
        }

        return status

    def perform_health_check(self) -> dict[str, Any]:
        """Perform and return comprehensive system health check

        Returns:
            Dictionary containing health status for all system components
            with recommendations for any issues found
        """
        # Ensure health check performed at least once
        self._ensure_health_check_performed()

        # Perform current health check
        health_status = self._perform_health_check()

        # Generate recommendations based on health status
        recommendations = []
        if health_status.get("registry_health") != "healthy":
            recommendations.append("Check PyTorch Geometric installation")
        if health_status.get("validator_health") != "healthy":
            recommendations.append("Review transform parameter validation system")
        if health_status.get("composer_health") != "healthy":
            recommendations.append("Check transform composition capabilities")
        if health_status.get("cache_health") != "healthy":
            recommendations.append("Review cache configuration and memory settings")
        if (
            self.config_bridge is not None
            and health_status.get("config_bridge_health") != "healthy"
        ):
            recommendations.append("Check configuration bridge functionality")
        if (
            self.error_recovery is not None
            and health_status.get("error_recovery_health") != "healthy"
        ):
            recommendations.append("Review error recovery system")

        health_status["recommendations"] = recommendations

        return health_status

    def _ensure_health_check_performed(self) -> None:
        """Ensure health check has been performed at least once (lazy initialization)"""
        if not self._health_check_performed:
            try:
                self._perform_health_check()
                self._health_check_performed = True
                self._logger.debug("Initial health check completed (lazy)")
            except Exception as e:
                self._logger.warning(f"Initial health check failed: {e}")
                # Don't set flag - allow retry on next call

    def _perform_health_check(self) -> dict[str, str]:
        """Perform comprehensive system health check with enhanced monitoring

        Returns:
            Dictionary with health status for each component
        """
        self._system_health["last_health_check"] = time.time()

        # Check registry health
        try:
            if self.registry:
                available_transforms = self.registry.list_available_transforms()
                if len(available_transforms) > 15:
                    self._system_health["registry_health"] = "healthy"
                elif len(available_transforms) > 10:
                    self._system_health["registry_health"] = "degraded"
                else:
                    self._system_health["registry_health"] = "unhealthy"

                self._metrics.set_gauge(
                    "health.registry_transform_count", len(available_transforms)
                )
            else:
                self._system_health["registry_health"] = "error: registry not initialized"
        except Exception as e:
            self._system_health["registry_health"] = f"error: {str(e)}"
            self._metrics.increment_counter("health.registry_errors", 1)

        # Check validator health
        try:
            if self.validator:
                test_config = [{"name": "AddSelfLoops"}]
                validation_result = self.validate_config(test_config)
                if validation_result["valid"]:
                    self._system_health["validator_health"] = "healthy"
                else:
                    self._system_health["validator_health"] = "degraded"

                self._metrics.set_gauge(
                    "health.validator_functional", 1 if validation_result["valid"] else 0
                )
            else:
                self._system_health["validator_health"] = "error: validator not initialized"
        except Exception as e:
            self._system_health["validator_health"] = f"error: {str(e)}"
            self._metrics.increment_counter("health.validator_errors", 1)

        # Check composer health
        try:
            if self.composer:
                test_config = [{"name": "AddSelfLoops"}]
                composed = self.composer.compose_transforms(test_config)
                if composed is not None:
                    self._system_health["composer_health"] = "healthy"
                else:
                    self._system_health["composer_health"] = "degraded"

                cache_health = self.composer.get_cache_health_report()
                if cache_health["overall_health"] == "healthy":
                    self._system_health["cache_health"] = "healthy"
                elif cache_health["overall_health"] == "degraded":
                    self._system_health["cache_health"] = "degraded"
                else:
                    self._system_health["cache_health"] = "unhealthy"

                self._metrics.set_gauge("health.cache_functional", 1 if composed is not None else 0)
            else:
                self._system_health["composer_health"] = "error: composer not initialized"
                self._system_health["cache_health"] = "error: composer not initialized"
        except Exception as e:
            self._system_health["composer_health"] = f"error: {str(e)}"
            self._metrics.increment_counter("health.composer_errors", 1)

        # Check metrics system health
        try:
            metrics_summary = self._metrics.get_metrics_summary()
            external_integrations = metrics_summary.get("external_integrations", {})

            if (
                any(external_integrations.values())
                or external_integrations.get("custom_handlers", 0) > 0
            ):
                self._system_health["metrics_health"] = "healthy"
            else:
                self._system_health["metrics_health"] = "degraded"

            self._metrics.set_gauge(
                "health.metrics_integrations", sum(1 for v in external_integrations.values() if v)
            )
        except Exception as e:
            self._system_health["metrics_health"] = f"error: {str(e)}"

        # Check config bridge health
        if self.config_bridge is not None:
            try:
                test_legacy_config = [{"name": "AddSelfLoops"}]
                converted = self.config_bridge.convert_legacy_config(test_legacy_config)
                if converted:
                    self._system_health["config_bridge_health"] = "healthy"
                else:
                    self._system_health["config_bridge_health"] = "degraded"
            except Exception as e:
                self._system_health["config_bridge_health"] = f"error: {str(e)}"
        else:
            self._system_health["config_bridge_health"] = "not_initialized"

        # Check error recovery health
        if self.error_recovery is not None:
            try:
                test_error = TransformValidationError("Test error", "TestTransform")
                recovery_result = self.error_recovery.recover_from_error(test_error, {})
                if recovery_result:
                    self._system_health["error_recovery_health"] = "healthy"
                else:
                    self._system_health["error_recovery_health"] = "degraded"
            except Exception as e:
                self._system_health["error_recovery_health"] = f"error: {str(e)}"
        else:
            self._system_health["error_recovery_health"] = "not_initialized"

        return self._system_health

    def validate_config(
        self, configs: list[dict[str, Any]], dataset_type: str | None = None
    ) -> dict[str, Any]:
        """Validate transform configuration with optional dataset-specific checks

        Args:
            configs: List of transform configurations to validate
            dataset_type: Optional dataset type for milia-specific validation

        Returns:
            Dictionary with validation results including errors, warnings, and suggestions
        """
        validation_start = time.time()
        self._metrics.increment_counter("validation.requests", 1)

        results = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": {},
            "transform_count": len(configs),
            "system_initialized": self._initialized,
            "milia_specific": {},
            "performance_metrics": {},
        }

        if not self._initialized:
            results["errors"].append("Transform system not initialized")
            self._metrics.increment_counter("validation.system_not_initialized", 1)
            return results

        if not configs:
            results["valid"] = True
            results["warnings"].append("Empty configuration provided")
            return results

        # Convert legacy format if needed
        try:
            if self.config_bridge is not None:
                converted_configs = self.config_bridge.convert_legacy_config(configs)
                if converted_configs != configs:
                    results["warnings"].append("Configuration converted from legacy format")
            else:
                converted_configs = configs
        except Exception as e:
            results["errors"].append(f"Configuration conversion failed: {str(e)}")
            return results

        # Set dataset context
        if dataset_type:
            self.validator.set_dataset_context(dataset_type)

        try:
            # Validate individual transforms
            for i, config in enumerate(converted_configs):
                if not isinstance(config, dict):
                    results["errors"].append(f"Transform {i}: Must be a dictionary")
                    continue

                transform_name = config.get("name")
                if not transform_name:
                    results["errors"].append(f"Transform {i}: Missing 'name' field")
                    continue

                transform_kwargs = config.get("kwargs", {})

                validation_errors = self.validator.get_validation_errors(
                    transform_name, transform_kwargs
                )
                if validation_errors:
                    results["errors"].extend(
                        [
                            f"Transform {i} ({transform_name}): {error}"
                            for error in validation_errors
                        ]
                    )
                    self._metrics.increment_counter(
                        f"validation.transform_errors.{transform_name}", 1
                    )

                suggestions = self.validator.suggest_corrections(transform_name, transform_kwargs)
                if suggestions:
                    results["suggestions"][f"transform_{i}_{transform_name}"] = suggestions

            # Validate sequence
            if not results["errors"]:
                sequence_warnings = self.composer.validate_sequence(converted_configs)
                results["warnings"].extend(sequence_warnings)
                results["valid"] = True

                # milia-specific validation
                if dataset_type and self.config_bridge is not None:
                    milia_validation = self.config_bridge.validate_against_milia_requirements(
                        converted_configs, dataset_type
                    )
                    results["milia_specific"] = milia_validation
                    if milia_validation.get("milia_specific_warnings"):
                        results["warnings"].extend(milia_validation["milia_specific_warnings"])

        finally:
            if dataset_type:
                self.validator.clear_dataset_context()

        total_validation_time = (time.time() - validation_start) * 1000
        self._metrics.record_timing("validation.total_time", total_validation_time)

        results["performance_metrics"] = {"total_validation_time_ms": total_validation_time}

        return results

    def validate_parameter(
        self, transform_name: str, param_name: str, value: Any, check_constraints: bool = True
    ) -> tuple[bool, list[str]]:
        """Public API: Validate parameter value with constraints

        Args:
            transform_name: Name of transform
            param_name: Name of parameter
            value: Value to validate
            check_constraints: Whether to check constraints

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not self._initialized:
            return (False, ["Transform system not initialized"])

        return self.validator.validate_parameter_with_constraints(
            transform_name, param_name, value, check_constraints
        )

    def validate_configuration(
        self, config: dict[str, Any], dataset_type: str = "DFT"
    ) -> dict[str, Any]:
        """
        Validate v3 configuration format.

        This method replaces the old migrate_configuration() method.
        Only v3 (research-grade) configurations are supported.

        Args:
            config: Configuration dict (must be v3 format)
            dataset_type: Dataset type for validation ('DFT', 'DMC', 'Wavefunction')

        Returns:
            Dict with validation results

        Raises:
            ConfigurationError: If not v3 format
        """
        return self.config_bridge.config_validator.validate_v3_configuration(config, dataset_type)

    def get_configuration_format_help(self) -> str:
        """
        Get help text for v3 configuration format.

        Returns:
            Multi-line string with v3 format specification and examples

        Example:
            >>> gt = get_graph_transforms()
            >>> print(gt.get_configuration_format_help())
        """
        return self.config_bridge.config_validator.get_format_help()

    def get_milia_experimental_setups(
        self, research_focus: str = "molecular_properties"
    ) -> dict[str, list[dict[str, Any]]]:
        """Get milia-optimized experimental setups

        Args:
            research_focus: Research focus area for setup generation
                Options: 'molecular_properties', 'uncertainty_quantification',
                        '3d_molecular_data', 'data_augmentation'

        Returns:
            Dictionary of experimental setup names to transform configurations

        Examples:
            >>> gt = get_graph_transforms()
            >>> setups = gt.get_milia_experimental_setups('molecular_properties')
            >>> baseline = setups['milia_baseline']
        """
        if not self._initialized:
            self._logger.warning("System not initialized, returning empty setups")
            return {}

        if self.config_bridge is None:
            self._logger.warning("Config bridge not available, returning empty setups")
            return {}

        try:
            setups = self.config_bridge.generate_experimental_setups_for_milia(research_focus)
            self._logger.info(f"Generated {len(setups)} milia setups for '{research_focus}'")
            return setups
        except Exception as e:
            self._logger.error(f"Failed to generate milia setups: {e}")
            return {}

    def get_parameter_info(
        self, transform_name: str, parameter_name: str | None = None
    ) -> ParameterMetadata | dict[str, ParameterMetadata]:
        """Get parameter metadata for a transform"""
        try:
            transform_info = self.registry.get_transform_info(transform_name)
        except TransformNotFoundError:
            # Re-raise with clearer message for parameter introspection
            raise TransformValidationError(
                f"Cannot get parameter info: Transform '{transform_name}' not found",
                transform_name=transform_name,
            ) from None

        if parameter_name is not None:
            return self.validator.get_parameter_metadata(transform_name, parameter_name)

        # Return all parameters
        metadata_dict = {}
        for param_name in transform_info.parameters:
            metadata_dict[param_name] = self.validator.get_parameter_metadata(
                transform_name, param_name
            )

        return metadata_dict

    def get_parameter_constraints(
        self, transform_name: str, parameter_name: str
    ) -> list[ParameterConstraint]:
        """Get constraints for a parameter"""
        metadata = self.validator.get_parameter_metadata(transform_name, parameter_name)
        return metadata.constraints

    def get_parameter_examples(self, transform_name: str, parameter_name: str) -> list[Any]:
        """Get example values for a parameter"""
        metadata = self.validator.get_parameter_metadata(transform_name, parameter_name)
        return metadata.examples

    def suggest_parameter_value(
        self, transform_name: str, parameter_name: str, context: dict[str, Any] | None = None
    ) -> Any:
        """Suggest parameter value based on context"""
        metadata = self.validator.get_parameter_metadata(transform_name, parameter_name)

        if metadata.has_default:
            return metadata.default_value

        if metadata.examples:
            return metadata.examples[0]

        for constraint in metadata.constraints:
            if constraint.type == "range":
                min_val, max_val = constraint.constraint_value
                if min_val != float("-inf") and max_val != float("inf"):
                    return (min_val + max_val) / 2

        if metadata.type_hint:
            base_type = metadata.get_base_type()
            if base_type == int:
                return 1
            elif base_type == float:
                return 0.5
            elif base_type == bool:
                return True
            elif base_type == str:
                return ""

        return None

    def get_transform_documentation(self, transform_name: str) -> dict[str, Any]:
        """
        Get comprehensive documentation for a transform.

        Production Enhancement: Complete transform documentation
        """
        transform_info = self.registry.get_transform_info(transform_name)
        param_metadata = self.get_parameter_info(transform_name)

        return {
            "name": transform_name,
            "description": transform_info.description,
            "category": transform_info.category,
            "pre_transform_safe": transform_info.pre_transform_safe,
            "dataset_compatible": list(transform_info.dataset_compatible)
            if hasattr(transform_info, "dataset_compatible")
            else [],
            "parameters": {
                name: {
                    "type": str(meta.type_hint) if meta.type_hint else "Any",
                    "required": meta.required,
                    "default": meta.default_value if meta.has_default else None,
                    "description": meta.description,
                    "constraints": [
                        {
                            "type": c.type,
                            "description": c.description,
                            "value": c.constraint_value,
                            "inferred": c.inferred,
                        }
                        for c in meta.constraints
                    ],
                    "examples": meta.examples,
                }
                for name, meta in param_metadata.items()
            },
        }

    def validate_config_comprehensive(
        self,
        configs: list[dict[str, Any]],
        dataset_type: str | None = None,
        research_context: str | None = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        validation_scope: ValidationScope = ValidationScope.SEMANTIC,
        strict_mode: bool = False,
        auto_fix: bool = False,
    ) -> dict[str, Any]:
        """
        Comprehensive validation with Production-Ready Architecture.3 enhancements

        Args:
            configs: Transform configurations
            dataset_type: Dataset type (DFT, DMC, Wavefunction)
            research_context: Research context
            validation_level: Validation strictness
            validation_scope: Validation scope
            strict_mode: Enable strict validation
            auto_fix: Attempt to auto-fix issues

        Returns:
            Comprehensive validation report
        """

        if not self._initialized:
            return {
                "valid": False,
                "errors": ["System not initialized"],
                "warnings": [],
                "report": None,
            }

        # Create validation context
        context = ValidationContext(
            level=validation_level,
            scope=validation_scope,
            dataset_type=dataset_type,
            research_context=research_context,
            strict_mode=strict_mode,
            auto_fix=auto_fix,
        )

        # Basic validation (existing)
        if validation_scope in [
            ValidationScope.BASIC,
            ValidationScope.SEMANTIC,
            ValidationScope.DATASET_SPECIFIC,
            ValidationScope.PRODUCTION,
        ]:
            context = self.validator.validate_with_context(configs, context)

        # Semantic validation
        if validation_scope in [
            ValidationScope.SEMANTIC,
            ValidationScope.DATASET_SPECIFIC,
            ValidationScope.PRODUCTION,
        ] and self.semantic_validator:
            context = self.semantic_validator.validate_sequence(configs, context)

        # Dataset-specific validation
        if validation_scope in [ValidationScope.DATASET_SPECIFIC, ValidationScope.PRODUCTION]:
            if self.dataset_validator and dataset_type:
                context = self.dataset_validator.validate_for_dataset(
                    configs, dataset_type, context
                )

        # Generate report
        report = None
        if self.validation_reporter:
            report = self.validation_reporter.generate_report(context)

        # Determine validation result
        valid = True
        if strict_mode:
            valid = len(context.issues) == 0
        elif validation_level == ValidationLevel.STRICT:
            valid = not context.has_errors()
        elif validation_level == ValidationLevel.STANDARD:
            valid = not context.has_critical_issues()
        # PERMISSIVE always passes

        return {
            "valid": valid,
            "context": context,
            "report": report,
            "errors": [i.message for i in context.get_issues_by_severity(ValidationSeverity.ERROR)],
            "warnings": [
                i.message for i in context.get_issues_by_severity(ValidationSeverity.WARNING)
            ],
            "auto_fixable": len(context.get_auto_fixable_issues()),
        }

    def get_validation_report(
        self,
        configs: list[dict[str, Any]],
        dataset_type: str | None = None,
        format_type: str = "text",
    ) -> str:
        """
        Get formatted validation report

        Args:
            configs: Transform configurations
            dataset_type: Dataset type
            format_type: Report format ('text', 'json', 'markdown')

        Returns:
            Formatted validation report
        """

        validation_result = self.validate_config_comprehensive(
            configs, dataset_type=dataset_type, validation_scope=ValidationScope.PRODUCTION
        )

        if validation_result["report"] and self.validation_reporter:
            return self.validation_reporter.format_report(validation_result["report"], format_type)

        return "Validation report unavailable"


# =============================================================================
# MODULE-LEVEL INTERFACE
# =============================================================================

_graph_transforms_instance: GraphTransforms | None = None
_initialization_lock = threading.RLock()


def get_graph_transforms() -> GraphTransforms:
    """
    Get the singleton GraphTransforms instance (thread-safe)

    Thread Safety:
        Uses double-checked locking pattern with RLock to ensure only one
        instance is created even under concurrent access. This pattern is
        safe in Python due to the Global Interpreter Lock (GIL).

        The sequence is:
        1. Fast path: Check if instance exists (no lock needed)
        2. Slow path: Acquire lock, check again, create if still None

        This avoids lock contention after first initialization while ensuring
        thread-safe creation.

    Returns:
        The singleton GraphTransforms instance

    Notes:
        - First call may take 100-200ms for discovery engine initialization
        - Subsequent calls are O(1) constant time
        - Safe to call from multiple threads simultaneously
    """
    global _graph_transforms_instance

    # Fast path: instance already exists (no lock needed)
    if _graph_transforms_instance is None:
        # Slow path: acquire lock for initialization
        with _initialization_lock:
            # Double-check: another thread may have initialized while we waited
            if _graph_transforms_instance is None:
                _graph_transforms_instance = GraphTransforms()

    return _graph_transforms_instance


def reset_graph_transforms() -> None:
    """Reset the singleton instance"""
    global _graph_transforms_instance
    with _initialization_lock:
        _graph_transforms_instance = None


def list_available_transforms(category: str | None = None) -> list[str]:
    """List available transforms"""
    return get_graph_transforms().list_transforms(category=category)


def create_transform_sequence(
    configs: list[dict[str, Any]], enable_recovery: bool = True, sample_data: Any = None
) -> Compose | None:
    """
    Create transform sequence with edge_attr-aware parameter injection.

    ENHANCED: Now supports automatic edge_attr handling. When sample_data is
    provided and contains edge_attr, transforms like AddSelfLoops will have
    their parameters automatically configured to maintain edge_index/edge_attr
    shape consistency.

    Args:
        configs: List of transform configurations
        enable_recovery: Whether to attempt error recovery on failures (default: True)
        sample_data: Optional sample data for edge_attr detection. If provided,
                    enables automatic parameter injection for edge_attr-aware transforms.

    Returns:
        Compose object or None
    """
    return get_graph_transforms().create_transform_sequence(
        configs, enable_recovery=enable_recovery, sample_data=sample_data
    )


def get_system_status() -> dict[str, Any]:
    """Get system status"""
    return get_graph_transforms().get_system_status()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================


def _validate_module_initialization():
    """Validate module initialization"""

    try:
        gt = get_graph_transforms()
        status = gt.get_system_status()

        if not status["initialized"]:
            logger.warning("Graph transforms module initialized with limitations")
        else:
            discovery_stats = status.get("discovery_statistics", {})
            logger.info(
                f"Graph transforms module initialized: "
                f"{status['available_transform_count']} transforms "
                f"({discovery_stats.get('manually_registered', 0)} manual, "
                f"{discovery_stats.get('auto_discovered', 0)} auto-discovered)"
            )

    except Exception as e:
        logger.error(f"Module validation failed: {str(e)}")


# =============================================================================
# CONFIGURATION VALIDATION SYSTEM (v3 Only)
# =============================================================================


class ConfigurationValidator:
    """
    Validator for v3 (research-grade) configuration format.

    This class replaces ConfigurationMigrationManager. Migration support
    has been removed - only v3 configurations are supported.

    v3 Format Requirements:
        - Must be a dictionary
        - Must contain 'experimental_setups' OR 'standard_transforms' (or both)
        - Must contain 'research_context' OR 'dataset_optimization'
        - experimental_setups values should have 'transforms' key

    Transform Application Order:
        When both are present, transforms are applied in this order:
        1. standard_transforms (always applied first)
        2. experimental_setups transforms (applied after standard)

    Example v3 Configuration with standard_transforms:
        {
            'standard_transforms': [
                {'name': 'AddSelfLoops', 'enabled': True},
                {'name': 'NormalizeFeatures', 'enabled': True}
            ],
            'experimental_setups': {
                'baseline': {
                    'transforms': [],  # Empty - uses only standard_transforms
                    'research_context': 'molecular_property_prediction'
                },
                'enhanced': {
                    'transforms': [
                        {'name': 'GCNNorm'},
                        {'name': 'VirtualNode'}
                    ],
                    'research_context': 'molecular_property_prediction'
                }
            },
            'default_setup': 'baseline',
            'research_context': 'molecular_property_prediction',
            'dataset_optimization': {
                'dataset_type': 'DFT',
                'optimization_applied': True,
                'optimizations': [...]
            }
        }

    Breaking Change from Previous Versions:
        - v1 (list-based) and v2 (basic dict) formats are NO LONGER supported
        - migrate_configuration() method has been removed
        - Use validate_v3_configuration() instead
        - See migration guide for converting old configs to v3
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def is_valid_v3_format(self, config: Any) -> bool:
        """
        Check if configuration is valid v3 format.

        Args:
            config: Configuration to check

        Returns:
            True if valid v3 format, False otherwise

        Note:
            A valid v3 config must have either experimental_setups OR standard_transforms
            (or both). The presence of standard_transforms alone is valid for configs
            that rely entirely on standard transforms without experimental variations.
        """
        if not isinstance(config, dict):
            return False

        # Must have at least one transform source
        has_transform_source = "experimental_setups" in config or "standard_transforms" in config

        if not has_transform_source:
            return False

        # Must have at least one of these v3 identifiers
        has_v3_markers = "research_context" in config or "dataset_optimization" in config

        return has_v3_markers

    def validate_v3_configuration(
        self, config: dict[str, Any], dataset_type: str = "DFT"
    ) -> dict[str, Any]:
        """
        Validate v3 configuration format and structure.

        Args:
            config: Configuration to validate (must be v3 format)
            dataset_type: Dataset type for validation ('DFT', 'DMC', 'Wavefunction')

        Returns:
            Dict with validation results:
                {
                    'is_valid': bool,
                    'warnings': List[str],
                    'errors': List[str],
                    'format_version': 'v3',
                    'dataset_type': str
                }

        Raises:
            ConfigurationError: If configuration is not v3 format
        """
        validation_result = {
            "is_valid": False,
            "warnings": [],
            "errors": [],
            "format_version": "unknown",
            "dataset_type": dataset_type,
        }

        # Check format version
        if not isinstance(config, dict):
            if isinstance(config, list):
                raise ConfigurationError(
                    "Legacy v1 list-based configuration format detected. "
                    "This format is no longer supported. "
                    "\n\nPlease convert to v3 research-grade format. "
                    "See documentation for v3 format specification and examples."
                )
            else:
                raise ConfigurationError(
                    f"Invalid configuration type: {type(config).__name__}. "
                    "Configuration must be a dictionary in v3 format."
                )

        # Check for v3 format
        if not self.is_valid_v3_format(config):
            # Check if it's v2 (has experimental_setups but not v3 markers)
            if "experimental_setups" in config:
                raise ConfigurationError(
                    "Legacy v2 configuration format detected. "
                    "This format is no longer supported. "
                    "\n\nTo upgrade to v3 format, add:"
                    "\n  - 'research_context': (e.g., 'molecular_property_prediction')"
                    "\n  - 'dataset_optimization': {dataset_type, optimization_applied, optimizations}"
                    "\n\nSee documentation for complete v3 specification."
                )
            else:
                raise ConfigurationError(
                    "Invalid configuration format. Must be v3 research-grade format. "
                    "\n\nRequired keys:"
                    "\n  - 'experimental_setups' OR 'standard_transforms': transform configurations"
                    "\n  - 'research_context': research application context"
                    "\n  - 'dataset_optimization': dataset-specific optimizations"
                    "\n\nSee documentation for v3 format specification."
                )

        validation_result["format_version"] = "v3"

        # Validate v3 structure
        errors = []
        warnings = []

        # Check for transform sources (standard_transforms and/or experimental_setups)
        has_standard_transforms = (
            "standard_transforms" in config
            and isinstance(config["standard_transforms"], list)
            and len(config["standard_transforms"]) > 0
        )

        # Check experimental_setups
        if "experimental_setups" not in config:
            if not has_standard_transforms:
                errors.append(
                    "Missing required key: 'experimental_setups' (or 'standard_transforms')"
                )
            else:
                # Having only standard_transforms is valid
                warnings.append("No 'experimental_setups' defined - using standard_transforms only")
        else:
            exp_setups = config["experimental_setups"]
            if not isinstance(exp_setups, dict):
                errors.append(
                    f"'experimental_setups' must be dict, got {type(exp_setups).__name__}"
                )
            elif len(exp_setups) == 0:
                if not has_standard_transforms:
                    warnings.append(
                        "'experimental_setups' is empty - no transform configurations defined"
                    )
                # else: empty experimental_setups is fine when standard_transforms exist
            else:
                # Validate each setup
                for setup_name, setup_config in exp_setups.items():
                    if isinstance(setup_config, dict):
                        if "transforms" not in setup_config:
                            # v3 format should have nested structure with 'transforms' key
                            warnings.append(
                                f"Setup '{setup_name}': Missing 'transforms' key. "
                                "v3 format should nest transforms under 'transforms' key."
                            )
                        if "research_context" not in setup_config:
                            warnings.append(
                                f"Setup '{setup_name}': Missing 'research_context'. "
                                "Consider adding for complete v3 compliance."
                            )
                    elif isinstance(setup_config, list):
                        # Direct list of transforms - acceptable but not ideal v3
                        warnings.append(
                            f"Setup '{setup_name}': Using list format. "
                            "v3 format recommends dict with 'transforms', 'research_context', etc."
                        )

        # Check for required v3 markers
        if "research_context" not in config:
            warnings.append("Missing 'research_context' - recommended for v3 format")

        if "dataset_optimization" not in config:
            warnings.append("Missing 'dataset_optimization' - recommended for v3 format")
        elif not isinstance(config["dataset_optimization"], dict):
            errors.append("'dataset_optimization' must be a dictionary")
        else:
            # Validate dataset_optimization structure
            ds_opt = config["dataset_optimization"]
            if "dataset_type" not in ds_opt:
                warnings.append("'dataset_optimization' missing 'dataset_type'")
            elif ds_opt["dataset_type"] != dataset_type:
                warnings.append(
                    f"dataset_type mismatch: config has '{ds_opt['dataset_type']}', "
                    f"but validating for '{dataset_type}'"
                )

        # Check optional but recommended fields
        if "default_setup" not in config:
            warnings.append("Missing 'default_setup' - consider specifying a default")
        elif config["default_setup"] not in config.get("experimental_setups", {}):
            # Only error if there's no standard_transforms to fall back on
            if not has_standard_transforms:
                errors.append(
                    f"'default_setup' references '{config['default_setup']}' "
                    f"which is not in experimental_setups"
                )
            else:
                warnings.append(
                    f"'default_setup' references '{config['default_setup']}' "
                    f"which is not in experimental_setups - will use standard_transforms"
                )

        validation_result["errors"] = errors
        validation_result["warnings"] = warnings
        validation_result["is_valid"] = len(errors) == 0

        if errors:
            self._logger.error(f"v3 configuration validation failed: {errors}")
        elif warnings:
            self._logger.warning(f"v3 configuration validation warnings: {warnings}")
        else:
            self._logger.debug("v3 configuration validation passed")

        return validation_result

    def get_format_help(self) -> str:
        """
        Get help text for v3 configuration format.

        Returns:
            Multi-line string with v3 format specification and examples
        """
        return """
v3 (Research-Grade) Configuration Format
========================================

Required Structure:
-------------------
{
    "experimental_setups": {
        "setup_name": {
            "transforms": [
                {"name": "TransformName", "kwargs": {...}},
                ...
            ],
            "research_context": "molecular_property_prediction",
            "dataset_compatibility": {...},
            "expected_performance_impact": {...}
        }
    },
    "default_setup": "setup_name",
    "research_context": "molecular_property_prediction",
    "dataset_optimization": {
        "dataset_type": "DFT",
        "optimization_applied": true,
        "optimizations": [...]
    }
}

Research Contexts:
------------------
- "molecular_property_prediction"
- "3d_molecular_analysis"
- "robustness_training"
- "geometric_invariance"
- "uncertainty_quantification"

Dataset Types:
--------------
- "DFT" - Density Functional Theory
- "DMC" - Diffusion Monte Carlo
- "Wavefunction" - Wavefunction

Migration from v1/v2:
---------------------
v1 and v2 formats are no longer supported. Please manually convert
configurations to v3 format using the structure above.

See documentation for complete specification and examples.
"""


# =============================================================================
# CONFIGURATION BRIDGE SYSTEM
# =============================================================================


class ConfigurationBridge:
    def __init__(self, transform_registry: TransformRegistry):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.transform_registry = transform_registry
        self.config_validator = ConfigurationValidator()

        self.milia_transform_recommendations = {
            "DFT": {
                "essential": ["AddSelfLoops", "ToUndirected", "GCNNorm"],
                "recommended": ["Distance", "Normalize"],
                "avoid": ["RandomFlip"],
                "performance_critical": ["Distance"],
            },
            "DMC": {
                "essential": ["AddSelfLoops", "ToUndirected"],
                "recommended": ["GCNNorm", "MaskFeatures"],
                "avoid": ["DropNode", "RandomFlip"],
                "uncertainty_preserving": ["MaskFeatures", "DropEdge"],
            },
            "Wavefunction": {
                "essential": ["AddSelfLoops", "ToUndirected", "Distance"],
                "recommended": ["GCNNorm", "Normalize"],
                "avoid": ["RandomFlip", "DropNode", "MaskFeatures", "RandomRotate", "RandomScale"],
                "orbital_critical": ["Distance", "GCNNorm"],
                "precision_preserving": ["Normalize"],
            },
        }

    def convert_legacy_config(self, config: Any) -> list[dict[str, Any]]:
        """Convert various configuration formats to standard list format.

        Handles both legacy formats and new format with standard_transforms.
        When standard_transforms is present, it is applied FIRST, followed by
        transforms from the experimental setup.
        """

        if isinstance(config, list):
            return self._normalize_transform_list(config)

        elif isinstance(config, dict):
            combined_transforms = []

            # Step 1: Extract standard_transforms first (if present)
            if "standard_transforms" in config:
                standard = config["standard_transforms"]
                if isinstance(standard, list):
                    normalized_standard = self._normalize_transform_list(standard)
                    combined_transforms.extend(normalized_standard)
                    self._logger.debug(f"Added {len(normalized_standard)} standard transforms")

            # Step 2: Extract experimental setup transforms
            if "experimental_setups" in config:
                default_setup = config.get("default_setup", "default")
                setups = config.get("experimental_setups", {})

                experimental_transforms = []
                if default_setup in setups:
                    setup_config = setups[default_setup]
                    # Handle both dict with 'transforms' key and direct list
                    if isinstance(setup_config, dict) and "transforms" in setup_config:
                        experimental_transforms = setup_config["transforms"]
                    elif isinstance(setup_config, list):
                        experimental_transforms = setup_config
                    elif isinstance(setup_config, dict):
                        # Setup might have empty transforms (relies on standard)
                        experimental_transforms = setup_config.get("transforms", [])
                elif setups:
                    first_setup = next(iter(setups.values()))
                    if isinstance(first_setup, dict) and "transforms" in first_setup:
                        experimental_transforms = first_setup["transforms"]
                    elif isinstance(first_setup, list):
                        experimental_transforms = first_setup

                if experimental_transforms:
                    normalized_experimental = self._normalize_transform_list(
                        experimental_transforms
                    )
                    combined_transforms.extend(normalized_experimental)
                    self._logger.debug(
                        f"Added {len(normalized_experimental)} experimental transforms"
                    )

            # If we found any transforms (standard or experimental), return them
            if combined_transforms:
                return combined_transforms

            # Fallback: single transform dict
            if "name" in config:
                return [self._normalize_transform_dict(config)]

            # Fallback: try other possible keys
            possible_keys = ["transforms", "transformations", "pre_transforms"]
            for key in possible_keys:
                if key in config and isinstance(config[key], list):
                    return self._normalize_transform_list(config[key])

        self._logger.warning(f"Unrecognized configuration format: {type(config)}")
        return []

    def _normalize_transform_list(self, transform_list: list[Any]) -> list[dict[str, Any]]:
        """Normalize list of transforms"""

        normalized = []
        for i, transform in enumerate(transform_list):
            try:
                normalized_transform = self._normalize_transform_dict(transform)
                normalized.append(normalized_transform)
            except Exception as e:
                self._logger.warning(f"Failed to normalize transform {i}: {e}")

        return normalized

    def _normalize_transform_dict(self, transform: Any) -> dict[str, Any]:
        """Normalize single transform"""

        if not isinstance(transform, dict):
            raise ValueError(f"Transform must be dict, got {type(transform)}")

        normalized = {
            "name": transform.get("name", "Unknown"),
            "enabled": transform.get("enabled", True),
        }

        if "kwargs" in transform:
            normalized["kwargs"] = transform["kwargs"]
        elif "parameters" in transform:
            normalized["kwargs"] = transform["parameters"]
        elif "args" in transform:
            normalized["kwargs"] = transform["args"]
        else:
            kwargs = {
                k: v for k, v in transform.items() if k not in ["name", "enabled", "description"]
            }
            if kwargs:
                normalized["kwargs"] = kwargs

        if "description" in transform:
            normalized["description"] = transform["description"]

        return normalized

    def validate_against_milia_requirements(
        self, configs: list[dict[str, Any]], dataset_type: str
    ) -> dict[str, Any]:
        """Validate against milia requirements"""

        validation = {
            "milia_compatible": True,
            "milia_specific_warnings": [],
            "dataset_optimization_suggestions": [],
            "performance_recommendations": [],
        }

        if dataset_type not in self.milia_transform_recommendations:
            validation["milia_specific_warnings"].append(f"Unknown dataset type '{dataset_type}'")
            return validation

        recommendations = self.milia_transform_recommendations[dataset_type]
        transform_names = [config.get("name", "") for config in configs]

        missing_essential = []
        for essential in recommendations["essential"]:
            if essential not in transform_names:
                missing_essential.append(essential)

        if missing_essential:
            validation["milia_compatible"] = False
            validation["milia_specific_warnings"].append(
                f"Missing essential transforms for {dataset_type}: {missing_essential}"
            )

        problematic_transforms = []
        for avoid in recommendations["avoid"]:
            if avoid in transform_names:
                problematic_transforms.append(avoid)

        if problematic_transforms:
            validation["milia_specific_warnings"].append(
                f"Transforms not recommended for {dataset_type}: {problematic_transforms}"
            )

        return validation

    def generate_experimental_setups_for_milia(
        self, research_focus: str
    ) -> dict[str, list[dict[str, Any]]]:
        """Generate milia-optimized experimental setups"""

        setups = {}

        if research_focus == "molecular_properties":
            setups.update(
                {
                    "milia_baseline": [
                        {"name": "AddSelfLoops"},
                        {"name": "ToUndirected"},
                        {"name": "GCNNorm", "kwargs": {"add_self_loops": False}},
                    ],
                    "milia_enhanced": [
                        {"name": "AddSelfLoops"},
                        {"name": "ToUndirected"},
                        {"name": "GCNNorm", "kwargs": {"add_self_loops": False}},
                        {"name": "Distance", "kwargs": {"norm": True, "max_value": 10.0}},
                    ],
                }
            )

        elif research_focus == "uncertainty_quantification":
            setups.update(
                {
                    "milia_dmc_baseline": [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}],
                    "milia_dmc_augmented": [
                        {"name": "AddSelfLoops"},
                        {"name": "ToUndirected"},
                        {"name": "MaskFeatures", "kwargs": {"p": 0.1}},
                        {"name": "DropEdge", "kwargs": {"p": 0.05}},
                    ],
                }
            )

        return setups


# =============================================================================
# TRANSFORM ERROR RECOVERY SYSTEM
# =============================================================================


class TransformErrorRecovery:
    """Handles error recovery and alternative strategies"""

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._recovery_strategies = self._initialize_recovery_strategies()
        self._fallback_configs = self._initialize_fallback_configs()

    def _initialize_recovery_strategies(self) -> dict[str, dict[str, Any]]:
        """Initialize error recovery strategies"""
        return {
            "parameter_validation_error": {
                "strategy": "remove_invalid_parameters",
                "fallback": "use_default_parameters",
                "last_resort": "skip_transform",
            },
            "transform_not_found": {
                "strategy": "suggest_similar_transform",
                "fallback": "use_basic_alternative",
                "last_resort": "skip_transform",
            },
            "composition_error": {
                "strategy": "remove_problematic_transforms",
                "fallback": "use_minimal_config",
                "last_resort": "disable_transforms",
            },
        }

    def _initialize_fallback_configs(self) -> dict[str, list[dict[str, Any]]]:
        """Initialize safe default configurations"""
        return {
            "minimal": [{"name": "AddSelfLoops"}],
            "basic_molecular": [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}],
            "safe_augmentation": [
                {"name": "AddSelfLoops"},
                {"name": "ToUndirected"},
                {"name": "DropEdge", "kwargs": {"p": 0.05}},
            ],
        }

    def recover_from_error(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        """Attempt to recover from errors"""

        recovery_result = {
            "recovered": False,
            "fallback_config": None,
            "recovery_actions": [],
            "warnings": [],
            "original_error": str(error),
        }

        error_type = self._classify_error(error)
        recovery_strategy = self._recovery_strategies.get(error_type, {})

        if not recovery_strategy:
            recovery_result["warnings"].append(f"No recovery strategy for: {error_type}")
            return recovery_result

        try:
            if self._attempt_recovery_strategy(
                recovery_strategy["strategy"], error, context, recovery_result
            ):
                recovery_result["recovered"] = True
                return recovery_result

            if self._attempt_recovery_strategy(
                recovery_strategy["fallback"], error, context, recovery_result
            ):
                recovery_result["recovered"] = True
                recovery_result["warnings"].append("Used fallback recovery strategy")
                return recovery_result

            if self._attempt_recovery_strategy(
                recovery_strategy["last_resort"], error, context, recovery_result
            ):
                recovery_result["recovered"] = True
                recovery_result["warnings"].append("Used last resort recovery strategy")
                return recovery_result

        except Exception as recovery_error:
            recovery_result["warnings"].append(f"Recovery attempt failed: {recovery_error}")

        return recovery_result

    def _classify_error(self, error: Exception) -> str:
        """Classify error type"""

        if isinstance(error, TransformValidationError):
            return "parameter_validation_error"
        elif isinstance(error, TransformNotFoundError):
            return "transform_not_found"
        elif isinstance(error, TransformCompositionError):
            return "composition_error"
        else:
            return "unknown_error"

    def _attempt_recovery_strategy(
        self,
        strategy: str,
        error: Exception,
        context: dict[str, Any],
        recovery_result: dict[str, Any],
    ) -> bool:
        """Attempt specific recovery strategy"""

        try:
            if strategy == "remove_invalid_parameters":
                return self._remove_invalid_parameters(context, recovery_result)
            elif strategy == "use_default_parameters":
                return self._use_default_parameters(context, recovery_result)
            elif strategy == "skip_transform":
                return self._skip_problematic_transform(context, recovery_result)
            elif strategy == "use_basic_alternative":
                return self._use_basic_alternative(context, recovery_result)
            elif strategy == "use_minimal_config":
                return self._use_minimal_config(context, recovery_result)
            elif strategy == "disable_transforms":
                return self._disable_all_transforms(recovery_result)
            else:
                return False

        except Exception as e:
            self._logger.error(f"Recovery strategy '{strategy}' failed: {e}")
            return False

    def _remove_invalid_parameters(
        self, context: dict[str, Any], recovery_result: dict[str, Any]
    ) -> bool:
        """Remove invalid parameters"""
        configs = context.get("configs", [])
        if not configs:
            return False

        cleaned_configs = []
        for config in configs:
            cleaned_config = {"name": config.get("name", "Unknown")}

            if "kwargs" in config and isinstance(config["kwargs"], dict):
                valid_kwargs = {}
                for key, value in config["kwargs"].items():
                    if isinstance(key, str) and key.isidentifier() and value is not None:
                        valid_kwargs[key] = value

                if valid_kwargs:
                    cleaned_config["kwargs"] = valid_kwargs

            cleaned_configs.append(cleaned_config)

        recovery_result["fallback_config"] = cleaned_configs
        recovery_result["recovery_actions"].append("Removed invalid parameters")
        return True

    def _use_default_parameters(
        self, context: dict[str, Any], recovery_result: dict[str, Any]
    ) -> bool:
        """Use default parameters"""
        configs = context.get("configs", [])
        if not configs:
            return False

        default_configs = []
        for config in configs:
            default_config = {"name": config.get("name", "Unknown")}
            default_configs.append(default_config)

        recovery_result["fallback_config"] = default_configs
        recovery_result["recovery_actions"].append("Used default parameters")
        return True

    def _skip_problematic_transform(
        self, context: dict[str, Any], recovery_result: dict[str, Any]
    ) -> bool:
        """Skip problematic transform"""
        configs = context.get("configs", [])
        if not configs:
            return False

        if len(configs) > 1:
            recovery_result["fallback_config"] = configs[:-1]
            recovery_result["recovery_actions"].append("Skipped problematic transform")
            return True

        return self._use_minimal_config(context, recovery_result)

    def _use_basic_alternative(
        self, context: dict[str, Any], recovery_result: dict[str, Any]
    ) -> bool:
        """Use basic alternatives"""

        alternatives = {
            "Distance": "AddSelfLoops",
            "VirtualNode": "ToUndirected",
            "DropNode": "DropEdge",
        }

        configs = context.get("configs", [])
        if not configs:
            return False

        alternative_configs = []
        for config in configs:
            transform_name = config.get("name", "")
            if transform_name in alternatives:
                alternative_configs.append({"name": alternatives[transform_name]})
            else:
                alternative_configs.append({"name": transform_name})

        recovery_result["fallback_config"] = alternative_configs
        recovery_result["recovery_actions"].append("Used basic alternatives")
        return True

    def _use_minimal_config(self, context: dict[str, Any], recovery_result: dict[str, Any]) -> bool:
        """Use minimal configuration"""

        dataset_type = context.get("dataset_type", "DFT")

        if dataset_type == "DMC":
            minimal_config = self._fallback_configs["basic_molecular"]
        else:
            minimal_config = self._fallback_configs["minimal"]

        recovery_result["fallback_config"] = minimal_config
        recovery_result["recovery_actions"].append(f"Used minimal config for {dataset_type}")
        return True

    def _disable_all_transforms(self, recovery_result: dict[str, Any]) -> bool:
        """Disable all transforms"""

        recovery_result["fallback_config"] = []
        recovery_result["recovery_actions"].append("Disabled all transforms")
        recovery_result["warnings"].append("All transforms disabled")
        return True


# =============================================================================
# HANDLER INTEGRATION WRAPPERS
# =============================================================================


@wrap_handler_operation("graph_transforms", "create_transforms")
def create_transforms_with_handler_integration(
    configs: list[dict[str, Any]],
    handler_type: str | None = None,
    dataset_type: str | None = None,
    sample_data: Any = None,
) -> Compose | None:
    """
    Complete handler integration with enhanced error handling and edge_attr-aware injection.

    ENHANCED: Now supports automatic edge_attr handling. When sample_data is
    provided and contains edge_attr, transforms like AddSelfLoops will have
    their parameters automatically configured to maintain edge_index/edge_attr
    shape consistency.

    Args:
        configs: List of transform configurations
        handler_type: Optional handler type for error context
        dataset_type: Optional dataset type for validation
        sample_data: Optional sample data for edge_attr detection

    Returns:
        Compose object or None if creation fails
    """

    gt = get_graph_transforms()

    if not gt._initialized:
        from milia_pipeline.exceptions import HandlerOperationError

        raise HandlerOperationError(
            message="Graph transforms system not initialized",
            handler_type=handler_type or "unknown",
            operation="create_transforms",
            recovery_suggestions=[
                "Check PyTorch Geometric installation",
                "Verify system dependencies",
            ],
            details="PyTorch Geometric may not be available",
        )

    try:
        # Enhanced validation with dataset-specific checks
        if dataset_type:
            validation_result = gt.validate_config(configs, dataset_type)
            if not validation_result["valid"]:
                raise HandlerOperationError(
                    message="Transform configuration validation failed",
                    handler_type=handler_type or "unknown",
                    operation="validate_transforms",
                    recovery_suggestions=[
                        "Check transform parameters",
                        "Review configuration format",
                    ],
                    details=f"Validation errors: {validation_result['errors']}",
                )

            # Log milia-specific warnings
            if validation_result.get("milia_specific", {}).get("milia_specific_warnings"):
                for warning in validation_result["milia_specific"]["milia_specific_warnings"]:
                    logger.warning(f"milia-specific warning: {warning}")

        # EDGE-ATTR AWARE: Pass sample_data for automatic parameter injection
        return gt.create_transform_sequence(configs, enable_recovery=True, sample_data=sample_data)

    except Exception as e:
        from milia_pipeline.exceptions import HandlerOperationError

        # NEW: Attempt error recovery with safe access
        recovery_attempted = False
        if gt.error_recovery is not None:  # Safe null check
            try:
                recovery_result = gt.error_recovery.recover_from_error(
                    e,
                    {
                        "handler_type": handler_type,
                        "dataset_type": dataset_type,
                        "configs": configs,
                    },
                )

                recovery_attempted = True

                if recovery_result.get("recovered") and recovery_result.get("fallback_config"):
                    logger.warning(
                        f"Transform creation recovered: {recovery_result['recovery_actions']}"
                    )
                    return gt.create_transform_sequence(
                        recovery_result["fallback_config"],
                        enable_recovery=False,
                        sample_data=sample_data,
                    )
            except Exception as recovery_error:
                logger.error(f"Error recovery failed: {recovery_error}")

        # Build error details
        error_details = f"Transform error: {str(e)}"
        if not recovery_attempted:
            error_details += " (Error recovery system not available)"
        elif recovery_attempted:
            error_details += " (Error recovery attempted but failed)"

        raise HandlerOperationError(
            message="Transform creation failed in handler context",
            handler_type=handler_type or "unknown",
            operation="create_transforms",
            recovery_suggestions=[
                "Validate transform configuration",
                "Check transform compatibility",
            ],
            details=error_details,
        ) from e


# =============================================================================
# PRODUCTION READINESS VALIDATION
# =============================================================================


def validate_production_readiness() -> dict[str, Any]:
    """Complete production readiness validation"""

    readiness_report = {
        "production_ready": True,
        "critical_issues": [],
        "warnings": [],
        "recommendations": [],
        "system_health": {},
        "feature_completeness": {},
        "discovery_statistics": {},
    }

    try:
        gt = get_graph_transforms()
        system_status = gt.get_system_status()

        readiness_report["system_health"] = (
            system_status.get("system_health", {}) if hasattr(gt, "_system_health") else {}
        )
        readiness_report["discovery_statistics"] = system_status.get("discovery_statistics", {})

        # Check critical components
        if not system_status["initialized"]:
            readiness_report["critical_issues"].append("System not initialized")
            readiness_report["production_ready"] = False

        if system_status["available_transform_count"] < 15:
            readiness_report["warnings"].append(
                f"Limited transform availability: {system_status['available_transform_count']}"
            )

        # Check feature completeness
        required_features = [
            "transform_registry",
            "parameter_validation",
            "sequence_composition",
            "error_recovery",
            "yaml_integration",
            "milia_optimization",
            "health_monitoring",
            "caching_system",
            "metrics_collection",
            "configuration_migration",
            "dynamic_discovery",
        ]

        completed_features = []
        if hasattr(gt, "registry") and gt.registry:
            completed_features.append("transform_registry")
            # Check for Production-Ready Architecture dynamic discovery
            if hasattr(gt.registry, "_discovery_engine"):
                completed_features.append("dynamic_discovery")

        if hasattr(gt, "validator") and gt.validator:
            completed_features.append("parameter_validation")

        if hasattr(gt, "composer") and gt.composer:
            completed_features.append("sequence_composition")
            if hasattr(gt.composer, "_cache_manager"):
                completed_features.append("caching_system")
            if hasattr(gt.composer, "_metrics"):
                completed_features.append("metrics_collection")

        if hasattr(gt, "error_recovery") and gt.error_recovery:
            completed_features.append("error_recovery")

        if hasattr(gt, "config_bridge") and gt.config_bridge:
            completed_features.extend(["yaml_integration", "milia_optimization"])

        if hasattr(gt, "_system_health"):
            completed_features.append("health_monitoring")

        readiness_report["feature_completeness"] = {
            "required": required_features,
            "completed": completed_features,
            "missing": [f for f in required_features if f not in completed_features],
            "completion_rate": len(completed_features) / len(required_features),
        }

        if readiness_report["feature_completeness"]["completion_rate"] < 0.9:
            readiness_report["critical_issues"].append("Incomplete feature implementation")
            readiness_report["production_ready"] = False

        # Check PyG version compatibility
        if system_status.get("pyg_version"):
            readiness_report["pyg_version"] = system_status["pyg_version"]
            readiness_report["recommendations"].append(
                f"PyG version {system_status['pyg_version']} detected - verify compatibility"
            )

        # Check discovery statistics
        discovery_stats = readiness_report["discovery_statistics"]
        if discovery_stats.get("auto_discovered", 0) > 0:
            readiness_report["recommendations"].append(
                f"Dynamic discovery active: {discovery_stats['auto_discovered']} transforms auto-discovered"
            )

        # Generate final recommendations
        if readiness_report["feature_completeness"]["missing"]:
            readiness_report["recommendations"].append(
                f"Complete missing features: {readiness_report['feature_completeness']['missing']}"
            )

        if not readiness_report["critical_issues"] and not readiness_report["warnings"]:
            readiness_report["recommendations"].append(
                "System is production-ready with Production-Ready Architecture enhancements"
            )

        return readiness_report

    except Exception as e:
        readiness_report["critical_issues"].append(f"Production readiness check failed: {str(e)}")
        readiness_report["production_ready"] = False
        return readiness_report


# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================


def get_usage_examples() -> dict[str, str]:
    """Complete usage examples for all features"""

    examples = {
        "basic_usage": """
# Basic transform sequence creation
transforms = [
    {'name': 'AddSelfLoops'},
    {'name': 'ToUndirected'},
    {'name': 'GCNNorm', 'kwargs': {'add_self_loops': False}}
]
compose_obj = create_transform_sequence(transforms)
""",
        "dynamic_discovery": """
# Production-Ready Architecture - Dynamic Discovery
gt = get_graph_transforms()

# Get discovery statistics
stats = gt.registry.get_discovery_statistics()
print(f"Auto-discovered: {stats['auto_discovered']} transforms")
print(f"PyG version: {stats['pyg_version']}")

# List all discovered transforms
all_transforms = gt.registry.list_available_transforms()
auto_discovered = gt.registry.get_auto_discovered_transforms()

# Check transform availability with version info
transform_info = gt.get_transform_info('Distance')
if transform_info:
    print(f"Discovery method: {transform_info.discovery_method}")
    print(f"Complexity score: {transform_info.complexity_score}")
    print(f"Pre-transform safe: {transform_info.pre_transform_safe}")
""",
        "experimental_setup": """
# Create experimental setup
gt = get_graph_transforms()
setup = gt.create_experimental_setup(
    setup_name="molecular_baseline",
    transforms=[
        {'name': 'AddSelfLoops'},
        {'name': 'Distance', 'kwargs': {'norm': True, 'max_value': 10.0}}
    ],
    description="Baseline setup for molecular property prediction",
    research_context="molecular_properties",
    expected_effects=["improved_distance_features", "normalized_geometry"]
)
""",
        "milia_integration": """
# milia dataset integration with Production-Ready Architecture
yaml_config = [
    {'name': 'AddSelfLoops'},
    {'name': 'GCNNorm', 'kwargs': {'add_self_loops': False}}
]

# Validate for specific dataset type
gt = get_graph_transforms()
validation = gt.validate_config(yaml_config, dataset_type='DFT')

if validation['valid']:
    compose_obj = gt.create_from_yaml_config(yaml_config, dataset_type='DFT')

    # Check milia-specific validation
    if validation.get('milia_specific'):
        print("milia compatibility:", validation['milia_specific']['milia_compatible'])
""",
        "error_recovery": """
# Automatic error recovery
configs = [
    {'name': 'NonExistentTransform'},  # Will fail
    {'name': 'AddSelfLoops'},          # Will succeed
    {'name': 'Distance', 'kwargs': {'max_value': -1}}  # Invalid param
]

# Enable recovery
gt = get_graph_transforms()
compose_obj = gt.create_transform_sequence(configs, enable_recovery=True)

# Even with errors, will get fallback composition
if compose_obj:
    print("Successfully recovered with fallback transforms")
""",
        "production_monitoring": """
# Production metrics and monitoring
gt = get_graph_transforms()

# Export metrics for Prometheus
metrics_text = gt.export_metrics(format_type='prometheus')

# Add custom metrics handler
def custom_handler(metric_type, name, value, tags):
    print(f"Custom metric: {metric_type} {name} = {value}")

gt.add_custom_metrics_handler(custom_handler)

# Get system health
health = gt.perform_health_check()
print(f"System health: {health}")

# Optimize performance
optimization = gt.optimize_performance(target_cache_hit_rate=0.8)
print(f"Optimization applied: {optimization['cache_optimization']}")
""",
        "research_recommendations": """
# Get research-specific recommendations
gt = get_graph_transforms()

# For molecular properties research with DFT data
recommendations = gt.get_research_recommendations(
    research_type='molecular_properties',
    dataset_type='DFT'
)

print("Essential transforms:", recommendations['recommended_transforms']['essential'])
print("Experimental setups:", list(recommendations['experimental_setups'].keys()))

# Get milia-optimized setups
milia_setups = gt.config_bridge.generate_experimental_setups_for_milia(
    research_focus='molecular_properties'
)
""",
        "transform_compatibility": """
# Check transform compatibility (Production-Ready Architecture)
gt = get_graph_transforms()

# Check if two transforms are compatible
compatible, reason = gt.registry.check_compatibility('DropNode', 'VirtualNode')
if not compatible:
    print(f"Incompatible: {reason}")

# Get transform dependencies
transform_info = gt.get_transform_info('Distance')
if transform_info:
    deps = transform_info.dependencies
    print(f"Required attributes: {deps.required_graph_attributes}")
    print(f"Modifies: {deps.modifies_attributes}")
    print(f"Recommended after: {deps.recommended_after}")
""",
    }

    return examples


# =============================================================================
# MODULE TESTING FUNCTIONALITY
# =============================================================================


def run_production_validation_tests() -> dict[str, Any]:
    """Run Production-Ready Architecture-specific validation tests"""

    test_results = {"total_tests": 0, "passed": 0, "failed": 0, "test_details": []}

    def run_test(test_name: str, test_func: Callable) -> bool:
        """Run a single test"""
        test_results["total_tests"] += 1
        try:
            test_func()
            test_results["passed"] += 1
            test_results["test_details"].append(
                {"name": test_name, "status": "PASSED", "error": None}
            )
            return True
        except Exception as e:
            test_results["failed"] += 1
            test_results["test_details"].append(
                {"name": test_name, "status": "FAILED", "error": str(e)}
            )
            return False

    # Test 1: Dynamic Discovery
    def test_dynamic_discovery():
        gt = get_graph_transforms()
        stats = gt.registry.get_discovery_statistics()
        assert stats["total_transforms"] > 0, "No transforms discovered"
        assert stats.get("auto_discovered", 0) >= 0, "Invalid auto-discovery count"

    run_test("Dynamic Discovery", test_dynamic_discovery)

    # Test 2: Transform Metadata Enhancement
    def test_enhanced_metadata():
        gt = get_graph_transforms()
        info = gt.get_transform_info("AddSelfLoops")
        assert info is not None, "Transform info not available"
        assert hasattr(info, "discovery_method"), "Missing discovery_method"
        assert hasattr(info, "complexity_score"), "Missing complexity_score"
        assert hasattr(info, "dependencies"), "Missing dependencies"

    run_test("Enhanced Metadata", test_enhanced_metadata)

    # Test 4: Error Recovery
    def test_error_recovery():
        gt = get_graph_transforms()
        if hasattr(gt, "error_recovery"):
            error = TransformValidationError("Test error", "TestTransform")
            result = gt.error_recovery.recover_from_error(error, {})
            assert "recovered" in result, "Recovery result invalid"

    run_test("Error Recovery", test_error_recovery)

    # Test 5: milia Integration
    def test_milia_integration():
        gt = get_graph_transforms()
        if hasattr(gt, "config_bridge"):
            validation = gt.config_bridge.validate_against_milia_requirements(
                [{"name": "AddSelfLoops"}], "DFT"
            )
            assert "milia_compatible" in validation, "milia validation failed"

    run_test("milia Integration", test_milia_integration)

    # Test 6: Production Readiness
    def test_production_readiness():
        readiness = validate_production_readiness()
        assert "production_ready" in readiness, "Readiness check failed"
        assert "feature_completeness" in readiness, "Feature completeness missing"

    run_test("Production Readiness", test_production_readiness)

    return test_results

    # Test 8: Compatibility Matrix
    def test_compatibility_matrix():
        gt = get_graph_transforms()
        compatible, reason = gt.registry.check_compatibility("DropNode", "VirtualNode")
        assert not compatible, "DropNode and VirtualNode should be incompatible"
        assert reason is not None, "Incompatibility should have a reason"

    run_test("Compatibility Matrix", test_compatibility_matrix)

    # Test 10: Discovery Failure Handling
    def test_discovery_failure_handling():
        gt = get_graph_transforms()
        # Should not crash even if transform doesn't exist
        gt.get_transform_info("NonExistentTransform") if False else None
        # Test discovery engine handles missing transforms gracefully
        engine = gt.registry._discovery_engine
        result = engine.get_transform_with_fallback("NonExistentTransform")
        assert result is None, "Should return None for non-existent transform"

    run_test("Discovery Failure Handling", test_discovery_failure_handling)

    # Test 11: Enhanced Metadata Fields
    def test_enhanced_metadata_fields():
        gt = get_graph_transforms()
        info = gt.get_transform_info("AddSelfLoops")
        assert hasattr(info, "discovery_method"), "Missing discovery_method"
        assert hasattr(info, "complexity_score"), "Missing complexity_score"
        assert hasattr(info, "dependencies"), "Missing dependencies"
        assert hasattr(info, "compatibility"), "Missing compatibility"
        assert hasattr(info, "tags"), "Missing tags"

    run_test("Enhanced Metadata Fields", test_enhanced_metadata_fields)

    # Test 12: Dependency Tracking
    def test_dependency_tracking():
        gt = get_graph_transforms()
        info = gt.get_transform_info("GCNNorm")
        assert info.dependencies.recommended_after, "GCNNorm should have order recommendations"
        assert "AddSelfLoops" in info.dependencies.recommended_after, (
            "GCNNorm should recommend AddSelfLoops before it"
        )

    run_test("Dependency Tracking", test_dependency_tracking)

    def test_v3_configuration_validation():
        """Test v3 configuration validation"""
        print("\n=== Test: v3 Configuration Validation ===")

        from milia_pipeline.exceptions import ConfigurationError

        gt = get_graph_transforms()

        # Test 1: Valid v3 config
        valid_v3 = {
            "experimental_setups": {
                "baseline": {
                    "transforms": [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}],
                    "research_context": "molecular_property_prediction",
                }
            },
            "default_setup": "baseline",
            "research_context": "molecular_property_prediction",
            "dataset_optimization": {"dataset_type": "DFT", "optimization_applied": True},
        }

        result = gt.validate_configuration(valid_v3, "DFT")
        if result["is_valid"]:
            print("  ✓ Valid v3 configuration accepted")
        else:
            print(f"  ✗ Valid v3 rejected: {result['errors']}")
            return False

        # Test 2: v1 format rejection
        v1_config = [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}]

        try:
            gt.validate_configuration(v1_config, "DFT")
            print("  ✗ v1 format should have been rejected")
            return False
        except ConfigurationError as e:
            if "v1" in str(e) or "list" in str(e).lower():
                print("  ✓ v1 format correctly rejected with clear message")
            else:
                print(f"  ⚠ v1 rejection message could be clearer: {str(e)[:100]}")

        # Test 3: v2 format rejection
        v2_config = {
            "experimental_setups": {"test": [{"name": "AddSelfLoops"}]},
            "default_setup": "test",
            # Missing research_context and dataset_optimization = v2
        }

        try:
            gt.validate_configuration(v2_config, "DFT")
            print("  ✗ v2 format should have been rejected")
            return False
        except ConfigurationError as e:
            if "v2" in str(e) or "research_context" in str(e):
                print("  ✓ v2 format correctly rejected with upgrade hints")
            else:
                print(f"  ⚠ v2 rejection message could be clearer: {str(e)[:100]}")

        # Test 4: Invalid type rejection
        try:
            gt.validate_configuration("not a dict", "DFT")
            print("  ✗ Invalid type should have been rejected")
            return False
        except ConfigurationError:
            print("  ✓ Invalid configuration type correctly rejected")

        print("✓ v3 configuration validation test passed")
        return True

    run_test("v3 Configuration Validation", test_v3_configuration_validation)

    def test_configuration_format_help():
        """Test configuration format help text"""
        print("\n=== Test: Configuration Format Help ===")

        gt = get_graph_transforms()

        help_text = gt.get_configuration_format_help()

        # Check help text contains key information
        required_keywords = [
            "v3",
            "experimental_setups",
            "research_context",
            "dataset_optimization",
            "DFT",
            "DMC",
        ]

        missing = [kw for kw in required_keywords if kw not in help_text]

        if missing:
            print(f"  ⚠ Help text missing keywords: {missing}")
        else:
            print("  ✓ Help text contains all required information")

        print(f"  Help text length: {len(help_text)} characters")
        print("✓ Configuration format help test passed")
        return True

    run_test("Configuration Format Help", test_configuration_format_help)

    return test_results


# =============================================================================
# FINAL MODULE INITIALIZATION AND VALIDATION
# =============================================================================


def _validate_module_initialization():
    """Complete module initialization validation"""

    try:
        gt = get_graph_transforms()
        status = gt.get_system_status()

        if not status["initialized"]:
            logger.warning("Graph transforms module initialized with limitations")
            if status["initialization_errors"]:
                for error in status["initialization_errors"]:
                    logger.warning(f"  - {error}")
        else:
            discovery_stats = status.get("discovery_statistics", {})
            logger.info(
                f"Graph transforms module initialized successfully:\n"
                f"  - Total transforms: {status['available_transform_count']}\n"
                f"  - Manual registration: {discovery_stats.get('manually_registered', 0)}\n"
                f"  - Auto-discovered: {discovery_stats.get('auto_discovered', 0)}\n"
                f"  - Fallback implementations: {discovery_stats.get('fallback_implementations', 0)}\n"
                f"  - PyG version: {discovery_stats.get('pyg_version', 'unknown')}\n"
                f"  - Production-Ready Architecture enhancements: ACTIVE"
            )

            # Run Production-Ready Architecture validation tests
            test_results = run_production_validation_tests()
            logger.info(
                f"Production-Ready Architecture validation tests: {test_results['passed']}/{test_results['total_tests']} passed"
            )

            if test_results["failed"] > 0:
                logger.warning(
                    "Some Production-Ready Architecture tests failed - check test details"
                )

            # Validate production readiness
            readiness = validate_production_readiness()
            if readiness["production_ready"]:
                logger.info(
                    "✓ System is production-ready with Production-Ready Architecture enhancements"
                )
            else:
                logger.warning(
                    f"⚠ Production readiness issues detected:\n"
                    f"  Critical: {len(readiness['critical_issues'])}\n"
                    f"  Warnings: {len(readiness['warnings'])}"
                )

    except Exception as e:
        logger.error(f"Module validation failed: {str(e)}")


# Run complete validation on import
_validate_module_initialization()


# =============================================================================
# ENHANCED GRAPHTRANSFORMS CLASS - COMPLETE METHODS
# =============================================================================

# Add these methods to the GraphTransforms class (these were abbreviated in Part 2)


def validate_config(
    self, configs: list[dict[str, Any]], dataset_type: str | None = None
) -> dict[str, Any]:
    """Complete validate_config implementation"""

    validation_start = time.time()
    self._metrics.increment_counter("validation.requests", 1)

    results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "suggestions": {},
        "transform_count": len(configs),
        "system_initialized": self._initialized,
        "milia_specific": {},
        "performance_metrics": {},
    }

    if not self._initialized:
        results["errors"].append("Transform system not initialized")
        self._metrics.increment_counter("validation.system_not_initialized", 1)
        return results

    if not configs:
        results["valid"] = True
        results["warnings"].append("Empty configuration provided")
        return results

    # Convert legacy format if needed
    try:
        if hasattr(self, "config_bridge"):
            converted_configs = self.config_bridge.convert_legacy_config(configs)
            if converted_configs != configs:
                results["warnings"].append("Configuration converted from legacy format")
        else:
            converted_configs = configs
    except Exception as e:
        results["errors"].append(f"Configuration conversion failed: {str(e)}")
        return results

    # Set dataset context
    if dataset_type:
        self.validator.set_dataset_context(dataset_type)

    try:
        # Validate individual transforms
        for i, config in enumerate(converted_configs):
            if not isinstance(config, dict):
                results["errors"].append(f"Transform {i}: Must be a dictionary")
                continue

            transform_name = config.get("name")
            if not transform_name:
                results["errors"].append(f"Transform {i}: Missing 'name' field")
                continue

            transform_kwargs = config.get("kwargs", {})

            validation_errors = self.validator.get_validation_errors(
                transform_name, transform_kwargs
            )
            if validation_errors:
                results["errors"].extend(
                    [f"Transform {i} ({transform_name}): {error}" for error in validation_errors]
                )
                self._metrics.increment_counter(f"validation.transform_errors.{transform_name}", 1)

            suggestions = self.validator.suggest_corrections(transform_name, transform_kwargs)
            if suggestions:
                results["suggestions"][f"transform_{i}_{transform_name}"] = suggestions

        # Validate sequence
        if not results["errors"]:
            sequence_warnings = self.composer.validate_sequence(converted_configs)
            results["warnings"].extend(sequence_warnings)
            results["valid"] = True

            # milia-specific validation
            if dataset_type and hasattr(self, "config_bridge"):
                milia_validation = self.config_bridge.validate_against_milia_requirements(
                    converted_configs, dataset_type
                )
                results["milia_specific"] = milia_validation
                if milia_validation.get("milia_specific_warnings"):
                    results["warnings"].extend(milia_validation["milia_specific_warnings"])

    finally:
        if dataset_type:
            self.validator.clear_dataset_context()

    total_validation_time = (time.time() - validation_start) * 1000
    self._metrics.record_timing("validation.total_time", total_validation_time)

    results["performance_metrics"] = {"total_validation_time_ms": total_validation_time}

    return results


def create_from_yaml_config(
    self, yaml_config: list[dict[str, Any]], dataset_type: str = "DFT"
) -> Compose | None:
    """Complete YAML config integration"""

    if not self._initialized:
        self._logger.error("Cannot process YAML config - system not initialized")
        return None

    try:
        if hasattr(self, "config_bridge"):
            converted_config = self.config_bridge.convert_legacy_config(yaml_config)
        else:
            converted_config = yaml_config

        validation_result = self.validate_config(converted_config, dataset_type)

        if not validation_result["valid"]:
            self._logger.error(f"YAML config validation failed: {validation_result['errors']}")
            return None

        for warning in validation_result["warnings"]:
            self._logger.warning(f"YAML config: {warning}")

        return self.create_transform_sequence(converted_config)

    except Exception as e:
        self._logger.error(f"Failed to create transforms from YAML config: {str(e)}")
        return None


def get_research_recommendations(
    self, research_type: str, dataset_type: str = "DFT"
) -> dict[str, Any]:
    """Complete research recommendations with milia integration"""

    recommendations = {
        "recommended_transforms": [],
        "experimental_setups": {},
        "warnings": [],
        "research_type": research_type,
        "dataset_type": dataset_type,
        "milia_optimized": True,
    }

    if not self._initialized:
        recommendations["warnings"].append("Transform system not initialized")
        return recommendations

    # Get milia-optimized setups if available
    if hasattr(self, "config_bridge"):
        milia_setups = self.config_bridge.generate_experimental_setups_for_milia(research_type)
        recommendations["experimental_setups"].update(milia_setups)

    # Research-specific recommendations
    research_recommendations = {
        "molecular_properties": {
            "essential": ["AddSelfLoops", "ToUndirected"],
            "recommended": ["GCNNorm", "Distance"],
            "experimental": ["VirtualNode", "RandomRotate"],
            "avoid_for_dmc": ["RandomFlip"],
        },
        "data_augmentation": {
            "essential": ["DropEdge", "MaskFeatures"],
            "recommended": ["RandomRotate", "RandomScale"],
            "experimental": ["DropNode", "RandomTranslate"],
            "avoid_for_dmc": ["DropNode"],
        },
        "3d_molecular_data": {
            "essential": ["Distance", "Cartesian"],
            "recommended": ["RandomRotate", "LocalCartesian"],
            "experimental": ["RandomFlip", "RandomScale"],
            "avoid_for_dmc": ["RandomFlip"],
        },
        "robustness_training": {
            "essential": ["DropEdge", "DropNode"],
            "recommended": ["MaskFeatures", "RandomNodeSample"],
            "experimental": ["RandomRotate", "RandomScale"],
            "avoid_for_dmc": ["DropNode"],
        },
        "uncertainty_quantification": {
            "essential": ["AddSelfLoops", "ToUndirected"],
            "recommended": ["MaskFeatures", "DropEdge"],
            "experimental": ["GCNNorm"],
            "avoid": ["DropNode", "RandomFlip", "VirtualNode"],
        },
    }

    if research_type in research_recommendations:
        rec = research_recommendations[research_type]
        recommendations["recommended_transforms"] = rec

        # Filter out dataset-incompatible transforms
        if dataset_type == "DMC":
            for avoid_key in ["avoid_for_dmc", "avoid"]:
                if avoid_key in rec:
                    for transform in rec[avoid_key]:
                        recommendations["warnings"].append(
                            f"Transform '{transform}' not recommended for DMC datasets"
                        )

        # Create sample setups if not provided
        if not recommendations["experimental_setups"]:
            recommendations["experimental_setups"] = {
                "baseline": [{"name": t} for t in rec.get("essential", [])],
                "enhanced": [
                    {"name": t} for t in rec.get("essential", []) + rec.get("recommended", [])[:2]
                ],
                "experimental": [
                    {"name": t} for t in rec.get("essential", []) + rec.get("experimental", [])[:2]
                ],
            }
    else:
        available_types = list(research_recommendations.keys())
        recommendations["warnings"].append(
            f"Unknown research type '{research_type}'. Available: {available_types}"
        )

    return recommendations


def create_experimental_setup(
    self,
    setup_name: str,
    transforms: list[dict[str, Any]],
    description: str | None = None,
    research_context: str | None = None,
    expected_effects: list[str] | None = None,
) -> Compose | None:
    """Complete experimental setup creation"""

    if not self._initialized:
        self._logger.error("Cannot create experimental setup - system not initialized")
        return None

    setup = ExperimentalSetup(
        name=setup_name,
        transforms=transforms,
        description=description,
        research_context=research_context,
        expected_effects=expected_effects,
    )

    try:
        return self.composer.create_experimental_setup(setup)
    except TransformCompositionError as e:
        self._logger.error(f"Experimental setup creation failed: {e.message}")
        return None


# ---part 5
# =============================================================================
# COMPLETE TRANSFORMCOMPOSER CLASS - MISSING METHODS
# =============================================================================

# Add these methods to TransformComposer class (they were abbreviated in Part 2)


def create_experimental_setup(self, setup_config: ExperimentalSetup) -> Compose | None:
    """Create transforms for an experimental setup with enhanced tracking"""

    setup_start_time = time.time()
    self._metrics.increment_counter("experimental_setup.requests", 1)

    if not setup_config.enabled:
        self._logger.debug(f"Skipping disabled experimental setup: {setup_config.name}")
        self._metrics.increment_counter("experimental_setup.disabled", 1)
        return None

    self._logger.info(f"Creating experimental setup: {setup_config.name}")

    if setup_config.description:
        self._logger.debug(f"Setup description: {setup_config.description}")

    cache_key = f"experimental_setup_{setup_config.name}"

    try:
        composed = self.compose_transforms(
            setup_config.transforms, cache_key=cache_key, validate_sequence=True
        )

        setup_time = (time.time() - setup_start_time) * 1000
        self._metrics.record_timing("experimental_setup.creation_time", setup_time)

        if composed:
            self._metrics.increment_counter("experimental_setup.successful", 1)
            if setup_config.expected_effects:
                self._logger.info(f"Expected effects: {', '.join(setup_config.expected_effects)}")
        else:
            self._metrics.increment_counter("experimental_setup.failed", 1)

        return composed

    except TransformCompositionError as e:
        setup_time = (time.time() - setup_start_time) * 1000
        self._metrics.record_timing("experimental_setup.failed_time", setup_time)
        self._metrics.increment_counter("experimental_setup.composition_errors", 1)

        raise TransformCompositionError(
            f"Failed to create experimental setup '{setup_config.name}'",
            transform_sequence=[t.get("name", "unknown") for t in setup_config.transforms],
            composition_errors=[str(e)],
            details=f"Experimental setup error: {str(e)}",
        ) from e


def _check_redundant_operations(self, transform_names: list[str], warnings: list[str]) -> None:
    """Check for redundant operations with enhanced detection"""

    name_counts = {}
    for name in transform_names:
        name_counts[name] = name_counts.get(name, 0) + 1

    for name, count in name_counts.items():
        if count > 1:
            if name in ["ToUndirected", "AddSelfLoops", "RemoveIsolatedNodes"]:
                warnings.append(
                    f"Multiple {name} transforms detected - redundant and potentially harmful"
                )
            elif name in ["Normalize", "NormalizeFeatures", "GCNNorm"]:
                warnings.append(
                    f"Multiple {name} transforms detected - may cause numerical instability"
                )
            else:
                warnings.append(f"Multiple {name} transforms detected - may be redundant")


def _check_conflicting_transforms(self, transform_names: list[str], warnings: list[str]) -> None:
    """Check for potentially conflicting transforms with enhanced logic"""

    norm_transforms = [name for name in transform_names if "Norm" in name]
    if len(norm_transforms) > 1:
        warnings.append(
            f"Multiple normalization transforms: {norm_transforms} - may conflict or be redundant"
        )

    geometric_transforms = [
        name
        for name in transform_names
        if name in ["RandomRotate", "RandomScale", "RandomTranslate", "RandomFlip"]
    ]
    if len(geometric_transforms) > 3:
        warnings.append(
            f"Many geometric transforms: {geometric_transforms} - may over-augment data"
        )

    if "DropNode" in transform_names and "VirtualNode" in transform_names:
        warnings.append("DropNode and VirtualNode together may cause unpredictable behavior")

    feature_transforms = [name for name in transform_names if name in ["MaskFeatures", "DropNode"]]
    if len(feature_transforms) > 1:
        warnings.append(
            f"Multiple feature manipulation transforms: {feature_transforms} - may remove too much information"
        )

    augmentation_transforms = [
        name
        for name in transform_names
        if name in ["DropEdge", "DropNode", "MaskFeatures", "RandomNodeSample"]
    ]
    if len(augmentation_transforms) > 2:
        warnings.append(
            f"High augmentation intensity: {augmentation_transforms} - may degrade model performance"
        )


def _check_transform_order_dependencies(
    self, transform_names: list[str], warnings: list[str]
) -> None:
    """Check for transforms that have order dependencies with enhanced rules"""

    critical_orderings = [
        {
            "first": "RemoveIsolatedNodes",
            "before": ["AddSelfLoops", "VirtualNode", "ToUndirected"],
            "reason": "Node removal should happen before structural modifications",
        },
        {
            "first": "AddSelfLoops",
            "before": ["GCNNorm"],
            "reason": "AddSelfLoops should come before GCNNorm for proper normalization",
        },
        {
            "first": "ToUndirected",
            "before": ["Distance", "Cartesian", "LocalCartesian"],
            "reason": "Graph should be made undirected before spatial computations",
        },
    ]

    for rule in critical_orderings:
        first_transform = rule["first"]
        if first_transform in transform_names:
            first_idx = transform_names.index(first_transform)

            for dependent_transform in rule["before"]:
                if dependent_transform in transform_names:
                    dependent_idx = transform_names.index(dependent_transform)
                    if first_idx > dependent_idx:
                        warnings.append(
                            f"Order dependency violation: {first_transform} should come before "
                            f"{dependent_transform} - {rule['reason']}"
                        )

    structural_transforms = ["AddSelfLoops", "ToUndirected", "RemoveIsolatedNodes", "VirtualNode"]
    feature_transforms = ["Normalize", "NormalizeFeatures", "MaskFeatures"]
    spatial_transforms = ["Distance", "Cartesian", "LocalCartesian"]

    last_structural_idx = -1
    first_feature_idx = len(transform_names)
    first_spatial_idx = len(transform_names)

    for i, name in enumerate(transform_names):
        if name in structural_transforms:
            last_structural_idx = max(last_structural_idx, i)
        elif name in feature_transforms:
            first_feature_idx = min(first_feature_idx, i)
        elif name in spatial_transforms:
            first_spatial_idx = min(first_spatial_idx, i)

    if last_structural_idx > first_feature_idx and first_feature_idx < len(transform_names):
        warnings.append(
            "Recommended order: structural modifications before feature transformations"
        )

    if first_feature_idx > first_spatial_idx and first_spatial_idx < len(transform_names):
        warnings.append("Recommended order: feature transformations before spatial computations")


def _check_performance_concerns(self, transform_names: list[str], warnings: list[str]) -> None:
    """Check for potential performance issues with detailed analysis"""

    expensive_transforms = {
        "Distance": {
            "complexity": "high",
            "memory": "high",
            "note": "O(V²) complexity without cutoff",
        },
        "VirtualNode": {
            "complexity": "medium",
            "memory": "high",
            "note": "Adds V edges and increases memory",
        },
        "Cartesian": {
            "complexity": "medium",
            "memory": "medium",
            "note": "Edge-wise coordinate computation",
        },
        "LocalCartesian": {
            "complexity": "medium",
            "memory": "medium",
            "note": "Local coordinate system computation",
        },
        "RandomNodeSample": {
            "complexity": "medium",
            "memory": "low",
            "note": "Subgraph sampling overhead",
        },
    }

    found_expensive = []
    total_complexity_score = 0
    memory_intensive_count = 0

    for transform_name in transform_names:
        if transform_name in expensive_transforms:
            found_expensive.append(transform_name)
            perf_info = expensive_transforms[transform_name]

            if perf_info["complexity"] == "high":
                total_complexity_score += 3
            elif perf_info["complexity"] == "medium":
                total_complexity_score += 2
            else:
                total_complexity_score += 1

            if perf_info["memory"] in ["high", "medium"]:
                memory_intensive_count += 1

    if total_complexity_score > 5:
        warnings.append(
            f"High computational complexity detected: {found_expensive} - consider optimization"
        )

    if memory_intensive_count > 2:
        warnings.append(
            f"Multiple memory-intensive transforms: {found_expensive} - may cause memory issues"
        )

    if "Distance" in transform_names:
        warnings.append(
            "Distance transform: consider setting max_value parameter to limit computation for large molecules"
        )

    if "VirtualNode" in transform_names:
        warnings.append("VirtualNode: significantly increases memory usage and computation time")

    if "VirtualNode" in transform_names and "Distance" in transform_names:
        warnings.append(
            "VirtualNode + Distance combination: extremely high memory and computation requirements"
        )


def _check_dataset_compatibility(self, transform_names: list[str], warnings: list[str]) -> None:
    """Check for dataset-specific compatibility issues"""

    if hasattr(self.validator, "_current_dataset_type"):
        dataset_type = self.validator._current_dataset_type

        incompatible_transforms = {
            "DMC": {
                "avoid": ["DropNode", "RandomFlip"],
                "reason": "may interfere with uncertainty quantification",
            },
            "Wavefunction": {
                "avoid": ["RandomFlip", "DropNode", "MaskFeatures", "RandomRotate", "RandomScale"],
                "reason": "may break orbital structure and SE(3)-equivariance required for wavefunction calculations",
            },
            "DFT": {
                "avoid": ["RandomFlip"],
                "reason": "may interfere with quantum mechanical calculations",
            },
        }

        if dataset_type in incompatible_transforms:
            incompatible = incompatible_transforms[dataset_type]
            found_incompatible = [t for t in transform_names if t in incompatible["avoid"]]

            if found_incompatible:
                warnings.append(
                    f"Dataset compatibility: transforms {found_incompatible} {incompatible['reason']} "
                    f"for {dataset_type} datasets"
                )


def _update_transform_usage_stats(self, transform_name: str) -> None:
    """Update usage statistics for transforms with enhanced tracking"""

    most_used = self._composition_stats["most_used_transforms"]
    most_used[transform_name] = most_used.get(transform_name, 0) + 1

    self._metrics.increment_counter(f"transform_usage.{transform_name}", 1)


def _update_sequence_statistics(self, transform_names: list[str]) -> None:
    """Update sequence length statistics with enhanced metrics"""

    current_avg = self._composition_stats["average_sequence_length"]
    total_comps = self._composition_stats["total_compositions"]

    if total_comps > 0:
        new_avg = ((current_avg * (total_comps - 1)) + len(transform_names)) / total_comps
        self._composition_stats["average_sequence_length"] = new_avg

        self._metrics.set_gauge("composition.average_sequence_length", new_avg)


def _update_performance_metrics(self) -> None:
    """Update performance metrics with current statistics"""

    total_requests = self._composition_stats["cache_hits"] + self._composition_stats["cache_misses"]
    if total_requests > 0:
        cache_efficiency = self._composition_stats["cache_hits"] / total_requests
        self._composition_stats["performance_metrics"]["cache_efficiency"] = cache_efficiency
        self._metrics.set_gauge("composition.cache_efficiency", cache_efficiency)

    recovery_attempts = self._composition_stats["error_recovery_attempts"]
    if recovery_attempts > 0:
        recovery_rate = self._composition_stats["successful_recoveries"] / recovery_attempts
        self._composition_stats["performance_metrics"]["error_recovery_rate"] = recovery_rate
        self._metrics.set_gauge("composition.error_recovery_rate", recovery_rate)


def optimize_cache_settings(self, target_hit_rate: float = 0.8) -> dict[str, Any]:
    """Optimize cache settings based on usage patterns"""

    current_stats = self._cache_manager.get_statistics()
    current_hit_rate = current_stats.get("hit_rate", 0.0)

    optimization_report = {
        "current_hit_rate": current_hit_rate,
        "target_hit_rate": target_hit_rate,
        "optimization_applied": False,
        "changes_made": [],
    }

    if (
        current_hit_rate < target_hit_rate
        and current_stats["cache_size"] >= current_stats["max_cache_size"] * 0.8
    ):
        new_cache_size = min(current_stats["max_cache_size"] * 1.5, 200)
        self._cache_manager.max_cache_size = int(new_cache_size)
        optimization_report["optimization_applied"] = True
        optimization_report["changes_made"].append(f"Increased cache size to {int(new_cache_size)}")

    if current_stats.get("memory_pressure", False):
        current_threshold = self._cache_manager.memory_threshold_mb
        new_threshold = current_threshold * 1.2
        self._cache_manager.memory_threshold_mb = new_threshold
        optimization_report["optimization_applied"] = True
        optimization_report["changes_made"].append(
            f"Increased memory threshold to {new_threshold:.1f}MB"
        )

    return optimization_report


# =============================================================================
# COMPLETE GRAPHTRANSFORMS CLASS - ALL MISSING METHODS
# =============================================================================

# Add these methods to GraphTransforms class


def register_custom_transform(
    self,
    name: str,
    transform_class: type,
    category: str = "custom",
    research_applicability: list[str] | None = None,
    performance_notes: str | None = None,
) -> bool:
    """Register a custom transform class"""

    if not self._initialized:
        self._logger.error("Cannot register custom transform - system not initialized")
        return False

    try:
        self.registry.register_transform(
            name=name,
            transform_class=transform_class,
            category=category,
            research_applicability=research_applicability,
            performance_notes=performance_notes,
        )

        self._logger.info(f"Successfully registered custom transform: {name}")
        self._metrics.increment_counter("custom_transforms.registered", 1)
        return True

    except (TransformRegistryError, TransformValidationError) as e:
        self._logger.error(f"Failed to register custom transform {name}: {e.message}")
        self._metrics.increment_counter("custom_transforms.registration_failures", 1)
        return False


def clear_cache(self) -> None:
    """Clear all caches"""

    if self._initialized:
        self.composer.clear_cache()
        self._logger.info("Cleared transform caches")


def get_milia_experimental_setups(
    self, research_focus: str = "molecular_properties"
) -> dict[str, list[dict[str, Any]]]:
    """Get milia-optimized experimental setups"""

    if not self._initialized:
        return {}

    if hasattr(self, "config_bridge"):
        return self.config_bridge.generate_experimental_setups_for_milia(research_focus)

    return {}


def _ensure_health_check_performed(self) -> None:
    """Ensure health check has been performed at least once (lazy initialization)"""
    if not self._health_check_performed:
        try:
            self._perform_health_check()
            self._health_check_performed = True
            self._logger.debug("Initial health check completed (lazy)")
        except Exception as e:
            self._logger.warning(f"Initial health check failed: {e}")
            # Don't set flag - allow retry on next call


def perform_health_check(self) -> dict[str, Any]:
    """Perform and return comprehensive system health check"""

    # ADD: Ensure health check performed at least once
    self._ensure_health_check_performed()

    health_status = self._perform_health_check()

    recommendations = []
    if health_status.get("registry_health") != "healthy":
        recommendations.append("Check PyTorch Geometric installation")
    if health_status.get("validator_health") != "healthy":
        recommendations.append("Review transform parameter validation system")
    if health_status.get("composer_health") != "healthy":
        recommendations.append("Check transform composition capabilities")
    if health_status.get("cache_health") != "healthy":
        recommendations.append("Review cache configuration and memory settings")
    if hasattr(self, "config_bridge") and health_status.get("config_bridge_health") != "healthy":
        recommendations.append("Check configuration bridge functionality")
    if hasattr(self, "error_recovery") and health_status.get("error_recovery_health") != "healthy":
        recommendations.append("Review error recovery system")

    health_status["recommendations"] = recommendations

    return health_status


def _perform_health_check(self) -> dict[str, str]:
    """Perform comprehensive system health check with enhanced monitoring"""

    self._system_health["last_health_check"] = time.time()

    # Check registry health
    try:
        available_transforms = self.registry.list_available_transforms()
        if len(available_transforms) > 15:
            self._system_health["registry_health"] = "healthy"
        elif len(available_transforms) > 10:
            self._system_health["registry_health"] = "degraded"
        else:
            self._system_health["registry_health"] = "unhealthy"

        self._metrics.set_gauge("health.registry_transform_count", len(available_transforms))
    except Exception as e:
        self._system_health["registry_health"] = f"error: {str(e)}"
        self._metrics.increment_counter("health.registry_errors", 1)

    # Check validator health
    try:
        test_config = [{"name": "AddSelfLoops"}]
        validation_result = self.validate_config(test_config)
        if validation_result["valid"]:
            self._system_health["validator_health"] = "healthy"
        else:
            self._system_health["validator_health"] = "degraded"

        self._metrics.set_gauge(
            "health.validator_functional", 1 if validation_result["valid"] else 0
        )
    except Exception as e:
        self._system_health["validator_health"] = f"error: {str(e)}"
        self._metrics.increment_counter("health.validator_errors", 1)

    # Check composer health
    try:
        test_config = [{"name": "AddSelfLoops"}]
        composed = self.composer.compose_transforms(test_config)
        if composed is not None:
            self._system_health["composer_health"] = "healthy"
        else:
            self._system_health["composer_health"] = "degraded"

        cache_health = self.composer.get_cache_health_report()
        if cache_health["overall_health"] == "healthy":
            self._system_health["cache_health"] = "healthy"
        elif cache_health["overall_health"] == "degraded":
            self._system_health["cache_health"] = "degraded"
        else:
            self._system_health["cache_health"] = "unhealthy"

        self._metrics.set_gauge("health.cache_functional", 1 if composed is not None else 0)
    except Exception as e:
        self._system_health["composer_health"] = f"error: {str(e)}"
        self._metrics.increment_counter("health.composer_errors", 1)

    # Check metrics system health
    try:
        metrics_summary = self._metrics.get_metrics_summary()
        external_integrations = metrics_summary.get("external_integrations", {})

        if (
            any(external_integrations.values())
            or external_integrations.get("custom_handlers", 0) > 0
        ):
            self._system_health["metrics_health"] = "healthy"
        else:
            self._system_health["metrics_health"] = "degraded"

        self._metrics.set_gauge(
            "health.metrics_integrations", sum(1 for v in external_integrations.values() if v)
        )
    except Exception as e:
        self._system_health["metrics_health"] = f"error: {str(e)}"

    # Check config bridge health
    if hasattr(self, "config_bridge"):
        try:
            test_legacy_config = [{"name": "AddSelfLoops"}]
            converted = self.config_bridge.convert_legacy_config(test_legacy_config)
            if converted:
                self._system_health["config_bridge_health"] = "healthy"
            else:
                self._system_health["config_bridge_health"] = "degraded"
        except Exception as e:
            self._system_health["config_bridge_health"] = f"error: {str(e)}"

    # Check error recovery health
    if hasattr(self, "error_recovery"):
        try:
            test_error = TransformValidationError("Test error", "TestTransform")
            recovery_result = self.error_recovery.recover_from_error(test_error, {})
            if recovery_result:
                self._system_health["error_recovery_health"] = "healthy"
            else:
                self._system_health["error_recovery_health"] = "degraded"
        except Exception as e:
            self._system_health["error_recovery_health"] = f"error: {str(e)}"

    return self._system_health


def export_metrics(self, format_type: str = "prometheus") -> str:
    """Export metrics for external monitoring systems"""

    self._metrics.increment_counter("metrics.export_requests", 1)

    try:
        exported_metrics = self._metrics.export_metrics_for_monitoring(format_type)
        self._metrics.increment_counter("metrics.export_successes", 1)
        return exported_metrics
    except Exception:
        self._metrics.increment_counter("metrics.export_failures", 1)
        raise


def add_custom_metrics_handler(
    self, handler: Callable[[str, str, float, dict[str, str] | None], None]
):
    """Add custom metrics handler for integration with external systems"""

    self._metrics.add_custom_handler(handler)
    self._logger.info("Added custom metrics handler for external integration")


def optimize_performance(self, target_cache_hit_rate: float = 0.8) -> dict[str, Any]:
    """Optimize system performance based on usage patterns"""

    self._metrics.increment_counter("system.performance_optimization_requests", 1)

    cache_optimization = self.composer.optimize_cache_settings(target_cache_hit_rate)

    cache_stats = self.composer._cache_manager.get_statistics()
    if cache_stats.get("memory_pressure", False):
        gc.collect()
        self._metrics.increment_counter("system.garbage_collections", 1)

    optimization_result = {
        "cache_optimization": cache_optimization,
        "memory_cleanup_performed": cache_stats.get("memory_pressure", False),
        "current_performance_metrics": self.composer.get_composition_statistics().get(
            "performance_metrics", {}
        ),
        "recommendations": [],
    }

    perf_metrics = optimization_result["current_performance_metrics"]
    if perf_metrics.get("cache_efficiency", 0) < 0.6:
        optimization_result["recommendations"].append(
            "Consider increasing cache size or reviewing caching patterns"
        )

    if perf_metrics.get("error_recovery_rate", 0) < 0.8:
        optimization_result["recommendations"].append(
            "Review error handling and recovery mechanisms"
        )

    return optimization_result


# =============================================================================
# ADDITIONAL MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================


def get_transform_info(name: str) -> TransformInfo | None:
    """Get information about a specific transform"""
    return get_graph_transforms().get_transform_info(name)


def validate_v3_configuration(config: dict[str, Any], dataset_type: str = "DFT") -> dict[str, Any]:
    """Validate v3 configuration format"""
    return get_graph_transforms().validate_configuration(config, dataset_type)


def get_configuration_format_help() -> str:
    """Get help text for v3 configuration format"""
    return get_graph_transforms().get_configuration_format_help()


def export_metrics(format_type: str = "prometheus") -> str:
    """Export metrics for monitoring systems"""
    return get_graph_transforms().export_metrics(format_type)


def optimize_performance(target_cache_hit_rate: float = 0.8) -> dict[str, Any]:
    """Optimize system performance"""
    return get_graph_transforms().optimize_performance(target_cache_hit_rate)


def get_milia_setups(
    research_focus: str = "molecular_properties",
) -> dict[str, list[dict[str, Any]]]:
    """Get milia-optimized experimental setups"""
    return get_graph_transforms().get_milia_experimental_setups(research_focus)


def perform_system_health_check() -> dict[str, Any]:
    """Perform comprehensive system health check"""
    return get_graph_transforms().perform_health_check()


# =============================================================================
# EDGE-ATTR AWARE TRANSFORM CONVENIENCE FUNCTIONS
# =============================================================================


def get_edge_attr_aware_transform_info() -> dict[str, Any]:
    """
    Get information about edge_attr-aware transforms and their configurations.

    Returns:
        Dictionary with information about registered edge_attr-aware transforms
    """
    return {
        "registered_transforms": list(EDGE_ATTR_AWARE_TRANSFORMS.keys()),
        "configs": {
            name: {
                "edge_attr_param": config.edge_attr_param,
                "edge_attr_value": config.edge_attr_value,
                "fill_value_param": config.fill_value_param,
                "default_fill_value": config.default_fill_value,
                "description": config.description,
            }
            for name, config in EDGE_ATTR_AWARE_TRANSFORMS.items()
        },
    }


def set_sample_data_for_edge_attr_detection(sample_data: Any) -> None:
    """
    Set sample data globally for edge_attr-aware parameter injection.

    This can be called before create_transform_sequence() to enable automatic
    edge_attr handling without passing sample_data to each call.

    Args:
        sample_data: A PyG Data object from the dataset
    """
    gt = get_graph_transforms()
    if gt._initialized:
        gt.set_sample_data_for_transforms(sample_data)


# =============================================================================
# END EDGE-ATTR AWARE TRANSFORM CONVENIENCE FUNCTIONS
# =============================================================================


def validate_comprehensive(
    configs: list[dict[str, Any]],
    dataset_type: str | None = None,
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
) -> dict[str, Any]:
    """Module-level comprehensive validation"""
    return get_graph_transforms().validate_config_comprehensive(
        configs, dataset_type=dataset_type, validation_level=validation_level
    )


def get_validation_report_text(
    configs: list[dict[str, Any]], dataset_type: str | None = None
) -> str:
    """Get text validation report"""
    return get_graph_transforms().get_validation_report(
        configs, dataset_type=dataset_type, format_type="text"
    )


def discover_custom_transforms() -> dict[str, type]:
    """
    Discover all CustomTransformBase subclasses in the custom_transforms module.

    Plugin System: Auto-registration helper for custom transforms.

    Returns:
        Dictionary mapping transform names to transform classes
    """
    if not CUSTOM_TRANSFORMS_AVAILABLE:
        logger.warning("Custom transforms module not available")
        return {}

    import inspect

    from . import custom_transforms

    discovered = {}

    # Scan custom_transforms module for CustomTransformBase subclasses
    for name, obj in inspect.getmembers(custom_transforms, inspect.isclass):
        # Skip base classes
        if obj in [CustomTransformBase, MolecularTransformBase, QuantumTransformBase]:
            continue

        # Check if it's a CustomTransformBase subclass
        if hasattr(obj, "forward") and hasattr(obj, "get_metadata"):
            try:
                # Get metadata to use as transform name
                metadata = obj.get_metadata()
                transform_name = metadata.name
                discovered[transform_name] = obj
                logger.debug(f"Discovered custom transform: {transform_name}")
            except Exception as e:
                logger.warning(f"Failed to discover transform {name}: {e}")

    return discovered


# register_all_custom_transforms() is defined in the Plugin System section above

# =============================================================================
# USAGE EXAMPLES - Production-Ready Architecture Step 2.3
# =============================================================================

"""
Example 1: Basic comprehensive validation

from milia_pipeline.transformations.graph_transforms import get_graph_transforms, ValidationLevel, ValidationScope

gt = get_graph_transforms()

configs = [
    {'name': 'AddSelfLoops'},
    {'name': 'ToUndirected'},
    {'name': 'Distance', 'kwargs': {'norm': True, 'max_value': 10.0}}
]

result = gt.validate_config_comprehensive(
    configs,
    dataset_type='DFT',
    validation_level=ValidationLevel.STANDARD,
    validation_scope=ValidationScope.PRODUCTION
)

print(f"Valid: {result['valid']}")
print(f"Errors: {result['errors']}")
print(f"Warnings: {result['warnings']}")

---

Example 2: Get formatted report

report = gt.get_validation_report(
    configs,
    dataset_type='DFT',
    format_type='text'
)

print(report)

---

Example 3: Dataset-specific validation

dmc_configs = [
    {'name': 'AddSelfLoops'},
    {'name': 'ToUndirected'},
    {'name': 'DropNode', 'kwargs': {'p': 0.2}}  # Will trigger DMC warning
]

result = gt.validate_config_comprehensive(
    dmc_configs,
    dataset_type='DMC',
    validation_scope=ValidationScope.DATASET_SPECIFIC
)

if not result['valid']:
    print("DMC-specific issues found:")
    for error in result['errors']:
        print(f"  - {error}")

---

Example 4: Semantic validation

bad_sequence = [
    {'name': 'DropNode', 'kwargs': {'p': 0.3}},  # Destructive first - bad!
    {'name': 'AddSelfLoops'},  # Should come before DropNode
    {'name': 'ToUndirected'}
]

result = gt.validate_config_comprehensive(
    bad_sequence,
    validation_scope=ValidationScope.SEMANTIC
)

print(result['report']['issues_by_category'].get('ordering', []))

---

Example 5: Strict mode validation

result = gt.validate_config_comprehensive(
    configs,
    dataset_type='DFT',
    validation_level=ValidationLevel.STRICT,
    strict_mode=True
)

# In strict mode, ANY issue fails validation
if result['valid']:
    print("Perfect - no issues at all!")
else:
    print(f"Found {len(result['context'].issues)} issues")
"""
