# molecule_filters.py - Transform System Integration
#                       Enhanced Parameter Introspection & Validation
#                       PHASE 6: Registry Integration for Dynamic Dataset Filtering

"""
Molecular Data Filtering with Handler-Only Architecture

This module provides functions for pre-filtering molecular data represented as PyG Data objects.
It includes filters based on atom count (min/max), the presence/absence of specific heavy atoms,
and dataset-specific filters. Custom exceptions are used to differentiate between data 
processing errors and intentional filter rejections.

Architecture Overview:
---------------------
Handler-Only Architecture - Zero Compatibility Layer Dependencies:
- Handlers are NEVER created in this module
- All handler-dependent functions accept handlers as parameters
- MoleculeFilter class accepts handler in constructor
- NO imports from compatibility layers (dataset_handler_compat)
- Zero backward compatibility mechanisms or fallback patterns

PHASE 6 UPDATE (Registry Integration):
- Dynamic dataset type discovery via registry
- Feature-based processing instead of type-specific checks
- Automatic support for new registered dataset types
- Legacy fallbacks when registry unavailable

Key Features:
------------
- Atom count filtering (min/max atoms per molecule)
- Heavy atom filtering (inclusion/exclusion by element)
- Dataset-specific filtering via handler delegation
- Transform-aware filtering with compatibility validation
- Parameter-level introspection for filter-transform conflicts
- Detailed conflict analysis with severity ratings
- Optimization suggestions for experimental configurations
- Dynamic support for any registered dataset type (Phase 6)

Handler Integration:
-------------------
Handlers passed as parameters for dataset-specific operations:
- Any registered dataset type: Via appropriate handler
- Handler parameter optional but recommended for full functionality
- When no handler provided, basic filtering still works

Transform Compatibility System:
------------------------------
Enhanced validation for filter-transform compatibility:
- Parameter introspection detects conflicts before processing
- Severity-based categorization (low/medium/high)
- Specific optimization suggestions per transform type
- Compatibility scoring for experimental setups
- Detailed reporting with categorical transform analysis

Current Architecture Status:
---------------------------
✓ Handler-only architecture (Step 5 complete)
✓ Zero compatibility layer imports
✓ All fallback mechanisms removed
✓ Handlers accepted as parameters only
✓ Consistent with cleaned modules
✓ Phase 6 registry integration complete

Usage Pattern:
-------------
```python
# Handlers created by caller and passed to filter module
from milia_pipeline.handlers import create_dataset_handler
from milia_pipeline.molecules.molecule_filters import create_molecule_filter

# Step 1: Create handler (done by caller)
handler = create_dataset_handler(
    dataset_config, filter_config, processing_config, logger
)

# Step 2: Pass handler to filter
filter_obj = create_molecule_filter(
    dataset_config=dataset_config,
    filter_config=filter_config,
    handler=handler,  # ← Passed as parameter
    logger=logger
)
```

Requirements:
------------
- Config containers required (DatasetConfig, FilterConfig)
- Handler optional but enables dataset-specific optimizations
- No raw dictionary configs accepted
- Transform config optional for enhanced validation

Exceptions:
----------
- MoleculeFilterRejectedError: Molecule rejected by filter criteria
- AtomFilterError: Atom-level filtering errors  
- HandlerOperationError: Handler-specific operation failures
- ConfigurationError: Invalid filter or transform configuration
"""

import logging
import torch
import numpy as np
from typing import Union, Dict, Set, List, Optional, Any

from torch_geometric.data import Data

from milia_pipeline.config.config_containers import DatasetConfig, FilterConfig
from milia_pipeline.config.config_constants import HEAVY_ATOM_SYMBOLS_TO_Z
from milia_pipeline.exceptions import (
    MoleculeProcessingError,
    MoleculeFilterRejectedError,
    AtomFilterError,
    ConfigurationError,
    HandlerError,
    HandlerOperationError,
    HandlerValidationError,
    DatasetSpecificHandlerError,
    HandlerIntegrationError,
    create_handler_error_context,
    create_dataset_handler_error,
    wrap_handler_operation,
    TransformConfigurationError,
    TransformValidationError,
    ValidationError
)

logger = logging.getLogger(__name__)

# ============================================================================
# PHASE 6: Registry Integration for Dynamic Dataset Filtering
# ============================================================================

# Registry availability flags - set during lazy initialization
_REGISTRY_INITIALIZED = False
_REGISTRY_AVAILABLE = False
_REGISTRY_IMPORT_ERROR = None

# Registry function placeholders (populated by _init_registry)
_registry_list_all = None
_registry_get = None
_registry_is_registered = None


def _init_registry() -> bool:
    """
    Lazily initialize registry imports to avoid circular import at module load time.
    
    The datasets/__init__.py imports implementations which may import this module
    indirectly. By deferring the registry import until first use, we allow both 
    modules to fully load first.
    
    Returns:
        True if registry is available, False otherwise
        
    ADDED Phase 6: Lazy initialization following Phase 3/6 pattern.
    """
    global _REGISTRY_INITIALIZED, _REGISTRY_AVAILABLE, _REGISTRY_IMPORT_ERROR
    global _registry_list_all, _registry_get, _registry_is_registered
    
    if _REGISTRY_INITIALIZED:
        return _REGISTRY_AVAILABLE
    
    _REGISTRY_INITIALIZED = True
    
    try:
        from milia_pipeline.datasets.registry import (
            list_all,
            get,
            is_registered,
        )
        _registry_list_all = list_all
        _registry_get = get
        _registry_is_registered = is_registered
        _REGISTRY_AVAILABLE = True
        logger.debug("Dataset registry initialized successfully for molecule filtering")
        return True
        
    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.debug(f"Dataset registry not available for molecule filtering: {e}")
        return False
        
    except Exception as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.debug(f"Dataset registry import failed: {e}")
        return False


def _get_available_dataset_types() -> List[str]:
    """
    Get list of available dataset types from registry or dynamic discovery.
    
    DYNAMIC APPROACH: Instead of hardcoded fallback, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers dataset implementations from filesystem
    3. Never uses hardcoded dataset type lists
    
    ADDED Phase 6: Dynamic dataset type discovery for supported_datasets lists.
    
    Returns:
        List of available dataset type names
    """
    _init_registry()
    if _REGISTRY_AVAILABLE and _registry_list_all is not None:
        try:
            return _registry_list_all()
        except Exception as e:
            logger.debug(f"Registry list_all failed: {e}")
    
    # DYNAMIC FALLBACK: Discover dataset types from implementations directory
    try:
        from pathlib import Path
        
        # Find the implementations directory
        implementations_dir = Path(__file__).parent.parent / 'datasets' / 'implementations'
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob('*.py'):
                if py_file.name.startswith('_'):
                    continue
                # Extract dataset name from filename (e.g., dft.py -> DFT, qm9.py -> QM9)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ['BASE', 'REGISTRY', 'UTILS', 'COMMON', 'PROTOCOLS']:
                    discovered_types.append(dataset_name)
            if discovered_types:
                logger.debug(f"Dynamically discovered dataset types: {discovered_types}")
                return discovered_types
    except Exception as e:
        logger.debug(f"Dynamic dataset type discovery failed: {e}")
    
    # Final fallback: return empty list with warning
    logger.warning("No dataset types available - registry not initialized and dynamic discovery failed")
    return []


def _is_dataset_type_registered(dataset_type: str) -> bool:
    """
    Check if dataset type is registered in registry or dynamically discovered.
    
    DYNAMIC APPROACH: Instead of hardcoded fallback check, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, uses _get_available_dataset_types() which does dynamic discovery
    3. Never uses hardcoded dataset type lists
    
    ADDED Phase 6: Dynamic dataset type validation.
    
    Args:
        dataset_type: Dataset type name to check
        
    Returns:
        True if dataset type is registered/discovered, False otherwise
    """
    _init_registry()
    if _REGISTRY_AVAILABLE and _registry_is_registered is not None:
        try:
            return _registry_is_registered(dataset_type)
        except Exception as e:
            logger.debug(f"Registry is_registered failed: {e}")
    
    # DYNAMIC FALLBACK: Check against dynamically discovered types
    available_types = _get_available_dataset_types()
    return dataset_type in available_types


def _get_dataset_feature(dataset_type: str, feature_name: str) -> bool:
    """
    Get a feature flag value for a dataset type from registry.
    
    Queries the registry for dataset feature flags. Used to determine
    dataset-specific behavior in filtering decisions and capability checks.
    
    Args:
        dataset_type: Dataset type name (e.g., 'DMC', 'DFT')
        feature_name: Feature name to query (e.g., 'uncertainty_handling')
        
    Returns:
        True if feature is enabled for this dataset type, False otherwise
    """
    _init_registry()
    
    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, 'features'):
                return getattr(dataset_class.features, feature_name, False)
        except Exception as e:
            logger.debug(f"Registry feature query failed for {dataset_type}.{feature_name}: {e}")
    
    # Registry unavailable or feature not found - return False
    return False


def _get_handler_error_type_for_dataset(handler_type_name: str):
    """
    Get the appropriate handler error class based on handler type name.
    
    Returns DatasetSpecificHandlerError for all dataset-specific handlers,
    falling back to HandlerOperationError for unrecognized handler types.
    This is fully dynamic — no dataset-specific names are checked.
    
    Args:
        handler_type_name: Handler class name (e.g., 'DMCDatasetHandler', 'QM9DatasetHandler')
        
    Returns:
        DatasetSpecificHandlerError if handler name contains 'DatasetHandler',
        HandlerOperationError otherwise
    """
    if "DatasetHandler" in handler_type_name:
        return DatasetSpecificHandlerError
    else:
        return HandlerOperationError


# ============================================================================
# END PHASE 6 Registry Integration Infrastructure
# ============================================================================

# Module-level singleton for default molecule filter
_default_filter = None

# Transform-aware filtering support
def validate_filter_compatibility_with_transforms(
    filter_config: Optional[FilterConfig] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    experimental_setup: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    include_parameter_introspection: bool = True,
    detailed_reporting: bool = True
) -> Dict[str, Any]:
    """
    ENHANCED: Validate filter and transform configuration compatibility with
    detailed parameter introspection and comprehensive reporting.
    
    Checks for potential conflicts between filters and transforms that could cause
    issues during systematic experimentation (e.g., filters that conflict with transforms
    that modify molecular structure).
    
    Args:
        filter_config: Filter configuration to validate
        transform_config: Transform configuration to validate against
        experimental_setup: Name of experimental setup for context
        logger: Logger instance
        include_parameter_introspection: NEW - Enable detailed parameter analysis
        detailed_reporting: NEW - Include detailed conflict analysis
        
    Returns:
        Dict with validation results including warnings, incompatibilities,
        recommendations, and optional parameter introspection
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    validation_results = {
        'compatible': True,
        'warnings': [],
        'incompatibilities': [],
        'recommendations': [],
        # Enhanced reporting fields
        'detailed_analysis': {} if detailed_reporting else None,
        'parameter_introspection': None
    }
    
    if filter_config is None or transform_config is None:
        validation_results['warnings'].append("Incomplete configuration - skipping compatibility check")
        return validation_results
    
    try:
        # Check for filter-transform conflicts
        transforms = transform_config.get('transforms', [])
        
        # Count transform types for reporting
        if detailed_reporting:
            transform_counts = {}
            for t in transforms:
                t_name = t.get('name')
                transform_counts[t_name] = transform_counts.get(t_name, 0) + 1
            validation_results['detailed_analysis']['transform_counts'] = transform_counts
        
        # Check for structural modification conflicts
        structural_transforms = ['RandomRotate', 'RandomScale', 'RandomTranslate', 'RandomFlip']
        has_structural_transforms = any(t.get('name') in structural_transforms for t in transforms)
        
        if has_structural_transforms and filter_config.heavy_atom_filter:
            warning_msg = (
                "Geometric transforms combined with heavy atom filters - "
                "ensure filters are applied before transforms"
            )
            validation_results['warnings'].append(warning_msg)
            
            # Add detailed context
            if detailed_reporting:
                validation_results['detailed_analysis']['structural_heavy_atom'] = {
                    'transforms': [t.get('name') for t in transforms 
                                  if t.get('name') in structural_transforms],
                    'filter': 'heavy_atom_filter',
                    'severity': 'low',
                    'rationale': 'Geometric transforms preserve atom types, minimal conflict'
                }
        
        # Check for node addition/removal conflicts
        node_modifying_transforms = ['VirtualNode', 'AddSelfLoops', 'DropNode']
        node_modifying_present = [t for t in transforms 
                                 if t.get('name') in node_modifying_transforms]
        
        if node_modifying_present and (filter_config.max_atoms or filter_config.min_atoms):
            transform_names = [t.get('name') for t in node_modifying_present]
            warning_msg = (
                f"Node-modifying transforms ({transform_names}) "
                "combined with atom count filters - atom counts may change after transforms"
            )
            validation_results['warnings'].append(warning_msg)
            
            recommendation = (
                "Consider applying atom count filters before transforms, "
                "or adjust limits to account for transform modifications"
            )
            validation_results['recommendations'].append(recommendation)
            
            # Specific recommendations per transform
            if detailed_reporting:
                detailed_recommendations = []
                for t in node_modifying_present:
                    t_name = t.get('name')
                    if t_name == 'VirtualNode' and filter_config.max_atoms:
                        detailed_recommendations.append({
                            'transform': t_name,
                            'suggestion': f'Increase max_atoms to {filter_config.max_atoms + 1} '
                                        'to accommodate virtual node'
                        })
                    elif t_name == 'DropNode' and filter_config.min_atoms:
                        drop_prob = t.get('params', {}).get('p', 0.0)
                        detailed_recommendations.append({
                            'transform': t_name,
                            'suggestion': f'With drop_prob={drop_prob}, consider reducing '
                                        f'min_atoms or lowering drop probability'
                        })
                
                validation_results['detailed_analysis']['node_modification_recommendations'] = \
                    detailed_recommendations
        
        # Check for edge modification conflicts
        edge_modifying_transforms = ['DropEdge', 'AddSelfLoops', 'ToUndirected']
        edge_modifying_present = [t for t in transforms 
                                 if t.get('name') in edge_modifying_transforms]
        
        if edge_modifying_present:
            transform_names = [t.get('name') for t in edge_modifying_present]
            recommendation = (
                f"Edge-modifying transforms present ({transform_names}) - "
                "ensure dataset-specific filters account for potential connectivity changes"
            )
            validation_results['recommendations'].append(recommendation)
            
            # Analyze edge modification severity
            if detailed_reporting:
                edge_analysis = []
                for t in edge_modifying_present:
                    t_name = t.get('name')
                    t_params = t.get('params', {})
                    
                    if t_name == 'DropEdge':
                        drop_prob = t_params.get('p', 0.0)
                        severity = 'high' if drop_prob > 0.5 else 'medium' if drop_prob > 0.2 else 'low'
                        edge_analysis.append({
                            'transform': t_name,
                            'parameter': f'p={drop_prob}',
                            'severity': severity,
                            'impact': f'May remove up to {drop_prob*100:.1f}% of edges'
                        })
                    else:
                        edge_analysis.append({
                            'transform': t_name,
                            'severity': 'low',
                            'impact': 'Connectivity structure modification'
                        })
                
                validation_results['detailed_analysis']['edge_modification_analysis'] = edge_analysis
        
        # Check for augmentation with strict filters
        augmentation_transforms = ['DropEdge', 'DropNode', 'MaskFeatures']
        has_augmentation = any(t.get('name') in augmentation_transforms for t in transforms)
        
        if has_augmentation and filter_config.dmc_uncertainty_filter:
            uncertainty_filter = filter_config.dmc_uncertainty_filter
            if uncertainty_filter.get('filter_invalid_uncertainties', True):
                warning_msg = (
                    "Augmentation transforms with strict uncertainty filtering - "
                    "ensure uncertainty handling is appropriate for augmented data"
                )
                validation_results['warnings'].append(warning_msg)
                
                # Detailed augmentation analysis
                if detailed_reporting:
                    aug_transforms = [t for t in transforms 
                                     if t.get('name') in augmentation_transforms]
                    validation_results['detailed_analysis']['augmentation_uncertainty'] = {
                        'augmentation_transforms': [t.get('name') for t in aug_transforms],
                        'uncertainty_filtering': 'strict',
                        'recommendation': 'Apply uncertainty filters before augmentation'
                    }
        
        # Add experimental setup context if available
        if experimental_setup:
            validation_results['experimental_setup'] = experimental_setup
            logger.debug(f"Validated filter-transform compatibility for experimental setup: {experimental_setup}")
        
        # Add parameter introspection if requested
        if include_parameter_introspection:
            param_introspection = introspect_transform_filter_parameters(
                filter_config=filter_config,
                transform_config=transform_config,
                logger=logger
            )
            validation_results['parameter_introspection'] = param_introspection
            
            # Merge parameter-level conflicts into main incompatibilities
            if param_introspection.get('parameter_conflicts'):
                for conflict in param_introspection['parameter_conflicts']:
                    if isinstance(conflict, dict) and conflict.get('severity') == 'high':
                        validation_results['incompatibilities'].append(
                            f"Parameter conflict: {conflict.get('conflict', 'Unknown conflict')}"
                        )
                        validation_results['compatible'] = False
        
        # Generate compatibility score
        if detailed_reporting:
            score = 100.0
            score -= len(validation_results['incompatibilities']) * 30
            score -= len(validation_results['warnings']) * 10
            score = max(0.0, score)
            
            validation_results['detailed_analysis']['compatibility_score'] = {
                'score': score,
                'rating': 'excellent' if score >= 90 else 'good' if score >= 70 else 
                         'fair' if score >= 50 else 'poor',
                'basis': f"{len(validation_results['incompatibilities'])} incompatibilities, "
                        f"{len(validation_results['warnings'])} warnings"
            }
        
    except Exception as e:
        logger.debug(f"Filter-transform compatibility check failed: {e}")
        validation_results['warnings'].append(f"Compatibility check error: {str(e)}")
    
    return validation_results


# Enhanced parameter introspection for filter-transform compatibility
def introspect_transform_filter_parameters(
    filter_config: Optional[FilterConfig] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Introspect transform and filter parameters for detailed compatibility analysis.
    
    Provides detailed parameter-level analysis of potential conflicts between filters
    and transforms, going beyond the basic compatibility checks.
    
    Args:
        filter_config: Filter configuration to analyze
        transform_config: Transform configuration to analyze
        logger: Logger instance
        
    Returns:
        Dict with detailed parameter introspection results including:
        - parameter_conflicts: Specific parameter-level conflicts
        - parameter_interactions: How parameters interact
        - optimization_suggestions: Suggestions for optimal configuration
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    introspection_results = {
        'parameter_conflicts': [],
        'parameter_interactions': [],
        'optimization_suggestions': [],
        'filter_parameters': {},
        'transform_parameters': {}
    }
    
    if filter_config is None or transform_config is None:
        introspection_results['parameter_conflicts'].append(
            "Incomplete configuration - cannot perform parameter introspection"
        )
        return introspection_results
    
    try:
        # Extract filter parameters
        filter_params = {
            'max_atoms': filter_config.max_atoms,
            'min_atoms': filter_config.min_atoms,
            'heavy_atom_filter': filter_config.heavy_atom_filter,
            'dmc_uncertainty_filter': filter_config.dmc_uncertainty_filter
        }
        introspection_results['filter_parameters'] = {
            k: v for k, v in filter_params.items() if v is not None
        }
        
        # Extract transform parameters
        transforms = transform_config.get('transforms', [])
        transform_params = []
        for t in transforms:
            transform_info = {
                'name': t.get('name'),
                'params': t.get('params', {})
            }
            transform_params.append(transform_info)
        introspection_results['transform_parameters'] = transform_params
        
        # Analyze atom count interactions
        if filter_config.max_atoms or filter_config.min_atoms:
            for t in transforms:
                t_name = t.get('name')
                t_params = t.get('params', {})
                
                # Virtual node transforms add nodes
                if t_name == 'VirtualNode':
                    if filter_config.max_atoms:
                        introspection_results['parameter_interactions'].append({
                            'type': 'atom_count_increase',
                            'transform': t_name,
                            'filter': 'max_atoms',
                            'impact': 'VirtualNode adds 1 node, effective limit becomes max_atoms - 1',
                            'severity': 'medium'
                        })
                        introspection_results['optimization_suggestions'].append(
                            f"Consider increasing max_atoms to {filter_config.max_atoms + 1} "
                            "to account for VirtualNode addition"
                        )
                
                # DropNode transforms remove nodes
                if t_name == 'DropNode':
                    drop_prob = t_params.get('p', 0.0)
                    if drop_prob > 0 and filter_config.min_atoms:
                        introspection_results['parameter_interactions'].append({
                            'type': 'atom_count_decrease',
                            'transform': t_name,
                            'filter': 'min_atoms',
                            'impact': f'DropNode with p={drop_prob} may remove nodes, '
                                     f'potentially violating min_atoms={filter_config.min_atoms}',
                            'severity': 'high' if drop_prob > 0.2 else 'medium'
                        })
                        
                        if drop_prob > 0.2:
                            introspection_results['parameter_conflicts'].append({
                                'transform': t_name,
                                'filter': 'min_atoms',
                                'conflict': f'High drop probability ({drop_prob}) with min_atoms filter',
                                'recommendation': 'Lower drop probability or remove min_atoms filter',
                                'severity': 'high'
                            })
        
        # Analyze edge modification interactions
        for t in transforms:
            t_name = t.get('name')
            t_params = t.get('params', {})
            
            if t_name == 'DropEdge':
                drop_prob = t_params.get('p', 0.0)
                if drop_prob > 0.5:
                    introspection_results['parameter_interactions'].append({
                        'type': 'connectivity_reduction',
                        'transform': t_name,
                        'impact': f'High edge drop probability ({drop_prob}) may significantly '
                                 'affect molecular connectivity',
                        'severity': 'high'
                    })
                    introspection_results['optimization_suggestions'].append(
                        f"DropEdge p={drop_prob} is aggressive - consider p < 0.5 "
                        "to maintain reasonable connectivity"
                    )
        
        # Analyze uncertainty filter interactions with augmentation
        if filter_config.dmc_uncertainty_filter:
            augmentation_transforms = [t for t in transforms 
                                      if t.get('name') in ['DropEdge', 'DropNode', 'MaskFeatures']]
            if augmentation_transforms:
                introspection_results['parameter_interactions'].append({
                    'type': 'uncertainty_augmentation',
                    'transforms': [t.get('name') for t in augmentation_transforms],
                    'filters': ['uncertainty_filter'],
                    'impact': 'Augmentation may interact with uncertainty-based filtering',
                    'severity': 'low'
                })
                introspection_results['optimization_suggestions'].append(
                    "Consider applying uncertainty filters before augmentation transforms "
                    "to ensure consistent filtering behavior"
                )
        
        # Analyze geometric transform interactions
        geometric_transforms = [t for t in transforms 
                               if t.get('name') in ['RandomRotate', 'RandomScale', 
                                                     'RandomTranslate', 'RandomFlip']]
        if geometric_transforms and filter_config.heavy_atom_filter:
            introspection_results['parameter_interactions'].append({
                'type': 'geometric_heavy_atom',
                'transforms': [t.get('name') for t in geometric_transforms],
                'filters': ['heavy_atom_filter'],
                'impact': 'Geometric transforms do not affect atom types, '
                         'heavy atom filters remain valid',
                'severity': 'info'
            })
        
        logger.debug(f"Parameter introspection completed: "
                    f"{len(introspection_results['parameter_conflicts'])} conflicts, "
                    f"{len(introspection_results['parameter_interactions'])} interactions")
        
    except Exception as e:
        logger.debug(f"Parameter introspection failed: {e}")
        introspection_results['parameter_conflicts'].append({
            'error': str(e),
            'type': 'introspection_error'
        })
    
    return introspection_results


@wrap_handler_operation("dataset_filter", "apply_dataset_specific_filters")
def apply_dataset_specific_filters(
    pyg_data: Data,
    dataset_config: Optional[DatasetConfig] = None,
    filter_config: Optional[FilterConfig] = None,
    logger: Optional[logging.Logger] = None,
    handler: Optional[object] = None
) -> None:
    """
    Applies dataset-specific filters using dataset handlers (HANDLER-ONLY).
    
    HANDLER-ONLY ARCHITECTURE: All filtering logic is delegated to handlers.
    No dataset_type conditionals remain - handlers encapsulate all dataset-specific behavior.
    
    CLEANUP PHASE: Removed all legacy fallback code and dataset_type conditionals.
    Handlers must implement apply_dataset_filters() if filtering is needed.
    
    PHASE 6 UPDATE: Uses centralized error type determination via 
    _get_handler_error_type_for_dataset() for consistent error handling.
    
    Args:
        pyg_data: The PyG Data object representing the molecule.
        dataset_config: Dataset configuration container (optional).
        filter_config: Filter configuration container (optional).
        logger: The logger instance for recording messages.
        handler: Dataset handler instance (REQUIRED).
        
    Raises:
        ValueError: If handler is None.
        MoleculeFilterRejectedError: If the molecule fails dataset-specific filters.
        DatasetSpecificHandlerError: If dataset-specific handler processing fails.
        HandlerOperationError: If generic handler operation fails.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Handler is required in handler-only architecture
    if handler is None:
        raise ValueError("Handler is required for dataset-specific filtering")
    
    mol_idx = getattr(pyg_data, 'original_mol_idx', 'N/A')
    handler_type = handler.__class__.__name__
    dataset_type = handler.get_dataset_type() if hasattr(handler, 'get_dataset_type') else 'Unknown'
    
    try:
        # HANDLER-ONLY: Delegate filtering to handler if implemented
        if hasattr(handler, 'apply_dataset_filters'):
            result = handler.apply_dataset_filters(pyg_data, filter_config)
            logger.debug(
                f"Handler {handler_type} applied dataset-specific filters to molecule {mol_idx}"
            )
            return result
        else:
            # Handler doesn't implement filtering - this is valid behavior
            # Not all dataset types require dataset-specific filters
            logger.debug(
                f"Handler {handler_type} does not implement apply_dataset_filters(). "
                f"No dataset-specific filtering applied to molecule {mol_idx}"
            )
            return None
            
    except HandlerError:
        # Re-raise handler-specific errors as-is
        raise
    except MoleculeFilterRejectedError:
        # Re-raise filter rejection errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors in appropriate handler-specific exception
        error_context = create_handler_error_context(
            handler_type=handler_type,
            operation="apply_dataset_specific_filters",
            molecule_index=mol_idx,
            additional_context={'dataset_type': dataset_type, 'filter_type': 'dataset_specific'}
        )
        
        logger.error(
            f"Dataset-specific filtering failed for molecule {mol_idx} "
            f"(handler: {handler_type}): {type(e).__name__}: {e}"
        )
        
        # Use centralized error type determination
        error_class = _get_handler_error_type_for_dataset(handler_type)
        
        if error_class == DatasetSpecificHandlerError:
            raise create_dataset_handler_error(
                message=f"Dataset handler filtering failed: {str(e)}",
                dataset_type=dataset_type,
                operation="apply_dataset_specific_filters",
                details=f"Original error: {type(e).__name__}: {str(e)}. Context: {error_context}"
            ) from e
        else:
            # Generic handler error for non-dataset handler types
            raise HandlerOperationError(
                message=f"Handler dataset filtering failed: {str(e)}",
                handler_type=handler_type,
                operation="apply_dataset_specific_filters",
                molecule_index=mol_idx,
                details=f"Original error: {type(e).__name__}: {str(e)}. Context: {error_context}"
            ) from e


def apply_atom_count_filters(
    pyg_data: Data,
    filter_config: Optional[FilterConfig] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Applies atom count filters (min_atoms, max_atoms) to a PyG Data object.
    
    Args:
        pyg_data: The PyG Data object representing the molecule.
        filter_config: Filter configuration container.
        logger: The logger instance for recording messages.
        ...
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    mol_idx: Union[int, str] = getattr(pyg_data, 'original_mol_idx', 'N/A')
    smiles: str = getattr(pyg_data, 'smiles', 'N/A')
    inchi: str = getattr(pyg_data, 'inchi', 'N/A')  
    num_nodes: Union[int, str] = getattr(pyg_data, 'num_nodes', 'N/A')

    # Use global config if none provided
    if filter_config is None:
        from milia_pipeline.config.config_containers import create_filter_config_from_global
        filter_config = create_filter_config_from_global()
        logger.debug(f"Molecule {mol_idx}: Using global filter configuration")

    try:
        # Validate atom count data availability
        if num_nodes == 'N/A' or not isinstance(num_nodes, int) or num_nodes <= 0:
            # Try to get atom count from z tensor
            if hasattr(pyg_data, 'z') and pyg_data.z is not None:
                num_nodes = pyg_data.z.numel()
            else:
                raise MoleculeProcessingError(
                    message="Cannot apply atom count filters: molecule has no valid atom count data",
                    molecule_index=mol_idx,
                    smiles=smiles,
                    inchi=inchi,
                    reason="Missing or invalid num_nodes and z tensor"
                )

        # Validate filter configuration
        if filter_config.max_atoms is not None:
            if not isinstance(filter_config.max_atoms, int) or filter_config.max_atoms <= 0:
                raise ConfigurationError(
                    message="max_atoms filter must be a positive integer",
                    config_key="filter_config.max_atoms",
                    actual_value=filter_config.max_atoms,
                    expected_value="positive integer"
                )

        if filter_config.min_atoms is not None:
            if not isinstance(filter_config.min_atoms, int) or filter_config.min_atoms <= 0:
                raise ConfigurationError(
                    message="min_atoms filter must be a positive integer",
                    config_key="filter_config.min_atoms",
                    actual_value=filter_config.min_atoms,
                    expected_value="positive integer"
                )

        # Check for logical inconsistency
        if (filter_config.max_atoms is not None and filter_config.min_atoms is not None and 
            filter_config.min_atoms > filter_config.max_atoms):
            raise ConfigurationError(
                message=f"min_atoms ({filter_config.min_atoms}) cannot be greater than max_atoms ({filter_config.max_atoms})",
                config_key="filter_config.min_atoms/max_atoms",
                actual_value=f"min={filter_config.min_atoms}, max={filter_config.max_atoms}"
            )

        # Max Atoms Filter
        if filter_config.max_atoms is not None and num_nodes > filter_config.max_atoms:
            raise MoleculeFilterRejectedError(
                molecule_index=mol_idx,
                inchi=inchi,
                reason=f"Molecule excluded due to 'max_atoms' filter: {num_nodes} atoms exceeds max_atoms={filter_config.max_atoms}",
                filter_name="max_atoms",
                filter_value=filter_config.max_atoms
            )

        # Min Atoms Filter
        if filter_config.min_atoms is not None and num_nodes < filter_config.min_atoms:
            raise MoleculeFilterRejectedError(
                molecule_index=mol_idx,
                inchi=inchi,
                reason=f"Molecule excluded due to 'min_atoms' filter: {num_nodes} atoms is below min_atoms={filter_config.min_atoms}",
                filter_name="min_atoms",
                filter_value=filter_config.min_atoms
            )
            
        logger.debug(f"Molecule {mol_idx} passed atom count filters ({num_nodes} atoms)")
        
    except (MoleculeFilterRejectedError, ConfigurationError, MoleculeProcessingError):
        # Re-raise expected exceptions as-is
        raise
    except Exception as e:
        # Convert unexpected exceptions to MoleculeProcessingError
        raise MoleculeProcessingError(
            message=f"Unexpected error during atom count filtering: {str(e)}",
            molecule_index=mol_idx,
            smiles=smiles,
            inchi=inchi,
            reason=f"Atom count filtering failed: {type(e).__name__}: {str(e)}"
        ) from e


def apply_heavy_atom_filters(
    pyg_data: Data,
    filter_config: Optional[FilterConfig] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Applies heavy atom filters to a PyG Data object.
    
    Args:
        pyg_data: The PyG Data object representing the molecule.
        filter_config: Filter configuration container.
        logger: The logger instance for recording messages.
        ...
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    mol_idx: Union[int, str] = getattr(pyg_data, 'original_mol_idx', 'N/A')
    smiles: str = getattr(pyg_data, 'smiles', 'N/A')
    inchi: str = getattr(pyg_data, 'inchi', 'N/A')

    # Use global config if none provided
    if filter_config is None:
        from milia_pipeline.config.config_containers import create_filter_config_from_global
        filter_config = create_filter_config_from_global()
        logger.debug(f"Molecule {mol_idx}: Using global filter configuration")
   
    heavy_atom_filter: Optional[Dict[str, Union[str, List[str]]]] = filter_config.heavy_atom_filter
    if not heavy_atom_filter:
        return
        
    try:
        mode: Optional[str] = heavy_atom_filter.get('mode')  # 'include' or 'exclude'
        filter_symbols: List[str] = heavy_atom_filter.get('atoms', [])

        if not mode:
            raise AtomFilterError(
                message="Heavy atom filter configured but 'mode' is missing",
                filter_config=heavy_atom_filter,
                details=f"Filter configuration validation failed for molecule {mol_idx} (InChI: {inchi})"
            )
        if not filter_symbols:
            raise AtomFilterError(
                message="Heavy atom filter configured but 'atoms' list is missing or empty",
                filter_config=heavy_atom_filter,
                details=f"Filter configuration validation failed for molecule {mol_idx} (InChI: {inchi})"
            )

        # Convert symbols to atomic numbers
        target_heavy_zs_for_filter: Set[int] = set()
        for symbol in filter_symbols:
            try:
                normalized_symbol: Optional[str] = None
                if len(symbol) == 1:
                    normalized_symbol = symbol.upper()
                elif len(symbol) == 2:
                    normalized_symbol = symbol[0].upper() + symbol[1].lower()
                else:
                    raise AtomFilterError(
                        message=f"Invalid atom symbol '{symbol}' in heavy atom filter configuration: unexpected symbol length",
                        filter_config=heavy_atom_filter,
                        atom_symbol=symbol,
                        details=f"Symbol length validation failed: expected 1-2 characters, got {len(symbol)}"
                    )

                atomic_num: Optional[int] = HEAVY_ATOM_SYMBOLS_TO_Z.get(normalized_symbol)
                if atomic_num:
                    target_heavy_zs_for_filter.add(atomic_num)
                else:
                    raise AtomFilterError(
                        message=f"Unknown heavy atom symbol '{symbol}' in filter configuration",
                        filter_config=heavy_atom_filter,
                        atom_symbol=symbol,
                        details=f"Symbol '{normalized_symbol}' not found in HEAVY_ATOM_SYMBOLS_TO_Z mapping"
                    )
                    
            except AtomFilterError:
                # Re-raise atom filter errors as-is
                raise
            except Exception as e:
                # Convert unexpected symbol processing errors
                raise AtomFilterError(
                    message=f"Error processing atom symbol '{symbol}': {str(e)}",
                    filter_config=heavy_atom_filter,
                    atom_symbol=symbol,
                    details=f"Symbol processing failed: {type(e).__name__}: {str(e)}"
                ) from e

        if not target_heavy_zs_for_filter:
            raise AtomFilterError(
                message="Heavy atom filter configured, but no valid atomic symbols were provided or recognized",
                filter_config=heavy_atom_filter,
                details=f"All symbols in {filter_symbols} were invalid or unrecognized"
            )

        # Check if atomic numbers are available
        if not hasattr(pyg_data, 'z') or pyg_data.z is None or pyg_data.z.numel() == 0:
            raise MoleculeProcessingError(
                message="Cannot apply heavy atom filter: 'z' (atomic numbers) is missing or empty in PyG Data",
                molecule_index=mol_idx,
                smiles=smiles,
                inchi=inchi,
                reason="Heavy atom filtering requires atomic number data (z tensor)"
            )

        molecule_all_zs: List[int] = pyg_data.z.unique().tolist()
        molecule_heavy_zs: Set[int] = {z for z in molecule_all_zs if z != 1}  # Exclude Hydrogen

        # Apply filtering logic based on mode
        if not molecule_heavy_zs:
            if mode == 'include':
                raise MoleculeFilterRejectedError(
                    molecule_index=mol_idx,
                    inchi=inchi,
                    reason=f"Molecule excluded due to 'heavy_atom_filter' ('include' mode): Molecule has no heavy atoms, but filter requires {filter_symbols}",
                    filter_name="heavy_atom_include",
                    filter_value=filter_symbols
                )
            elif mode == 'exclude':
                logger.debug(f"Molecule {mol_idx} passed heavy atom exclude filter (no heavy atoms to exclude)")
        else:
            if mode == 'include':
                if not molecule_heavy_zs.issubset(target_heavy_zs_for_filter):
                    unallowed_atoms: List[int] = list(molecule_heavy_zs - target_heavy_zs_for_filter)
                    raise MoleculeFilterRejectedError(
                        molecule_index=mol_idx,
                        inchi=inchi,
                        reason=f"Molecule excluded due to 'heavy_atom_filter' ('include' mode): Molecule contains unallowed heavy atoms {unallowed_atoms}. Filtered for {filter_symbols}",
                        filter_name="heavy_atom_include",
                        filter_value=filter_symbols
                    )
                else:
                    logger.debug(f"Molecule {mol_idx} passed heavy atom include filter")
                    
            elif mode == 'exclude':
                overlap_atoms: List[int] = list(molecule_heavy_zs.intersection(target_heavy_zs_for_filter))
                if len(overlap_atoms) > 0:
                    raise MoleculeFilterRejectedError(
                        molecule_index=mol_idx,
                        inchi=inchi,
                        reason=f"Molecule excluded due to 'heavy_atom_filter' ('exclude' mode) because it contains excluded heavy atom(s) {overlap_atoms}. Filtered against {filter_symbols}",
                        filter_name="heavy_atom_exclude",
                        filter_value=filter_symbols
                    )
                else:
                    logger.debug(f"Molecule {mol_idx} passed heavy atom exclude filter")
            else:
                raise AtomFilterError(
                    message=f"Unknown heavy atom filter mode '{mode}'. Expected 'include' or 'exclude'",
                    filter_config=heavy_atom_filter,
                    details=f"Invalid mode value in filter configuration"
                )
                
    except (MoleculeFilterRejectedError, AtomFilterError, MoleculeProcessingError):
        # Re-raise expected exceptions as-is
        raise
    except Exception as e:
        # Convert unexpected exceptions to AtomFilterError
        raise AtomFilterError(
            message=f"Unexpected error during heavy atom filtering: {str(e)}",
            filter_config=heavy_atom_filter,
            details=f"Heavy atom filtering failed for molecule {mol_idx}: {type(e).__name__}: {str(e)}"
        ) from e


def validate_filter_configuration(
    dataset_config: Optional[DatasetConfig] = None,
    filter_config: Optional[FilterConfig] = None,
    logger: Optional[logging.Logger] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    experimental_setup: Optional[str] = None
) -> None:
    """
    Validates the filter configuration for consistency and completeness.
    
    ENHANCED: Added transform compatibility validation.
    PHASE 6 UPDATE: Uses feature-based queries for uncertainty validation
    instead of hardcoded dataset type checks.
    
    Args:
        dataset_config (Optional[DatasetConfig]): Dataset configuration container.
        filter_config (Optional[FilterConfig]): Filter configuration container.
        logger (Optional[logging.Logger]): Logger instance for validation messages.
        transform_config (Optional[Dict[str, Any]]): NEW - Transform configuration for compatibility check.
        experimental_setup (Optional[str]): NEW - Experimental setup name for context.
        
    Raises:
        ConfigurationError: If the filter configuration is invalid.
        HandlerIntegrationError: If handler integration validation fails.
        TransformConfigurationError: If transform-filter compatibility issues detected.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Use global configs if none provided
    if dataset_config is None:
        from milia_pipeline.config.config_containers import create_dataset_config_from_global
        dataset_config = create_dataset_config_from_global()
        
    if filter_config is None:
        from milia_pipeline.config.config_containers import create_filter_config_from_global
        filter_config = create_filter_config_from_global()

    elif filter_config is None:
        from milia_pipeline.config.config_containers import create_filter_config_from_global
        filter_config = create_filter_config_from_global()

    if filter_config is None:
        logger.debug("No filter configuration to validate")
        return
    
    try:
        # Validate atom count filters
        if filter_config.max_atoms is not None:
            if not isinstance(filter_config.max_atoms, int) or filter_config.max_atoms <= 0:
                raise ConfigurationError(
                    message="max_atoms must be a positive integer",
                    config_key="filter_config.max_atoms",
                    actual_value=filter_config.max_atoms,
                    expected_value="positive integer"
                )

        if filter_config.min_atoms is not None:
            if not isinstance(filter_config.min_atoms, int) or filter_config.min_atoms <= 0:
                raise ConfigurationError(
                    message="min_atoms must be a positive integer",
                    config_key="filter_config.min_atoms",
                    actual_value=filter_config.min_atoms,
                    expected_value="positive integer"
                )

        # Check for logical inconsistency
        if (filter_config.max_atoms is not None and filter_config.min_atoms is not None and 
            filter_config.min_atoms > filter_config.max_atoms):
            raise ConfigurationError(
                message=f"min_atoms ({filter_config.min_atoms}) cannot be greater than max_atoms ({filter_config.max_atoms})",
                config_key="filter_config.min_atoms/max_atoms",
                actual_value=f"min={filter_config.min_atoms}, max={filter_config.max_atoms}"
            )
    
        # PHASE 6: Validate uncertainty-enabled dataset configuration via feature query
        # instead of hardcoded if dataset_config.dataset_type == "DMC":
        if _get_dataset_feature(dataset_config.dataset_type, 'uncertainty_handling'):
            dmc_filter_config = filter_config.dmc_uncertainty_filter or {}
            
            # Validate uncertainty threshold values
            max_uncertainty = dmc_filter_config.get('max_uncertainty')
            if max_uncertainty is not None and (not isinstance(max_uncertainty, (int, float)) or max_uncertainty < 0):
                raise ConfigurationError(
                    message="Uncertainty filter 'max_uncertainty' must be a non-negative number",
                    config_key="filter_config.dmc_uncertainty_filter.max_uncertainty",
                    actual_value=max_uncertainty,
                    expected_value="non-negative number"
                )
            
            # Validate uncertainty filtering is enabled when uncertainty filters are configured
            if max_uncertainty is not None:
                if not dataset_config.is_uncertainty_enabled:
                    logger.warning(
                        f"{dataset_config.dataset_type} uncertainty filters are configured but "
                        "uncertainty handling is disabled. Filters will not be applied."
                    )
                    
                # Check uncertainty configuration completeness
                if not dataset_config.uncertainty_config:
                    raise HandlerIntegrationError(
                        message=f"{dataset_config.dataset_type} uncertainty filters configured but uncertainty_config is missing",
                        handler_type=dataset_config.dataset_type,
                        integration_point="uncertainty_configuration",
                        details="uncertainty_config is required when uncertainty filters are active. "
                                "Ensure molecules have 'uncertainty' or 'std' attributes populated."
                    )
        
        # Validate heavy atom filter configuration
        if filter_config.heavy_atom_filter:
            heavy_atom_filter = filter_config.heavy_atom_filter
            mode = heavy_atom_filter.get('mode')
            atoms = heavy_atom_filter.get('atoms', [])
            
            if mode not in ['include', 'exclude']:
                raise ConfigurationError(
                    message=f"Invalid heavy atom filter mode: '{mode}'. Must be 'include' or 'exclude'",
                    config_key="filter_config.heavy_atom_filter.mode",
                    actual_value=mode,
                    expected_value="'include' or 'exclude'"
                )
            
            if not atoms:
                raise ConfigurationError(
                    message="Heavy atom filter 'atoms' list is empty or missing",
                    config_key="filter_config.heavy_atom_filter.atoms",
                    actual_value=atoms,
                    expected_value="non-empty list of atom symbols"
                )
            
            # Validate all atom symbols are recognized
            for symbol in atoms:
                if isinstance(symbol, str) and len(symbol) in [1, 2]:
                    normalized = symbol[0].upper() + (symbol[1].lower() if len(symbol) == 2 else "")
                    if normalized not in HEAVY_ATOM_SYMBOLS_TO_Z:
                        raise ConfigurationError(
                            message=f"Unknown atom symbol in filter configuration: '{symbol}'",
                            config_key="filter_config.heavy_atom_filter.atoms",
                            actual_value=symbol,
                            expected_value="valid chemical symbol"
                        )
                else:
                    raise ConfigurationError(
                        message=f"Invalid atom symbol format: '{symbol}'. Must be 1-2 character string",
                        config_key="filter_config.heavy_atom_filter.atoms",
                        actual_value=symbol,
                        expected_value="1-2 character string"
                    )
        
        # Validate transform compatibility
        if transform_config is not None:
            try:
                compatibility_results = validate_filter_compatibility_with_transforms(
                    filter_config=filter_config,
                    transform_config=transform_config,
                    experimental_setup=experimental_setup,
                    logger=logger
                )
                
                # Log warnings
                for warning in compatibility_results.get('warnings', []):
                    logger.warning(f"Filter-Transform Compatibility: {warning}")
                
                # Log recommendations
                for recommendation in compatibility_results.get('recommendations', []):
                    logger.info(f"Filter-Transform Recommendation: {recommendation}")
                
                # Check for critical incompatibilities
                if compatibility_results.get('incompatibilities'):
                    incompatibilities = compatibility_results['incompatibilities']
                    raise TransformConfigurationError(
                        message="Critical filter-transform incompatibilities detected",
                        config_source="filter_validation",
                        experimental_setup=experimental_setup,
                        validation_errors=incompatibilities
                    )
                    
            except TransformConfigurationError:
                # Re-raise transform configuration errors
                raise
            except Exception as e:
                logger.debug(f"Transform compatibility validation failed: {e}")
                logger.warning("Could not validate filter-transform compatibility - proceeding with caution")
                    
        logger.debug("Filter configuration validation completed successfully")
        
    except (ConfigurationError, HandlerIntegrationError, TransformConfigurationError):
        # Re-raise configuration and integration errors as-is
        raise
    except Exception as e:
        # Convert unexpected exceptions to ConfigurationError
        raise ConfigurationError(
            message=f"Unexpected error during filter configuration validation: {str(e)}",
            config_key="filter_configuration_validation",
            actual_value="validation_failed",
            details=f"Validation error: {type(e).__name__}: {str(e)}"
        ) from e


def create_handler_aware_filter_stats() -> Dict[str, Any]:
    """
    Creates statistics for filter operations with handler awareness.
    
    ENHANCED: Added transform system integration status.
    ENHANCED: Added parameter introspection and detailed reporting capabilities.
    HANDLER-ONLY: Module uses handler-only architecture with zero fallback mechanisms.
    PHASE 6 UPDATE: Uses dynamic registry for supported_datasets list.
    
    Returns:
        Dictionary with filter statistics including handler, transform integration,
        and parameter introspection capabilities.
    """
    # Handler-only architecture is established - no need to query compatibility layer
    handler_status = 'handler_only'
    
    # Check transform system availability
    transform_system_available = False
    try:
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        gt = get_graph_transforms()
        transform_system_available = gt.is_available()
    except ImportError:
        pass
    
    stats = {
        'filter_types_available': [
            'atom_count',
            'heavy_atom',
            'dataset_specific'
        ],
        'handler_integration': {
            'status': handler_status,
            'supports_uncertainty_filtering': True,
            'supports_dataset_specific_filtering': True,
            'handler_required': True,  # Handler is mandatory for dataset-specific filtering
            'architecture': 'handler_only'  # Indicates handler-only architecture
        },
        # PHASE 6: Dynamic supported_datasets from registry
        'supported_datasets': _get_available_dataset_types(),
        'exception_hierarchy': {
            'filter_rejections': 'MoleculeFilterRejectedError',
            'processing_errors': 'MoleculeProcessingError',
            'configuration_errors': 'ConfigurationError',
            'handler_errors': ['HandlerOperationError', 'DatasetSpecificHandlerError'],
            'atom_filter_errors': 'AtomFilterError',
            'handler_requirement_error': 'ValueError'  # Error when handler is None
        },
        'transform_integration': {
            'transform_system_available': transform_system_available,
            'supports_compatibility_validation': True,
            'experimental_setup_aware': True,
            'transform_aware_filtering': True,
            # Enhanced capabilities
            'supports_parameter_introspection': True,
            'supports_detailed_reporting': True,
            'compatibility_scoring': True
        },
        # Parameter introspection capabilities
        'parameter_introspection': {
            'supported_analyses': [
                'atom_count_interactions',
                'edge_modification_analysis',
                'uncertainty_augmentation_analysis',
                'geometric_transform_analysis'
            ],
            'conflict_detection_levels': ['parameter', 'transform', 'pipeline'],
            'severity_ratings': ['info', 'low', 'medium', 'high'],
            'optimization_suggestions': True
        },
        # Handler-only architecture info
        'architecture_info': {
            'version': 'handler_only_v1',
            'legacy_support': False,
            'fallback_mechanisms': None,  # Explicitly None - no fallbacks
            'requires_explicit_handler': True,
            'supports_global_config_fallback': True  # Only for configs, not handlers
        },
        # PHASE 6: Registry integration status
        'registry_integration': {
            'phase_6_complete': True,
            'registry_available': _REGISTRY_AVAILABLE,
            'dynamic_dataset_discovery': True
        }
    }
    
    return stats


@wrap_handler_operation("pre_filter", "apply_pre_filters")
def apply_pre_filters(
    pyg_data: Data,
    dataset_config: Optional[DatasetConfig] = None,
    filter_config: Optional[FilterConfig] = None,
    logger: Optional[logging.Logger] = None,
    handler: Optional[object] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    experimental_setup: Optional[str] = None
) -> bool:
    """
    Applies all pre-filters to a PyG Data object.
    
    This function orchestrates the application of all configured filters:
    1. Atom count filters (min/max atoms)
    2. Heavy atom filters (include/exclude specific elements)
    3. Dataset-specific filters via handler delegation
    
    HANDLER-ONLY ARCHITECTURE: No lazy handler initialization.
    All handlers must be passed as parameters.
    
    ENHANCED: Added transform compatibility validation.
    ENHANCED: Added experimental setup context support.
    
    Args:
        pyg_data: The PyG Data object representing the molecule.
        dataset_config: Dataset configuration container.
        filter_config: Filter configuration container.
        logger: The logger instance for recording messages.
        handler: Dataset handler instance (REQUIRED for dataset-specific filters).
        transform_config: NEW - Transform configuration for compatibility validation.
        experimental_setup: NEW - Experimental setup name for context.
        
    Raises:
        MoleculeFilterRejectedError: If the molecule fails any filter.
        MoleculeProcessingError: If processing fails due to data issues.
        AtomFilterError: If atom filtering fails.
        ConfigurationError: If filter configuration is invalid.
        HandlerOperationError: If handler operation fails unexpectedly.
        DatasetSpecificHandlerError: If dataset-specific handler operation fails.
        HandlerIntegrationError: If handler integration encounters issues.
        TransformConfigurationError: If filter-transform compatibility issues detected.

    Returns:
        bool: **True** if the molecule successfully passes all active filters.
              (Note: This return is only reached if no exceptions are raised.)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Handler is required for dataset-specific filtering
    # No lazy initialization - caller must provide handler
    if handler is None:
        logger.warning(
            "No handler provided to apply_pre_filters(). "
            "Dataset-specific filtering will be skipped. "
            "Pass a handler instance to enable dataset-specific optimizations."
        )
    
    handler_available = handler is not None
    handler_type = handler.__class__.__name__ if handler else 'None'

    # Get configurations (use global fallback if needed)
    try:
        if dataset_config is None:
            from milia_pipeline.config.config_containers import create_dataset_config_from_global
            dataset_config = create_dataset_config_from_global()
            logger.debug("Using global dataset configuration")
            
        if filter_config is None:
            from milia_pipeline.config.config_containers import create_filter_config_from_global
            filter_config = create_filter_config_from_global()
            logger.debug("Using global filter configuration")
            
    except Exception as e:
        raise HandlerIntegrationError(
            message=f"Failed to initialize filter configuration: {str(e)}",
            integration_point="configuration_initialization",
            details=f"Configuration setup failed: {type(e).__name__}: {str(e)}"
        ) from e

    # Check if any filters are configured
    if (filter_config is None or 
        (filter_config.max_atoms is None and 
         filter_config.min_atoms is None and
         filter_config.heavy_atom_filter is None and
         filter_config.dmc_uncertainty_filter is None)):
        logger.debug("No filters configured. Skipping pre-filtering.")
        return True
    
    # Validate filter configuration first (with transform compatibility if provided)
    try:
        validate_filter_configuration(
            dataset_config, 
            filter_config, 
            logger,
            transform_config=transform_config,
            experimental_setup=experimental_setup
        )
    except Exception as e:
        if not isinstance(e, (ConfigurationError, HandlerIntegrationError, TransformConfigurationError)):
            raise HandlerIntegrationError(
                message=f"Filter configuration validation failed: {str(e)}",
                integration_point="configuration_validation",
                details=f"Validation error: {type(e).__name__}: {str(e)}"
            ) from e
        else:
            raise
    
    mol_idx: Union[int, str] = getattr(pyg_data, 'original_mol_idx', 'N/A')
    dataset_type = dataset_config.dataset_type if dataset_config else 'Unknown'
    
    logger.debug(
        f"Applying pre-filters to {dataset_type} molecule {mol_idx} "
        f"(Handler: {handler_type}, Available: {handler_available})"
    )
    
    # Log transform context if available
    if experimental_setup:
        logger.debug(f"Filtering molecule {mol_idx} for experimental setup: {experimental_setup}")
    
    try:
        # Apply common filters (not dataset-specific)
        apply_atom_count_filters(pyg_data, filter_config, logger)
        apply_heavy_atom_filters(pyg_data, filter_config, logger)
        
        # Apply dataset-specific filters using handlers
        apply_dataset_specific_filters(
            pyg_data, 
            dataset_config, 
            filter_config, 
            logger, 
            handler
        )
        
        logger.debug(f"{dataset_type} molecule {mol_idx} passed all pre-filters")
        return True  # The molecule passes all applied filters
        
    except (MoleculeFilterRejectedError, AtomFilterError, ConfigurationError, 
            HandlerOperationError, DatasetSpecificHandlerError, 
            HandlerIntegrationError, MoleculeProcessingError, TransformConfigurationError,
            ValueError):  # Added ValueError for handler requirement
        # Re-raise expected exceptions as-is
        raise
    except Exception as e:
        # Convert unexpected exceptions to appropriate error type
        error_context = create_handler_error_context(
            handler_type=handler_type,
            operation="apply_pre_filters",
            molecule_index=mol_idx,
            additional_context={
                'dataset_type': dataset_type,
                'handler_available': handler_available,
                'filter_types': ['atom_count', 'heavy_atom', 'dataset_specific'],
                'experimental_setup': experimental_setup
            }
        )
        
        raise HandlerIntegrationError(
            message=f"Unexpected error during pre-filtering: {str(e)}",
            handler_type=handler_type,
            integration_point="pre_filtering_pipeline",
            details=f"Pipeline error for molecule {mol_idx}: {type(e).__name__}: {str(e)}. Context: {error_context}"
        ) from e


# =============================================================================
# MOLECULE FILTER CLASS (ENHANCED)
# =============================================================================

class MoleculeFilter:
    """
    High-level interface for molecule filtering operations
    
    This class provides a simplified facade over the molecule filtering system,
    making it easier to integrate with external systems like miliaDataset.
    
    ENHANCED: Added transform-aware filtering capabilities.
    PHASE 6 UPDATE: Uses registry-based feature queries for dataset-specific
    behavior instead of hardcoded type checks.
    
    Example usage with optimization suggestions:
        >>> # Create filter with validation
        >>> filter = create_molecule_filter(
        ...     filter_config=config,
        ...     transform_config=transforms,
        ...     validate_on_init=True,
        ...     show_optimization_suggestions=True
        ... )
        >>> 
        >>> # Check for issues before processing
        >>> if filter.has_high_severity_conflicts():
        ...     filter.print_optimization_report()
        ...     raise ValueError("Fix conflicts before proceeding")
        >>> 
        >>> # Review and acknowledge suggestions
        >>> if filter.has_unacknowledged_suggestions():
        ...     filter.print_optimization_report()
        ...     filter.acknowledge_suggestions()
        >>> 
        >>> # Now safe to process
        >>> for molecule in molecules:
        ...     try:
        ...         filter.apply_filters(molecule)
        ...     except MoleculeFilterRejectedError:
        ...         pass
    """
    
    def __init__(self, 
                 dataset_config: Optional[DatasetConfig] = None,
                 filter_config: Optional[FilterConfig] = None,
                 logger: Optional[logging.Logger] = None,
                 enable_handler_integration: bool = True,
                 handler: Optional[object] = None,
                 transform_config: Optional[Dict[str, Any]] = None,
                 experimental_setup: Optional[str] = None,
                 acknowledge_high_severity_conflicts: bool = False):
        """
        Initialize the Molecule Filter
        
        ENHANCED: Added transform configuration support.
        
        Args:
            dataset_config: Dataset configuration container
            filter_config: Filter configuration container
            logger: Logger instance for filtering operations
            enable_handler_integration: Whether to use dataset handlers when available
            handler: Dataset handler instance (passed as parameter)
            transform_config: NEW - Transform configuration for compatibility validation
            experimental_setup: NEW - Experimental setup name for context
            acknowledge_high_severity_conflicts: Whether to proceed despite high-severity conflicts
        """
        self.dataset_config = dataset_config
        self.filter_config = filter_config
        self.logger = logger or logging.getLogger(__name__ + ".MoleculeFilter")
        self.enable_handler_integration = enable_handler_integration
        
        # Transform integration
        self.transform_config = transform_config
        self.experimental_setup = experimental_setup
        
        # Initialize handler from parameter or None
        self._handler = handler
        if self.enable_handler_integration:
            if self._handler:
                self.logger.debug(f"Initialized with handler: {self._handler.__class__.__name__}")
            else:
                # Warn when handlers enabled but no handler provided
                self.logger.warning(
                    "⚠️  Handler integration enabled but no handler provided. "
                    "Dataset-specific optimizations disabled. "
                    "Pass a handler instance to enable uncertainty filtering and optimizations."
                )
                
        # Statistics tracking
        self._stats = {
            'molecules_processed': 0,
            'molecules_passed': 0,
            'molecules_rejected': 0,
            'rejections_by_filter': {},
            'processing_errors': 0,
            'handler_operations': 0,
            'transform_aware_filtering': transform_config is not None,
            'experimental_setup_name': experimental_setup,
            'optimization_suggestions': [],
            'high_severity_conflicts': [],
            'suggestions_acknowledged': False
        }
        
        # Validate compatibility on initialization
        if transform_config is not None and filter_config is not None:
            try:
                compatibility_results = validate_filter_compatibility_with_transforms(
                    filter_config=filter_config,
                    transform_config=transform_config,
                    experimental_setup=experimental_setup,
                    logger=self.logger
                )
                
                self._stats['compatibility_validation'] = compatibility_results
                
                # Store suggestions for tracking
                suggestions = compatibility_results.get('recommendations', [])
                if compatibility_results.get('parameter_introspection'):
                    param_suggestions = compatibility_results['parameter_introspection'].get(
                        'optimization_suggestions', []
                    )
                    suggestions.extend(param_suggestions)
                
                self._stats['optimization_suggestions'] = suggestions
                
                # Extract high-severity conflicts
                if compatibility_results.get('parameter_introspection'):
                    for conflict in compatibility_results['parameter_introspection'].get('parameter_conflicts', []):
                        if isinstance(conflict, dict) and conflict.get('severity') == 'high':
                            self._stats['high_severity_conflicts'].append(conflict)
                
                # Log warnings
                for warning in compatibility_results.get('warnings', []):
                    self.logger.warning(f"Filter-Transform Compatibility: {warning}")
                
                # Handle high-severity conflicts
                if self._stats['high_severity_conflicts']:
                    conflict_count = len(self._stats['high_severity_conflicts'])
                    self.logger.error(
                        f"⚠️  {conflict_count} HIGH-SEVERITY conflicts detected in filter-transform configuration!"
                    )
                    for conflict in self._stats['high_severity_conflicts']:
                        self.logger.error(f"  - {conflict.get('conflict', 'Unknown conflict')}")
                    
                    if not acknowledge_high_severity_conflicts:
                        raise ConfigurationError(
                            message=f"{conflict_count} high-severity filter-transform conflicts detected. "
                                    "Set acknowledge_high_severity_conflicts=True to proceed anyway (not recommended).",
                            config_key="filter_transform_compatibility",
                            actual_value="high_severity_conflicts_present",
                            details=f"Conflicts: {self._stats['high_severity_conflicts']}"
                        )
                    else:
                        self.logger.warning(
                            "⚠️  Proceeding with high-severity conflicts (acknowledged by user)"
                        )
                        self._stats['suggestions_acknowledged'] = True
                
                # Log optimization suggestions with counts
                if suggestions:
                    self.logger.info(
                        f"💡 {len(suggestions)} optimization suggestions available. "
                        "Call get_optimization_suggestions() to review."
                    )
                    
            except ConfigurationError:
                # Re-raise configuration errors
                raise
            except Exception as e:
                self.logger.warning(f"Filter-transform compatibility validation failed: {e}")
  
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about the filter.
        
        ENHANCED: Now includes detailed handler integration status.
        """
        status = {
            'initialized': True,
            'dataset_config_available': self.dataset_config is not None,
            'filter_config_available': self.filter_config is not None,
            'handler_available': self._handler is not None,
            'handler_integration_enabled': self.enable_handler_integration,
            'statistics': self._stats.copy(),
            'transform_aware': self.transform_config is not None,
            'experimental_setup': self.experimental_setup,
            'compatibility_validated': 'compatibility_validation' in self._stats,
            'handler_details': {
                'handler_type': self._handler.__class__.__name__ if self._handler else None,
                'handler_module': self._handler.__module__ if self._handler else None,
                'integration_working': (
                    self.enable_handler_integration and 
                    self._handler is not None and
                    self._stats.get('molecules_processed', 0) > 0 and
                    self._stats.get('handler_operations', 0) > 0
                ),
                'usage_statistics': {
                    'handler_operations': self._stats.get('handler_operations', 0),
                    'handler_usage_rate': (
                        self._stats.get('handler_operations', 0) / 
                        self._stats.get('molecules_processed', 1)
                        if self._stats.get('molecules_processed', 0) > 0 else 0.0
                    )
                }
            }
        }
        return status


    def get_optimization_suggestions(self, 
                                    include_applied: bool = False) -> List[Dict[str, Any]]:
        """
        Get all optimization suggestions from validation.
        
        Args:
            include_applied: If True, include suggestions marked as applied
            
        Returns:
            List of optimization suggestions
        """
        suggestions = self._stats.get('optimization_suggestions', [])
        
        if include_applied:
            return suggestions
        
        # Filter out applied suggestions if not included
        return [s for s in suggestions if not (isinstance(s, dict) and s.get('applied', False))]
    
    def has_high_severity_conflicts(self) -> bool:
        """Check if there are any high-severity conflicts."""
        return len(self._stats.get('high_severity_conflicts', [])) > 0
    
    def has_unacknowledged_suggestions(self) -> bool:
        """Check if there are unacknowledged optimization suggestions."""
        return (
            len(self._stats.get('optimization_suggestions', [])) > 0 and 
            not self._stats.get('suggestions_acknowledged', False)
        )
    
    def acknowledge_suggestions(self) -> None:
        """Mark all current suggestions as acknowledged."""
        self._stats['suggestions_acknowledged'] = True
        self.logger.info("Optimization suggestions acknowledged")
    
    def print_optimization_report(self) -> None:
        """Print a detailed optimization report to the logger."""
        print("\n" + "="*70)
        print("MOLECULE FILTER OPTIMIZATION REPORT")
        print("="*70)
        
        # High-severity conflicts
        conflicts = self._stats.get('high_severity_conflicts', [])
        if conflicts:
            print(f"\n⚠️  HIGH-SEVERITY CONFLICTS ({len(conflicts)}):")
            print("-"*50)
            for i, conflict in enumerate(conflicts, 1):
                if isinstance(conflict, dict):
                    print(f"  {i}. Transform: {conflict.get('transform', 'N/A')}")
                    print(f"     Filter: {conflict.get('filter', 'N/A')}")
                    print(f"     Conflict: {conflict.get('conflict', 'N/A')}")
                    print(f"     Recommendation: {conflict.get('recommendation', 'N/A')}")
                else:
                    print(f"  {i}. {conflict}")
        else:
            print("\n✓ No high-severity conflicts detected")
        
        # Optimization suggestions
        suggestions = self._stats.get('optimization_suggestions', [])
        if suggestions:
            print(f"\n💡 OPTIMIZATION SUGGESTIONS ({len(suggestions)}):")
            print("-"*50)
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("\n✓ No optimization suggestions")
        
        # Acknowledgment status
        if self._stats.get('suggestions_acknowledged'):
            print("\n✓ Suggestions have been acknowledged")
        elif suggestions or conflicts:
            print("\n⚠️  Suggestions have NOT been acknowledged")
            print("   Call acknowledge_suggestions() to proceed")
        
        print("="*70 + "\n")

    
    def validate_configuration(self, 
                                  # Enhanced validation options
                                  include_parameter_introspection: bool = True,
                                  detailed_reporting: bool = True) -> Dict[str, Any]:
            """
            Validate the current filter configuration with enhanced reporting.
            
            NEW: Added detailed parameter introspection and comprehensive reporting.
            PHASE 6 UPDATE: Uses feature-based queries for uncertainty-related warnings.
            
            Args:
                include_parameter_introspection: Enable detailed parameter analysis
                detailed_reporting: Include detailed conflict analysis
            
            Returns:
                Dictionary with enhanced validation results including:
                - Basic validation status
                - Parameter introspection results
                - Detailed compatibility analysis
                - Optimization suggestions
            """
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'configuration_status': {},
                # Enhanced reporting fields
                'detailed_analysis': None,
                'parameter_introspection': None,
                'optimization_suggestions': []
            }
            
            try:
                # Validate filter configuration with enhanced options
                validate_filter_configuration(
                    self.dataset_config, 
                    self.filter_config, 
                    self.logger,
                    transform_config=self.transform_config,
                    experimental_setup=self.experimental_setup
                )
                validation_result['configuration_status']['filter_config'] = 'valid'
            except ConfigurationError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Filter configuration error: {e}")
                validation_result['configuration_status']['filter_config'] = 'invalid'
            except TransformConfigurationError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Transform compatibility error: {e}")
                validation_result['configuration_status']['transform_compatibility'] = 'invalid'
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Validation error: {e}")
                validation_result['configuration_status']['filter_config'] = 'error'
            
            # Enhanced compatibility validation with parameter introspection
            if self.transform_config is not None and self.filter_config is not None:
                try:
                    compatibility_results = validate_filter_compatibility_with_transforms(
                        filter_config=self.filter_config,
                        transform_config=self.transform_config,
                        experimental_setup=self.experimental_setup,
                        logger=self.logger,
                        include_parameter_introspection=include_parameter_introspection,
                        detailed_reporting=detailed_reporting
                    )
                    
                    # Store detailed analysis
                    validation_result['detailed_analysis'] = compatibility_results.get('detailed_analysis')
                    validation_result['parameter_introspection'] = compatibility_results.get('parameter_introspection')
                    
                    # Merge warnings and recommendations
                    validation_result['warnings'].extend(compatibility_results.get('warnings', []))
                    validation_result['optimization_suggestions'].extend(
                        compatibility_results.get('recommendations', [])
                    )
                    
                    # Add parameter-level suggestions
                    if compatibility_results.get('parameter_introspection'):
                        param_suggestions = compatibility_results['parameter_introspection'].get(
                            'optimization_suggestions', []
                        )
                        validation_result['optimization_suggestions'].extend(param_suggestions)
                    
                    # Check for incompatibilities
                    if compatibility_results.get('incompatibilities'):
                        validation_result['valid'] = False
                        validation_result['errors'].extend(compatibility_results['incompatibilities'])
                    
                    # Add compatibility score to status
                    if detailed_reporting and compatibility_results.get('detailed_analysis'):
                        score_info = compatibility_results['detailed_analysis'].get('compatibility_score')
                        if score_info:
                            validation_result['configuration_status']['compatibility_score'] = score_info
                    
                except Exception as e:
                    validation_result['warnings'].append(
                        f"Enhanced compatibility validation failed: {e}"
                    )
            
            # PHASE 6: Check dataset configuration compatibility using feature query
            # instead of hardcoded if self.dataset_config.dataset_type == "DMC"
            if self.dataset_config and self.filter_config:
                if (_get_dataset_feature(self.dataset_config.dataset_type, 'uncertainty_handling') and 
                    self.dataset_config.is_uncertainty_enabled and
                    not self.filter_config.dmc_uncertainty_filter):
                    validation_result['warnings'].append(
                        f"{self.dataset_config.dataset_type} dataset with uncertainty enabled but "
                        "no uncertainty filters configured. "
                        "Note: Molecules must have 'uncertainty' or 'std' attributes populated."
                    )
                    validation_result['optimization_suggestions'].append(
                        f"Consider adding uncertainty filters (max_uncertainty) for "
                        f"{self.dataset_config.dataset_type} datasets to improve data quality"
                    )
            
            return validation_result

    
    def apply_filters(self, 
                     pyg_data: Data,
                     override_dataset_config: Optional[DatasetConfig] = None,
                     override_filter_config: Optional[FilterConfig] = None) -> bool:
        """
        Apply all configured filters to a molecule
        
        ENHANCED: Integrated transform context.
        
        Args:
            pyg_data: PyG Data object representing the molecule
            override_dataset_config: Optional override for dataset configuration
            override_filter_config: Optional override for filter configuration
            
        Returns:
            True if molecule passes all filters
            
        Raises:
            MoleculeFilterRejectedError: If molecule fails any filter
            MoleculeProcessingError: If processing fails
            Various other filter-specific exceptions
        """
        self._stats['molecules_processed'] += 1
        
        # Use override configs if provided
        dataset_config = override_dataset_config or self.dataset_config
        filter_config = override_filter_config or self.filter_config
        
        mol_idx = getattr(pyg_data, 'original_mol_idx', 'N/A')
        
        try:
            result = apply_pre_filters(
                pyg_data=pyg_data,
                dataset_config=dataset_config,
                filter_config=filter_config,
                logger=self.logger,
                handler=self._handler if self.enable_handler_integration else None,
                #
                transform_config=self.transform_config,
                experimental_setup=self.experimental_setup
            )
            
            if result:
                self._stats['molecules_passed'] += 1
                self.logger.debug(f"Molecule {mol_idx} passed all filters")
            
            if self._handler:
                self._stats['handler_operations'] += 1
                
            return result
            
        except MoleculeFilterRejectedError as e:
            self._stats['molecules_rejected'] += 1
            
            # Track rejections by filter type
            filter_name = e.filter_name or 'unknown'
            if filter_name not in self._stats['rejections_by_filter']:
                self._stats['rejections_by_filter'][filter_name] = 0
            self._stats['rejections_by_filter'][filter_name] += 1
            
            self.logger.debug(f"Molecule {mol_idx} rejected by {filter_name} filter: {e.reason}")
            raise
            
        except (MoleculeProcessingError, AtomFilterError, ConfigurationError, 
                HandlerOperationError, DatasetSpecificHandlerError, 
                TransformConfigurationError) as e:
            self._stats['processing_errors'] += 1
            self.logger.error(f"Error processing molecule {mol_idx}: {e}")
            raise
            
        except Exception as e:
            self._stats['processing_errors'] += 1
            self.logger.error(f"Unexpected error processing molecule {mol_idx}: {e}")
            raise MoleculeProcessingError(
                message=f"Unexpected filtering error: {str(e)}",
                molecule_index=mol_idx,
                reason=f"Filter processing failed: {type(e).__name__}: {str(e)}"
            ) from e
    
    def apply_atom_count_filters(self, 
                                pyg_data: Data,
                                override_filter_config: Optional[FilterConfig] = None) -> bool:
        """
        Apply only atom count filters to a molecule
        
        Args:
            pyg_data: PyG Data object representing the molecule
            override_filter_config: Optional override for filter configuration
            
        Returns:
            True if molecule passes atom count filters
            
        Raises:
            MoleculeFilterRejectedError: If molecule fails atom count filters
        """
        filter_config = override_filter_config or self.filter_config
        
        try:
            apply_atom_count_filters(pyg_data, filter_config, self.logger)
            return True
        except MoleculeFilterRejectedError:
            raise
        except Exception as e:
            mol_idx = getattr(pyg_data, 'original_mol_idx', 'N/A')
            raise MoleculeProcessingError(
                message=f"Atom count filter error: {str(e)}",
                molecule_index=mol_idx,
                reason=f"Atom count filtering failed: {type(e).__name__}: {str(e)}"
            ) from e
    
    def apply_heavy_atom_filters(self, 
                                pyg_data: Data,
                                override_filter_config: Optional[FilterConfig] = None) -> bool:
        """
        Apply only heavy atom filters to a molecule
        
        Args:
            pyg_data: PyG Data object representing the molecule
            override_filter_config: Optional override for filter configuration
            
        Returns:
            True if molecule passes heavy atom filters
            
        Raises:
            MoleculeFilterRejectedError: If molecule fails heavy atom filters
        """
        filter_config = override_filter_config or self.filter_config
        
        try:
            apply_heavy_atom_filters(pyg_data, filter_config, self.logger)
            return True
        except (MoleculeFilterRejectedError, AtomFilterError):
            raise
        except Exception as e:
            mol_idx = getattr(pyg_data, 'original_mol_idx', 'N/A')
            raise MoleculeProcessingError(
                message=f"Heavy atom filter error: {str(e)}",
                molecule_index=mol_idx,
                reason=f"Heavy atom filtering failed: {type(e).__name__}: {str(e)}"
            ) from e
    
    def apply_uncertainty_filters(self, 
                                 pyg_data: Data,
                                 override_dataset_config: Optional[DatasetConfig] = None,
                                 override_filter_config: Optional[FilterConfig] = None) -> bool:
        """
        Apply only uncertainty filters to a molecule (uncertainty-enabled datasets only)
        
        PHASE 6 UPDATE: Works with any uncertainty-enabled dataset, not just DMC.
        
        Args:
            pyg_data: PyG Data object representing the molecule
            override_dataset_config: Optional override for dataset configuration
            override_filter_config: Optional override for filter configuration
            
        Returns:
            True if molecule passes uncertainty filters
            
        Raises:
            MoleculeFilterRejectedError: If molecule fails uncertainty filters
        """
        dataset_config = override_dataset_config or self.dataset_config
        filter_config = override_filter_config or self.filter_config
        
        try:
            apply_dataset_specific_filters(
                pyg_data, 
                dataset_config, 
                filter_config, 
                self.logger,
                self._handler if self.enable_handler_integration else None
            )
            return True
        except (MoleculeFilterRejectedError, DatasetSpecificHandlerError):
            raise
        except Exception as e:
            mol_idx = getattr(pyg_data, 'original_mol_idx', 'N/A')
            raise MoleculeProcessingError(
                message=f"Uncertainty filter error: {str(e)}",
                molecule_index=mol_idx,
                reason=f"Uncertainty filtering failed: {type(e).__name__}: {str(e)}"
            ) from e
    
    def apply_dataset_specific_filters(self, 
                                      pyg_data: Data,
                                      override_dataset_config: Optional[DatasetConfig] = None,
                                      override_filter_config: Optional[FilterConfig] = None) -> bool:
        """
        Apply only dataset-specific filters to a molecule
        
        Args:
            pyg_data: PyG Data object representing the molecule
            override_dataset_config: Optional override for dataset configuration
            override_filter_config: Optional override for filter configuration
            
        Returns:
            True if molecule passes dataset-specific filters
            
        Raises:
            MoleculeFilterRejectedError: If molecule fails dataset-specific filters
        """
        dataset_config = override_dataset_config or self.dataset_config
        filter_config = override_filter_config or self.filter_config
        
        try:
            apply_dataset_specific_filters(
                pyg_data, 
                dataset_config, 
                filter_config, 
                self.logger,
                self._handler if self.enable_handler_integration else None
            )
            return True
        except (MoleculeFilterRejectedError, DatasetSpecificHandlerError, HandlerOperationError):
            raise
        except Exception as e:
            mol_idx = getattr(pyg_data, 'original_mol_idx', 'N/A')
            raise MoleculeProcessingError(
                message=f"Dataset-specific filter error: {str(e)}",
                molecule_index=mol_idx,
                reason=f"Dataset-specific filtering failed: {type(e).__name__}: {str(e)}"
            ) from e

    def check_molecule_compatibility(self, pyg_data: Data) -> Dict[str, Any]:
        """
        Check if a molecule is compatible with the current filter configuration.
        
        PHASE 6 UPDATE: Uses feature-based queries for uncertainty checks.
        
        Args:
            pyg_data: PyG Data object to check
            
        Returns:
            Dictionary with compatibility information
        """
        compatibility = {
            'compatible': True,
            'issues': [],
            'warnings': [],
            'molecule_info': {},
            'filter_predictions': {}
        }
        
        try:
            mol_idx = getattr(pyg_data, 'original_mol_idx', 'N/A')
            compatibility['molecule_info']['mol_idx'] = mol_idx
            
            # Check basic data availability
            if hasattr(pyg_data, 'z') and pyg_data.z is not None:
                num_atoms = pyg_data.z.numel()
                compatibility['molecule_info']['num_atoms'] = num_atoms
                
                # Predict atom count filter results
                if self.filter_config:
                    if self.filter_config.min_atoms and num_atoms < self.filter_config.min_atoms:
                        compatibility['filter_predictions']['atom_count'] = 'would_reject'
                        compatibility['issues'].append(f"Too few atoms: {num_atoms} < {self.filter_config.min_atoms}")
                    elif self.filter_config.max_atoms and num_atoms > self.filter_config.max_atoms:
                        compatibility['filter_predictions']['atom_count'] = 'would_reject'
                        compatibility['issues'].append(f"Too many atoms: {num_atoms} > {self.filter_config.max_atoms}")
                    else:
                        compatibility['filter_predictions']['atom_count'] = 'would_pass'
            else:
                compatibility['issues'].append("Missing atomic number data (z tensor)")
                compatibility['compatible'] = False
            
            # PHASE 6: Check uncertainty data for uncertainty-enabled datasets
            # instead of hardcoded if self.dataset_config.dataset_type == "DMC"
            if (self.dataset_config and 
                _get_dataset_feature(self.dataset_config.dataset_type, 'uncertainty_handling') and 
                self.dataset_config.is_uncertainty_enabled):
                
                if hasattr(pyg_data, 'uncertainty') or hasattr(pyg_data, 'std'):
                    compatibility['molecule_info']['has_uncertainty'] = True
                    compatibility['filter_predictions']['uncertainty'] = 'data_available'
                    # Check if the data is actually valid
                    unc_val = getattr(pyg_data, 'uncertainty', getattr(pyg_data, 'std', None))
                    if unc_val is None:
                        compatibility['warnings'].append("Uncertainty attribute exists but is None")
                else:
                    compatibility['issues'].append(
                        f"{self.dataset_config.dataset_type} molecule missing uncertainty data "
                        "(no 'uncertainty' or 'std' attribute). "
                        "Ensure uncertainty data is populated before filtering."
                    )
                    compatibility['compatible'] = False
            
            # Check heavy atom compatibility if filters are configured
            if (self.filter_config and 
                self.filter_config.heavy_atom_filter and
                hasattr(pyg_data, 'z') and pyg_data.z is not None):
                
                unique_z = pyg_data.z.unique().tolist()
                heavy_atoms = [z for z in unique_z if z != 1]  # Exclude hydrogen
                compatibility['molecule_info']['heavy_atoms'] = heavy_atoms
                
                filter_config = self.filter_config.heavy_atom_filter
                mode = filter_config.get('mode')
                target_symbols = filter_config.get('atoms', [])
                
                if mode and target_symbols:
                    # Convert symbols to atomic numbers for comparison
                    target_z = set()
                    for symbol in target_symbols:
                        normalized = symbol[0].upper() + (symbol[1].lower() if len(symbol) == 2 else "")
                        z = HEAVY_ATOM_SYMBOLS_TO_Z.get(normalized)
                        if z:
                            target_z.add(z)
                    
                    heavy_atom_set = set(heavy_atoms)
                    if mode == 'include':
                        if heavy_atom_set.issubset(target_z):
                            compatibility['filter_predictions']['heavy_atom'] = 'would_pass'
                        else:
                            compatibility['filter_predictions']['heavy_atom'] = 'would_reject'
                            unallowed = list(heavy_atom_set - target_z)
                            compatibility['issues'].append(f"Contains unallowed heavy atoms: {unallowed}")
                    elif mode == 'exclude':
                        overlap = heavy_atom_set.intersection(target_z)
                        if not overlap:
                            compatibility['filter_predictions']['heavy_atom'] = 'would_pass'
                        else:
                            compatibility['filter_predictions']['heavy_atom'] = 'would_reject'
                            compatibility['issues'].append(f"Contains excluded heavy atoms: {list(overlap)}")
            
            # Set overall compatibility
            if compatibility['issues']:
                compatibility['compatible'] = False
            
        except Exception as e:
            compatibility['compatible'] = False
            compatibility['issues'].append(f"Compatibility check error: {str(e)}")
        
        return compatibility
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive filtering statistics
        
        ENHANCED: Added transform integration statistics.
        """
        stats = self._stats.copy()
        
        # Calculate rates
        if stats['molecules_processed'] > 0:
            stats['pass_rate'] = stats['molecules_passed'] / stats['molecules_processed']
            stats['rejection_rate'] = stats['molecules_rejected'] / stats['molecules_processed']
            stats['error_rate'] = stats['processing_errors'] / stats['molecules_processed']
        else:
            stats['pass_rate'] = 0.0
            stats['rejection_rate'] = 0.0
            stats['error_rate'] = 0.0
        
        # Add handler usage statistics
        if stats['molecules_processed'] > 0:
            stats['handler_usage_rate'] = stats['handler_operations'] / stats['molecules_processed']
        else:
            stats['handler_usage_rate'] = 0.0
        
        return stats

    def check_handler_usage(self, 
                           min_handler_usage_rate: float = 0.9,
                           min_molecules_for_check: int = 10) -> Dict[str, Any]:
        """
        Check if handler integration is working properly.
        
        Args:
            min_handler_usage_rate: Minimum acceptable handler usage rate (default 90%)
            min_molecules_for_check: Minimum molecules needed for meaningful check
            
        Returns:
            Dictionary with handler usage analysis and recommendations
        """
        analysis = {
            'handler_available': self._handler is not None,
            'handler_enabled': self.enable_handler_integration,
            'handler_type': self._handler.__class__.__name__ if self._handler else None,
            'handler_usage_sufficient': False,
            'handler_usage_rate': 0.0,
            'warnings': [],
            'recommendations': []
        }
        
        stats = self.get_filter_statistics()
        
        # Check if we have enough data
        if stats['molecules_processed'] < min_molecules_for_check:
            analysis['warnings'].append(
                f"Only {stats['molecules_processed']} molecules processed. "
                f"Need at least {min_molecules_for_check} for meaningful analysis."
            )
            return analysis
        
        analysis['handler_usage_rate'] = stats['handler_usage_rate']
        
        # Check if handler integration is working
        if not self.enable_handler_integration:
            analysis['warnings'].append(
                "Handler integration is DISABLED. "
                "Enable with enable_handler_integration=True for better performance."
            )
            analysis['recommendations'].append(
                "Enable handler integration: MoleculeFilter(..., enable_handler_integration=True)"
            )
        elif not self._handler:
            analysis['warnings'].append(
                "Handler integration ENABLED but no handler available. "
                "Missing dataset-specific optimizations."
            )
            analysis['recommendations'].append(
                "Check handler availability. Ensure handlers are properly installed and configured."
            )
        elif stats['handler_usage_rate'] < min_handler_usage_rate:
            analysis['warnings'].append(
                f"Low handler usage rate: {stats['handler_usage_rate']:.1%} "
                f"(threshold: {min_handler_usage_rate:.1%}). "
                f"Handler available but not being used effectively."
            )
            analysis['recommendations'].append(
                "Investigate why handler is not being used. Check dataset type compatibility."
            )
            analysis['handler_usage_sufficient'] = False
        else:
            analysis['handler_usage_sufficient'] = True
        
        return analysis

    def warn_if_low_handler_usage(self,
                                   min_handler_usage_rate: float = 0.9,
                                   min_molecules: int = 10) -> bool:
        """
        Quick check and warning if handler usage is too low.
        
        Args:
            min_handler_usage_rate: Minimum acceptable rate (default 90%)
            min_molecules: Minimum molecules needed for check
            
        Returns:
            True if handler usage is low, False otherwise
        """
        stats = self.get_filter_statistics()
        
        if stats['molecules_processed'] < min_molecules:
            return False
        
        if stats['handler_usage_rate'] < min_handler_usage_rate:
            self.logger.warning(
                f"⚠️  LOW HANDLER USAGE DETECTED ⚠️\n"
                f"Handler usage rate: {stats['handler_usage_rate']:.1%} "
                f"(threshold: {min_handler_usage_rate:.1%})\n"
                f"Handler available: {self._handler is not None}\n"
                f"Handler enabled: {self.enable_handler_integration}\n"
                f"\nRecommendation: Check handler availability and dataset type compatibility"
            )
            return True
        
        return False
    
    def reset_statistics(self) -> None:
        """Reset filtering statistics"""
        processed_count = self._stats['molecules_processed']
        
        self._stats = {
            'molecules_processed': 0,
            'molecules_passed': 0,
            'molecules_rejected': 0,
            'rejections_by_filter': {},
            'processing_errors': 0,
            'handler_operations': 0,
            # Preserve configuration info
            'transform_aware_filtering': self._stats.get('transform_aware_filtering', False),
            'experimental_setup_name': self._stats.get('experimental_setup_name')
        }
        
        self.logger.debug(f"Filter statistics reset (previously processed: {processed_count})")
    
    def create_filter_report(self, 
                                # Enhanced reporting options
                                include_parameter_introspection: bool = True,
                                detailed_reporting: bool = True,
                                check_handler_usage: bool = True,
                                min_handler_usage_rate: float = 0.9) -> Dict[str, Any]:
            """
            Create comprehensive filter report with enhancements.
            
            NEW: Enhanced with detailed parameter introspection and
            comprehensive transform compatibility analysis.
            PHASE 6 UPDATE: Uses dynamic registry for supported_datasets.
            
            Args:
                include_parameter_introspection: Include detailed parameter analysis
                detailed_reporting: Include detailed conflict analysis
                check_handler_usage: Include handler usage analysis
                min_handler_usage_rate: Minimum handler usage rate for warnings
            
            Returns:
                Comprehensive report with enhanced information
            """
            report = {
                'filter_status': self.get_status(),
                'configuration_validation': self.validate_configuration(
                    include_parameter_introspection=include_parameter_introspection,
                    detailed_reporting=detailed_reporting
                ),
                'statistics': self.get_filter_statistics(),
                'capabilities': {
                    'supported_filters': [
                        'atom_count',
                        'heavy_atom',
                        'uncertainty',
                        'dataset_specific'
                    ],
                    # PHASE 6: Dynamic supported_datasets from registry
                    'supported_datasets': _get_available_dataset_types(),
                    'handler_integration': self.enable_handler_integration,
                    'transform_aware': self.transform_config is not None,
                    'compatibility_validation': True,
                    'parameter_introspection': include_parameter_introspection,
                    'detailed_reporting': detailed_reporting
                },
                'transform_analysis': None,
                'handler_analysis': None
            }
            
            # Add transform analysis if available
            if self.transform_config is not None and detailed_reporting:
                try:
                    transforms = self.transform_config.get('transforms', [])
                    transform_summary = {
                        'total_transforms': len(transforms),
                        'transform_types': {},
                        'categories': {
                            'geometric': [],
                            'augmentation': [],
                            'structural': [],
                            'feature': []
                        }
                    }
                    
                    # Categorize transforms
                    geometric_transforms = {'RandomRotate', 'RandomScale', 'RandomTranslate', 'RandomFlip'}
                    augmentation_transforms = {'DropEdge', 'DropNode', 'MaskFeatures'}
                    structural_transforms = {'VirtualNode', 'AddSelfLoops', 'ToUndirected'}
                    
                    for t in transforms:
                        t_name = t.get('name')
                        transform_summary['transform_types'][t_name] = \
                            transform_summary['transform_types'].get(t_name, 0) + 1
                        
                        if t_name in geometric_transforms:
                            transform_summary['categories']['geometric'].append(t_name)
                        elif t_name in augmentation_transforms:
                            transform_summary['categories']['augmentation'].append(t_name)
                        elif t_name in structural_transforms:
                            transform_summary['categories']['structural'].append(t_name)
                        else:
                            transform_summary['categories']['feature'].append(t_name)
                    
                    report['transform_analysis'] = transform_summary
                    
                except Exception as e:
                    self.logger.debug(f"Failed to generate transform analysis: {e}")

            # Add handler usage check
            if check_handler_usage:
                try:
                    handler_analysis = self.check_handler_usage(
                        min_handler_usage_rate=min_handler_usage_rate
                    )
                    report['handler_analysis'] = handler_analysis
                    
                    # Log warnings if handler usage is problematic
                    if handler_analysis['warnings']:
                        for warning in handler_analysis['warnings']:
                            self.logger.warning(f"Handler Usage: {warning}")
                            
                except Exception as e:
                    self.logger.debug(f"Failed to perform handler usage analysis: {e}")
            
            return report

    def get_registry_integration_status(self) -> Dict[str, Any]:
        """
        PHASE 6: Get the status of registry integration for filtering.
        
        Returns:
            Dict containing registry availability and dataset-specific information
        """
        _init_registry()
        
        dataset_type = self.dataset_config.dataset_type if self.dataset_config else None
        
        status = {
            'registry_available': _REGISTRY_AVAILABLE,
            'registry_initialized': _REGISTRY_INITIALIZED,
            'registry_import_error': _REGISTRY_IMPORT_ERROR,
            'available_dataset_types': _get_available_dataset_types(),
            'current_dataset_type': dataset_type,
            'current_dataset_registered': _is_dataset_type_registered(dataset_type) if dataset_type else None,
            'phase_6_complete': True
        }
        
        # Add feature information for current dataset
        if dataset_type and _is_dataset_type_registered(dataset_type):
            status['dataset_features'] = {
                'uncertainty_handling': _get_dataset_feature(dataset_type, 'uncertainty_handling'),
                'vibrational_analysis': _get_dataset_feature(dataset_type, 'vibrational_analysis'),
                'atomization_energy': _get_dataset_feature(dataset_type, 'atomization_energy'),
                'orbital_analysis': _get_dataset_feature(dataset_type, 'orbital_analysis'),
                'frequency_analysis': _get_dataset_feature(dataset_type, 'frequency_analysis'),
            }
        
        return status

    
    def __repr__(self) -> str:
        """String representation of the molecule filter"""
        handler_status = "with handler" if self._handler else "without handler"
        transform_status = f", transform-aware={self.transform_config is not None}"
        return f"MoleculeFilter({handler_status}{transform_status}, processed={self._stats['molecules_processed']})"
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        status = f"MoleculeFilter: {self._stats['molecules_processed']} molecules processed"
        if self._stats['molecules_processed'] > 0:
            pass_rate = self._stats['molecules_passed'] / self._stats['molecules_processed'] * 100
            status += f", {pass_rate:.1f}% pass rate"
        if self.experimental_setup:
            status += f" (setup: {self.experimental_setup})"
        return status


# =============================================================================
# FACTORY FUNCTIONS FOR MOLECULE FILTER
# =============================================================================

def create_molecule_filter(
    dataset_config: Optional[DatasetConfig] = None,
    filter_config: Optional[FilterConfig] = None,
    logger: Optional[logging.Logger] = None,
    enable_handler_integration: bool = True,
    handler: Optional[object] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    experimental_setup: Optional[str] = None,
    validate_on_init: bool = True,
    include_parameter_introspection: bool = True,
    show_optimization_suggestions: bool = True,
    acknowledge_high_severity_conflicts: bool = False,
    check_handler_status: bool = True
) -> MoleculeFilter:
    """
    Factory function to create a MoleculeFilter instance with enhancements.
    
    ENHANCED: Added transform configuration support.
    ENHANCED: Added initialization validation and parameter introspection options.
    
    Args:
        dataset_config: Dataset configuration container
        filter_config: Filter configuration container
        logger: Logger instance
        enable_handler_integration: Whether to enable handler integration
        handler: Handler instance to use
        transform_config: Transform configuration for compatibility validation
        experimental_setup: Experimental setup name for context
        validate_on_init: NEW - Validate configuration on initialization
        include_parameter_introspection: NEW - Enable parameter introspection
        show_optimization_suggestions: NEW - Show optimization suggestions
        acknowledge_high_severity_conflicts: NEW - Acknowledge high-severity conflicts
        check_handler_status: NEW - Check handler status on init
        
    Returns:
        MoleculeFilter instance with enhanced capabilities
    """
    if logger is None:
        logger = logging.getLogger(__name__ + ".create_molecule_filter")
    
    # Create filter instance
    molecule_filter = MoleculeFilter(
        dataset_config=dataset_config,
        filter_config=filter_config,
        logger=logger,
        enable_handler_integration=enable_handler_integration,
        handler=handler,
        transform_config=transform_config,
        experimental_setup=experimental_setup,
        acknowledge_high_severity_conflicts=acknowledge_high_severity_conflicts  
    )
    
    # Validate on initialization if requested
    if validate_on_init and filter_config is not None:
        try:
            validation_results = molecule_filter.validate_configuration(
                include_parameter_introspection=include_parameter_introspection,
                detailed_reporting=True
            )
            
            if not validation_results['valid']:
                logger.warning(
                    f"Filter configuration validation found issues: "
                    f"{len(validation_results['errors'])} errors, "
                    f"{len(validation_results['warnings'])} warnings"
                )
                for error in validation_results['errors']:
                    logger.error(f"Configuration error: {error}")
            
            # Show optimization suggestions
            suggestions = validation_results.get('optimization_suggestions', [])
            if suggestions and show_optimization_suggestions:
                logger.info(f"\n{'='*60}")
                logger.info(f"💡 {len(suggestions)} OPTIMIZATION SUGGESTIONS AVAILABLE")
                logger.info(f"{'='*60}")
                for i, suggestion in enumerate(suggestions, 1):
                    logger.info(f"{i}. {suggestion}")
                logger.info(f"{'='*60}\n")
            
            # Check parameter conflicts
            if validation_results.get('parameter_introspection'):
                conflicts = validation_results['parameter_introspection'].get('parameter_conflicts', [])
                high_severity = [c for c in conflicts if isinstance(c, dict) and c.get('severity') == 'high']
                
                if high_severity:
                    logger.error(f"\n{'='*60}")
                    logger.error(f"⚠️  {len(high_severity)} HIGH-SEVERITY CONFLICTS DETECTED")
                    logger.error(f"{'='*60}")
                    for conflict in high_severity:
                        logger.error(f"  Transform: {conflict.get('transform')}")
                        logger.error(f"  Filter: {conflict.get('filter')}")
                        logger.error(f"  Conflict: {conflict.get('conflict')}")
                        logger.error(f"  Recommendation: {conflict.get('recommendation')}")
                        logger.error("")
                    logger.error(f"{'='*60}\n")
                    
        except Exception as e:
            logger.warning(f"Filter configuration validation failed: {e}")

    # Check handler integration status
    if check_handler_status:
        status = molecule_filter.get_status()
        handler_details = status['handler_details']
        
        if enable_handler_integration and not handler_details['handler_type']:
            logger.warning(
                "⚠️  Handler integration enabled but no handler available!\n"
                "Dataset-specific optimizations will not be used.\n"
                "This may impact performance, especially for uncertainty-enabled datasets."
            )
        elif enable_handler_integration and handler_details['handler_type']:
            logger.info(
                f"✓ Handler integration active: {handler_details['handler_type']}"
            )
        elif not enable_handler_integration:
            logger.warning(
                "⚠️  Handler integration disabled - dataset-specific filtering unavailable"
            )
    
    return molecule_filter


def get_default_molecule_filter() -> MoleculeFilter:
    """Get the default molecule filter singleton"""
    global _default_filter
    if _default_filter is None:
        _default_filter = MoleculeFilter()
    return _default_filter
