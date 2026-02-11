"""
Target Selection Configuration Container

DYNAMIC: Supports multiple selection modes (properties, indices, ranges)
PRODUCTION-READY: Comprehensive validation and error handling
FUTURE-PROOF: Extensible design for new selection modes

Pydantic V2 Migration (Phase 25):
    - Migrated TargetSelectionConfig from @dataclass to Pydantic BaseModel (mutable)
    - Uses Field(default_factory=dict) for mutable default (raw_config)
    - Preserves custom to_dict() for backward compatibility (enum .name serialization)
    - NON-BREAKING: Same constructor API, attribute access, and method signatures
    - Follows established pattern from device_manager.py (Phase 7)

Location: milia_pipeline/models/factory/target_selection_config.py
Version: 1.1.0
"""

import logging
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)


class SelectionMode(Enum):
    """Enumeration of supported selection modes."""
    PROPERTIES = auto()   # Select by property name(s)
    INDICES = auto()      # Select by index/indices
    RANGE = auto()        # Select by range string (e.g., "0:3")
    ALL = auto()          # Select all (default)


class TargetLevel(Enum):
    """
    Target level for task-specific target selection.
    
    DYNAMIC: Inferred automatically from task_type or explicitly configured
    PRODUCTION-READY: Clear naming aligned with task_type prefixes
    FUTURE-PROOF: Extensible for new levels (e.g., SUBGRAPH)
    
    Inference rules (from task_type):
    - task_type.startswith('graph_') -> GRAPH
    - task_type.startswith('node_') -> NODE
    - task_type.startswith('edge_') or task_type.startswith('link_') -> EDGE
    - Otherwise -> GRAPH (default)
    """
    GRAPH = "graph"    # Graph-level targets: y.shape = [1, *] or scalar per graph
    NODE = "node"      # Node-level targets: y.shape = [num_nodes, *]
    EDGE = "edge"      # Edge-level targets: y.shape = [num_edges, *]


class TargetSource(Enum):
    """
    Source attribute for target data extraction.
    
    DYNAMIC: Auto-inferred from TargetLevel or explicitly configured
    PRODUCTION-READY: Standard PyG attributes plus custom support
    FUTURE-PROOF: Pattern allows extension for any attribute
    
    Default inference rules (from TargetLevel):
    - GRAPH -> Y (standard target attribute)
    - NODE -> Y first, fallback to X if shape mismatch
    - EDGE -> Y first, fallback to EDGE_ATTR if shape mismatch
    
    Special cases:
    - link_prediction -> EDGE_LABEL
    - edge_regression -> EDGE_Y or EDGE_ATTR
    """
    Y = "y"                    # Standard target attribute (PyG convention)
    X = "x"                    # Node features (fallback for node-level)
    EDGE_ATTR = "edge_attr"    # Edge features (fallback for edge-level)
    EDGE_LABEL = "edge_label"  # Binary edge labels (link prediction)
    EDGE_Y = "edge_y"          # Edge targets (edge regression)
    CUSTOM = "custom"          # Placeholder for custom attribute names


class TargetSelectionConfig(BaseModel):
    """
    Configuration container for target/property selection.
    
    DYNAMIC: Adapts to any selection specification, level/source configurable or auto-inferred
    PRODUCTION-READY: Validates all inputs, provides clear errors
    FUTURE-PROOF: Extensible via enums, supports automatic inference with override
    
    Pydantic V2 Migration (Phase 25):
        - Inherits from BaseModel (mutable, no frozen=True)
        - Uses Field(default_factory=dict) for raw_config
        - Preserves custom to_dict() for backward compatibility
    
    Level and source can be:
    - Explicitly set in config (target_level, target_source)
    - Auto-inferred from task_type when set to "auto" (default)
    
    Attributes:
        mode: The selection mode determined from config
        properties: List of property names to select (if mode=PROPERTIES)
        indices: List of indices to select (if mode=INDICES)
        range_spec: Range specification string (if mode=RANGE)
        strict: Whether to fail on invalid selections
        raw_config: Original config dict for debugging
        resolved_indices: Resolved indices after resolution
        resolved_names: Resolved property names after resolution
        total_available: Total number of targets in dataset
        config_level: Config-specified level ("auto", "graph", "node", "edge")
        config_source: Config-specified source ("auto", "y", "x", "edge_attr", or custom)
        resolved_level: Resolved TargetLevel after inference
        resolved_source: Resolved TargetSource after inference
        resolved_source_attr: Actual attribute name (for custom sources)
    """
    mode: SelectionMode = SelectionMode.ALL
    properties: Optional[List[str]] = None
    indices: Optional[List[int]] = None
    range_spec: Optional[str] = None
    strict: bool = True
    raw_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Resolved values (populated after resolution against dataset)
    resolved_indices: Optional[List[int]] = None
    resolved_names: Optional[List[str]] = None
    total_available: Optional[int] = None
    
    # Config-specified level and source (from config.yaml target_selection)
    config_level: Optional[str] = "auto"      # "auto", "graph", "node", "edge"
    config_source: Optional[str] = "auto"     # "auto", "y", "x", "edge_attr", or custom
    
    # Resolved level and source (after inference/validation)
    resolved_level: Optional[TargetLevel] = None
    resolved_source: Optional[TargetSource] = None
    resolved_source_attr: Optional[str] = None  # Actual attribute name (for custom sources)
    
    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> 'TargetSelectionConfig':
        """
        Create TargetSelectionConfig from configuration dictionary.
        
        DYNAMIC: Handles all supported config formats
        PRODUCTION-READY: Validates and normalizes input
        
        Args:
            config: Target selection configuration dict, or None for defaults
            
        Returns:
            TargetSelectionConfig instance
            
        Raises:
            ConfigurationError: If config is invalid or ambiguous
        """
        from milia_pipeline.exceptions import ConfigurationError
        
        if config is None:
            logger.debug("No target_selection config provided, using ALL targets")
            return cls(mode=SelectionMode.ALL, raw_config={})
        
        # If already a TargetSelectionConfig instance, return it directly
        # This handles the case where model_factory receives an already-created config
        # from hpo_manager.py instead of a raw dict
        if isinstance(config, cls):
            logger.debug("Config is already a TargetSelectionConfig instance, returning as-is")
            return config
        
        # Store raw config for debugging
        instance = cls(raw_config=config.copy())
        instance.strict = config.get('strict', True)
        
        # Determine selection mode (mutually exclusive)
        has_properties = 'properties' in config and config['properties'] is not None
        has_indices = 'indices' in config and config['indices'] is not None
        has_all = config.get('all', False)
        
        # Validate mutual exclusivity
        modes_specified = sum([has_properties, has_indices, has_all])
        if modes_specified > 1:
            raise ConfigurationError(
                message="Multiple selection modes specified in target_selection",
                config_key="target_selection",
                actual_value=str(config),
                expected_value="Only ONE of: properties, indices, all"
            )
        
        # Parse based on mode
        if has_properties:
            instance.mode = SelectionMode.PROPERTIES
            instance.properties = cls._normalize_list(config['properties'], str)
            logger.debug(f"Target selection mode: PROPERTIES, values: {instance.properties}")
            
        elif has_indices:
            raw_indices = config['indices']
            if isinstance(raw_indices, str) and ':' in raw_indices:
                # Range specification: "0:3" or "::2"
                instance.mode = SelectionMode.RANGE
                instance.range_spec = raw_indices
                logger.debug(f"Target selection mode: RANGE, spec: {instance.range_spec}")
            else:
                instance.mode = SelectionMode.INDICES
                instance.indices = cls._normalize_list(raw_indices, int)
                logger.debug(f"Target selection mode: INDICES, values: {instance.indices}")
                
        else:
            instance.mode = SelectionMode.ALL
            logger.debug("Target selection mode: ALL (default)")
        
        # =========================================================================
        # Parse target_level (default: "auto")
        # =========================================================================
        config_level = config.get('target_level', 'auto')
        if config_level is not None:
            config_level = str(config_level).lower()
            valid_levels = {'auto', 'graph', 'node', 'edge'}
            if config_level not in valid_levels:
                logger.warning(
                    f"Invalid target_level '{config_level}', must be one of {valid_levels}. "
                    f"Defaulting to 'auto'."
                )
                config_level = 'auto'
        else:
            config_level = 'auto'
        instance.config_level = config_level
        
        # =========================================================================
        # Parse target_source (default: "auto")
        # =========================================================================
        config_source = config.get('target_source', 'auto')
        if config_source is not None:
            config_source = str(config_source).lower()
            # Known sources + "auto" + allow any custom attribute name
            known_sources = {'auto', 'y', 'x', 'edge_attr', 'edge_label', 'edge_y'}
            if config_source not in known_sources:
                # Treat as custom attribute name (valid - user knows their data)
                logger.info(
                    f"target_source '{config_source}' is not a standard PyG attribute. "
                    f"Will attempt to use as custom attribute name."
                )
        else:
            config_source = 'auto'
        instance.config_source = config_source
        
        logger.debug(
            f"Target selection config: level={instance.config_level}, "
            f"source={instance.config_source}"
        )
        
        return instance
    
    @staticmethod
    def _normalize_list(value: Any, item_type: type) -> List:
        """Normalize single value or list to list of specified type."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [item_type(v) for v in value]
        return [item_type(value)]
    
    # =========================================================================
    # LEVEL AND SOURCE INFERENCE METHODS
    # =========================================================================
    
    @classmethod
    def infer_level_from_task_type(cls, task_type: Optional[str]) -> TargetLevel:
        """
        Infer target level from task_type string.
        
        DYNAMIC: Pattern-based detection, not hardcoded list
        PRODUCTION-READY: Handles None/empty gracefully
        FUTURE-PROOF: Any task following naming convention works
        
        Args:
            task_type: Task type string (e.g., 'node_classification', 'graph_regression')
            
        Returns:
            Inferred TargetLevel
            
        Examples:
            >>> TargetSelectionConfig.infer_level_from_task_type('node_classification')
            TargetLevel.NODE
            >>> TargetSelectionConfig.infer_level_from_task_type('graph_regression')
            TargetLevel.GRAPH
            >>> TargetSelectionConfig.infer_level_from_task_type('link_prediction')
            TargetLevel.EDGE
        """
        if task_type is None:
            logger.debug("No task_type provided, defaulting to GRAPH level")
            return TargetLevel.GRAPH
        
        task_lower = task_type.lower()
        
        # Pattern-based detection (matches model_factory.py logic)
        if task_lower.startswith('node_'):
            return TargetLevel.NODE
        elif task_lower.startswith('edge_') or task_lower.startswith('link_'):
            return TargetLevel.EDGE
        elif task_lower.startswith('graph_'):
            return TargetLevel.GRAPH
        else:
            # Unknown pattern - default to GRAPH (most common)
            logger.warning(
                f"Unknown task_type pattern '{task_type}', defaulting to GRAPH level"
            )
            return TargetLevel.GRAPH
    
    @classmethod
    def infer_source_from_level(
        cls, 
        level: TargetLevel, 
        task_type: Optional[str] = None
    ) -> TargetSource:
        """
        Infer fallback target source from level and task_type.
        
        DYNAMIC: Adapts source based on level
        PRODUCTION-READY: Handles special cases (link_prediction)
        FUTURE-PROOF: Easy to extend for new task types
        
        NOTE: This returns the FALLBACK source when "auto" is used and y doesn't match.
        The actual resolution in resolve_for_task() tries Y first (PyG convention).
        
        Args:
            level: Inferred TargetLevel
            task_type: Original task_type for special case handling
            
        Returns:
            Fallback TargetSource for when y doesn't have correct shape
        """
        task_lower = task_type.lower() if task_type else ''
        
        # Special cases for specific edge tasks
        if task_lower == 'link_prediction':
            return TargetSource.EDGE_LABEL
        elif task_lower == 'edge_regression':
            return TargetSource.EDGE_Y  # Will fallback to EDGE_ATTR if not available
        
        # Fallback mapping based on level (used when y shape doesn't match)
        # NOTE: Y is tried FIRST in resolve_for_task(), these are fallbacks
        level_to_fallback = {
            TargetLevel.GRAPH: TargetSource.Y,      # Graph always uses Y
            TargetLevel.NODE: TargetSource.X,       # Fallback: extract from node features
            TargetLevel.EDGE: TargetSource.EDGE_ATTR,  # Fallback: extract from edge features
        }
        
        return level_to_fallback.get(level, TargetSource.Y)
    
    def resolve_for_task(
        self, 
        task_type: str,
        data_sample: Optional[Any] = None,
    ) -> 'TargetSelectionConfig':
        """
        Resolve level and source for a specific task type.
        
        DYNAMIC: Uses config values if specified, auto-infers when "auto"
        PRODUCTION-READY: Validates against data, provides clear logging
        FUTURE-PROOF: Handles custom attributes, smart fallback logic
        
        Resolution logic:
        1. Level: Use config_level if not "auto", else infer from task_type
        2. Source: Use config_source if not "auto", else:
           a. Try Y first (PyG convention)
           b. If Y shape doesn't match level, use fallback (X for node, EDGE_ATTR for edge)
        
        Args:
            task_type: Task type string
            data_sample: Optional sample data for shape validation
            
        Returns:
            Self with resolved_level, resolved_source, resolved_source_attr populated
        """
        # =========================================================================
        # Idempotency check: Skip if already resolved
        # This prevents duplicate resolution when the same config is passed to
        # multiple functions (e.g., train/val prep and test prep in train_final_model)
        # =========================================================================
        if self.resolved_level is not None and self.resolved_source is not None:
            logger.debug(
                f"Target selection already resolved for task '{task_type}': "
                f"level={self.resolved_level.name}, source={self.resolved_source.name}"
            )
            return self
        
        # =========================================================================
        # Step 1: Resolve level
        # =========================================================================
        if self.config_level and self.config_level != 'auto':
            # Use explicit config value
            level_map = {
                'graph': TargetLevel.GRAPH,
                'node': TargetLevel.NODE,
                'edge': TargetLevel.EDGE,
            }
            self.resolved_level = level_map.get(self.config_level, TargetLevel.GRAPH)
            logger.debug(f"Using config-specified level: {self.resolved_level.name}")
        else:
            # Auto-infer from task_type
            self.resolved_level = self.infer_level_from_task_type(task_type)
            logger.debug(f"Auto-inferred level from task_type '{task_type}': {self.resolved_level.name}")
        
        # =========================================================================
        # Step 2: Resolve source
        # =========================================================================
        if self.config_source and self.config_source != 'auto':
            # Use explicit config value
            source_map = {
                'y': TargetSource.Y,
                'x': TargetSource.X,
                'edge_attr': TargetSource.EDGE_ATTR,
                'edge_label': TargetSource.EDGE_LABEL,
                'edge_y': TargetSource.EDGE_Y,
            }
            if self.config_source in source_map:
                self.resolved_source = source_map[self.config_source]
                self.resolved_source_attr = self.config_source
            else:
                # Custom attribute name
                self.resolved_source = TargetSource.CUSTOM
                self.resolved_source_attr = self.config_source
            logger.debug(f"Using config-specified source: {self.resolved_source_attr}")
        else:
            # Auto-infer: Try Y first (PyG convention), then fallback
            self.resolved_source = self._resolve_source_auto(
                self.resolved_level, task_type, data_sample
            )
            self.resolved_source_attr = self.resolved_source.value
        
        logger.info(
            f"Target selection resolved for task '{task_type}': "
            f"level={self.resolved_level.name}, source={self.resolved_source.name} "
            f"(attr='{self.resolved_source_attr}')"
        )
        
        return self
    
    def _resolve_source_auto(
        self,
        level: TargetLevel,
        task_type: str,
        data_sample: Optional[Any],
    ) -> TargetSource:
        """
        Auto-resolve source attribute following PyG convention.
        
        Logic:
        1. Try Y first (standard PyG convention)
        2. If Y exists and has correct shape for level → use Y
        3. If Y doesn't exist or shape mismatch → use level-based fallback
        
        Args:
            level: Resolved TargetLevel
            task_type: Task type for special cases
            data_sample: Sample data for shape validation
            
        Returns:
            Resolved TargetSource
        """
        # Special cases first
        task_lower = task_type.lower() if task_type else ''
        if task_lower == 'link_prediction':
            return TargetSource.EDGE_LABEL
        elif task_lower == 'edge_regression':
            return TargetSource.EDGE_Y
        
        # If no data sample, use Y as default (PyG convention)
        if data_sample is None:
            logger.debug("No data sample provided, defaulting to Y (PyG convention)")
            return TargetSource.Y
        
        # Check if Y exists and has correct shape
        if hasattr(data_sample, 'y') and data_sample.y is not None:
            y = data_sample.y
            y_first_dim = y.size(0) if y.dim() >= 1 else 1
            
            # Get expected dimension for level
            if level == TargetLevel.GRAPH:
                # Graph-level: y should be scalar or [1, *]
                return TargetSource.Y
            
            elif level == TargetLevel.NODE:
                num_nodes = data_sample.num_nodes if hasattr(data_sample, 'num_nodes') else None
                if num_nodes is None and hasattr(data_sample, 'x') and data_sample.x is not None:
                    num_nodes = data_sample.x.size(0)
                
                if num_nodes is not None and y_first_dim == num_nodes:
                    logger.debug(f"Y has node-level shape ({y_first_dim} == num_nodes), using Y")
                    return TargetSource.Y
                else:
                    logger.debug(
                        f"Y shape mismatch for node-level: y.size(0)={y_first_dim}, "
                        f"num_nodes={num_nodes}. Falling back to X."
                    )
                    return TargetSource.X
            
            elif level == TargetLevel.EDGE:
                num_edges = data_sample.edge_index.size(1) if hasattr(data_sample, 'edge_index') else None
                
                if num_edges is not None and y_first_dim == num_edges:
                    logger.debug(f"Y has edge-level shape ({y_first_dim} == num_edges), using Y")
                    return TargetSource.Y
                else:
                    logger.debug(
                        f"Y shape mismatch for edge-level: y.size(0)={y_first_dim}, "
                        f"num_edges={num_edges}. Falling back to EDGE_ATTR."
                    )
                    return TargetSource.EDGE_ATTR
        
        # Y doesn't exist - use fallback
        fallback = self.infer_source_from_level(level, task_type)
        logger.debug(f"Y not available, using fallback source: {fallback.name}")
        return fallback

    def resolve(
        self,
        available_names: Optional[List[str]],
        total_count: int
    ) -> 'TargetSelectionConfig':
        """
        Resolve selection specification against actual dataset.
        
        DYNAMIC: Works with or without property names
        PRODUCTION-READY: Validates against actual dataset
        
        Args:
            available_names: List of property names from dataset (may be None)
            total_count: Total number of targets in y tensor
            
        Returns:
            Self with resolved_indices and resolved_names populated
            
        Raises:
            ConfigurationError: If strict=True and selection is invalid
        """
        from milia_pipeline.exceptions import ConfigurationError
        
        self.total_available = total_count
        
        if self.mode == SelectionMode.ALL:
            self.resolved_indices = list(range(total_count))
            self.resolved_names = available_names.copy() if available_names else None
            
        elif self.mode == SelectionMode.PROPERTIES:
            self._resolve_properties(available_names, total_count)
            
        elif self.mode == SelectionMode.INDICES:
            self._resolve_indices(total_count, available_names)
            
        elif self.mode == SelectionMode.RANGE:
            self._resolve_range(total_count, available_names)
        
        # Log resolution result
        logger.info(
            f"Target selection resolved: {len(self.resolved_indices)} of {total_count} targets "
            f"[indices: {self.resolved_indices}]"
            + (f" [names: {self.resolved_names}]" if self.resolved_names else "")
        )
        
        return self
    
    def _resolve_properties(self, available_names: Optional[List[str]], total_count: int):
        """Resolve property names to indices."""
        from milia_pipeline.exceptions import ConfigurationError
        
        if available_names is None:
            if self.strict:
                raise ConfigurationError(
                    message="Cannot use property-based selection: dataset has no property names (y_property_names)",
                    config_key="target_selection.properties",
                    actual_value=str(self.properties),
                    expected_value="Use 'indices' instead, or ensure dataset has y_property_names"
                )
            else:
                logger.warning(
                    "Property-based selection requested but no property names available. "
                    "Falling back to ALL targets."
                )
                self.resolved_indices = list(range(total_count))
                self.resolved_names = None
                return
        
        # Build name → index mapping
        name_to_index = {name: i for i, name in enumerate(available_names)}
        
        resolved_indices = []
        resolved_names = []
        invalid_names = []
        
        for prop in self.properties:
            if prop in name_to_index:
                resolved_indices.append(name_to_index[prop])
                resolved_names.append(prop)
            else:
                invalid_names.append(prop)
        
        if invalid_names:
            msg = f"Invalid property name(s): {invalid_names}. Available: {available_names}"
            if self.strict:
                raise ConfigurationError(
                    message=msg,
                    config_key="target_selection.properties",
                    actual_value=str(self.properties),
                    expected_value=str(available_names)
                )
            else:
                logger.warning(msg + " (skipping invalid)")
        
        if not resolved_indices:
            raise ConfigurationError(
                message="No valid properties resolved",
                config_key="target_selection.properties",
                actual_value=str(self.properties),
                expected_value=str(available_names)
            )
        
        self.resolved_indices = resolved_indices
        self.resolved_names = resolved_names
    
    def _resolve_indices(self, total_count: int, available_names: Optional[List[str]]):
        """Resolve index specification."""
        from milia_pipeline.exceptions import ConfigurationError
        
        resolved_indices = []
        invalid_indices = []
        
        for idx in self.indices:
            # Handle negative indices (Python-style)
            actual_idx = idx if idx >= 0 else total_count + idx
            if 0 <= actual_idx < total_count:
                resolved_indices.append(actual_idx)
            else:
                invalid_indices.append(idx)
        
        if invalid_indices:
            msg = f"Invalid index/indices: {invalid_indices}. Valid range: 0 to {total_count - 1}"
            if self.strict:
                raise ConfigurationError(
                    message=msg,
                    config_key="target_selection.indices",
                    actual_value=str(self.indices),
                    expected_value=f"Integers in range [0, {total_count - 1}] or negative indices"
                )
            else:
                logger.warning(msg + " (skipping invalid)")
        
        if not resolved_indices:
            raise ConfigurationError(
                message="No valid indices resolved",
                config_key="target_selection.indices",
                actual_value=str(self.indices),
                expected_value=f"Integers in range [0, {total_count - 1}]"
            )
        
        self.resolved_indices = resolved_indices
        self.resolved_names = (
            [available_names[i] for i in resolved_indices]
            if available_names else None
        )
    
    def _resolve_range(self, total_count: int, available_names: Optional[List[str]]):
        """Resolve range specification (e.g., '0:3', '::2')."""
        from milia_pipeline.exceptions import ConfigurationError
        
        try:
            # Parse slice notation
            parts = self.range_spec.split(':')
            if len(parts) == 2:
                start = int(parts[0]) if parts[0] else 0
                stop = int(parts[1]) if parts[1] else total_count
                step = 1
            elif len(parts) == 3:
                start = int(parts[0]) if parts[0] else 0
                stop = int(parts[1]) if parts[1] else total_count
                step = int(parts[2]) if parts[2] else 1
            else:
                raise ValueError(f"Invalid slice format: {self.range_spec}")
            
            # Create slice and get indices
            s = slice(start, stop, step)
            self.resolved_indices = list(range(*s.indices(total_count)))
            
            if not self.resolved_indices:
                raise ConfigurationError(
                    message=f"Range '{self.range_spec}' produced no valid indices",
                    config_key="target_selection.indices",
                    actual_value=self.range_spec,
                    expected_value=f"Range producing indices in [0, {total_count - 1}]"
                )
            
            self.resolved_names = (
                [available_names[i] for i in self.resolved_indices]
                if available_names else None
            )
            
        except (ValueError, TypeError) as e:
            raise ConfigurationError(
                message=f"Invalid range specification: {self.range_spec}",
                config_key="target_selection.indices",
                actual_value=self.range_spec,
                expected_value="Format: 'start:stop' or 'start:stop:step' (e.g., '0:3', '::2')"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model_info and logging."""
        return {
            'mode': self.mode.name,
            # Config-specified values
            'config_level': self.config_level,
            'config_source': self.config_source,
            # Resolved values
            'resolved_level': self.resolved_level.name if self.resolved_level else None,
            'resolved_source': self.resolved_source.name if self.resolved_source else None,
            'resolved_source_attr': self.resolved_source_attr,
            # Selection values
            'specified': self.properties or self.indices or self.range_spec or 'ALL',
            'resolved_indices': self.resolved_indices,
            'resolved_names': self.resolved_names,
            'total_available': self.total_available,
            'strict': self.strict,
        }
    
    def __repr__(self) -> str:
        level_str = self.resolved_level.name if self.resolved_level else self.config_level
        source_str = self.resolved_source_attr or self.config_source
        return (
            f"TargetSelectionConfig(mode={self.mode.name}, "
            f"level={level_str}, source={source_str}, "
            f"resolved={self.resolved_indices})"
        )
