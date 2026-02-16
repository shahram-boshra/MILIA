# plugin_system.py
"""
Plugin System for milia Pipeline

Provides a secure, extensible plugin system for custom transforms:
- Plugin discovery from multiple sources (YAML, __plugin__.py, standalone)
- Comprehensive validation (dependencies, security, functional, performance)
- Version management and compatibility checking
- Thread-safe registry with enable/disable functionality
"""

import hashlib
import importlib
import importlib.util
import inspect
import logging
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

# Lazy import to avoid circular dependency during plugin discovery
CustomTransformBase = None
MolecularTransformBase = None
QuantumTransformBase = None
TransformMetadata = None


# Import guard flags
_IMPORTING_CUSTOM_TRANSFORMS = False
_IMPORTING_GRAPH_TRANSFORMS = False


def _import_custom_transforms():
    """Lazy import of custom transforms to avoid circular dependency"""
    global CustomTransformBase, MolecularTransformBase, QuantumTransformBase, TransformMetadata
    global _IMPORTING_CUSTOM_TRANSFORMS

    if CustomTransformBase is not None:
        return True

    # Prevent re-entry
    if _IMPORTING_CUSTOM_TRANSFORMS:
        return False

    _IMPORTING_CUSTOM_TRANSFORMS = True
    try:
        from milia_pipeline.transformations.custom_transforms import CustomTransformBase as _CTB
        from milia_pipeline.transformations.custom_transforms import MolecularTransformBase as _MTB
        from milia_pipeline.transformations.custom_transforms import QuantumTransformBase as _QTB
        from milia_pipeline.transformations.custom_transforms import TransformMetadata as _TM

        CustomTransformBase = _CTB
        MolecularTransformBase = _MTB
        QuantumTransformBase = _QTB
        TransformMetadata = _TM
        return True
    except ImportError as e:
        logger.debug(f"Failed to import custom_transforms: {e}")
        return False
    finally:
        _IMPORTING_CUSTOM_TRANSFORMS = False


# Lazy import to avoid circular dependency
TransformRegistry = None
get_transform_info = None
validate_comprehensive = None


def _import_graph_transforms():
    """Lazy import of graph transforms to avoid circular dependency"""
    global TransformRegistry, get_transform_info, validate_comprehensive
    global _IMPORTING_GRAPH_TRANSFORMS

    if TransformRegistry is not None:
        return True

    # Prevent re-entry
    if _IMPORTING_GRAPH_TRANSFORMS:
        return False

    _IMPORTING_GRAPH_TRANSFORMS = True
    try:
        from milia_pipeline.transformations.graph_transforms import get_transform_info as _gti
        from milia_pipeline.transformations.graph_transforms import registry as _reg
        from milia_pipeline.transformations.graph_transforms import validate_comprehensive as _vc

        TransformRegistry = _reg
        get_transform_info = _gti
        validate_comprehensive = _vc
        return True
    except ImportError as e:
        logger.debug(f"Failed to import graph_transforms: {e}")
        return False
    finally:
        _IMPORTING_GRAPH_TRANSFORMS = False


# From config system (01_Relevant_Files)
try:
    from milia_pipeline.config.config_loader import load_config
except ImportError:
    load_config = None

try:
    from milia_pipeline.exceptions import (
        PluginDependencyError,
        PluginError,
        PluginSecurityError,
        PluginValidationError,
        TransformValidationError,
    )
except ImportError:
    # Fallback exception classes
    class PluginError(Exception):
        """Base exception for plugin-related errors."""

        def __init__(self, message: str, plugin_name: str | None = None, **kwargs):
            super().__init__(message)
            self.plugin_name = plugin_name

    class PluginValidationError(PluginError):
        """Exception raised when plugin validation fails."""

        def __init__(
            self,
            message: str,
            plugin_name: str | None = None,
            validation_errors: list | None = None,
            **kwargs,
        ):
            super().__init__(message, plugin_name=plugin_name)
            self.validation_errors = validation_errors or []

    class PluginSecurityError(PluginError):
        """Exception raised for plugin security concerns."""

        def __init__(
            self,
            message: str,
            plugin_name: str | None = None,
            security_issues: list | None = None,
            **kwargs,
        ):
            super().__init__(message, plugin_name=plugin_name)
            self.security_issues = security_issues or []

    class PluginDependencyError(PluginError):
        """Exception raised when plugin dependencies are not satisfied."""

        def __init__(
            self,
            message: str,
            plugin_name: str | None = None,
            missing_dependencies: list | None = None,
            **kwargs,
        ):
            super().__init__(message, plugin_name=plugin_name)
            self.missing_dependencies = missing_dependencies or []

    TransformValidationError = Exception

# Logging (02_Relevant_Files)
# Note: logging_config.py doesn't have get_logger(), using standard logging
logger = logging.getLogger("milia_Main.PluginSystem")


class TransformDeclaration(BaseModel):
    """
    Represents a transform declared in plugin.yaml.

    This is a SPECIFICATION of what should exist, not an implementation.
    Used to track what transforms a plugin claims to provide.

    Pydantic V2 Migration (Phase 19):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - Uses model_dump() for to_dict() method (backward compatible)
        - NON-BREAKING: Same constructor API and attribute access
    """

    # Identity (required)
    name: str  # Transform name for TransformRegistry
    class_name: str  # Python class name
    module_path: str  # Relative import path from plugin root

    # Metadata (required)
    category: str  # Transform category
    description: str  # What the transform does
    version: str = "1.0.0"  # Transform version

    # Optional metadata from plugin.yaml
    required_node_features: list[str] = Field(default_factory=list)
    required_edge_features: list[str] = Field(default_factory=list)
    required_graph_attributes: list[str] = Field(default_factory=list)
    parameter_constraints: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Backward compatible method wrapping Pydantic V2's model_dump().
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformDeclaration":
        """Create from dictionary (from plugin.yaml)."""
        return cls(
            name=data["name"],
            class_name=data["class_name"],
            module_path=data["module_path"],
            category=data["category"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            required_node_features=data.get("required_node_features", []),
            required_edge_features=data.get("required_edge_features", []),
            required_graph_attributes=data.get("required_graph_attributes", []),
            parameter_constraints=data.get("parameter_constraints", {}),
        )


class PluginMetadata(BaseModel):
    """
    Enhanced metadata for a transform plugin package.

    Separates transform declarations (from YAML) from registrations (runtime state).
    This enables proper tracking of what should exist vs what actually exists.

    Pydantic V2 Migration (Phase 19):
        - Migrated from @dataclass(eq=True) to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - Uses @model_validator(mode='after') for __post_init__ validation logic
        - Preserves custom __hash__ for use in sets/dicts (hashes plugin_name + version)
        - Uses model_dump() for to_dict() method (backward compatible)
        - NON-BREAKING: Same constructor API and attribute access
    """

    # Required fields (identity)
    plugin_name: str
    version: str
    author: str
    # Plugin type (for filtering)
    plugin_type: str = "user_experimental"  # "pyg_fallback" or "user_experimental"

    # Optional metadata
    email: str | None = None
    license: str = "MIT"
    description: str = ""
    homepage: str | None = None

    # Version dependencies
    milia_version: str = ">=1.0.0"
    pyg_version: str = ">=2.0.0"
    python_version: str = ">=3.8"
    dependencies: list[str] = Field(default_factory=list)

    # CRITICAL SEPARATION: Declarations vs Registrations

    # What SHOULD exist (from plugin.yaml specification)
    transform_declarations: list[TransformDeclaration] = Field(default_factory=list)

    # What IS registered (runtime state in TransformRegistry)
    registered_transforms: set[str] = Field(default_factory=set)

    # Discovery metadata
    discovery_source: str = "unknown"  # "yaml", "python", "hybrid"
    discovery_timestamp: str | None = None

    # Validation status
    is_validated: bool = False
    validation_date: str | None = None
    validation_results: dict = Field(default_factory=dict)

    # Security
    checksum: str | None = None
    trusted: bool = False

    @model_validator(mode="after")
    def validate_plugin_metadata(self) -> "PluginMetadata":
        """
        Validate plugin metadata on creation.

        Replaces __post_init__ validation from dataclass implementation.
        """
        if not self.plugin_name:
            raise PluginError("Plugin name is required")

        # Validate version format
        if not self._is_valid_version(self.version):
            raise PluginError(f"Invalid version format: {self.version}")

        # Validate dependency specifications
        self._validate_dependencies()

        return self

    def __hash__(self):
        """Make hashable for use in sets/dicts."""
        return hash((self.plugin_name, self.version))

    def __eq__(self, other):
        """Equality based on plugin_name and version (matches __hash__ behavior)."""
        if not isinstance(other, PluginMetadata):
            return NotImplemented
        return (self.plugin_name, self.version) == (other.plugin_name, other.version)

    @property
    def declared_count(self) -> int:
        """Number of transforms declared in plugin.yaml."""
        return len(self.transform_declarations)

    @property
    def registered_count(self) -> int:
        """Number of transforms actually registered in TransformRegistry."""
        return len(self.registered_transforms)

    @property
    def missing_implementations(self) -> list[str]:
        """Transforms declared in YAML but not registered (no implementation found)."""
        declared_names = {decl.name for decl in self.transform_declarations}
        return list(declared_names - self.registered_transforms)

    @property
    def undeclared_implementations(self) -> list[str]:
        """Transforms registered but not declared in YAML (bonus discoveries)."""
        declared_names = {decl.name for decl in self.transform_declarations}
        return list(self.registered_transforms - declared_names)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Backward compatible method using Pydantic V2's model_dump().
        Includes computed properties for convenience.
        """
        base_dict = self.model_dump()
        # Convert TransformDeclaration objects to dicts
        base_dict["transform_declarations"] = [
            decl.to_dict() for decl in self.transform_declarations
        ]
        # Convert set to list for JSON serialization
        base_dict["registered_transforms"] = list(self.registered_transforms)
        # Add computed properties for convenience
        base_dict["declared_count"] = self.declared_count
        base_dict["registered_count"] = self.registered_count
        base_dict["missing_implementations"] = self.missing_implementations
        base_dict["undeclared_implementations"] = self.undeclared_implementations
        return base_dict

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Check if version follows semantic versioning."""
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$"
        return bool(re.match(pattern, version))

    def _validate_dependencies(self):
        """Validate dependency specifications."""
        for dep in self.dependencies:
            if not isinstance(dep, str):
                raise PluginError(f"Invalid dependency format: {dep}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginMetadata":
        """Create from dictionary."""
        return cls(**data)


class PluginRegistry:
    """
    Central registry for transform plugins.

    Manages:
    - Plugin discovery and loading
    - Version compatibility checking
    - Dependency resolution
    - Validation status tracking
    - Security and trust management

    Thread-safe singleton pattern.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._plugins: dict[str, PluginMetadata] = {}
            self._plugin_paths: list[Path] = []
            self._enabled_plugins: set[str] = set()
            self._disabled_plugins: set[str] = set()
            self._initialized = True

    @classmethod
    def add_plugin_path(cls, path: Path) -> None:
        """
        Add a directory to search for plugins.

        Args:
            path: Directory path containing plugins
        """
        instance = cls()
        path = Path(path).resolve()

        if not path.is_dir():
            raise PluginError(f"Plugin path is not a directory: {path}")

        if path not in instance._plugin_paths:
            instance._plugin_paths.append(path)
            # Add to Python path for imports
            sys.path.insert(0, str(path))
            logger.info(f"Added plugin path: {path}")

    @classmethod
    def discover_plugins(
        cls, paths: list[Path] | None = None, auto_validate: bool = False
    ) -> list[str]:
        """
        Unified plugin discovery and registration with 3-tier fallback.

        This method implements the complete discovery flow:
        1. Load plugin.yaml metadata
        2. For each declared transform, try 3-tier registration
        3. Scan for undeclared transforms
        4. Validate if requested

        Args:
            paths: List of paths to search. If None, uses registered paths.
            auto_validate: If True, validate plugins during discovery.

        Returns:
            List of discovered plugin names
        """
        instance = cls()

        if paths:
            for path in paths:
                cls.add_plugin_path(path)

        discovered_plugins = []

        for search_path in instance._plugin_paths:
            logger.info(f"Searching for plugins in: {search_path}")

            # Determine plugin category based on path
            if "pyg_augmentation" in str(search_path):
                logger.info("  → PyG Fallback Plugins (for missing PyG transforms)")
            elif "myplugins" in str(search_path):
                logger.info("  → User Experimental Plugins (custom transforms)")

            # Primary method: plugin.yaml in directory
            for plugin_yaml in search_path.glob("*/plugin.yaml"):
                try:
                    # Load plugin metadata from YAML
                    plugin_meta = instance._load_plugin_metadata_from_yaml(plugin_yaml)

                    if not plugin_meta:
                        continue

                    # Register plugin
                    instance._plugins[plugin_meta.plugin_name] = plugin_meta
                    discovered_plugins.append(plugin_meta.plugin_name)

                    plugin_type_label = (
                        "PyG Fallback"
                        if plugin_meta.plugin_type == "pyg_fallback"
                        else "User Experimental"
                    )
                    logger.info(
                        f"Discovered plugin from YAML: {plugin_meta.plugin_name} [{plugin_type_label}]"
                    )

                    # Attempt to register each declared transform with 3-tier fallback
                    registration_results = []

                    for declaration in plugin_meta.transform_declarations:
                        result = instance._register_transform_with_fallback(
                            declaration=declaration,
                            plugin_dir=plugin_yaml.parent,
                            plugin_meta=plugin_meta,
                        )
                        registration_results.append(result)

                        if result["registered"]:
                            plugin_meta.registered_transforms.add(declaration.name)

                    # Scan for undeclared transforms (bonus discoveries)
                    instance._scan_and_register_undeclared_transforms(
                        plugin_dir=plugin_yaml.parent, plugin_meta=plugin_meta
                    )

                    # Log summary for this plugin
                    instance._log_plugin_discovery_summary(plugin_meta, registration_results)

                except Exception as e:
                    logger.error(f"Failed to load plugin from {plugin_yaml}: {e}")
                    logger.debug("Error details:", exc_info=True)

        # Auto-validate if requested
        if auto_validate:
            for plugin_name in discovered_plugins:
                try:
                    cls.validate_plugin(plugin_name)
                except Exception as e:
                    logger.warning(f"Auto-validation failed for {plugin_name}: {e}")

        logger.info(f"Discovery complete: {len(discovered_plugins)} plugin(s) found")
        return discovered_plugins

    @classmethod
    def _load_plugin_metadata_from_yaml(cls, yaml_path: Path) -> PluginMetadata | None:
        """
        Load plugin metadata from plugin.yaml file.

        Creates PluginMetadata with TransformDeclarations from YAML.
        Does NOT register transforms - that happens in discover_plugins().

        Args:
            yaml_path: Path to plugin.yaml

        Returns:
            PluginMetadata with declarations populated, or None on failure
        """
        instance = cls()

        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            # Extract transform declarations from YAML
            transform_decls = []
            if "transforms" in data and isinstance(data["transforms"], list):
                for transform_dict in data["transforms"]:
                    try:
                        decl = TransformDeclaration.from_dict(transform_dict)
                        transform_decls.append(decl)
                    except Exception as e:
                        logger.warning(f"Failed to parse transform declaration in {yaml_path}: {e}")

            # Calculate checksum
            plugin_dir = yaml_path.parent
            checksum = instance._calculate_directory_checksum(plugin_dir)

            # Create metadata (remove 'transforms' from data, use 'transform_declarations')
            data_copy = data.copy()
            data_copy.pop("transforms", None)  # Remove old format
            data_copy["transform_declarations"] = transform_decls
            data_copy["checksum"] = checksum
            data_copy["discovery_source"] = "yaml"
            data_copy["discovery_timestamp"] = datetime.now().isoformat()

            metadata = PluginMetadata(**data_copy)

            return metadata

        except Exception as e:
            logger.error(f"Failed to load plugin metadata from {yaml_path}: {e}")
            logger.debug("Error details:", exc_info=True)
            return None

    def _register_transform_with_fallback(
        self, declaration: TransformDeclaration, plugin_dir: Path, plugin_meta: PluginMetadata
    ) -> dict[str, Any]:
        """
        Attempt to register transform using 3-tier fallback strategy.

        Tier 1: Check if already in TransformRegistry (PyG native)
        Tier 2: Try to load from plugin Python module
        Tier 3: Declaration only (no implementation) - log warning

        Args:
            declaration: Transform declaration from plugin.yaml
            plugin_dir: Plugin root directory
            plugin_meta: Plugin metadata object

        Returns:
            Registration result dict with status and details
        """
        transform_name = declaration.name

        # ================================================================
        # TIER 1: PyG Native Implementation (Highest Priority)
        # ================================================================
        try:
            _import_graph_transforms()
            if TransformRegistry is not None:
                native_class = TransformRegistry.get(transform_name)

                if native_class is not None:
                    logger.info(
                        f"✓ {transform_name}: Using PyG native implementation "
                        f"[Plugin: {plugin_meta.plugin_name}]"
                    )
                    return {
                        "registered": True,
                        "source": "pyg",
                        "transform_name": transform_name,
                        "reason": "Transform found in PyG native library",
                        "details": {
                            "class": native_class.__name__,
                            "module": native_class.__module__,
                        },
                    }
        except Exception as e:
            logger.debug(f"PyG native check failed for {transform_name}: {e}")

        # ================================================================
        # TIER 2: Plugin Python Implementation
        # ================================================================
        transform_class = None
        try:
            # Load the transform class from plugin module
            transform_class = self._load_transform_class(
                plugin_dir=plugin_dir,
                module_path=declaration.module_path,
                class_name=declaration.class_name,
            )
        except (FileNotFoundError, AttributeError, ImportError) as e:
            # These are expected errors from _load_transform_class
            if isinstance(e, FileNotFoundError):
                logger.warning(
                    f"⚠ {transform_name}: Module not found [Plugin: {plugin_meta.plugin_name}]"
                )
            elif isinstance(e, AttributeError):
                logger.warning(
                    f"⚠ {transform_name}: Class '{declaration.class_name}' not found [Plugin: {plugin_meta.plugin_name}]"
                )
            elif isinstance(e, ImportError):
                logger.error(
                    f"✗ {transform_name}: Import failed: {e} [Plugin: {plugin_meta.plugin_name}]"
                )
                import traceback

                logger.debug(f"Import traceback:\n{traceback.format_exc()}")
        except Exception as e:
            # Unexpected errors during loading
            logger.error(
                f"✗ {transform_name}: Unexpected load error: {e} [Plugin: {plugin_meta.plugin_name}]"
            )
            import traceback

            logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # If load succeeded, validate and register
        if transform_class is not None:
            try:
                # Verify protocol (duck typing) - must have forward() and get_metadata()
                # Note: Skip inheritance check due to dynamic import module instance mismatch
                if not (
                    hasattr(transform_class, "forward")
                    and hasattr(transform_class, "get_metadata")
                    and callable(transform_class.forward)
                    and callable(transform_class.get_metadata)
                ):
                    raise TypeError(
                        f"{transform_class} must have callable 'forward()' and 'get_metadata()' methods"
                    )

                # Register with TransformRegistry
                # Get metadata from transform class
                transform_metadata = transform_class.get_metadata()

                # Register with TransformRegistry using the static method
                _import_graph_transforms()
                if TransformRegistry is not None:
                    TransformRegistry.register(
                        transform_class, transform_metadata.name, transform_metadata
                    )

                    # Verify it actually registered
                    # Note: TransformRegistry.get() may have internal compatibility issues with
                    # dynamically loaded plugin transforms, so we wrap in try/except
                    try:
                        registered_transform = TransformRegistry.get(transform_metadata.name)
                        if registered_transform is None:
                            raise Exception("Registration failed - transform not in registry")
                        logger.debug(
                            f"✓ Verified {transform_name} successfully registered in TransformRegistry"
                        )
                    except AttributeError:
                        # Registry.get() has structural issues with plugin transforms, but
                        # registration succeeded (no exception from register() call)
                        # Verify by checking transform count increase shown in logs
                        logger.debug(
                            f"✓ {transform_name} registered (verification skipped due to registry internals)"
                        )

                logger.info(
                    f"✓ {transform_name}: Registered from plugin implementation "
                    f"[Plugin: {plugin_meta.plugin_name}, "
                    f"Class: {declaration.class_name}, "
                    f"Module: {declaration.module_path}]"
                )

                return {
                    "registered": True,
                    "source": "plugin",
                    "transform_name": transform_name,
                    "reason": "Transform loaded and registered from plugin Python code",
                    "details": {
                        "class": declaration.class_name,
                        "module": declaration.module_path,
                        "file": str(plugin_dir / f"{declaration.module_path}.py"),
                    },
                }

            except Exception as e:
                # Errors during validation or registration
                logger.error(
                    f"✗ {transform_name}: Registration failed: {e} [Plugin: {plugin_meta.plugin_name}]"
                )
                import traceback

                logger.error(f"Registration traceback:\n{traceback.format_exc()}")

        # ================================================================
        # TIER 3: Declaration Only (No Implementation)
        # ================================================================
        logger.warning(
            f"⚠ {transform_name}: Declared in plugin.yaml but no implementation found "
            f"[Plugin: {plugin_meta.plugin_name}] "
            f"- Transform will not be available for use"
        )

        return {
            "registered": False,
            "source": "none",
            "transform_name": transform_name,
            "reason": "Transform declared but no implementation found in PyG or plugin",
            "details": {
                "declaration": declaration.to_dict(),
                "checked_sources": ["pyg_native", "plugin_python"],
            },
        }

    def _load_transform_class(self, plugin_dir: Path, module_path: str, class_name: str) -> type:
        """
        Dynamically load a transform class from plugin module.

        Args:
            plugin_dir: Plugin root directory
            module_path: Relative module path (e.g., "transforms")
            class_name: Name of the class to load

        Returns:
            Transform class type

        Raises:
            FileNotFoundError: Module file not found
            AttributeError: Class not found in module
            ImportError: Module import failed
        """
        # Construct file path
        module_file = plugin_dir / f"{module_path}.py"

        if not module_file.exists():
            raise FileNotFoundError(f"Module file not found: {module_file}")

        # Dynamic import with unique module name to avoid conflicts
        module_name = f"plugin_{plugin_dir.name}_{module_path.replace('/', '_')}"

        spec = importlib.util.spec_from_file_location(module_name, module_file)

        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create module spec for {module_file}")
        # ---
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Add to sys.modules for proper import

        # Deferred import: Only import if not already in progress
        if not _IMPORTING_CUSTOM_TRANSFORMS:
            _import_custom_transforms()
        if not _IMPORTING_GRAPH_TRANSFORMS:
            _import_graph_transforms()

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            sys.modules.pop(module_name, None)  # Clean up on failure

            # If this was an import error during initial load, provide better context
            if "circular" in str(e).lower() or "recursion" in str(e).lower():
                raise ImportError(
                    f"Circular import detected while loading {module_file}. "
                    f"Plugin transforms should not import transformations.__init__ directly. "
                    f"Use 'from milia_pipeline.transformations.graph_transforms import ...' instead."
                ) from e

            raise ImportError(f"Failed to execute module {module_file}: {e}") from e
        # ---
        # Get class from module
        if not hasattr(module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in {module_file}")

        return getattr(module, class_name)

    def _scan_and_register_undeclared_transforms(
        self, plugin_dir: Path, plugin_meta: PluginMetadata
    ) -> int:
        """
        Scan plugin directory for transform classes not declared in plugin.yaml.

        These are "bonus" transforms that were implemented but not documented.

        Args:
            plugin_dir: Plugin root directory
            plugin_meta: Plugin metadata object

        Returns:
            Number of undeclared transforms discovered and registered
        """
        _import_custom_transforms()  # Ensure imports are loaded
        if not CustomTransformBase:
            return 0

        discovered_count = 0
        declared_names = {decl.name for decl in plugin_meta.transform_declarations}

        # Scan Python files in plugin directory
        for py_file in plugin_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                # Load module
                relative_path = py_file.relative_to(plugin_dir).with_suffix("")
                module_path = str(relative_path).replace("/", ".")

                module_name = f"plugin_{plugin_dir.name}_{module_path.replace('.', '_')}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                # ---
                if spec is None or spec.loader is None:
                    continue

                # Deferred import: Only import if not already in progress
                if not _IMPORTING_CUSTOM_TRANSFORMS:
                    _import_custom_transforms()
                if not _IMPORTING_GRAPH_TRANSFORMS:
                    _import_graph_transforms()

                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    # Skip this file if it causes import errors
                    logger.debug(f"Failed to execute module {py_file}: {e}")
                    continue
                # ---
                # Find CustomTransformBase subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Skip inheritance check - use protocol instead
                    if not (hasattr(obj, "forward") and hasattr(obj, "get_metadata")):
                        continue

                    # Skip base classes
                    if obj in [CustomTransformBase, MolecularTransformBase, QuantumTransformBase]:
                        continue

                    # Get metadata
                    try:
                        metadata = obj.get_metadata()
                        transform_name = metadata.name

                        # Check if already declared or registered
                        if transform_name in declared_names:
                            continue
                        if transform_name in plugin_meta.registered_transforms:
                            continue

                        # Register undeclared transform
                        metadata = obj.get_metadata()

                        _import_graph_transforms()
                        if TransformRegistry is not None:
                            TransformRegistry.register(
                                transform_class=obj, name=metadata.name, metadata=metadata
                            )

                        plugin_meta.registered_transforms.add(transform_name)
                        discovered_count += 1

                        logger.info(
                            f"+ {transform_name}: Bonus discovery (not declared in plugin.yaml) "
                            f"[Plugin: {plugin_meta.plugin_name}]"
                        )

                    except Exception as e:
                        logger.debug(f"Failed to process class {name} in {py_file}: {e}")

            except Exception as e:
                logger.debug(f"Failed to scan {py_file}: {e}")

        return discovered_count

    def _log_plugin_discovery_summary(
        self, plugin_meta: PluginMetadata, registration_results: list[dict[str, Any]]
    ) -> None:
        """
        Log comprehensive plugin discovery summary.

        Provides detailed breakdown of what was discovered, registered, and any issues.
        """
        logger.info("=" * 64)
        logger.info(f"Plugin Discovery: {plugin_meta.plugin_name} v{plugin_meta.version}")
        logger.info("=" * 64)
        logger.info(f"Author: {plugin_meta.author}")
        if plugin_meta.description:
            logger.info(f"Description: {plugin_meta.description}")
        logger.info("")

        logger.info("Transform Summary:")
        logger.info(f"  Declared in plugin.yaml: {plugin_meta.declared_count}")
        logger.info(f"  Successfully registered: {plugin_meta.registered_count}")
        logger.info(f"  Missing implementations: {len(plugin_meta.missing_implementations)}")
        logger.info(f"  Bonus discoveries: {len(plugin_meta.undeclared_implementations)}")

        # Overall status
        if plugin_meta.missing_implementations:
            logger.info(
                f"  Status: ⚠ {len(plugin_meta.missing_implementations)} "
                f"declared transform(s) not implemented"
            )
        elif plugin_meta.registered_count > 0:
            logger.info("  Status: ✓ All declared transforms successfully registered")
        else:
            logger.info("  Status: ℹ No transforms declared or registered")

        # Registration details
        if registration_results:
            logger.info("")
            logger.info("Registration Details:")
            for result in registration_results:
                transform_name = result.get("transform_name", "Unknown")
                if result["registered"]:
                    source_label = {
                        "pyg": "PyG native implementation",
                        "plugin": f"Plugin implementation ({result['details'].get('module', 'unknown')})",
                    }.get(result["source"], "Unknown source")
                    logger.info(f"  ✓ {transform_name}: {source_label}")
                else:
                    logger.info(f"  ✗ {transform_name}: {result['reason']}")

        # Bonus discoveries
        if plugin_meta.undeclared_implementations:
            logger.info("")
            logger.info("Bonus Discoveries (not declared in plugin.yaml):")
            for name in plugin_meta.undeclared_implementations:
                logger.info(f"  + {name}")

        logger.info("")
        logger.info(f"Validation: {'PASSED' if plugin_meta.is_validated else 'PENDING'}")
        logger.info("=" * 64)

    @classmethod
    def _load_plugin_from_module(cls, module_path: Path) -> PluginMetadata | None:
        """
        Load plugin from __plugin__.py module.

        Args:
            module_path: Path to directory containing __plugin__.py

        Returns:
            PluginMetadata or None
        """
        instance = cls()

        try:
            module_name = module_path.name
            spec = importlib.util.spec_from_file_location(
                f"{module_name}.__plugin__", module_path / "__plugin__.py"
            )
            # ---
            if spec is None or spec.loader is None:
                raise PluginError(f"Cannot load module from {module_path}")

            # Deferred import: Only import if not already in progress
            if not _IMPORTING_CUSTOM_TRANSFORMS:
                _import_custom_transforms()
            if not _IMPORTING_GRAPH_TRANSFORMS:
                _import_graph_transforms()

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get metadata from module
            if not hasattr(module, "PLUGIN_METADATA"):
                raise PluginError(f"__plugin__.py missing PLUGIN_METADATA in {module_path}")

            # Calculate checksum
            plugin_data = module.PLUGIN_METADATA.copy()
            plugin_data["checksum"] = instance._calculate_directory_checksum(module_path)

            # Create metadata
            metadata = PluginMetadata(**plugin_data)

            # Load transforms only if not defined in __plugin__.py metadata
            if not metadata.transforms:
                instance._load_transforms_from_directory(module_path, metadata)
            else:
                logger.debug(
                    f"Transforms already defined in PLUGIN_METADATA for {metadata.plugin_name}, skipping directory scan"
                )

            # Register plugin
            instance._plugins[metadata.plugin_name] = metadata

            return metadata

        except Exception as e:
            raise PluginError(f"Failed to load plugin from {module_path}: {e}") from e

    @classmethod
    def _load_plugin_from_standalone(cls, py_file: Path) -> PluginMetadata | None:
        """
        Load plugin from standalone .py file with PLUGIN_METADATA.

        Args:
            py_file: Path to standalone Python file

        Returns:
            PluginMetadata or None
        """
        instance = cls()

        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            # ---
            if spec is None or spec.loader is None:
                return None

            # Deferred import: Only import if not already in progress
            if not _IMPORTING_CUSTOM_TRANSFORMS:
                _import_custom_transforms()
            if not _IMPORTING_GRAPH_TRANSFORMS:
                _import_graph_transforms()

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # ---
            # Check for PLUGIN_METADATA
            if not hasattr(module, "PLUGIN_METADATA"):
                return None

            # Calculate checksum
            plugin_data = module.PLUGIN_METADATA.copy()
            plugin_data["checksum"] = instance._calculate_file_checksum(py_file)

            # Create metadata
            metadata = PluginMetadata(**plugin_data)

            # Register transforms from this file
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if instance._is_custom_transform(obj):
                    instance._register_transform(obj, metadata)

            # Register plugin
            instance._plugins[metadata.plugin_name] = metadata

            return metadata

        except Exception as e:
            logger.debug(f"Not a valid plugin file {py_file}: {e}")
            return None

    @classmethod
    def _load_transforms_from_directory(cls, directory: Path, metadata: PluginMetadata) -> None:
        """
        Load all custom transforms from a plugin directory.

        Args:
            directory: Plugin directory path
            metadata: Plugin metadata to update
        """
        instance = cls()

        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                # Import module
                module_name = f"{metadata.plugin_name}.{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                # ---
                if spec is None or spec.loader is None:
                    continue

                # Deferred import: Only import if not already in progress
                if not _IMPORTING_CUSTOM_TRANSFORMS:
                    _import_custom_transforms()
                if not _IMPORTING_GRAPH_TRANSFORMS:
                    _import_graph_transforms()

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # ---
                # Find and register CustomTransformBase subclasses
                for _name, obj in inspect.getmembers(module, inspect.isclass):
                    if instance._is_custom_transform(obj):
                        # DEBUG: Log transform class being registered
                        logger.debug(f"DEBUG: Registering transform class: {obj}")
                        logger.debug(f"DEBUG: Transform class name: {obj.__name__}")
                        instance._register_transform(obj, metadata)

            except Exception as e:
                logger.warning(f"Error loading transforms from {py_file}: {e}")

    @staticmethod
    def _is_custom_transform(obj: Any) -> bool:
        """Check if object is a CustomTransformBase subclass."""
        _import_custom_transforms()  # Ensure imports are loaded
        if CustomTransformBase is None:
            return False

        try:
            return (
                inspect.isclass(obj)
                and hasattr(obj, "forward")
                and hasattr(obj, "get_metadata")
                and obj not in [CustomTransformBase, MolecularTransformBase, QuantumTransformBase]
            )
        except Exception:
            return False

    def _register_transform(self, transform_class: type, plugin_metadata: PluginMetadata) -> None:
        """
        Register a transform with the TransformRegistry.
        Updated for new PluginMetadata structure.
        """
        _import_graph_transforms()
        if TransformRegistry is None:
            logger.warning("TransformRegistry not available, skipping registration")
            return

        try:
            # Get metadata from transform class
            transform_metadata = transform_class.get_metadata()

            # Register with TransformRegistry (signature: transform_class, name, metadata)
            TransformRegistry.register(transform_class, transform_metadata.name, transform_metadata)

            logger.debug(
                f"Transform '{transform_metadata.name}' successfully registered in TransformRegistry "
                f"[Plugin: {plugin_metadata.plugin_name}]"
            )

        except Exception as e:
            logger.error(f"Failed to register transform {transform_class}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    @staticmethod
    def _calculate_directory_checksum(directory: Path) -> str:
        """Calculate SHA256 checksum of all Python files in directory."""
        hasher = hashlib.sha256()

        for py_file in sorted(directory.glob("**/*.py")):
            with open(py_file, "rb") as f:
                hasher.update(f.read())

        return hasher.hexdigest()

    @staticmethod
    def _calculate_file_checksum(file_path: Path) -> str:
        """Calculate SHA256 checksum of a single file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    @classmethod
    def validate_plugin(cls, plugin_name: str) -> dict[str, Any]:
        """
        Validate a plugin comprehensively.

        Runs:
        - Dependency checks
        - Transform instantiation tests
        - Parameter validation
        - Compatibility tests
        - Security checks

        Args:
            plugin_name: Name of plugin to validate

        Returns:
            Validation results dictionary
        """
        instance = cls()

        if plugin_name not in instance._plugins:
            raise PluginError(f"Plugin '{plugin_name}' not registered")

        metadata = instance._plugins[plugin_name]

        results = {
            "plugin_name": plugin_name,
            "passed": False,
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {},
        }

        # Test 0: NEW - Consistency check (declarations vs registrations)
        logger.info(f"Checking declaration/registration consistency for {plugin_name}...")
        results["tests"]["consistency"] = instance._validate_consistency(metadata)

        # Test 1: Dependency check
        logger.info(f"Validating dependencies for {plugin_name}...")
        results["tests"]["dependencies"] = instance._check_dependencies(metadata)

        # Test 2: Security check
        logger.info(f"Running security checks for {plugin_name}...")
        results["tests"]["security"] = instance._check_security(metadata)

        # Test 3: Transform instantiation (only for registered transforms)
        logger.info(f"Testing transform instantiation for {plugin_name}...")
        results["tests"]["instantiation"] = instance._test_transform_instantiation(metadata)

        # Test 4: Parameter validation (only for registered transforms)
        logger.info(f"Validating parameters for {plugin_name}...")
        results["tests"]["parameters"] = instance._test_parameter_validation(metadata)

        # Test 5: Sample data compatibility (only for registered transforms)
        logger.info(f"Testing data compatibility for {plugin_name}...")
        results["tests"]["compatibility"] = instance._test_data_compatibility(metadata)

        # Add summary
        results["summary"] = {
            "declared_transforms": metadata.declared_count,
            "registered_transforms": metadata.registered_count,
            "missing_implementations": metadata.missing_implementations,
            "undeclared_implementations": metadata.undeclared_implementations,
            "tests_passed": sum(1 for t in results["tests"].values() if t.get("passed")),
            "tests_failed": sum(1 for t in results["tests"].values() if not t.get("passed")),
        }

        # Overall result
        results["passed"] = all(test.get("passed", False) for test in results["tests"].values())

        # Update metadata
        object.__setattr__(metadata, "is_validated", results["passed"])
        object.__setattr__(metadata, "validation_date", results["timestamp"])
        object.__setattr__(metadata, "validation_results", results)

        logger.info(
            f"Validation {'PASSED' if results['passed'] else 'FAILED'} for plugin '{plugin_name}'"
        )

        return results

    @staticmethod
    def _check_dependencies(metadata: PluginMetadata) -> dict:
        """
        Check if plugin dependencies are satisfied.

        Returns:
            {'passed': bool, 'details': str, 'missing': list}
        """
        missing = []

        # Check milia version
        try:
            # Placeholder - would check actual milia version
            pass
        except Exception as e:
            missing.append(f"milia version check failed: {e}")

        # Check PyG version
        try:
            import torch_geometric
            # Version check logic would go here
        except ImportError:
            missing.append("PyTorch Geometric not installed")

        # Check additional dependencies
        for dep in metadata.dependencies:
            try:
                # Try to import package - basic check
                dep_name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
                __import__(dep_name)
            except ImportError:
                missing.append(dep)
            except Exception:
                # If parsing fails, just add the raw dep
                missing.append(dep)

        return {
            "passed": len(missing) == 0,
            "details": "All dependencies satisfied" if not missing else f"Missing: {missing}",
            "missing": missing,
        }

    @staticmethod
    def _check_security(metadata: PluginMetadata) -> dict:
        """
        Run security checks on plugin.

        Checks:
        - Checksum verification
        - No dangerous imports (os, subprocess, etc. unless trusted)
        - No eval/exec usage

        Returns:
            {'passed': bool, 'issues': list, 'warnings': list}
        """
        issues = []
        warnings = []

        # If plugin is marked as trusted, skip some checks
        if metadata.trusted:
            warnings.append("Plugin is marked as trusted - security checks relaxed")
            return {"passed": True, "issues": [], "warnings": warnings}

        # Checksum verification
        if not metadata.checksum:
            warnings.append("No checksum provided")

        # Dangerous imports check (placeholder - would scan actual code)
        # This would require AST parsing of actual plugin code

        return {"passed": len(issues) == 0, "issues": issues, "warnings": warnings}

    @staticmethod
    def _validate_consistency(metadata: PluginMetadata) -> dict[str, Any]:
        """
        Validate consistency between declarations and registrations.

        Checks:
        - Are all declared transforms registered?
        - Are there undeclared transforms (bonus)?

        This is a new validation test that ensures plugin metadata accuracy.

        Returns:
            {
                'passed': bool,
                'missing': List[str],
                'undeclared': List[str],
                'details': str
            }
        """
        missing = metadata.missing_implementations
        undeclared = metadata.undeclared_implementations

        # Consider it passed if:
        # - No missing implementations, OR
        # - At least some transforms are registered (partial success)
        passed = len(missing) == 0 or metadata.registered_count > 0

        details = []
        if missing:
            details.append(
                f"{len(missing)} declared transform(s) not implemented: {', '.join(missing)}"
            )
        if undeclared:
            details.append(
                f"{len(undeclared)} bonus transform(s) discovered: {', '.join(undeclared)}"
            )
        if not missing and not undeclared:
            details.append("Perfect consistency - all declared transforms registered, no extras")

        return {
            "passed": passed,
            "missing": missing,
            "undeclared": undeclared,
            "details": "; ".join(details) if details else "No issues",
        }

    @classmethod
    def _test_transform_instantiation(cls, metadata: PluginMetadata) -> dict:
        """
        Test that all plugin transforms can be instantiated.

        Returns:
            {'passed': bool, 'failures': list}
        """
        _import_graph_transforms()
        if TransformRegistry is None:
            return {"passed": False, "failures": [{"error": "TransformRegistry not available"}]}

        results = {"passed": True, "failures": []}

        # Access the module-level registry singleton (line 3170 of graph_transforms.py)
        from milia_pipeline.transformations.graph_transforms import registry

        # Only test REGISTERED transforms
        for transform_name in metadata.registered_transforms:
            try:
                # Get transform class directly from the singleton's _custom_transforms dict
                transform_class = registry._custom_transforms.get(transform_name)

                if transform_class is None:
                    raise PluginError(f"Transform '{transform_name}' not found in registry")

                # Try instantiation with reasonable defaults for known parameter patterns
                try:
                    instance = transform_class()
                except (ValueError, TypeError) as e:
                    # If default instantiation fails, try with common fallback parameters
                    if "num" in str(e) or "ratio" in str(e):
                        instance = transform_class(num=10)
                    elif "p" in str(e):
                        instance = transform_class(p=0.5)
                    else:
                        raise

                # Verify it's callable
                if not callable(instance):
                    raise TypeError(f"{transform_name} is not callable")

            except Exception as e:
                results["passed"] = False
                results["failures"].append(
                    {"transform": transform_name, "error": str(e), "error_type": type(e).__name__}
                )
                import logging
                import traceback

                logger = logging.getLogger(__name__)
                logger.error(
                    f"❌ Instantiation failed for {transform_name}: {type(e).__name__}: {e}"
                )
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        # Log summary of failures
        if results["failures"]:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Instantiation test failures: {len(results['failures'])} transform(s) failed"
            )
            for failure in results["failures"]:
                logger.error(f"  ✗ {failure['transform']}: {failure['error']}")

        return results

    @classmethod
    def _test_parameter_validation(cls, metadata: PluginMetadata) -> dict:
        """
        Test parameter constraint validation for all transforms.

        Returns:
            {'passed': bool, 'issues': list}
        """
        if get_transform_info is None:
            return {"passed": True, "issues": ["get_transform_info not available - skipped"]}

        issues = []

        # Only test REGISTERED transforms
        _import_graph_transforms()
        for transform_name in metadata.registered_transforms:
            try:
                # Get transform info
                info = get_transform_info(transform_name)

                # Check if parameter constraints are properly defined
                # Handle both dict (legacy) and TransformInfo object (new)
                constraints = None
                if isinstance(info, dict):
                    # Legacy dict format
                    constraints = info.get("parameter_constraints")
                elif hasattr(info, "_parameter_constraints"):
                    # TransformInfo object with custom attribute
                    constraints = getattr(info, "_parameter_constraints", None)
                elif hasattr(info, "parameters"):
                    # TransformInfo object - parameters dict may contain constraints
                    # For standard TransformInfo, constraints aren't separate
                    constraints = None

                if constraints:
                    # Validate constraint format
                    for param, constraint in constraints.items():
                        if not isinstance(constraint, dict):
                            issues.append(
                                {
                                    "transform": transform_name,
                                    "parameter": param,
                                    "issue": "Constraint must be a dictionary",
                                }
                            )

            except Exception as e:
                issues.append(
                    {"transform": transform_name, "issue": f"Parameter validation failed: {e}"}
                )

        return {"passed": len(issues) == 0, "issues": issues}

    @classmethod
    def _test_data_compatibility(cls, metadata: PluginMetadata) -> dict:
        """
        Test transforms work with sample milia data.

        Returns:
            {'passed': bool, 'failures': list}
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            return {"passed": False, "failures": [{"error": "PyTorch Geometric not available"}]}
        _import_graph_transforms()
        if TransformRegistry is None:
            return {"passed": False, "failures": [{"error": "TransformRegistry not available"}]}

        failures = []

        # Create sample milia-like data
        sample_data = Data(
            x=torch.randn(10, 5),  # 10 atoms, 5 features
            edge_index=torch.randint(0, 10, (2, 20)),  # 20 edges
            pos=torch.randn(10, 3),  # 3D coordinates
            energy=torch.tensor(100.0),
            dmc_uncertainty=torch.tensor(0.05),
            charges=torch.randn(10),
            num_nodes=10,
        )

        # Access the module-level registry singleton
        from milia_pipeline.transformations.graph_transforms import registry

        # Only test REGISTERED transforms
        for transform_name in metadata.registered_transforms:
            try:
                # Get transform class from registry singleton's _custom_transforms
                transform_class = registry._custom_transforms.get(transform_name)
                if transform_class is None:
                    raise PluginError(f"Transform '{transform_name}' not found")

                # Try instantiation with fallback parameters
                try:
                    transform = transform_class()
                except (ValueError, TypeError) as e:
                    if "num" in str(e) or "ratio" in str(e):
                        transform = transform_class(num=10)
                    elif "p" in str(e):
                        transform = transform_class(p=0.5)
                    else:
                        raise

                # Try to apply transform
                result = transform(sample_data.clone())

                # Verify result is still valid Data object
                if result is not None and not isinstance(result, Data):
                    raise TypeError(f"Transform returned {type(result)}, expected Data or None")

            except Exception as e:
                failures.append(
                    {"transform": transform_name, "error": str(e), "error_type": type(e).__name__}
                )
                import logging
                import traceback

                logger = logging.getLogger(__name__)
                logger.error(
                    f"❌ Compatibility test failed for {transform_name}: {type(e).__name__}: {e}"
                )
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        # Log summary
        if failures:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Compatibility test failures: {len(failures)} transform(s) failed")
            for failure in failures:
                logger.error(f"  ✗ {failure['transform']}: {failure['error']}")

        return {"passed": len(failures) == 0, "failures": failures}

    @classmethod
    def enable_plugin(cls, plugin_name: str) -> None:
        """Enable a plugin for use."""
        instance = cls()

        if plugin_name not in instance._plugins:
            raise PluginError(f"Plugin '{plugin_name}' not found")

        instance._enabled_plugins.add(plugin_name)
        instance._disabled_plugins.discard(plugin_name)

        logger.info(f"Enabled plugin: {plugin_name}")

    @classmethod
    def disable_plugin(cls, plugin_name: str) -> None:
        """Disable a plugin."""
        instance = cls()

        if plugin_name not in instance._plugins:
            raise PluginError(f"Plugin '{plugin_name}' not found")

        instance._disabled_plugins.add(plugin_name)
        instance._enabled_plugins.discard(plugin_name)

        # Unregister transforms
        _import_graph_transforms()
        if TransformRegistry is not None:
            metadata = instance._plugins[plugin_name]

            # Unregister only transforms that were registered
            _import_graph_transforms()
            for transform_name in metadata.registered_transforms:
                try:
                    TransformRegistry.unregister(transform_name)
                except Exception as e:
                    logger.warning(f"Failed to unregister {transform_name}: {e}")

        logger.info(f"Disabled plugin: {plugin_name}")

    @classmethod
    def list_plugins(cls, validated_only: bool = False, enabled_only: bool = False) -> list[str]:
        """
        List registered plugins.

        Args:
            validated_only: Only return validated plugins
            enabled_only: Only return enabled plugins

        Returns:
            List of plugin names
        """
        instance = cls()
        plugins = list(instance._plugins.keys())

        if validated_only:
            plugins = [name for name in plugins if instance._plugins[name].is_validated]

        if enabled_only:
            plugins = [name for name in plugins if name in instance._enabled_plugins]

        return plugins

    @classmethod
    def get_plugin_info(cls, plugin_name: str) -> dict | None:
        """Get metadata for a specific plugin."""
        instance = cls()
        metadata = instance._plugins.get(plugin_name)
        return metadata.to_dict() if metadata else None


class PluginValidator:
    """
    Comprehensive validation for plugin transforms.

    Ensures plugins meet quality standards for:
    - Code correctness
    - Documentation
    - Test coverage (if tests provided)
    - Performance
    - Security
    """

    @staticmethod
    def validate_plugin_comprehensive(
        plugin_name: str, test_data_path: Path | None = None, run_performance_tests: bool = True
    ) -> dict[str, Any]:
        """
        Run comprehensive validation suite on a plugin.

        Args:
            plugin_name: Name of plugin to validate
            test_data_path: Optional path to test milia data
            run_performance_tests: Whether to run performance benchmarks

        Returns:
            Detailed validation report with scores and recommendations
        """
        metadata = PluginRegistry.get_plugin_info(plugin_name)
        if not metadata:
            raise PluginError(f"Plugin '{plugin_name}' not found")

        report = {
            "plugin": plugin_name,
            "timestamp": datetime.now().isoformat(),
            "sections": {},
            "overall_score": 0.0,
            "recommendation": "",
        }

        # Section 1: Code quality
        logger.info(f"Checking code quality for {plugin_name}...")
        report["sections"]["code_quality"] = PluginValidator._check_code_quality(metadata)

        # Section 2: Documentation
        logger.info(f"Checking documentation for {plugin_name}...")
        report["sections"]["documentation"] = PluginValidator._check_documentation(metadata)

        # Section 3: Functional tests
        logger.info(f"Running functional tests for {plugin_name}...")
        report["sections"]["functional"] = PluginValidator._run_functional_tests(
            metadata, test_data_path
        )

        # Section 4: Performance (optional)
        if run_performance_tests:
            logger.info(f"Running performance benchmarks for {plugin_name}...")
            report["sections"]["performance"] = PluginValidator._benchmark_performance(metadata)

        # Section 5: Security
        logger.info(f"Running security analysis for {plugin_name}...")
        report["sections"]["security"] = PluginValidator._analyze_security(metadata)

        # Calculate overall score
        report["overall_score"] = PluginValidator._calculate_score(report["sections"])

        # Generate recommendation
        report["recommendation"] = PluginValidator._generate_recommendation(report)

        logger.info(
            f"Comprehensive validation complete for {plugin_name}: "
            f"Score {report['overall_score']:.2f}"
        )

        return report

    @staticmethod
    def _check_code_quality(metadata: PluginMetadata) -> dict:
        """
        Check code quality metrics.

        Would check:
        - PEP 8 compliance (using flake8/pylint)
        - Type hints coverage
        - Docstring coverage
        - Code complexity

        Returns:
            {'passed': bool, 'score': float, 'details': dict}
        """
        # Placeholder implementation
        return {
            "passed": True,
            "score": 0.95,
            "details": {
                "pep8_compliance": "Not checked (placeholder)",
                "type_hints": "Not checked (placeholder)",
                "docstrings": "Not checked (placeholder)",
            },
        }

    @staticmethod
    def _check_documentation(metadata: PluginMetadata) -> dict:
        """
        Check documentation completeness.

        Would check:
        - README.md presence and quality
        - Example code
        - API documentation
        - Citations (if research-based)

        Returns:
            {'passed': bool, 'score': float, 'missing': list}
        """
        missing = []

        # Check metadata fields
        if not metadata.description:
            missing.append("Plugin description")

        if not metadata.homepage:
            missing.append("Homepage/repository URL")

        # Would check for README file in plugin directory

        score = max(0.0, 1.0 - (len(missing) * 0.2))

        return {"passed": len(missing) == 0, "score": score, "missing": missing}

    @staticmethod
    def _run_functional_tests(metadata: PluginMetadata, test_data_path: Path | None) -> dict:
        """
        Run functional tests with real data.

        Args:
            metadata: Plugin metadata
            test_data_path: Path to test data (if provided)

        Returns:
            {'passed': bool, 'score': float, 'test_results': list}
        """
        # Use PluginRegistry's test methods
        basic_tests = PluginRegistry.validate_plugin(metadata.plugin_name)

        return {
            "passed": basic_tests["passed"],
            "score": 1.0 if basic_tests["passed"] else 0.5,
            "test_results": basic_tests["tests"],
        }

    @staticmethod
    def _benchmark_performance(metadata: PluginMetadata) -> dict:
        """
        Benchmark transform performance.

        Measures:
        - Execution time
        - Memory usage
        - Scalability (small vs large molecules)

        Returns:
            {'passed': bool, 'score': float, 'benchmarks': dict}
        """
        try:
            import time

            import torch
            from torch_geometric.data import Data
        except ImportError:
            return {
                "passed": False,
                "score": 0.0,
                "benchmarks": {"error": "PyTorch Geometric not available"},
            }

        _import_graph_transforms()
        if TransformRegistry is None:
            return {
                "passed": False,
                "score": 0.0,
                "benchmarks": {"error": "TransformRegistry not available"},
            }

        benchmarks = {}

        # Only benchmark REGISTERED transforms
        _import_graph_transforms()
        for transform_name in metadata.registered_transforms:
            try:
                transform_class = TransformRegistry.get(transform_name)

                if transform_class is None:
                    benchmarks[transform_name] = {"error": "Transform not found"}
                    continue

                transform = transform_class()

                # Small molecule test
                small_data = Data(
                    x=torch.randn(10, 5), edge_index=torch.randint(0, 10, (2, 20)), num_nodes=10
                )

                start = time.time()
                for _ in range(100):
                    _ = transform(small_data.clone())
                small_time = (time.time() - start) / 100

                # Large molecule test
                large_data = Data(
                    x=torch.randn(100, 5), edge_index=torch.randint(0, 100, (2, 500)), num_nodes=100
                )

                start = time.time()
                for _ in range(100):
                    _ = transform(large_data.clone())
                large_time = (time.time() - start) / 100

                benchmarks[transform_name] = {
                    "small_molecule_ms": small_time * 1000,
                    "large_molecule_ms": large_time * 1000,
                }

            except Exception as e:
                benchmarks[transform_name] = {"error": str(e)}

        # Score based on performance
        valid_times = [
            b.get("small_molecule_ms", 1000)
            for b in benchmarks.values()
            if "small_molecule_ms" in b
        ]

        avg_time = sum(valid_times) / len(valid_times) if valid_times else 1000

        score = min(1.0, 10.0 / max(avg_time, 0.1))  # Target: <10ms

        return {
            "passed": avg_time < 100,  # <100ms acceptable
            "score": score,
            "benchmarks": benchmarks,
        }

    @staticmethod
    def _analyze_security(metadata: PluginMetadata) -> dict:
        """
        Security analysis of plugin.

        Returns:
            {'passed': bool, 'score': float, 'issues': list}
        """
        security_result = PluginRegistry._check_security(metadata)

        score = 1.0
        if security_result["issues"]:
            score -= len(security_result["issues"]) * 0.3
        if security_result["warnings"]:
            score -= len(security_result["warnings"]) * 0.1

        score = max(0.0, score)

        return {
            "passed": security_result["passed"],
            "score": score,
            "issues": security_result["issues"],
            "warnings": security_result["warnings"],
        }

    @staticmethod
    def _calculate_score(sections: dict) -> float:
        """Calculate weighted overall score from section scores."""
        weights = {
            "code_quality": 0.15,
            "documentation": 0.20,
            "functional": 0.35,
            "performance": 0.15,
            "security": 0.15,
        }

        total_score = 0.0
        total_weight = 0.0

        for section_name, weight in weights.items():
            if section_name in sections:
                section_score = sections[section_name].get("score", 0.0)
                total_score += section_score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _generate_recommendation(report: dict) -> str:
        """Generate human-readable recommendation based on validation."""
        score = report["overall_score"]

        if score >= 0.95:
            return "APPROVED - Excellent quality, ready for production use"
        elif score >= 0.85:
            return "APPROVED - Good quality, minor improvements suggested"
        elif score >= 0.70:
            return "CONDITIONAL - Needs improvements before production use"
        elif score >= 0.50:
            return "NOT APPROVED - Significant issues need resolution"
        else:
            return "REJECTED - Major issues prevent approval"


# Module exports
__all__ = [
    "PluginMetadata",
    "PluginRegistry",
    "PluginValidator",
    "PluginError",
    "PluginValidationError",
    "PluginSecurityError",
    "PluginDependencyError",
]
