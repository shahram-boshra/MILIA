"""
Descriptor Plugin System for milia Pipeline

Provides secure, extensible plugin system for custom molecular descriptors:
- Plugin discovery from YAML configuration
- Comprehensive validation (structure, security, dependencies)
- Version management and compatibility checking
- Thread-safe registry with enable/disable functionality

Mirrors the pattern from milia_pipeline/transformations/plugin_system.py

Pydantic V2 Migration (Phase 32):
    - Migrated DescriptorDeclaration from @dataclass to Pydantic BaseModel (mutable)
    - Migrated DescriptorPluginMetadata from @dataclass to Pydantic BaseModel (mutable)
    - Uses Field(default_factory=list/set/dict) for mutable defaults
    - Uses @model_validator(mode='after') to replace __post_init__ validation
    - Custom __hash__ preserved for set/dict usage
    - Added to_dict() methods wrapping model_dump() for backward compatibility
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: milia Team
Version: 1.1.0
"""

import hashlib
import importlib
import importlib.util
import inspect
import logging
import re
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from milia_pipeline.exceptions import (
    DescriptorPluginError,
)

from .descriptor_categories import DescriptorCategory, DescriptorMetadata
from .descriptor_registry import registry as descriptor_registry

logger = logging.getLogger(__name__)


# =============================================================================
# DESCRIPTOR DECLARATION DATACLASS
# =============================================================================


class DescriptorDeclaration(BaseModel):
    """
    Declaration of a descriptor from plugin.yaml.

    Pydantic V2 Migration (Phase 32):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Added to_dict() method wrapping model_dump() for backward compatibility
        - NON-BREAKING: Same constructor API and attribute access preserved

    Attributes:
        name: Descriptor name (must be unique)
        function_name: Name of the function in Python module
        module_path: Relative path to Python module (e.g., "descriptors")
        category: Descriptor category
        description: Brief description
        requires_3d: Whether 3D coordinates required
        requires_charges: Whether partial charges required
        version: Descriptor version
    """

    name: str
    function_name: str
    module_path: str
    category: str = "constitutional"
    description: str = ""
    requires_3d: bool = False
    requires_charges: bool = False
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Backward compatible method wrapping Pydantic V2's model_dump().

        Returns:
            Dictionary with all 8 fields
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DescriptorDeclaration":
        """Create from dictionary (from plugin.yaml)"""
        return cls(
            name=data["name"],
            function_name=data["function_name"],
            module_path=data["module_path"],
            category=data.get("category", "constitutional"),
            description=data.get("description", ""),
            requires_3d=data.get("requires_3d", False),
            requires_charges=data.get("requires_charges", False),
            version=data.get("version", "1.0.0"),
        )


# =============================================================================
# PLUGIN METADATA DATACLASS
# =============================================================================


class DescriptorPluginMetadata(BaseModel):
    """
    Metadata for a descriptor plugin package.

    Separates descriptor declarations (from YAML) from registrations (runtime state).

    Pydantic V2 Migration (Phase 32):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=list/set/dict) for mutable defaults
        - Uses @model_validator(mode='after') to replace __post_init__ validation
        - Custom __hash__ preserved for set/dict usage
        - NON-BREAKING: Same constructor API and attribute access preserved

    Attributes:
        plugin_name: Unique plugin identifier
        version: Semantic version string
        author: Author name
        email: Optional contact email
        license: License type (default: MIT)
        description: Plugin description
        homepage: Optional homepage URL
        milia_version: Required milia version
        python_version: Required Python version
        dependencies: List of required Python packages
        descriptor_declarations: Descriptors declared in plugin.yaml
        registered_descriptors: Descriptors actually registered (runtime)
        discovery_source: How plugin was discovered
        discovery_timestamp: When plugin was discovered
        is_validated: Whether plugin passed validation
        validation_date: When validation occurred
        validation_results: Validation result details
        checksum: Plugin directory checksum
        trusted: Whether plugin is trusted
    """

    # Required fields
    plugin_name: str
    version: str
    author: str

    # Optional metadata
    email: str | None = None
    license: str = "MIT"
    description: str = ""
    homepage: str | None = None

    # Version dependencies
    milia_version: str = ">=1.0.0"
    python_version: str = ">=3.8"
    dependencies: list[str] = Field(default_factory=list)

    # Declarations vs Registrations (CRITICAL SEPARATION)
    descriptor_declarations: list[DescriptorDeclaration] = Field(default_factory=list)
    registered_descriptors: set[str] = Field(default_factory=set)

    # Discovery metadata
    discovery_source: str = "unknown"  # "yaml", "python", "hybrid"
    discovery_timestamp: str | None = None

    # Validation status
    is_validated: bool = False
    validation_date: str | None = None
    validation_results: dict[str, Any] = Field(default_factory=dict)

    # Security
    checksum: str | None = None
    trusted: bool = False

    def __hash__(self) -> int:
        """Make hashable for use in sets/dicts"""
        return hash((self.plugin_name, self.version))

    def __eq__(self, other: object) -> bool:
        """Equality comparison for consistency with __hash__"""
        if isinstance(other, DescriptorPluginMetadata):
            return self.plugin_name == other.plugin_name and self.version == other.version
        return False

    @property
    def declared_count(self) -> int:
        """Number of descriptors declared in plugin.yaml"""
        return len(self.descriptor_declarations)

    @property
    def registered_count(self) -> int:
        """Number of descriptors actually registered"""
        return len(self.registered_descriptors)

    @property
    def missing_implementations(self) -> list[str]:
        """Descriptors declared but not registered"""
        declared_names = {decl.name for decl in self.descriptor_declarations}
        return list(declared_names - self.registered_descriptors)

    @property
    def undeclared_implementations(self) -> list[str]:
        """Descriptors registered but not declared (bonus discoveries)"""
        declared_names = {decl.name for decl in self.descriptor_declarations}
        return list(self.registered_descriptors - declared_names)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Backward compatible method that includes computed properties.
        Uses model_dump() for base fields and adds computed properties.

        Returns:
            Dictionary with all 19 fields + 4 computed properties
        """
        # Get base fields from model_dump, excluding nested models for manual handling
        result = {
            "plugin_name": self.plugin_name,
            "version": self.version,
            "author": self.author,
            "email": self.email,
            "license": self.license,
            "description": self.description,
            "homepage": self.homepage,
            "milia_version": self.milia_version,
            "python_version": self.python_version,
            "dependencies": self.dependencies,
            "descriptor_declarations": [decl.to_dict() for decl in self.descriptor_declarations],
            "registered_descriptors": list(self.registered_descriptors),
            "discovery_source": self.discovery_source,
            "is_validated": self.is_validated,
            "validation_date": self.validation_date,
            "checksum": self.checksum,
            "trusted": self.trusted,
            # Computed properties
            "declared_count": self.declared_count,
            "registered_count": self.registered_count,
            "missing_implementations": self.missing_implementations,
            "undeclared_implementations": self.undeclared_implementations,
        }
        return result

    @model_validator(mode="after")
    def validate_plugin_metadata(self) -> "DescriptorPluginMetadata":
        """
        Validate plugin metadata on creation.

        Replaces __post_init__ validation from dataclass pattern.
        """
        if not self.plugin_name:
            raise DescriptorPluginError("Plugin name is required")

        # Validate version format
        if not self._is_valid_version(self.version):
            raise DescriptorPluginError(
                f"Invalid version format: {self.version}", plugin_name=self.plugin_name
            )

        # Validate dependencies
        self._validate_dependencies()

        return self

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Check if version follows semantic versioning"""
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$"
        return bool(re.match(pattern, version))

    def _validate_dependencies(self):
        """Validate dependency specifications"""
        for dep in self.dependencies:
            if not isinstance(dep, str):
                raise DescriptorPluginError(
                    f"Invalid dependency format: {dep}", plugin_name=self.plugin_name
                )


# =============================================================================
# DESCRIPTOR PLUGIN LOADER
# =============================================================================


class DescriptorPluginLoader:
    """
    Loader for descriptor plugins.

    Handles plugin discovery, validation, and descriptor registration.
    Thread-safe singleton pattern.

    Usage:
        >>> loader = DescriptorPluginLoader()
        >>> loader.add_plugin_path(Path("./plugins/descriptors"))
        >>> plugins = loader.discover_plugins()
        >>> loader.validate_plugin("my_plugin")
    """

    _instance: Optional["DescriptorPluginLoader"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        """Implement singleton pattern with thread safety"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize loader (only once due to singleton)"""
        if not hasattr(self, "_initialized"):
            self._plugins: dict[str, DescriptorPluginMetadata] = {}
            self._plugin_paths: list[Path] = []
            self._enabled_plugins: set[str] = set()
            self._disabled_plugins: set[str] = set()
            self._initialized = True
            logger.info("DescriptorPluginLoader initialized")

    # =========================================================================
    # PLUGIN PATH MANAGEMENT
    # =========================================================================

    def add_plugin_path(self, path: Path) -> None:
        """
        Add a directory to search for plugins.

        Args:
            path: Directory path containing plugins

        Raises:
            DescriptorPluginError: If path is invalid
        """
        path = Path(path).resolve()

        if not path.is_dir():
            raise DescriptorPluginError(f"Plugin path is not a directory: {path}")

        if path not in self._plugin_paths:
            self._plugin_paths.append(path)
            # Add to Python path for imports
            sys.path.insert(0, str(path))
            logger.info(f"Added plugin path: {path}")

    # =========================================================================
    # PLUGIN DISCOVERY
    # =========================================================================

    def discover_plugins(
        self, paths: list[Path] | None = None, auto_validate: bool = False
    ) -> list[str]:
        """
        Discover and load descriptor plugins.

        Discovery flow:
        1. Load plugin.yaml metadata
        2. For each declared descriptor, register with DescriptorRegistry
        3. Scan for undeclared descriptors
        4. Validate if requested

        Args:
            paths: List of paths to search (if None, uses registered paths)
            auto_validate: If True, validate plugins during discovery

        Returns:
            List of discovered plugin names
        """
        if paths:
            for path in paths:
                self.add_plugin_path(path)

        discovered_plugins = []

        for search_path in self._plugin_paths:
            logger.info(f"Searching for descriptor plugins in: {search_path}")

            # Primary method: plugin.yaml in directory
            for plugin_yaml in search_path.glob("*/plugin.yaml"):
                try:
                    # Load plugin metadata from YAML
                    plugin_meta = self._load_plugin_metadata_from_yaml(plugin_yaml)

                    if not plugin_meta:
                        continue

                    # Register plugin
                    self._plugins[plugin_meta.plugin_name] = plugin_meta
                    discovered_plugins.append(plugin_meta.plugin_name)

                    logger.info(
                        f"Discovered plugin: {plugin_meta.plugin_name} v{plugin_meta.version}"
                    )

                    # Register each declared descriptor
                    registration_results = []

                    for declaration in plugin_meta.descriptor_declarations:
                        result = self._register_descriptor_from_declaration(
                            declaration=declaration,
                            plugin_dir=plugin_yaml.parent,
                            plugin_meta=plugin_meta,
                        )
                        registration_results.append(result)

                        if result["registered"]:
                            plugin_meta.registered_descriptors.add(declaration.name)

                    # Scan for undeclared descriptors
                    bonus_count = self._scan_and_register_undeclared_descriptors(
                        plugin_dir=plugin_yaml.parent, plugin_meta=plugin_meta
                    )

                    # Log summary
                    self._log_plugin_discovery_summary(plugin_meta, registration_results)

                    # Validate if requested
                    if auto_validate:
                        self.validate_plugin(plugin_meta.plugin_name)

                except Exception as e:
                    logger.error(f"Failed to load plugin from {plugin_yaml}: {e}")
                    logger.debug("Error details:", exc_info=True)

        logger.info(f"Plugin discovery complete: {len(discovered_plugins)} plugins discovered")
        return discovered_plugins

    def _load_plugin_metadata_from_yaml(self, yaml_path: Path) -> DescriptorPluginMetadata | None:
        """
        Load plugin metadata from plugin.yaml file.

        Args:
            yaml_path: Path to plugin.yaml

        Returns:
            DescriptorPluginMetadata or None if loading fails
        """
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning(f"Empty plugin.yaml: {yaml_path}")
                return None

            # Parse descriptor declarations
            descriptor_decls = []
            if "descriptors" in data:
                for desc_dict in data["descriptors"]:
                    try:
                        decl = DescriptorDeclaration.from_dict(desc_dict)
                        descriptor_decls.append(decl)
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse descriptor declaration in {yaml_path}: {e}"
                        )

            # Calculate checksum
            plugin_dir = yaml_path.parent
            checksum = self._calculate_directory_checksum(plugin_dir)

            # Create metadata
            data_copy = data.copy()
            data_copy.pop("descriptors", None)  # Remove old format
            data_copy["descriptor_declarations"] = descriptor_decls
            data_copy["checksum"] = checksum
            data_copy["discovery_source"] = "yaml"
            data_copy["discovery_timestamp"] = datetime.now().isoformat()

            metadata = DescriptorPluginMetadata(**data_copy)

            return metadata

        except Exception as e:
            logger.error(f"Failed to load plugin metadata from {yaml_path}: {e}")
            logger.debug("Error details:", exc_info=True)
            return None

    # =========================================================================
    # DESCRIPTOR REGISTRATION
    # =========================================================================

    def _register_descriptor_from_declaration(
        self,
        declaration: DescriptorDeclaration,
        plugin_dir: Path,
        plugin_meta: DescriptorPluginMetadata,
    ) -> dict[str, Any]:
        """
        Register descriptor from declaration.

        Args:
            declaration: Descriptor declaration from plugin.yaml
            plugin_dir: Plugin root directory
            plugin_meta: Plugin metadata object

        Returns:
            Registration result dict
        """
        descriptor_name = declaration.name

        try:
            # Load the descriptor function from plugin module
            descriptor_func = self._load_descriptor_function(
                plugin_dir=plugin_dir,
                module_path=declaration.module_path,
                function_name=declaration.function_name,
            )

            # Create metadata
            category = self._parse_category(declaration.category)
            metadata = DescriptorMetadata(
                name=descriptor_name,
                category=category,
                requires_3d=declaration.requires_3d,
                requires_charges=declaration.requires_charges,
                description=declaration.description
                or f"Plugin descriptor from {plugin_meta.plugin_name}",
            )

            # Register with DescriptorRegistry
            descriptor_registry.register_descriptor(
                name=descriptor_name,
                function=descriptor_func,
                metadata=metadata,
                is_builtin=False,
                plugin_name=plugin_meta.plugin_name,
            )

            logger.info(
                f"✓ {descriptor_name}: Registered from plugin "
                f"[Plugin: {plugin_meta.plugin_name}, "
                f"Function: {declaration.function_name}, "
                f"Module: {declaration.module_path}]"
            )

            return {
                "registered": True,
                "descriptor_name": descriptor_name,
                "reason": "Descriptor loaded and registered from plugin",
                "details": {
                    "function": declaration.function_name,
                    "module": declaration.module_path,
                    "file": str(plugin_dir / f"{declaration.module_path}.py"),
                },
            }

        except FileNotFoundError:
            logger.warning(
                f"⚠ {descriptor_name}: Module not found [Plugin: {plugin_meta.plugin_name}]"
            )
            return {
                "registered": False,
                "descriptor_name": descriptor_name,
                "reason": "Module file not found",
            }

        except AttributeError as e:
            logger.warning(
                f"⚠ {descriptor_name}: Function '{declaration.function_name}' not found "
                f"[Plugin: {plugin_meta.plugin_name}]"
            )
            return {
                "registered": False,
                "descriptor_name": descriptor_name,
                "reason": f"Function not found: {str(e)}",
            }

        except Exception as e:
            logger.error(
                f"✗ {descriptor_name}: Registration failed: {e} [Plugin: {plugin_meta.plugin_name}]"
            )
            logger.debug("Registration error details:", exc_info=True)
            return {
                "registered": False,
                "descriptor_name": descriptor_name,
                "reason": f"Registration error: {str(e)}",
            }

    def _load_descriptor_function(
        self, plugin_dir: Path, module_path: str, function_name: str
    ) -> Callable:
        """
        Dynamically load a descriptor function from plugin module.

        Args:
            plugin_dir: Plugin root directory
            module_path: Relative module path (e.g., "descriptors")
            function_name: Name of the function to load

        Returns:
            Descriptor function callable

        Raises:
            FileNotFoundError: Module file not found
            AttributeError: Function not found in module
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

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Add to sys.modules

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            sys.modules.pop(module_name, None)  # Clean up on failure
            raise ImportError(f"Failed to execute module {module_file}: {e}") from e

        # Get function from module
        if not hasattr(module, function_name):
            raise AttributeError(f"Function '{function_name}' not found in {module_file}")

        func = getattr(module, function_name)

        # Verify it's callable
        if not callable(func):
            raise TypeError(f"{function_name} is not callable")

        return func

    def _parse_category(self, category_str: str) -> DescriptorCategory:
        """Parse category string to DescriptorCategory enum"""
        try:
            return DescriptorCategory(category_str.lower())
        except ValueError:
            logger.warning(f"Unknown category '{category_str}', using CONSTITUTIONAL")
            return DescriptorCategory.CONSTITUTIONAL

    # =========================================================================
    # UNDECLARED DESCRIPTOR DISCOVERY
    # =========================================================================

    def _scan_and_register_undeclared_descriptors(
        self, plugin_dir: Path, plugin_meta: DescriptorPluginMetadata
    ) -> int:
        """
        Scan plugin directory for descriptor functions not declared in plugin.yaml.

        These are "bonus" descriptors that were implemented but not documented.

        Args:
            plugin_dir: Plugin root directory
            plugin_meta: Plugin metadata object

        Returns:
            Number of undeclared descriptors discovered and registered
        """
        discovered_count = 0
        declared_names = {decl.name for decl in plugin_meta.descriptor_declarations}

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

                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    logger.debug(f"Failed to execute module {py_file}: {e}")
                    continue

                # Find descriptor functions (by convention: functions that take mol as first arg)
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    # Skip private functions
                    if name.startswith("_"):
                        continue

                    # Check if already declared or registered
                    if name in declared_names:
                        continue
                    if name in plugin_meta.registered_descriptors:
                        continue

                    # Check function signature
                    try:
                        sig = inspect.signature(obj)
                        params = list(sig.parameters.keys())

                        # Must have at least one parameter (mol)
                        if not params:
                            continue

                        # Register as undeclared descriptor
                        metadata = DescriptorMetadata(
                            name=name,
                            category=DescriptorCategory.CONSTITUTIONAL,
                            description=f"Undeclared descriptor from {plugin_meta.plugin_name}",
                        )

                        descriptor_registry.register_descriptor(
                            name=name,
                            function=obj,
                            metadata=metadata,
                            is_builtin=False,
                            plugin_name=plugin_meta.plugin_name,
                        )

                        plugin_meta.registered_descriptors.add(name)
                        discovered_count += 1

                        logger.info(
                            f"+ {name}: Bonus discovery (not declared in plugin.yaml) "
                            f"[Plugin: {plugin_meta.plugin_name}]"
                        )

                    except Exception as e:
                        logger.debug(f"Failed to process function {name} in {py_file}: {e}")

            except Exception as e:
                logger.debug(f"Failed to scan {py_file}: {e}")

        return discovered_count

    # =========================================================================
    # PLUGIN VALIDATION
    # =========================================================================

    def validate_plugin(self, plugin_name: str) -> tuple[bool, list[str]]:
        """
        Validate a plugin.

        Validation checks:
        - All declared descriptors are implemented
        - Descriptor functions are callable
        - No security issues

        Args:
            plugin_name: Name of plugin to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if plugin_name not in self._plugins:
            return False, [f"Plugin '{plugin_name}' not found"]

        plugin_meta = self._plugins[plugin_name]
        errors = []
        warnings = []

        # Check for missing implementations
        missing = plugin_meta.missing_implementations
        if missing:
            errors.append(f"Missing implementations: {', '.join(missing)}")

        # Verify registered descriptors are callable
        for desc_name in plugin_meta.registered_descriptors:
            func = descriptor_registry.get_descriptor(desc_name)
            if not callable(func):
                errors.append(f"Descriptor '{desc_name}' is not callable")

        # Security checks (basic)
        # TODO: Implement more comprehensive security checks

        # Update validation status
        plugin_meta.is_validated = len(errors) == 0
        plugin_meta.validation_date = datetime.now().isoformat()
        plugin_meta.validation_results = {"errors": errors, "warnings": warnings}

        return len(errors) == 0, errors

    # =========================================================================
    # PLUGIN MANAGEMENT
    # =========================================================================

    def enable_plugin(self, plugin_name: str) -> None:
        """Enable a plugin"""
        if plugin_name not in self._plugins:
            raise DescriptorPluginError(f"Plugin '{plugin_name}' not found")

        self._enabled_plugins.add(plugin_name)
        self._disabled_plugins.discard(plugin_name)
        logger.info(f"Enabled plugin: {plugin_name}")

    def disable_plugin(self, plugin_name: str) -> None:
        """Disable a plugin"""
        if plugin_name not in self._plugins:
            raise DescriptorPluginError(f"Plugin '{plugin_name}' not found")

        self._disabled_plugins.add(plugin_name)
        self._enabled_plugins.discard(plugin_name)
        logger.info(f"Disabled plugin: {plugin_name}")

    def is_enabled(self, plugin_name: str) -> bool:
        """Check if plugin is enabled"""
        return plugin_name in self._enabled_plugins

    def list_plugins(self) -> list[str]:
        """List all discovered plugins"""
        return list(self._plugins.keys())

    def get_plugin_info(self, plugin_name: str) -> dict[str, Any] | None:
        """Get plugin information"""
        if plugin_name not in self._plugins:
            return None
        return self._plugins[plugin_name].to_dict()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate checksum of plugin directory"""
        hasher = hashlib.sha256()

        for file_path in sorted(directory.glob("**/*.py")):
            try:
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
            except Exception as e:
                logger.debug(f"Failed to read {file_path} for checksum: {e}")

        return hasher.hexdigest()

    def _log_plugin_discovery_summary(
        self, plugin_meta: DescriptorPluginMetadata, registration_results: list[dict[str, Any]]
    ) -> None:
        """Log comprehensive plugin discovery summary"""
        logger.info("=" * 64)
        logger.info(f"Plugin Discovery: {plugin_meta.plugin_name} v{plugin_meta.version}")
        logger.info("=" * 64)
        logger.info(f"Author: {plugin_meta.author}")
        if plugin_meta.description:
            logger.info(f"Description: {plugin_meta.description}")
        logger.info("")

        logger.info("Descriptor Summary:")
        logger.info(f"  Declared in plugin.yaml: {plugin_meta.declared_count}")
        logger.info(f"  Successfully registered: {plugin_meta.registered_count}")
        logger.info(f"  Missing implementations: {len(plugin_meta.missing_implementations)}")
        logger.info(f"  Bonus discoveries: {len(plugin_meta.undeclared_implementations)}")

        # Overall status
        if plugin_meta.missing_implementations:
            logger.info(
                f"  Status: ⚠ {len(plugin_meta.missing_implementations)} "
                f"declared descriptor(s) not implemented"
            )
        elif plugin_meta.registered_count > 0:
            logger.info("  Status: ✓ All declared descriptors successfully registered")
        else:
            logger.info("  Status: ℹ No descriptors declared or registered")

        # Registration details
        if registration_results:
            logger.info("")
            logger.info("Registration Details:")
            for result in registration_results:
                desc_name = result.get("descriptor_name", "Unknown")
                if result["registered"]:
                    logger.info(f"  ✓ {desc_name}: {result['reason']}")
                else:
                    logger.info(f"  ✗ {desc_name}: {result['reason']}")

        # Bonus discoveries
        if plugin_meta.undeclared_implementations:
            logger.info("")
            logger.info("Bonus Discoveries (not declared in plugin.yaml):")
            for name in plugin_meta.undeclared_implementations:
                logger.info(f"  + {name}")

        logger.info("")
        logger.info(f"Validation: {'PASSED' if plugin_meta.is_validated else 'PENDING'}")
        logger.info("=" * 64)


# =============================================================================
# GLOBAL PLUGIN LOADER INSTANCE
# =============================================================================

# Global plugin loader instance
plugin_loader = DescriptorPluginLoader()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def discover_plugins(paths: list[Path] | None = None, auto_validate: bool = False) -> list[str]:
    """Discover plugins using global loader"""
    return plugin_loader.discover_plugins(paths=paths, auto_validate=auto_validate)


def validate_plugin(plugin_name: str) -> tuple[bool, list[str]]:
    """Validate plugin using global loader"""
    return plugin_loader.validate_plugin(plugin_name)


def list_plugins() -> list[str]:
    """List plugins using global loader"""
    return plugin_loader.list_plugins()


def get_plugin_info(plugin_name: str) -> dict[str, Any] | None:
    """Get plugin info using global loader"""
    return plugin_loader.get_plugin_info(plugin_name)
