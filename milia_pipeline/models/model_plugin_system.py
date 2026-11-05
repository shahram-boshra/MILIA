"""
Model Plugin System for Milia Pipeline

Provides a secure, extensible plugin system for custom models:
- Plugin discovery from YAML metadata (plugin.yaml)
- Model registration into ModelRegistry
- Comprehensive validation (dependencies, security, functional)
- Version management and compatibility checking
- Thread-safe registry with enable/disable functionality

This module follows the established pattern from transformations/plugin_system.py
and integrates with the models module infrastructure.

Pydantic V2 Migration (Phase 38):
    - Migrated ModelDeclaration from @dataclass to Pydantic BaseModel (mutable)
    - Migrated ModelPluginMetadata from @dataclass to Pydantic BaseModel (mutable)
    - Uses Field(default_factory=list) for mutable defaults
    - Added to_dict() methods for backward compatibility
    - Custom to_dict() in ModelPluginMetadata preserved (computes num_models, extracts model names)
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: milia Team
Version: 1.1.0
"""

import importlib
import importlib.machinery
import importlib.util
import keyword
import logging
import re
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import torch.nn as nn
import yaml
from pydantic import BaseModel, Field

# Import model registry and categories
try:
    from milia_pipeline.models.registry.model_registry import ModelRegistry, registry
    from milia_pipeline.models.registry.pyg_introspector import ModelCategory, ModelMetadata

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    ModelRegistry = None
    registry = None
    ModelMetadata = None
    ModelCategory = None

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import (
        ModelError,
        PluginDependencyError,
        PluginError,
        PluginModelError,
        PluginSecurityError,
        PluginValidationError,
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
        """Exception raised for plugin dependency issues."""

        def __init__(
            self,
            message: str,
            plugin_name: str | None = None,
            missing_dependencies: list | None = None,
            **kwargs,
        ):
            super().__init__(message, plugin_name=plugin_name)
            self.missing_dependencies = missing_dependencies or []

    class ModelError(Exception):
        """Exception raised for model-related errors."""

        pass

    class PluginModelError(ModelError, PluginError):
        """Exception raised for plugin model issues."""

        def __init__(
            self,
            message: str,
            plugin_name: str | None = None,
            plugin_path: str | None = None,
            **kwargs,
        ):
            PluginError.__init__(self, message, plugin_name=plugin_name)
            self.plugin_path = plugin_path


logger = logging.getLogger(__name__)

# Modules of directory plugins are imported under this package, one subpackage
# per plugin (``<namespace>.<plugin_name>.<module_path>``), so plugins never
# share module names and ``sys.path`` is left unchanged.
PLUGIN_MODULE_NAMESPACE = "milia_pipeline.models._loaded_plugins"


def _create_plugin_package(package_name: str, location: Path) -> ModuleType:
    """
    Create the package for one plugin directory and register it in ``sys.modules``.

    A directory with ``__init__.py`` becomes a regular package (the file is
    executed); otherwise it becomes a namespace-style package with no code. Any
    modules left from an earlier package of the same name are removed first.
    """
    for name in [n for n in sys.modules if n == package_name or n.startswith(package_name + ".")]:
        del sys.modules[name]

    init_file = location / "__init__.py"
    if init_file.is_file():
        spec = importlib.util.spec_from_file_location(
            package_name, init_file, submodule_search_locations=[str(location)]
        )
    else:
        spec = importlib.machinery.ModuleSpec(package_name, None, is_package=True)
        spec.submodule_search_locations = [str(location)]

    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    try:
        if spec.loader is not None:
            spec.loader.exec_module(package)
    except BaseException:
        del sys.modules[package_name]
        raise

    parent = sys.modules.get(PLUGIN_MODULE_NAMESPACE)
    if parent is not None:
        setattr(parent, package_name.rpartition(".")[2], package)
    return package


@dataclass
class _LoadAttempt:
    """A plugin load in progress: its owner thread, completion event and outcome."""

    owner: int
    done: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None


def _same_plugin_dir(first: Path | None, second: Path | None) -> bool:
    """Whether two plugin directories are the same directory."""
    if first is None or second is None:
        return first is second
    return Path(first).resolve() == Path(second).resolve()


def _import_plugin_module(plugin_name: str, plugin_dir: Path, module_path: str) -> ModuleType:
    """
    Import ``module_path`` from a plugin directory under the plugin's own package.

    The module is imported as ``<PLUGIN_MODULE_NAMESPACE>.<plugin_name>.<module_path>``
    by the standard import system, so dotted module paths, subpackages and
    relative imports between the plugin's modules work, and a module that fails
    to execute is not left in ``sys.modules``. ``sys.path`` is not changed.

    Raises:
        PluginModelError: If ``plugin_name`` cannot be used as a module name
        ImportError: If the module cannot be found or imported
    """
    if not plugin_name.isidentifier() or keyword.iskeyword(plugin_name):
        raise PluginModelError(
            f"Plugin name '{plugin_name}' cannot be used as a Python module name",
            plugin_name=plugin_name,
            plugin_path=str(plugin_dir),
        )

    importlib.import_module(PLUGIN_MODULE_NAMESPACE)
    package_name = f"{PLUGIN_MODULE_NAMESPACE}.{plugin_name}"
    location = Path(plugin_dir).resolve()
    package = sys.modules.get(package_name)
    if package is None or list(getattr(package, "__path__", [])) != [str(location)]:
        _create_plugin_package(package_name, location)

    return importlib.import_module(f"{package_name}.{module_path}")


# =============================================================================
# PLUGIN METADATA
# =============================================================================


class ModelDeclaration(BaseModel):
    """
    Declaration of a model within a plugin.

    This represents a single model that a plugin provides,
    with all necessary metadata for registration.

    Pydantic V2 Migration (Phase 38):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Added to_dict() method for backward compatibility
        - NON-BREAKING: Same constructor API and attribute access preserved

    Attributes:
        name: Model name (unique identifier)
        class_name: Python class name
        module_path: Import path to module containing the class
        category: Model category (e.g., 'gnn', 'custom')
        description: Brief description of the model
        supported_tasks: List of supported task types
        hyperparameters: Dict of hyperparameter schemas
        plugin_name: Name of the plugin providing this model
        requires_edge_index: Whether model requires edge_index
        requires_edge_features: Whether model requires edge features
        requires_edge_weights: Whether model requires edge weights
        supports_batch: Whether model supports batched graphs
        supports_heterogeneous: Whether model supports heterogeneous graphs
        min_pyg_version: Minimum PyG version required
        reference_paper: Reference paper citation
        reference_url: URL to reference paper or documentation
    """

    name: str
    class_name: str
    module_path: str
    category: str
    description: str
    supported_tasks: list[str]
    hyperparameters: dict[str, Any]
    plugin_name: str
    requires_edge_index: bool = True
    requires_edge_features: bool = False
    requires_edge_weights: bool = False
    supports_batch: bool = True
    supports_heterogeneous: bool = False
    min_pyg_version: str | None = None
    reference_paper: str | None = None
    reference_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Backward compatible method wrapping Pydantic V2's model_dump().

        Returns:
            Dictionary with all 16 fields
        """
        return self.model_dump()


class ModelPluginMetadata(BaseModel):
    """
    Complete metadata for a model plugin.

    This contains all information about a plugin, including
    the models it provides and validation requirements.

    Pydantic V2 Migration (Phase 38):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=list) for mutable defaults
        - Custom to_dict() preserved (computes num_models, extracts model names)
        - NON-BREAKING: Same constructor API and attribute access preserved

    Attributes:
        plugin_name: Unique plugin identifier
        version: Semantic version string
        author: Author name
        description: Plugin description
        plugin_type: Type of plugin (e.g., 'user_experimental')
        milia_version: Required milia version
        pyg_version: Required PyG version
        python_version: Required Python version
        license: License type
        model_declarations: List of model declarations
        dependencies: List of required dependencies
        optional_dependencies: List of optional dependencies
        homepage: Optional homepage URL
        repository: Optional repository URL
        documentation: Optional documentation URL
        plugin_path: Path to plugin directory
        loaded: Whether plugin is loaded
        enabled: Whether plugin is enabled
        load_time: When plugin was loaded
        validation_errors: List of validation errors
    """

    # Required fields
    plugin_name: str
    version: str
    author: str
    description: str
    plugin_type: str
    milia_version: str
    pyg_version: str
    python_version: str
    license: str
    model_declarations: list[ModelDeclaration]

    # Optional fields with mutable defaults
    dependencies: list[str] = Field(default_factory=list)
    optional_dependencies: list[str] = Field(default_factory=list)
    homepage: str | None = None
    repository: str | None = None
    documentation: str | None = None
    plugin_path: Path | None = None

    # Internal tracking
    loaded: bool = False
    enabled: bool = True
    load_time: datetime | None = None
    validation_errors: list[str] = Field(default_factory=list)

    # Pydantic V2 config for Path type
    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Custom implementation preserved from original - computes num_models
        and extracts model names for convenience.

        Returns:
            Dictionary with plugin metadata and computed fields
        """
        return {
            "plugin_name": self.plugin_name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "plugin_type": self.plugin_type,
            "milia_version": self.milia_version,
            "pyg_version": self.pyg_version,
            "python_version": self.python_version,
            "license": self.license,
            "num_models": len(self.model_declarations),
            "models": [decl.name for decl in self.model_declarations],
            "dependencies": self.dependencies,
            "loaded": self.loaded,
            "enabled": self.enabled,
        }


# =============================================================================
# PLUGIN LOADER
# =============================================================================


class ModelPluginLoader:
    """
    Loader for custom model plugins.

    This class handles discovery, loading, and registration of custom
    models from plugin directories. It follows the established pattern
    from the transformations plugin system.

    Features:
    - Auto-discovery from plugin.yaml files
    - Comprehensive validation (security, dependencies, functionality)
    - Thread-safe plugin management
    - Integration with ModelRegistry
    - Enable/disable functionality

    Usage:
        >>> loader = ModelPluginLoader()
        >>> plugins = loader.discover_plugins([Path("./plugins/models")])
        >>> print(f"Discovered {len(plugins)} plugins")
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize plugin loader."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._plugins: dict[str, ModelPluginMetadata] = {}
        self._loaded_modules: dict[str, Any] = {}
        # Guards loader state only; plugin code never runs while it is held.
        self._lock = threading.Lock()
        # Loads in progress, by plugin name; other callers wait on their event.
        self._loading: dict[str, _LoadAttempt] = {}

        logger.debug("ModelPluginLoader initialized")

    # =========================================================================
    # PLUGIN DISCOVERY
    # =========================================================================

    def discover_plugins(
        self, paths: list[Path], auto_validate: bool = True, validation_level: str = "standard"
    ) -> list[str]:
        """
        Discover plugins from specified paths.

        Searches for plugin.yaml files in subdirectories and loads
        the metadata. Optionally validates plugins before registration.

        Args:
            paths: List of paths to search for plugins
            auto_validate: Whether to validate plugins automatically
            validation_level: Validation strictness (permissive, standard, strict)

        Returns:
            List of discovered plugin names

        Example:
            >>> loader = ModelPluginLoader()
            >>> plugin_paths = [Path("./plugins/models")]
            >>> discovered = loader.discover_plugins(plugin_paths)
            >>> print(f"Found {len(discovered)} plugins")
        """
        discovered = []

        for search_path in paths:
            if not search_path.exists():
                logger.warning(f"Plugin search path does not exist: {search_path}")
                continue

            # Find all plugin.yaml files
            for plugin_yaml in search_path.glob("*/plugin.yaml"):
                try:
                    logger.debug(f"Discovering plugin from: {plugin_yaml}")

                    # Load plugin metadata
                    metadata = self._load_plugin_metadata(plugin_yaml)

                    # Validate if requested
                    if auto_validate:
                        validation_result = self._validate_plugin(metadata, level=validation_level)

                        if not validation_result["valid"]:
                            logger.warning(
                                f"Plugin '{metadata.plugin_name}' failed validation: "
                                f"{validation_result['errors']}"
                            )
                            if validation_level == "strict":
                                continue

                    # Register plugin. A name is unique across all sources: the first
                    # plugin found keeps it, and discovering the same directory again
                    # keeps its current state.
                    with self._lock:
                        existing = self._plugins.get(metadata.plugin_name)
                        if existing is None:
                            self._plugins[metadata.plugin_name] = metadata
                        elif not _same_plugin_dir(existing.plugin_path, metadata.plugin_path):
                            raise PluginModelError(
                                f"Duplicate plugin name '{metadata.plugin_name}': already "
                                f"discovered at {existing.plugin_path}, ignoring "
                                f"{metadata.plugin_path}",
                                plugin_name=metadata.plugin_name,
                                plugin_path=str(metadata.plugin_path),
                            )
                        else:
                            logger.debug(f"Plugin '{metadata.plugin_name}' already discovered")
                            discovered.append(metadata.plugin_name)
                            continue

                    discovered.append(metadata.plugin_name)
                    logger.info(
                        f"Discovered plugin '{metadata.plugin_name}' "
                        f"with {len(metadata.model_declarations)} model(s)"
                    )

                except Exception as e:
                    logger.error(f"Failed to discover plugin from {plugin_yaml}: {e}")
                    if validation_level == "strict":
                        raise

        logger.info(f"Plugin discovery complete: {len(discovered)} plugins found")
        return discovered

    def _load_plugin_metadata(self, plugin_yaml: Path) -> ModelPluginMetadata:
        """
        Load plugin metadata from plugin.yaml.

        Args:
            plugin_yaml: Path to plugin.yaml file

        Returns:
            ModelPluginMetadata instance

        Raises:
            PluginError: If metadata cannot be loaded
        """
        try:
            with open(plugin_yaml) as f:
                data = yaml.safe_load(f)

            if not data:
                raise PluginError(f"Empty plugin.yaml: {plugin_yaml}")

            # Extract basic metadata
            plugin_name = data.get("plugin_name")
            if not plugin_name:
                raise PluginError(f"Missing plugin_name in {plugin_yaml}")

            # Parse model declarations
            models_data = data.get("models", [])
            if not models_data:
                logger.warning(f"Plugin '{plugin_name}' declares no models")

            model_declarations = []
            for model_data in models_data:
                decl = ModelDeclaration(
                    name=model_data["name"],
                    class_name=model_data["class_name"],
                    module_path=model_data["module_path"],
                    category=model_data.get("category", "custom"),
                    description=model_data.get("description", ""),
                    supported_tasks=model_data.get("supported_tasks", []),
                    hyperparameters=model_data.get("hyperparameters", {}),
                    plugin_name=plugin_name,
                    requires_edge_index=model_data.get("requires_edge_index", True),
                    requires_edge_features=model_data.get("requires_edge_features", False),
                    requires_edge_weights=model_data.get("requires_edge_weights", False),
                    supports_batch=model_data.get("supports_batch", True),
                    supports_heterogeneous=model_data.get("supports_heterogeneous", False),
                    min_pyg_version=model_data.get("min_pyg_version"),
                    reference_paper=model_data.get("reference_paper"),
                    reference_url=model_data.get("reference_url"),
                )
                model_declarations.append(decl)

            # Create metadata
            metadata = ModelPluginMetadata(
                plugin_name=plugin_name,
                version=data.get("version", "0.0.0"),
                author=data.get("author", "Unknown"),
                description=data.get("description", ""),
                plugin_type=data.get("plugin_type", "user_experimental"),
                milia_version=data.get("milia_version", ">=4.0.0"),
                pyg_version=data.get("pyg_version", ">=2.0.0"),
                python_version=data.get("python_version", ">=3.8"),
                license=data.get("license", "Unknown"),
                model_declarations=model_declarations,
                dependencies=data.get("dependencies", []),
                optional_dependencies=data.get("optional_dependencies", []),
                homepage=data.get("homepage"),
                repository=data.get("repository"),
                documentation=data.get("documentation"),
                plugin_path=plugin_yaml.parent,
            )

            return metadata

        except yaml.YAMLError as e:
            raise PluginError(f"Invalid YAML in {plugin_yaml}: {e}") from e
        except KeyError as e:
            raise PluginError(f"Missing required field in {plugin_yaml}: {e}") from e
        except Exception as e:
            raise PluginError(f"Failed to load plugin metadata from {plugin_yaml}: {e}") from e

    # =========================================================================
    # PLUGIN VALIDATION
    # =========================================================================

    def _validate_plugin(
        self, metadata: ModelPluginMetadata, level: str = "standard"
    ) -> dict[str, Any]:
        """
        Validate plugin metadata and requirements.

        Args:
            metadata: Plugin metadata to validate
            level: Validation level (permissive, standard, strict)

        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []

        # Validate basic metadata
        if not metadata.plugin_name:
            errors.append("Missing plugin_name")

        if not metadata.version:
            warnings.append("Missing version")

        if not metadata.model_declarations:
            warnings.append("No models declared")

        # Validate dependencies
        dep_result = self._validate_dependencies(metadata)
        if not dep_result["satisfied"]:
            if level == "strict":
                errors.extend(dep_result["missing"])
            else:
                warnings.extend(dep_result["missing"])

        # Validate model declarations
        for decl in metadata.model_declarations:
            decl_errors = self._validate_model_declaration(decl)
            if decl_errors:
                if level == "strict":
                    errors.extend(decl_errors)
                else:
                    warnings.extend(decl_errors)

        # Security checks
        if level in ["standard", "strict"]:
            security_result = self._security_check(metadata)
            if security_result["issues"]:
                if level == "strict":
                    errors.extend(security_result["issues"])
                else:
                    warnings.extend(security_result["issues"])

        # Store validation errors in metadata
        metadata.validation_errors = errors + warnings

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_dependencies(self, metadata: ModelPluginMetadata) -> dict[str, Any]:
        """Validate plugin dependencies."""
        missing = []
        satisfied = True

        for dep in metadata.dependencies:
            try:
                # Try to import the dependency
                importlib.import_module(dep.split("[")[0])  # Handle extras like torch[cuda]
            except ImportError:
                missing.append(f"Missing required dependency: {dep}")
                satisfied = False

        return {"satisfied": satisfied, "missing": missing}

    def _validate_model_declaration(self, declaration: ModelDeclaration) -> list[str]:
        """Validate a single model declaration."""
        errors = []

        if not declaration.name:
            errors.append("Model declaration missing name")

        if not declaration.class_name:
            errors.append(f"Model '{declaration.name}' missing class_name")

        if not declaration.module_path:
            errors.append(f"Model '{declaration.name}' missing module_path")

        if not declaration.supported_tasks:
            errors.append(f"Model '{declaration.name}' declares no supported tasks")

        # Validate hyperparameter schema
        for param_name, param_schema in declaration.hyperparameters.items():
            if not isinstance(param_schema, dict):
                errors.append(
                    f"Model '{declaration.name}' hyperparameter '{param_name}' "
                    f"must be a dict with schema"
                )

        return errors

    def _security_check(self, metadata: ModelPluginMetadata) -> dict[str, Any]:
        """Perform security checks on plugin."""
        issues = []

        # Check for suspicious imports
        if metadata.plugin_path:
            suspicious_patterns = [
                r"import\s+os\.system",
                r"import\s+subprocess",
                r"eval\(",
                r"exec\(",
                r"__import__\(",
            ]

            for py_file in metadata.plugin_path.glob("**/*.py"):
                try:
                    with open(py_file) as f:
                        content = f.read()

                    for pattern in suspicious_patterns:
                        if re.search(pattern, content):
                            issues.append(f"Suspicious code pattern in {py_file.name}: {pattern}")
                except Exception as e:
                    logger.debug(f"Could not scan {py_file}: {e}")

        return {"issues": issues}

    # =========================================================================
    # PLUGIN LOADING AND REGISTRATION
    # =========================================================================

    def load_plugin(self, plugin_name: str, register_models: bool = True) -> bool:
        """
        Load a plugin and optionally register its models.

        Args:
            plugin_name: Name of plugin to load
            register_models: Whether to register models with ModelRegistry

        Returns:
            True if successful

        Raises:
            PluginError: If plugin cannot be loaded
        """
        # 1. Under the lock: check state and claim the load, or join a load in progress.
        with self._lock:
            if plugin_name not in self._plugins:
                raise PluginError(f"Plugin '{plugin_name}' not discovered")

            metadata = self._plugins[plugin_name]

            if metadata.loaded:
                logger.debug(f"Plugin '{plugin_name}' already loaded")
                return True

            if not metadata.enabled:
                raise PluginError(f"Plugin '{plugin_name}' is disabled")

            attempt = self._loading.get(plugin_name)
            if attempt is None:
                attempt = _LoadAttempt(owner=threading.get_ident())
                self._loading[plugin_name] = attempt
                declarations = list(metadata.model_declarations)
                plugin_path = metadata.plugin_path
                owner = True
            elif attempt.owner == threading.get_ident():
                raise PluginError(
                    f"Plugin '{plugin_name}' is already being loaded by this thread "
                    f"(the plugin's code loads it again)",
                    plugin_name=plugin_name,
                )
            else:
                owner = False

        if not owner:
            attempt.done.wait()
            if attempt.error is not None:
                raise PluginError(
                    f"Failed to load plugin '{plugin_name}': {attempt.error}",
                    plugin_name=plugin_name,
                ) from attempt.error
            return True

        # 2. Without the lock: run plugin code (module import, model registration).
        error: Exception | None = None
        try:
            logger.info(f"Loading plugin '{plugin_name}'...")
            if register_models:
                for decl in declarations:
                    self._register_plugin_model(decl, plugin_path)
        except Exception as e:
            error = e

        # 3. Under the lock: commit the result and release waiting callers.
        with self._lock:
            if error is None:
                metadata.loaded = True
                metadata.load_time = datetime.now()
            del self._loading[plugin_name]
        attempt.error = error
        attempt.done.set()

        if error is not None:
            logger.error(f"Failed to load plugin '{plugin_name}': {error}")
            raise PluginError(
                f"Failed to load plugin '{plugin_name}': {error}", plugin_name=plugin_name
            ) from error

        logger.info(f"Successfully loaded plugin '{plugin_name}' with {len(declarations)} model(s)")
        return True

    def _register_plugin_model(self, declaration: ModelDeclaration, plugin_dir: Path):
        """
        Register a model from plugin with ModelRegistry.

        Args:
            declaration: Model declaration from plugin
            plugin_dir: Plugin directory path. Its modules are imported under the
                plugin's own package; without a directory, ``module_path`` is an
                installed module's absolute name.

        Raises:
            PluginError: If model cannot be registered
        """
        if not MODELS_AVAILABLE or registry is None:
            raise PluginError("ModelRegistry not available - cannot register plugin models")

        try:
            # Import model class
            if plugin_dir is not None:
                module = _import_plugin_module(
                    declaration.plugin_name, plugin_dir, declaration.module_path
                )
            else:
                module = importlib.import_module(declaration.module_path)
            model_class = getattr(module, declaration.class_name)

            # Verify it's a proper PyTorch module
            if not issubclass(model_class, nn.Module):
                raise PluginError(
                    f"Model class '{declaration.class_name}' must inherit from nn.Module"
                )

            # Create ModelMetadata
            if ModelMetadata is None:
                raise PluginError("ModelMetadata not available")

            metadata = ModelMetadata(
                name=declaration.name,
                category=declaration.category,
                import_path=f"{declaration.module_path}.{declaration.class_name}",
                description=declaration.description,
                supported_tasks=declaration.supported_tasks,
                hyperparameters=declaration.hyperparameters,
                requires_edge_index=declaration.requires_edge_index,
                requires_edge_features=declaration.requires_edge_features,
                requires_edge_weights=declaration.requires_edge_weights,
                supports_batch=declaration.supports_batch,
                supports_heterogeneous=declaration.supports_heterogeneous,
                min_pyg_version=declaration.min_pyg_version,
                reference_paper=declaration.reference_paper,
                reference_url=declaration.reference_url,
                is_plugin=True,
                plugin_name=declaration.plugin_name,
            )

            # Register with registry
            registry.register_model(
                name=declaration.name,
                model_class=model_class,
                metadata=metadata,
                plugin_name=declaration.plugin_name,
            )

            logger.info(
                f"Registered plugin model '{declaration.name}' "
                f"from plugin '{declaration.plugin_name}'"
            )

        except ImportError as e:
            raise PluginError(
                f"Failed to import model '{declaration.name}': {e}",
                plugin_name=declaration.plugin_name,
            ) from e
        except AttributeError as e:
            raise PluginError(
                f"Model class '{declaration.class_name}' not found in module: {e}",
                plugin_name=declaration.plugin_name,
            ) from e
        except Exception as e:
            raise PluginError(
                f"Failed to register model '{declaration.name}': {e}",
                plugin_name=declaration.plugin_name,
            ) from e

    def load_all_plugins(self, register_models: bool = True) -> dict[str, bool]:
        """
        Load all discovered plugins.

        Args:
            register_models: Whether to register models with ModelRegistry

        Returns:
            Dictionary mapping plugin names to success status
        """
        results = {}

        for plugin_name in list(self._plugins.keys()):
            try:
                success = self.load_plugin(plugin_name, register_models)
                results[plugin_name] = success
            except Exception as e:
                logger.error(f"Failed to load plugin '{plugin_name}': {e}")
                results[plugin_name] = False

        return results

    # =========================================================================
    # PLUGIN MANAGEMENT
    # =========================================================================

    def enable_plugin(self, plugin_name: str):
        """Enable a plugin."""
        with self._lock:
            if plugin_name not in self._plugins:
                raise PluginError(f"Plugin '{plugin_name}' not found")
            self._plugins[plugin_name].enabled = True
            logger.info(f"Enabled plugin '{plugin_name}'")

    def disable_plugin(self, plugin_name: str):
        """Disable a plugin."""
        with self._lock:
            if plugin_name not in self._plugins:
                raise PluginError(f"Plugin '{plugin_name}' not found")
            self._plugins[plugin_name].enabled = False
            logger.info(f"Disabled plugin '{plugin_name}'")

    def unload_plugin(self, plugin_name: str):
        """
        Unload a plugin.

        Note: This does not unregister models from ModelRegistry.
        """
        with self._lock:
            if plugin_name not in self._plugins:
                raise PluginError(f"Plugin '{plugin_name}' not found")

            metadata = self._plugins[plugin_name]
            metadata.loaded = False
            metadata.load_time = None

            logger.info(f"Unloaded plugin '{plugin_name}'")

    def list_plugins(self, loaded_only: bool = False, enabled_only: bool = False) -> list[str]:
        """
        List available plugins.

        Args:
            loaded_only: Only return loaded plugins
            enabled_only: Only return enabled plugins

        Returns:
            List of plugin names
        """
        with self._lock:
            plugins = list(self._plugins.keys())

            if loaded_only:
                plugins = [name for name in plugins if self._plugins[name].loaded]

            if enabled_only:
                plugins = [name for name in plugins if self._plugins[name].enabled]

            return plugins

    def get_plugin_info(self, plugin_name: str) -> dict[str, Any]:
        """
        Get information about a plugin.

        Args:
            plugin_name: Name of plugin

        Returns:
            Dictionary with plugin information
        """
        with self._lock:
            if plugin_name not in self._plugins:
                raise PluginError(f"Plugin '{plugin_name}' not found")

            return self._plugins[plugin_name].to_dict()

    def get_plugin_models(self, plugin_name: str) -> list[str]:
        """
        Get list of models provided by a plugin.

        Args:
            plugin_name: Name of plugin

        Returns:
            List of model names
        """
        with self._lock:
            if plugin_name not in self._plugins:
                raise PluginError(f"Plugin '{plugin_name}' not found")

            metadata = self._plugins[plugin_name]
            return [decl.name for decl in metadata.model_declarations]


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

# Singleton instance
_plugin_loader = None


def get_plugin_loader() -> ModelPluginLoader:
    """
    Get the singleton plugin loader instance.

    Returns:
        ModelPluginLoader instance
    """
    global _plugin_loader
    if _plugin_loader is None:
        _plugin_loader = ModelPluginLoader()
    return _plugin_loader


def discover_plugins(
    paths: list[Path] | None = None, auto_validate: bool = True, validation_level: str = "standard"
) -> list[str]:
    """
    Discover model plugins from paths.

    Args:
        paths: List of paths to search (default: from config)
        auto_validate: Whether to validate automatically
        validation_level: Validation strictness

    Returns:
        List of discovered plugin names

    Example:
        >>> from milia_pipeline.models import discover_plugins
        >>> from pathlib import Path
        >>> plugins = discover_plugins([Path("./plugins/models")])
        >>> print(f"Discovered {len(plugins)} plugins")
    """
    loader = get_plugin_loader()

    if paths is None:
        # Try to get paths from config
        try:
            from milia_pipeline.models.utils.config_bridge import get_plugins_config

            config = get_plugins_config()
            paths = [Path(p) for p in config.plugin_paths]
        except Exception as e:
            logger.warning(f"Could not load plugin paths from config: {e}")
            paths = [Path("./plugins/models")]

    return loader.discover_plugins(paths, auto_validate, validation_level)


def load_plugin(plugin_name: str, register_models: bool = True) -> bool:
    """
    Load a specific plugin.

    Args:
        plugin_name: Name of plugin to load
        register_models: Whether to register models

    Returns:
        True if successful
    """
    loader = get_plugin_loader()
    return loader.load_plugin(plugin_name, register_models)


def list_plugins(loaded_only: bool = False, enabled_only: bool = False) -> list[str]:
    """
    List available plugins.

    Args:
        loaded_only: Only return loaded plugins
        enabled_only: Only return enabled plugins

    Returns:
        List of plugin names
    """
    loader = get_plugin_loader()
    return loader.list_plugins(loaded_only, enabled_only)


def get_plugin_info(plugin_name: str) -> dict[str, Any]:
    """
    Get information about a plugin.

    Args:
        plugin_name: Name of plugin

    Returns:
        Dictionary with plugin information
    """
    loader = get_plugin_loader()
    return loader.get_plugin_info(plugin_name)


def validate_plugin(plugin_name: str, level: str = "standard") -> dict[str, Any]:
    """
    Validate a plugin.

    Args:
        plugin_name: Name of plugin
        level: Validation level

    Returns:
        Validation results
    """
    loader = get_plugin_loader()
    if plugin_name not in loader._plugins:
        raise PluginError(f"Plugin '{plugin_name}' not found")

    metadata = loader._plugins[plugin_name]
    return loader._validate_plugin(metadata, level)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================


def get_plugin_loader_instance() -> ModelPluginLoader:
    """
    Get the singleton plugin loader instance.

    This is an alias for get_plugin_loader() provided for clarity
    in contexts where the singleton pattern needs to be explicit.

    Returns:
        ModelPluginLoader: The singleton plugin loader instance

    Example:
        >>> from milia_pipeline.models.model_plugin_system import get_plugin_loader_instance
        >>> loader = get_plugin_loader_instance()
        >>> plugins = loader.list_plugins()
    """
    return get_plugin_loader()


def discover_and_load_plugins(
    paths: list[Path] | None = None,
    auto_validate: bool = True,
    validation_level: str = "standard",
    register_models: bool = True,
) -> dict[str, bool]:
    """
    Discover and load plugins in one operation.

    This is a convenience function that combines discovery and loading,
    useful for initialization scenarios where you want to discover and
    immediately load all available plugins.

    Args:
        paths: List of paths to search for plugins (default: from config)
        auto_validate: Whether to validate plugins automatically during discovery
        validation_level: Validation strictness ('permissive', 'standard', 'strict')
        register_models: Whether to register models in ModelRegistry

    Returns:
        Dictionary mapping plugin names to load success status

    Example:
        >>> from milia_pipeline.models.model_plugin_system import discover_and_load_plugins
        >>> from pathlib import Path
        >>> results = discover_and_load_plugins(
        ...     paths=[Path("./plugins/models")],
        ...     validation_level="strict"
        ... )
        >>> print(f"Loaded {sum(results.values())} of {len(results)} plugins")
    """
    loader = get_plugin_loader()

    # Discover plugins
    plugin_names = loader.discover_plugins(
        paths=paths, auto_validate=auto_validate, validation_level=validation_level
    )

    # Load all discovered plugins
    results = {}
    for plugin_name in plugin_names:
        try:
            success = loader.load_plugin(plugin_name, register_models=register_models)
            results[plugin_name] = success
        except Exception:
            results[plugin_name] = False

    return results


def get_all_plugin_models() -> dict[str, list[str]]:
    """
    Get all models provided by all plugins.

    Returns a mapping of plugin names to the list of model names
    they provide. Useful for getting a complete inventory of
    available plugin models.

    Returns:
        Dictionary mapping plugin names to lists of model names

    Example:
        >>> from milia_pipeline.models.model_plugin_system import get_all_plugin_models
        >>> models = get_all_plugin_models()
        >>> for plugin, model_list in models.items():
        ...     print(f"{plugin}: {', '.join(model_list)}")
    """
    loader = get_plugin_loader()
    all_plugins = loader.list_plugins()

    result = {}
    for plugin_name in all_plugins:
        try:
            models = loader.get_plugin_models(plugin_name)
            result[plugin_name] = models
        except Exception:
            result[plugin_name] = []

    return result


def is_plugin_loaded(plugin_name: str) -> bool:
    """
    Check if a plugin is loaded.

    Args:
        plugin_name: Name of the plugin to check

    Returns:
        True if plugin is loaded, False otherwise

    Example:
        >>> from milia_pipeline.models.model_plugin_system import is_plugin_loaded
        >>> if is_plugin_loaded("my_custom_model"):
        ...     print("Plugin is ready to use")
    """
    loader = get_plugin_loader()
    loaded_plugins = loader.list_plugins(loaded_only=True)
    return plugin_name in loaded_plugins


def is_plugin_enabled(plugin_name: str) -> bool:
    """
    Check if a plugin is enabled.

    Args:
        plugin_name: Name of the plugin to check

    Returns:
        True if plugin is enabled, False otherwise

    Example:
        >>> from milia_pipeline.models.model_plugin_system import is_plugin_enabled
        >>> if is_plugin_enabled("my_custom_model"):
        ...     print("Plugin is active")
    """
    loader = get_plugin_loader()
    enabled_plugins = loader.list_plugins(enabled_only=True)
    return plugin_name in enabled_plugins


def get_plugin_summary() -> dict[str, Any]:
    """
    Get a summary of all plugins and their status.

    Returns:
        Dictionary with comprehensive plugin information including:
        - Total number of plugins
        - Number of loaded plugins
        - Number of enabled plugins
        - List of plugin names by status

    Example:
        >>> from milia_pipeline.models.model_plugin_system import get_plugin_summary
        >>> summary = get_plugin_summary()
        >>> print(f"Total plugins: {summary['total']}")
        >>> print(f"Loaded: {summary['loaded_count']}")
    """
    loader = get_plugin_loader()

    all_plugins = loader.list_plugins()
    loaded_plugins = loader.list_plugins(loaded_only=True)
    enabled_plugins = loader.list_plugins(enabled_only=True)

    return {
        "total": len(all_plugins),
        "loaded_count": len(loaded_plugins),
        "enabled_count": len(enabled_plugins),
        "all_plugins": all_plugins,
        "loaded_plugins": loaded_plugins,
        "enabled_plugins": enabled_plugins,
        "disabled_plugins": [p for p in all_plugins if p not in enabled_plugins],
        "discovered_but_not_loaded": [p for p in all_plugins if p not in loaded_plugins],
    }


# =============================================================================
# ENHANCED EXCEPTION HANDLING
# =============================================================================


def safe_load_plugin(plugin_name: str, register_models: bool = True) -> tuple[bool, str | None]:
    """
    Safely load a plugin with enhanced error handling.

    This function wraps the load_plugin function with additional
    error handling and returns both success status and error message
    if loading fails.

    Args:
        plugin_name: Name of plugin to load
        register_models: Whether to register models in ModelRegistry

    Returns:
        Tuple of (success: bool, error_message: Optional[str])

    Example:
        >>> from milia_pipeline.models.model_plugin_system import safe_load_plugin
        >>> success, error = safe_load_plugin("my_custom_model")
        >>> if not success:
        ...     print(f"Failed to load: {error}")
    """
    try:
        success = load_plugin(plugin_name, register_models=register_models)
        return (success, None)
    except PluginValidationError as e:
        return (False, f"Validation failed: {str(e)}")
    except PluginSecurityError as e:
        return (False, f"Security issue: {str(e)}")
    except PluginDependencyError as e:
        return (False, f"Missing dependencies: {str(e)}")
    except PluginError as e:
        return (False, f"Plugin error: {str(e)}")
    except Exception as e:
        return (False, f"Unexpected error: {str(e)}")


def safe_discover_plugins(
    paths: list[Path] | None = None, auto_validate: bool = True, validation_level: str = "standard"
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Safely discover plugins with enhanced error handling.

    Returns both successfully discovered plugins and any errors
    encountered during discovery.

    Args:
        paths: List of paths to search for plugins
        auto_validate: Whether to validate plugins automatically
        validation_level: Validation strictness

    Returns:
        Tuple of (discovered_plugins: List[str], errors: List[Tuple[path, error]])

    Example:
        >>> from milia_pipeline.models.model_plugin_system import safe_discover_plugins
        >>> from pathlib import Path
        >>> plugins, errors = safe_discover_plugins([Path("./plugins")])
        >>> if errors:
        ...     print(f"Encountered {len(errors)} errors during discovery")
    """
    errors = []
    discovered = []

    try:
        discovered = discover_plugins(
            paths=paths, auto_validate=auto_validate, validation_level=validation_level
        )
    except Exception as e:
        errors.append(("discovery", str(e)))

    return (discovered, errors)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Classes
    "ModelPluginLoader",
    "ModelPluginMetadata",
    "ModelDeclaration",
    # Functions
    "get_plugin_loader",
    "discover_plugins",
    "load_plugin",
    "list_plugins",
    "get_plugin_info",
    "validate_plugin",
    # Exceptions (re-exported for convenience)
    "PluginError",
    "PluginValidationError",
    "PluginSecurityError",
    "PluginDependencyError",
    # Convenience functions
    "get_plugin_loader_instance",
    "discover_and_load_plugins",
    "get_all_plugin_models",
    "is_plugin_loaded",
    "is_plugin_enabled",
    "get_plugin_summary",
    "safe_load_plugin",
    "safe_discover_plugins",
]
