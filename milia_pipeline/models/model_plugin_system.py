"""
Model Plugin System for Milia Pipeline

Provides an extensible plugin system for custom models:
- Plugin discovery from YAML metadata (plugin.yaml), validated by the versioned
  manifest schema (``_model_plugin_manifest.parse_manifest``)
- Model registration into ModelRegistry
- Validation of manifests, versions and dependencies

Trust model: loading a plugin imports its modules, which runs their code with
the privileges of the MILIA process. Plugins are therefore trusted code: a
directory plugin is code you chose to put on a plugin path, and an entry-point
plugin is an installed distribution. Nothing here sandboxes or vets plugin code;
the code scan in ``ModelPluginLoader._security_check`` is a lint hint, not a
security control.
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
import importlib.metadata
import importlib.resources
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
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

# Import model registry and categories
try:
    from milia_pipeline.models.registry.model_registry import ModelRegistry, registry
    from milia_pipeline.models.registry.pyg_introspector import (
        ModelCategory,
        ModelMetadata,
        build_metadata_for_class,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    ModelRegistry = None
    registry = None
    ModelMetadata = None
    ModelCategory = None
    build_metadata_for_class = None

from milia_pipeline.exceptions import (
    PluginDependencyError,
    PluginError,
    PluginModelError,
    PluginSecurityError,
    PluginValidationError,
)
from milia_pipeline.models._model_plugin_manifest import parse_manifest

logger = logging.getLogger(__name__)

# Modules of directory plugins are imported under this package, one subpackage
# per plugin (``<namespace>.<plugin_name>.<module_path>``), so plugins never
# share module names and ``sys.path`` is left unchanged.
PLUGIN_MODULE_NAMESPACE = "milia_pipeline.models._loaded_plugins"

# Entry-point group through which installed distributions provide model plugins.
# Each entry point names a package holding ``plugin.yaml``; its name is the
# ``plugin_name``.
ENTRY_POINT_GROUP = "milia.models"


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


def _check_requirement(requirement: str, environment: dict[str, str] | None = None) -> str | None:
    """
    Check one PEP 508 requirement against the installed distributions.

    Args:
        requirement: The PEP 508 requirement string.
        environment: Marker variables overriding the current environment (for
            example ``{"extra": "descriptors-cdk"}``).

    Returns:
        ``None`` if it is satisfied or its marker excludes this environment;
        otherwise a message saying it is invalid, missing or unsatisfied.
    """
    try:
        req = Requirement(requirement)
    except InvalidRequirement as e:
        reason = str(e).splitlines()[0] if str(e) else type(e).__name__
        return f"Invalid dependency specifier: {requirement!r} ({reason})"

    if req.marker is not None and not req.marker.evaluate(environment):
        return None

    try:
        installed = importlib.metadata.version(req.name)
    except importlib.metadata.PackageNotFoundError:
        return f"Missing required dependency: {requirement}"

    if not req.specifier.contains(installed, prereleases=True):
        return f"Unsatisfied dependency: {requirement} (installed: {req.name} {installed})"
    return None


# The distribution whose optional extras a plugin's ``requires_extra`` names.
MILIA_DISTRIBUTION = "milia-py"


def _installed_versions() -> dict[str, str | None]:
    """
    The versions the ``requirements`` gates are checked against (``None`` = not installed).

    MILIA's is its ``__version__`` (the single source of the package version);
    Python's is ``major.minor.micro``, as pip checks ``Requires-Python``;
    PyG's is read from its installed distribution metadata, without importing it.
    """
    import milia_pipeline

    try:
        pyg = importlib.metadata.version("torch_geometric")
    except importlib.metadata.PackageNotFoundError:
        pyg = None
    return {
        "milia": getattr(milia_pipeline, "__version__", None),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_geometric": pyg,
    }


def _version_gate(label: str, specifier: str, installed: str | None) -> str | None:
    """
    Check one ``requirements`` entry (a PEP 440 specifier) against an installed version.

    Returns:
        ``None`` if it is satisfied or empty (no constraint); otherwise the reason.
    """
    if not specifier:
        return None
    if installed is None:
        return f"requires {label} {specifier}, which is not installed"
    try:
        satisfied = SpecifierSet(specifier).contains(Version(installed), prereleases=True)
    except (InvalidSpecifier, InvalidVersion) as e:
        return f"cannot check {label} {specifier!r} against {installed} ({e})"
    if not satisfied:
        return f"requires {label} {specifier} (installed: {installed})"
    return None


def _extra_problems(extra: str) -> list[str]:
    """
    Check that MILIA's optional extra ``extra`` is installed.

    The extra's requirements are read from MILIA's distribution metadata: the
    entries whose marker holds for ``extra == <extra>`` but not without it. Each
    is checked like a ``dependencies`` entry (nothing is imported). Extra names
    compare in PEP 685 normalised form.

    Returns:
        One message per problem; empty if the extra is satisfied.
    """
    try:
        distribution = importlib.metadata.distribution(MILIA_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return [f"cannot check extra '{extra}': {MILIA_DISTRIBUTION} is not installed"]

    declared = distribution.metadata.get_all("Provides-Extra") or []
    provided = {canonicalize_name(name) for name in declared}
    if canonicalize_name(extra) not in provided:
        return [f"{MILIA_DISTRIBUTION} has no extra '{extra}'"]

    problems = []
    for requirement in distribution.requires or []:
        try:
            marker = Requirement(requirement).marker
        except InvalidRequirement:
            continue  # not ours to report: the installer accepted MILIA's metadata
        if marker is None or marker.evaluate({"extra": ""}):
            continue  # a base requirement, not part of the extra
        problem = _check_requirement(requirement, {"extra": extra})
        if problem:
            problems.append(problem)
    return problems


_FORWARD_CONVENTIONS = (("x", "edge_index"), ("z", "pos"))
_CHAINED_OUTPUT_PARAMETERS = ("out_channels", "hidden_channels")


def _min_version(specifier: str) -> str | None:
    """The highest version a ``>=``, ``~=`` or ``==`` clause of ``specifier`` requires."""
    if not specifier:
        return None
    bounds = [
        Version(clause.version)
        for clause in SpecifierSet(specifier)
        if clause.operator in (">=", "~=", "==") and "*" not in clause.version
    ]
    return str(max(bounds)) if bounds else None


def _plugin_model_metadata(
    declaration: "ModelDeclaration", model_class: type, import_path: str, pyg_requirement: str
):
    """
    Build a plugin model's metadata: introspected from the class, then the manifest on top.

    The class gives ``parameters``, ``forward_parameters``, the generated
    ``hyperparameters`` schema and inferred defaults, exactly as for PyG models.
    The manifest's declarations override them; ``supported_tasks`` and
    ``description`` are inferred when the manifest leaves them out. The merge is
    validated as a whole (``model_validate``; ``model_copy(update=...)`` would
    skip validation).

    Raises:
        PluginModelError: If a hyperparameter override names a parameter the
            constructor does not have
    """
    base = build_metadata_for_class(declaration.name, model_class, import_path)

    unknown = sorted(
        name
        for name in declaration.hyperparameters
        if name not in base.parameters and not base.accepts_kwargs
    )
    if unknown:
        raise PluginModelError(
            f"Hyperparameter override(s) {unknown} for model '{declaration.name}' are not "
            f"parameters of {model_class.__name__}.__init__ "
            f"(parameters: {sorted(base.parameters)})",
            plugin_name=declaration.plugin_name,
            model_name=declaration.name,
        )
    hyperparameters = dict(base.hyperparameters)
    for name, override in declaration.hyperparameters.items():
        hyperparameters[name] = {**hyperparameters.get(name, {}), **override}

    overrides: dict[str, Any] = {
        "category": declaration.category,
        "hyperparameters": hyperparameters,
        "requires_edge_index": declaration.requires_edge_index,
        "requires_edge_features": declaration.requires_edge_features,
        "requires_edge_weights": declaration.requires_edge_weights,
        "supports_heterogeneous": declaration.supports_heterogeneous,
    }
    if declaration.supported_tasks:
        overrides["supported_tasks"] = declaration.supported_tasks
    if declaration.description:
        overrides["description"] = declaration.description
    if declaration.reference_paper:
        overrides["paper_title"] = declaration.reference_paper
    if declaration.reference_url:
        overrides["paper_url"] = declaration.reference_url
    min_pyg = declaration.min_pyg_version or _min_version(pyg_requirement)
    if min_pyg:
        overrides["min_pyg_version"] = min_pyg

    return ModelMetadata.model_validate({**dict(base), **overrides})


def _composability(metadata) -> tuple[bool, bool]:
    """
    Whether a model follows the composer's conventions (blueprint contract).

    Returns:
        ``(composable, chainable)``: ``composable`` when the required ``forward``
        parameters are ``(x, edge_index)`` or ``(z, pos)`` (optional ones may
        follow); ``chainable`` when it is composable and the constructor accepts
        ``in_channels`` and ``out_channels`` or ``hidden_channels`` (or ``**kwargs``).
    """
    required = tuple(name for name, info in metadata.forward_parameters.items() if info.required)
    composable = required in _FORWARD_CONVENTIONS
    parameters = metadata.parameters
    constructor_chains = metadata.accepts_kwargs or (
        "in_channels" in parameters and any(p in parameters for p in _CHAINED_OUTPUT_PARAMETERS)
    )
    return composable, composable and constructor_chains


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


def _plugin_source(metadata) -> str:
    """Where a plugin was discovered: its entry point, or its directory."""
    if metadata.entry_point is not None:
        return f"entry point '{metadata.entry_point}'"
    return str(metadata.plugin_path)


def _same_plugin_source(first, second) -> bool:
    """Whether two discoveries of a plugin name come from the same source."""
    return first.entry_point == second.entry_point and _same_plugin_dir(
        first.plugin_path, second.plugin_path
    )


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
        category: Model category, a ``ModelCategory`` value (e.g., 'basic_gnn', 'general')
        description: Brief description of the model
        supported_tasks: List of supported task types (empty: not declared in the manifest)
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
    # Set when the model is registered (composability contract); None before
    composable: bool | None = None  # forward() follows a composer convention
    chainable: bool | None = None  # constructor chains dimensions (sequential/hierarchical)

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
        entry_point: For a plugin from an installed distribution, its entry point
            (``"<name> = <package>"``); its ``class`` references are absolute
            import paths. ``None`` for a directory plugin.
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
    entry_point: str | None = None
    requires_extra: str | None = None
    schema_version: int | None = None  # manifest schema version; None if not read from a manifest
    legacy_manifest: bool = False  # the manifest used the deprecated legacy format

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
    - Validation of manifests, versions and dependencies (plugins are trusted
      code: see the module docstring)
    - Thread-safe plugin management
    - Integration with ModelRegistry
    - Enable/disable functionality

    Usage:
        >>> loader = ModelPluginLoader()
        >>> plugins = loader.discover_plugins([Path("/path/to/my_model_plugins")])
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

        A plugin is not registered (and is logged with the reason) when its
        ``requirements`` exclude the installed MILIA, Python or PyG version, when
        its ``requires_extra`` is not installed, or when validation reports errors.

        Args:
            paths: List of paths to search for plugins
            auto_validate: Whether to validate plugins automatically
            validation_level: Validation strictness (permissive, standard, strict);
                see ``_validate_plugin``

        Returns:
            List of discovered plugin names

        Example:
            >>> loader = ModelPluginLoader()
            >>> plugin_paths = [Path("/path/to/my_model_plugins")]
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

                    metadata = self._load_plugin_metadata(plugin_yaml)
                    if self._admit_plugin(metadata, auto_validate, validation_level):
                        discovered.append(metadata.plugin_name)

                except Exception as e:
                    logger.error(f"Failed to discover plugin from {plugin_yaml}: {e}")
                    if validation_level == "strict":
                        raise

        logger.info(f"Plugin discovery complete: {len(discovered)} plugins found")
        return discovered

    def discover_entry_point_plugins(
        self, auto_validate: bool = True, validation_level: str = "standard"
    ) -> list[str]:
        """
        Discover plugins provided by installed distributions.

        Each entry point in the ``milia.models`` group names a package holding
        ``plugin.yaml`` (for example ``my_plugin = "my_dist.milia_plugin"``); the
        entry-point name must equal the manifest's ``plugin_name``, and the
        manifest's ``class`` references are absolute import paths. Loading the
        entry point imports that package. The plugin is then checked and
        registered like a directory plugin.

        A failing entry point is logged and skipped, so the others are still
        discovered (re-raised at ``strict``, as for directory plugins).

        Args:
            auto_validate: Whether to validate plugins automatically
            validation_level: Validation strictness (permissive, standard, strict)

        Returns:
            List of discovered plugin names
        """
        discovered = []
        for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            try:
                metadata = self._load_entry_point_metadata(entry_point)
                if self._admit_plugin(metadata, auto_validate, validation_level):
                    discovered.append(metadata.plugin_name)
            except Exception as e:
                logger.error(
                    f"Failed to discover plugin from entry point "
                    f"'{entry_point.name} = {entry_point.value}': {e}"
                )
                if validation_level == "strict":
                    raise

        logger.info(f"Entry-point plugin discovery complete: {len(discovered)} plugins found")
        return discovered

    def _load_entry_point_metadata(
        self, entry_point: importlib.metadata.EntryPoint
    ) -> ModelPluginMetadata:
        """
        Load plugin metadata from the package an entry point names.

        Raises:
            PluginModelError: If the entry point does not name a package holding
                ``plugin.yaml``, the manifest is empty or not valid YAML, or its
                ``plugin_name`` is not the entry-point name
            PluginValidationError: If the manifest is not valid
        """
        reference = f"{entry_point.name} = {entry_point.value}"
        if entry_point.attr:
            raise PluginModelError(
                f"Entry point '{reference}' must name a package, not an object in it",
                plugin_name=entry_point.name,
            )

        package = entry_point.load()
        if not hasattr(package, "__path__"):
            raise PluginModelError(
                f"Entry point '{reference}' must name a package holding plugin.yaml; "
                f"'{entry_point.module}' is not a package",
                plugin_name=entry_point.name,
            )

        manifest_file = importlib.resources.files(package) / "plugin.yaml"
        if not manifest_file.is_file():
            raise PluginModelError(
                f"Entry point '{reference}': package '{entry_point.module}' has no plugin.yaml",
                plugin_name=entry_point.name,
            )

        source = f"{entry_point.module}/plugin.yaml"
        try:
            data = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise PluginModelError(
                f"Invalid YAML in {source}: {e}", plugin_name=entry_point.name
            ) from e

        # The package directory, when the distribution is installed on disk.
        plugin_path = manifest_file.parent if isinstance(manifest_file, Path) else None
        metadata = self._metadata_from_manifest(
            data, source, plugin_path=plugin_path, entry_point=reference
        )
        if metadata.plugin_name != entry_point.name:
            raise PluginModelError(
                f"Entry point '{reference}': name '{entry_point.name}' does not match the "
                f"manifest's plugin_name '{metadata.plugin_name}'",
                plugin_name=entry_point.name,
            )
        return metadata

    def _admit_plugin(
        self, metadata: ModelPluginMetadata, auto_validate: bool, validation_level: str
    ) -> bool:
        """
        Check a discovered plugin and register it.

        Returns:
            ``True`` if the plugin is registered (or was already discovered from the
            same source); ``False`` if it was skipped, with the reason logged.

        Raises:
            PluginModelError: If another plugin already has this name
        """
        # Compatibility gates, at every level and before validation: a plugin for
        # another MILIA, Python or PyG version, or whose install extra is missing,
        # is skipped with the reason.
        gate_failures = self._version_gate_failures(metadata)
        if gate_failures:
            logger.warning(
                f"Plugin '{metadata.plugin_name}' not registered: {'; '.join(gate_failures)}"
            )
            return False

        if metadata.requires_extra:
            extra_problems = _extra_problems(metadata.requires_extra)
            if extra_problems:
                logger.info(
                    f"Skipping plugin '{metadata.plugin_name}': optional extra "
                    f"'{metadata.requires_extra}' not installed ({'; '.join(extra_problems)})"
                )
                return False

        # A plugin whose validation reports errors is never registered.
        if auto_validate:
            validation_result = self._validate_plugin(metadata, level=validation_level)
            if not validation_result["valid"]:
                logger.warning(
                    f"Plugin '{metadata.plugin_name}' not registered: failed "
                    f"validation: {validation_result['errors']}"
                )
                return False

        # Register plugin. A name is unique across all sources: the first plugin
        # found keeps it, and discovering the same source again keeps its state.
        with self._lock:
            existing = self._plugins.get(metadata.plugin_name)
            if existing is None:
                self._plugins[metadata.plugin_name] = metadata
            elif not _same_plugin_source(existing, metadata):
                raise PluginModelError(
                    f"Duplicate plugin name '{metadata.plugin_name}': already "
                    f"discovered at {_plugin_source(existing)}, ignoring "
                    f"{_plugin_source(metadata)}",
                    plugin_name=metadata.plugin_name,
                    plugin_path=str(metadata.plugin_path),
                )
            else:
                logger.debug(f"Plugin '{metadata.plugin_name}' already discovered")
                return True

        logger.info(
            f"Discovered plugin '{metadata.plugin_name}' "
            f"with {len(metadata.model_declarations)} model(s)"
        )
        return True

    def _load_plugin_metadata(self, plugin_yaml: Path) -> ModelPluginMetadata:
        """
        Load plugin metadata from plugin.yaml.

        Args:
            plugin_yaml: Path to plugin.yaml file

        Returns:
            ModelPluginMetadata instance

        The manifest is validated by ``parse_manifest`` (schema version 2, or the
        deprecated legacy format converted to it).

        Raises:
            PluginValidationError: If the manifest is not valid (field-located
                ``validation_errors``)
            PluginModelError: If the file is empty or not valid YAML
        """
        try:
            with open(plugin_yaml) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise PluginModelError(
                f"Invalid YAML in {plugin_yaml}: {e}", plugin_path=str(plugin_yaml)
            ) from e

        return self._metadata_from_manifest(data, plugin_yaml, plugin_path=plugin_yaml.parent)

    def _metadata_from_manifest(
        self,
        data: Any,
        source: Path | str,
        plugin_path: Path | None,
        entry_point: str | None = None,
    ) -> ModelPluginMetadata:
        """
        Build plugin metadata from a loaded manifest.

        Args:
            data: The manifest as loaded from YAML
            source: The manifest file, for messages and warnings
            plugin_path: The plugin directory, if the plugin has one on disk
            entry_point: The entry point, for a plugin from an installed distribution

        Raises:
            PluginValidationError: If the manifest is not valid
            PluginModelError: If the manifest is empty
        """
        if not data:
            raise PluginModelError(f"Empty plugin.yaml: {source}", plugin_path=str(source))

        manifest = parse_manifest(data, source)
        if not manifest.models:
            logger.warning(f"Plugin '{manifest.plugin_name}' declares no models")

        model_declarations = [
            ModelDeclaration(
                name=entry.name,
                class_name=entry.class_name,
                module_path=entry.module,
                category=entry.category,
                description=entry.description,
                supported_tasks=list(entry.supported_tasks or []),  # empty: inferred later
                hyperparameters=dict(entry.hyperparameters),
                plugin_name=manifest.plugin_name,
                requires_edge_index=entry.requires_edge_index,
                requires_edge_features=entry.requires_edge_features,
                requires_edge_weights=entry.requires_edge_weights,
                supports_heterogeneous=entry.supports_heterogeneous,
                reference_paper=entry.reference.paper,
                reference_url=entry.reference.url,
            )
            for entry in manifest.models
        ]

        requirements = manifest.requirements
        return ModelPluginMetadata(
            plugin_name=manifest.plugin_name,
            version=manifest.version,
            author=manifest.author,
            description=manifest.description,
            plugin_type=manifest.plugin_type,
            # "" = no constraint (an empty specifier set matches every version)
            milia_version=requirements.milia or "",
            pyg_version=requirements.torch_geometric or "",
            python_version=requirements.python or "",
            license=manifest.license,
            model_declarations=model_declarations,
            dependencies=list(manifest.dependencies),
            homepage=manifest.homepage,
            plugin_path=plugin_path,
            entry_point=entry_point,
            requires_extra=manifest.requires_extra,
            schema_version=manifest.schema_version,
            legacy_manifest="schema_version" not in data,
        )

    # =========================================================================
    # PLUGIN VALIDATION
    # =========================================================================

    def _validate_plugin(
        self, metadata: ModelPluginMetadata, level: str = "standard"
    ) -> dict[str, Any]:
        """
        Validate plugin metadata and requirements.

        Errors are the same at every level: a missing ``plugin_name``, an invalid,
        missing or unsatisfied dependency, and an incomplete model declaration. A
        plugin with errors is never registered. Warnings: a missing version and no
        models; ``strict`` turns every warning into an error.

        The code scan (``_security_check``) is neither: its findings are lint hints,
        logged at INFO at ``standard`` and ``strict`` (``permissive`` skips the
        scan), and never affect registration.

        Args:
            metadata: Plugin metadata to validate
            level: Validation level (permissive, standard, strict)

        Returns:
            ``{"valid": bool, "errors": [str], "warnings": [str]}``
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
        errors.extend(dep_result["missing"])

        # Validate model declarations
        for decl in metadata.model_declarations:
            errors.extend(self._validate_model_declaration(decl))

        # Lint hints: reported, never blocking (plugins are trusted code)
        if level in ["standard", "strict"]:
            for hint in self._security_check(metadata)["issues"]:
                logger.info(f"Plugin '{metadata.plugin_name}' lint hint: {hint}")

        if level == "strict":
            errors.extend(warnings)
            warnings = []

        # Store validation errors in metadata
        metadata.validation_errors = errors + warnings

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _version_gate_failures(self, metadata: ModelPluginMetadata) -> list[str]:
        """
        Check the plugin's ``requirements`` against the installed versions.

        Each is a PEP 440 specifier, checked with
        ``SpecifierSet(spec).contains(Version(v), prereleases=True)``; ``""`` is no
        constraint.

        Returns:
            One reason per failed gate; empty if the plugin is compatible.
        """
        installed = _installed_versions()
        gates = (
            ("milia", metadata.milia_version),
            ("python", metadata.python_version),
            ("torch_geometric", metadata.pyg_version),
        )
        return [
            reason
            for label, specifier in gates
            if (reason := _version_gate(label, specifier, installed[label])) is not None
        ]

    def _validate_dependencies(self, metadata: ModelPluginMetadata) -> dict[str, Any]:
        """
        Check plugin dependencies against the installed distributions.

        Each entry is a PEP 508 requirement on a distribution name (for example
        ``"torch>=2.4"`` or ``"scikit-learn"``). It is checked from installed package
        metadata; nothing is imported. An entry whose environment marker is false
        is skipped.

        Returns:
            ``{"satisfied": bool, "missing": [str]}``; ``missing`` lists one message
            per missing, unsatisfied or invalid entry.
        """
        problems = []

        for dep in metadata.dependencies:
            problem = _check_requirement(dep)
            if problem:
                problems.append(problem)

        return {"satisfied": not problems, "missing": problems}

    def _validate_model_declaration(self, declaration: ModelDeclaration) -> list[str]:
        """Validate a single model declaration."""
        errors = []

        if not declaration.name:
            errors.append("Model declaration missing name")

        if not declaration.class_name:
            errors.append(f"Model '{declaration.name}' missing class_name")

        if not declaration.module_path:
            errors.append(f"Model '{declaration.name}' missing module_path")

        # Validate hyperparameter schema
        for param_name, param_schema in declaration.hyperparameters.items():
            if not isinstance(param_schema, dict):
                errors.append(
                    f"Model '{declaration.name}' hyperparameter '{param_name}' "
                    f"must be a dict with schema"
                )

        return errors

    def _security_check(self, metadata: ModelPluginMetadata) -> dict[str, Any]:
        """
        Scan a directory plugin's Python files for dynamic-execution patterns.

        This is a lint hint for the plugin's reviewer, not a security control.
        Importing a plugin runs its code, and a pattern list cannot decide what code
        does (an incomplete list of disallowed inputs, CWE-184): obfuscated code
        passes and harmless code may match. Plugins are trusted code; see the module
        docstring. A plugin without a ``plugin_path`` on disk is not scanned.

        Returns:
            ``{"issues": [str]}``, one hint per file and pattern.
        """
        issues = []

        if metadata.plugin_path:
            suspicious_patterns = [
                r"\bos\.system\(",
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
                # An entry-point plugin's modules are imported by their absolute names.
                plugin_path = None if metadata.entry_point is not None else metadata.plugin_path
                pyg_requirement = metadata.pyg_version
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
                    self._register_plugin_model(decl, plugin_path, pyg_requirement)
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

    def _register_plugin_model(
        self, declaration: ModelDeclaration, plugin_dir: Path, pyg_requirement: str = ""
    ):
        """
        Register a model from plugin with ModelRegistry.

        Args:
            declaration: Model declaration from plugin
            plugin_dir: Plugin directory path. Its modules are imported under the
                plugin's own package; without a directory, ``module_path`` is an
                installed module's absolute name.
            pyg_requirement: The plugin's ``torch_geometric`` requirement; its lower
                bound becomes the model's ``min_pyg_version``.

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

            if ModelMetadata is None or build_metadata_for_class is None:
                raise PluginError("ModelMetadata not available")

            # Complete metadata from the class (as for PyG models), then the manifest's
            # declarations on top, validated as a whole.
            metadata = _plugin_model_metadata(
                declaration,
                model_class,
                f"{model_class.__module__}.{model_class.__qualname__}",
                pyg_requirement,
            )
            declaration.composable, declaration.chainable = _composability(metadata)
            if not declaration.composable:
                logger.warning(
                    f"Plugin model '{declaration.name}': forward() does not follow "
                    "(x, edge_index[, edge_attr][, batch]) or (z, pos[, batch]); "
                    "registered; not composable in ensembles"
                )
            elif not declaration.chainable:
                logger.warning(
                    f"Plugin model '{declaration.name}': constructor lacks in_channels and "
                    "out_channels/hidden_channels; registered; not chainable in sequential "
                    "or hierarchical ensembles"
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
            raise PluginModelError(
                f"Failed to import model '{declaration.name}': {e}",
                plugin_name=declaration.plugin_name,
                model_name=declaration.name,
                plugin_path=None if plugin_dir is None else str(plugin_dir),
            ) from e
        except AttributeError as e:
            raise PluginModelError(
                f"Model class '{declaration.class_name}' not found in module: {e}",
                plugin_name=declaration.plugin_name,
                model_name=declaration.name,
                plugin_path=None if plugin_dir is None else str(plugin_dir),
            ) from e
        except Exception as e:
            raise PluginModelError(
                f"Failed to register model '{declaration.name}': {e}",
                plugin_name=declaration.plugin_name,
                model_name=declaration.name,
                plugin_path=None if plugin_dir is None else str(plugin_dir),
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
        paths: List of paths to search. Default: the configured sources, that is
            the directories from ``models.plugins`` and, when
            ``discover_entry_points`` is true, the ``milia.models`` entry points of
            installed distributions (directories first, so a directory plugin keeps
            its name against an entry-point plugin). Given paths are searched alone.
        auto_validate: Whether to validate automatically
        validation_level: Validation strictness

    Returns:
        List of discovered plugin names

    Example:
        >>> from milia_pipeline.models.model_plugin_system import discover_plugins
        >>> from pathlib import Path
        >>> plugins = discover_plugins([Path("/path/to/my_model_plugins")])
        >>> print(f"Discovered {len(plugins)} plugins")
    """
    loader = get_plugin_loader()

    if paths is not None:
        return loader.discover_plugins(paths, auto_validate, validation_level)

    paths, entry_points = _configured_sources()
    discovered = loader.discover_plugins(paths, auto_validate, validation_level)
    if entry_points:
        discovered += loader.discover_entry_point_plugins(auto_validate, validation_level)
    return discovered


def _configured_plugin_paths(config_path: Path | None = None) -> list[Path]:
    """The model plugin directories the configuration selects (see ``_configured_sources``)."""
    return _configured_sources(config_path)[0]


def _configured_sources(config_path: Path | None = None) -> tuple[list[Path], bool]:
    """
    The model plugin sources the configuration selects.

    Args:
        config_path: The configuration directory or file (default: ``./configs``).

    Returns:
        ``(directories, discover_entry_points)``; see ``_configured_plugins``.
    """
    config, paths = _configured_plugins(config_path)
    return paths, config.discover_entry_points


def _configured_plugins(config_path: Path | None = None):
    """
    The model plugins configuration and the directories it selects.

    The directories are the built-in container (unless ``include_builtin`` is
    false) and the user ``plugin_paths``, resolved against the configuration root.

    Args:
        config_path: The configuration directory or file (default: ``./configs``).

    Returns:
        ``(PluginsConfig, directories)``: existing, resolved directories. If the
        configuration cannot be loaded, the ``PluginsConfig`` defaults apply (the
        built-in container and entry points), with a warning.
    """
    from milia_pipeline.config.config_loader import get_config_root
    from milia_pipeline.models._model_plugin_paths import (
        builtin_model_plugins_directory,
        resolve_model_plugin_paths,
    )
    from milia_pipeline.models.utils.config_bridge import PluginsConfig, get_plugins_config

    try:
        config = get_plugins_config(config_path)
        config_root = get_config_root(config_path)
    except Exception as e:
        logger.warning(
            f"Could not load plugin paths from config: {e}; using the built-in container "
            f"and installed entry points"
        )
        return PluginsConfig(), [builtin_model_plugins_directory()]

    paths = resolve_model_plugin_paths(
        config.plugin_paths, config_root=config_root, include_builtin=config.include_builtin
    )
    return config, paths


# Process-wide automatic discovery: the first caller runs it, concurrent callers
# wait for it, later callers get its result. The lock guards only this state.
_auto_discovery_lock = threading.Lock()
_auto_discovery: _LoadAttempt | None = None
_auto_discovered: list[str] = []


def ensure_model_plugins_discovered(config_path: Path | None = None) -> list[str]:
    """
    Discover and load the configured model plugins, once per process.

    Called when models are built (``ModelFactory()``, HPO search-space filtering),
    not at import time. Gated by ``models.plugins.enabled`` and ``auto_discover``;
    sources, ``auto_validate`` and ``validation_level`` come from the same
    configuration. The first call does the work; concurrent calls wait for it and
    later calls return its result, whatever ``config_path`` they pass. A call made
    by plugin code during that discovery returns ``[]`` (not reentrant).

    Never raises: a discovery error is logged and gives ``[]``, and a plugin that
    fails to load is logged and left out, so building models still works.

    Args:
        config_path: The configuration directory or file (default: ``./configs``).

    Returns:
        The names of the plugins loaded (their models are registered).
    """
    global _auto_discovery, _auto_discovered

    with _auto_discovery_lock:
        attempt = _auto_discovery
        if attempt is None:
            attempt = _auto_discovery = _LoadAttempt(owner=threading.get_ident())
            owner = True
        else:
            owner = False

    if not owner:
        if attempt.owner == threading.get_ident() and not attempt.done.is_set():
            logger.warning("Model plugin discovery was requested again while it runs; skipped")
            return []
        attempt.done.wait()
        return list(_auto_discovered)

    loaded: list[str] = []
    try:
        loaded = _discover_and_load_configured(config_path)
    except Exception as e:
        attempt.error = e
        logger.error(f"Automatic model plugin discovery failed: {e}")
    finally:
        with _auto_discovery_lock:
            _auto_discovered = loaded
        attempt.done.set()
    return list(loaded)


def _discover_and_load_configured(config_path: Path | None) -> list[str]:
    """Discover the configured plugins and load each enabled one; return those loaded."""
    config, paths = _configured_plugins(config_path)
    if not (config.enabled and config.auto_discover):
        logger.debug("Automatic model plugin discovery is disabled by the configuration")
        return []

    loader = get_plugin_loader()
    names = loader.discover_plugins(paths, config.auto_validate, config.validation_level)
    if config.discover_entry_points:
        names += loader.discover_entry_point_plugins(config.auto_validate, config.validation_level)

    enabled = set(loader.list_plugins(enabled_only=True))
    loaded = []
    for name in dict.fromkeys(names):
        if name not in enabled:
            continue
        try:
            loader.load_plugin(name)
        except Exception as e:
            logger.error(f"Model plugin '{name}' not loaded: {e}")
            continue
        loaded.append(name)
    logger.info(f"Automatic model plugin discovery: {len(loaded)} plugin(s) loaded")
    return loaded


def _reset_model_plugin_discovery() -> None:
    """Forget automatic discovery, so the next ``ensure_model_plugins_discovered`` runs it (tests)."""
    global _auto_discovery, _auto_discovered
    with _auto_discovery_lock:
        _auto_discovery = None
        _auto_discovered = []


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
        ...     paths=[Path("/path/to/my_model_plugins")],
        ...     validation_level="strict"
        ... )
        >>> print(f"Loaded {sum(results.values())} of {len(results)} plugins")
    """
    loader = get_plugin_loader()

    # Discover plugins (paths=None: the configured sources, as discover_plugins)
    plugin_names = discover_plugins(
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
        >>> plugins, errors = safe_discover_plugins([Path("/path/to/my_model_plugins")])
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
    # Automatic discovery when models are built
    "ensure_model_plugins_discovered",
]
