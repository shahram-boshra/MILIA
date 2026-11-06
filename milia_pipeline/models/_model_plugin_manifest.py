"""Model plugin manifest (``plugin.yaml``): schema version 2 and the legacy adapter.

A manifest with ``schema_version: 2`` is validated strictly: unknown keys are
rejected. A manifest without ``schema_version`` is the legacy format. It is
converted to version 2 first; every legacy key that is mapped or ignored emits a
``DeprecationWarning`` attributed to the manifest file (removal in MILIA 2.0.0).

``parse_manifest`` is the only entry point. Validation errors are raised as
``PluginValidationError`` with one ``<field location>: <message>`` line per error.
"""

from __future__ import annotations

import keyword
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Literal

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from milia_pipeline.exceptions import PluginValidationError
from milia_pipeline.models.registry.pyg_introspector import ModelCategory

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
LEGACY_REMOVAL_VERSION = "2.0.0"

# Task names MILIA's model metadata uses (the vocabulary of task inference in
# ``registry/pyg_introspector.py``).
KNOWN_TASKS = frozenset(
    {
        "node_classification",
        "node_regression",
        "graph_classification",
        "graph_regression",
        "link_prediction",
        "edge_regression",
        "edge_classification",
        "node_embedding",
    }
)

PLUGIN_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Normalised extra name (PEP 685).
EXTRA_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# Semantic Versioning 2.0.0, the regular expression published at semver.org.
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

_CONTEXT_LEGACY = "legacy"
_CONTEXT_SOURCE = "source"


def _is_identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def _deprecated(
    message: str, info: ValidationInfo | None = None, source: str | None = None
) -> None:
    """Emit a manifest deprecation, attributed to the manifest file."""
    if source is None and info is not None and info.context:
        source = info.context.get(_CONTEXT_SOURCE)
    text = (
        f"{message} This form is deprecated and will be removed in MILIA {LEGACY_REMOVAL_VERSION}."
    )
    warnings.warn_explicit(
        text, DeprecationWarning, filename=str(source or "plugin.yaml"), lineno=0
    )
    logger.warning(f"{source or 'plugin.yaml'}: {text}")


def _is_legacy(info: ValidationInfo) -> bool:
    return bool(info.context and info.context.get(_CONTEXT_LEGACY))


def _check_specifier(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        SpecifierSet(value)
    except InvalidSpecifier as e:
        raise ValueError(f"invalid version specifier {value!r}: {e}") from e
    return value


# =============================================================================
# SCHEMA VERSION 2
# =============================================================================


class ModelReference(BaseModel):
    """Where a model is described."""

    model_config = ConfigDict(extra="forbid")

    paper: str | None = None
    url: str | None = None


class ModelEntry(BaseModel):
    """One model a plugin provides."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    class_ref: str = Field(alias="class")
    category: str = ModelCategory.GENERAL.value
    supported_tasks: list[str] | None = None  # None: inferred from the class
    description: str = ""
    requires_edge_index: bool = True
    requires_edge_features: bool = False
    requires_edge_weights: bool = False
    supports_heterogeneous: bool = False
    # Overrides of the introspected schema, in the registry's metadata format: a
    # parameter name maps to a default value or to a schema mapping.
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    reference: ModelReference = Field(default_factory=ModelReference)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not MODEL_NAME_PATTERN.fullmatch(value):
            raise ValueError(f"must match {MODEL_NAME_PATTERN.pattern}")
        return value

    @field_validator("class_ref")
    @classmethod
    def _valid_class_ref(cls, value: str) -> str:
        module, sep, attr = value.partition(":")
        parts = module.split(".") + attr.split(".")
        if not sep or not all(_is_identifier(p) for p in parts):
            raise ValueError(
                "must be '<module>:<class>', e.g. 'model:MyNet' (dotted names allowed)"
            )
        return value

    @field_validator("category")
    @classmethod
    def _valid_category(cls, value: str) -> str:
        allowed = sorted(c.value for c in ModelCategory)
        if value not in allowed:
            raise ValueError(f"unknown category {value!r}; expected one of {allowed}")
        return value

    @field_validator("supported_tasks")
    @classmethod
    def _valid_tasks(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("must list at least one task, or be omitted to infer them")
        unknown = [task for task in value if task not in KNOWN_TASKS]
        if unknown:
            raise ValueError(f"unknown task(s) {unknown}; expected some of {sorted(KNOWN_TASKS)}")
        return value

    @property
    def module(self) -> str:
        """Module part of ``class``."""
        return self.class_ref.partition(":")[0]

    @property
    def class_name(self) -> str:
        """Attribute part of ``class``."""
        return self.class_ref.partition(":")[2]


class PluginRequirements(BaseModel):
    """PEP 440 version specifiers the environment must satisfy."""

    model_config = ConfigDict(extra="forbid")

    milia: str | None = None
    python: str | None = None
    torch_geometric: str | None = None

    @field_validator("milia", "python", "torch_geometric")
    @classmethod
    def _valid_specifier(cls, value: str | None) -> str | None:
        return _check_specifier(value)


class PluginManifestV2(BaseModel):
    """A model plugin manifest, schema version 2."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    plugin_name: str
    version: str
    author: str = ""
    description: str = ""
    license: str = ""
    plugin_type: Literal["first_party", "user"] = "user"
    homepage: str | None = None
    requirements: PluginRequirements = Field(default_factory=PluginRequirements)
    dependencies: list[str] = Field(default_factory=list)
    requires_extra: str | None = None
    models: list[ModelEntry] = Field(default_factory=list)

    @field_validator("plugin_name")
    @classmethod
    def _valid_plugin_name(cls, value: str, info: ValidationInfo) -> str:
        if PLUGIN_NAME_PATTERN.fullmatch(value):
            return value
        if _is_legacy(info) and _is_identifier(value):
            _deprecated(
                f"plugin_name {value!r} does not match {PLUGIN_NAME_PATTERN.pattern}.", info
            )
            return value
        raise ValueError(f"must match {PLUGIN_NAME_PATTERN.pattern}")

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str, info: ValidationInfo) -> str:
        if SEMVER_PATTERN.fullmatch(value):
            return value
        if _is_legacy(info):
            _deprecated(f"version {value!r} is not a Semantic Versioning 2.0.0 version.", info)
            return value
        raise ValueError("must be a Semantic Versioning 2.0.0 version, e.g. '1.0.0'")

    @field_validator("dependencies")
    @classmethod
    def _valid_dependencies(cls, value: list[str]) -> list[str]:
        for dependency in value:
            try:
                Requirement(dependency)
            except InvalidRequirement as e:
                reason = str(e).splitlines()[0] if str(e) else type(e).__name__
                raise ValueError(f"invalid PEP 508 requirement {dependency!r}: {reason}") from e
        return value

    @field_validator("requires_extra")
    @classmethod
    def _valid_extra(cls, value: str | None) -> str | None:
        if value is not None and not EXTRA_NAME_PATTERN.fullmatch(value):
            raise ValueError(f"must be a normalised extra name ({EXTRA_NAME_PATTERN.pattern})")
        return value

    @model_validator(mode="after")
    def _unique_model_names(self) -> PluginManifestV2:
        names = [model.name for model in self.models]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate model names: {duplicates}")
        return self


# =============================================================================
# LEGACY ADAPTER
# =============================================================================

_LEGACY_TOP_LEVEL_REQUIREMENTS = {
    "milia_version": "milia",
    "python_version": "python",
    "pyg_version": "torch_geometric",
}
_LEGACY_IGNORED_TOP_LEVEL = ("email", "repository", "documentation", "optional_dependencies")
_LEGACY_IGNORED_MODEL = ("version", "supports_batch", "plugin_name")
_V2_TOP_LEVEL = set(PluginManifestV2.model_fields)
_V2_MODEL = {"class" if f == "class_ref" else f for f in ModelEntry.model_fields}


def _legacy_to_v2(data: dict[str, Any], source: str) -> dict[str, Any]:
    """Convert a legacy manifest to a schema version 2 mapping, warning per legacy key."""
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    requirements: dict[str, Any] = {}
    pyg_specifiers: list[str] = []
    for key, value in (data.get("requirements") or {}).items():
        target = _LEGACY_TOP_LEVEL_REQUIREMENTS.get(key, key)
        if key in _LEGACY_TOP_LEVEL_REQUIREMENTS:
            _deprecated(
                f"'requirements.{key}' is a legacy key; use 'requirements.{target}'.", source=source
            )
        elif key == "min_pyg_version":
            _deprecated(
                "'requirements.min_pyg_version' is a legacy key; use "
                "'requirements.torch_geometric: \">=X\"'.",
                source=source,
            )
            target, value = "torch_geometric", None if value is None else f">={value}"
        if value is None:
            continue
        if target == "torch_geometric":
            pyg_specifiers.append(str(value))
        else:
            requirements[target] = value  # unknown keys are rejected by validation

    for key, value in data.items():
        if key == "requirements":
            continue
        if key in _LEGACY_TOP_LEVEL_REQUIREMENTS:
            target = _LEGACY_TOP_LEVEL_REQUIREMENTS[key]
            _deprecated(f"'{key}' is a legacy key; use 'requirements.{target}'.", source=source)
            if value is None:
                continue
            if target == "torch_geometric":
                pyg_specifiers.append(str(value))
            else:
                requirements[target] = str(value)
        elif key == "min_pyg_version":
            _deprecated(
                "'min_pyg_version' is a legacy key; use 'requirements.torch_geometric: \">=X\"'.",
                source=source,
            )
            if value is not None:
                pyg_specifiers.append(f">={value}")
        elif key in _LEGACY_IGNORED_TOP_LEVEL:
            _deprecated(
                f"'{key}' is not part of the manifest schema and is ignored.", source=source
            )
        elif key == "plugin_type":
            if value in ("first_party", "user"):
                result[key] = value
            else:
                _deprecated(
                    f"plugin_type {value!r} is a legacy value; it is read as 'user' "
                    "(allowed: 'first_party', 'user').",
                    source=source,
                )
                result[key] = "user"
        elif key == "models":
            result["models"] = [
                _legacy_model_to_v2(model, index, source, pyg_specifiers)
                for index, model in enumerate(value or [])
            ]
        elif key in _V2_TOP_LEVEL:
            if value is None and key == "dependencies":
                continue  # an empty YAML key (every item commented out) means no dependencies
            result[key] = value
        else:
            _deprecated(
                f"'{key}' is not part of the manifest schema and is ignored.", source=source
            )

    if "version" not in result:
        _deprecated("'version' is missing; it is read as '0.0.0'.", source=source)
        result["version"] = "0.0.0"
    if pyg_specifiers:
        requirements["torch_geometric"] = ",".join(pyg_specifiers)
    if requirements:
        result["requirements"] = requirements
    return result


def _legacy_model_to_v2(model: Any, index: int, source: str, pyg_specifiers: list[str]) -> Any:
    """Convert one legacy model entry; non-mapping entries are left for validation to reject."""
    if not isinstance(model, dict):
        return model
    where = f"models[{index}]"
    result: dict[str, Any] = {}
    module_path = model.get("module_path")
    class_name = model.get("class_name")
    if module_path is not None or class_name is not None:
        _deprecated(
            f"'{where}.class_name' and '{where}.module_path' are legacy keys; "
            "use 'class: \"<module>:<class>\"'.",
            source=source,
        )
        module = str(module_path or "")
        if module.endswith(".py"):
            module = module[: -len(".py")]
        result["class"] = f"{module}:{class_name or ''}"

    reference: dict[str, Any] = dict(model.get("reference") or {})
    for key, value in model.items():
        if key in ("module_path", "class_name", "reference"):
            continue
        if key in ("reference_paper", "reference_url"):
            _deprecated(
                f"'{where}.{key}' is a legacy key; use '{where}.reference.{key[len('reference_') :]}'.",
                source=source,
            )
            if value is not None:
                reference[key[len("reference_") :]] = value
        elif key == "min_pyg_version":
            _deprecated(
                f"'{where}.min_pyg_version' is a legacy key; use the plugin's "
                "'requirements.torch_geometric'.",
                source=source,
            )
            if value is not None:
                pyg_specifiers.append(f">={value}")
        elif key in _LEGACY_IGNORED_MODEL:
            _deprecated(
                f"'{where}.{key}' is not part of the manifest schema and is ignored.", source=source
            )
        elif key in _V2_MODEL:
            if value is None and key in ("hyperparameters", "supported_tasks", "description"):
                continue  # an empty YAML key means the default
            result[key] = value
        else:
            _deprecated(
                f"'{where}.{key}' is not part of the manifest schema and is ignored.", source=source
            )

    if reference:
        result["reference"] = reference
    return result


# =============================================================================
# ENTRY POINT
# =============================================================================


def _format_errors(error: ValidationError) -> list[str]:
    lines = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "<manifest>"
        message = item["msg"].removeprefix("Value error, ")
        lines.append(f"{location}: {message}")
    return lines


def parse_manifest(data: dict, source: Path | str) -> PluginManifestV2:
    """
    Validate a model plugin manifest and return it as schema version 2.

    Args:
        data: The manifest as loaded from YAML
        source: The manifest file, used in warnings and error messages

    Returns:
        The validated manifest

    Raises:
        PluginValidationError: If the manifest is not valid; ``validation_errors``
            holds one ``<field location>: <message>`` line per error
    """
    source = str(source)
    if not isinstance(data, dict):
        raise PluginValidationError(
            f"Invalid plugin manifest {source}: expected a mapping, got {type(data).__name__}",
            validation_errors=[f"<manifest>: expected a mapping, got {type(data).__name__}"],
        )

    plugin_name = data.get("plugin_name") if isinstance(data.get("plugin_name"), str) else None
    legacy = "schema_version" not in data
    if not legacy and data["schema_version"] != SCHEMA_VERSION:
        message = f"schema_version: unsupported value {data['schema_version']!r}; expected {SCHEMA_VERSION}"
        raise PluginValidationError(
            f"Invalid plugin manifest {source}: {message}",
            plugin_name=plugin_name,
            validation_errors=[message],
        )

    candidate = _legacy_to_v2(data, source) if legacy else data
    try:
        return PluginManifestV2.model_validate(
            candidate, context={_CONTEXT_LEGACY: legacy, _CONTEXT_SOURCE: source}
        )
    except ValidationError as e:
        errors = _format_errors(e)
        raise PluginValidationError(
            f"Invalid plugin manifest {source}:\n" + "\n".join(f"  {line}" for line in errors),
            plugin_name=plugin_name,
            validation_errors=errors,
        ) from e
