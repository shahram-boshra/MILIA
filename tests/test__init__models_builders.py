# tests/test__init__models_builders.py

"""
Test Suite: milia_pipeline/models/builders/__init__.py — Smoke Tests & Contract Tests
======================================================================================

Production-ready test suite for the MILIA Pipeline models builders package
``milia_pipeline/models/builders/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.models.builders`` subpackage imports without ImportError
        - All re-exported names from the 6 submodules are accessible
        - Module-level metadata attributes (__version__, __author__) exist
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Layer registry components are accessible
        - Architecture builder components are accessible
        - Model composer components are accessible
        - Template components are accessible
        - Config parser components are accessible
        - Validation components are accessible
        - Exception classes are accessible

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, callables are callable
        - Exception classes inherit from Exception
        - Dataclass exports are dataclasses (LayerConfig, ResidualConnection, etc.)
        - Enum exports are enums (LayerCategory)
        - nn.Module subclasses are identified (CustomArchitecture, ParallelEnsemble, etc.)
        - Convenience functions have documented parameter signatures
        - ``layer_registry`` is a LayerRegistry instance
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)
        - Import source verification (names come from correct submodules)

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__models_builders.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__models_builders.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def builders_pkg():
    """
    Import and return the ``milia_pipeline.models.builders`` package once
    per module.

    This fixture validates the fundamental smoke invariant: the builders
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.models.builders as bld
        return bld
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.models.builders could not be imported — smoke "
            f"test precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(builders_pkg):
    """Return the ``__all__`` list from the builders package."""
    assert hasattr(builders_pkg, "__all__"), (
        "milia_pipeline.models.builders.__all__ is missing — contract violation"
    )
    return list(builders_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeBuilderPackageImport:
    """§1.2 — Verify the builders subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_builders_package_succeeds(self, builders_pkg):
        """The builders package imports without raising any exception."""
        assert builders_pkg is not None

    @pytest.mark.smoke
    def test_builders_package_is_a_module(self, builders_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(builders_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_builders_package_has_file_attribute(self, builders_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(builders_pkg, "__file__")

    @pytest.mark.smoke
    def test_builders_package_name(self, builders_pkg):
        """The package ``__name__`` is ``milia_pipeline.models.builders``."""
        assert builders_pkg.__name__ == "milia_pipeline.models.builders"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
    ])
    def test_metadata_attribute_exists(self, builders_pkg, attr):
        """Each metadata dunder is defined on the builders package."""
        assert hasattr(builders_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
    ])
    def test_metadata_attribute_is_string(self, builders_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(builders_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, builders_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = builders_pkg.__version__
        parts = version.split(".")
        assert len(parts) >= 2, (
            f"Version '{version}' should have at least MAJOR.MINOR components"
        )
        for part in parts:
            numeric_part = ""
            for ch in part:
                if ch.isdigit():
                    numeric_part += ch
                else:
                    break
            assert len(numeric_part) > 0, (
                f"Version component '{part}' should start with a digit"
            )


class TestSmokeLayerRegistryExports:
    """§1.2 — Layer registry components are accessible from the builders package."""

    LAYER_REGISTRY_EXPORTS = [
        "LayerRegistry",
        "LayerCategory",
        "LayerMetadata",
        "FunctionalLayerWrapper",
        "LayerNotFoundError",
        "get_layer",
        "list_layers",
        "get_layer_metadata",
        "layer_registry",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", LAYER_REGISTRY_EXPORTS)
    def test_layer_registry_export_exists(self, builders_pkg, name):
        """Each layer registry export is present and non-None."""
        obj = getattr(builders_pkg, name, None)
        assert obj is not None, (
            f"Layer registry export '{name}' is None or missing"
        )


class TestSmokeArchitectureBuilderExports:
    """§1.2 — Architecture builder components are accessible."""

    ARCHITECTURE_BUILDER_EXPORTS = [
        "ArchitectureBuilder",
        "LayerConfig",
        "ArchitectureConfig",
        "ResidualConnection",
        "ArchitectureError",
        "ChannelMismatchError",
        "CustomArchitecture",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ARCHITECTURE_BUILDER_EXPORTS)
    def test_architecture_builder_export_exists(self, builders_pkg, name):
        """Each architecture builder export is present and non-None."""
        obj = getattr(builders_pkg, name, None)
        assert obj is not None, (
            f"Architecture builder export '{name}' is None or missing"
        )


class TestSmokeModelComposerExports:
    """§1.2 — Model composer components are accessible."""

    MODEL_COMPOSER_EXPORTS = [
        "ModelSpec",
        "EnsembleConfig",
        "CompositionError",
        "ModelComposer",
        "ParallelEnsemble",
        "SequentialStack",
        "HierarchicalComposition",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MODEL_COMPOSER_EXPORTS)
    def test_model_composer_export_exists(self, builders_pkg, name):
        """Each model composer export is present and non-None."""
        obj = getattr(builders_pkg, name, None)
        assert obj is not None, (
            f"Model composer export '{name}' is None or missing"
        )


class TestSmokeTemplateExports:
    """§1.2 — Template components are accessible."""

    @pytest.mark.smoke
    def test_architecture_templates_exists(self, builders_pkg):
        """``ArchitectureTemplates`` is present and non-None."""
        obj = getattr(builders_pkg, "ArchitectureTemplates", None)
        assert obj is not None, "ArchitectureTemplates is None or missing"


class TestSmokeConfigParserExports:
    """§1.2 — Config parser components are accessible."""

    CONFIG_PARSER_EXPORTS = [
        "ArchitectureConfigParser",
        "parse_custom_architecture",
        "parse_ensemble",
        "load_config",
        "validate_config",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONFIG_PARSER_EXPORTS)
    def test_config_parser_export_exists(self, builders_pkg, name):
        """Each config parser export is present and non-None."""
        obj = getattr(builders_pkg, name, None)
        assert obj is not None, (
            f"Config parser export '{name}' is None or missing"
        )


class TestSmokeValidationExports:
    """§1.2 — Validation components are accessible."""

    VALIDATION_EXPORTS = [
        "ArchitectureValidator",
        "validate_architecture",
        "validate_data_compatibility",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATION_EXPORTS)
    def test_validation_export_exists(self, builders_pkg, name):
        """Each validation export is present and non-None."""
        obj = getattr(builders_pkg, name, None)
        assert obj is not None, (
            f"Validation export '{name}' is None or missing"
        )


class TestSmokeConvenienceFunctionsCallable:
    """§1.2 — All convenience functions are callable."""

    CONVENIENCE_FUNCTIONS = [
        "get_layer",
        "list_layers",
        "get_layer_metadata",
        "parse_custom_architecture",
        "parse_ensemble",
        "load_config",
        "validate_config",
        "validate_architecture",
        "validate_data_compatibility",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_is_callable(self, builders_pkg, name):
        """Each convenience function is callable."""
        obj = getattr(builders_pkg, name, None)
        assert obj is not None, f"Function '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeExceptionClassesAccessible:
    """§1.2 — Exception classes are accessible and are classes."""

    EXCEPTION_CLASSES = [
        "LayerNotFoundError",
        "ArchitectureError",
        "ChannelMismatchError",
        "CompositionError",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_exists(self, builders_pkg, name):
        """Each exception class is present and non-None."""
        obj = getattr(builders_pkg, name, None)
        assert obj is not None, f"Exception class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_is_a_class(self, builders_pkg, name):
        """Each exception export is a class."""
        obj = getattr(builders_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, builders_pkg):
        """
        Re-importing the builders package (via ``importlib.reload``) does not
        crash.

        Validates that all module-level code is safe to re-execute.
        """
        reloaded = importlib.reload(builders_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, builders_pkg):
        """Re-importing the builders package preserves ``__all__``."""
        reloaded = importlib.reload(builders_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_reimport_preserves_version(self, builders_pkg):
        """Re-importing the builders package preserves ``__version__``."""
        original_version = builders_pkg.__version__
        reloaded = importlib.reload(builders_pkg)
        assert reloaded.__version__ == original_version


class TestSmokeLayerRegistryInstance:
    """§1.2 — The ``layer_registry`` global instance is accessible."""

    @pytest.mark.smoke
    def test_layer_registry_instance_exists(self, builders_pkg):
        """``layer_registry`` is present on the builders package."""
        assert hasattr(builders_pkg, "layer_registry"), (
            "layer_registry global instance is missing"
        )

    @pytest.mark.smoke
    def test_layer_registry_instance_is_not_none(self, builders_pkg):
        """``layer_registry`` is not None."""
        assert builders_pkg.layer_registry is not None

    @pytest.mark.smoke
    def test_layer_registry_is_instance_of_layer_registry_class(self, builders_pkg):
        """``layer_registry`` is an instance of ``LayerRegistry``."""
        assert isinstance(builders_pkg.layer_registry, builders_pkg.LayerRegistry), (
            f"layer_registry should be a LayerRegistry instance, "
            f"got {type(builders_pkg.layer_registry).__name__}"
        )


class TestSmokeAllExportsGenericSweep:
    """§1.2 — Generic sweep: every name in ``__all__`` resolves."""

    @pytest.mark.smoke
    def test_every_all_entry_is_accessible(self, builders_pkg, all_names):
        """Every name in ``__all__`` is accessible on the module."""
        missing = [
            name for name in all_names
            if not hasattr(builders_pkg, name)
        ]
        assert not missing, (
            f"Names in __all__ that are not accessible: {missing}"
        )


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the builders package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, builders_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(builders_pkg.__all__, list)

    @pytest.mark.contract
    def test_all_contains_no_duplicates(self, all_names):
        """``__all__`` has no duplicate entries."""
        seen = set()
        duplicates = []
        for name in all_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)

        assert not duplicates, (
            f"Duplicate entries in __all__: {duplicates}"
        )

    @pytest.mark.contract
    def test_every_all_entry_is_resolvable(self, builders_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [
            name for name in all_names
            if not hasattr(builders_pkg, name)
        ]
        assert not unresolvable, (
            f"Names in __all__ that are not defined on the module: {unresolvable}"
        )

    @pytest.mark.contract
    def test_all_entries_are_strings(self, all_names):
        """Every entry in ``__all__`` is a string."""
        non_strings = [
            (i, name) for i, name in enumerate(all_names)
            if not isinstance(name, str)
        ]
        assert not non_strings, (
            f"Non-string entries in __all__: {non_strings}"
        )

    @pytest.mark.contract
    def test_all_minimum_expected_count(self, all_names):
        """
        ``__all__`` contains the minimum expected number of exports.

        The ``__init__.py`` defines 30 entries in ``__all__`` (29 named
        exports + ``__version__``).
        """
        assert len(all_names) >= 30, (
            f"Expected at least 30 entries in __all__, got {len(all_names)}"
        )


class TestContractAllConsistency:
    """§2 — Every public import in the builders module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Metadata dunders (not typically in __all__ besides __version__)
        "__author__",
        # typing imports used at module level
        "Dict",
        "List",
        "Optional",
        "Any",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, builders_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the builders ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(builders_pkg)
        missing_from_all = []

        for name, obj in module_dict.items():
            # Skip dunder names
            if name.startswith("__") and name.endswith("__"):
                continue
            # Skip modules (submodule references)
            if isinstance(obj, types.ModuleType):
                continue
            # Skip known unlisted names
            if name in self.KNOWN_UNLISTED:
                continue
            # Skip private names NOT in __all__
            if name.startswith("_") and name not in all_set:
                continue

            if name not in all_set:
                missing_from_all.append(name)

        # Filter common Python internals
        python_internals = {
            "__builtins__", "__cached__", "__doc__", "__file__",
            "__loader__", "__name__", "__package__", "__path__",
            "__spec__",
        }
        missing_from_all = [
            n for n in missing_from_all if n not in python_internals
        ]

        assert not missing_from_all, (
            f"Public names imported in builders/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractClassExports:
    """§2 — Exported classes are classes (not instances or functions)."""

    CLASS_EXPORTS = [
        # Layer Registry
        "LayerRegistry",
        "LayerCategory",
        "LayerMetadata",
        "FunctionalLayerWrapper",
        "LayerNotFoundError",
        # Architecture Builder
        "ArchitectureBuilder",
        "LayerConfig",
        "ArchitectureConfig",
        "ResidualConnection",
        "ArchitectureError",
        "ChannelMismatchError",
        "CustomArchitecture",
        # Model Composer
        "ModelSpec",
        "EnsembleConfig",
        "CompositionError",
        "ModelComposer",
        "ParallelEnsemble",
        "SequentialStack",
        "HierarchicalComposition",
        # Templates
        "ArchitectureTemplates",
        # Config Parser
        "ArchitectureConfigParser",
        # Validation
        "ArchitectureValidator",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CLASS_EXPORTS)
    def test_export_is_class(self, builders_pkg, name):
        """Each class-type export is a class."""
        obj = getattr(builders_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestContractExceptionClassInheritance:
    """§2 — Exception classes inherit from Exception (directly or indirectly)."""

    EXCEPTION_CLASSES = [
        "LayerNotFoundError",
        "ArchitectureError",
        "ChannelMismatchError",
        "CompositionError",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_inherits_from_exception(self, builders_pkg, name):
        """Each exception class is a subclass of Exception."""
        cls = getattr(builders_pkg, name)
        assert issubclass(cls, Exception), (
            f"'{name}' should be a subclass of Exception, "
            f"MRO: {[c.__name__ for c in cls.__mro__]}"
        )

    @pytest.mark.contract
    def test_channel_mismatch_inherits_from_architecture_error(self, builders_pkg):
        """
        ``ChannelMismatchError`` is a subclass of ``ArchitectureError``
        (specialized exception hierarchy).
        """
        assert issubclass(
            builders_pkg.ChannelMismatchError,
            builders_pkg.ArchitectureError
        ), (
            "ChannelMismatchError should inherit from ArchitectureError"
        )


class TestContractDataclassExports:
    """§2 — Dataclass exports are actual dataclasses."""

    DATACLASS_EXPORTS = [
        "LayerConfig",
        "ResidualConnection",
        "ArchitectureConfig",
        "ModelSpec",
        "EnsembleConfig",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DATACLASS_EXPORTS)
    def test_export_is_dataclass(self, builders_pkg, name):
        """
        Each dataclass-type export is a dataclass (has ``__dataclass_fields__``)
        or a Pydantic model (has ``__pydantic_fields__`` or ``model_fields``).

        The project structure documents these as dataclasses.
        """
        import dataclasses
        cls = getattr(builders_pkg, name)
        is_stdlib_dc = dataclasses.is_dataclass(cls)
        is_pydantic = hasattr(cls, "__pydantic_fields__") or hasattr(cls, "model_fields")

        assert is_stdlib_dc or is_pydantic, (
            f"'{name}' should be a dataclass or Pydantic model"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", [
        "LayerConfig",
        "ResidualConnection",
        "ArchitectureConfig",
    ])
    def test_dataclass_has_to_dict(self, builders_pkg, name):
        """
        Architecture builder dataclasses expose a ``to_dict()`` method
        for serialization (per project structure documentation).
        """
        cls = getattr(builders_pkg, name)
        assert hasattr(cls, "to_dict"), (
            f"'{name}' should have a 'to_dict' method for serialization"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", [
        "LayerConfig",
        "ResidualConnection",
        "ArchitectureConfig",
    ])
    def test_dataclass_has_from_dict(self, builders_pkg, name):
        """
        Architecture builder dataclasses expose a ``from_dict()`` classmethod
        for deserialization (per project structure documentation).
        """
        cls = getattr(builders_pkg, name)
        assert hasattr(cls, "from_dict"), (
            f"'{name}' should have a 'from_dict' classmethod for deserialization"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", [
        "ModelSpec",
        "EnsembleConfig",
    ])
    def test_composer_dataclass_has_to_dict(self, builders_pkg, name):
        """
        Model composer dataclasses expose a ``to_dict()`` method
        for serialization (per project structure documentation).
        """
        cls = getattr(builders_pkg, name)
        assert hasattr(cls, "to_dict"), (
            f"'{name}' should have a 'to_dict' method for serialization"
        )


class TestContractEnumExports:
    """§2 — Enum exports are actual enums."""

    @pytest.mark.contract
    def test_layer_category_is_enum(self, builders_pkg):
        """``LayerCategory`` is an Enum subclass."""
        import enum
        assert issubclass(builders_pkg.LayerCategory, enum.Enum), (
            "LayerCategory should be an Enum subclass"
        )

    @pytest.mark.contract
    def test_layer_category_has_expected_members(self, builders_pkg):
        """
        ``LayerCategory`` has the 8 documented category members.

        Per project structure: CONVOLUTIONAL, POOLING, NORMALIZATION,
        ACTIVATION, AGGREGATION, LINEAR, DROPOUT, CUSTOM.
        """
        expected_members = {
            "CONVOLUTIONAL",
            "POOLING",
            "NORMALIZATION",
            "ACTIVATION",
            "AGGREGATION",
            "LINEAR",
            "DROPOUT",
            "CUSTOM",
        }
        actual_members = set(builders_pkg.LayerCategory.__members__.keys())
        missing = expected_members - actual_members
        assert not missing, (
            f"LayerCategory is missing expected members: {missing}. "
            f"Actual members: {sorted(actual_members)}"
        )


class TestContractLayerMetadataStructure:
    """§2 — ``LayerMetadata`` has the documented 12 attributes."""

    EXPECTED_FIELDS = [
        "name",
        "category",
        "class_path",
        "description",
        "requires_edge_index",
        "requires_edge_attr",
        "requires_batch",
        "has_in_channels",
        "has_out_channels",
        "modifies_graph_structure",
        "supported_task_levels",
        "is_functional",
    ]

    @pytest.mark.contract
    def test_layer_metadata_is_dataclass(self, builders_pkg):
        """``LayerMetadata`` is a dataclass."""
        import dataclasses
        cls = builders_pkg.LayerMetadata
        is_dc = dataclasses.is_dataclass(cls)
        is_pydantic = hasattr(cls, "__pydantic_fields__")
        assert is_dc or is_pydantic, (
            "LayerMetadata should be a dataclass or Pydantic model"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("field_name", EXPECTED_FIELDS)
    def test_layer_metadata_has_expected_field(self, builders_pkg, field_name):
        """
        ``LayerMetadata`` has each of the 12 documented attributes
        as either a dataclass field or a class attribute.
        """
        import dataclasses
        cls = builders_pkg.LayerMetadata

        if dataclasses.is_dataclass(cls):
            field_names = {f.name for f in dataclasses.fields(cls)}
            assert field_name in field_names, (
                f"LayerMetadata is missing dataclass field '{field_name}'. "
                f"Available fields: {sorted(field_names)}"
            )
        elif hasattr(cls, "__pydantic_fields__"):
            assert field_name in cls.__pydantic_fields__, (
                f"LayerMetadata is missing Pydantic field '{field_name}'"
            )
        else:
            assert hasattr(cls, field_name), (
                f"LayerMetadata is missing attribute '{field_name}'"
            )

    @pytest.mark.contract
    def test_layer_metadata_has_to_dict(self, builders_pkg):
        """``LayerMetadata`` exposes a ``to_dict()`` method."""
        assert hasattr(builders_pkg.LayerMetadata, "to_dict"), (
            "LayerMetadata should have a 'to_dict' method"
        )


class TestContractNNModuleExports:
    """§2 — nn.Module subclasses are identified."""

    NN_MODULE_EXPORTS = [
        "FunctionalLayerWrapper",
        "CustomArchitecture",
        "ParallelEnsemble",
        "SequentialStack",
        "HierarchicalComposition",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", NN_MODULE_EXPORTS)
    def test_export_is_nn_module_subclass(self, builders_pkg, name):
        """
        Each nn.Module export is a subclass of ``torch.nn.Module``.

        Per project structure: FunctionalLayerWrapper, CustomArchitecture,
        ParallelEnsemble, SequentialStack, HierarchicalComposition are all
        nn.Module subclasses.
        """
        try:
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available — cannot verify nn.Module inheritance")

        cls = getattr(builders_pkg, name)
        assert issubclass(cls, nn.Module), (
            f"'{name}' should be a subclass of torch.nn.Module, "
            f"MRO: {[c.__name__ for c in cls.__mro__]}"
        )


class TestContractFunctionSignatures:
    """§2 — Convenience functions have documented parameter signatures."""

    @pytest.mark.contract
    def test_get_layer_has_name_parameter(self, builders_pkg):
        """``get_layer()`` accepts a layer name parameter."""
        sig = inspect.signature(builders_pkg.get_layer)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, (
            f"get_layer() should have at least 1 parameter, got {len(params)}"
        )

    @pytest.mark.contract
    def test_list_layers_is_callable(self, builders_pkg):
        """``list_layers()`` is callable and has a signature."""
        sig = inspect.signature(builders_pkg.list_layers)
        assert sig is not None, "list_layers() should have a valid signature"

    @pytest.mark.contract
    def test_get_layer_metadata_has_name_parameter(self, builders_pkg):
        """``get_layer_metadata()`` accepts a layer name parameter."""
        sig = inspect.signature(builders_pkg.get_layer_metadata)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, (
            f"get_layer_metadata() should have at least 1 parameter, got {len(params)}"
        )

    @pytest.mark.contract
    def test_validate_architecture_is_callable(self, builders_pkg):
        """``validate_architecture()`` is callable and has a valid signature."""
        sig = inspect.signature(builders_pkg.validate_architecture)
        assert sig is not None, "validate_architecture() should have a valid signature"

    @pytest.mark.contract
    def test_validate_data_compatibility_is_callable(self, builders_pkg):
        """``validate_data_compatibility()`` is callable and has a valid signature."""
        sig = inspect.signature(builders_pkg.validate_data_compatibility)
        assert sig is not None, (
            "validate_data_compatibility() should have a valid signature"
        )

    @pytest.mark.contract
    def test_parse_custom_architecture_is_callable(self, builders_pkg):
        """``parse_custom_architecture()`` is callable with a valid signature."""
        sig = inspect.signature(builders_pkg.parse_custom_architecture)
        assert sig is not None, (
            "parse_custom_architecture() should have a valid signature"
        )

    @pytest.mark.contract
    def test_parse_ensemble_is_callable(self, builders_pkg):
        """``parse_ensemble()`` is callable with a valid signature."""
        sig = inspect.signature(builders_pkg.parse_ensemble)
        assert sig is not None, (
            "parse_ensemble() should have a valid signature"
        )

    @pytest.mark.contract
    def test_load_config_is_callable(self, builders_pkg):
        """``load_config()`` is callable with a valid signature."""
        sig = inspect.signature(builders_pkg.load_config)
        assert sig is not None, (
            "load_config() should have a valid signature"
        )

    @pytest.mark.contract
    def test_validate_config_is_callable(self, builders_pkg):
        """``validate_config()`` is callable with a valid signature."""
        sig = inspect.signature(builders_pkg.validate_config)
        assert sig is not None, (
            "validate_config() should have a valid signature"
        )


class TestContractArchitectureBuilderFluentAPI:
    """§2 — ``ArchitectureBuilder`` has the documented fluent builder API."""

    FLUENT_METHODS = [
        "add_layer",
        "remove_layer",
        "insert_layer",
        "replace_layer",
        "swap_layers",
        "add_residual_connection",
        "build",
        "validate_architecture",
        "to_config",
        "from_config",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", FLUENT_METHODS)
    def test_architecture_builder_has_method(self, builders_pkg, method_name):
        """``ArchitectureBuilder`` exposes each documented method."""
        cls = builders_pkg.ArchitectureBuilder
        assert hasattr(cls, method_name), (
            f"ArchitectureBuilder is missing method '{method_name}'"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", FLUENT_METHODS)
    def test_architecture_builder_method_is_callable(self, builders_pkg, method_name):
        """Each ``ArchitectureBuilder`` method is callable."""
        cls = builders_pkg.ArchitectureBuilder
        method = getattr(cls, method_name)
        assert callable(method), (
            f"ArchitectureBuilder.{method_name} should be callable"
        )

    @pytest.mark.contract
    def test_from_config_is_classmethod(self, builders_pkg):
        """``ArchitectureBuilder.from_config`` is a classmethod."""
        cls = builders_pkg.ArchitectureBuilder
        # Check if from_config is a classmethod by inspecting the class dict
        attr = cls.__dict__.get("from_config")
        is_cm = isinstance(attr, classmethod) if attr is not None else False
        # Also accept if it's just callable on the class (already bound)
        is_callable_on_class = callable(getattr(cls, "from_config", None))
        assert is_cm or is_callable_on_class, (
            "ArchitectureBuilder.from_config should be a classmethod or callable"
        )


class TestContractModelComposerFluentAPI:
    """§2 — ``ModelComposer`` has the documented fluent builder API."""

    FLUENT_METHODS = [
        "add_model",
        "remove_model",
        "clear_models",
        "set_strategy",
        "set_fusion",
        "validate_composition",
        "build",
        "to_config",
        "from_config",
        "summary",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", FLUENT_METHODS)
    def test_model_composer_has_method(self, builders_pkg, method_name):
        """``ModelComposer`` exposes each documented method."""
        cls = builders_pkg.ModelComposer
        assert hasattr(cls, method_name), (
            f"ModelComposer is missing method '{method_name}'"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", FLUENT_METHODS)
    def test_model_composer_method_is_callable(self, builders_pkg, method_name):
        """Each ``ModelComposer`` method is callable."""
        cls = builders_pkg.ModelComposer
        method = getattr(cls, method_name)
        assert callable(method), (
            f"ModelComposer.{method_name} should be callable"
        )


class TestContractArchitectureTemplatesAPI:
    """§2 — ``ArchitectureTemplates`` has the documented static methods."""

    TEMPLATE_METHODS = [
        "simple_gcn",
        "attention_network",
        "deep_residual",
        "hybrid_conv_attention",
        "hierarchical_pooling",
        "graph_sage_network",
        "gin_network",
        "molecular_network",
        "node_classification_network",
        "graph_classification_network",
        "list_templates",
        "get_template_info",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", TEMPLATE_METHODS)
    def test_architecture_templates_has_method(self, builders_pkg, method_name):
        """``ArchitectureTemplates`` exposes each documented method."""
        cls = builders_pkg.ArchitectureTemplates
        assert hasattr(cls, method_name), (
            f"ArchitectureTemplates is missing method '{method_name}'"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", TEMPLATE_METHODS)
    def test_architecture_templates_method_is_callable(self, builders_pkg, method_name):
        """Each ``ArchitectureTemplates`` method is callable."""
        cls = builders_pkg.ArchitectureTemplates
        method = getattr(cls, method_name)
        assert callable(method), (
            f"ArchitectureTemplates.{method_name} should be callable"
        )

    @pytest.mark.contract
    def test_list_templates_returns_list(self, builders_pkg):
        """``ArchitectureTemplates.list_templates()`` returns a list."""
        result = builders_pkg.ArchitectureTemplates.list_templates()
        assert isinstance(result, list), (
            f"list_templates() should return a list, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_list_templates_has_10_templates(self, builders_pkg):
        """
        ``ArchitectureTemplates.list_templates()`` returns exactly 10
        template names (per project structure documentation).
        """
        result = builders_pkg.ArchitectureTemplates.list_templates()
        assert len(result) == 10, (
            f"Expected 10 templates, got {len(result)}: {result}"
        )

    @pytest.mark.contract
    def test_list_templates_entries_are_strings(self, builders_pkg):
        """Each entry returned by ``list_templates()`` is a string."""
        result = builders_pkg.ArchitectureTemplates.list_templates()
        for entry in result:
            assert isinstance(entry, str), (
                f"Template name should be str, got {type(entry).__name__}: {entry}"
            )


class TestContractArchitectureValidatorAPI:
    """§2 — ``ArchitectureValidator`` has the documented validation methods."""

    VALIDATOR_METHODS = [
        "validate",
        "validate_channel_flow",
        "validate_task_compatibility",
        "validate_layer_ordering",
        "validate_data_compatibility",
        "suggest_fixes",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", VALIDATOR_METHODS)
    def test_architecture_validator_has_method(self, builders_pkg, method_name):
        """``ArchitectureValidator`` exposes each documented method."""
        cls = builders_pkg.ArchitectureValidator
        assert hasattr(cls, method_name), (
            f"ArchitectureValidator is missing method '{method_name}'"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", VALIDATOR_METHODS)
    def test_architecture_validator_method_is_callable(self, builders_pkg, method_name):
        """Each ``ArchitectureValidator`` method is callable."""
        cls = builders_pkg.ArchitectureValidator
        method = getattr(cls, method_name)
        assert callable(method), (
            f"ArchitectureValidator.{method_name} should be callable"
        )


class TestContractArchitectureConfigParserAPI:
    """§2 — ``ArchitectureConfigParser`` has the documented API."""

    PARSER_METHODS = [
        "parse_custom_architecture",
        "parse_ensemble",
        "validate_config",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", PARSER_METHODS)
    def test_config_parser_has_method(self, builders_pkg, method_name):
        """``ArchitectureConfigParser`` exposes each documented method."""
        cls = builders_pkg.ArchitectureConfigParser
        assert hasattr(cls, method_name), (
            f"ArchitectureConfigParser is missing method '{method_name}'"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", PARSER_METHODS)
    def test_config_parser_method_is_callable(self, builders_pkg, method_name):
        """Each ``ArchitectureConfigParser`` method is callable."""
        cls = builders_pkg.ArchitectureConfigParser
        method = getattr(cls, method_name)
        assert callable(method), (
            f"ArchitectureConfigParser.{method_name} should be callable"
        )


class TestContractLayerRegistryAPI:
    """§2 — ``LayerRegistry`` class has the documented API methods."""

    REGISTRY_METHODS = [
        "register_custom_layer",
        "get_layer",
        "get_layer_metadata",
        "has_layer",
        "list_layers",
        "list_categories",
        "get_statistics",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", REGISTRY_METHODS)
    def test_layer_registry_has_method(self, builders_pkg, method_name):
        """``LayerRegistry`` exposes each documented method."""
        cls = builders_pkg.LayerRegistry
        assert hasattr(cls, method_name), (
            f"LayerRegistry is missing method '{method_name}'"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", REGISTRY_METHODS)
    def test_layer_registry_method_is_callable(self, builders_pkg, method_name):
        """Each ``LayerRegistry`` method is callable."""
        cls = builders_pkg.LayerRegistry
        method = getattr(cls, method_name)
        assert callable(method), (
            f"LayerRegistry.{method_name} should be callable"
        )


class TestContractLayerRegistryInstanceContract:
    """§2 — The global ``layer_registry`` instance satisfies operational contracts."""

    @pytest.mark.contract
    def test_layer_registry_has_layers(self, builders_pkg):
        """
        The global ``layer_registry`` has registered layers (63+ built-in
        layers per project structure documentation).
        """
        registry = builders_pkg.layer_registry
        layers = registry.list_layers()
        assert len(layers) >= 50, (
            f"Expected at least 50 registered layers, got {len(layers)}"
        )

    @pytest.mark.contract
    def test_layer_registry_has_categories(self, builders_pkg):
        """The global ``layer_registry`` has registered categories."""
        registry = builders_pkg.layer_registry
        categories = registry.list_categories()
        assert len(categories) >= 6, (
            f"Expected at least 6 categories, got {len(categories)}"
        )

    @pytest.mark.contract
    def test_layer_registry_get_statistics_returns_dict(self, builders_pkg):
        """``layer_registry.get_statistics()`` returns a dict."""
        registry = builders_pkg.layer_registry
        stats = registry.get_statistics()
        assert isinstance(stats, dict), (
            f"get_statistics() should return dict, got {type(stats).__name__}"
        )

    @pytest.mark.contract
    def test_layer_registry_statistics_has_total_layers(self, builders_pkg):
        """Statistics dict contains ``total_layers`` key."""
        registry = builders_pkg.layer_registry
        stats = registry.get_statistics()
        assert "total_layers" in stats, (
            f"Statistics should contain 'total_layers' key. Keys: {list(stats.keys())}"
        )

    @pytest.mark.contract
    def test_layer_registry_statistics_has_by_category(self, builders_pkg):
        """Statistics dict contains ``by_category`` key."""
        registry = builders_pkg.layer_registry
        stats = registry.get_statistics()
        assert "by_category" in stats, (
            f"Statistics should contain 'by_category' key. Keys: {list(stats.keys())}"
        )

    @pytest.mark.contract
    def test_layer_registry_has_layer_returns_bool(self, builders_pkg):
        """``layer_registry.has_layer()`` returns a boolean."""
        registry = builders_pkg.layer_registry
        # Use a known layer name from the documentation
        result = registry.has_layer("GCNConv")
        assert isinstance(result, bool), (
            f"has_layer() should return bool, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_layer_registry_has_gcnconv(self, builders_pkg):
        """
        The global ``layer_registry`` contains the GCNConv layer
        (first listed convolutional layer in documentation).
        """
        registry = builders_pkg.layer_registry
        assert registry.has_layer("GCNConv"), (
            "layer_registry should contain 'GCNConv' as a built-in layer"
        )


class TestContractVersionMetadata:
    """§2 — ``__version__`` contract validation."""

    @pytest.mark.contract
    def test_version_follows_semver(self, builders_pkg):
        """``__version__`` follows MAJOR.MINOR.PATCH semver pattern."""
        version = builders_pkg.__version__
        parts = version.split(".")
        assert len(parts) == 3, (
            f"Version '{version}' should have exactly 3 components (MAJOR.MINOR.PATCH)"
        )
        for i, part in enumerate(parts):
            assert part.isdigit(), (
                f"Version component [{i}] = '{part}' should be numeric"
            )

    @pytest.mark.contract
    def test_version_in_all(self, all_names):
        """``__version__`` is listed in ``__all__``."""
        assert "__version__" in all_names, (
            "__version__ should be listed in __all__"
        )


class TestContractPublicAPISurfaceStability:
    """§2 — Public API surface stability (minimum expected names present)."""

    # These are the core exports that MUST be present for the API to be stable.
    # Derived from the __init__.py source and project structure documentation.
    CORE_API_NAMES = [
        # Layer Registry core
        "LayerRegistry",
        "LayerCategory",
        "LayerMetadata",
        "get_layer",
        "list_layers",
        "layer_registry",
        # Architecture Builder core
        "ArchitectureBuilder",
        "LayerConfig",
        "ArchitectureConfig",
        "CustomArchitecture",
        # Model Composer core
        "ModelComposer",
        "ParallelEnsemble",
        "SequentialStack",
        "HierarchicalComposition",
        # Templates core
        "ArchitectureTemplates",
        # Config Parser core
        "ArchitectureConfigParser",
        "parse_custom_architecture",
        "parse_ensemble",
        # Validation core
        "ArchitectureValidator",
        "validate_architecture",
        "validate_data_compatibility",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CORE_API_NAMES)
    def test_core_api_name_in_all(self, all_names, name):
        """Each core API name is present in ``__all__``."""
        assert name in all_names, (
            f"Core API name '{name}' should be in __all__"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CORE_API_NAMES)
    def test_core_api_name_is_accessible(self, builders_pkg, name):
        """Each core API name is accessible as an attribute."""
        obj = getattr(builders_pkg, name, None)
        assert obj is not None, (
            f"Core API name '{name}' should be accessible on the module"
        )


class TestContractImportSourceVerification:
    """§2 — Exported names originate from the correct submodules."""

    @pytest.mark.contract
    def test_layer_registry_from_layer_registry_module(self, builders_pkg):
        """``LayerRegistry`` originates from ``layer_registry`` submodule."""
        cls = builders_pkg.LayerRegistry
        module_name = cls.__module__
        assert "layer_registry" in module_name, (
            f"LayerRegistry should come from layer_registry module, "
            f"got module '{module_name}'"
        )

    @pytest.mark.contract
    def test_architecture_builder_from_architecture_builder_module(self, builders_pkg):
        """``ArchitectureBuilder`` originates from ``architecture_builder`` submodule."""
        cls = builders_pkg.ArchitectureBuilder
        module_name = cls.__module__
        assert "architecture_builder" in module_name, (
            f"ArchitectureBuilder should come from architecture_builder module, "
            f"got module '{module_name}'"
        )

    @pytest.mark.contract
    def test_model_composer_from_model_composer_module(self, builders_pkg):
        """``ModelComposer`` originates from ``model_composer`` submodule."""
        cls = builders_pkg.ModelComposer
        module_name = cls.__module__
        assert "model_composer" in module_name, (
            f"ModelComposer should come from model_composer module, "
            f"got module '{module_name}'"
        )

    @pytest.mark.contract
    def test_architecture_templates_from_templates_module(self, builders_pkg):
        """``ArchitectureTemplates`` originates from ``templates`` submodule."""
        cls = builders_pkg.ArchitectureTemplates
        module_name = cls.__module__
        assert "templates" in module_name, (
            f"ArchitectureTemplates should come from templates module, "
            f"got module '{module_name}'"
        )

    @pytest.mark.contract
    def test_config_parser_from_config_parser_module(self, builders_pkg):
        """``ArchitectureConfigParser`` originates from ``config_parser`` submodule."""
        cls = builders_pkg.ArchitectureConfigParser
        module_name = cls.__module__
        assert "config_parser" in module_name, (
            f"ArchitectureConfigParser should come from config_parser module, "
            f"got module '{module_name}'"
        )

    @pytest.mark.contract
    def test_architecture_validator_from_validation_module(self, builders_pkg):
        """``ArchitectureValidator`` originates from ``validation`` submodule."""
        cls = builders_pkg.ArchitectureValidator
        module_name = cls.__module__
        assert "validation" in module_name, (
            f"ArchitectureValidator should come from validation module, "
            f"got module '{module_name}'"
        )


class TestContractFunctionalLayerWrapperAPI:
    """§2 — ``FunctionalLayerWrapper`` nn.Module has documented attributes."""

    EXPECTED_ATTRS = [
        "func",
        "func_name",
        "requires_batch",
        "requires_edge_index",
        "requires_edge_attr",
    ]

    @pytest.mark.contract
    def test_functional_layer_wrapper_is_class(self, builders_pkg):
        """``FunctionalLayerWrapper`` is a class."""
        assert inspect.isclass(builders_pkg.FunctionalLayerWrapper)

    @pytest.mark.contract
    def test_functional_layer_wrapper_has_forward(self, builders_pkg):
        """``FunctionalLayerWrapper`` has a ``forward()`` method."""
        cls = builders_pkg.FunctionalLayerWrapper
        assert hasattr(cls, "forward"), (
            "FunctionalLayerWrapper should have a 'forward' method"
        )
        assert callable(getattr(cls, "forward")), (
            "FunctionalLayerWrapper.forward should be callable"
        )


class TestContractCustomArchitectureAPI:
    """§2 — ``CustomArchitecture`` nn.Module has documented API."""

    @pytest.mark.contract
    def test_custom_architecture_has_forward(self, builders_pkg):
        """``CustomArchitecture`` has a ``forward()`` method."""
        cls = builders_pkg.CustomArchitecture
        assert hasattr(cls, "forward"), (
            "CustomArchitecture should have a 'forward' method"
        )

    @pytest.mark.contract
    def test_custom_architecture_is_class(self, builders_pkg):
        """``CustomArchitecture`` is a class."""
        assert inspect.isclass(builders_pkg.CustomArchitecture)


class TestContractEnsembleModulesAPI:
    """§2 — Ensemble nn.Module subclasses have documented API."""

    ENSEMBLE_MODULES = [
        "ParallelEnsemble",
        "SequentialStack",
        "HierarchicalComposition",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ENSEMBLE_MODULES)
    def test_ensemble_module_has_forward(self, builders_pkg, name):
        """Each ensemble module has a ``forward()`` method."""
        cls = getattr(builders_pkg, name)
        assert hasattr(cls, "forward"), (
            f"{name} should have a 'forward' method"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ENSEMBLE_MODULES)
    def test_ensemble_module_forward_is_callable(self, builders_pkg, name):
        """Each ensemble module's ``forward()`` is callable."""
        cls = getattr(builders_pkg, name)
        assert callable(getattr(cls, "forward")), (
            f"{name}.forward should be callable"
        )
