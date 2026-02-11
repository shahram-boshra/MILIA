# tests/test__init__molecules.py

"""
Test Suite: milia_pipeline/molecules/__init__.py — Smoke Tests & Contract Tests
================================================================================

Production-ready test suite for the MILIA Pipeline molecules package
``milia_pipeline/molecules/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.molecules`` subpackage imports without ImportError
        - All re-exported names from the 6 submodules are accessible
        - Module-level metadata attributes (__version__) exist and are typed
        - Module initialization (logging, registry status checks) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Registry integration status flags are present and correctly typed
        - Phase 6 registry status logging blocks execute without exceptions
        - Core classes (MoleculeDataConverter, MoleculeFilter) are importable
        - Conversion, validation, enrichment, and filtering exports are accessible
        - Handler-only operation functions are accessible and callable
        - Phase 6 diagnostics functions are accessible

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, callables are callable
        - Core classes (MoleculeDataConverter, MoleculeFilter) are classes
        - Conversion functions are callable
        - Validation functions are callable
        - Enrichment functions are callable
        - Handler-only functions are callable
        - Filtering factory functions are callable
        - Filtering core functions are callable
        - Filtering validation functions are callable
        - Filtering utility functions are callable
        - Phase 6 registry diagnostics functions return dicts
        - Phase 6 registry integration functions are callable
        - ``get_filter_registry_status()`` returns dict with documented keys
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)
        - Registry flags are correctly typed (bool or str|None)
        - Phase-aliased imports do not collide (renamed with prefixes)
        - Namespace contains expected logger attribute
        - Feature function ``get_available_features`` is callable

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__molecules.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import importlib
import inspect
import sys
import types
import logging
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__molecules.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def mol_pkg():
    """
    Import and return the ``milia_pipeline.molecules`` package once per module.

    This fixture validates the fundamental smoke invariant: the molecules
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.molecules as mol
        return mol
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.molecules could not be imported — smoke test "
            f"precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(mol_pkg):
    """Return the ``__all__`` list from the molecules package."""
    assert hasattr(mol_pkg, "__all__"), (
        "milia_pipeline.molecules.__all__ is missing — contract violation"
    )
    return list(mol_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeMoleculesPackageImport:
    """§1.2 — Verify the molecules subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_molecules_package_succeeds(self, mol_pkg):
        """The molecules package imports without raising any exception."""
        assert mol_pkg is not None

    @pytest.mark.smoke
    def test_molecules_package_is_a_module(self, mol_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(mol_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_molecules_package_has_file_attribute(self, mol_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(mol_pkg, "__file__")

    @pytest.mark.smoke
    def test_molecules_package_name(self, mol_pkg):
        """The package ``__name__`` is ``milia_pipeline.molecules``."""
        assert mol_pkg.__name__ == "milia_pipeline.molecules"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    def test_version_attribute_exists(self, mol_pkg):
        """``__version__`` is defined on the molecules package."""
        assert hasattr(mol_pkg, "__version__"), "Missing attribute: __version__"

    @pytest.mark.smoke
    def test_version_attribute_is_string(self, mol_pkg):
        """``__version__`` is a non-empty string."""
        value = mol_pkg.__version__
        assert isinstance(value, str), (
            f"__version__ should be str, got {type(value)}"
        )
        assert len(value) > 0, "__version__ should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, mol_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = mol_pkg.__version__
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


class TestSmokeCoreClassExports:
    """§1.2 — Core classes are accessible from the molecules package."""

    CORE_CLASSES = [
        "MoleculeDataConverter",
        "MoleculeFilter",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_CLASSES)
    def test_core_class_exists(self, mol_pkg, name):
        """Each core class is importable from the molecules package."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, f"Core class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_CLASSES)
    def test_core_class_is_a_class(self, mol_pkg, name):
        """Each core class export is a class (not an instance or function)."""
        obj = getattr(mol_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeConversionFunctionExports:
    """§1.2 — Core conversion function exports are accessible."""

    CONVERSION_FUNCTIONS = [
        "create_rdkit_mol",
        "mol_to_pyg_data",
        "create_mol_with_dataset_support",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVERSION_FUNCTIONS)
    def test_conversion_function_exists(self, mol_pkg, name):
        """Each conversion function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, f"Conversion function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVERSION_FUNCTIONS)
    def test_conversion_function_is_callable(self, mol_pkg, name):
        """Each conversion function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeFeatureFunctionExports:
    """§1.2 — Structural feature function exports are accessible."""

    FEATURE_FUNCTIONS = [
        "add_structural_features",
        "get_available_features",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FEATURE_FUNCTIONS)
    def test_feature_function_exists(self, mol_pkg, name):
        """Each structural feature function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, f"Feature function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FEATURE_FUNCTIONS)
    def test_feature_function_is_callable(self, mol_pkg, name):
        """Each structural feature function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeValidationFunctionExports:
    """§1.2 — Validation function exports are accessible."""

    VALIDATION_FUNCTIONS = [
        "validate_molecular_structure",
        "check_dataset_compatibility",
        "validate_pyg_data_completeness",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATION_FUNCTIONS)
    def test_validation_function_exists(self, mol_pkg, name):
        """Each validation function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, f"Validation function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATION_FUNCTIONS)
    def test_validation_function_is_callable(self, mol_pkg, name):
        """Each validation function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeEnrichmentFunctionExports:
    """§1.2 — Enrichment function exports are accessible."""

    ENRICHMENT_FUNCTIONS = [
        "enrich_pyg_data_with_properties",
        "calculate_atomization_energy",
        "estimate_molecular_properties",
        "get_molecule_identifiers",
        "get_structural_feature_summary",
        "get_feature_extraction_diagnostics",
        "analyze_structural_feature_capabilities",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ENRICHMENT_FUNCTIONS)
    def test_enrichment_function_exists(self, mol_pkg, name):
        """Each enrichment function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, f"Enrichment function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ENRICHMENT_FUNCTIONS)
    def test_enrichment_function_is_callable(self, mol_pkg, name):
        """Each enrichment function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeHandlerOnlyFunctionExports:
    """§1.2 — Handler-only operation function exports are accessible."""

    HANDLER_ONLY_FUNCTIONS = [
        "estimate_properties_with_handler",
        "analyze_capabilities_with_handler",
        "create_handler_compatible_fingerprint",
        "validate_feature_extraction_with_handler",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", HANDLER_ONLY_FUNCTIONS)
    def test_handler_only_function_exists(self, mol_pkg, name):
        """Each handler-only function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, (
            f"Handler-only function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", HANDLER_ONLY_FUNCTIONS)
    def test_handler_only_function_is_callable(self, mol_pkg, name):
        """Each handler-only function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeFilteringFactoryExports:
    """§1.2 — Filtering factory function exports are accessible."""

    FACTORY_FUNCTIONS = [
        "create_molecule_filter",
        "get_default_molecule_filter",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FACTORY_FUNCTIONS)
    def test_factory_function_exists(self, mol_pkg, name):
        """Each filtering factory function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, f"Factory function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FACTORY_FUNCTIONS)
    def test_factory_function_is_callable(self, mol_pkg, name):
        """Each filtering factory function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeFilteringCoreFunctionExports:
    """§1.2 — Filtering core function exports are accessible."""

    CORE_FILTER_FUNCTIONS = [
        "apply_pre_filters",
        "apply_atom_count_filters",
        "apply_heavy_atom_filters",
        "apply_dataset_specific_filters",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_FILTER_FUNCTIONS)
    def test_core_filter_function_exists(self, mol_pkg, name):
        """Each core filtering function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, f"Core filter function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_FILTER_FUNCTIONS)
    def test_core_filter_function_is_callable(self, mol_pkg, name):
        """Each core filtering function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeFilteringValidationExports:
    """§1.2 — Filtering validation and utility function exports are accessible."""

    FILTER_VALIDATION_FUNCTIONS = [
        "validate_filter_configuration",
        "validate_filter_compatibility_with_transforms",
    ]

    FILTER_UTILITY_FUNCTIONS = [
        "introspect_transform_filter_parameters",
        "create_handler_aware_filter_stats",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FILTER_VALIDATION_FUNCTIONS + FILTER_UTILITY_FUNCTIONS)
    def test_filter_validation_utility_exists(self, mol_pkg, name):
        """Each filtering validation/utility function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, (
            f"Filter validation/utility '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FILTER_VALIDATION_FUNCTIONS + FILTER_UTILITY_FUNCTIONS)
    def test_filter_validation_utility_is_callable(self, mol_pkg, name):
        """Each filtering validation/utility function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeDiagnosticsFunctionExports:
    """§1.2 — Phase 6 diagnostics function exports are accessible."""

    DIAGNOSTICS_FUNCTIONS = [
        "get_registry_integration_status",
        "get_enricher_registry_status",
        "get_validator_registry_status",
        "get_filter_registry_status",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DIAGNOSTICS_FUNCTIONS)
    def test_diagnostics_function_exists(self, mol_pkg, name):
        """Each diagnostics function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, (
            f"Diagnostics function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DIAGNOSTICS_FUNCTIONS)
    def test_diagnostics_function_is_callable(self, mol_pkg, name):
        """Each diagnostics function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeRegistryStatusFlags:
    """§1.2 — Registry integration status flags exist and are correctly typed."""

    BOOL_FLAGS = [
        "_FILTER_REGISTRY_INITIALIZED",
        "_FILTER_REGISTRY_AVAILABLE",
    ]

    ERROR_FLAGS = [
        "_FILTER_REGISTRY_IMPORT_ERROR",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", BOOL_FLAGS)
    def test_registry_bool_flag_exists(self, mol_pkg, flag):
        """Each registry boolean flag is defined on the molecules package."""
        assert hasattr(mol_pkg, flag), f"Flag '{flag}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", BOOL_FLAGS)
    def test_registry_bool_flag_is_bool(self, mol_pkg, flag):
        """Each registry boolean status flag is actually a bool."""
        value = getattr(mol_pkg, flag)
        assert isinstance(value, bool), (
            f"Flag '{flag}' should be bool, got {type(value).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", ERROR_FLAGS)
    def test_registry_import_error_flag_exists(self, mol_pkg, flag):
        """Each registry import error flag is defined."""
        assert hasattr(mol_pkg, flag), f"Flag '{flag}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", ERROR_FLAGS)
    def test_registry_import_error_flag_is_str_or_none(self, mol_pkg, flag):
        """Each import error flag is either None or a string."""
        value = getattr(mol_pkg, flag)
        assert value is None or isinstance(value, str), (
            f"Flag '{flag}' should be None or str, got {type(value).__name__}"
        )


class TestSmokeRegistryInitFunctions:
    """§1.2 — Registry initialization functions are accessible and callable."""

    INIT_FUNCTIONS = [
        "_filter_init_registry",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INIT_FUNCTIONS)
    def test_init_registry_function_exists(self, mol_pkg, name):
        """Each registry init function is present and non-None."""
        obj = getattr(mol_pkg, name, None)
        assert obj is not None, (
            f"Registry init function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INIT_FUNCTIONS)
    def test_init_registry_function_is_callable(self, mol_pkg, name):
        """Each registry init function is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokePhase6InternalRegistryExports:
    """§1.2 — Phase 6 internal registry helper exports are accessible."""

    PHASE6_INTERNAL_EXPORTS = [
        "_filter_init_registry",
        "_filter_get_available_dataset_types",
        "_filter_is_dataset_type_registered",
        "_filter_get_dataset_feature",
        "_filter_get_handler_error_type_for_dataset",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PHASE6_INTERNAL_EXPORTS)
    def test_phase6_internal_export_exists(self, mol_pkg, name):
        """Each Phase 6 internal registry export is present."""
        assert hasattr(mol_pkg, name), (
            f"Phase 6 internal export '{name}' is missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PHASE6_INTERNAL_EXPORTS)
    def test_phase6_internal_export_is_callable(self, mol_pkg, name):
        """Each Phase 6 internal registry export is callable."""
        obj = getattr(mol_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, mol_pkg):
        """
        Re-importing the molecules package (via ``importlib.reload``) does
        not crash.

        Validates that all module-level code (logging, registry status checks)
        is safe to re-execute.
        """
        reloaded = importlib.reload(mol_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, mol_pkg):
        """
        Re-importing the molecules package preserves ``__all__``.
        """
        reloaded = importlib.reload(mol_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_logger_attribute_exists(self, mol_pkg):
        """
        The ``_logger`` attribute is defined on the molecules package.

        The ``__init__.py`` creates ``_logger = logging.getLogger(__name__)``.
        """
        assert hasattr(mol_pkg, "_logger"), (
            "_logger attribute is missing from molecules package"
        )

    @pytest.mark.smoke
    def test_logger_is_logger_instance(self, mol_pkg):
        """
        ``_logger`` is a ``logging.Logger`` instance (or LoggerAdapter).
        """
        logger = mol_pkg._logger
        assert isinstance(logger, (logging.Logger, logging.LoggerAdapter)), (
            f"_logger should be a Logger instance, got {type(logger).__name__}"
        )


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the molecules package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, mol_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(mol_pkg.__all__, list)

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
    def test_every_all_entry_is_resolvable(self, mol_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [
            name for name in all_names
            if not hasattr(mol_pkg, name)
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


class TestContractAllConsistency:
    """§2 — Every public import in the molecules module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Metadata dunders
        "__version__",
        # Internal logger
        "_logger",
        # Phase 6 internal registry re-exports (prefixed with underscore)
        "_filter_init_registry",
        "_filter_get_available_dataset_types",
        "_filter_is_dataset_type_registered",
        "_filter_get_dataset_feature",
        "_filter_get_handler_error_type_for_dataset",
        "_FILTER_REGISTRY_INITIALIZED",
        "_FILTER_REGISTRY_AVAILABLE",
        "_FILTER_REGISTRY_IMPORT_ERROR",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, mol_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the molecules ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(mol_pkg)
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
            # Private names that start with underscore but ARE in __all__
            # are already covered. Skip private names NOT in __all__.
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
            f"Public names imported in molecules/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractCoreClassTypes:
    """§2 — Core classes are actual classes with expected nature."""

    @pytest.mark.contract
    def test_molecule_data_converter_is_class(self, mol_pkg):
        """``MoleculeDataConverter`` is a class."""
        assert inspect.isclass(mol_pkg.MoleculeDataConverter), (
            f"MoleculeDataConverter should be a class, got "
            f"{type(mol_pkg.MoleculeDataConverter).__name__}"
        )

    @pytest.mark.contract
    def test_molecule_filter_is_class(self, mol_pkg):
        """``MoleculeFilter`` is a class."""
        assert inspect.isclass(mol_pkg.MoleculeFilter), (
            f"MoleculeFilter should be a class, got "
            f"{type(mol_pkg.MoleculeFilter).__name__}"
        )


class TestContractConversionFunctionSignatures:
    """§2 — Conversion functions are functions (not classes) with parameters."""

    CONVERSION_FUNCTIONS = [
        "create_rdkit_mol",
        "mol_to_pyg_data",
        "create_mol_with_dataset_support",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CONVERSION_FUNCTIONS)
    def test_conversion_is_function(self, mol_pkg, name):
        """Each conversion export is a function (not a class or unbound method)."""
        obj = getattr(mol_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CONVERSION_FUNCTIONS)
    def test_conversion_function_has_parameters(self, mol_pkg, name):
        """Each conversion function accepts at least one parameter."""
        sig = inspect.signature(getattr(mol_pkg, name))
        assert len(sig.parameters) >= 1, (
            f"'{name}' should accept at least one parameter"
        )


class TestContractValidationFunctionTypes:
    """§2 — Validation functions are functions (not classes)."""

    VALIDATION_FUNCTIONS = [
        "validate_molecular_structure",
        "check_dataset_compatibility",
        "validate_pyg_data_completeness",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", VALIDATION_FUNCTIONS)
    def test_validation_is_function(self, mol_pkg, name):
        """Each validation export is a function."""
        obj = getattr(mol_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )


class TestContractEnrichmentFunctionTypes:
    """§2 — Enrichment functions are functions (not classes)."""

    ENRICHMENT_FUNCTIONS = [
        "enrich_pyg_data_with_properties",
        "calculate_atomization_energy",
        "estimate_molecular_properties",
        "get_molecule_identifiers",
        "get_structural_feature_summary",
        "get_feature_extraction_diagnostics",
        "analyze_structural_feature_capabilities",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ENRICHMENT_FUNCTIONS)
    def test_enrichment_is_function(self, mol_pkg, name):
        """Each enrichment export is a function."""
        obj = getattr(mol_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )


class TestContractHandlerOnlyFunctionTypes:
    """§2 — Handler-only operation functions are functions."""

    HANDLER_ONLY_FUNCTIONS = [
        "estimate_properties_with_handler",
        "analyze_capabilities_with_handler",
        "create_handler_compatible_fingerprint",
        "validate_feature_extraction_with_handler",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", HANDLER_ONLY_FUNCTIONS)
    def test_handler_only_is_function(self, mol_pkg, name):
        """Each handler-only export is a function."""
        obj = getattr(mol_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", HANDLER_ONLY_FUNCTIONS)
    def test_handler_only_function_has_parameters(self, mol_pkg, name):
        """Each handler-only function accepts at least one parameter."""
        sig = inspect.signature(getattr(mol_pkg, name))
        assert len(sig.parameters) >= 1, (
            f"'{name}' should accept at least one parameter (handler)"
        )


class TestContractFilteringFactoryFunctions:
    """§2 — Filtering factory functions are functions."""

    FACTORY_FUNCTIONS = [
        "create_molecule_filter",
        "get_default_molecule_filter",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FACTORY_FUNCTIONS)
    def test_factory_is_function(self, mol_pkg, name):
        """Each factory export is a function."""
        obj = getattr(mol_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )


class TestContractFilteringCoreFunctions:
    """§2 — Filtering core functions are functions with parameters."""

    CORE_FILTER_FUNCTIONS = [
        "apply_pre_filters",
        "apply_atom_count_filters",
        "apply_heavy_atom_filters",
        "apply_dataset_specific_filters",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CORE_FILTER_FUNCTIONS)
    def test_core_filter_is_function(self, mol_pkg, name):
        """Each core filtering export is a function."""
        obj = getattr(mol_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CORE_FILTER_FUNCTIONS)
    def test_core_filter_function_has_parameters(self, mol_pkg, name):
        """Each core filter function accepts at least one parameter."""
        sig = inspect.signature(getattr(mol_pkg, name))
        assert len(sig.parameters) >= 1, (
            f"'{name}' should accept at least one parameter"
        )


class TestContractFilteringValidationFunctions:
    """§2 — Filtering validation functions are functions."""

    FILTER_VALIDATION = [
        "validate_filter_configuration",
        "validate_filter_compatibility_with_transforms",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FILTER_VALIDATION)
    def test_filter_validation_is_function(self, mol_pkg, name):
        """Each filter validation export is a function."""
        obj = getattr(mol_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )


class TestContractFilteringUtilityFunctions:
    """§2 — Filtering utility functions are functions."""

    FILTER_UTILITIES = [
        "introspect_transform_filter_parameters",
        "create_handler_aware_filter_stats",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FILTER_UTILITIES)
    def test_filter_utility_is_function(self, mol_pkg, name):
        """Each filter utility export is a function."""
        obj = getattr(mol_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )


class TestContractFeatureFunctionTypes:
    """§2 — Structural feature functions are functions."""

    FEATURE_FUNCTIONS = [
        "add_structural_features",
        "get_available_features",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FEATURE_FUNCTIONS)
    def test_feature_is_function(self, mol_pkg, name):
        """Each structural feature export is a function."""
        obj = getattr(mol_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )


class TestContractRegistryStatusReportingFunctions:
    """§2 — Registry status reporting functions return dicts."""

    STATUS_FUNCTIONS = [
        "get_registry_integration_status",
        "get_enricher_registry_status",
        "get_validator_registry_status",
        "get_filter_registry_status",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", STATUS_FUNCTIONS)
    def test_registry_status_function_returns_dict(self, mol_pkg, name):
        """Each registry status reporting function returns a dict."""
        func = getattr(mol_pkg, name)
        result = func()
        assert isinstance(result, dict), (
            f"'{name}()' should return dict, got {type(result).__name__}"
        )


class TestContractGetFilterRegistryStatusKeys:
    """§2 — ``get_filter_registry_status()`` returns dict with documented keys."""

    @pytest.mark.contract
    def test_filter_registry_status_has_registry_available(self, mol_pkg):
        """``get_filter_registry_status()`` result includes ``registry_available``."""
        result = mol_pkg.get_filter_registry_status()
        assert "registry_available" in result, (
            "get_filter_registry_status() missing 'registry_available' key"
        )

    @pytest.mark.contract
    def test_filter_registry_status_has_registry_initialized(self, mol_pkg):
        """``get_filter_registry_status()`` result includes ``registry_initialized``."""
        result = mol_pkg.get_filter_registry_status()
        assert "registry_initialized" in result, (
            "get_filter_registry_status() missing 'registry_initialized' key"
        )

    @pytest.mark.contract
    def test_filter_registry_status_has_phase_6_complete(self, mol_pkg):
        """``get_filter_registry_status()`` result includes ``phase_6_complete``."""
        result = mol_pkg.get_filter_registry_status()
        assert "phase_6_complete" in result, (
            "get_filter_registry_status() missing 'phase_6_complete' key"
        )

    @pytest.mark.contract
    def test_filter_registry_status_phase_6_is_true(self, mol_pkg):
        """``get_filter_registry_status()['phase_6_complete']`` is True."""
        result = mol_pkg.get_filter_registry_status()
        assert result["phase_6_complete"] is True, (
            "get_filter_registry_status()['phase_6_complete'] should be True"
        )

    @pytest.mark.contract
    def test_filter_registry_status_has_available_dataset_types(self, mol_pkg):
        """``get_filter_registry_status()`` result includes ``available_dataset_types``."""
        result = mol_pkg.get_filter_registry_status()
        assert "available_dataset_types" in result, (
            "get_filter_registry_status() missing 'available_dataset_types' key"
        )

    @pytest.mark.contract
    def test_filter_registry_status_available_types_is_list(self, mol_pkg):
        """``get_filter_registry_status()['available_dataset_types']`` is a list."""
        result = mol_pkg.get_filter_registry_status()
        assert isinstance(result["available_dataset_types"], list), (
            f"'available_dataset_types' should be list, got "
            f"{type(result['available_dataset_types']).__name__}"
        )

    @pytest.mark.contract
    def test_filter_registry_status_has_features_available(self, mol_pkg):
        """``get_filter_registry_status()`` result includes ``features_available``."""
        result = mol_pkg.get_filter_registry_status()
        assert "features_available" in result, (
            "get_filter_registry_status() missing 'features_available' key"
        )

    @pytest.mark.contract
    def test_filter_registry_status_features_is_list(self, mol_pkg):
        """``get_filter_registry_status()['features_available']`` is a list."""
        result = mol_pkg.get_filter_registry_status()
        assert isinstance(result["features_available"], list), (
            f"'features_available' should be list, got "
            f"{type(result['features_available']).__name__}"
        )

    @pytest.mark.contract
    def test_filter_registry_status_features_expected_entries(self, mol_pkg):
        """
        ``get_filter_registry_status()['features_available']`` contains
        expected feature names as documented in the __init__.py.
        """
        result = mol_pkg.get_filter_registry_status()
        features = set(result["features_available"])
        expected_features = {
            "uncertainty_handling",
            "vibrational_analysis",
            "atomization_energy",
            "rotational_constants",
            "frequency_analysis",
            "orbital_analysis",
            "homo_lumo_gap",
            "mo_energies",
        }
        missing = expected_features - features
        assert not missing, (
            f"Expected features missing from features_available: {sorted(missing)}"
        )

    @pytest.mark.contract
    def test_filter_registry_status_has_import_error(self, mol_pkg):
        """``get_filter_registry_status()`` result includes ``registry_import_error``."""
        result = mol_pkg.get_filter_registry_status()
        assert "registry_import_error" in result, (
            "get_filter_registry_status() missing 'registry_import_error' key"
        )


class TestContractRegistryFlagsConsistency:
    """§2 — Phase 6 molecule_filters registry flags are internally consistent."""

    @pytest.mark.contract
    def test_filter_registry_flags_consistent(self, mol_pkg):
        """
        If ``_FILTER_REGISTRY_AVAILABLE`` is True, then
        ``_FILTER_REGISTRY_IMPORT_ERROR`` should be None.
        """
        available = mol_pkg._FILTER_REGISTRY_AVAILABLE
        error = mol_pkg._FILTER_REGISTRY_IMPORT_ERROR

        if available:
            assert error is None, (
                f"_FILTER_REGISTRY_AVAILABLE is True but "
                f"_FILTER_REGISTRY_IMPORT_ERROR is '{error}'"
            )

    @pytest.mark.contract
    def test_filter_registry_status_consistent_with_flags(self, mol_pkg):
        """
        ``get_filter_registry_status()['registry_available']`` is consistent
        with ``_FILTER_REGISTRY_AVAILABLE``.
        """
        status = mol_pkg.get_filter_registry_status()
        flag = mol_pkg._FILTER_REGISTRY_AVAILABLE
        assert status["registry_available"] == flag, (
            f"get_filter_registry_status()['registry_available']={status['registry_available']} "
            f"but _FILTER_REGISTRY_AVAILABLE={flag}"
        )


class TestContractPhaseAliasNonCollision:
    """
    §2 — Phase-aliased imports do not collide.

    The ``__init__.py`` imports registry functions from ``molecule_filters``
    with ``as`` aliases (prefixed with ``_filter_``). Each aliased name must
    resolve to a distinct module-level attribute.
    """

    @pytest.mark.contract
    def test_filter_registry_aliases_are_distinct_attributes(self, mol_pkg):
        """
        ``_filter_init_registry``, ``_filter_get_available_dataset_types``,
        ``_filter_is_dataset_type_registered``, ``_filter_get_dataset_feature``,
        ``_filter_get_handler_error_type_for_dataset``
        are all distinct attributes on the molecules package.
        """
        aliases = [
            "_filter_init_registry",
            "_filter_get_available_dataset_types",
            "_filter_is_dataset_type_registered",
            "_filter_get_dataset_feature",
            "_filter_get_handler_error_type_for_dataset",
        ]
        for alias in aliases:
            assert hasattr(mol_pkg, alias), (
                f"Aliased function '{alias}' is missing from molecules package"
            )
            assert callable(getattr(mol_pkg, alias)), (
                f"Aliased function '{alias}' should be callable"
            )

    @pytest.mark.contract
    def test_diagnostics_aliases_are_distinct(self, mol_pkg):
        """
        ``get_registry_integration_status``, ``get_enricher_registry_status``,
        ``get_validator_registry_status``, ``get_filter_registry_status``
        are all distinct callable attributes.
        """
        aliases = [
            "get_registry_integration_status",
            "get_enricher_registry_status",
            "get_validator_registry_status",
            "get_filter_registry_status",
        ]
        resolved = {}
        for alias in aliases:
            obj = getattr(mol_pkg, alias, None)
            assert obj is not None, f"'{alias}' is None or missing"
            assert callable(obj), f"'{alias}' should be callable"
            resolved[alias] = id(obj)

        # All four should be distinct functions (from different submodules)
        unique_ids = set(resolved.values())
        assert len(unique_ids) == len(aliases), (
            f"Expected {len(aliases)} distinct diagnostics functions, "
            f"but only {len(unique_ids)} are unique: {resolved}"
        )


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    # Based on the documented Module Exports from the project structure doc
    MINIMUM_API = {
        # Core classes
        "MoleculeDataConverter",
        "MoleculeFilter",
        # Conversion
        "create_rdkit_mol",
        "mol_to_pyg_data",
        "create_mol_with_dataset_support",
        # Features
        "add_structural_features",
        "get_available_features",
        # Validation
        "validate_molecular_structure",
        "check_dataset_compatibility",
        "validate_pyg_data_completeness",
        # Enrichment
        "enrich_pyg_data_with_properties",
        "calculate_atomization_energy",
        "estimate_molecular_properties",
        "get_molecule_identifiers",
        "get_structural_feature_summary",
        "get_feature_extraction_diagnostics",
        "analyze_structural_feature_capabilities",
        # Handler-only
        "estimate_properties_with_handler",
        "analyze_capabilities_with_handler",
        "create_handler_compatible_fingerprint",
        "validate_feature_extraction_with_handler",
        # Filtering - Factory
        "create_molecule_filter",
        "get_default_molecule_filter",
        # Filtering - Core
        "apply_pre_filters",
        "apply_atom_count_filters",
        "apply_heavy_atom_filters",
        "apply_dataset_specific_filters",
        # Filtering - Validation
        "validate_filter_configuration",
        "validate_filter_compatibility_with_transforms",
        # Filtering - Utilities
        "introspect_transform_filter_parameters",
        "create_handler_aware_filter_stats",
        # Diagnostics (Phase 6)
        "get_registry_integration_status",
        "get_enricher_registry_status",
        "get_validator_registry_status",
        "get_filter_registry_status",
    }

    @pytest.mark.contract
    def test_minimum_api_in_all(self, all_names):
        """The minimum expected public API is present in ``__all__``."""
        all_set = set(all_names)
        missing = self.MINIMUM_API - all_set
        assert not missing, (
            f"Minimum API names missing from __all__: {sorted(missing)}"
        )

    @pytest.mark.contract
    def test_all_has_expected_length(self, all_names):
        """
        ``__all__`` contains a substantial number of entries.

        Based on the __init__.py source, the molecules package exports ~50
        names. This test guards against catastrophic loss (e.g., accidental
        truncation of __all__) while allowing for organic growth.
        """
        actual = len(all_names)
        # The __init__.py has ~50 entries in __all__
        # We set a floor well below the actual count to allow changes
        # while catching catastrophic loss.
        MINIMUM_EXPECTED = 30
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least {MINIMUM_EXPECTED}. "
            f"This suggests __all__ may have been accidentally truncated."
        )


class TestContractValidatorRegistryStatusReturnType:
    """§2 — ``get_validator_registry_status()`` return type contract."""

    @pytest.mark.contract
    def test_validator_registry_status_returns_dict(self, mol_pkg):
        """``get_validator_registry_status()`` returns a dict."""
        result = mol_pkg.get_validator_registry_status()
        assert isinstance(result, dict), (
            f"get_validator_registry_status() should return dict, got "
            f"{type(result).__name__}"
        )


class TestContractEnricherRegistryStatusReturnType:
    """§2 — ``get_enricher_registry_status()`` return type contract."""

    @pytest.mark.contract
    def test_enricher_registry_status_returns_dict(self, mol_pkg):
        """``get_enricher_registry_status()`` returns a dict."""
        result = mol_pkg.get_enricher_registry_status()
        assert isinstance(result, dict), (
            f"get_enricher_registry_status() should return dict, got "
            f"{type(result).__name__}"
        )


class TestContractPropertyEnrichmentRegistryStatusReturnType:
    """§2 — ``get_registry_integration_status()`` return type contract."""

    @pytest.mark.contract
    def test_property_enrichment_status_returns_dict(self, mol_pkg):
        """``get_registry_integration_status()`` returns a dict."""
        result = mol_pkg.get_registry_integration_status()
        assert isinstance(result, dict), (
            f"get_registry_integration_status() should return dict, got "
            f"{type(result).__name__}"
        )


class TestContractFilterGetAvailableDatasetTypesReturnType:
    """§2 — ``_filter_get_available_dataset_types()`` return type contract."""

    @pytest.mark.contract
    def test_filter_get_available_dataset_types_returns_list(self, mol_pkg):
        """``_filter_get_available_dataset_types()`` returns a list."""
        # Ensure registry is initialized first
        mol_pkg._filter_init_registry()
        result = mol_pkg._filter_get_available_dataset_types()
        assert isinstance(result, list), (
            f"_filter_get_available_dataset_types() should return list, got "
            f"{type(result).__name__}"
        )

    @pytest.mark.contract
    def test_filter_get_available_dataset_types_entries_are_strings(self, mol_pkg):
        """Each entry from ``_filter_get_available_dataset_types()`` is a string."""
        mol_pkg._filter_init_registry()
        result = mol_pkg._filter_get_available_dataset_types()
        for entry in result:
            assert isinstance(entry, str), (
                f"Dataset type entry should be str, got {type(entry).__name__}"
            )


class TestContractFilterIsDatasetTypeRegisteredReturnType:
    """§2 — ``_filter_is_dataset_type_registered()`` return type contract."""

    @pytest.mark.contract
    def test_filter_is_dataset_type_registered_returns_bool(self, mol_pkg):
        """
        ``_filter_is_dataset_type_registered()`` returns a bool for a
        known dataset type name.
        """
        mol_pkg._filter_init_registry()
        # Use a dataset type that is known to exist per the project structure
        result = mol_pkg._filter_is_dataset_type_registered("DFT")
        assert isinstance(result, bool), (
            f"_filter_is_dataset_type_registered('DFT') should return bool, got "
            f"{type(result).__name__}"
        )

    @pytest.mark.contract
    def test_filter_is_dataset_type_registered_false_for_unknown(self, mol_pkg):
        """
        ``_filter_is_dataset_type_registered()`` returns False for an
        unknown dataset type.
        """
        mol_pkg._filter_init_registry()
        result = mol_pkg._filter_is_dataset_type_registered(
            "NONEXISTENT_DATASET_TYPE_XYZ"
        )
        assert result is False, (
            "_filter_is_dataset_type_registered('NONEXISTENT_DATASET_TYPE_XYZ') "
            "should return False"
        )


class TestContractMoleculeDataConverterInterface:
    """§2 — MoleculeDataConverter class exposes expected methods."""

    EXPECTED_METHODS = [
        "convert",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", EXPECTED_METHODS)
    def test_converter_has_expected_method(self, mol_pkg, method_name):
        """MoleculeDataConverter exposes expected method(s)."""
        cls = mol_pkg.MoleculeDataConverter
        assert hasattr(cls, method_name), (
            f"MoleculeDataConverter should have method '{method_name}'"
        )
        method = getattr(cls, method_name)
        assert callable(method), (
            f"MoleculeDataConverter.{method_name} should be callable"
        )


class TestContractMoleculeFilterInterface:
    """§2 — MoleculeFilter class exposes expected methods."""

    EXPECTED_METHODS = [
        "apply_filters",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", EXPECTED_METHODS)
    def test_filter_has_expected_method(self, mol_pkg, method_name):
        """MoleculeFilter exposes expected method(s)."""
        cls = mol_pkg.MoleculeFilter
        assert hasattr(cls, method_name), (
            f"MoleculeFilter should have method '{method_name}'"
        )
        method = getattr(cls, method_name)
        assert callable(method), (
            f"MoleculeFilter.{method_name} should be callable"
        )


class TestContractVersionContract:
    """§2 — ``__version__`` follows documented contract."""

    @pytest.mark.contract
    def test_version_is_1_4_x(self, mol_pkg):
        """
        ``__version__`` starts with '1.4' per the Phase 6 molecule_filters
        integration documented in the source.
        """
        version = mol_pkg.__version__
        assert version.startswith("1.4"), (
            f"Expected __version__ to start with '1.4' (Phase 6), got '{version}'"
        )

    @pytest.mark.contract
    def test_version_has_three_components(self, mol_pkg):
        """``__version__`` has exactly three MAJOR.MINOR.PATCH components."""
        version = mol_pkg.__version__
        parts = version.split(".")
        assert len(parts) == 3, (
            f"Expected 3 version components (MAJOR.MINOR.PATCH), "
            f"got {len(parts)} in '{version}'"
        )
