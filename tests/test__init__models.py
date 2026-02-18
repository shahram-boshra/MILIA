#!/usr/bin/env python3
# tests/test__init__models.py

"""
Test Suite: milia_pipeline/models/__init__.py — Smoke Tests & Contract Tests
============================================================================

Production-ready test suite for the MILIA Pipeline models package
``milia_pipeline/models/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.models`` subpackage imports without ImportError
        - All re-exported names from the submodules are accessible
        - Module-level metadata attributes (__version__, __author__, etc.) exist
        - Module initialization (logging, registry statistics) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Categories & metadata exports from pyg_introspector are accessible
        - Registry system exports from model_registry are accessible
        - Factory & validator exports from model_factory are accessible
        - Phase 7 builders module exports are accessible
        - Phase 8 HPO module exports are accessible (conditional)
        - Training infrastructure exports are accessible
        - Exception exports are accessible
        - Convenience functions (get_module_info, print_module_summary) exist

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, callables are callable
        - Fallback classes (import failure stubs) have correct structure
        - HPO_AVAILABLE flag is boolean
        - _MISSING_COMPONENTS is a list
        - Convenience functions have expected signatures and return types
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)
        - Exception classes are proper Exception subclasses
        - Registry instance (``registry``) has ``list_available_models`` method
        - Module metadata (__author__, __license__, __status__) are strings

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__models.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import importlib
import inspect
import logging
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__models.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def models_pkg():
    """
    Import and return the ``milia_pipeline.models`` package once per module.

    This fixture validates the fundamental smoke invariant: the models
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.models as mdl

        return mdl
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.models could not be imported — smoke test precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(models_pkg):
    """Return the ``__all__`` list from the models package."""
    assert hasattr(models_pkg, "__all__"), (
        "milia_pipeline.models.__all__ is missing — contract violation"
    )
    return list(models_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeModelsPackageImport:
    """§1.2 — Verify the models subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_models_package_succeeds(self, models_pkg):
        """The models package imports without raising any exception."""
        assert models_pkg is not None

    @pytest.mark.smoke
    def test_models_package_is_a_module(self, models_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(models_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_models_package_has_file_attribute(self, models_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(models_pkg, "__file__")

    @pytest.mark.smoke
    def test_models_package_name(self, models_pkg):
        """The package ``__name__`` is ``milia_pipeline.models``."""
        assert models_pkg.__name__ == "milia_pipeline.models"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "attr",
        [
            "__version__",
            "__author__",
            "__license__",
            "__maintainer__",
            "__status__",
        ],
    )
    def test_metadata_attribute_exists(self, models_pkg, attr):
        """Each metadata dunder is defined on the models package."""
        assert hasattr(models_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "attr",
        [
            "__version__",
            "__author__",
            "__license__",
            "__maintainer__",
            "__status__",
        ],
    )
    def test_metadata_attribute_is_string(self, models_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(models_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, models_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = models_pkg.__version__
        parts = version.split(".")
        assert len(parts) >= 2, f"Version '{version}' should have at least MAJOR.MINOR components"
        for part in parts:
            numeric_part = ""
            for ch in part:
                if ch.isdigit():
                    numeric_part += ch
                else:
                    break
            assert len(numeric_part) > 0, f"Version component '{part}' should start with a digit"

    @pytest.mark.smoke
    def test_status_is_production(self, models_pkg):
        """``__status__`` is 'Production'."""
        assert models_pkg.__status__ == "Production"


class TestSmokeCategoriesAndMetadataExports:
    """§1.2 — Categories and metadata exports from pyg_introspector are accessible."""

    INTROSPECTOR_CLASSES = [
        "ModelCategory",
        "ModelMetadata",
    ]

    INTROSPECTOR_FUNCTIONS = [
        "get_model_metadata",
        "get_all_model_names",
        "get_models_by_category",
        "get_models_by_task",
        "get_models_by_tag",
        "search_models",
        "get_category_statistics",
    ]

    INTROSPECTOR_DATA = [
        "ALL_MODELS",
        "MODELS_BY_CATEGORY",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INTROSPECTOR_CLASSES)
    def test_introspector_class_exists(self, models_pkg, name):
        """Each introspector class is present on the models package."""
        assert hasattr(models_pkg, name), f"Introspector class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INTROSPECTOR_FUNCTIONS)
    def test_introspector_function_exists(self, models_pkg, name):
        """Each introspector function is present and non-None."""
        obj = getattr(models_pkg, name, None)
        assert obj is not None, f"Introspector function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INTROSPECTOR_FUNCTIONS)
    def test_introspector_function_is_callable(self, models_pkg, name):
        """Each introspector function is callable."""
        obj = getattr(models_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INTROSPECTOR_DATA)
    def test_introspector_data_exists(self, models_pkg, name):
        """Each introspector data export is present."""
        assert hasattr(models_pkg, name), f"Introspector data '{name}' is missing"


class TestSmokeRegistrySystemExports:
    """§1.2 — Registry system exports from model_registry are accessible."""

    REGISTRY_CLASSES = [
        "ModelRegistry",
        "ModelRegistration",
    ]

    REGISTRY_FUNCTIONS = [
        "get_model",
        "has_model",
        "list_models",
        "get_model_info",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", REGISTRY_CLASSES)
    def test_registry_class_exists(self, models_pkg, name):
        """Each registry class is present on the models package."""
        assert hasattr(models_pkg, name), f"Registry class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", REGISTRY_FUNCTIONS)
    def test_registry_function_exists(self, models_pkg, name):
        """Each registry function is present and non-None."""
        obj = getattr(models_pkg, name, None)
        assert obj is not None, f"Registry function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", REGISTRY_FUNCTIONS)
    def test_registry_function_is_callable(self, models_pkg, name):
        """Each registry function is callable."""
        obj = getattr(models_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    def test_registry_instance_exists(self, models_pkg):
        """The ``registry`` singleton instance is present."""
        assert hasattr(models_pkg, "registry"), "registry instance is missing"
        assert models_pkg.registry is not None, "registry should not be None"


class TestSmokeFactoryAndValidatorExports:
    """§1.2 — Factory and validator exports from model_factory are accessible."""

    FACTORY_CLASSES = [
        "ModelFactory",
        "ModelValidator",
        "GraphLevelModelWrapper",
        "EdgeLevelModelWrapper",
    ]

    FACTORY_FUNCTIONS = [
        "create_model",
        "get_factory",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FACTORY_CLASSES)
    def test_factory_class_exists(self, models_pkg, name):
        """Each factory class is present on the models package."""
        assert hasattr(models_pkg, name), f"Factory class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FACTORY_FUNCTIONS)
    def test_factory_function_exists(self, models_pkg, name):
        """Each factory function is present and non-None."""
        obj = getattr(models_pkg, name, None)
        assert obj is not None, f"Factory function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FACTORY_FUNCTIONS)
    def test_factory_function_is_callable(self, models_pkg, name):
        """Each factory function is callable."""
        obj = getattr(models_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokePhase7BuildersExports:
    """§1.2 — Phase 7 builders module exports are accessible."""

    BUILDER_CLASSES = [
        "LayerRegistry",
        "LayerCategory",
        "LayerMetadata",
        "ArchitectureBuilder",
        "LayerConfig",
        "ArchitectureConfig",
        "ModelComposer",
        "EnsembleConfig",
        "ArchitectureValidator",
        "ArchitectureTemplates",
    ]

    BUILDER_FUNCTIONS = [
        "validate_architecture",
        "validate_data_compatibility",
        "parse_custom_architecture",
        "parse_ensemble",
        "load_config",
        "validate_config",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", BUILDER_CLASSES)
    def test_builder_class_exists(self, models_pkg, name):
        """Each builder class is present on the models package."""
        assert hasattr(models_pkg, name), f"Builder class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", BUILDER_FUNCTIONS)
    def test_builder_function_exists(self, models_pkg, name):
        """Each builder function is present and non-None."""
        obj = getattr(models_pkg, name, None)
        assert obj is not None, f"Builder function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", BUILDER_FUNCTIONS)
    def test_builder_function_is_callable(self, models_pkg, name):
        """Each builder function is callable."""
        obj = getattr(models_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    def test_layer_registry_instance_exists(self, models_pkg):
        """The ``layer_registry`` instance is present."""
        assert hasattr(models_pkg, "layer_registry"), "layer_registry instance is missing"


class TestSmokePhase8HPOExports:
    """§1.2 — Phase 8 HPO module exports are accessible (conditional)."""

    HPO_NAMES = [
        "HPOManager",
        "HPOConfig",
        "is_hpo_enabled",
        "get_best_params",
        "create_hpo_manager",
        "infer_task_type",
        "OptunaPruningCallback",
    ]

    @pytest.mark.smoke
    def test_hpo_available_flag_exists(self, models_pkg):
        """``HPO_AVAILABLE`` flag is defined on the models package."""
        assert hasattr(models_pkg, "HPO_AVAILABLE"), "HPO_AVAILABLE is missing"

    @pytest.mark.smoke
    def test_hpo_available_is_bool(self, models_pkg):
        """``HPO_AVAILABLE`` is a boolean."""
        assert isinstance(models_pkg.HPO_AVAILABLE, bool), (
            f"HPO_AVAILABLE should be bool, got {type(models_pkg.HPO_AVAILABLE).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", HPO_NAMES)
    def test_hpo_export_exists(self, models_pkg, name):
        """Each HPO export name is present on the models package."""
        assert hasattr(models_pkg, name), f"HPO export '{name}' is missing"

    @pytest.mark.smoke
    def test_hpo_exports_consistent_with_flag(self, models_pkg):
        """
        When ``HPO_AVAILABLE`` is True, HPO exports are non-None.
        When False, HPO exports are None (graceful degradation).
        """
        if models_pkg.HPO_AVAILABLE:
            for name in self.HPO_NAMES:
                obj = getattr(models_pkg, name)
                assert obj is not None, f"HPO_AVAILABLE is True but '{name}' is None"
        else:
            for name in self.HPO_NAMES:
                obj = getattr(models_pkg, name)
                assert obj is None, f"HPO_AVAILABLE is False but '{name}' is not None"


class TestSmokeTrainingInfrastructureExports:
    """§1.2 — Training infrastructure exports are accessible."""

    TRAINER_EXPORTS = [
        "Trainer",
    ]

    CALLBACK_CLASSES = [
        "Callback",
        "EarlyStopping",
        "ModelCheckpoint",
        "TensorBoardLogger",
        "LearningRateMonitor",
        "ProgressBar",
        "GradientMonitor",
        "CallbackFactory",
    ]

    DATA_SPLITTING_CLASSES = [
        "DataSplitter",
    ]

    DATA_SPLITTING_FUNCTIONS = [
        "random_split",
        "stratified_split",
        "temporal_split",
        "scaffold_split",
        "k_fold_split",
    ]

    LOSS_CLASSES = [
        "LossRegistry",
        "FocalLoss",
        "WeightedMSELoss",
        "RMSELoss",
    ]

    LOSS_FUNCTIONS = [
        "get_loss",
        "list_losses",
    ]

    OPTIMIZER_CLASSES = [
        "OptimizerRegistry",
    ]

    OPTIMIZER_FUNCTIONS = [
        "get_optimizer",
        "list_optimizers",
    ]

    SCHEDULER_CLASSES = [
        "SchedulerRegistry",
    ]

    SCHEDULER_FUNCTIONS = [
        "get_scheduler",
        "list_schedulers",
        "create_warmup_scheduler",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", TRAINER_EXPORTS)
    def test_trainer_export_exists(self, models_pkg, name):
        """Trainer export is present on the models package."""
        assert hasattr(models_pkg, name), f"Trainer export '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CALLBACK_CLASSES)
    def test_callback_class_exists(self, models_pkg, name):
        """Each callback class is present on the models package."""
        assert hasattr(models_pkg, name), f"Callback class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DATA_SPLITTING_CLASSES)
    def test_data_splitting_class_exists(self, models_pkg, name):
        """Each data splitting class is present."""
        assert hasattr(models_pkg, name), f"Data splitting class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DATA_SPLITTING_FUNCTIONS)
    def test_data_splitting_function_exists(self, models_pkg, name):
        """Each data splitting function is present and non-None."""
        obj = getattr(models_pkg, name, None)
        assert obj is not None, f"Data splitting function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DATA_SPLITTING_FUNCTIONS)
    def test_data_splitting_function_is_callable(self, models_pkg, name):
        """Each data splitting function is callable."""
        obj = getattr(models_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", LOSS_CLASSES)
    def test_loss_class_exists(self, models_pkg, name):
        """Each loss class is present on the models package."""
        assert hasattr(models_pkg, name), f"Loss class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", LOSS_FUNCTIONS)
    def test_loss_function_exists(self, models_pkg, name):
        """Each loss function is present and non-None."""
        obj = getattr(models_pkg, name, None)
        assert obj is not None, f"Loss function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", LOSS_FUNCTIONS)
    def test_loss_function_is_callable(self, models_pkg, name):
        """Each loss function is callable."""
        obj = getattr(models_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", OPTIMIZER_CLASSES)
    def test_optimizer_class_exists(self, models_pkg, name):
        """Each optimizer class is present on the models package."""
        assert hasattr(models_pkg, name), f"Optimizer class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", OPTIMIZER_FUNCTIONS)
    def test_optimizer_function_exists(self, models_pkg, name):
        """Each optimizer function is present and non-None."""
        obj = getattr(models_pkg, name, None)
        assert obj is not None, f"Optimizer function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", OPTIMIZER_FUNCTIONS)
    def test_optimizer_function_is_callable(self, models_pkg, name):
        """Each optimizer function is callable."""
        obj = getattr(models_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", SCHEDULER_CLASSES)
    def test_scheduler_class_exists(self, models_pkg, name):
        """Each scheduler class is present on the models package."""
        assert hasattr(models_pkg, name), f"Scheduler class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", SCHEDULER_FUNCTIONS)
    def test_scheduler_function_exists(self, models_pkg, name):
        """Each scheduler function is present and non-None."""
        obj = getattr(models_pkg, name, None)
        assert obj is not None, f"Scheduler function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", SCHEDULER_FUNCTIONS)
    def test_scheduler_function_is_callable(self, models_pkg, name):
        """Each scheduler function is callable."""
        obj = getattr(models_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeExceptionExports:
    """§1.2 — Exception exports are accessible."""

    EXCEPTION_CLASSES = [
        "ModelError",
        "ModelNotFoundError",
        "ModelValidationError",
        "ModelInstantiationError",
        "HyperparameterError",
        "DataCompatibilityError",
        "TrainingError",
        "CheckpointError",
        "DataError",
        "PluginModelError",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_exists(self, models_pkg, name):
        """Each exception class is present on the models package."""
        assert hasattr(models_pkg, name), f"Exception class '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_is_a_class(self, models_pkg, name):
        """Each exception export is a class."""
        obj = getattr(models_pkg, name)
        assert inspect.isclass(obj), f"'{name}' should be a class, got {type(obj).__name__}"


class TestSmokeConvenienceFunctions:
    """§1.2 — Convenience functions are accessible and callable."""

    @pytest.mark.smoke
    def test_get_module_info_exists(self, models_pkg):
        """``get_module_info`` is present on the models package."""
        assert hasattr(models_pkg, "get_module_info"), "get_module_info is missing"

    @pytest.mark.smoke
    def test_get_module_info_is_callable(self, models_pkg):
        """``get_module_info`` is callable."""
        assert callable(models_pkg.get_module_info), "get_module_info should be callable"

    @pytest.mark.smoke
    def test_print_module_summary_exists(self, models_pkg):
        """``print_module_summary`` is present on the models package."""
        assert hasattr(models_pkg, "print_module_summary"), "print_module_summary is missing"

    @pytest.mark.smoke
    def test_print_module_summary_is_callable(self, models_pkg):
        """``print_module_summary`` is callable."""
        assert callable(models_pkg.print_module_summary), "print_module_summary should be callable"


class TestSmokeMissingComponentsTracking:
    """§1.2 — _MISSING_COMPONENTS tracking list is present."""

    @pytest.mark.smoke
    def test_missing_components_exists(self, models_pkg):
        """``_MISSING_COMPONENTS`` list is defined on the models package."""
        assert hasattr(models_pkg, "_MISSING_COMPONENTS"), "_MISSING_COMPONENTS is missing"

    @pytest.mark.smoke
    def test_missing_components_is_list(self, models_pkg):
        """``_MISSING_COMPONENTS`` is a list."""
        assert isinstance(models_pkg._MISSING_COMPONENTS, list), (
            f"_MISSING_COMPONENTS should be list, got "
            f"{type(models_pkg._MISSING_COMPONENTS).__name__}"
        )


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, models_pkg):
        """
        Re-importing the models package (via ``importlib.reload``) does not
        crash.

        Validates that all module-level code (logging, registry statistics,
        _MISSING_COMPONENTS tracking) is safe to re-execute.
        """
        reloaded = importlib.reload(models_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, models_pkg):
        """
        Re-importing the models package preserves ``__all__``.
        """
        reloaded = importlib.reload(models_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_logger_is_accessible(self, models_pkg):
        """
        The ``logger`` object is present on the models package (module-level
        logger created via ``logging.getLogger(__name__)``).
        """
        assert hasattr(models_pkg, "logger"), "logger is missing"
        obj = models_pkg.logger
        assert isinstance(obj, logging.Logger), (
            f"logger should be logging.Logger, got {type(obj).__name__}"
        )


class TestSmokeDocstringPresent:
    """§1.2 — Module docstring is present and informative."""

    @pytest.mark.smoke
    def test_module_docstring_exists(self, models_pkg):
        """The models package has a non-empty ``__doc__`` attribute."""
        assert models_pkg.__doc__ is not None, "__doc__ is None"
        assert isinstance(models_pkg.__doc__, str), "__doc__ should be str"
        assert len(models_pkg.__doc__) > 100, "__doc__ should be a substantial docstring"


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the models package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, models_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(models_pkg.__all__, list)

    @pytest.mark.contract
    def test_all_contains_no_duplicates(self, all_names):
        """``__all__`` has no duplicate entries."""
        seen = set()
        duplicates = []
        for name in all_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)

        assert not duplicates, f"Duplicate entries in __all__: {duplicates}"

    @pytest.mark.contract
    def test_every_all_entry_is_resolvable(self, models_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [name for name in all_names if not hasattr(models_pkg, name)]
        assert not unresolvable, (
            f"Names in __all__ that are not defined on the module: {unresolvable}"
        )

    @pytest.mark.contract
    def test_all_entries_are_strings(self, all_names):
        """Every entry in ``__all__`` is a string."""
        non_strings = [(i, name) for i, name in enumerate(all_names) if not isinstance(name, str)]
        assert not non_strings, f"Non-string entries in __all__: {non_strings}"


class TestContractAllConsistency:
    """§2 — Every public import in the models module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Metadata dunders (not typically in __all__)
        "__version__",
        "__author__",
        "__license__",
        "__maintainer__",
        "__status__",
        # Internal tracking
        "_MISSING_COMPONENTS",
        # Module-level logger
        "logger",
        # Internal renamed import
        "get_factory_model_info",
        # Module-level initialization variables (transient, not public API)
        # Created at lines 684-690 during registry statistics logging
        "available_models",
        "stats",
        # Convenience functions defined after __all__ (lines 700-811)
        # Not included in __all__ by the module author
        "get_module_info",
        "print_module_summary",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, models_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the models ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(models_pkg)
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
            "__builtins__",
            "__cached__",
            "__doc__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__path__",
            "__spec__",
        }
        missing_from_all = [n for n in missing_from_all if n not in python_internals]

        assert not missing_from_all, (
            f"Public names imported in models/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractClassTypes:
    """§2 — Re-exported classes are actually classes."""

    EXPECTED_CLASSES = [
        # Registry
        "ModelRegistry",
        "ModelRegistration",
        # Factory
        "ModelFactory",
        "ModelValidator",
        "GraphLevelModelWrapper",
        "EdgeLevelModelWrapper",
        # Categories & Metadata
        "ModelCategory",
        "ModelMetadata",
        # Builders
        "LayerRegistry",
        "LayerCategory",
        "LayerMetadata",
        "ArchitectureBuilder",
        "LayerConfig",
        "ArchitectureConfig",
        "ModelComposer",
        "EnsembleConfig",
        "ArchitectureValidator",
        "ArchitectureTemplates",
        # Training
        "Trainer",
        "Callback",
        "EarlyStopping",
        "ModelCheckpoint",
        "TensorBoardLogger",
        "LearningRateMonitor",
        "ProgressBar",
        "GradientMonitor",
        "CallbackFactory",
        "DataSplitter",
        "LossRegistry",
        "FocalLoss",
        "WeightedMSELoss",
        "RMSELoss",
        "OptimizerRegistry",
        "SchedulerRegistry",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", EXPECTED_CLASSES)
    def test_export_is_class(self, models_pkg, name):
        """Each expected class export is actually a class."""
        obj = getattr(models_pkg, name)
        assert inspect.isclass(obj), f"'{name}' should be a class, got {type(obj).__name__}"


class TestContractCallableTypes:
    """§2 — Re-exported functions/callables are callable."""

    EXPECTED_CALLABLES = [
        # Registry functions
        "get_model",
        "has_model",
        "list_models",
        "get_model_info",
        # Factory functions
        "create_model",
        "get_factory",
        # Introspector functions
        "get_model_metadata",
        "get_all_model_names",
        "get_models_by_category",
        "get_models_by_task",
        "get_models_by_tag",
        "search_models",
        "get_category_statistics",
        # Builder functions
        "validate_architecture",
        "validate_data_compatibility",
        "parse_custom_architecture",
        "parse_ensemble",
        "load_config",
        "validate_config",
        # Data splitting
        "random_split",
        "stratified_split",
        "temporal_split",
        "scaffold_split",
        "k_fold_split",
        # Loss functions
        "get_loss",
        "list_losses",
        # Optimizer functions
        "get_optimizer",
        "list_optimizers",
        # Scheduler functions
        "get_scheduler",
        "list_schedulers",
        "create_warmup_scheduler",
        # Convenience
        "get_module_info",
        "print_module_summary",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", EXPECTED_CALLABLES)
    def test_export_is_callable(self, models_pkg, name):
        """Each expected callable export is callable."""
        obj = getattr(models_pkg, name)
        assert callable(obj), f"'{name}' should be callable, got {type(obj).__name__}"


class TestContractExceptionHierarchy:
    """§2 — Exception classes are proper Exception subclasses."""

    EXCEPTION_CLASSES = [
        "ModelError",
        "ModelNotFoundError",
        "ModelValidationError",
        "ModelInstantiationError",
        "HyperparameterError",
        "DataCompatibilityError",
        "TrainingError",
        "CheckpointError",
        "DataError",
        "PluginModelError",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_is_exception_subclass(self, models_pkg, name):
        """Each exception class is a subclass of Exception."""
        cls = getattr(models_pkg, name)
        assert issubclass(cls, Exception), f"'{name}' should be a subclass of Exception"

    @pytest.mark.contract
    def test_model_error_is_base_for_others(self, models_pkg):
        """
        ``ModelError`` is the base class for model-specific exceptions.

        The following should be subclasses of ``ModelError``:
        ModelNotFoundError, ModelValidationError, ModelInstantiationError,
        HyperparameterError, DataCompatibilityError, PluginModelError.
        """
        model_error = models_pkg.ModelError
        expected_children = [
            "ModelNotFoundError",
            "ModelValidationError",
            "ModelInstantiationError",
            "HyperparameterError",
            "DataCompatibilityError",
            "PluginModelError",
        ]
        for child_name in expected_children:
            child_cls = getattr(models_pkg, child_name)
            assert issubclass(child_cls, model_error), (
                f"'{child_name}' should be a subclass of ModelError"
            )


class TestContractRegistryInstance:
    """§2 — The ``registry`` singleton has expected interface."""

    @pytest.mark.contract
    def test_registry_has_list_available_models(self, models_pkg):
        """``registry`` has ``list_available_models`` method."""
        reg = models_pkg.registry
        assert hasattr(reg, "list_available_models"), (
            "registry should have list_available_models method"
        )

    @pytest.mark.contract
    def test_registry_list_available_models_is_callable(self, models_pkg):
        """``registry.list_available_models`` is callable."""
        method = models_pkg.registry.list_available_models
        assert callable(method), "registry.list_available_models should be callable"

    @pytest.mark.contract
    def test_registry_list_available_models_returns_list(self, models_pkg):
        """``registry.list_available_models()`` returns a list."""
        result = models_pkg.registry.list_available_models()
        assert isinstance(result, list), (
            f"registry.list_available_models() should return list, got {type(result).__name__}"
        )


class TestContractGetModuleInfoReturnType:
    """§2 — ``get_module_info()`` returns a dict with expected keys."""

    @pytest.mark.contract
    def test_get_module_info_returns_dict(self, models_pkg):
        """``get_module_info()`` returns a dict."""
        result = models_pkg.get_module_info()
        assert isinstance(result, dict), (
            f"get_module_info() should return dict, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_module_info_has_version(self, models_pkg):
        """``get_module_info()`` result includes 'version' key."""
        result = models_pkg.get_module_info()
        assert "version" in result, "get_module_info() result missing 'version' key"

    @pytest.mark.contract
    def test_get_module_info_version_matches(self, models_pkg):
        """``get_module_info()['version']`` matches ``__version__``."""
        result = models_pkg.get_module_info()
        assert result["version"] == models_pkg.__version__, (
            f"get_module_info() version '{result['version']}' does not match "
            f"__version__ '{models_pkg.__version__}'"
        )

    @pytest.mark.contract
    def test_get_module_info_has_expected_keys_on_success(self, models_pkg):
        """
        ``get_module_info()`` result includes expected keys when all
        components are loaded successfully (no 'error' key).
        """
        result = models_pkg.get_module_info()
        if "error" not in result:
            expected_keys = {
                "version",
                "total_models",
                "categories",
                "category_breakdown",
                "available_losses",
                "available_optimizers",
                "available_schedulers",
                "builders_available",
                "hpo_available",
                "phase_7_features",
                "phase_8_features",
            }
            missing = expected_keys - set(result.keys())
            assert not missing, f"get_module_info() missing expected keys: {sorted(missing)}"

    @pytest.mark.contract
    def test_get_module_info_phase_7_features_structure(self, models_pkg):
        """
        ``get_module_info()['phase_7_features']`` is a dict with expected
        feature flag keys.
        """
        result = models_pkg.get_module_info()
        if "error" in result:
            pytest.skip("get_module_info() returned error")
        p7 = result.get("phase_7_features")
        assert isinstance(p7, dict), f"phase_7_features should be dict, got {type(p7).__name__}"
        expected_flags = {
            "custom_architectures",
            "ensemble_models",
            "architecture_templates",
            "layer_registry",
        }
        missing = expected_flags - set(p7.keys())
        assert not missing, f"phase_7_features missing keys: {sorted(missing)}"

    @pytest.mark.contract
    def test_get_module_info_phase_8_features_structure(self, models_pkg):
        """
        ``get_module_info()['phase_8_features']`` is a dict with expected
        feature flag keys.
        """
        result = models_pkg.get_module_info()
        if "error" in result:
            pytest.skip("get_module_info() returned error")
        p8 = result.get("phase_8_features")
        assert isinstance(p8, dict), f"phase_8_features should be dict, got {type(p8).__name__}"
        expected_flags = {
            "hyperparameter_optimization",
            "optuna_backend",
            "pruning_callbacks",
            "search_space_builder",
            "study_analyzer",
        }
        missing = expected_flags - set(p8.keys())
        assert not missing, f"phase_8_features missing keys: {sorted(missing)}"


class TestContractPrintModuleSummary:
    """§2 — ``print_module_summary()`` executes without error."""

    @pytest.mark.contract
    def test_print_module_summary_runs_without_error(self, models_pkg, capsys):
        """
        ``print_module_summary()`` runs without raising exceptions and
        produces output to stdout.
        """
        models_pkg.print_module_summary()
        captured = capsys.readouterr()
        assert len(captured.out) > 0, "print_module_summary() should produce stdout output"

    @pytest.mark.contract
    def test_print_module_summary_includes_version(self, models_pkg, capsys):
        """
        ``print_module_summary()`` output includes the module version.
        """
        models_pkg.print_module_summary()
        captured = capsys.readouterr()
        assert models_pkg.__version__ in captured.out, (
            f"print_module_summary() output should include version '{models_pkg.__version__}'"
        )


class TestContractHPOConditionalImports:
    """§2 — HPO conditional imports follow the correct pattern."""

    @pytest.mark.contract
    def test_hpo_available_consistent_with_missing_components(self, models_pkg):
        """
        If ``HPO_AVAILABLE`` is False, 'hpo' should be in
        ``_MISSING_COMPONENTS``, and vice versa.
        """
        if not models_pkg.HPO_AVAILABLE:
            assert "hpo" in models_pkg._MISSING_COMPONENTS, (
                "HPO_AVAILABLE is False but 'hpo' not in _MISSING_COMPONENTS"
            )
        else:
            assert "hpo" not in models_pkg._MISSING_COMPONENTS, (
                "HPO_AVAILABLE is True but 'hpo' in _MISSING_COMPONENTS"
            )

    @pytest.mark.contract
    def test_hpo_callable_exports_when_available(self, models_pkg):
        """
        When ``HPO_AVAILABLE`` is True, HPO function exports should be
        callable (not None).
        """
        if not models_pkg.HPO_AVAILABLE:
            pytest.skip("HPO not available")

        callable_names = [
            "is_hpo_enabled",
            "get_best_params",
            "create_hpo_manager",
            "infer_task_type",
        ]
        for name in callable_names:
            obj = getattr(models_pkg, name)
            assert callable(obj), f"HPO function '{name}' should be callable when HPO_AVAILABLE"


class TestContractFallbackStubs:
    """§2 — Fallback stubs for missing imports have correct structure.

    When a submodule import fails, the ``__init__.py`` defines fallback
    classes/functions. This test verifies the fallback contract:
    - Fallback classes are classes
    - Fallback functions raise ``NotImplementedError`` or return safe defaults
    """

    @pytest.mark.contract
    def test_registry_fallback_list_available_models(self, models_pkg):
        """
        If the real ModelRegistry is not loaded,
        ``registry.list_available_models()`` returns a list (possibly empty)
        rather than crashing.
        """
        result = models_pkg.registry.list_available_models()
        assert isinstance(result, list), (
            f"registry.list_available_models() should return list even in "
            f"fallback mode, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_category_statistics_returns_dict(self, models_pkg):
        """
        ``get_category_statistics()`` returns a dict (even in fallback mode).
        """
        result = models_pkg.get_category_statistics()
        assert isinstance(result, dict), (
            f"get_category_statistics() should return dict, got {type(result).__name__}"
        )


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    MINIMUM_API = {
        # Version
        "__version__",
        # Registry
        "ModelRegistry",
        "ModelRegistration",
        "registry",
        "get_model",
        "has_model",
        "list_models",
        "get_model_info",
        # Factory
        "ModelFactory",
        "ModelValidator",
        "create_model",
        "get_factory",
        # Categories
        "ModelCategory",
        "ModelMetadata",
        "get_model_metadata",
        "get_all_model_names",
        "search_models",
        "ALL_MODELS",
        "MODELS_BY_CATEGORY",
        # Training
        "Trainer",
        "Callback",
        "EarlyStopping",
        "ModelCheckpoint",
        "DataSplitter",
        "random_split",
        # Loss/Optimizer/Scheduler
        "LossRegistry",
        "get_loss",
        "OptimizerRegistry",
        "get_optimizer",
        "SchedulerRegistry",
        "get_scheduler",
        # Exceptions
        "ModelError",
        "ModelNotFoundError",
        "TrainingError",
        "CheckpointError",
        # HPO
        "HPO_AVAILABLE",
        "HPOManager",
        "HPOConfig",
        # Builders
        "ArchitectureBuilder",
        "ModelComposer",
        "LayerRegistry",
    }

    @pytest.mark.contract
    def test_minimum_api_in_all(self, all_names):
        """The minimum expected public API is present in ``__all__``."""
        all_set = set(all_names)
        missing = self.MINIMUM_API - all_set
        assert not missing, f"Minimum API names missing from __all__: {sorted(missing)}"

    @pytest.mark.contract
    def test_all_has_expected_length(self, all_names):
        """
        ``__all__`` contains a substantial number of entries.

        Based on the __init__.py source, the models package exports ~80+
        names. This test guards against catastrophic loss (e.g., accidental
        truncation of __all__) while allowing for organic growth.
        """
        actual = len(all_names)
        # The __init__.py has ~80 entries in __all__ (lines 522-663)
        # We set a floor well below the actual count to allow changes
        # while catching catastrophic loss.
        MINIMUM_EXPECTED = 60
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least {MINIMUM_EXPECTED}. "
            f"This suggests __all__ may have been accidentally truncated."
        )


class TestContractCategorySections:
    """§2 — __all__ contains entries organized by the expected category sections."""

    @pytest.mark.contract
    def test_all_has_version_section(self, all_names):
        """``__all__`` includes the __version__ entry."""
        assert "__version__" in all_names

    @pytest.mark.contract
    def test_all_has_registry_section(self, all_names):
        """``__all__`` includes core registry entries."""
        registry_entries = {"ModelRegistry", "registry", "get_model", "list_models"}
        all_set = set(all_names)
        missing = registry_entries - all_set
        assert not missing, f"Registry section missing entries: {sorted(missing)}"

    @pytest.mark.contract
    def test_all_has_factory_section(self, all_names):
        """``__all__`` includes core factory entries."""
        factory_entries = {"ModelFactory", "create_model", "get_factory"}
        all_set = set(all_names)
        missing = factory_entries - all_set
        assert not missing, f"Factory section missing entries: {sorted(missing)}"

    @pytest.mark.contract
    def test_all_has_training_section(self, all_names):
        """``__all__`` includes core training entries."""
        training_entries = {"Trainer", "EarlyStopping", "ModelCheckpoint", "DataSplitter"}
        all_set = set(all_names)
        missing = training_entries - all_set
        assert not missing, f"Training section missing entries: {sorted(missing)}"

    @pytest.mark.contract
    def test_all_has_exceptions_section(self, all_names):
        """``__all__`` includes core exception entries."""
        exception_entries = {"ModelError", "ModelNotFoundError", "TrainingError", "CheckpointError"}
        all_set = set(all_names)
        missing = exception_entries - all_set
        assert not missing, f"Exceptions section missing entries: {sorted(missing)}"

    @pytest.mark.contract
    def test_all_has_hpo_section(self, all_names):
        """``__all__`` includes HPO entries."""
        hpo_entries = {"HPO_AVAILABLE", "HPOManager", "HPOConfig"}
        all_set = set(all_names)
        missing = hpo_entries - all_set
        assert not missing, f"HPO section missing entries: {sorted(missing)}"

    @pytest.mark.contract
    def test_all_has_builders_section(self, all_names):
        """``__all__`` includes builders entries."""
        builder_entries = {
            "LayerRegistry",
            "ArchitectureBuilder",
            "ModelComposer",
            "ArchitectureTemplates",
        }
        all_set = set(all_names)
        missing = builder_entries - all_set
        assert not missing, f"Builders section missing entries: {sorted(missing)}"


class TestContractLossRegistryInterface:
    """§2 — LossRegistry has expected static interface."""

    @pytest.mark.contract
    def test_loss_registry_has_list_available(self, models_pkg):
        """``LossRegistry`` has ``list_available`` static method."""
        assert hasattr(models_pkg.LossRegistry, "list_available"), (
            "LossRegistry should have list_available method"
        )

    @pytest.mark.contract
    def test_loss_registry_list_available_returns_list(self, models_pkg):
        """``LossRegistry.list_available()`` returns a list."""
        result = models_pkg.LossRegistry.list_available()
        assert isinstance(result, list), (
            f"LossRegistry.list_available() should return list, got {type(result).__name__}"
        )


class TestContractOptimizerRegistryInterface:
    """§2 — OptimizerRegistry has expected static interface."""

    @pytest.mark.contract
    def test_optimizer_registry_has_list_available(self, models_pkg):
        """``OptimizerRegistry`` has ``list_available`` static method."""
        assert hasattr(models_pkg.OptimizerRegistry, "list_available"), (
            "OptimizerRegistry should have list_available method"
        )

    @pytest.mark.contract
    def test_optimizer_registry_list_available_returns_list(self, models_pkg):
        """``OptimizerRegistry.list_available()`` returns a list."""
        result = models_pkg.OptimizerRegistry.list_available()
        assert isinstance(result, list), (
            f"OptimizerRegistry.list_available() should return list, got {type(result).__name__}"
        )


class TestContractSchedulerRegistryInterface:
    """§2 — SchedulerRegistry has expected static interface."""

    @pytest.mark.contract
    def test_scheduler_registry_has_list_available(self, models_pkg):
        """``SchedulerRegistry`` has ``list_available`` static method."""
        assert hasattr(models_pkg.SchedulerRegistry, "list_available"), (
            "SchedulerRegistry should have list_available method"
        )

    @pytest.mark.contract
    def test_scheduler_registry_list_available_returns_list(self, models_pkg):
        """``SchedulerRegistry.list_available()`` returns a list."""
        result = models_pkg.SchedulerRegistry.list_available()
        assert isinstance(result, list), (
            f"SchedulerRegistry.list_available() should return list, got {type(result).__name__}"
        )


class TestContractWrapperClasses:
    """§2 — GraphLevelModelWrapper and EdgeLevelModelWrapper are proper classes."""

    @pytest.mark.contract
    def test_graph_level_wrapper_is_class(self, models_pkg):
        """``GraphLevelModelWrapper`` is a class."""
        assert inspect.isclass(models_pkg.GraphLevelModelWrapper)

    @pytest.mark.contract
    def test_edge_level_wrapper_is_class(self, models_pkg):
        """``EdgeLevelModelWrapper`` is a class."""
        assert inspect.isclass(models_pkg.EdgeLevelModelWrapper)


class TestContractConvenienceFunctionSignatures:
    """§2 — Convenience functions have expected parameter signatures."""

    @pytest.mark.contract
    def test_get_module_info_takes_no_args(self, models_pkg):
        """``get_module_info()`` accepts zero required arguments."""
        sig = inspect.signature(models_pkg.get_module_info)
        required = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required) == 0, (
            f"get_module_info() should take no required args, found: {[p.name for p in required]}"
        )

    @pytest.mark.contract
    def test_print_module_summary_takes_no_args(self, models_pkg):
        """``print_module_summary()`` accepts zero required arguments."""
        sig = inspect.signature(models_pkg.print_module_summary)
        required = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required) == 0, (
            f"print_module_summary() should take no required args, "
            f"found: {[p.name for p in required]}"
        )

    @pytest.mark.contract
    def test_get_module_info_is_function(self, models_pkg):
        """``get_module_info`` is a function (not a class or method)."""
        assert inspect.isfunction(models_pkg.get_module_info)

    @pytest.mark.contract
    def test_print_module_summary_is_function(self, models_pkg):
        """``print_module_summary`` is a function (not a class or method)."""
        assert inspect.isfunction(models_pkg.print_module_summary)


class TestContractMissingComponentsIntegrity:
    """§2 — _MISSING_COMPONENTS list contains only strings."""

    @pytest.mark.contract
    def test_missing_components_entries_are_strings(self, models_pkg):
        """Every entry in ``_MISSING_COMPONENTS`` is a string."""
        for item in models_pkg._MISSING_COMPONENTS:
            assert isinstance(item, str), (
                f"_MISSING_COMPONENTS entry should be str, got {type(item).__name__}: {item}"
            )

    @pytest.mark.contract
    def test_missing_components_no_duplicates(self, models_pkg):
        """``_MISSING_COMPONENTS`` has no duplicate entries."""
        items = models_pkg._MISSING_COMPONENTS
        duplicates = [item for item in items if items.count(item) > 1]
        assert not duplicates, f"Duplicate entries in _MISSING_COMPONENTS: {set(duplicates)}"


class TestContractModuleDocstring:
    """§2 — Module docstring is comprehensive and includes key components."""

    @pytest.mark.contract
    def test_docstring_mentions_pyg(self, models_pkg):
        """Module docstring mentions PyTorch Geometric."""
        doc = models_pkg.__doc__
        assert "PyTorch Geometric" in doc or "PyG" in doc, (
            "Module docstring should mention PyTorch Geometric or PyG"
        )

    @pytest.mark.contract
    def test_docstring_mentions_phase_7(self, models_pkg):
        """Module docstring mentions Phase 7 (builders)."""
        doc = models_pkg.__doc__
        assert "PHASE 7" in doc or "Phase 7" in doc, "Module docstring should mention Phase 7"

    @pytest.mark.contract
    def test_docstring_mentions_phase_8(self, models_pkg):
        """Module docstring mentions Phase 8 (HPO)."""
        doc = models_pkg.__doc__
        assert "PHASE 8" in doc or "Phase 8" in doc, "Module docstring should mention Phase 8"

    @pytest.mark.contract
    def test_docstring_mentions_trainer(self, models_pkg):
        """Module docstring mentions Trainer."""
        doc = models_pkg.__doc__
        assert "Trainer" in doc, "Module docstring should mention Trainer"

    @pytest.mark.contract
    def test_docstring_mentions_model_registry(self, models_pkg):
        """Module docstring mentions ModelRegistry."""
        doc = models_pkg.__doc__
        assert "ModelRegistry" in doc, "Module docstring should mention ModelRegistry"
