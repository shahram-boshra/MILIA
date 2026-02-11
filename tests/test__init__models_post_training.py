# tests/test__init__models_post_training.py

"""
Test Suite: milia_pipeline/models/post_training/__init__.py — Smoke Tests & Contract Tests
===========================================================================================

Production-ready test suite for the MILIA Pipeline post-training package
``milia_pipeline/models/post_training/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.models.post_training`` subpackage imports without
          ImportError or CircularImportError
        - All re-exported names from checkpoint, inference, data_preparation,
          and transfer_learning submodules are accessible
        - Module-level metadata attributes (__version__, __author__) exist
        - Module initialization (logging, availability flags) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Availability flags (_DATA_PREPARATION_AVAILABLE,
          _TRANSFER_LEARNING_AVAILABLE) are present and boolean
        - Convenience functions (get_available_components,
          print_available_components, get_implementation_status) execute
          without exceptions

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, callables are callable
        - CheckpointManager, ModelLoader, Predictor are classes
        - load_model, load_model_only, predict are callable functions
        - CHECKPOINT_FORMAT_VERSION is a non-empty string
        - Conditional exports (data_preparation, transfer_learning) follow
          availability flags
        - get_available_components() returns Dict[str, List[str]]
        - get_implementation_status() returns Dict[str, bool]
        - Dependency Injection pattern: path_utils removed (no resolve_path)
        - __version__ follows semver pattern (MAJOR.MINOR.PATCH)
        - Public API surface stability (minimum expected names present)
        - Sectioned __all__ sub-lists consistency

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__models_post_training.py -v --tb=short

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
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__models_post_training.py``
# from the project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def post_training_pkg():
    """
    Import and return the ``milia_pipeline.models.post_training`` package
    once per module.

    This fixture validates the fundamental smoke invariant: the post_training
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.models.post_training as pt
        return pt
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.models.post_training could not be imported — "
            f"smoke test precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(post_training_pkg):
    """Return the ``__all__`` list from the post_training package."""
    assert hasattr(post_training_pkg, "__all__"), (
        "milia_pipeline.models.post_training.__all__ is missing — "
        "contract violation"
    )
    return list(post_training_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokePostTrainingPackageImport:
    """§1.2 — Verify the post_training subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_post_training_package_succeeds(self, post_training_pkg):
        """The post_training package imports without raising any exception."""
        assert post_training_pkg is not None

    @pytest.mark.smoke
    def test_post_training_package_is_a_module(self, post_training_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(post_training_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_post_training_package_has_file_attribute(self, post_training_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(post_training_pkg, "__file__")

    @pytest.mark.smoke
    def test_post_training_package_name(self, post_training_pkg):
        """The package ``__name__`` is ``milia_pipeline.models.post_training``."""
        assert post_training_pkg.__name__ == "milia_pipeline.models.post_training"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
    ])
    def test_metadata_attribute_exists(self, post_training_pkg, attr):
        """Each metadata dunder is defined on the post_training package."""
        assert hasattr(post_training_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
    ])
    def test_metadata_attribute_is_string(self, post_training_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(post_training_pkg, attr)
        assert isinstance(value, str), (
            f"{attr} should be str, got {type(value)}"
        )
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, post_training_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = post_training_pkg.__version__
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


class TestSmokeCheckpointExports:
    """§1.2 — Checkpoint management exports (Phase 1) are accessible."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", [
        "CheckpointManager",
        "CHECKPOINT_FORMAT_VERSION",
    ])
    def test_checkpoint_export_exists(self, post_training_pkg, name):
        """Each checkpoint export is present and non-None."""
        obj = getattr(post_training_pkg, name, None)
        assert obj is not None, (
            f"Checkpoint export '{name}' is None or missing"
        )


class TestSmokeInferenceExports:
    """§1.2 — Model loading and inference exports (Phase 2) are accessible."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", [
        "ModelLoader",
        "load_model",
        "load_model_only",
        "Predictor",
        "predict",
    ])
    def test_inference_export_exists(self, post_training_pkg, name):
        """Each inference export is present and non-None."""
        obj = getattr(post_training_pkg, name, None)
        assert obj is not None, (
            f"Inference export '{name}' is None or missing"
        )


class TestSmokeAvailabilityFlags:
    """§1.2 — Availability flags exist and are boolean."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", [
        "_DATA_PREPARATION_AVAILABLE",
        "_TRANSFER_LEARNING_AVAILABLE",
    ])
    def test_availability_flag_exists(self, post_training_pkg, flag):
        """Each availability flag is defined on the post_training package."""
        assert hasattr(post_training_pkg, flag), (
            f"Availability flag '{flag}' is missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", [
        "_DATA_PREPARATION_AVAILABLE",
        "_TRANSFER_LEARNING_AVAILABLE",
    ])
    def test_availability_flag_is_bool(self, post_training_pkg, flag):
        """Each availability flag is a boolean."""
        value = getattr(post_training_pkg, flag)
        assert isinstance(value, bool), (
            f"Flag '{flag}' should be bool, got {type(value).__name__}"
        )


class TestSmokeConditionalDataPreparationExports:
    """§1.2 — Data preparation exports (Phase 3) follow availability flag."""

    DATA_PREPARATION_NAMES = [
        # Protocol
        "DataConverterProtocol",
        # Registry
        "DataConverterRegistry",
        "get_registry",
        "register_converter",
        # Base class
        "BaseDataConverter",
        # Built-in converters
        "PyGDataConverter",
        "DictConverter",
        "SMILESConverter",
        "InChIConverter",
        "XYZConverter",
        "ASEAtomsConverter",
        "SDFConverter",
        # Convenience functions
        "convert_to_pyg",
        "convert_batch_to_pyg",
        "convert_sdf_to_pyg_list",
        "list_available_formats",
        "list_all_formats",
        "smiles_to_data",
    ]

    @pytest.mark.smoke
    def test_data_prep_exports_follow_flag(self, post_training_pkg):
        """
        If ``_DATA_PREPARATION_AVAILABLE`` is True, all data preparation
        exports must be resolvable. If False, they should not be in
        ``__all__``.
        """
        available = post_training_pkg._DATA_PREPARATION_AVAILABLE
        if available:
            for name in self.DATA_PREPARATION_NAMES:
                obj = getattr(post_training_pkg, name, None)
                assert obj is not None, (
                    f"Data preparation export '{name}' is None but "
                    f"_DATA_PREPARATION_AVAILABLE is True"
                )
        else:
            # When unavailable, these should NOT be in __all__
            all_set = set(post_training_pkg.__all__)
            for name in self.DATA_PREPARATION_NAMES:
                assert name not in all_set, (
                    f"'{name}' found in __all__ but "
                    f"_DATA_PREPARATION_AVAILABLE is False"
                )


class TestSmokeConditionalTransferLearningExports:
    """§1.2 — Transfer learning exports (Phase 4) follow availability flag."""

    TRANSFER_LEARNING_NAMES = [
        "FineTuner",
        "FreezeStrategy",
    ]

    @pytest.mark.smoke
    def test_transfer_learning_exports_follow_flag(self, post_training_pkg):
        """
        If ``_TRANSFER_LEARNING_AVAILABLE`` is True, all transfer learning
        exports must be resolvable. If False, they should not be in
        ``__all__``.
        """
        available = post_training_pkg._TRANSFER_LEARNING_AVAILABLE
        if available:
            for name in self.TRANSFER_LEARNING_NAMES:
                obj = getattr(post_training_pkg, name, None)
                assert obj is not None, (
                    f"Transfer learning export '{name}' is None but "
                    f"_TRANSFER_LEARNING_AVAILABLE is True"
                )
        else:
            all_set = set(post_training_pkg.__all__)
            for name in self.TRANSFER_LEARNING_NAMES:
                assert name not in all_set, (
                    f"'{name}' found in __all__ but "
                    f"_TRANSFER_LEARNING_AVAILABLE is False"
                )


class TestSmokeSectionedAllSubLists:
    """§1.2 — Sectioned ``__all__`` sub-lists are defined and are lists."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("sublist_name", [
        "__all_path_utils__",
        "__all_checkpoint__",
        "__all_inference__",
        "__all_data_preparation__",
        "__all_transfer_learning__",
    ])
    def test_all_sublist_exists(self, post_training_pkg, sublist_name):
        """Each sectioned __all__ sub-list is defined."""
        assert hasattr(post_training_pkg, sublist_name), (
            f"Sectioned sub-list '{sublist_name}' is missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("sublist_name", [
        "__all_path_utils__",
        "__all_checkpoint__",
        "__all_inference__",
        "__all_data_preparation__",
        "__all_transfer_learning__",
    ])
    def test_all_sublist_is_a_list(self, post_training_pkg, sublist_name):
        """Each sectioned __all__ sub-list is a list."""
        obj = getattr(post_training_pkg, sublist_name)
        assert isinstance(obj, list), (
            f"'{sublist_name}' should be a list, got {type(obj).__name__}"
        )


class TestSmokeConvenienceFunctions:
    """§1.2 — Module-level convenience functions are accessible and callable."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", [
        "get_available_components",
        "print_available_components",
        "get_implementation_status",
    ])
    def test_convenience_function_exists(self, post_training_pkg, name):
        """Each convenience function is present and non-None."""
        obj = getattr(post_training_pkg, name, None)
        assert obj is not None, (
            f"Convenience function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", [
        "get_available_components",
        "print_available_components",
        "get_implementation_status",
    ])
    def test_convenience_function_is_callable(self, post_training_pkg, name):
        """Each convenience function is callable."""
        obj = getattr(post_training_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeGetAvailableComponentsExecution:
    """§1.2 — ``get_available_components()`` executes without crashing."""

    @pytest.mark.smoke
    def test_get_available_components_runs(self, post_training_pkg):
        """``get_available_components()`` runs without exception."""
        result = post_training_pkg.get_available_components()
        assert result is not None

    @pytest.mark.smoke
    def test_get_available_components_returns_dict(self, post_training_pkg):
        """``get_available_components()`` returns a dictionary."""
        result = post_training_pkg.get_available_components()
        assert isinstance(result, dict), (
            f"get_available_components() should return dict, "
            f"got {type(result).__name__}"
        )


class TestSmokeGetImplementationStatusExecution:
    """§1.2 — ``get_implementation_status()`` executes without crashing."""

    @pytest.mark.smoke
    def test_get_implementation_status_runs(self, post_training_pkg):
        """``get_implementation_status()`` runs without exception."""
        result = post_training_pkg.get_implementation_status()
        assert result is not None

    @pytest.mark.smoke
    def test_get_implementation_status_returns_dict(self, post_training_pkg):
        """``get_implementation_status()`` returns a dictionary."""
        result = post_training_pkg.get_implementation_status()
        assert isinstance(result, dict), (
            f"get_implementation_status() should return dict, "
            f"got {type(result).__name__}"
        )


class TestSmokePrintAvailableComponentsExecution:
    """§1.2 — ``print_available_components()`` executes without crashing."""

    @pytest.mark.smoke
    def test_print_available_components_runs(self, post_training_pkg, capsys):
        """``print_available_components()`` runs without exception and
        produces output."""
        post_training_pkg.print_available_components()
        captured = capsys.readouterr()
        assert len(captured.out) > 0, (
            "print_available_components() should produce console output"
        )


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, post_training_pkg):
        """
        Re-importing the post_training package (via ``importlib.reload``)
        does not crash.

        Validates that all module-level code (logging, availability checks)
        is safe to re-execute.
        """
        reloaded = importlib.reload(post_training_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, post_training_pkg):
        """
        Re-importing the post_training package preserves ``__all__``.
        """
        reloaded = importlib.reload(post_training_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, (list, tuple))
        assert len(reloaded.__all__) > 0


class TestSmokeLoggerExists:
    """§1.2 — Module-level logger is defined."""

    @pytest.mark.smoke
    def test_logger_attribute_exists(self, post_training_pkg):
        """The module defines a ``logger`` attribute."""
        assert hasattr(post_training_pkg, "logger")

    @pytest.mark.smoke
    def test_logger_is_logging_logger(self, post_training_pkg):
        """The ``logger`` is a ``logging.Logger`` instance."""
        logger_obj = post_training_pkg.logger
        assert isinstance(logger_obj, logging.Logger), (
            f"logger should be logging.Logger, got {type(logger_obj).__name__}"
        )


class TestSmokePathUtilsRemoved:
    """§1.2 — Path utilities have been removed (v2.0.0 DI refactoring)."""

    @pytest.mark.smoke
    def test_resolve_path_not_exported(self, post_training_pkg):
        """``resolve_path`` should NOT be available (removed in v2.0.0)."""
        assert not hasattr(post_training_pkg, "resolve_path"), (
            "resolve_path should have been removed in v2.0.0 "
            "(Dependency Injection refactoring)"
        )

    @pytest.mark.smoke
    def test_path_utils_sublist_is_empty(self, post_training_pkg):
        """``__all_path_utils__`` should be an empty list (v2.0.0)."""
        sublist = post_training_pkg.__all_path_utils__
        assert sublist == [], (
            f"__all_path_utils__ should be [] (v2.0.0), got {sublist}"
        )


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the post_training
    package."""

    @pytest.mark.contract
    def test_all_is_a_list_or_tuple(self, post_training_pkg):
        """``__all__`` is a list or tuple."""
        assert isinstance(post_training_pkg.__all__, (list, tuple))

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
    def test_every_all_entry_is_resolvable(self, post_training_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [
            name for name in all_names
            if not hasattr(post_training_pkg, name)
        ]
        assert not unresolvable, (
            f"Names in __all__ that are not defined on the module: "
            f"{unresolvable}"
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
    """§2 — Every public import in the post_training module is listed in
    ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Metadata dunders
        "__version__",
        "__author__",
        # Internal availability flags
        "_DATA_PREPARATION_AVAILABLE",
        "_TRANSFER_LEARNING_AVAILABLE",
        # Sectioned sub-lists (not public API symbols)
        "__all_path_utils__",
        "__all_checkpoint__",
        "__all_inference__",
        "__all_data_preparation__",
        "__all_transfer_learning__",
        # Logger
        "logger",
        # Typing imports used in function annotations but not
        # cleaned from namespace (Dict, Any, List from typing)
        "Dict",
        "Any",
        "List",
        # Module-level convenience functions defined locally but
        # not included in __all__ (they are introspection helpers,
        # not part of the core post-training public API)
        "get_available_components",
        "print_available_components",
        "get_implementation_status",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, post_training_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the post_training ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(post_training_pkg)
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
            # Skip private names that start with underscore and are
            # NOT in __all__
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
            f"Public names imported in post_training/__init__.py but not "
            f"in __all__: {sorted(missing_from_all)}"
        )


class TestContractSectionedSubListComposition:
    """§2 — ``__all__`` equals the concatenation of sectioned sub-lists."""

    @pytest.mark.contract
    def test_all_equals_concatenated_sublists(self, post_training_pkg):
        """
        ``__all__`` is composed by concatenating __all_path_utils__,
        __all_checkpoint__, __all_inference__, __all_data_preparation__,
        and __all_transfer_learning__.
        """
        expected = (
            post_training_pkg.__all_path_utils__
            + post_training_pkg.__all_checkpoint__
            + post_training_pkg.__all_inference__
            + post_training_pkg.__all_data_preparation__
            + post_training_pkg.__all_transfer_learning__
        )
        actual = list(post_training_pkg.__all__)
        assert actual == expected, (
            f"__all__ does not equal the concatenation of its sub-lists. "
            f"Difference: expected {len(expected)} items, got {len(actual)}"
        )


class TestContractCheckpointClassTypes:
    """§2 — Checkpoint management exports have expected types."""

    @pytest.mark.contract
    def test_checkpoint_manager_is_class(self, post_training_pkg):
        """``CheckpointManager`` is a class."""
        assert inspect.isclass(post_training_pkg.CheckpointManager), (
            f"CheckpointManager should be a class, got "
            f"{type(post_training_pkg.CheckpointManager).__name__}"
        )

    @pytest.mark.contract
    def test_checkpoint_format_version_is_string(self, post_training_pkg):
        """``CHECKPOINT_FORMAT_VERSION`` is a non-empty string."""
        v = post_training_pkg.CHECKPOINT_FORMAT_VERSION
        assert isinstance(v, str), (
            f"CHECKPOINT_FORMAT_VERSION should be str, got "
            f"{type(v).__name__}"
        )
        assert len(v) > 0, (
            "CHECKPOINT_FORMAT_VERSION should be non-empty"
        )


class TestContractInferenceClassTypes:
    """§2 — Inference exports have expected types (class vs function)."""

    @pytest.mark.contract
    def test_model_loader_is_class(self, post_training_pkg):
        """``ModelLoader`` is a class."""
        assert inspect.isclass(post_training_pkg.ModelLoader), (
            f"ModelLoader should be a class, got "
            f"{type(post_training_pkg.ModelLoader).__name__}"
        )

    @pytest.mark.contract
    def test_predictor_is_class(self, post_training_pkg):
        """``Predictor`` is a class."""
        assert inspect.isclass(post_training_pkg.Predictor), (
            f"Predictor should be a class, got "
            f"{type(post_training_pkg.Predictor).__name__}"
        )

    @pytest.mark.contract
    def test_load_model_is_callable(self, post_training_pkg):
        """``load_model`` is callable."""
        assert callable(post_training_pkg.load_model), (
            "load_model should be callable"
        )

    @pytest.mark.contract
    def test_load_model_only_is_callable(self, post_training_pkg):
        """``load_model_only`` is callable."""
        assert callable(post_training_pkg.load_model_only), (
            "load_model_only should be callable"
        )

    @pytest.mark.contract
    def test_predict_is_callable(self, post_training_pkg):
        """``predict`` is callable."""
        assert callable(post_training_pkg.predict), (
            "predict should be callable"
        )


class TestContractConditionalDataPreparationTypes:
    """§2 — Data preparation exports have expected types when available."""

    @pytest.mark.contract
    def test_data_preparation_protocol_is_class(self, post_training_pkg):
        """If available, ``DataConverterProtocol`` is a class."""
        if not post_training_pkg._DATA_PREPARATION_AVAILABLE:
            pytest.skip("Data preparation not available")
        assert inspect.isclass(post_training_pkg.DataConverterProtocol)

    @pytest.mark.contract
    def test_data_converter_registry_is_class(self, post_training_pkg):
        """If available, ``DataConverterRegistry`` is a class."""
        if not post_training_pkg._DATA_PREPARATION_AVAILABLE:
            pytest.skip("Data preparation not available")
        assert inspect.isclass(post_training_pkg.DataConverterRegistry)

    @pytest.mark.contract
    def test_base_data_converter_is_class(self, post_training_pkg):
        """If available, ``BaseDataConverter`` is a class."""
        if not post_training_pkg._DATA_PREPARATION_AVAILABLE:
            pytest.skip("Data preparation not available")
        assert inspect.isclass(post_training_pkg.BaseDataConverter)

    @pytest.mark.contract
    @pytest.mark.parametrize("name", [
        "PyGDataConverter",
        "DictConverter",
        "SMILESConverter",
        "InChIConverter",
        "XYZConverter",
        "ASEAtomsConverter",
        "SDFConverter",
    ])
    def test_builtin_converter_is_class(self, post_training_pkg, name):
        """Each built-in converter is a class when data_preparation is
        available."""
        if not post_training_pkg._DATA_PREPARATION_AVAILABLE:
            pytest.skip("Data preparation not available")
        obj = getattr(post_training_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", [
        "get_registry",
        "register_converter",
        "convert_to_pyg",
        "convert_batch_to_pyg",
        "convert_sdf_to_pyg_list",
        "list_available_formats",
        "list_all_formats",
        "smiles_to_data",
    ])
    def test_data_prep_function_is_callable(self, post_training_pkg, name):
        """Each data preparation function is callable when available."""
        if not post_training_pkg._DATA_PREPARATION_AVAILABLE:
            pytest.skip("Data preparation not available")
        obj = getattr(post_training_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestContractConditionalTransferLearningTypes:
    """§2 — Transfer learning exports have expected types when available."""

    @pytest.mark.contract
    def test_fine_tuner_is_class(self, post_training_pkg):
        """If available, ``FineTuner`` is a class."""
        if not post_training_pkg._TRANSFER_LEARNING_AVAILABLE:
            pytest.skip("Transfer learning not available")
        assert inspect.isclass(post_training_pkg.FineTuner), (
            f"FineTuner should be a class, got "
            f"{type(post_training_pkg.FineTuner).__name__}"
        )

    @pytest.mark.contract
    def test_freeze_strategy_is_class(self, post_training_pkg):
        """If available, ``FreezeStrategy`` is a class (enum)."""
        if not post_training_pkg._TRANSFER_LEARNING_AVAILABLE:
            pytest.skip("Transfer learning not available")
        assert inspect.isclass(post_training_pkg.FreezeStrategy), (
            f"FreezeStrategy should be a class, got "
            f"{type(post_training_pkg.FreezeStrategy).__name__}"
        )


class TestContractGetAvailableComponentsContract:
    """§2 — ``get_available_components()`` return type and structure contract."""

    @pytest.mark.contract
    def test_returns_dict(self, post_training_pkg):
        """``get_available_components()`` returns a dict."""
        result = post_training_pkg.get_available_components()
        assert isinstance(result, dict)

    @pytest.mark.contract
    def test_has_expected_keys(self, post_training_pkg):
        """``get_available_components()`` contains the four expected keys."""
        result = post_training_pkg.get_available_components()
        expected_keys = {
            "checkpoint",
            "inference",
            "data_preparation",
            "transfer_learning",
        }
        assert expected_keys == set(result.keys()), (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    @pytest.mark.contract
    def test_values_are_lists_of_strings(self, post_training_pkg):
        """Each value in ``get_available_components()`` is a list of strings."""
        result = post_training_pkg.get_available_components()
        for key, value in result.items():
            assert isinstance(value, list), (
                f"Value for key '{key}' should be list, "
                f"got {type(value).__name__}"
            )
            for item in value:
                assert isinstance(item, str), (
                    f"Each item in '{key}' should be str, "
                    f"got {type(item).__name__}"
                )

    @pytest.mark.contract
    def test_checkpoint_components_nonempty(self, post_training_pkg):
        """Checkpoint components are always non-empty (always available)."""
        result = post_training_pkg.get_available_components()
        assert len(result["checkpoint"]) > 0, (
            "Checkpoint components should always be non-empty"
        )

    @pytest.mark.contract
    def test_inference_components_nonempty(self, post_training_pkg):
        """Inference components are always non-empty (always available)."""
        result = post_training_pkg.get_available_components()
        assert len(result["inference"]) > 0, (
            "Inference components should always be non-empty"
        )

    @pytest.mark.contract
    def test_data_prep_matches_flag(self, post_training_pkg):
        """
        ``data_preparation`` list is non-empty iff
        ``_DATA_PREPARATION_AVAILABLE`` is True.
        """
        result = post_training_pkg.get_available_components()
        available = post_training_pkg._DATA_PREPARATION_AVAILABLE
        if available:
            assert len(result["data_preparation"]) > 0
        else:
            assert len(result["data_preparation"]) == 0

    @pytest.mark.contract
    def test_transfer_learning_matches_flag(self, post_training_pkg):
        """
        ``transfer_learning`` list is non-empty iff
        ``_TRANSFER_LEARNING_AVAILABLE`` is True.
        """
        result = post_training_pkg.get_available_components()
        available = post_training_pkg._TRANSFER_LEARNING_AVAILABLE
        if available:
            assert len(result["transfer_learning"]) > 0
        else:
            assert len(result["transfer_learning"]) == 0


class TestContractGetImplementationStatusContract:
    """§2 — ``get_implementation_status()`` return type and structure
    contract."""

    @pytest.mark.contract
    def test_returns_dict(self, post_training_pkg):
        """``get_implementation_status()`` returns a dict."""
        result = post_training_pkg.get_implementation_status()
        assert isinstance(result, dict)

    @pytest.mark.contract
    def test_values_are_booleans(self, post_training_pkg):
        """All values in ``get_implementation_status()`` are booleans."""
        result = post_training_pkg.get_implementation_status()
        for key, value in result.items():
            assert isinstance(value, bool), (
                f"Value for key '{key}' should be bool, "
                f"got {type(value).__name__}"
            )

    @pytest.mark.contract
    def test_has_expected_phase_keys(self, post_training_pkg):
        """``get_implementation_status()`` contains expected phase entries."""
        result = post_training_pkg.get_implementation_status()
        # At minimum, Phases 1, 2, and 7 should always be True
        keys = list(result.keys())
        assert len(keys) >= 3, (
            f"get_implementation_status() should have at least 3 phases, "
            f"got {len(keys)}"
        )

    @pytest.mark.contract
    def test_core_phases_always_true(self, post_training_pkg):
        """
        Phase 1 (Checkpoint), Phase 2 (Inference), and Phase 7 (DI)
        should always be True per the __init__.py source.
        """
        result = post_training_pkg.get_implementation_status()
        # Find phases by substring matching (keys are descriptive strings)
        for key, value in result.items():
            if "Checkpoint" in key or "Inference" in key or "Dependency" in key:
                assert value is True, (
                    f"Core phase '{key}' should always be True, got {value}"
                )

    @pytest.mark.contract
    def test_data_prep_phase_matches_flag(self, post_training_pkg):
        """
        Phase 3 (Data Preparation) status matches
        ``_DATA_PREPARATION_AVAILABLE``.
        """
        result = post_training_pkg.get_implementation_status()
        expected = post_training_pkg._DATA_PREPARATION_AVAILABLE
        for key, value in result.items():
            if "Data Preparation" in key:
                assert value == expected, (
                    f"Phase 3 status should match "
                    f"_DATA_PREPARATION_AVAILABLE={expected}, got {value}"
                )
                break

    @pytest.mark.contract
    def test_transfer_learning_phase_matches_flag(self, post_training_pkg):
        """
        Phase 4 (Transfer Learning) status matches
        ``_TRANSFER_LEARNING_AVAILABLE``.
        """
        result = post_training_pkg.get_implementation_status()
        expected = post_training_pkg._TRANSFER_LEARNING_AVAILABLE
        for key, value in result.items():
            if "Transfer Learning" in key:
                assert value == expected, (
                    f"Phase 4 status should match "
                    f"_TRANSFER_LEARNING_AVAILABLE={expected}, got {value}"
                )
                break


class TestContractCheckpointSublistContent:
    """§2 — ``__all_checkpoint__`` sub-list contains expected names."""

    @pytest.mark.contract
    def test_checkpoint_sublist_has_expected_names(self, post_training_pkg):
        """``__all_checkpoint__`` contains the documented exports."""
        sublist = post_training_pkg.__all_checkpoint__
        expected = {"CheckpointManager", "CHECKPOINT_FORMAT_VERSION"}
        actual = set(sublist)
        assert expected == actual, (
            f"__all_checkpoint__ expected {expected}, got {actual}"
        )


class TestContractInferenceSublistContent:
    """§2 — ``__all_inference__`` sub-list contains expected names."""

    @pytest.mark.contract
    def test_inference_sublist_has_expected_names(self, post_training_pkg):
        """``__all_inference__`` contains the documented exports."""
        sublist = post_training_pkg.__all_inference__
        expected = {
            "ModelLoader",
            "load_model",
            "load_model_only",
            "Predictor",
            "predict",
        }
        actual = set(sublist)
        assert expected == actual, (
            f"__all_inference__ expected {expected}, got {actual}"
        )


class TestContractDataPrepSublistContent:
    """§2 — ``__all_data_preparation__`` sub-list matches availability."""

    EXPECTED_DATA_PREP_NAMES = {
        "DataConverterProtocol",
        "DataConverterRegistry",
        "get_registry",
        "register_converter",
        "BaseDataConverter",
        "PyGDataConverter",
        "DictConverter",
        "SMILESConverter",
        "InChIConverter",
        "XYZConverter",
        "ASEAtomsConverter",
        "SDFConverter",
        "convert_to_pyg",
        "convert_batch_to_pyg",
        "convert_sdf_to_pyg_list",
        "list_available_formats",
        "list_all_formats",
        "smiles_to_data",
    }

    @pytest.mark.contract
    def test_data_prep_sublist_when_available(self, post_training_pkg):
        """
        When data preparation is available, ``__all_data_preparation__``
        contains all expected names.
        """
        if not post_training_pkg._DATA_PREPARATION_AVAILABLE:
            pytest.skip("Data preparation not available")

        sublist = set(post_training_pkg.__all_data_preparation__)
        assert self.EXPECTED_DATA_PREP_NAMES == sublist, (
            f"Missing from __all_data_preparation__: "
            f"{self.EXPECTED_DATA_PREP_NAMES - sublist}. "
            f"Extra: {sublist - self.EXPECTED_DATA_PREP_NAMES}"
        )

    @pytest.mark.contract
    def test_data_prep_sublist_empty_when_unavailable(self, post_training_pkg):
        """
        When data preparation is NOT available,
        ``__all_data_preparation__`` is empty.
        """
        if post_training_pkg._DATA_PREPARATION_AVAILABLE:
            pytest.skip("Data preparation IS available")

        sublist = post_training_pkg.__all_data_preparation__
        assert sublist == [], (
            f"__all_data_preparation__ should be empty when unavailable, "
            f"got {sublist}"
        )


class TestContractTransferLearningSublistContent:
    """§2 — ``__all_transfer_learning__`` sub-list matches availability."""

    EXPECTED_TRANSFER_NAMES = {"FineTuner", "FreezeStrategy"}

    @pytest.mark.contract
    def test_transfer_sublist_when_available(self, post_training_pkg):
        """
        When transfer learning is available,
        ``__all_transfer_learning__`` contains expected names.
        """
        if not post_training_pkg._TRANSFER_LEARNING_AVAILABLE:
            pytest.skip("Transfer learning not available")

        sublist = set(post_training_pkg.__all_transfer_learning__)
        assert self.EXPECTED_TRANSFER_NAMES == sublist, (
            f"__all_transfer_learning__ expected "
            f"{self.EXPECTED_TRANSFER_NAMES}, got {sublist}"
        )

    @pytest.mark.contract
    def test_transfer_sublist_empty_when_unavailable(self, post_training_pkg):
        """
        When transfer learning is NOT available,
        ``__all_transfer_learning__`` is empty.
        """
        if post_training_pkg._TRANSFER_LEARNING_AVAILABLE:
            pytest.skip("Transfer learning IS available")

        sublist = post_training_pkg.__all_transfer_learning__
        assert sublist == [], (
            f"__all_transfer_learning__ should be empty when unavailable, "
            f"got {sublist}"
        )


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST always be present in __all__
    # (regardless of optional dependency availability)
    MINIMUM_API = {
        # Checkpoint (always available)
        "CheckpointManager",
        "CHECKPOINT_FORMAT_VERSION",
        # Inference (always available)
        "ModelLoader",
        "load_model",
        "load_model_only",
        "Predictor",
        "predict",
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
    def test_all_has_minimum_length(self, all_names):
        """
        ``__all__`` contains at least the core exports (7 minimum).

        The core always exports 7 names (2 checkpoint + 5 inference).
        Conditional exports can add up to 20 more.
        """
        actual = len(all_names)
        MINIMUM_EXPECTED = 7
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least "
            f"{MINIMUM_EXPECTED}. This suggests __all__ may have been "
            f"accidentally truncated."
        )


class TestContractDependencyInjectionPattern:
    """§2 — Dependency Injection pattern validation (v2.0.0)."""

    @pytest.mark.contract
    def test_no_path_utils_export(self, post_training_pkg, all_names):
        """
        No path utility names (removed in v2.0.0) appear in ``__all__``.
        """
        removed_names = {"resolve_path", "get_working_root", "PathResolver"}
        all_set = set(all_names)
        found = removed_names & all_set
        assert not found, (
            f"Path utility names still in __all__ after v2.0.0 removal: "
            f"{found}"
        )

    @pytest.mark.contract
    def test_version_is_2_or_later(self, post_training_pkg):
        """
        ``__version__`` is 2.0.0 or later (DI refactoring version).
        """
        version = post_training_pkg.__version__
        major = int(version.split(".")[0])
        assert major >= 2, (
            f"Post-training module version should be >= 2.0.0 (DI pattern), "
            f"got {version}"
        )


class TestContractInferenceFunctionSignatures:
    """§2 — Key inference functions have expected parameter signatures."""

    @pytest.mark.contract
    def test_load_model_has_parameters(self, post_training_pkg):
        """``load_model`` accepts parameters."""
        sig = inspect.signature(post_training_pkg.load_model)
        assert len(sig.parameters) >= 1, (
            "load_model should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_load_model_only_has_parameters(self, post_training_pkg):
        """``load_model_only`` accepts parameters."""
        sig = inspect.signature(post_training_pkg.load_model_only)
        assert len(sig.parameters) >= 1, (
            "load_model_only should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_predict_has_parameters(self, post_training_pkg):
        """``predict`` accepts parameters."""
        sig = inspect.signature(post_training_pkg.predict)
        assert len(sig.parameters) >= 1, (
            "predict should accept at least one parameter"
        )


class TestContractConvenienceFunctionSignatures:
    """§2 — Convenience functions have expected signatures."""

    @pytest.mark.contract
    def test_get_available_components_takes_no_args(self, post_training_pkg):
        """``get_available_components`` takes no required arguments."""
        sig = inspect.signature(
            post_training_pkg.get_available_components
        )
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required) == 0, (
            "get_available_components should take no required arguments"
        )

    @pytest.mark.contract
    def test_get_implementation_status_takes_no_args(self, post_training_pkg):
        """``get_implementation_status`` takes no required arguments."""
        sig = inspect.signature(
            post_training_pkg.get_implementation_status
        )
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required) == 0, (
            "get_implementation_status should take no required arguments"
        )

    @pytest.mark.contract
    def test_print_available_components_takes_no_args(self, post_training_pkg):
        """``print_available_components`` takes no required arguments."""
        sig = inspect.signature(
            post_training_pkg.print_available_components
        )
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required) == 0, (
            "print_available_components should take no required arguments"
        )


class TestContractConvenienceFunctionReturnTypeAnnotations:
    """§2 — Convenience functions have return type annotations."""

    @pytest.mark.contract
    def test_get_available_components_has_return_annotation(
        self, post_training_pkg
    ):
        """``get_available_components`` has a return type annotation."""
        sig = inspect.signature(
            post_training_pkg.get_available_components
        )
        assert sig.return_annotation is not inspect.Signature.empty, (
            "get_available_components should have a return type annotation"
        )

    @pytest.mark.contract
    def test_get_implementation_status_has_return_annotation(
        self, post_training_pkg
    ):
        """``get_implementation_status`` has a return type annotation."""
        sig = inspect.signature(
            post_training_pkg.get_implementation_status
        )
        assert sig.return_annotation is not inspect.Signature.empty, (
            "get_implementation_status should have a return type annotation"
        )


class TestContractComponentsMatchSubLists:
    """§2 — ``get_available_components()`` returns the same names as the
    sectioned __all__ sub-lists."""

    @pytest.mark.contract
    def test_checkpoint_components_match_sublist(self, post_training_pkg):
        """Checkpoint components match ``__all_checkpoint__``."""
        result = post_training_pkg.get_available_components()
        assert result["checkpoint"] == post_training_pkg.__all_checkpoint__

    @pytest.mark.contract
    def test_inference_components_match_sublist(self, post_training_pkg):
        """Inference components match ``__all_inference__``."""
        result = post_training_pkg.get_available_components()
        assert result["inference"] == post_training_pkg.__all_inference__

    @pytest.mark.contract
    def test_data_prep_components_match_sublist(self, post_training_pkg):
        """Data preparation components match ``__all_data_preparation__``."""
        result = post_training_pkg.get_available_components()
        assert (
            result["data_preparation"]
            == post_training_pkg.__all_data_preparation__
        )

    @pytest.mark.contract
    def test_transfer_components_match_sublist(self, post_training_pkg):
        """Transfer learning components match
        ``__all_transfer_learning__``."""
        result = post_training_pkg.get_available_components()
        assert (
            result["transfer_learning"]
            == post_training_pkg.__all_transfer_learning__
        )


class TestContractVersionConsistency:
    """§2 — Version string is consistent and well-formed."""

    @pytest.mark.contract
    def test_version_has_three_parts(self, post_training_pkg):
        """``__version__`` has MAJOR.MINOR.PATCH format."""
        version = post_training_pkg.__version__
        parts = version.split(".")
        assert len(parts) == 3, (
            f"Version '{version}' should have exactly 3 parts "
            f"(MAJOR.MINOR.PATCH), got {len(parts)}"
        )

    @pytest.mark.contract
    def test_version_parts_are_numeric(self, post_training_pkg):
        """Each version part is numeric."""
        version = post_training_pkg.__version__
        for part in version.split("."):
            assert part.isdigit(), (
                f"Version part '{part}' should be numeric"
            )

    @pytest.mark.contract
    def test_author_is_milia_team(self, post_training_pkg):
        """``__author__`` is 'MILIA Team'."""
        assert post_training_pkg.__author__ == "MILIA Team", (
            f"__author__ should be 'MILIA Team', "
            f"got '{post_training_pkg.__author__}'"
        )
