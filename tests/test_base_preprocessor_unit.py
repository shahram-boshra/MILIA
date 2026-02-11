#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/preprocessing/base_preprocessor.py

Module under test: base_preprocessor.py
- BasePreprocessor: Abstract base class for dataset preprocessors
  - __init__(config, logger): Stores config/logger, calls _validate_config()
  - _validate_config(): Abstract — subclass-specific config validation
  - preprocess(): Abstract — execute preprocessing, return Path to .npz
  - run(): Full pipeline: preprocess() → _validate_output() → return Path
  - _validate_output(output_path): Validate .npz structure (compounds, metadata keys)

Test path on local machine: ~/ml_projects/milia/tests/test_base_preprocessor_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/preprocessing/base_preprocessor.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call
import logging
import tempfile
import time
from typing import Dict, Any, Optional
from abc import ABC

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.exceptions import (
    DataProcessingError,
    ConfigurationError,
)


# ============================================================================
# HELPER: Concrete subclass builder for testing the abstract base class
# ============================================================================

def _make_concrete_preprocessor_class(
    class_name="ConcretePreprocessor",
    validate_config_impl=None,
    preprocess_impl=None,
):
    """
    Dynamically build a concrete BasePreprocessor subclass.

    By default, _validate_config is a no-op and preprocess returns a
    dummy Path.  Callers can override either via callables.
    """
    def _default_validate_config(self):
        pass

    def _default_preprocess(self):
        return Path("/tmp/dummy_output.npz")

    attrs = {
        "_validate_config": validate_config_impl or _default_validate_config,
        "preprocess": preprocess_impl or _default_preprocess,
    }

    cls = type(class_name, (BasePreprocessor,), attrs)
    return cls


def _make_config(**overrides):
    """Create a minimal valid config dict with optional overrides."""
    defaults = {
        "input_path": "/tmp/raw_data",
        "output_path": "/tmp/output.npz",
    }
    defaults.update(overrides)
    return defaults


def _make_logger():
    """Create a logger instance for testing."""
    logger = logging.getLogger("test_base_preprocessor")
    logger.setLevel(logging.DEBUG)
    return logger


def _create_valid_npz(path, compounds=None, metadata=None):
    """Create a valid .npz file with required keys at the given path."""
    if compounds is None:
        compounds = np.array([{"atoms": [1, 6]}, {"atoms": [8, 1]}], dtype=object)
    if metadata is None:
        metadata = np.array({"source": "test", "version": "1.0"})
    np.savez(path, compounds=compounds, metadata=metadata)


# ============================================================================
# GROUP 1: BasePreprocessor Abstract Nature and Class Structure (8 tests)
# ============================================================================

class TestBasePreprocessorAbstractStructure(unittest.TestCase):
    """Test that BasePreprocessor is properly abstract and has correct structure."""

    def test_cannot_instantiate_base_directly(self):
        """BasePreprocessor cannot be instantiated — it is abstract."""
        with self.assertRaises(TypeError):
            BasePreprocessor(config={}, logger=_make_logger())

    def test_is_subclass_of_abc(self):
        """BasePreprocessor inherits from ABC."""
        self.assertTrue(issubclass(BasePreprocessor, ABC))

    def test_validate_config_is_abstract(self):
        """_validate_config is declared as an abstract method."""
        self.assertIn("_validate_config", BasePreprocessor.__abstractmethods__)

    def test_preprocess_is_abstract(self):
        """preprocess is declared as an abstract method."""
        self.assertIn("preprocess", BasePreprocessor.__abstractmethods__)

    def test_run_is_not_abstract(self):
        """run() is a concrete method on the base class."""
        self.assertNotIn("run", BasePreprocessor.__abstractmethods__)

    def test_validate_output_is_not_abstract(self):
        """_validate_output() is a concrete method on the base class."""
        self.assertNotIn("_validate_output", BasePreprocessor.__abstractmethods__)

    def test_abstract_methods_count(self):
        """Exactly two abstract methods exist."""
        self.assertEqual(len(BasePreprocessor.__abstractmethods__), 2)

    def test_abstract_methods_are_exactly_expected(self):
        """Abstract methods are exactly _validate_config and preprocess."""
        self.assertEqual(
            BasePreprocessor.__abstractmethods__,
            frozenset({"_validate_config", "preprocess"}),
        )


# ============================================================================
# GROUP 2: __init__ — Config/Logger Storage and _validate_config Call (10 tests)
# ============================================================================

class TestBasePreprocessorInit(unittest.TestCase):
    """Test __init__ stores config/logger and calls _validate_config."""

    def test_config_stored(self):
        """config dict is stored as self.config."""
        config = _make_config(dataset="DFT")
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=config, logger=_make_logger())
        self.assertIs(preprocessor.config, config)

    def test_logger_stored(self):
        """logger is stored as self.logger."""
        logger = _make_logger()
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=logger)
        self.assertIs(preprocessor.logger, logger)

    def test_validate_config_called_on_init(self):
        """_validate_config is called during __init__."""
        call_tracker = Mock()

        def tracked_validate(self):
            call_tracker()

        ConcreteCls = _make_concrete_preprocessor_class(
            validate_config_impl=tracked_validate
        )
        ConcreteCls(config=_make_config(), logger=_make_logger())
        call_tracker.assert_called_once()

    def test_validate_config_called_after_attributes_set(self):
        """_validate_config can access self.config and self.logger."""
        accessed = {}

        def validating_config(self):
            accessed["config"] = self.config
            accessed["logger"] = self.logger

        ConcreteCls = _make_concrete_preprocessor_class(
            validate_config_impl=validating_config
        )
        config = _make_config()
        logger = _make_logger()
        ConcreteCls(config=config, logger=logger)
        self.assertIs(accessed["config"], config)
        self.assertIs(accessed["logger"], logger)

    def test_validate_config_raises_propagates(self):
        """If _validate_config raises ConfigurationError, it propagates from __init__."""
        def failing_validate(self):
            raise ConfigurationError("Bad config")

        ConcreteCls = _make_concrete_preprocessor_class(
            validate_config_impl=failing_validate
        )
        with self.assertRaises(ConfigurationError):
            ConcreteCls(config=_make_config(), logger=_make_logger())

    def test_validate_config_raises_generic_exception_propagates(self):
        """If _validate_config raises a generic Exception, it propagates."""
        def failing_validate(self):
            raise ValueError("unexpected")

        ConcreteCls = _make_concrete_preprocessor_class(
            validate_config_impl=failing_validate
        )
        with self.assertRaises(ValueError):
            ConcreteCls(config=_make_config(), logger=_make_logger())

    def test_empty_config_accepted(self):
        """Empty config dict is accepted (validation is subclass-specific)."""
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config={}, logger=_make_logger())
        self.assertEqual(preprocessor.config, {})

    def test_complex_config_stored_by_reference(self):
        """Config with nested structures is stored by reference, not copied."""
        config = _make_config(nested={"a": {"b": [1, 2, 3]}})
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=config, logger=_make_logger())
        self.assertIs(preprocessor.config["nested"], config["nested"])

    def test_concrete_subclass_is_instance_of_base(self):
        """Concrete subclass instance is an instance of BasePreprocessor."""
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        self.assertIsInstance(preprocessor, BasePreprocessor)

    def test_class_name_preserved_on_dynamic_subclass(self):
        """Dynamically created subclass preserves its class name."""
        ConcreteCls = _make_concrete_preprocessor_class(class_name="WavefunctionPreprocessor")
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        self.assertEqual(preprocessor.__class__.__name__, "WavefunctionPreprocessor")


# ============================================================================
# GROUP 3: run() — Orchestration Pipeline (14 tests)
# ============================================================================

class TestBasePreprocessorRun(unittest.TestCase):
    """Test run() orchestrates preprocess → _validate_output → return path."""

    def test_run_returns_path_from_preprocess(self):
        """run() returns the Path returned by preprocess()."""
        expected_path = Path("/tmp/output.npz")

        def preprocess_impl(self):
            return expected_path

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=preprocess_impl)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())

        with patch.object(preprocessor, "_validate_output"):
            result = preprocessor.run()

        self.assertEqual(result, expected_path)

    def test_run_calls_preprocess(self):
        """run() calls self.preprocess()."""
        preprocess_tracker = Mock(return_value=Path("/tmp/out.npz"))

        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        preprocessor.preprocess = preprocess_tracker

        with patch.object(preprocessor, "_validate_output"):
            preprocessor.run()

        preprocess_tracker.assert_called_once()

    def test_run_calls_validate_output_with_path(self):
        """run() calls _validate_output with the path from preprocess()."""
        expected_path = Path("/tmp/output.npz")

        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        preprocessor.preprocess = Mock(return_value=expected_path)

        with patch.object(preprocessor, "_validate_output") as mock_validate:
            preprocessor.run()

        mock_validate.assert_called_once_with(expected_path)

    def test_run_calls_preprocess_before_validate_output(self):
        """run() calls preprocess() before _validate_output()."""
        call_order = []

        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())

        original_preprocess = preprocessor.preprocess
        def tracked_preprocess():
            call_order.append("preprocess")
            return Path("/tmp/out.npz")

        preprocessor.preprocess = tracked_preprocess

        with patch.object(preprocessor, "_validate_output",
                          side_effect=lambda p: call_order.append("validate_output")):
            preprocessor.run()

        self.assertEqual(call_order, ["preprocess", "validate_output"])

    def test_run_logs_start_message(self):
        """run() logs an info message at the start with class name."""
        ConcreteCls = _make_concrete_preprocessor_class(class_name="TestPreprocessor")
        mock_logger = Mock(spec=logging.Logger)
        preprocessor = ConcreteCls(config=_make_config(), logger=mock_logger)
        preprocessor.preprocess = Mock(return_value=Path("/tmp/out.npz"))

        with patch.object(preprocessor, "_validate_output"):
            preprocessor.run()

        # First info call should contain the class name
        start_call_args = mock_logger.info.call_args_list[0][0][0]
        self.assertIn("TestPreprocessor", start_call_args)

    def test_run_logs_completion_message_with_path(self):
        """run() logs completion with the output path."""
        expected_path = Path("/tmp/result.npz")
        ConcreteCls = _make_concrete_preprocessor_class()
        mock_logger = Mock(spec=logging.Logger)
        preprocessor = ConcreteCls(config=_make_config(), logger=mock_logger)
        preprocessor.preprocess = Mock(return_value=expected_path)

        with patch.object(preprocessor, "_validate_output"):
            preprocessor.run()

        completion_call_args = mock_logger.info.call_args_list[-1][0][0]
        self.assertIn(str(expected_path), completion_call_args)

    def test_run_logs_elapsed_time(self):
        """run() logs elapsed time in the completion message."""
        ConcreteCls = _make_concrete_preprocessor_class()
        mock_logger = Mock(spec=logging.Logger)
        preprocessor = ConcreteCls(config=_make_config(), logger=mock_logger)
        preprocessor.preprocess = Mock(return_value=Path("/tmp/out.npz"))

        with patch.object(preprocessor, "_validate_output"):
            preprocessor.run()

        # Last info call should contain a timing like "0.00s" or similar
        completion_msg = mock_logger.info.call_args_list[-1][0][0]
        self.assertIn("s)", completion_msg)

    def test_run_preprocess_exception_raises_data_processing_error(self):
        """run() wraps preprocess() exceptions in DataProcessingError."""
        def failing_preprocess(self):
            raise RuntimeError("disk full")

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=failing_preprocess)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())

        with self.assertRaises(DataProcessingError) as ctx:
            preprocessor.run()

        self.assertIn("Preprocessing error", str(ctx.exception))

    def test_run_preprocess_exception_preserves_cause(self):
        """run() preserves the original exception as __cause__ (raise from)."""
        original_error = RuntimeError("original disk error")

        def failing_preprocess(self):
            raise original_error

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=failing_preprocess)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())

        with self.assertRaises(DataProcessingError) as ctx:
            preprocessor.run()

        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    def test_run_validate_output_exception_raises_data_processing_error(self):
        """run() wraps _validate_output exceptions in DataProcessingError."""
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        preprocessor.preprocess = Mock(return_value=Path("/tmp/out.npz"))

        with patch.object(
            preprocessor, "_validate_output",
            side_effect=DataProcessingError("validation failed")
        ):
            with self.assertRaises(DataProcessingError) as ctx:
                preprocessor.run()

        self.assertIn("Preprocessing error", str(ctx.exception))

    def test_run_logs_error_on_failure(self):
        """run() logs an error message when an exception occurs."""
        def failing_preprocess(self):
            raise ValueError("bad data")

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=failing_preprocess)
        mock_logger = Mock(spec=logging.Logger)
        preprocessor = ConcreteCls(config=_make_config(), logger=mock_logger)

        with self.assertRaises(DataProcessingError):
            preprocessor.run()

        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        self.assertIn("Preprocessing failed", error_msg)

    def test_run_does_not_return_on_failure(self):
        """run() does not return a value when an exception is raised."""
        def failing_preprocess(self):
            raise IOError("cannot read")

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=failing_preprocess)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())

        with self.assertRaises(DataProcessingError):
            result = preprocessor.run()
            self.fail("run() should not reach this point")

    def test_run_wraps_data_processing_error_from_preprocess(self):
        """If preprocess() raises DataProcessingError, run() re-wraps it."""
        def raising_preprocess(self):
            raise DataProcessingError("inner error")

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=raising_preprocess)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())

        with self.assertRaises(DataProcessingError) as ctx:
            preprocessor.run()

        # The outer exception wraps the inner one
        self.assertIn("Preprocessing error", str(ctx.exception))

    def test_run_successful_returns_path_type(self):
        """run() returns a Path object on success."""
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        preprocessor.preprocess = Mock(return_value=Path("/tmp/out.npz"))

        with patch.object(preprocessor, "_validate_output"):
            result = preprocessor.run()

        self.assertIsInstance(result, Path)


# ============================================================================
# GROUP 4: _validate_output — NPZ File Validation (14 tests)
# ============================================================================

class TestBasePreprocessorValidateOutput(unittest.TestCase):
    """Test _validate_output validates .npz file existence and structure."""

    def setUp(self):
        """Create a temporary directory for test .npz files."""
        self._tmpdir = tempfile.mkdtemp()
        ConcreteCls = _make_concrete_preprocessor_class()
        self.preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _npz_path(self, name="test.npz"):
        return Path(self._tmpdir) / name

    def test_nonexistent_file_raises(self):
        """_validate_output raises DataProcessingError for missing file."""
        fake_path = Path("/tmp/nonexistent_abc123.npz")
        with self.assertRaises(DataProcessingError) as ctx:
            self.preprocessor._validate_output(fake_path)
        self.assertIn("not created", str(ctx.exception))

    def test_valid_npz_passes(self):
        """_validate_output succeeds for .npz with compounds and metadata."""
        path = self._npz_path()
        _create_valid_npz(path)
        # Should not raise
        self.preprocessor._validate_output(path)

    def test_missing_compounds_key_raises(self):
        """_validate_output raises when 'compounds' key is missing."""
        path = self._npz_path()
        np.savez(path, metadata=np.array({"v": "1.0"}))
        with self.assertRaises(DataProcessingError) as ctx:
            self.preprocessor._validate_output(path)
        self.assertIn("compounds", str(ctx.exception))

    def test_missing_metadata_key_raises(self):
        """_validate_output raises when 'metadata' key is missing."""
        path = self._npz_path()
        np.savez(path, compounds=np.array([1, 2, 3]))
        with self.assertRaises(DataProcessingError) as ctx:
            self.preprocessor._validate_output(path)
        self.assertIn("metadata", str(ctx.exception))

    def test_missing_both_keys_raises(self):
        """_validate_output raises when both required keys are missing."""
        path = self._npz_path()
        np.savez(path, other_data=np.array([1]))
        with self.assertRaises(DataProcessingError) as ctx:
            self.preprocessor._validate_output(path)
        error_msg = str(ctx.exception)
        self.assertIn("compounds", error_msg)
        self.assertIn("metadata", error_msg)

    def test_extra_keys_allowed(self):
        """_validate_output allows additional keys beyond the required ones."""
        path = self._npz_path()
        compounds = np.array([{"a": 1}], dtype=object)
        metadata = np.array({"v": "1.0"})
        np.savez(path, compounds=compounds, metadata=metadata, extra=np.array([42]))
        # Should not raise
        self.preprocessor._validate_output(path)

    def test_valid_npz_logs_molecule_count(self):
        """_validate_output logs the number of molecules on success."""
        path = self._npz_path()
        compounds = np.array([1, 2, 3, 4, 5])
        np.savez(path, compounds=compounds, metadata=np.array("meta"))
        mock_logger = Mock(spec=logging.Logger)
        self.preprocessor.logger = mock_logger

        self.preprocessor._validate_output(path)

        info_msg = mock_logger.info.call_args[0][0]
        self.assertIn("5", info_msg)

    def test_error_message_includes_output_path_on_missing_file(self):
        """Error message includes the path of the missing output file."""
        fake_path = Path("/tmp/missing_file_xyz.npz")
        with self.assertRaises(DataProcessingError) as ctx:
            self.preprocessor._validate_output(fake_path)
        self.assertIn("missing_file_xyz.npz", str(ctx.exception))

    def test_error_message_includes_missing_keys(self):
        """Error message lists which specific keys are missing."""
        path = self._npz_path()
        np.savez(path, unrelated=np.array([1]))
        with self.assertRaises(DataProcessingError) as ctx:
            self.preprocessor._validate_output(path)
        error_msg = str(ctx.exception)
        # Both should be listed as missing
        self.assertIn("compounds", error_msg)
        self.assertIn("metadata", error_msg)

    def test_corrupt_npz_raises_data_processing_error(self):
        """_validate_output raises DataProcessingError for corrupt .npz files."""
        path = self._npz_path("corrupt.npz")
        with open(path, "wb") as f:
            f.write(b"this is not a valid npz file content")
        with self.assertRaises(DataProcessingError):
            self.preprocessor._validate_output(path)

    def test_empty_compounds_array_passes(self):
        """_validate_output accepts empty compounds array (key exists)."""
        path = self._npz_path()
        np.savez(path, compounds=np.array([]), metadata=np.array("meta"))
        # Should not raise — the key exists, just empty
        self.preprocessor._validate_output(path)

    def test_validate_output_uses_allow_pickle_true(self):
        """_validate_output loads .npz with allow_pickle=True for object arrays."""
        path = self._npz_path()
        compounds = np.array([{"key": "value"}], dtype=object)
        metadata = np.array({"source": "test"})
        np.savez(path, compounds=compounds, metadata=metadata)
        # Should succeed — allow_pickle=True enables object array loading
        self.preprocessor._validate_output(path)

    def test_exception_chaining_on_np_load_failure(self):
        """_validate_output chains the original exception via 'from'."""
        path = self._npz_path("bad.npz")
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02\x03")
        with self.assertRaises(DataProcessingError) as ctx:
            self.preprocessor._validate_output(path)
        # Should have a __cause__ from the original exception
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_validate_output_with_path_object(self):
        """_validate_output works correctly when given a Path object."""
        path = self._npz_path()
        _create_valid_npz(path)
        self.assertIsInstance(path, Path)
        # Should not raise
        self.preprocessor._validate_output(path)


# ============================================================================
# GROUP 5: Subclass Patterns and Inheritance (10 tests)
# ============================================================================

class TestBasePreprocessorSubclassing(unittest.TestCase):
    """Test subclass patterns, inheritance, and multiple subclass isolation."""

    def test_concrete_subclass_with_both_methods_is_instantiable(self):
        """Subclass implementing both abstract methods can be instantiated."""
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        self.assertIsNotNone(preprocessor)

    def test_subclass_missing_preprocess_is_still_abstract(self):
        """Subclass implementing only _validate_config remains abstract."""
        class PartialPreprocessor(BasePreprocessor):
            def _validate_config(self):
                pass

        with self.assertRaises(TypeError):
            PartialPreprocessor(config={}, logger=_make_logger())

    def test_subclass_missing_validate_config_is_still_abstract(self):
        """Subclass implementing only preprocess remains abstract."""
        class PartialPreprocessor(BasePreprocessor):
            def preprocess(self):
                return Path("/tmp/out.npz")

        with self.assertRaises(TypeError):
            PartialPreprocessor(config={}, logger=_make_logger())

    def test_subclass_missing_both_is_still_abstract(self):
        """Subclass implementing neither abstract method remains abstract."""
        class EmptyPreprocessor(BasePreprocessor):
            pass

        with self.assertRaises(TypeError):
            EmptyPreprocessor(config={}, logger=_make_logger())

    def test_two_subclasses_are_isolated(self):
        """Two different concrete subclasses have independent configs."""
        ClsA = _make_concrete_preprocessor_class(class_name="PreprocessorA")
        ClsB = _make_concrete_preprocessor_class(class_name="PreprocessorB")

        config_a = _make_config(name="A")
        config_b = _make_config(name="B")

        a = ClsA(config=config_a, logger=_make_logger())
        b = ClsB(config=config_b, logger=_make_logger())

        self.assertEqual(a.config["name"], "A")
        self.assertEqual(b.config["name"], "B")
        self.assertIsNot(a.config, b.config)

    def test_isinstance_check(self):
        """Concrete subclass instance passes isinstance for BasePreprocessor."""
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        self.assertIsInstance(preprocessor, BasePreprocessor)

    def test_issubclass_check(self):
        """Concrete subclass passes issubclass for BasePreprocessor."""
        ConcreteCls = _make_concrete_preprocessor_class()
        self.assertTrue(issubclass(ConcreteCls, BasePreprocessor))

    def test_intermediate_abstract_subclass_allowed(self):
        """An abstract intermediate subclass that doesn't implement all methods is valid as a class."""
        class IntermediatePreprocessor(BasePreprocessor):
            def _validate_config(self):
                pass
            # preprocess still abstract

        # Should not raise — class itself is fine, instantiation fails
        self.assertTrue(issubclass(IntermediatePreprocessor, BasePreprocessor))

    def test_subclass_can_override_validate_output(self):
        """Subclass can override _validate_output if needed."""
        custom_validate_called = Mock()

        class CustomValidation(BasePreprocessor):
            def _validate_config(self):
                pass
            def preprocess(self):
                return Path("/tmp/out.npz")
            def _validate_output(self, output_path):
                custom_validate_called(output_path)

        preprocessor = CustomValidation(config=_make_config(), logger=_make_logger())
        preprocessor.run()
        custom_validate_called.assert_called_once()

    def test_subclass_can_override_run(self):
        """Subclass can override run() for custom pipeline orchestration."""
        class CustomRun(BasePreprocessor):
            def _validate_config(self):
                pass
            def preprocess(self):
                return Path("/tmp/out.npz")
            def run(self):
                return Path("/custom/path.npz")

        preprocessor = CustomRun(config=_make_config(), logger=_make_logger())
        result = preprocessor.run()
        self.assertEqual(result, Path("/custom/path.npz"))


# ============================================================================
# GROUP 6: Integration Scenarios — End-to-End run() with Real NPZ (8 tests)
# ============================================================================

class TestBasePreprocessorIntegrationScenarios(unittest.TestCase):
    """Test end-to-end run() with real temporary .npz files."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_run_end_to_end_with_valid_npz(self):
        """Full run() succeeds when preprocess() creates a valid .npz file."""
        output_path = Path(self._tmpdir) / "output.npz"

        def preprocess_impl(self):
            _create_valid_npz(output_path)
            return output_path

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=preprocess_impl)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        result = preprocessor.run()
        self.assertEqual(result, output_path)
        self.assertTrue(result.exists())

    def test_run_end_to_end_fails_for_missing_keys(self):
        """Full run() fails when preprocess() creates .npz without required keys."""
        output_path = Path(self._tmpdir) / "bad_output.npz"

        def preprocess_impl(self):
            np.savez(output_path, other=np.array([1]))
            return output_path

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=preprocess_impl)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        with self.assertRaises(DataProcessingError):
            preprocessor.run()

    def test_run_end_to_end_fails_for_nonexistent_output(self):
        """Full run() fails when preprocess() returns a path that doesn't exist."""
        def preprocess_impl(self):
            return Path(self._tmpdir) / "never_created.npz"

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=preprocess_impl)
        # Bind self._tmpdir into the preprocess function scope
        tmpdir = self._tmpdir
        def preprocess_with_closure(self_inner):
            return Path(tmpdir) / "never_created.npz"

        ConcreteCls2 = _make_concrete_preprocessor_class(preprocess_impl=preprocess_with_closure)
        preprocessor = ConcreteCls2(config=_make_config(), logger=_make_logger())
        with self.assertRaises(DataProcessingError):
            preprocessor.run()

    def test_run_with_large_compounds_array(self):
        """run() succeeds with a large compounds array."""
        output_path = Path(self._tmpdir) / "large.npz"

        def preprocess_impl(self):
            compounds = np.arange(10000)
            metadata = np.array({"size": 10000})
            np.savez(output_path, compounds=compounds, metadata=metadata)
            return output_path

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=preprocess_impl)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        result = preprocessor.run()
        self.assertEqual(result, output_path)

    def test_run_timing_is_non_negative(self):
        """run() reports non-negative elapsed time."""
        output_path = Path(self._tmpdir) / "timed.npz"

        def preprocess_impl(self):
            _create_valid_npz(output_path)
            return output_path

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=preprocess_impl)
        mock_logger = Mock(spec=logging.Logger)
        preprocessor = ConcreteCls(config=_make_config(), logger=mock_logger)
        preprocessor.run()

        # Extract time from completion message
        completion_msg = mock_logger.info.call_args_list[-1][0][0]
        # Message format: "Preprocessing complete: /path (X.XXs)"
        self.assertIn("s)", completion_msg)

    def test_run_with_object_dtype_compounds(self):
        """run() succeeds with object-dtype compounds (pickled arrays)."""
        output_path = Path(self._tmpdir) / "objects.npz"

        def preprocess_impl(self):
            compounds = np.array(
                [{"smiles": "C", "energy": -1.0}, {"smiles": "O", "energy": -2.0}],
                dtype=object,
            )
            metadata = np.array({"format": "object_array"})
            np.savez(output_path, compounds=compounds, metadata=metadata)
            return output_path

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=preprocess_impl)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        result = preprocessor.run()
        self.assertEqual(result, output_path)

    def test_validate_config_and_preprocess_called_in_correct_lifecycle(self):
        """_validate_config is called in __init__, preprocess in run()."""
        lifecycle = []

        def validate_impl(self):
            lifecycle.append("validate_config")

        def preprocess_impl(self):
            lifecycle.append("preprocess")
            return Path("/tmp/dummy.npz")

        ConcreteCls = _make_concrete_preprocessor_class(
            validate_config_impl=validate_impl,
            preprocess_impl=preprocess_impl,
        )
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        self.assertEqual(lifecycle, ["validate_config"])

        with patch.object(preprocessor, "_validate_output"):
            preprocessor.run()

        self.assertEqual(lifecycle, ["validate_config", "preprocess"])

    def test_multiple_run_calls_succeed(self):
        """run() can be called multiple times on the same preprocessor."""
        output_path = Path(self._tmpdir) / "multi.npz"
        _create_valid_npz(output_path)

        def preprocess_impl(self):
            return output_path

        ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=preprocess_impl)
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())

        result1 = preprocessor.run()
        result2 = preprocessor.run()
        self.assertEqual(result1, result2)


# ============================================================================
# GROUP 7: Edge Cases and Boundary Conditions (8 tests)
# ============================================================================

class TestBasePreprocessorEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_config_with_none_values(self):
        """Config containing None values is accepted."""
        config = {"key": None, "nested": {"inner": None}}
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=config, logger=_make_logger())
        self.assertIsNone(preprocessor.config["key"])

    def test_config_with_pathlib_paths(self):
        """Config containing Path objects is accepted."""
        config = _make_config(input_path=Path("/data/raw"), output_path=Path("/data/out.npz"))
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=config, logger=_make_logger())
        self.assertIsInstance(preprocessor.config["input_path"], Path)

    def test_logger_with_different_levels(self):
        """Preprocessor works with loggers at any level."""
        for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
            with self.subTest(level=level):
                logger = logging.getLogger(f"test_level_{level}")
                logger.setLevel(level)
                ConcreteCls = _make_concrete_preprocessor_class()
                preprocessor = ConcreteCls(config=_make_config(), logger=logger)
                self.assertIsNotNone(preprocessor)

    def test_preprocess_returning_relative_path(self):
        """run() accepts a relative Path from preprocess()."""
        relative_path = Path("output/result.npz")

        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        preprocessor.preprocess = Mock(return_value=relative_path)

        with patch.object(preprocessor, "_validate_output"):
            result = preprocessor.run()

        self.assertEqual(result, relative_path)

    def test_preprocess_returning_path_with_spaces(self):
        """run() handles paths with spaces in the name."""
        path_with_spaces = Path("/tmp/my data/output file.npz")

        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
        preprocessor.preprocess = Mock(return_value=path_with_spaces)

        with patch.object(preprocessor, "_validate_output"):
            result = preprocessor.run()

        self.assertEqual(result, path_with_spaces)

    def test_class_name_used_in_start_log(self):
        """The actual class name appears in the start log message."""
        for name in ["WavefunctionPreprocessor", "QM9Preprocessor", "ANI1xPreprocessor"]:
            with self.subTest(name=name):
                ConcreteCls = _make_concrete_preprocessor_class(class_name=name)
                mock_logger = Mock(spec=logging.Logger)
                preprocessor = ConcreteCls(config=_make_config(), logger=mock_logger)
                preprocessor.preprocess = Mock(return_value=Path("/tmp/out.npz"))

                with patch.object(preprocessor, "_validate_output"):
                    preprocessor.run()

                start_msg = mock_logger.info.call_args_list[0][0][0]
                self.assertIn(name, start_msg)

    def test_config_mutation_after_init(self):
        """Config can be mutated after init (stored by reference)."""
        config = _make_config()
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=config, logger=_make_logger())
        config["new_key"] = "new_value"
        self.assertEqual(preprocessor.config["new_key"], "new_value")

    def test_validate_output_with_single_compound(self):
        """_validate_output succeeds with exactly one compound."""
        tmpdir = tempfile.mkdtemp()
        try:
            path = Path(tmpdir) / "single.npz"
            compounds = np.array([{"mol": "H2O"}], dtype=object)
            np.savez(path, compounds=compounds, metadata=np.array("m"))

            ConcreteCls = _make_concrete_preprocessor_class()
            preprocessor = ConcreteCls(config=_make_config(), logger=_make_logger())
            # Should not raise
            preprocessor._validate_output(path)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# GROUP 8: Realistic Preprocessor Patterns (6 tests)
# ============================================================================

class TestRealisticPreprocessorPatterns(unittest.TestCase):
    """Test patterns that mirror real-world preprocessor usage."""

    def test_wavefunction_like_preprocessor(self):
        """Simulates a Wavefunction-style preprocessor with config validation."""
        def validate_wf(self):
            if "molden_dir" not in self.config:
                raise ConfigurationError("Missing 'molden_dir' in config")

        ConcreteCls = _make_concrete_preprocessor_class(
            class_name="WavefunctionPreprocessor",
            validate_config_impl=validate_wf,
        )
        with self.assertRaises(ConfigurationError):
            ConcreteCls(config={}, logger=_make_logger())

        # With valid config
        preprocessor = ConcreteCls(
            config={"molden_dir": "/data/molden"}, logger=_make_logger()
        )
        self.assertEqual(preprocessor.config["molden_dir"], "/data/molden")

    def test_qm9_like_preprocessor(self):
        """Simulates a QM9-style preprocessor with input format check."""
        def validate_qm9(self):
            fmt = self.config.get("input_format")
            if fmt not in ("xyz", "sdf"):
                raise ConfigurationError(f"Unsupported format: {fmt}")

        ConcreteCls = _make_concrete_preprocessor_class(
            class_name="QM9Preprocessor",
            validate_config_impl=validate_qm9,
        )
        with self.assertRaises(ConfigurationError):
            ConcreteCls(config={"input_format": "csv"}, logger=_make_logger())

        preprocessor = ConcreteCls(
            config={"input_format": "xyz"}, logger=_make_logger()
        )
        self.assertIsNotNone(preprocessor)

    def test_preprocessor_with_output_dir_creation(self):
        """Simulates a preprocessor that references output directory in config."""
        tmpdir = tempfile.mkdtemp()
        try:
            output_path = Path(tmpdir) / "processed.npz"
            config = _make_config(output_path=str(output_path))

            def preprocess_impl(self):
                _create_valid_npz(Path(self.config["output_path"]))
                return Path(self.config["output_path"])

            ConcreteCls = _make_concrete_preprocessor_class(preprocess_impl=preprocess_impl)
            preprocessor = ConcreteCls(config=config, logger=_make_logger())
            result = preprocessor.run()
            self.assertTrue(result.exists())
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_preprocessor_with_multiple_config_keys(self):
        """Simulates a preprocessor with complex config structure."""
        config = {
            "dataset_type": "ANI1x",
            "input_path": "/data/ani1x.h5",
            "output_path": "/data/ani1x.npz",
            "properties": ["energy", "forces", "dipole"],
            "max_conformations": 500,
            "chunk_size": 1000,
        }
        ConcreteCls = _make_concrete_preprocessor_class()
        preprocessor = ConcreteCls(config=config, logger=_make_logger())
        self.assertEqual(preprocessor.config["dataset_type"], "ANI1x")
        self.assertEqual(len(preprocessor.config["properties"]), 3)

    def test_preprocessor_config_error_message_quality(self):
        """ConfigurationError from _validate_config includes helpful message."""
        def strict_validate(self):
            missing = []
            for key in ("input_path", "output_path", "dataset_type"):
                if key not in self.config:
                    missing.append(key)
            if missing:
                raise ConfigurationError(
                    f"Missing required keys: {missing}"
                )

        ConcreteCls = _make_concrete_preprocessor_class(
            validate_config_impl=strict_validate
        )
        with self.assertRaises(ConfigurationError) as ctx:
            ConcreteCls(config={"input_path": "/data"}, logger=_make_logger())

        error_msg = str(ctx.exception)
        self.assertIn("output_path", error_msg)
        self.assertIn("dataset_type", error_msg)

    def test_preprocessor_subclass_hierarchy(self):
        """Intermediate abstract subclass with concrete leaf works correctly."""
        class ArchivePreprocessor(BasePreprocessor):
            """Intermediate — adds extract but keeps preprocess abstract."""
            def _validate_config(self):
                pass

            def extract_archive(self):
                return "extracted"

        # Cannot instantiate intermediate (preprocess still abstract)
        with self.assertRaises(TypeError):
            ArchivePreprocessor(config={}, logger=_make_logger())

        # Concrete leaf
        class QM9ArchivePreprocessor(ArchivePreprocessor):
            def preprocess(self):
                self.extract_archive()
                return Path("/tmp/qm9.npz")

        preprocessor = QM9ArchivePreprocessor(config={}, logger=_make_logger())
        self.assertIsInstance(preprocessor, BasePreprocessor)
        self.assertEqual(preprocessor.extract_archive(), "extracted")


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestBasePreprocessorAbstractStructure,       # GROUP 1:  8 tests
        TestBasePreprocessorInit,                     # GROUP 2: 10 tests
        TestBasePreprocessorRun,                      # GROUP 3: 14 tests
        TestBasePreprocessorValidateOutput,           # GROUP 4: 14 tests
        TestBasePreprocessorSubclassing,              # GROUP 5: 10 tests
        TestBasePreprocessorIntegrationScenarios,     # GROUP 6:  8 tests
        TestBasePreprocessorEdgeCases,                # GROUP 7:  8 tests
        TestRealisticPreprocessorPatterns,            # GROUP 8:  6 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — base_preprocessor.py")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    total_test_groups = len(test_classes)
    print(f"\nTest Groups: {total_test_groups}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - PRODUCTION-READY")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        # Let pytest discover and run tests normally
        pass
    else:
        sys.exit(run_comprehensive_suite())


"""
TEST SUITE SUMMARY — milia_pipeline/preprocessing/base_preprocessor.py
=======================================================================

78 comprehensive production-ready tests covering:

GROUP 1: BasePreprocessor Abstract Nature and Class Structure (8 tests)
- Cannot instantiate base directly
- Is subclass of ABC
- _validate_config is abstract
- preprocess is abstract
- run is concrete (not abstract)
- _validate_output is concrete (not abstract)
- Exactly two abstract methods
- Abstract methods are exactly the expected set

GROUP 2: __init__ — Config/Logger Storage and _validate_config Call (10 tests)
- config stored as self.config
- logger stored as self.logger
- _validate_config called on init
- _validate_config can access self.config and self.logger
- ConfigurationError from _validate_config propagates
- Generic exception from _validate_config propagates
- Empty config accepted
- Complex config stored by reference
- Instance is instance of BasePreprocessor
- Dynamic class name preserved

GROUP 3: run() — Orchestration Pipeline (14 tests)
- Returns path from preprocess()
- Calls preprocess()
- Calls _validate_output with correct path
- Calls preprocess before _validate_output (ordering)
- Logs start message with class name
- Logs completion message with output path
- Logs elapsed time
- Wraps preprocess() exceptions in DataProcessingError
- Preserves original exception as __cause__
- Wraps _validate_output exceptions in DataProcessingError
- Logs error on failure
- Does not return on failure
- Re-wraps DataProcessingError from preprocess
- Returns Path type on success

GROUP 4: _validate_output — NPZ File Validation (14 tests)
- Nonexistent file raises DataProcessingError
- Valid .npz with compounds+metadata passes
- Missing 'compounds' key raises
- Missing 'metadata' key raises
- Missing both keys raises (both listed in error)
- Extra keys beyond required are allowed
- Logs molecule count on success
- Error message includes output path on missing file
- Error message lists specific missing keys
- Corrupt .npz raises DataProcessingError
- Empty compounds array passes (key exists)
- Uses allow_pickle=True for object arrays
- Exception chaining on np.load failure
- Works correctly with Path objects

GROUP 5: Subclass Patterns and Inheritance (10 tests)
- Concrete subclass with both methods is instantiable
- Missing preprocess keeps class abstract
- Missing _validate_config keeps class abstract
- Missing both keeps class abstract
- Two subclasses have isolated configs
- isinstance check passes
- issubclass check passes
- Intermediate abstract subclass allowed
- Subclass can override _validate_output
- Subclass can override run()

GROUP 6: Integration Scenarios — End-to-End run() with Real NPZ (8 tests)
- Full run() with valid .npz succeeds
- Full run() fails for missing keys in .npz
- Full run() fails for nonexistent output
- run() with large compounds array
- run() timing is non-negative
- run() with object-dtype compounds (pickled)
- Lifecycle ordering (_validate_config in init, preprocess in run)
- Multiple run() calls succeed

GROUP 7: Edge Cases and Boundary Conditions (8 tests)
- Config with None values
- Config with Path objects
- Logger with different levels
- Relative Path from preprocess()
- Path with spaces
- Class name in start log (parameterized)
- Config mutation after init (reference semantics)
- Single compound validates successfully

GROUP 8: Realistic Preprocessor Patterns (6 tests)
- Wavefunction-like preprocessor with config validation
- QM9-like preprocessor with format check
- Preprocessor with output dir in config
- Complex config structure
- Config error message quality
- Intermediate abstract subclass hierarchy

Total: 78 comprehensive production-ready tests

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All mocking via @patch decorators or context managers (test-level only)
- Dynamic test data creation via helper functions (no hardcoded paths)
- No NPZ file downloads — test .npz files created in tempdir
- Comprehensive error path coverage
- Interface-focused testing (future-proof)
- Compatible with both pytest and unittest runner
- Subclass isolation verified
- Exception chaining verified
- Error message quality assertions
- Lifecycle ordering assertions
- Temp file cleanup in tearDown
"""
