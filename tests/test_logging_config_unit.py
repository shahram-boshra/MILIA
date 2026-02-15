#!/usr/bin/env python3
"""
Unit tests for logging_config.py module (Transformation System Integration)

Test file: test_logging_config_unit.py
Module under test: ~/ml_projects/milia/milia_pipeline/milia_pipeline/logging_config.py

This test suite validates the logging_config module including handler logging,
migration logging, transformation system logging, logger adapters, decorators,
performance logging, exception context logging, and utility functions.

PRODUCTION-READY REQUIREMENTS:
- NO sys.modules pollution at module level (mocks use @patch decorators)
- Proper teardown_module() cleanup for safety
- Test isolation via fixtures
- Comprehensive coverage of all public API methods

Key Test Areas:
1.  HandlerLoggerAdapter - initialization and process()
2.  MigrationLoggerAdapter - initialization and process()
3.  TransformLoggerAdapter - initialization and process()
4.  setup_logging() - full configuration with all toggles
5.  _configure_third_party_loggers() - third-party silencing
6.  _setup_handler_loggers() - handler logger creation
7.  _setup_migration_loggers() - migration logger creation
8.  _setup_transform_loggers() - transform logger creation
9.  create_handler_logger() - factory function
10. create_migration_logger() - factory function
11. create_transform_logger() - factory function
12. log_handler_operation() - decorator
13. log_migration_step() - decorator
14. log_transform_operation() - decorator
15. log_handler_performance() - structured performance logging
16. log_transform_performance() - structured performance logging
17. get_handler_logger_by_type() - logger retrieval
18. get_migration_logger_by_phase() - logger retrieval
19. get_transform_logger_by_operation() - logger retrieval
20. log_exception_with_context() - exception logging
21. log_experimental_setup_switch() - setup switch logging
22. log_transform_validation_results() - validation result logging
23. log_transform_composition_summary() - composition summary logging
24. configure_debug_logging_for_handlers() - debug configuration
25. configure_debug_logging_for_transforms() - debug configuration
26. disable_verbose_third_party_logging() - third-party silencing
27. create_experimental_setup_hash() - deterministic hashing
28. log_experimental_setup_summary() - comprehensive setup summary
29. setup_basic_logging() - backward compatibility alias
30. Edge cases and boundary conditions
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path("/app/milia")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Standard library imports
import logging
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
import milia_pipeline.logging_config as logging_config
from milia_pipeline.exceptions import (
    ExperimentalSetupError,
    HandlerError,
    LoggingConfigurationError,
    MigrationError,
    TransformCompositionError,
    TransformConfigurationError,
    TransformValidationError,
)
from milia_pipeline.logging_config import (
    HandlerLoggerAdapter,
    MigrationLoggerAdapter,
    TransformLoggerAdapter,
    configure_debug_logging_for_handlers,
    configure_debug_logging_for_transforms,
    create_experimental_setup_hash,
    create_handler_logger,
    create_migration_logger,
    create_transform_logger,
    disable_verbose_third_party_logging,
    get_handler_logger_by_type,
    get_migration_logger_by_phase,
    get_transform_logger_by_operation,
    log_exception_with_context,
    log_experimental_setup_summary,
    log_experimental_setup_switch,
    log_handler_operation,
    log_handler_performance,
    log_migration_step,
    log_transform_composition_summary,
    log_transform_operation,
    log_transform_performance,
    log_transform_validation_results,
    setup_basic_logging,
    setup_logging,
)

# ==============================================================================
# PYTEST FIXTURE: AUTOMATIC LOGGER CLEANUP
# ==============================================================================


@pytest.fixture(autouse=True)
def cleanup_loggers():
    """
    Automatic fixture to prevent logger handler accumulation across tests.

    Problem: logging.getLogger() returns singleton instances by name. Handlers
    added during setup_logging() persist across tests causing handler duplication,
    duplicate log output, and cross-test contamination.

    Solution: This fixture automatically clears handlers on all loggers that
    the logging_config module creates or touches, both before and after each test.
    """
    logger_names_to_clean = [
        "milia_pipeline.logging_config",
        "logging_config",
        "rdkit",
        "matplotlib",
        "PIL",
        "urllib3",
        "requests",
        "torch",
        "torch_geometric",
        "numpy",
        "scipy",
        "handler.dft",
        "handler.dmc",
        "handler.generic",
        "migration.phase_6F",
        "migration.phase_6G",
        "migration.phase_6H",
        "migration.phase_6I",
        "migration.phase_6J",
        "migration.phase_6f",
        "migration.phase_6g",
        "migration.phase_6h",
        "migration.phase_6i",
        "migration.phase_6j",
        "transform.registry",
        "transform.validation",
        "transform.composition",
        "transform.experimental",
    ]

    # Pre-test cleanup
    for name in logger_names_to_clean:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)
        # CRITICAL: Reset parent to natural hierarchy parent.
        # _setup_handler_loggers/_setup_migration_loggers/_setup_transform_loggers
        # set logger.parent = base_logger (a MagicMock in tests). Since loggers
        # are singletons, this mock parent persists across tests. When a decorator
        # error path calls logger.error(), Python's callHandlers() walks c.parent
        # and hits the Mock, crashing on c.handlers. Fix: use the logging Manager
        # to resolve the correct parent from the logger name hierarchy.
        logger.parent = (
            logger.manager.getLogger(name.rsplit(".", 1)[0]) if "." in name else logging.getLogger()
        )

    # Ensure all loggers have at least a NullHandler so that logging calls
    # inside decorators (error paths) don't crash when walking the parent
    # handler chain. Without this, Python's logging callHandlers() traverses
    # up to root and may encounter loggers in unexpected states.
    for name in logger_names_to_clean:
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())

    root_logger = logging.getLogger()
    original_root_handlers = root_logger.handlers[:]
    original_root_level = root_logger.level

    yield

    # Post-test cleanup
    for name in logger_names_to_clean:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)
        # Reset parent to natural hierarchy parent (same as pre-test)
        logger.parent = (
            logger.manager.getLogger(name.rsplit(".", 1)[0]) if "." in name else logging.getLogger()
        )

    root_logger.handlers = original_root_handlers
    root_logger.level = original_root_level


# ==============================================================================
# TEST AREA 1: HandlerLoggerAdapter
# ==============================================================================


class TestHandlerLoggerAdapter(unittest.TestCase):
    """Tests for HandlerLoggerAdapter initialization and process method."""

    def setUp(self):
        """Create a mock base logger for adapter tests."""
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.mock_logger.manager = MagicMock()
        self.mock_logger.parent = None

    def test_init_basic(self):
        """Test basic initialization with handler_type and no extra context."""
        adapter = HandlerLoggerAdapter(self.mock_logger, handler_type="DFT")
        self.assertEqual(adapter.handler_type, "DFT")
        self.assertEqual(adapter.extra["handler_type"], "DFT")

    def test_init_with_extra_context(self):
        """Test initialization with additional extra context."""
        extra = {"batch_id": "batch_001", "run_id": "run_42"}
        adapter = HandlerLoggerAdapter(self.mock_logger, handler_type="DMC", extra=extra)
        self.assertEqual(adapter.handler_type, "DMC")
        self.assertEqual(adapter.extra["batch_id"], "batch_001")
        self.assertEqual(adapter.extra["run_id"], "run_42")

    def test_init_extra_none(self):
        """Test initialization when extra is explicitly None."""
        adapter = HandlerLoggerAdapter(self.mock_logger, handler_type="GENERIC", extra=None)
        self.assertEqual(adapter.extra, {"handler_type": "GENERIC"})

    def test_process_basic_message(self):
        """Test process adds handler prefix to message."""
        adapter = HandlerLoggerAdapter(self.mock_logger, handler_type="DFT")
        processed_msg, _ = adapter.process("Processing molecule", {})
        self.assertEqual(processed_msg, "[DFT] Processing molecule")

    def test_process_with_molecule_index_in_kwargs(self):
        """Test process adds molecule context when molecule_index is in kwargs."""
        adapter = HandlerLoggerAdapter(self.mock_logger, handler_type="DMC")
        processed_msg, _ = adapter.process("Validation failed", {"molecule_index": 42})
        self.assertEqual(processed_msg, "[Mol:42] [DMC] Validation failed")

    def test_process_without_molecule_index(self):
        """Test process does not add molecule context when molecule_index is absent."""
        adapter = HandlerLoggerAdapter(self.mock_logger, handler_type="DFT")
        processed_msg, _ = adapter.process("Batch complete", {"other_key": "value"})
        self.assertEqual(processed_msg, "[DFT] Batch complete")

    def test_process_returns_kwargs_unchanged(self):
        """Test process returns kwargs unmodified."""
        adapter = HandlerLoggerAdapter(self.mock_logger, handler_type="DFT")
        original_kwargs = {"molecule_index": 5, "extra_data": "test"}
        _, returned_kwargs = adapter.process("test", original_kwargs)
        self.assertIs(returned_kwargs, original_kwargs)

    def test_extra_context_update_behavior(self):
        """Test that extra dict update follows dict.update() semantics (last write wins)."""
        extra = {"handler_type": "OVERRIDDEN_VALUE"}
        adapter = HandlerLoggerAdapter(self.mock_logger, handler_type="DFT", extra=extra)
        self.assertEqual(adapter.extra["handler_type"], "OVERRIDDEN_VALUE")


# ==============================================================================
# TEST AREA 2: MigrationLoggerAdapter
# ==============================================================================


class TestMigrationLoggerAdapter(unittest.TestCase):
    """Tests for MigrationLoggerAdapter initialization and process method."""

    def setUp(self):
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.mock_logger.manager = MagicMock()
        self.mock_logger.parent = None

    def test_init_basic(self):
        """Test basic initialization with migration_phase."""
        adapter = MigrationLoggerAdapter(self.mock_logger, migration_phase="6F")
        self.assertEqual(adapter.migration_phase, "6F")
        self.assertEqual(adapter.extra["migration_phase"], "6F")

    def test_init_with_extra_context(self):
        """Test initialization with additional extra context."""
        extra = {"module": "property_enrichment.py"}
        adapter = MigrationLoggerAdapter(self.mock_logger, migration_phase="6G", extra=extra)
        self.assertEqual(adapter.migration_phase, "6G")
        self.assertEqual(adapter.extra["module"], "property_enrichment.py")

    def test_init_extra_none(self):
        """Test initialization when extra is explicitly None."""
        adapter = MigrationLoggerAdapter(self.mock_logger, migration_phase="6H", extra=None)
        self.assertEqual(adapter.extra, {"migration_phase": "6H"})

    def test_process_basic_message(self):
        """Test process adds migration prefix to message."""
        adapter = MigrationLoggerAdapter(self.mock_logger, migration_phase="6F")
        processed_msg, _ = adapter.process("Starting migration", {})
        self.assertEqual(processed_msg, "[Migration-6F] Starting migration")

    def test_process_with_migration_step_in_kwargs(self):
        """Test process adds step context when migration_step is in kwargs."""
        adapter = MigrationLoggerAdapter(self.mock_logger, migration_phase="6G")
        processed_msg, _ = adapter.process("Executing step", {"migration_step": "validate_schema"})
        self.assertEqual(processed_msg, "[Step:validate_schema] [Migration-6G] Executing step")

    def test_process_without_migration_step(self):
        """Test process does not add step context when migration_step is absent."""
        adapter = MigrationLoggerAdapter(self.mock_logger, migration_phase="6I")
        processed_msg, _ = adapter.process("Phase complete", {"other": "data"})
        self.assertEqual(processed_msg, "[Migration-6I] Phase complete")

    def test_process_returns_kwargs_unchanged(self):
        """Test process returns kwargs unmodified."""
        adapter = MigrationLoggerAdapter(self.mock_logger, migration_phase="6J")
        original_kwargs = {"migration_step": "rollback", "detail": "test"}
        _, returned_kwargs = adapter.process("test", original_kwargs)
        self.assertIs(returned_kwargs, original_kwargs)


# ==============================================================================
# TEST AREA 3: TransformLoggerAdapter
# ==============================================================================


class TestTransformLoggerAdapter(unittest.TestCase):
    """Tests for TransformLoggerAdapter initialization and process method."""

    def setUp(self):
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.mock_logger.manager = MagicMock()
        self.mock_logger.parent = None

    def test_init_basic_no_context(self):
        """Test initialization with no experimental_setup or transform_context."""
        adapter = TransformLoggerAdapter(self.mock_logger)
        self.assertIsNone(adapter.experimental_setup)
        self.assertIsNone(adapter.transform_context)
        self.assertEqual(adapter.extra, {})

    def test_init_with_experimental_setup(self):
        """Test initialization with experimental_setup."""
        adapter = TransformLoggerAdapter(self.mock_logger, experimental_setup="baseline")
        self.assertEqual(adapter.experimental_setup, "baseline")
        self.assertEqual(adapter.extra["experimental_setup"], "baseline")

    def test_init_with_transform_context(self):
        """Test initialization with transform_context."""
        adapter = TransformLoggerAdapter(self.mock_logger, transform_context="validation")
        self.assertEqual(adapter.transform_context, "validation")
        self.assertEqual(adapter.extra["transform_context"], "validation")

    def test_init_with_both_setup_and_context(self):
        """Test initialization with both experimental_setup and transform_context."""
        adapter = TransformLoggerAdapter(
            self.mock_logger, experimental_setup="augmented", transform_context="composition"
        )
        self.assertEqual(adapter.experimental_setup, "augmented")
        self.assertEqual(adapter.transform_context, "composition")

    def test_init_with_extra_context(self):
        """Test initialization with additional extra context."""
        extra = {"config_source": "yaml", "version": "1.0"}
        adapter = TransformLoggerAdapter(
            self.mock_logger, experimental_setup="baseline", extra=extra
        )
        self.assertEqual(adapter.extra["experimental_setup"], "baseline")
        self.assertEqual(adapter.extra["config_source"], "yaml")

    def test_process_with_experimental_setup_only(self):
        """Test process adds setup prefix when only experimental_setup is set."""
        adapter = TransformLoggerAdapter(self.mock_logger, experimental_setup="baseline")
        processed_msg, _ = adapter.process("Running transforms", {})
        self.assertEqual(processed_msg, "[Setup:baseline] Running transforms")

    def test_process_with_transform_context_only(self):
        """Test process adds context prefix when only transform_context is set."""
        adapter = TransformLoggerAdapter(self.mock_logger, transform_context="validation")
        processed_msg, _ = adapter.process("Validating config", {})
        self.assertEqual(processed_msg, "[Context:validation] Validating config")

    def test_process_with_both_setup_and_context(self):
        """Test process includes both setup and context in prefix."""
        adapter = TransformLoggerAdapter(
            self.mock_logger, experimental_setup="baseline", transform_context="composition"
        )
        processed_msg, _ = adapter.process("Composing", {})
        self.assertEqual(processed_msg, "[Setup:baseline|Context:composition] Composing")

    def test_process_with_transform_name_in_kwargs(self):
        """Test process adds transform name when present in kwargs."""
        adapter = TransformLoggerAdapter(self.mock_logger, experimental_setup="baseline")
        processed_msg, _ = adapter.process("Processing", {"transform_name": "AddSelfLoops"})
        self.assertEqual(processed_msg, "[Setup:baseline|Transform:AddSelfLoops] Processing")

    def test_process_with_all_prefix_parts(self):
        """Test process with setup, context, and transform_name."""
        adapter = TransformLoggerAdapter(
            self.mock_logger, experimental_setup="augmented", transform_context="validation"
        )
        processed_msg, _ = adapter.process("Validating", {"transform_name": "RandomRotate"})
        self.assertEqual(
            processed_msg, "[Setup:augmented|Context:validation|Transform:RandomRotate] Validating"
        )

    def test_process_no_prefix_parts(self):
        """Test process with no prefix parts produces clean message."""
        adapter = TransformLoggerAdapter(self.mock_logger)
        processed_msg, _ = adapter.process("Simple message", {})
        self.assertEqual(processed_msg, "Simple message")

    def test_process_returns_kwargs_unchanged(self):
        """Test process returns kwargs unmodified."""
        adapter = TransformLoggerAdapter(self.mock_logger)
        original_kwargs = {"transform_name": "GCNNorm", "extra": "data"}
        _, returned_kwargs = adapter.process("test", original_kwargs)
        self.assertIs(returned_kwargs, original_kwargs)


# ==============================================================================
# TEST AREA 4: setup_logging()
# ==============================================================================


class TestSetupLogging(unittest.TestCase):
    """Tests for the setup_logging() function."""

    def setUp(self):
        logger_name = "milia_pipeline.logging_config"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()

    def tearDown(self):
        logger_name = "milia_pipeline.logging_config"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()

    @patch("milia_pipeline.logging_config._setup_transform_loggers")
    @patch("milia_pipeline.logging_config._setup_migration_loggers")
    @patch("milia_pipeline.logging_config._setup_handler_loggers")
    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_default_parameters(
        self, mock_inspect, mock_third_party, mock_handler, mock_migration, mock_transform
    ):
        """Test setup_logging with default parameters enables all logging."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        logger = setup_logging()
        self.assertIsInstance(logger, logging.Logger)
        mock_third_party.assert_called_once()
        mock_handler.assert_called_once_with(logger)
        mock_migration.assert_called_once_with(logger)
        mock_transform.assert_called_once_with(logger)

    @patch("milia_pipeline.logging_config._setup_transform_loggers")
    @patch("milia_pipeline.logging_config._setup_migration_loggers")
    @patch("milia_pipeline.logging_config._setup_handler_loggers")
    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_disable_handler_logging(
        self, mock_inspect, mock_third_party, mock_handler, mock_migration, mock_transform
    ):
        """Test setup_logging with handler logging disabled."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        setup_logging(enable_handler_logging=False)
        mock_handler.assert_not_called()
        mock_migration.assert_called_once()
        mock_transform.assert_called_once()

    @patch("milia_pipeline.logging_config._setup_transform_loggers")
    @patch("milia_pipeline.logging_config._setup_migration_loggers")
    @patch("milia_pipeline.logging_config._setup_handler_loggers")
    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_disable_migration_logging(
        self, mock_inspect, mock_third_party, mock_handler, mock_migration, mock_transform
    ):
        """Test setup_logging with migration logging disabled."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        setup_logging(enable_migration_logging=False)
        mock_handler.assert_called_once()
        mock_migration.assert_not_called()
        mock_transform.assert_called_once()

    @patch("milia_pipeline.logging_config._setup_transform_loggers")
    @patch("milia_pipeline.logging_config._setup_migration_loggers")
    @patch("milia_pipeline.logging_config._setup_handler_loggers")
    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_disable_transform_logging(
        self, mock_inspect, mock_third_party, mock_handler, mock_migration, mock_transform
    ):
        """Test setup_logging with transform logging disabled."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        setup_logging(enable_transform_logging=False)
        mock_handler.assert_called_once()
        mock_migration.assert_called_once()
        mock_transform.assert_not_called()

    @patch("milia_pipeline.logging_config._setup_transform_loggers")
    @patch("milia_pipeline.logging_config._setup_migration_loggers")
    @patch("milia_pipeline.logging_config._setup_handler_loggers")
    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_all_disabled(
        self, mock_inspect, mock_third_party, mock_handler, mock_migration, mock_transform
    ):
        """Test setup_logging with all enhanced logging disabled."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        setup_logging(
            enable_handler_logging=False,
            enable_migration_logging=False,
            enable_transform_logging=False,
        )
        mock_handler.assert_not_called()
        mock_migration.assert_not_called()
        mock_transform.assert_not_called()
        mock_third_party.assert_called_once()

    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_log_level_debug(self, mock_inspect, mock_third_party):
        """Test setup_logging sets DEBUG level correctly."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        logger = setup_logging(log_level="DEBUG")
        self.assertEqual(logger.level, logging.DEBUG)

    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_log_level_warning(self, mock_inspect, mock_third_party):
        """Test setup_logging sets WARNING level correctly."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        logger = setup_logging(log_level="WARNING")
        self.assertEqual(logger.level, logging.WARNING)

    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_log_level_case_insensitive(self, mock_inspect, mock_third_party):
        """Test setup_logging handles case-insensitive log level strings."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        logger = setup_logging(log_level="error")
        self.assertEqual(logger.level, logging.ERROR)

    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_no_handler_duplication(self, mock_inspect, mock_third_party):
        """Test setup_logging does not duplicate handlers on multiple calls."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        logger1 = setup_logging()
        handler_count_first = len(logger1.handlers)
        logger2 = setup_logging()
        handler_count_second = len(logger2.handlers)
        self.assertEqual(handler_count_first, handler_count_second)
        self.assertIs(logger1, logger2)

    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_file_handler_oserror_raises(self, mock_inspect, mock_third_party):
        """Test setup_logging raises LoggingConfigurationError on file handler OSError."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/nonexistent/deeply/nested/path/test_script.py"
        with self.assertRaises(LoggingConfigurationError):
            setup_logging()

    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_enhanced_formatter_when_enabled(self, mock_inspect, mock_third_party):
        """Test enhanced formatter is used when any logging enhancement is enabled."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        logger = setup_logging(enable_handler_logging=True)
        for handler in logger.handlers:
            if handler.formatter:
                fmt = handler.formatter._fmt
                self.assertIn("%(name)s", fmt)
                self.assertIn("%(lineno)d", fmt)
                break

    @patch("milia_pipeline.logging_config._configure_third_party_loggers")
    @patch("milia_pipeline.logging_config.inspect")
    def test_setup_logging_standard_formatter_when_all_disabled(
        self, mock_inspect, mock_third_party
    ):
        """Test standard formatter is used when all enhancements are disabled."""
        mock_inspect.currentframe.return_value = MagicMock()
        mock_inspect.getfile.return_value = "/tmp/test_script.py"
        logger = setup_logging(
            enable_handler_logging=False,
            enable_migration_logging=False,
            enable_transform_logging=False,
        )
        for handler in logger.handlers:
            if handler.formatter:
                fmt = handler.formatter._fmt
                self.assertNotIn("%(name)s", fmt)
                self.assertNotIn("%(lineno)d", fmt)
                break


# ==============================================================================
# TEST AREAS 5-8: Internal setup functions
# ==============================================================================


class TestConfigureThirdPartyLoggers(unittest.TestCase):
    """Tests for _configure_third_party_loggers() internal function."""

    def test_rdkit_logger_set_to_error(self):
        """Test RDKit logger is set to ERROR level."""
        logging_config._configure_third_party_loggers()
        rdkit_logger = logging.getLogger("rdkit")
        self.assertEqual(rdkit_logger.level, logging.ERROR)

    def test_verbose_loggers_set_to_warning(self):
        """Test verbose third-party loggers are set to WARNING level."""
        logging_config._configure_third_party_loggers()
        expected = ["matplotlib", "PIL", "urllib3", "requests", "torch", "torch_geometric"]
        for logger_name in expected:
            logger = logging.getLogger(logger_name)
            self.assertEqual(logger.level, logging.WARNING, f"'{logger_name}' should be WARNING")


class TestSetupHandlerLoggers(unittest.TestCase):
    """Tests for _setup_handler_loggers() internal function."""

    def test_handler_loggers_created(self):
        """Test handler loggers are created for DFT, DMC, GENERIC."""
        base_logger = MagicMock(spec=logging.Logger)
        base_logger.level = logging.INFO
        logging_config._setup_handler_loggers(base_logger)
        for handler_type in ["dft", "dmc", "generic"]:
            logger = logging.getLogger(f"handler.{handler_type}")
            self.assertEqual(logger.level, base_logger.level)
            self.assertTrue(logger.propagate)

    def test_handler_loggers_inherit_level(self):
        """Test handler loggers inherit the base logger's level."""
        base_logger = MagicMock(spec=logging.Logger)
        base_logger.level = logging.DEBUG
        logging_config._setup_handler_loggers(base_logger)
        for handler_type in ["dft", "dmc", "generic"]:
            logger = logging.getLogger(f"handler.{handler_type}")
            self.assertEqual(logger.level, logging.DEBUG)


class TestSetupMigrationLoggers(unittest.TestCase):
    """Tests for _setup_migration_loggers() internal function."""

    def test_migration_loggers_created(self):
        """Test migration loggers are created for all phases."""
        base_logger = MagicMock(spec=logging.Logger)
        base_logger.level = logging.INFO
        logging_config._setup_migration_loggers(base_logger)
        for phase in ["6F", "6G", "6H", "6I", "6J"]:
            logger = logging.getLogger(f"migration.phase_{phase}")
            self.assertEqual(logger.level, base_logger.level)
            self.assertTrue(logger.propagate)


class TestSetupTransformLoggers(unittest.TestCase):
    """Tests for _setup_transform_loggers() internal function."""

    def test_transform_loggers_created(self):
        """Test transform loggers are created for all operations."""
        base_logger = MagicMock(spec=logging.Logger)
        base_logger.level = logging.INFO
        logging_config._setup_transform_loggers(base_logger)
        for operation in ["registry", "validation", "composition", "experimental"]:
            logger = logging.getLogger(f"transform.{operation}")
            self.assertEqual(logger.level, base_logger.level)
            self.assertTrue(logger.propagate)


# ==============================================================================
# TEST AREAS 9-11: Factory functions
# ==============================================================================


class TestCreateHandlerLogger(unittest.TestCase):
    """Tests for create_handler_logger() factory function."""

    def test_create_with_defaults(self):
        adapter = create_handler_logger("DFT")
        self.assertIsInstance(adapter, HandlerLoggerAdapter)
        self.assertEqual(adapter.handler_type, "DFT")

    def test_create_with_custom_base_logger(self):
        custom_logger = logging.getLogger("custom.handler")
        adapter = create_handler_logger("DMC", base_logger=custom_logger)
        self.assertIs(adapter.logger, custom_logger)

    def test_create_with_extra_context(self):
        extra = {"batch_id": "batch_001"}
        adapter = create_handler_logger("DFT", extra_context=extra)
        self.assertEqual(adapter.extra["batch_id"], "batch_001")

    def test_default_base_logger_name(self):
        adapter = create_handler_logger("DFT")
        self.assertEqual(adapter.logger.name, "handler.dft")


class TestCreateMigrationLogger(unittest.TestCase):
    """Tests for create_migration_logger() factory function."""

    def test_create_with_defaults(self):
        adapter = create_migration_logger("6F")
        self.assertIsInstance(adapter, MigrationLoggerAdapter)
        self.assertEqual(adapter.migration_phase, "6F")

    def test_create_with_custom_base_logger(self):
        custom_logger = logging.getLogger("custom.migration")
        adapter = create_migration_logger("6G", base_logger=custom_logger)
        self.assertIs(adapter.logger, custom_logger)

    def test_create_with_extra_context(self):
        extra = {"module": "property_enrichment.py"}
        adapter = create_migration_logger("6F", extra_context=extra)
        self.assertEqual(adapter.extra["module"], "property_enrichment.py")

    def test_default_base_logger_name(self):
        adapter = create_migration_logger("6H")
        self.assertEqual(adapter.logger.name, "migration.phase_6H")


class TestCreateTransformLogger(unittest.TestCase):
    """Tests for create_transform_logger() factory function."""

    def test_create_with_no_args(self):
        adapter = create_transform_logger()
        self.assertIsInstance(adapter, TransformLoggerAdapter)
        self.assertIsNone(adapter.experimental_setup)
        self.assertIsNone(adapter.transform_context)

    def test_create_with_experimental_setup(self):
        adapter = create_transform_logger(experimental_setup="baseline")
        self.assertEqual(adapter.experimental_setup, "baseline")

    def test_create_with_transform_context(self):
        adapter = create_transform_logger(transform_context="validation")
        self.assertEqual(adapter.transform_context, "validation")
        self.assertEqual(adapter.logger.name, "transform.validation")

    def test_create_with_both_setup_and_context(self):
        adapter = create_transform_logger(
            experimental_setup="augmented", transform_context="composition"
        )
        self.assertEqual(adapter.experimental_setup, "augmented")
        self.assertEqual(adapter.transform_context, "composition")

    def test_default_logger_name_with_context(self):
        adapter = create_transform_logger(transform_context="registry")
        self.assertEqual(adapter.logger.name, "transform.registry")

    def test_default_logger_name_without_context(self):
        adapter = create_transform_logger()
        self.assertEqual(adapter.logger.name, "transform.experimental")

    def test_create_with_custom_base_logger(self):
        custom_logger = logging.getLogger("custom.transform")
        adapter = create_transform_logger(base_logger=custom_logger)
        self.assertIs(adapter.logger, custom_logger)

    def test_create_with_extra_context(self):
        extra = {"config_source": "yaml"}
        adapter = create_transform_logger(experimental_setup="baseline", extra_context=extra)
        self.assertEqual(adapter.extra["config_source"], "yaml")


# ==============================================================================
# TEST AREAS 12-14: Decorators
# ==============================================================================


class TestLogHandlerOperationDecorator(unittest.TestCase):
    """Tests for log_handler_operation() decorator."""

    def test_decorator_preserves_return_value(self):
        mock_handler = MagicMock()
        mock_handler.get_dataset_type.return_value = "DFT"

        @log_handler_operation("test_operation")
        def my_func(self_arg):
            return "result_value"

        self.assertEqual(my_func(mock_handler), "result_value")

    def test_decorator_preserves_function_name(self):
        @log_handler_operation("test_op")
        def my_special_function(self_arg):
            pass

        self.assertEqual(my_special_function.__name__, "my_special_function")

    def test_decorator_handler_error_reraises(self):
        mock_handler = MagicMock()
        mock_handler.get_dataset_type.return_value = "DFT"

        @log_handler_operation("validate")
        def failing_func(self_arg):
            raise HandlerError(message="handler failure")

        with self.assertRaises(HandlerError):
            failing_func(mock_handler)

    def test_decorator_unexpected_error_reraises(self):
        mock_handler = MagicMock()
        mock_handler.get_dataset_type.return_value = "DFT"

        @log_handler_operation("process")
        def failing_func(self_arg):
            raise ValueError("unexpected")

        with self.assertRaises(ValueError):
            failing_func(mock_handler)

    def test_decorator_extracts_handler_type(self):
        mock_handler = MagicMock()
        mock_handler.get_dataset_type.return_value = "DMC"

        @log_handler_operation("test_op")
        def my_func(self_arg):
            return "ok"

        my_func(mock_handler)
        mock_handler.get_dataset_type.assert_called()

    def test_decorator_handles_no_args(self):
        @log_handler_operation("test_op")
        def my_func():
            return "ok"

        self.assertEqual(my_func(), "ok")


class TestLogMigrationStepDecorator(unittest.TestCase):
    """Tests for log_migration_step() decorator."""

    def test_decorator_preserves_return_value(self):
        @log_migration_step("test_step", "6F")
        def migrate_func():
            return "migration_result"

        self.assertEqual(migrate_func(), "migration_result")

    def test_decorator_preserves_function_name(self):
        @log_migration_step("test_step", "6G")
        def my_migration():
            pass

        self.assertEqual(my_migration.__name__, "my_migration")

    def test_decorator_migration_error_reraises(self):
        @log_migration_step("failing_step", "6H")
        def failing_migration():
            raise MigrationError(message="migration failed", migration_phase="6H")

        with self.assertRaises(MigrationError):
            failing_migration()

    def test_decorator_unexpected_error_reraises(self):
        @log_migration_step("test_step", "6I")
        def failing_migration():
            raise RuntimeError("unexpected")

        with self.assertRaises(RuntimeError):
            failing_migration()


class TestLogTransformOperationDecorator(unittest.TestCase):
    """Tests for log_transform_operation() decorator."""

    def test_decorator_preserves_return_value(self):
        @log_transform_operation("validate_params", transform_context="validation")
        def validate_func():
            return "validated"

        self.assertEqual(validate_func(), "validated")

    def test_decorator_preserves_function_name(self):
        @log_transform_operation("compose", transform_context="composition")
        def compose_transforms():
            pass

        self.assertEqual(compose_transforms.__name__, "compose_transforms")

    def test_decorator_transform_config_error_reraises(self):
        @log_transform_operation("configure", transform_context="validation")
        def failing_func():
            raise TransformConfigurationError(message="config error")

        with self.assertRaises(TransformConfigurationError):
            failing_func()

    def test_decorator_transform_validation_error_reraises(self):
        @log_transform_operation("validate", transform_context="validation")
        def failing_func():
            raise TransformValidationError(
                message="validation error", transform_name="TestTransform"
            )

        with self.assertRaises(TransformValidationError):
            failing_func()

    def test_decorator_transform_composition_error_reraises(self):
        @log_transform_operation("compose", transform_context="composition")
        def failing_func():
            raise TransformCompositionError(message="composition error")

        with self.assertRaises(TransformCompositionError):
            failing_func()

    def test_decorator_experimental_setup_error_reraises(self):
        @log_transform_operation("setup", transform_context="experimental")
        def failing_func():
            raise ExperimentalSetupError(message="setup error")

        with self.assertRaises(ExperimentalSetupError):
            failing_func()

    def test_decorator_unexpected_error_reraises(self):
        @log_transform_operation("test_op")
        def failing_func():
            raise TypeError("unexpected")

        with self.assertRaises(TypeError):
            failing_func()

    def test_decorator_uses_experimental_setup_from_kwargs(self):
        @log_transform_operation("test_op")
        def my_func(experimental_setup=None):
            return "ok"

        self.assertEqual(my_func(experimental_setup="baseline"), "ok")


# ==============================================================================
# TEST AREAS 15-16: Performance logging
# ==============================================================================


class TestLogHandlerPerformance(unittest.TestCase):
    """Tests for log_handler_performance() function."""

    def test_basic_performance_logging(self):
        mock_logger = MagicMock()
        log_handler_performance(mock_logger, "batch_validation", 100, 45.2)
        mock_logger.info.assert_called_once()
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("Performance metrics", call_msg)
        self.assertIn("operation=batch_validation", call_msg)
        self.assertIn("molecule_count=100", call_msg)

    def test_performance_logging_with_errors(self):
        mock_logger = MagicMock()
        log_handler_performance(mock_logger, "validation", 100, 10.0, error_count=5)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("error_count=5", call_msg)
        self.assertIn("success_rate=95.0", call_msg)

    def test_performance_logging_with_additional_metrics(self):
        mock_logger = MagicMock()
        log_handler_performance(
            mock_logger, "process", 50, 25.0, additional_metrics={"memory_used_mb": 128.5}
        )
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("memory_used_mb=128.5", call_msg)

    def test_performance_logging_zero_execution_time(self):
        mock_logger = MagicMock()
        log_handler_performance(mock_logger, "fast_op", 10, 0.0)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("molecules_per_sec=0", call_msg)

    def test_performance_logging_zero_molecule_count(self):
        mock_logger = MagicMock()
        log_handler_performance(mock_logger, "empty_op", 0, 1.0)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("success_rate=0", call_msg)


class TestLogTransformPerformance(unittest.TestCase):
    """Tests for log_transform_performance() function."""

    def test_basic_transform_performance(self):
        mock_logger = MagicMock()
        log_transform_performance(mock_logger, "composition", 5, 0.023)
        mock_logger.info.assert_called_once()
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("Transform performance", call_msg)
        self.assertIn("operation=composition", call_msg)

    def test_transform_performance_with_setup(self):
        mock_logger = MagicMock()
        log_transform_performance(
            mock_logger, "composition", 5, 0.023, experimental_setup="baseline"
        )
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("experimental_setup=baseline", call_msg)

    def test_transform_performance_with_errors(self):
        mock_logger = MagicMock()
        log_transform_performance(mock_logger, "validation", 10, 1.0, error_count=2)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("error_count=2", call_msg)
        self.assertIn("success_rate=80.0", call_msg)

    def test_transform_performance_with_additional_metrics(self):
        mock_logger = MagicMock()
        log_transform_performance(
            mock_logger, "composition", 3, 0.5, additional_metrics={"cache_hits": 2}
        )
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("cache_hits=2", call_msg)

    def test_transform_performance_zero_execution_time(self):
        mock_logger = MagicMock()
        log_transform_performance(mock_logger, "fast_op", 5, 0.0)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("transforms_per_sec=0", call_msg)

    def test_transform_performance_zero_count(self):
        mock_logger = MagicMock()
        log_transform_performance(mock_logger, "empty_op", 0, 1.0)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("success_rate=0", call_msg)


# ==============================================================================
# TEST AREAS 17-19: Logger retrieval functions
# ==============================================================================


class TestLoggerRetrievalFunctions(unittest.TestCase):
    """Tests for logger retrieval functions."""

    def test_get_handler_logger_by_type(self):
        logger = get_handler_logger_by_type("DFT")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "handler.dft")

    def test_get_handler_logger_by_type_dmc(self):
        logger = get_handler_logger_by_type("DMC")
        self.assertEqual(logger.name, "handler.dmc")

    def test_get_migration_logger_by_phase(self):
        logger = get_migration_logger_by_phase("6F")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "migration.phase_6F")

    def test_get_migration_logger_by_phase_all_phases(self):
        for phase in ["6F", "6G", "6H", "6I", "6J"]:
            logger = get_migration_logger_by_phase(phase)
            self.assertEqual(logger.name, f"migration.phase_{phase}")

    def test_get_transform_logger_by_operation(self):
        logger = get_transform_logger_by_operation("validation")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "transform.validation")

    def test_get_transform_logger_by_operation_all_operations(self):
        for op in ["registry", "validation", "composition", "experimental"]:
            logger = get_transform_logger_by_operation(op)
            self.assertEqual(logger.name, f"transform.{op}")


# ==============================================================================
# TEST AREA 20: log_exception_with_context()
# ==============================================================================


class TestLogExceptionWithContext(unittest.TestCase):
    """Tests for log_exception_with_context() function."""

    def test_basic_exception_logging(self):
        mock_logger = MagicMock()
        exc = ValueError("test error")
        with patch("milia_pipeline.exceptions.get_exception_recovery_suggestions", return_value=[]):
            log_exception_with_context(mock_logger, exc, "test_operation")
        mock_logger.error.assert_called_once()
        call_msg = mock_logger.error.call_args[0][0]
        self.assertIn("Exception in test_operation", call_msg)
        self.assertIn("operation=test_operation", call_msg)

    def test_exception_with_context_dict(self):
        mock_logger = MagicMock()
        exc = ValueError("test error")
        context = {"molecule_index": 42, "handler_type": "DFT"}
        with patch("milia_pipeline.exceptions.get_exception_recovery_suggestions", return_value=[]):
            log_exception_with_context(mock_logger, exc, "validation", context=context)
        call_msg = mock_logger.error.call_args[0][0]
        self.assertIn("molecule_index=42", call_msg)
        self.assertIn("handler_type=DFT", call_msg)

    def test_handler_error_adds_handler_context(self):
        mock_logger = MagicMock()
        exc = HandlerError(message="handler failure")
        exc.handler_type = "DFT"
        exc.handler_operation = "validate"
        with patch("milia_pipeline.exceptions.get_exception_recovery_suggestions", return_value=[]):
            log_exception_with_context(mock_logger, exc, "handler_op")
        call_msg = mock_logger.error.call_args[0][0]
        self.assertIn("handler_type=DFT", call_msg)
        self.assertIn("handler_operation=validate", call_msg)

    def test_transform_error_adds_transform_context(self):
        mock_logger = MagicMock()
        exc = TransformConfigurationError(message="config error")
        exc.experimental_setup = "baseline"
        exc.config_source = "yaml"
        exc.transform_name = "AddSelfLoops"
        with patch("milia_pipeline.exceptions.get_exception_recovery_suggestions", return_value=[]):
            log_exception_with_context(mock_logger, exc, "transform_op")
        call_msg = mock_logger.error.call_args[0][0]
        self.assertIn("experimental_setup=baseline", call_msg)
        self.assertIn("config_source=yaml", call_msg)
        self.assertIn("transform_name=AddSelfLoops", call_msg)

    def test_exception_with_molecule_index_attribute(self):
        mock_logger = MagicMock()
        exc = ValueError("test")
        exc.molecule_index = 99
        with patch("milia_pipeline.exceptions.get_exception_recovery_suggestions", return_value=[]):
            log_exception_with_context(mock_logger, exc, "process")
        call_msg = mock_logger.error.call_args[0][0]
        self.assertIn("molecule_index=99", call_msg)

    def test_recovery_suggestions_logged_when_available(self):
        mock_logger = MagicMock()
        exc = ValueError("test")
        suggestions = ["Try reconfiguring the handler", "Check input data"]
        with patch(
            "milia_pipeline.exceptions.get_exception_recovery_suggestions", return_value=suggestions
        ):
            log_exception_with_context(mock_logger, exc, "operation")
        mock_logger.error.assert_called_once()
        mock_logger.info.assert_called_once()
        info_msg = mock_logger.info.call_args[0][0]
        self.assertIn("Recovery suggestions", info_msg)

    def test_no_recovery_suggestions_no_info_log(self):
        mock_logger = MagicMock()
        exc = ValueError("test")
        with patch("milia_pipeline.exceptions.get_exception_recovery_suggestions", return_value=[]):
            log_exception_with_context(mock_logger, exc, "operation")
        mock_logger.error.assert_called_once()
        mock_logger.info.assert_not_called()


# ==============================================================================
# TEST AREA 21: log_experimental_setup_switch()
# ==============================================================================


class TestLogExperimentalSetupSwitch(unittest.TestCase):
    """Tests for log_experimental_setup_switch() function."""

    def test_successful_switch(self):
        mock_logger = MagicMock()
        log_experimental_setup_switch(mock_logger, "baseline", "augmented", True)
        mock_logger.info.assert_called_once()
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("successful", call_msg)
        self.assertIn("baseline", call_msg)
        self.assertIn("augmented", call_msg)

    def test_successful_switch_with_transform_count(self):
        mock_logger = MagicMock()
        log_experimental_setup_switch(mock_logger, "baseline", "augmented", True, transform_count=5)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("5 transforms", call_msg)

    def test_successful_switch_from_none(self):
        mock_logger = MagicMock()
        log_experimental_setup_switch(mock_logger, None, "baseline", True)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("None", call_msg)
        self.assertIn("baseline", call_msg)

    def test_failed_switch(self):
        mock_logger = MagicMock()
        log_experimental_setup_switch(mock_logger, "baseline", "augmented", False)
        mock_logger.error.assert_called_once()
        call_msg = mock_logger.error.call_args[0][0]
        self.assertIn("failed", call_msg)

    def test_failed_switch_with_error(self):
        mock_logger = MagicMock()
        error = ExperimentalSetupError(message="Invalid config")
        log_experimental_setup_switch(mock_logger, "baseline", "broken", False, error=error)
        call_msg = mock_logger.error.call_args[0][0]
        self.assertIn("ExperimentalSetupError", call_msg)


# ==============================================================================
# TEST AREA 22: log_transform_validation_results()
# ==============================================================================


class TestLogTransformValidationResults(unittest.TestCase):
    """Tests for log_transform_validation_results() function."""

    def test_valid_results_no_warnings(self):
        mock_logger = MagicMock()
        results = {"valid": True, "warnings": [], "errors": []}
        log_transform_validation_results(mock_logger, "AddSelfLoops", results)
        mock_logger.info.assert_called_once()
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("AddSelfLoops", call_msg)
        self.assertIn("passed", call_msg)

    def test_valid_results_with_warnings(self):
        mock_logger = MagicMock()
        results = {"valid": True, "warnings": ["Parameter X may cause issues"], "errors": []}
        log_transform_validation_results(mock_logger, "RandomRotate", results)
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_called()

    def test_invalid_results_with_errors(self):
        mock_logger = MagicMock()
        results = {"valid": False, "warnings": [], "errors": ["Missing param", "Invalid type"]}
        log_transform_validation_results(mock_logger, "BadTransform", results)
        mock_logger.error.assert_called()
        error_calls = [c[0][0] for c in mock_logger.error.call_args_list]
        self.assertTrue(any("failed" in msg for msg in error_calls))

    def test_with_experimental_setup_prefix(self):
        mock_logger = MagicMock()
        results = {"valid": True, "warnings": [], "errors": []}
        log_transform_validation_results(
            mock_logger, "AddSelfLoops", results, experimental_setup="baseline"
        )
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("[Setup:baseline]", call_msg)

    def test_without_experimental_setup_prefix(self):
        mock_logger = MagicMock()
        results = {"valid": True, "warnings": [], "errors": []}
        log_transform_validation_results(mock_logger, "AddSelfLoops", results)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertNotIn("[Setup:", call_msg)


# ==============================================================================
# TEST AREA 23: log_transform_composition_summary()
# ==============================================================================


class TestLogTransformCompositionSummary(unittest.TestCase):
    """Tests for log_transform_composition_summary() function."""

    def test_basic_composition_summary(self):
        mock_logger = MagicMock()
        results = {
            "transform_count": 5,
            "composition_time": 0.023,
            "cache_hit": True,
            "warnings": [],
            "transform_sequence": ["AddSelfLoops", "ToUndirected"],
        }
        log_transform_composition_summary(mock_logger, results)
        info_msg = mock_logger.info.call_args_list[0][0][0]
        self.assertIn("5 transforms", info_msg)
        self.assertIn("HIT", info_msg)

    def test_composition_summary_cache_miss(self):
        mock_logger = MagicMock()
        results = {
            "transform_count": 3,
            "composition_time": 0.05,
            "cache_hit": False,
            "warnings": [],
            "transform_sequence": [],
        }
        log_transform_composition_summary(mock_logger, results)
        info_msg = mock_logger.info.call_args_list[0][0][0]
        self.assertIn("MISS", info_msg)

    def test_composition_summary_with_warnings(self):
        mock_logger = MagicMock()
        results = {
            "transform_count": 5,
            "composition_time": 0.023,
            "cache_hit": True,
            "warnings": ["ToUndirected appears multiple times"],
            "transform_sequence": ["AddSelfLoops", "ToUndirected"],
        }
        log_transform_composition_summary(mock_logger, results)
        mock_logger.warning.assert_called_once()

    def test_composition_summary_with_experimental_setup(self):
        mock_logger = MagicMock()
        results = {
            "transform_count": 2,
            "composition_time": 0.01,
            "cache_hit": False,
            "warnings": [],
            "transform_sequence": ["AddSelfLoops"],
        }
        log_transform_composition_summary(mock_logger, results, experimental_setup="baseline")
        info_msg = mock_logger.info.call_args_list[0][0][0]
        self.assertIn("[Setup:baseline]", info_msg)

    def test_composition_summary_empty_results(self):
        mock_logger = MagicMock()
        log_transform_composition_summary(mock_logger, {})
        mock_logger.info.assert_called()


# ==============================================================================
# TEST AREAS 24-26: Debug configuration and third-party silencing
# ==============================================================================


class TestConfigureDebugLoggingForHandlers(unittest.TestCase):
    """Tests for configure_debug_logging_for_handlers() function."""

    def test_handler_loggers_set_to_debug(self):
        configure_debug_logging_for_handlers()
        for name in ["handler.dft", "handler.dmc", "handler.generic"]:
            logger = logging.getLogger(name)
            self.assertEqual(logger.level, logging.DEBUG)

    def test_migration_loggers_set_to_debug(self):
        configure_debug_logging_for_handlers()
        for phase in ["6f", "6g", "6h", "6i", "6j"]:
            logger = logging.getLogger(f"migration.phase_{phase}")
            self.assertEqual(logger.level, logging.DEBUG)


class TestConfigureDebugLoggingForTransforms(unittest.TestCase):
    """Tests for configure_debug_logging_for_transforms() function."""

    def test_transform_loggers_set_to_debug(self):
        configure_debug_logging_for_transforms()
        for op in ["registry", "validation", "composition", "experimental"]:
            logger = logging.getLogger(f"transform.{op}")
            self.assertEqual(logger.level, logging.DEBUG)


class TestDisableVerboseThirdPartyLogging(unittest.TestCase):
    """Tests for disable_verbose_third_party_logging() function."""

    def test_rdkit_set_to_error(self):
        disable_verbose_third_party_logging()
        self.assertEqual(logging.getLogger("rdkit").level, logging.ERROR)

    def test_science_libraries_set_to_warning(self):
        disable_verbose_third_party_logging()
        for name in [
            "matplotlib",
            "PIL",
            "urllib3",
            "requests",
            "torch",
            "torch_geometric",
            "numpy",
            "scipy",
        ]:
            self.assertEqual(
                logging.getLogger(name).level, logging.WARNING, f"'{name}' should be WARNING"
            )


# ==============================================================================
# TEST AREA 27: create_experimental_setup_hash()
# ==============================================================================


class TestCreateExperimentalSetupHash(unittest.TestCase):
    """Tests for create_experimental_setup_hash() function."""

    def test_basic_hash_generation(self):
        config = [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}]
        hash_val = create_experimental_setup_hash(config)
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 12)

    def test_hash_is_deterministic(self):
        config = [{"name": "AddSelfLoops"}, {"name": "RandomRotate", "kwargs": {"degrees": 180}}]
        self.assertEqual(
            create_experimental_setup_hash(config), create_experimental_setup_hash(config)
        )

    def test_different_configs_different_hashes(self):
        config1 = [{"name": "AddSelfLoops"}]
        config2 = [{"name": "ToUndirected"}]
        self.assertNotEqual(
            create_experimental_setup_hash(config1), create_experimental_setup_hash(config2)
        )

    def test_order_independent_hashing(self):
        """Test hash is independent of spec order (sorted internally)."""
        config1 = [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}]
        config2 = [{"name": "ToUndirected"}, {"name": "AddSelfLoops"}]
        self.assertEqual(
            create_experimental_setup_hash(config1), create_experimental_setup_hash(config2)
        )

    def test_hash_with_kwargs(self):
        config_no_kwargs = [{"name": "RandomRotate"}]
        config_with_kwargs = [{"name": "RandomRotate", "kwargs": {"degrees": 180}}]
        self.assertNotEqual(
            create_experimental_setup_hash(config_no_kwargs),
            create_experimental_setup_hash(config_with_kwargs),
        )

    def test_hash_empty_config(self):
        hash_val = create_experimental_setup_hash([])
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 12)

    def test_hash_is_hexadecimal(self):
        hash_val = create_experimental_setup_hash([{"name": "AddSelfLoops"}])
        int(hash_val, 16)  # Raises ValueError if not valid hex


# ==============================================================================
# TEST AREA 28: log_experimental_setup_summary()
# ==============================================================================


class TestLogExperimentalSetupSummary(unittest.TestCase):
    """Tests for log_experimental_setup_summary() function."""

    def test_basic_summary_logging(self):
        mock_logger = MagicMock()
        config = [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}]
        log_experimental_setup_summary(mock_logger, "baseline", config)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        self.assertTrue(any("baseline" in msg for msg in info_calls))
        self.assertTrue(any("Transform count: 2" in msg for msg in info_calls))

    def test_summary_with_validation_passed(self):
        mock_logger = MagicMock()
        config = [{"name": "AddSelfLoops"}]
        validation = {"valid": True, "warnings": [], "errors": []}
        log_experimental_setup_summary(
            mock_logger, "baseline", config, validation_results=validation
        )
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        self.assertTrue(any("PASSED" in msg for msg in info_calls))

    def test_summary_with_validation_failed(self):
        mock_logger = MagicMock()
        config = [{"name": "BadTransform"}]
        validation = {"valid": False, "warnings": ["W1"], "errors": ["E1", "E2"]}
        log_experimental_setup_summary(mock_logger, "broken", config, validation_results=validation)
        mock_logger.warning.assert_called()
        mock_logger.error.assert_called()

    def test_summary_without_validation(self):
        mock_logger = MagicMock()
        config = [{"name": "AddSelfLoops"}]
        log_experimental_setup_summary(mock_logger, "baseline", config)
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_summary_logs_transform_sequence(self):
        mock_logger = MagicMock()
        config = [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}, {"name": "GCNNorm"}]
        log_experimental_setup_summary(mock_logger, "test", config)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        self.assertTrue(any("AddSelfLoops" in msg and "ToUndirected" in msg for msg in info_calls))

    def test_summary_truncates_warnings_to_three(self):
        mock_logger = MagicMock()
        config = [{"name": "Test"}]
        validation = {"valid": False, "warnings": ["W1", "W2", "W3", "W4", "W5"], "errors": []}
        log_experimental_setup_summary(mock_logger, "test", config, validation_results=validation)
        warning_calls = mock_logger.warning.call_args_list
        # 1 FAILED warning + 3 individual warnings (first 3 of 5)
        self.assertEqual(len(warning_calls), 4)

    def test_summary_truncates_errors_to_three(self):
        mock_logger = MagicMock()
        config = [{"name": "Test"}]
        validation = {"valid": False, "warnings": [], "errors": ["E1", "E2", "E3", "E4", "E5"]}
        log_experimental_setup_summary(mock_logger, "test", config, validation_results=validation)
        error_calls = mock_logger.error.call_args_list
        # "Validation errors: 5" + 3 individual errors
        self.assertEqual(len(error_calls), 4)


# ==============================================================================
# TEST AREA 29: setup_basic_logging() backward compatibility
# ==============================================================================


class TestSetupBasicLogging(unittest.TestCase):
    """Tests for setup_basic_logging() backward compatibility function."""

    @patch("milia_pipeline.logging_config.setup_logging")
    def test_calls_setup_logging_with_all_disabled(self, mock_setup):
        mock_setup.return_value = MagicMock(spec=logging.Logger)
        setup_basic_logging()
        mock_setup.assert_called_once_with(
            enable_handler_logging=False,
            enable_migration_logging=False,
            enable_transform_logging=False,
        )

    @patch("milia_pipeline.logging_config.setup_logging")
    def test_returns_logger(self, mock_setup):
        expected_logger = MagicMock(spec=logging.Logger)
        mock_setup.return_value = expected_logger
        self.assertIs(setup_basic_logging(), expected_logger)


# ==============================================================================
# TEST AREA 30: Edge cases and boundary conditions
# ==============================================================================


class TestEdgeCasesAndBoundaryConditions(unittest.TestCase):
    """Tests for edge cases and boundary conditions across all functions."""

    def test_handler_adapter_empty_handler_type(self):
        mock_logger = MagicMock(spec=logging.Logger)
        mock_logger.manager = MagicMock()
        mock_logger.parent = None
        adapter = HandlerLoggerAdapter(mock_logger, handler_type="")
        processed_msg, _ = adapter.process("test", {})
        self.assertEqual(processed_msg, "[] test")

    def test_migration_adapter_empty_phase(self):
        mock_logger = MagicMock(spec=logging.Logger)
        mock_logger.manager = MagicMock()
        mock_logger.parent = None
        adapter = MigrationLoggerAdapter(mock_logger, migration_phase="")
        processed_msg, _ = adapter.process("test", {})
        self.assertEqual(processed_msg, "[Migration-] test")

    def test_transform_adapter_empty_strings(self):
        """Test TransformLoggerAdapter with empty string setup and context.
        Empty strings are falsy in Python, so they are skipped by the
        'if self.experimental_setup:' / 'if self.transform_context:' checks."""
        mock_logger = MagicMock(spec=logging.Logger)
        mock_logger.manager = MagicMock()
        mock_logger.parent = None
        adapter = TransformLoggerAdapter(mock_logger, experimental_setup="", transform_context="")
        processed_msg, _ = adapter.process("test", {})
        self.assertEqual(processed_msg, "test")

    def test_handler_performance_large_numbers(self):
        mock_logger = MagicMock()
        log_handler_performance(mock_logger, "large_batch", 1_000_000, 3600.0, error_count=100)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("molecule_count=1000000", call_msg)

    def test_transform_performance_very_small_time(self):
        mock_logger = MagicMock()
        log_transform_performance(mock_logger, "micro_op", 1, 0.001)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertIn("execution_time_sec=0.001", call_msg)

    def test_experimental_setup_hash_config_missing_name(self):
        config = [{"kwargs": {"degrees": 180}}]
        hash_val = create_experimental_setup_hash(config)
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 12)

    def test_experimental_setup_hash_config_missing_kwargs(self):
        config = [{"name": "AddSelfLoops"}]
        hash_val = create_experimental_setup_hash(config)
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 12)

    def test_log_exception_with_context_none_context(self):
        mock_logger = MagicMock()
        exc = ValueError("test")
        with patch("milia_pipeline.exceptions.get_exception_recovery_suggestions", return_value=[]):
            log_exception_with_context(mock_logger, exc, "op", context=None)
        mock_logger.error.assert_called_once()

    def test_composition_summary_missing_keys(self):
        mock_logger = MagicMock()
        log_transform_composition_summary(mock_logger, {"transform_count": 0})
        mock_logger.info.assert_called()

    def test_setup_summary_unknown_transform_names(self):
        mock_logger = MagicMock()
        config = [{"kwargs": {"param": 1}}, {}]
        log_experimental_setup_summary(mock_logger, "test", config)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        self.assertTrue(any("unknown" in msg for msg in info_calls))

    def test_handler_operation_decorator_with_kwargs(self):
        mock_handler = MagicMock()
        mock_handler.get_dataset_type.return_value = "DFT"

        @log_handler_operation("test_op")
        def my_func(self_arg, key1=None, key2=None):
            return (key1, key2)

        self.assertEqual(my_func(mock_handler, key1="a", key2="b"), ("a", "b"))

    def test_transform_operation_decorator_extracts_transform_name_kwarg(self):
        @log_transform_operation("test_op", transform_context="validation")
        def my_func(transform_name=None):
            return transform_name

        self.assertEqual(my_func(transform_name="AddSelfLoops"), "AddSelfLoops")

    def test_transform_operation_decorator_extracts_name_kwarg(self):
        @log_transform_operation("test_op", transform_context="validation")
        def my_func(name=None):
            return name

        self.assertEqual(my_func(name="GCNNorm"), "GCNNorm")

    def test_validation_results_empty_valid_key(self):
        """Test log_transform_validation_results when 'valid' key is missing."""
        mock_logger = MagicMock()
        results = {"warnings": [], "errors": []}
        log_transform_validation_results(mock_logger, "Test", results)
        mock_logger.error.assert_called()

    def test_experimental_setup_switch_no_transform_count(self):
        mock_logger = MagicMock()
        log_experimental_setup_switch(mock_logger, "old", "new", True, transform_count=None)
        call_msg = mock_logger.info.call_args[0][0]
        self.assertNotIn("transforms", call_msg)

    def test_experimental_setup_switch_no_error_on_failure(self):
        mock_logger = MagicMock()
        log_experimental_setup_switch(mock_logger, "old", "new", False, error=None)
        call_msg = mock_logger.error.call_args[0][0]
        self.assertIn("failed", call_msg)


# ==============================================================================
# MODULE TEARDOWN
# ==============================================================================


def teardown_module():
    """
    Clean up any injected mocks at module teardown.

    CRITICAL: This function ensures no mock pollution persists after test module execution.
    It removes any modules that might have been injected during test setup.

    Note: This is a safety net - tests should use @patch decorators instead of
    sys.modules injection to avoid pollution in the first place.
    """
    mocks_to_remove = [
        "milia_pipeline.logging_config",
        "milia_pipeline.exceptions",
    ]
    for mock_module in mocks_to_remove:
        sys.modules.pop(mock_module, None)


# ==============================================================================
# MAIN TEST EXECUTION
# ==============================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHandlerLoggerAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestMigrationLoggerAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformLoggerAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestSetupLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigureThirdPartyLoggers))
    suite.addTests(loader.loadTestsFromTestCase(TestSetupHandlerLoggers))
    suite.addTests(loader.loadTestsFromTestCase(TestSetupMigrationLoggers))
    suite.addTests(loader.loadTestsFromTestCase(TestSetupTransformLoggers))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateHandlerLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateMigrationLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateTransformLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestLogHandlerOperationDecorator))
    suite.addTests(loader.loadTestsFromTestCase(TestLogMigrationStepDecorator))
    suite.addTests(loader.loadTestsFromTestCase(TestLogTransformOperationDecorator))
    suite.addTests(loader.loadTestsFromTestCase(TestLogHandlerPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestLogTransformPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestLoggerRetrievalFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestLogExceptionWithContext))
    suite.addTests(loader.loadTestsFromTestCase(TestLogExperimentalSetupSwitch))
    suite.addTests(loader.loadTestsFromTestCase(TestLogTransformValidationResults))
    suite.addTests(loader.loadTestsFromTestCase(TestLogTransformCompositionSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigureDebugLoggingForHandlers))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigureDebugLoggingForTransforms))
    suite.addTests(loader.loadTestsFromTestCase(TestDisableVerboseThirdPartyLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateExperimentalSetupHash))
    suite.addTests(loader.loadTestsFromTestCase(TestLogExperimentalSetupSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestSetupBasicLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCasesAndBoundaryConditions))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    # Print test area coverage summary
    print("\nTEST AREA COVERAGE:")
    print("-" * 70)
    test_areas = [
        ("TestHandlerLoggerAdapter", "HandlerLoggerAdapter init & process"),
        ("TestMigrationLoggerAdapter", "MigrationLoggerAdapter init & process"),
        ("TestTransformLoggerAdapter", "TransformLoggerAdapter init & process"),
        ("TestSetupLogging", "setup_logging() configuration"),
        ("TestConfigureThirdPartyLoggers", "_configure_third_party_loggers()"),
        ("TestSetupHandlerLoggers", "_setup_handler_loggers()"),
        ("TestSetupMigrationLoggers", "_setup_migration_loggers()"),
        ("TestSetupTransformLoggers", "_setup_transform_loggers()"),
        ("TestCreateHandlerLogger", "create_handler_logger() factory"),
        ("TestCreateMigrationLogger", "create_migration_logger() factory"),
        ("TestCreateTransformLogger", "create_transform_logger() factory"),
        ("TestLogHandlerOperationDecorator", "log_handler_operation() decorator"),
        ("TestLogMigrationStepDecorator", "log_migration_step() decorator"),
        ("TestLogTransformOperationDecorator", "log_transform_operation() decorator"),
        ("TestLogHandlerPerformance", "log_handler_performance()"),
        ("TestLogTransformPerformance", "log_transform_performance()"),
        ("TestLoggerRetrievalFunctions", "Logger retrieval functions"),
        ("TestLogExceptionWithContext", "log_exception_with_context()"),
        ("TestLogExperimentalSetupSwitch", "log_experimental_setup_switch()"),
        ("TestLogTransformValidationResults", "log_transform_validation_results()"),
        ("TestLogTransformCompositionSummary", "log_transform_composition_summary()"),
        ("TestConfigureDebugLoggingForHandlers", "configure_debug_logging_for_handlers()"),
        ("TestConfigureDebugLoggingForTransforms", "configure_debug_logging_for_transforms()"),
        ("TestDisableVerboseThirdPartyLogging", "disable_verbose_third_party_logging()"),
        ("TestCreateExperimentalSetupHash", "create_experimental_setup_hash()"),
        ("TestLogExperimentalSetupSummary", "log_experimental_setup_summary()"),
        ("TestSetupBasicLogging", "setup_basic_logging() backward compat"),
        ("TestEdgeCasesAndBoundaryConditions", "Edge cases & boundary conditions"),
    ]
    print(f"Test Classes: {len(test_areas)}")
    for cls_name, description in test_areas:
        print(f"  \u2022 {cls_name}: {description}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
