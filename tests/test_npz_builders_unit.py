#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/preprocessing/utils/npz_builders.py

Module under test: npz_builders.py
- build_npz(): Build compressed .npz file from features and metadata dicts
- validate_npz_structure(): Validate .npz file structure and return summary

Test path on local machine: ~/ml_projects/milia/tests/test_npz_builders_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/preprocessing/utils/npz_builders.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.exceptions import DataProcessingError
from milia_pipeline.preprocessing.utils.npz_builders import (
    build_npz,
    validate_npz_structure,
)

# ============================================================================
# HELPERS: Mock data builders for molecular features
# ============================================================================


def _make_logger():
    """Create a logger instance for testing."""
    logger = logging.getLogger("test_npz_builders")
    logger.setLevel(logging.DEBUG)
    return logger


def _build_minimal_features(num_molecules=3):
    """
    Build a minimal valid features dict with the three required keys.

    Returns:
        Dictionary with 'compounds', 'atoms', 'coordinates' numpy arrays.
    """
    compounds = np.array([f"mol_{i}" for i in range(num_molecules)], dtype=object)
    atoms = np.array(
        [np.array([6, 1, 1, 1, 1], dtype=np.int64) for _ in range(num_molecules)],
        dtype=object,
    )
    coordinates = np.array(
        [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
            for _ in range(num_molecules)
        ],
        dtype=object,
    )
    return {
        "compounds": compounds,
        "atoms": atoms,
        "coordinates": coordinates,
    }


def _build_features_with_extras(num_molecules=3):
    """
    Build a features dict with the required keys plus extra feature arrays.

    Returns:
        Dictionary with required keys and additional 'energies' and 'charges' arrays.
    """
    features = _build_minimal_features(num_molecules)
    features["energies"] = np.array(
        [np.float64(-76.0 + i * 0.1) for i in range(num_molecules)],
        dtype=np.float64,
    )
    features["charges"] = np.array(
        [np.zeros(5, dtype=np.float64) for _ in range(num_molecules)],
        dtype=object,
    )
    return features


def _build_sample_metadata():
    """Build a simple metadata dictionary."""
    return {
        "version": "1.0",
        "source": "test_suite",
        "feature_tier": "standard",
    }


# ============================================================================
# GROUP 1: build_npz — Input Validation (10 tests)
# ============================================================================


class TestBuildNpzInputValidation(unittest.TestCase):
    """Test build_npz input validation: empty features, missing keys, shape mismatches."""

    def setUp(self):
        """Create a temporary directory for .npz output."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._output_path = Path(self._tmp_dir) / "test_output.npz"
        self._logger = _make_logger()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_empty_features_raises_data_processing_error(self):
        """Empty features dict raises DataProcessingError."""
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz({}, _build_sample_metadata(), self._output_path, logger=self._logger)
        self.assertIn("empty features", str(ctx.exception))

    def test_empty_features_error_operation_field(self):
        """Empty features error has operation='npz_creation'."""
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz({}, _build_sample_metadata(), self._output_path, logger=self._logger)
        self.assertEqual(ctx.exception.operation, "npz_creation")

    def test_missing_compounds_key_raises_error(self):
        """Missing 'compounds' key raises DataProcessingError."""
        features = _build_minimal_features()
        del features["compounds"]
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(features, _build_sample_metadata(), self._output_path, logger=self._logger)
        self.assertIn("compounds", str(ctx.exception))

    def test_missing_atoms_key_raises_error(self):
        """Missing 'atoms' key raises DataProcessingError."""
        features = _build_minimal_features()
        del features["atoms"]
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(features, _build_sample_metadata(), self._output_path, logger=self._logger)
        self.assertIn("atoms", str(ctx.exception))

    def test_missing_coordinates_key_raises_error(self):
        """Missing 'coordinates' key raises DataProcessingError."""
        features = _build_minimal_features()
        del features["coordinates"]
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(features, _build_sample_metadata(), self._output_path, logger=self._logger)
        self.assertIn("coordinates", str(ctx.exception))

    def test_missing_multiple_keys_reports_all(self):
        """Missing multiple required keys are all listed in the error."""
        features = {"compounds": np.array(["mol_0"], dtype=object)}
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(features, _build_sample_metadata(), self._output_path, logger=self._logger)
        error_msg = str(ctx.exception)
        self.assertIn("atoms", error_msg)
        self.assertIn("coordinates", error_msg)

    def test_missing_keys_error_includes_available_keys(self):
        """Error details include available keys when required keys are missing."""
        features = {"compounds": np.array(["mol_0"], dtype=object), "extra_key": np.array([1.0])}
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(features, _build_sample_metadata(), self._output_path, logger=self._logger)
        self.assertIsNotNone(ctx.exception.details)

    def test_shape_mismatch_atoms_raises_error(self):
        """Mismatched 'atoms' length relative to 'compounds' raises DataProcessingError."""
        features = _build_minimal_features(num_molecules=3)
        # Overwrite atoms with wrong length
        features["atoms"] = np.array([np.array([1, 1]) for _ in range(5)], dtype=object)
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(features, _build_sample_metadata(), self._output_path, logger=self._logger)
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_shape_mismatch_coordinates_raises_error(self):
        """Mismatched 'coordinates' length relative to 'compounds' raises DataProcessingError."""
        features = _build_minimal_features(num_molecules=3)
        features["coordinates"] = np.array(
            [np.array([[0.0, 0.0, 0.0]]) for _ in range(2)], dtype=object
        )
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(features, _build_sample_metadata(), self._output_path, logger=self._logger)
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_shape_mismatch_error_includes_counts(self):
        """Shape mismatch error message includes both actual and expected counts."""
        features = _build_minimal_features(num_molecules=3)
        features["atoms"] = np.array([np.array([1]) for _ in range(5)], dtype=object)
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(features, _build_sample_metadata(), self._output_path, logger=self._logger)
        error_msg = str(ctx.exception)
        self.assertIn("5", error_msg)
        self.assertIn("3", error_msg)


# ============================================================================
# GROUP 2: build_npz — Happy Path / File Creation (12 tests)
# ============================================================================


class TestBuildNpzHappyPath(unittest.TestCase):
    """Test build_npz successful creation of .npz files."""

    def setUp(self):
        """Create a temporary directory and standard test data."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._output_path = Path(self._tmp_dir) / "output.npz"
        self._logger = _make_logger()
        self._features = _build_minimal_features(num_molecules=3)
        self._metadata = _build_sample_metadata()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_creates_npz_file(self):
        """build_npz creates a .npz file at the specified path."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        self.assertTrue(self._output_path.exists())

    def test_created_file_is_nonzero_size(self):
        """Created .npz file has nonzero size."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        self.assertGreater(self._output_path.stat().st_size, 0)

    def test_npz_contains_compounds_key(self):
        """Created .npz contains 'compounds' key."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        self.assertIn("compounds", data.files)

    def test_npz_contains_atoms_key(self):
        """Created .npz contains 'atoms' key."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        self.assertIn("atoms", data.files)

    def test_npz_contains_coordinates_key(self):
        """Created .npz contains 'coordinates' key."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        self.assertIn("coordinates", data.files)

    def test_npz_contains_metadata_key(self):
        """Created .npz contains 'metadata' key."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        self.assertIn("metadata", data.files)

    def test_compounds_data_preserved(self):
        """Compound names are preserved in the .npz file."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        np.testing.assert_array_equal(data["compounds"], self._features["compounds"])

    def test_atoms_data_preserved(self):
        """Atoms arrays are preserved in the .npz file."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        loaded_atoms = data["atoms"]
        for i in range(len(self._features["atoms"])):
            np.testing.assert_array_equal(loaded_atoms[i], self._features["atoms"][i])

    def test_coordinates_data_preserved(self):
        """Coordinate arrays are preserved in the .npz file."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        loaded_coords = data["coordinates"]
        for i in range(len(self._features["coordinates"])):
            np.testing.assert_array_almost_equal(loaded_coords[i], self._features["coordinates"][i])

    def test_extra_features_preserved(self):
        """Extra feature arrays beyond the required three are preserved."""
        features = _build_features_with_extras(num_molecules=3)
        build_npz(features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        self.assertIn("energies", data.files)
        self.assertIn("charges", data.files)

    def test_creates_parent_directories(self):
        """build_npz creates parent directories if they don't exist."""
        nested_path = Path(self._tmp_dir) / "sub" / "deep" / "output.npz"
        build_npz(self._features, self._metadata, nested_path, logger=self._logger)
        self.assertTrue(nested_path.exists())

    def test_returns_none(self):
        """build_npz returns None on success (void function)."""
        result = build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        self.assertIsNone(result)


# ============================================================================
# GROUP 3: build_npz — Metadata Handling (10 tests)
# ============================================================================


class TestBuildNpzMetadata(unittest.TestCase):
    """Test build_npz metadata embedding and enhancement."""

    def setUp(self):
        """Create temporary directory and standard test data."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._output_path = Path(self._tmp_dir) / "output.npz"
        self._logger = _make_logger()
        self._features = _build_minimal_features(num_molecules=3)
        self._metadata = _build_sample_metadata()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _load_metadata(self):
        """Helper to load metadata from the created .npz file."""
        data = np.load(self._output_path, allow_pickle=True)
        return data["metadata"][0]

    def test_metadata_stored_as_numpy_array(self):
        """Metadata is stored as a numpy array in the .npz file."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        self.assertIsInstance(data["metadata"], np.ndarray)

    def test_metadata_array_has_single_element(self):
        """Metadata numpy array contains exactly one element (the dict)."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        data = np.load(self._output_path, allow_pickle=True)
        self.assertEqual(len(data["metadata"]), 1)

    def test_metadata_preserves_original_keys(self):
        """Original metadata keys are preserved in the .npz file."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        meta = self._load_metadata()
        self.assertEqual(meta["version"], "1.0")
        self.assertEqual(meta["source"], "test_suite")
        self.assertEqual(meta["feature_tier"], "standard")

    def test_metadata_enhanced_with_num_molecules(self):
        """Metadata is enhanced with 'num_molecules' count."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        meta = self._load_metadata()
        self.assertEqual(meta["num_molecules"], 3)

    def test_metadata_enhanced_with_feature_keys(self):
        """Metadata is enhanced with 'feature_keys' list."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        meta = self._load_metadata()
        self.assertIn("feature_keys", meta)
        self.assertIsInstance(meta["feature_keys"], list)

    def test_metadata_feature_keys_match_input(self):
        """Enhanced metadata 'feature_keys' matches the input feature dictionary keys."""
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        meta = self._load_metadata()
        self.assertEqual(sorted(meta["feature_keys"]), sorted(list(self._features.keys())))

    def test_metadata_num_molecules_matches_compounds_length(self):
        """Enhanced 'num_molecules' matches the length of 'compounds' array."""
        features = _build_minimal_features(num_molecules=7)
        build_npz(features, self._metadata, self._output_path, logger=self._logger)
        meta = self._load_metadata()
        self.assertEqual(meta["num_molecules"], 7)

    def test_original_metadata_not_mutated(self):
        """build_npz does not mutate the original metadata dictionary."""
        metadata_copy = self._metadata.copy()
        build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        self.assertEqual(self._metadata, metadata_copy)

    def test_empty_metadata_dict_accepted(self):
        """An empty metadata dict is accepted and enhanced with auto-fields."""
        build_npz(self._features, {}, self._output_path, logger=self._logger)
        meta = self._load_metadata()
        self.assertIn("num_molecules", meta)
        self.assertIn("feature_keys", meta)

    def test_metadata_with_extra_feature_arrays(self):
        """Metadata 'feature_keys' includes extra feature array names."""
        features = _build_features_with_extras(num_molecules=3)
        build_npz(features, self._metadata, self._output_path, logger=self._logger)
        meta = self._load_metadata()
        self.assertIn("energies", meta["feature_keys"])
        self.assertIn("charges", meta["feature_keys"])


# ============================================================================
# GROUP 4: build_npz — Logger Behavior (8 tests)
# ============================================================================


class TestBuildNpzLogging(unittest.TestCase):
    """Test build_npz logging behavior with custom and fallback loggers."""

    def setUp(self):
        """Create temporary directory and standard test data."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._output_path = Path(self._tmp_dir) / "output.npz"
        self._features = _build_minimal_features(num_molecules=2)
        self._metadata = _build_sample_metadata()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_custom_logger_used_when_provided(self):
        """When a custom logger is provided, it is used for logging."""
        mock_logger = MagicMock(spec=logging.Logger)
        build_npz(self._features, self._metadata, self._output_path, logger=mock_logger)
        self.assertTrue(mock_logger.info.called)

    def test_logs_molecule_count(self):
        """Logger.info is called with molecule count."""
        mock_logger = MagicMock(spec=logging.Logger)
        build_npz(self._features, self._metadata, self._output_path, logger=mock_logger)
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        joined = " ".join(info_messages)
        self.assertIn("2", joined)

    def test_logs_output_path(self):
        """Logger.info is called with the output path."""
        mock_logger = MagicMock(spec=logging.Logger)
        build_npz(self._features, self._metadata, self._output_path, logger=mock_logger)
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        joined = " ".join(info_messages)
        self.assertIn(str(self._output_path), joined)

    def test_logs_file_size(self):
        """Logger.info is called with file size information after creation."""
        mock_logger = MagicMock(spec=logging.Logger)
        build_npz(self._features, self._metadata, self._output_path, logger=mock_logger)
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        joined = " ".join(info_messages)
        self.assertIn("MB", joined)

    def test_logs_feature_count(self):
        """Logger.info is called with feature array count."""
        mock_logger = MagicMock(spec=logging.Logger)
        build_npz(self._features, self._metadata, self._output_path, logger=mock_logger)
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        joined = " ".join(info_messages)
        # 3 feature arrays (compounds, atoms, coordinates)
        self.assertIn("3", joined)

    def test_debug_logs_feature_summary(self):
        """Logger.debug is called with feature summary details."""
        mock_logger = MagicMock(spec=logging.Logger)
        build_npz(self._features, self._metadata, self._output_path, logger=mock_logger)
        self.assertTrue(mock_logger.debug.called)

    def test_fallback_logger_when_none(self):
        """When logger=None, the module-level logger is used (no crash)."""
        # Should not raise — falls back to module logger
        build_npz(self._features, self._metadata, self._output_path, logger=None)
        self.assertTrue(self._output_path.exists())

    def test_debug_logs_dtype_info(self):
        """Logger.debug includes dtype or length info for feature arrays."""
        mock_logger = MagicMock(spec=logging.Logger)
        build_npz(self._features, self._metadata, self._output_path, logger=mock_logger)
        debug_messages = [str(c) for c in mock_logger.debug.call_args_list]
        joined = " ".join(debug_messages)
        # At least one of 'dtype' or 'shape' or 'length' should appear
        self.assertTrue(
            "dtype" in joined or "shape" in joined or "length" in joined,
            f"Expected dtype/shape/length info in debug logs, got: {joined}",
        )


# ============================================================================
# GROUP 5: build_npz — Error Paths / Exception Wrapping (8 tests)
# ============================================================================


class TestBuildNpzErrorPaths(unittest.TestCase):
    """Test build_npz exception wrapping when np.savez_compressed fails."""

    def setUp(self):
        """Create temporary directory and standard test data."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._output_path = Path(self._tmp_dir) / "output.npz"
        self._logger = _make_logger()
        self._features = _build_minimal_features(num_molecules=2)
        self._metadata = _build_sample_metadata()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    @patch("milia_pipeline.preprocessing.utils.npz_builders.np.savez_compressed")
    def test_savez_failure_raises_data_processing_error(self, mock_savez):
        """If np.savez_compressed raises, DataProcessingError is raised."""
        mock_savez.side_effect = OSError("Disk full")
        with self.assertRaises(DataProcessingError):
            build_npz(self._features, self._metadata, self._output_path, logger=self._logger)

    @patch("milia_pipeline.preprocessing.utils.npz_builders.np.savez_compressed")
    def test_savez_failure_error_includes_file_path(self, mock_savez):
        """DataProcessingError from savez failure includes file_path."""
        mock_savez.side_effect = OSError("Disk full")
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        self.assertEqual(ctx.exception.file_path, str(self._output_path))

    @patch("milia_pipeline.preprocessing.utils.npz_builders.np.savez_compressed")
    def test_savez_failure_error_operation_field(self, mock_savez):
        """DataProcessingError from savez failure has operation='npz_creation'."""
        mock_savez.side_effect = OSError("Disk full")
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        self.assertEqual(ctx.exception.operation, "npz_creation")

    @patch("milia_pipeline.preprocessing.utils.npz_builders.np.savez_compressed")
    def test_savez_failure_preserves_cause(self, mock_savez):
        """DataProcessingError.__cause__ is the original exception."""
        original_error = OSError("Disk full")
        mock_savez.side_effect = original_error
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        self.assertIs(ctx.exception.__cause__, original_error)

    @patch("milia_pipeline.preprocessing.utils.npz_builders.np.savez_compressed")
    def test_savez_failure_error_message_includes_cause(self, mock_savez):
        """DataProcessingError message includes the original error message."""
        mock_savez.side_effect = ValueError("Invalid array")
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        self.assertIn("Invalid array", str(ctx.exception))

    @patch("milia_pipeline.preprocessing.utils.npz_builders.np.savez_compressed")
    def test_file_not_created_after_savez_raises_error(self, mock_savez):
        """If savez_compressed succeeds but file doesn't exist, DataProcessingError raised."""
        # savez_compressed returns normally but file is not actually created
        mock_savez.return_value = None
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(self._features, self._metadata, self._output_path, logger=self._logger)
        self.assertIn("not created", str(ctx.exception).lower())

    @patch("milia_pipeline.preprocessing.utils.npz_builders.np.savez_compressed")
    def test_readonly_directory_raises_error(self, mock_savez):
        """Writing to a read-only location raises DataProcessingError."""
        # Simulate savez raising PermissionError due to read-only filesystem
        mock_savez.side_effect = PermissionError("Read-only file system")
        with self.assertRaises(DataProcessingError):
            build_npz(self._features, self._metadata, self._output_path, logger=self._logger)

    @patch("milia_pipeline.preprocessing.utils.npz_builders.np.savez_compressed")
    def test_permission_error_wrapped(self, mock_savez):
        """PermissionError from savez is wrapped in DataProcessingError."""
        mock_savez.side_effect = PermissionError("Permission denied")
        with self.assertRaises(DataProcessingError):
            build_npz(self._features, self._metadata, self._output_path, logger=self._logger)


# ============================================================================
# GROUP 6: build_npz — Edge Cases (8 tests)
# ============================================================================


class TestBuildNpzEdgeCases(unittest.TestCase):
    """Test build_npz with edge case inputs: single molecule, large data, etc."""

    def setUp(self):
        """Create temporary directory."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._logger = _make_logger()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_single_molecule(self):
        """build_npz handles a single-molecule features dict."""
        output_path = Path(self._tmp_dir) / "single.npz"
        features = _build_minimal_features(num_molecules=1)
        build_npz(features, _build_sample_metadata(), output_path, logger=self._logger)
        data = np.load(output_path, allow_pickle=True)
        self.assertEqual(len(data["compounds"]), 1)

    def test_large_molecule_count(self):
        """build_npz handles a larger number of molecules."""
        output_path = Path(self._tmp_dir) / "large.npz"
        features = _build_minimal_features(num_molecules=100)
        build_npz(features, _build_sample_metadata(), output_path, logger=self._logger)
        data = np.load(output_path, allow_pickle=True)
        self.assertEqual(len(data["compounds"]), 100)

    def test_extra_non_required_keys_ignored_in_shape_check(self):
        """Extra keys not in required set are not subject to shape validation."""
        output_path = Path(self._tmp_dir) / "extra.npz"
        features = _build_minimal_features(num_molecules=3)
        # Add an extra key with different length — should NOT raise
        features["global_property"] = np.array([42.0])
        build_npz(features, _build_sample_metadata(), output_path, logger=self._logger)
        data = np.load(output_path, allow_pickle=True)
        self.assertIn("global_property", data.files)

    def test_overwrite_existing_file(self):
        """build_npz overwrites an existing .npz file without error."""
        output_path = Path(self._tmp_dir) / "overwrite.npz"
        features = _build_minimal_features(num_molecules=2)
        build_npz(features, _build_sample_metadata(), output_path, logger=self._logger)
        _size_first = output_path.stat().st_size

        features_v2 = _build_minimal_features(num_molecules=5)
        build_npz(features_v2, _build_sample_metadata(), output_path, logger=self._logger)
        data = np.load(output_path, allow_pickle=True)
        self.assertEqual(len(data["compounds"]), 5)

    def test_metadata_with_nested_structures(self):
        """Metadata with nested dicts/lists is preserved via object array."""
        output_path = Path(self._tmp_dir) / "nested_meta.npz"
        metadata = {
            "version": "2.0",
            "nested": {"key1": "value1", "key2": [1, 2, 3]},
        }
        features = _build_minimal_features(num_molecules=2)
        build_npz(features, metadata, output_path, logger=self._logger)
        data = np.load(output_path, allow_pickle=True)
        loaded_meta = data["metadata"][0]
        self.assertEqual(loaded_meta["nested"]["key1"], "value1")

    def test_features_dict_not_mutated(self):
        """build_npz does not mutate the original features dictionary."""
        features = _build_minimal_features(num_molecules=2)
        original_keys = set(features.keys())
        output_path = Path(self._tmp_dir) / "no_mutate.npz"
        build_npz(features, _build_sample_metadata(), output_path, logger=self._logger)
        # The function adds 'metadata' to npz_data (a copy), not to the original
        self.assertEqual(set(features.keys()), original_keys)

    def test_path_without_npz_extension_raises_error(self):
        """Path without .npz extension causes DataProcessingError because
        numpy saves to <path>.npz but the module verifies the original path."""
        output_path = Path(self._tmp_dir) / "no_ext"
        features = _build_minimal_features(num_molecules=2)
        with self.assertRaises(DataProcessingError) as ctx:
            build_npz(features, _build_sample_metadata(), output_path, logger=self._logger)
        self.assertIn("not created", str(ctx.exception).lower())

    def test_compounds_with_special_characters(self):
        """Compound names with special characters are preserved."""
        output_path = Path(self._tmp_dir) / "special_chars.npz"
        features = _build_minimal_features(num_molecules=2)
        features["compounds"] = np.array(["mol/1:α", "mol#2—β"], dtype=object)
        build_npz(features, _build_sample_metadata(), output_path, logger=self._logger)
        data = np.load(output_path, allow_pickle=True)
        np.testing.assert_array_equal(data["compounds"], features["compounds"])


# ============================================================================
# GROUP 7: validate_npz_structure — Happy Path (10 tests)
# ============================================================================


class TestValidateNpzHappyPath(unittest.TestCase):
    """Test validate_npz_structure with valid .npz files."""

    def setUp(self):
        """Create a temporary directory and a valid .npz file for validation."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._npz_path = Path(self._tmp_dir) / "valid.npz"
        self._logger = _make_logger()

        # Create a valid .npz file via build_npz
        features = _build_minimal_features(num_molecules=4)
        metadata = _build_sample_metadata()
        build_npz(features, metadata, self._npz_path, logger=self._logger)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_returns_dict(self):
        """validate_npz_structure returns a dictionary."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        self.assertIsInstance(result, dict)

    def test_valid_field_is_true(self):
        """Result contains 'valid': True for a valid .npz file."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        self.assertTrue(result["valid"])

    def test_path_field_matches(self):
        """Result 'path' matches the input path string."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        self.assertEqual(result["path"], str(self._npz_path))

    def test_file_size_mb_positive(self):
        """Result 'file_size_mb' is a positive float."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        self.assertIsInstance(result["file_size_mb"], float)
        self.assertGreater(result["file_size_mb"], 0.0)

    def test_num_molecules_correct(self):
        """Result 'num_molecules' matches the number of compounds in the file."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        self.assertEqual(result["num_molecules"], 4)

    def test_num_features_positive(self):
        """Result 'num_features' is a positive integer."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        self.assertIsInstance(result["num_features"], int)
        self.assertGreater(result["num_features"], 0)

    def test_feature_keys_is_list(self):
        """Result 'feature_keys' is a list of strings."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        self.assertIsInstance(result["feature_keys"], list)
        for key in result["feature_keys"]:
            self.assertIsInstance(key, str)

    def test_feature_keys_contains_required(self):
        """Result 'feature_keys' includes the required keys plus metadata."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        for key in ["compounds", "atoms", "coordinates", "metadata"]:
            self.assertIn(key, result["feature_keys"])

    def test_metadata_is_dict(self):
        """Result 'metadata' is a dictionary."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        self.assertIsInstance(result["metadata"], dict)

    def test_metadata_contains_enhanced_fields(self):
        """Result 'metadata' includes the enhanced fields from build_npz."""
        result = validate_npz_structure(self._npz_path, logger=self._logger)
        self.assertIn("num_molecules", result["metadata"])
        self.assertIn("feature_keys", result["metadata"])


# ============================================================================
# GROUP 8: validate_npz_structure — Error Paths (10 tests)
# ============================================================================


class TestValidateNpzErrorPaths(unittest.TestCase):
    """Test validate_npz_structure error handling."""

    def setUp(self):
        """Create a temporary directory."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._logger = _make_logger()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_nonexistent_file_raises_error(self):
        """Nonexistent .npz path raises DataProcessingError."""
        fake_path = Path(self._tmp_dir) / "nonexistent.npz"
        with self.assertRaises(DataProcessingError):
            validate_npz_structure(fake_path, logger=self._logger)

    def test_nonexistent_file_error_operation(self):
        """Error for nonexistent file has operation='npz_validation'."""
        fake_path = Path(self._tmp_dir) / "nonexistent.npz"
        with self.assertRaises(DataProcessingError) as ctx:
            validate_npz_structure(fake_path, logger=self._logger)
        self.assertEqual(ctx.exception.operation, "npz_validation")

    def test_nonexistent_file_error_includes_path(self):
        """Error for nonexistent file includes the file path."""
        fake_path = Path(self._tmp_dir) / "nonexistent.npz"
        with self.assertRaises(DataProcessingError) as ctx:
            validate_npz_structure(fake_path, logger=self._logger)
        self.assertEqual(ctx.exception.file_path, str(fake_path))

    def test_missing_compounds_key_raises_error(self):
        """NPZ file missing 'compounds' key raises DataProcessingError."""
        npz_path = Path(self._tmp_dir) / "no_compounds.npz"
        np.savez_compressed(
            npz_path,
            metadata=np.array([{"version": "1.0"}], dtype=object),
            atoms=np.array([np.array([1, 1])], dtype=object),
        )
        with self.assertRaises(DataProcessingError) as ctx:
            validate_npz_structure(npz_path, logger=self._logger)
        self.assertIn("compounds", str(ctx.exception))

    def test_missing_metadata_key_raises_error(self):
        """NPZ file missing 'metadata' key raises DataProcessingError."""
        npz_path = Path(self._tmp_dir) / "no_metadata.npz"
        np.savez_compressed(
            npz_path,
            compounds=np.array(["mol_0"], dtype=object),
        )
        with self.assertRaises(DataProcessingError) as ctx:
            validate_npz_structure(npz_path, logger=self._logger)
        self.assertIn("metadata", str(ctx.exception))

    def test_missing_both_required_keys(self):
        """NPZ file missing both required validation keys raises error listing both."""
        npz_path = Path(self._tmp_dir) / "empty_npz.npz"
        np.savez_compressed(npz_path, dummy=np.array([1]))
        with self.assertRaises(DataProcessingError) as ctx:
            validate_npz_structure(npz_path, logger=self._logger)
        error_msg = str(ctx.exception)
        self.assertIn("compounds", error_msg)
        self.assertIn("metadata", error_msg)

    def test_corrupted_file_raises_error(self):
        """A corrupted (non-npz) file raises DataProcessingError."""
        corrupted_path = Path(self._tmp_dir) / "corrupted.npz"
        corrupted_path.write_text("This is not a valid npz file")
        with self.assertRaises(DataProcessingError):
            validate_npz_structure(corrupted_path, logger=self._logger)

    def test_corrupted_file_error_operation(self):
        """Corrupted file error has operation='npz_validation'."""
        corrupted_path = Path(self._tmp_dir) / "corrupted.npz"
        corrupted_path.write_text("Not npz")
        with self.assertRaises(DataProcessingError) as ctx:
            validate_npz_structure(corrupted_path, logger=self._logger)
        self.assertEqual(ctx.exception.operation, "npz_validation")

    def test_validation_error_preserves_cause(self):
        """DataProcessingError from validation wraps the original exception."""
        corrupted_path = Path(self._tmp_dir) / "corrupted2.npz"
        corrupted_path.write_bytes(b"\x00\x01\x02\x03")
        with self.assertRaises(DataProcessingError) as ctx:
            validate_npz_structure(corrupted_path, logger=self._logger)
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_fallback_logger_when_none(self):
        """validate_npz_structure uses module logger when logger=None without crashing."""
        npz_path = Path(self._tmp_dir) / "valid.npz"
        features = _build_minimal_features(num_molecules=2)
        build_npz(features, _build_sample_metadata(), npz_path, logger=self._logger)
        # Should not raise with logger=None
        result = validate_npz_structure(npz_path, logger=None)
        self.assertTrue(result["valid"])


# ============================================================================
# GROUP 9: validate_npz_structure — Logging (6 tests)
# ============================================================================


class TestValidateNpzLogging(unittest.TestCase):
    """Test validate_npz_structure logging behavior."""

    def setUp(self):
        """Create a temporary directory and a valid .npz file."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._npz_path = Path(self._tmp_dir) / "valid.npz"
        self._logger = _make_logger()
        features = _build_minimal_features(num_molecules=5)
        build_npz(features, _build_sample_metadata(), self._npz_path, logger=self._logger)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_logs_validation_passed(self):
        """Logger.info is called indicating validation passed."""
        mock_logger = MagicMock(spec=logging.Logger)
        validate_npz_structure(self._npz_path, logger=mock_logger)
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        joined = " ".join(info_messages)
        self.assertIn("Validation passed", joined)

    def test_logs_file_name(self):
        """Logger.info includes the file name."""
        mock_logger = MagicMock(spec=logging.Logger)
        validate_npz_structure(self._npz_path, logger=mock_logger)
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        joined = " ".join(info_messages)
        self.assertIn(self._npz_path.name, joined)

    def test_logs_molecule_count(self):
        """Logger.info includes the molecule count."""
        mock_logger = MagicMock(spec=logging.Logger)
        validate_npz_structure(self._npz_path, logger=mock_logger)
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        joined = " ".join(info_messages)
        self.assertIn("5", joined)

    def test_logs_feature_count(self):
        """Logger.info includes the feature count."""
        mock_logger = MagicMock(spec=logging.Logger)
        validate_npz_structure(self._npz_path, logger=mock_logger)
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        joined = " ".join(info_messages)
        self.assertIn("Features", joined)

    def test_logs_file_size(self):
        """Logger.info includes the file size in MB."""
        mock_logger = MagicMock(spec=logging.Logger)
        validate_npz_structure(self._npz_path, logger=mock_logger)
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        joined = " ".join(info_messages)
        self.assertIn("MB", joined)

    def test_custom_logger_receives_all_info_calls(self):
        """Custom logger receives at least 4 info calls (passed, molecules, features, size)."""
        mock_logger = MagicMock(spec=logging.Logger)
        validate_npz_structure(self._npz_path, logger=mock_logger)
        self.assertGreaterEqual(mock_logger.info.call_count, 4)


# ============================================================================
# GROUP 10: Integration — build_npz + validate_npz_structure (8 tests)
# ============================================================================


class TestNpzBuildersIntegration(unittest.TestCase):
    """Integration tests: build_npz followed by validate_npz_structure."""

    def setUp(self):
        """Create a temporary directory."""
        self._tmp_dir = tempfile.mkdtemp(prefix="test_npz_builders_")
        self._logger = _make_logger()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_build_then_validate_roundtrip(self):
        """A file created by build_npz passes validate_npz_structure."""
        npz_path = Path(self._tmp_dir) / "roundtrip.npz"
        features = _build_minimal_features(num_molecules=3)
        metadata = _build_sample_metadata()
        build_npz(features, metadata, npz_path, logger=self._logger)

        result = validate_npz_structure(npz_path, logger=self._logger)
        self.assertTrue(result["valid"])

    def test_roundtrip_molecule_count_consistency(self):
        """Validated molecule count matches the number built."""
        npz_path = Path(self._tmp_dir) / "consistency.npz"
        num_mols = 7
        features = _build_minimal_features(num_molecules=num_mols)
        build_npz(features, _build_sample_metadata(), npz_path, logger=self._logger)

        result = validate_npz_structure(npz_path, logger=self._logger)
        self.assertEqual(result["num_molecules"], num_mols)

    def test_roundtrip_feature_keys_include_required_plus_metadata(self):
        """Validated feature_keys include original feature keys plus 'metadata'."""
        npz_path = Path(self._tmp_dir) / "keys_check.npz"
        features = _build_minimal_features(num_molecules=2)
        build_npz(features, _build_sample_metadata(), npz_path, logger=self._logger)

        result = validate_npz_structure(npz_path, logger=self._logger)
        for key in ["compounds", "atoms", "coordinates", "metadata"]:
            self.assertIn(key, result["feature_keys"])

    def test_roundtrip_with_extra_features(self):
        """Extra features survive the build→validate roundtrip."""
        npz_path = Path(self._tmp_dir) / "extras.npz"
        features = _build_features_with_extras(num_molecules=3)
        build_npz(features, _build_sample_metadata(), npz_path, logger=self._logger)

        result = validate_npz_structure(npz_path, logger=self._logger)
        self.assertIn("energies", result["feature_keys"])
        self.assertIn("charges", result["feature_keys"])

    def test_roundtrip_metadata_preserved(self):
        """Original metadata fields are accessible after build→validate roundtrip."""
        npz_path = Path(self._tmp_dir) / "meta_rt.npz"
        metadata = {"version": "2.5", "author": "test"}
        features = _build_minimal_features(num_molecules=2)
        build_npz(features, metadata, npz_path, logger=self._logger)

        result = validate_npz_structure(npz_path, logger=self._logger)
        self.assertEqual(result["metadata"]["version"], "2.5")
        self.assertEqual(result["metadata"]["author"], "test")

    def test_roundtrip_num_features_count(self):
        """Validated num_features equals the number of stored arrays."""
        npz_path = Path(self._tmp_dir) / "count.npz"
        features = _build_features_with_extras(num_molecules=3)
        build_npz(features, _build_sample_metadata(), npz_path, logger=self._logger)

        result = validate_npz_structure(npz_path, logger=self._logger)
        # features dict has 5 keys + 1 metadata key added by build_npz = 6
        expected_count = len(features) + 1  # +1 for 'metadata'
        self.assertEqual(result["num_features"], expected_count)

    def test_multiple_builds_validate_independently(self):
        """Multiple .npz files can be built and validated independently."""
        for i in range(3):
            npz_path = Path(self._tmp_dir) / f"multi_{i}.npz"
            features = _build_minimal_features(num_molecules=i + 1)
            build_npz(features, _build_sample_metadata(), npz_path, logger=self._logger)

            result = validate_npz_structure(npz_path, logger=self._logger)
            self.assertTrue(result["valid"])
            self.assertEqual(result["num_molecules"], i + 1)

    def test_roundtrip_file_size_positive(self):
        """Validated file_size_mb is a positive number after roundtrip."""
        npz_path = Path(self._tmp_dir) / "size_check.npz"
        features = _build_minimal_features(num_molecules=10)
        build_npz(features, _build_sample_metadata(), npz_path, logger=self._logger)

        result = validate_npz_structure(npz_path, logger=self._logger)
        self.assertGreater(result["file_size_mb"], 0.0)


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestBuildNpzInputValidation,  # GROUP  1: 10 tests
        TestBuildNpzHappyPath,  # GROUP  2: 12 tests
        TestBuildNpzMetadata,  # GROUP  3: 10 tests
        TestBuildNpzLogging,  # GROUP  4:  8 tests
        TestBuildNpzErrorPaths,  # GROUP  5:  8 tests
        TestBuildNpzEdgeCases,  # GROUP  6:  8 tests
        TestValidateNpzHappyPath,  # GROUP  7: 10 tests
        TestValidateNpzErrorPaths,  # GROUP  8: 10 tests
        TestValidateNpzLogging,  # GROUP  9:  6 tests
        TestNpzBuildersIntegration,  # GROUP 10:  8 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — npz_builders.py")
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
TEST SUITE SUMMARY — milia_pipeline/preprocessing/utils/npz_builders.py
============================================================================

90 comprehensive production-ready tests covering:

GROUP 1: build_npz — Input Validation (10 tests)
- Empty features dict raises DataProcessingError
- Empty features error has operation='npz_creation'
- Missing 'compounds' key raises error
- Missing 'atoms' key raises error
- Missing 'coordinates' key raises error
- Missing multiple keys reports all in error
- Missing keys error details include available keys
- Shape mismatch on 'atoms' raises error
- Shape mismatch on 'coordinates' raises error
- Shape mismatch error includes actual and expected counts

GROUP 2: build_npz — Happy Path / File Creation (12 tests)
- Creates .npz file at specified path
- Created file has nonzero size
- NPZ contains 'compounds' key
- NPZ contains 'atoms' key
- NPZ contains 'coordinates' key
- NPZ contains 'metadata' key
- Compound names preserved
- Atoms arrays preserved
- Coordinate arrays preserved
- Extra feature arrays preserved
- Parent directories created automatically
- Returns None on success

GROUP 3: build_npz — Metadata Handling (10 tests)
- Metadata stored as numpy array
- Metadata array has single element
- Original metadata keys preserved
- Enhanced with num_molecules
- Enhanced with feature_keys list
- Feature keys match input dict keys
- num_molecules matches compounds length
- Original metadata dict not mutated
- Empty metadata dict accepted
- Extra features included in feature_keys

GROUP 4: build_npz — Logger Behavior (8 tests)
- Custom logger used when provided
- Logs molecule count
- Logs output path
- Logs file size in MB
- Logs feature array count
- Debug logs feature summary
- Fallback to module logger when None
- Debug logs include dtype/shape/length info

GROUP 5: build_npz — Error Paths / Exception Wrapping (8 tests)
- savez failure raises DataProcessingError
- savez failure error includes file_path
- savez failure error has operation='npz_creation'
- savez failure preserves __cause__
- savez failure error message includes cause
- File not created after savez raises error
- Read-only filesystem raises error (mocked savez)
- PermissionError wrapped in DataProcessingError

GROUP 6: build_npz — Edge Cases (8 tests)
- Single molecule features
- Large molecule count (100)
- Extra non-required keys skip shape validation
- Overwrite existing file
- Nested metadata structures preserved
- Features dict not mutated
- Path without .npz extension raises DataProcessingError
- Compound names with special characters preserved

GROUP 7: validate_npz_structure — Happy Path (10 tests)
- Returns dict
- valid=True for valid file
- path field matches input
- file_size_mb is positive float
- num_molecules correct
- num_features is positive int
- feature_keys is list of strings
- feature_keys contains required keys
- metadata is dict
- metadata contains enhanced fields

GROUP 8: validate_npz_structure — Error Paths (10 tests)
- Nonexistent file raises error
- Nonexistent file error operation='npz_validation'
- Nonexistent file error includes path
- Missing 'compounds' key raises error
- Missing 'metadata' key raises error
- Missing both required keys raises error
- Corrupted file raises error
- Corrupted file error operation='npz_validation'
- Validation error preserves __cause__
- Fallback logger when None

GROUP 9: validate_npz_structure — Logging (6 tests)
- Logs validation passed
- Logs file name
- Logs molecule count
- Logs feature count
- Logs file size in MB
- Custom logger receives at least 4 info calls

GROUP 10: Integration — build_npz + validate_npz_structure (8 tests)
- Build then validate roundtrip
- Molecule count consistency
- Feature keys include required plus metadata
- Extra features survive roundtrip
- Metadata preserved in roundtrip
- num_features count correct
- Multiple files built and validated independently
- file_size_mb positive after roundtrip

Total: 90 comprehensive production-ready tests

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All mocking via @patch decorators or context managers (test-level only)
- Temporary directory cleanup in tearDown
- Comprehensive error path coverage
- Exception message content verification
- Exception attribute verification (operation, file_path, __cause__)
- Numpy dtype and shape preservation testing
- Real .npz file I/O for integration tests
- Mock logger for logging behavior verification
- Edge case coverage (single molecule, large count, special characters)
- Metadata immutability checks
- Features dict immutability checks
- Compatible with both pytest and unittest runner
- No downloaded files — all data created in-memory via helpers
"""
