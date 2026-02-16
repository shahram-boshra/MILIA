#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/preprocessing/preprocessors/ani2x.py

Module under test: ani2x.py
- iter_data_buckets_ani2x: Module-level generator yielding conformer dicts from HDF5
  - Supports 'species' (ANI-2x convention) or 'atomic_numbers' (ANI-1x convention) keys
  - Default keys: ['energies']
  - Filters out conformers with NaN values in floating-point properties
  - Filters out zero-padding atoms (Z=0)
  - Handles both 1D and 2D atomic_numbers arrays
  - Per-atom properties filtered by non_zero_mask when shapes match
  - Explicit dtype enforcement: energies->float64, forces->float32
- ANI2X_SUPPORTED_ELEMENTS: {1, 6, 7, 8, 9, 16, 17} -- H, C, N, O, F, S, Cl
- ANI2xPreprocessor: Preprocessor for ANI-2x quantum chemistry dataset (HDF5 format)
  - Inherits BasePreprocessor ABC (2 abstract methods: _validate_config, preprocess)
  - Registered via @PreprocessorRegistry.register("ANI2x")
  - CRITICAL: BasePreprocessor.__init__() calls self._validate_config() during construction
  - Pipeline: _get_h5_path -> _parse_ani2x_h5 -> _build_npz (3-step)
  - Config keys: raw_archive_path, output_npz_path, num_molecules, property_keys
  - Supports: .h5, .hdf5, .tar.gz, .tgz file extensions (warning on unrecognized extension)
  - _get_h5_path: Returns Path as-is for .h5/.hdf5, extracts from tar.gz otherwise
  - Auto-skip if output .npz already exists
  - Wraps all errors in DataProcessingError (operation="ani2x_preprocessing")
  - Private methods: _get_h5_path, _parse_ani2x_h5, _build_npz
  - _parse_ani2x_h5 validates elements against ANI2X_SUPPORTED_ELEMENTS
  - DEFAULT_PROPERTY_KEYS = ['energies', 'forces']

Test path on local machine: ~/ml_projects/milia/tests/test_preprocessor_ani2x_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/preprocessing/preprocessor/ani2x.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution
- iter_data_buckets_ani2x h5py mocking uses patch.dict('sys.modules', ...) at test level
  to intercept the LOCAL ``import h5py`` inside the function body (ani2x.py line 84).
  patch.dict restores original sys.modules state on context manager exit.

NPZ file paths (mocked, never downloaded):
- ~/Chem_Data/MILIA_PyG_Dataset/raw/ANI-2x-wB97X-631Gd.tar.gz

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import contextlib

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.preprocessors.ani2x import (
    ANI2X_SUPPORTED_ELEMENTS,
    ANI2xPreprocessor,
    iter_data_buckets_ani2x,
)
from milia_pipeline.preprocessing.registry import PreprocessorRegistry

# ============================================================================
# HELPERS: Build realistic config and mock objects
# ============================================================================


def _make_config(**overrides):
    """
    Build a minimal config dict for ANI2xPreprocessor tests.

    Based on ANI2xPreprocessor._validate_config requirements:
    - Required: 'raw_archive_path', 'output_npz_path'
    - Optional: 'num_molecules', 'property_keys'
    """
    config = {
        "raw_archive_path": overrides.get(
            "raw_archive_path", "/tmp/test_data/raw/ANI-2x-wB97X-631Gd.h5"
        ),
        "output_npz_path": overrides.get("output_npz_path", "/tmp/test_data/processed/ani2x.npz"),
    }
    for key in ["num_molecules", "property_keys"]:
        if key in overrides:
            config[key] = overrides[key]
    for key in list(config.keys()):
        if overrides.get(f"_remove_{key}", False):
            del config[key]
    return config


def _make_logger():
    """Build a test logger."""
    return logging.getLogger("test.preprocessor.ani2x")


def _make_preprocessor(config=None, logger=None):
    """
    Build an ANI2xPreprocessor instance with configurable mocks.

    CRITICAL: BasePreprocessor.__init__() calls self._validate_config() during
    construction. Therefore Path.exists MUST be patched BEFORE calling this
    helper (for valid configs), or the constructor will raise ConfigurationError.
    """
    if config is None:
        config = _make_config()
    if logger is None:
        logger = _make_logger()
    return ANI2xPreprocessor(config=config, logger=logger)


def _path_exists_factory(archive_path_str, output_path_str):
    """
    Create a Path.exists side_effect that controls which paths 'exist'.

    Returns True for archive_path (so __init__ validation passes),
    False for output_path (so preprocess doesn't skip).
    """
    archive_p = Path(archive_path_str)
    output_p = Path(output_path_str)

    def exists_side_effect(self_path):
        if self_path == archive_p:
            return True
        if self_path == output_p:
            return False
        return False

    return exists_side_effect


def _make_mock_features_and_metadata():
    """
    Build realistic mock features and metadata dicts as returned by _parse_ani2x_h5.
    """
    atoms_arr = np.empty(2, dtype=object)
    atoms_arr[0] = np.array([6, 1, 1, 1, 1], dtype=np.uint8)
    atoms_arr[1] = np.array([8, 1, 1], dtype=np.uint8)

    coords_arr = np.empty(2, dtype=object)
    coords_arr[0] = np.random.randn(5, 3).astype(np.float32)
    coords_arr[1] = np.random.randn(3, 3).astype(np.float32)

    mol_id_arr = np.empty(2, dtype=object)
    mol_id_arr[0] = "C1H4"
    mol_id_arr[1] = "H2O1"

    features = {
        "atoms": atoms_arr,
        "coordinates": coords_arr,
        "energy": np.array([-40.5, -76.4], dtype=np.float64),
        "molecule_id": mol_id_arr,
    }

    metadata = {
        "total_conformers": 2,
        "skipped_nan": 0,
        "skipped_unknown_element": 0,
        "mean_atoms": 4.0,
        "max_atoms": 5,
        "min_atoms": 3,
        "properties_extracted": ["atoms", "coordinates", "energy", "molecule_id"],
    }

    return features, metadata


def _create_and_run_pipeline(config, mock_parse, mock_build, parse_return=None):
    """
    Helper: create preprocessor with proper Path.exists handling and run preprocess.

    _get_h5_path is NOT mocked here because for .h5/.hdf5 archive paths it
    simply returns the path as-is (ani2x.py line 364). Only non-.h5 archives
    trigger tar extraction.
    """
    mock_parse.return_value = parse_return or _make_mock_features_and_metadata()

    exists_fn = _path_exists_factory(config["raw_archive_path"], config["output_npz_path"])

    with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
        preprocessor = _make_preprocessor(config=config)
        result = preprocessor.preprocess()

    return preprocessor, result


def _make_h5_mock_for_iter(mol_groups):
    """
    Build a mock h5py module + File context manager from mol_groups.

    Shared helper for iter_data_buckets_ani2x tests.

    Args:
        mol_groups: Dict of group_name -> dict of dataset arrays.

    Returns:
        mock_h5py_module with .File() returning a context manager.
    """
    mock_file = MagicMock()
    mock_file.keys.return_value = list(mol_groups.keys())

    group_mocks = {}
    for group_name, datasets in mol_groups.items():
        mock_group = MagicMock()
        mock_group.__contains__ = lambda s, key, ds=datasets: key in ds
        mock_group.__getitem__ = lambda s, key, ds=datasets: ds[key]
        group_mocks[group_name] = mock_group

    mock_file.__getitem__ = lambda s, name, gm=group_mocks: gm[name]
    mock_file.__enter__ = Mock(return_value=mock_file)
    mock_file.__exit__ = Mock(return_value=False)

    mock_h5py = MagicMock()
    mock_h5py.File.return_value = mock_file
    return mock_h5py


# ============================================================================
# GROUP 1: ANI2xPreprocessor - Identity and Registration (6 tests)
# ============================================================================


class TestANI2xPreprocessorIdentity(unittest.TestCase):
    """Test ANI2xPreprocessor identity, registration, and basic attributes."""

    def test_is_subclass_of_base_preprocessor(self):
        """ANI2xPreprocessor is a proper BasePreprocessor subclass."""
        self.assertTrue(issubclass(ANI2xPreprocessor, BasePreprocessor))

    def test_registered_in_preprocessor_registry(self):
        """ANI2xPreprocessor is registered as 'ANI2x' in PreprocessorRegistry."""
        self.assertTrue(PreprocessorRegistry.supports_preprocessing("ANI2x"))

    def test_registry_returns_correct_class(self):
        """PreprocessorRegistry.get_preprocessor('ANI2x') returns ANI2xPreprocessor."""
        cls = PreprocessorRegistry.get_preprocessor("ANI2x")
        self.assertIs(cls, ANI2xPreprocessor)

    @patch("pathlib.Path.exists", return_value=True)
    def test_preprocessor_stores_config(self, mock_exists):
        """Preprocessor stores config dict passed during construction."""
        config = _make_config()
        preprocessor = _make_preprocessor(config=config)
        self.assertIs(preprocessor.config, config)

    @patch("pathlib.Path.exists", return_value=True)
    def test_preprocessor_stores_logger(self, mock_exists):
        """Preprocessor stores logger passed during construction."""
        logger = _make_logger()
        preprocessor = ANI2xPreprocessor(config=_make_config(), logger=logger)
        self.assertIs(preprocessor.logger, logger)

    def test_ani2x_in_list_preprocessors(self):
        """'ANI2x' appears in PreprocessorRegistry.list_preprocessors()."""
        available = PreprocessorRegistry.list_preprocessors()
        self.assertIn("ANI2x", available)


# ============================================================================
# GROUP 2: ANI2X_SUPPORTED_ELEMENTS Module Constant (4 tests)
# ============================================================================


class TestANI2xSupportedElements(unittest.TestCase):
    """Test the ANI2X_SUPPORTED_ELEMENTS module-level constant."""

    def test_supported_elements_is_set(self):
        """ANI2X_SUPPORTED_ELEMENTS is a set."""
        self.assertIsInstance(ANI2X_SUPPORTED_ELEMENTS, set)

    def test_supported_elements_count(self):
        """ANI2X_SUPPORTED_ELEMENTS contains exactly 7 elements.

        Evidence: ani2x.py line 59: {1, 6, 7, 8, 9, 16, 17}
        """
        self.assertEqual(len(ANI2X_SUPPORTED_ELEMENTS), 7)

    def test_supported_elements_values(self):
        """ANI2X_SUPPORTED_ELEMENTS contains H(1), C(6), N(7), O(8), F(9), S(16), Cl(17).

        Evidence: ani2x.py line 59.
        """
        expected = {1, 6, 7, 8, 9, 16, 17}
        self.assertEqual(ANI2X_SUPPORTED_ELEMENTS, expected)

    def test_extended_elements_beyond_ani1x(self):
        """ANI2x extends ANI-1x elements (H,C,N,O) with S(16), F(9), Cl(17).

        Evidence: ani2x.py lines 31, 59.
        """
        ani1x_elements = {1, 6, 7, 8}
        self.assertTrue(ani1x_elements.issubset(ANI2X_SUPPORTED_ELEMENTS))
        extensions = ANI2X_SUPPORTED_ELEMENTS - ani1x_elements
        self.assertEqual(extensions, {9, 16, 17})


# ============================================================================
# GROUP 3: _validate_config - Success Paths (4 tests)
# ============================================================================


class TestValidateConfigSuccess(unittest.TestCase):
    """Test _validate_config success paths for valid configuration."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_minimal_valid_config(self, mock_exists):
        """Minimal config with only required keys passes validation."""
        _make_preprocessor(config=_make_config())

    @patch("pathlib.Path.exists", return_value=True)
    def test_valid_config_with_num_molecules(self, mock_exists):
        """Config with valid num_molecules passes validation."""
        _make_preprocessor(config=_make_config(num_molecules=100))

    @patch("pathlib.Path.exists", return_value=True)
    def test_valid_config_with_property_keys(self, mock_exists):
        """Config with explicit property_keys passes validation."""
        _make_preprocessor(config=_make_config(property_keys=["energies", "forces"]))

    @patch("pathlib.Path.exists", return_value=True)
    def test_valid_config_with_all_optional_keys(self, mock_exists):
        """Config with all optional keys passes validation."""
        _make_preprocessor(
            config=_make_config(num_molecules=500, property_keys=["energies", "forces"])
        )


# ============================================================================
# GROUP 4: _validate_config - Missing Required Keys (4 tests)
# ============================================================================


class TestValidateConfigMissingKeys(unittest.TestCase):
    """Test _validate_config error paths for missing required configuration keys."""

    def test_missing_raw_archive_path_raises(self):
        """Missing 'raw_archive_path' raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(config=_make_config(_remove_raw_archive_path=True))
        self.assertIn("raw_archive_path", str(ctx.exception))

    def test_missing_output_npz_path_raises(self):
        """Missing 'output_npz_path' raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(config=_make_config(_remove_output_npz_path=True))
        self.assertIn("output_npz_path", str(ctx.exception))

    def test_missing_both_required_keys_raises(self):
        """Missing both required keys raises ConfigurationError listing both."""
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(config={})
        error_msg = str(ctx.exception)
        self.assertIn("raw_archive_path", error_msg)
        self.assertIn("output_npz_path", error_msg)

    def test_missing_key_error_is_configuration_error(self):
        """ConfigurationError is the correct exception type for missing keys."""
        with self.assertRaises(ConfigurationError):
            _make_preprocessor(config={})


# ============================================================================
# GROUP 5: _validate_config - Path Validation (3 tests)
# ============================================================================


class TestValidateConfigPathValidation(unittest.TestCase):
    """Test _validate_config error paths for invalid file paths."""

    @patch("pathlib.Path.exists", return_value=False)
    def test_nonexistent_archive_path_raises(self, mock_exists):
        """Non-existent raw_archive_path raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(config=_make_config())
        self.assertIn("not found", str(ctx.exception).lower())

    @patch("pathlib.Path.exists", return_value=False)
    def test_nonexistent_archive_path_mentions_path(self, mock_exists):
        """Error for non-existent path mentions the actual path."""
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(config=_make_config())
        self.assertIn("ANI-2x-wB97X-631Gd.h5", str(ctx.exception))

    @patch("pathlib.Path.exists", return_value=False)
    def test_nonexistent_path_error_type(self, mock_exists):
        """Path validation error is ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            _make_preprocessor(config=_make_config())


# ============================================================================
# GROUP 6: _validate_config - File Extension Validation (6 tests)
# ============================================================================


class TestValidateConfigExtension(unittest.TestCase):
    """Test _validate_config behavior for various file extensions.

    ANI-2x supports .h5, .hdf5, .tar.gz, .tgz extensions.
    Evidence: ani2x.py line 242: valid_extensions = ('.h5', '.hdf5', '.tar.gz', '.tgz')
    """

    @patch("pathlib.Path.exists", return_value=True)
    def test_h5_extension_accepted(self, mock_exists):
        """Archive with .h5 extension passes validation without warning."""
        _make_preprocessor(config=_make_config(raw_archive_path="/tmp/data/ANI-2x-wB97X-631Gd.h5"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_hdf5_extension_accepted(self, mock_exists):
        """Archive with .hdf5 extension passes validation without warning."""
        _make_preprocessor(
            config=_make_config(raw_archive_path="/tmp/data/ANI-2x-wB97X-631Gd.hdf5")
        )

    @patch("pathlib.Path.exists", return_value=True)
    def test_tar_gz_extension_accepted(self, mock_exists):
        """Archive with .tar.gz extension passes validation without warning.

        Evidence: ani2x.py line 242.
        """
        _make_preprocessor(
            config=_make_config(raw_archive_path="/tmp/data/ANI-2x-wB97X-631Gd.tar.gz")
        )

    @patch("pathlib.Path.exists", return_value=True)
    def test_tgz_extension_accepted(self, mock_exists):
        """Archive with .tgz extension passes validation without warning.

        Evidence: ani2x.py line 242.
        """
        _make_preprocessor(config=_make_config(raw_archive_path="/tmp/data/ANI-2x-wB97X-631Gd.tgz"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_uppercase_h5_extension_accepted(self, mock_exists):
        """Archive with .H5 extension (case insensitive) passes validation.

        Evidence: ani2x.py line 243: str(archive_path).lower().endswith(ext)
        """
        _make_preprocessor(config=_make_config(raw_archive_path="/tmp/data/ANI-2x-wB97X-631Gd.H5"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_unrecognized_extension_logs_warning(self, mock_exists):
        """Unrecognized file extension logs warning but does not raise.

        Evidence: ani2x.py lines 243-247 -- warns but proceeds.
        """
        logger = _make_logger()
        with patch.object(logger, "warning") as mock_warn:
            _make_preprocessor(
                config=_make_config(raw_archive_path="/tmp/data/ani2x.dat"), logger=logger
            )
            mock_warn.assert_called_once()
            self.assertIn("not recognized", mock_warn.call_args[0][0].lower())


# ============================================================================
# GROUP 7: _validate_config - num_molecules Validation (5 tests)
# ============================================================================


class TestValidateConfigNumMolecules(unittest.TestCase):
    """Test _validate_config error paths for invalid num_molecules values."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_zero_num_molecules_raises(self, mock_exists):
        """num_molecules=0 raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            _make_preprocessor(config=_make_config(num_molecules=0))

    @patch("pathlib.Path.exists", return_value=True)
    def test_negative_num_molecules_raises(self, mock_exists):
        """Negative num_molecules raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            _make_preprocessor(config=_make_config(num_molecules=-5))

    @patch("pathlib.Path.exists", return_value=True)
    def test_float_num_molecules_raises(self, mock_exists):
        """Float num_molecules raises ConfigurationError (must be int)."""
        with self.assertRaises(ConfigurationError):
            _make_preprocessor(config=_make_config(num_molecules=10.5))

    @patch("pathlib.Path.exists", return_value=True)
    def test_string_num_molecules_raises(self, mock_exists):
        """String num_molecules raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            _make_preprocessor(config=_make_config(num_molecules="100"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_none_num_molecules_is_valid(self, mock_exists):
        """num_molecules=None is valid (extract all conformers)."""
        _make_preprocessor(config=_make_config(num_molecules=None))


# ============================================================================
# GROUP 8: preprocess - Output Already Exists (Early Return) (3 tests)
# ============================================================================


class TestPreprocessOutputExists(unittest.TestCase):
    """Test preprocess() early return when output .npz already exists."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_existing_output_returns_path_without_processing(self, mock_exists):
        """When output .npz exists, returns path immediately without processing."""
        config = _make_config()
        preprocessor = _make_preprocessor(config=config)

        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = Mock(st_size=1024 * 1024 * 50)
            result = preprocessor.preprocess()

        self.assertEqual(result, Path(config["output_npz_path"]))

    @patch("pathlib.Path.exists", return_value=True)
    def test_existing_output_skips_h5_parsing(self, mock_exists):
        """When output .npz exists, HDF5 parsing is never called."""
        preprocessor = _make_preprocessor(config=_make_config())
        with patch.object(Path, "stat", return_value=Mock(st_size=1024)):
            with patch.object(preprocessor, "_parse_ani2x_h5") as mock_parse:
                preprocessor.preprocess()
        mock_parse.assert_not_called()

    @patch("pathlib.Path.exists", return_value=True)
    def test_existing_output_skips_npz_build(self, mock_exists):
        """When output .npz exists, NPZ building is never called."""
        preprocessor = _make_preprocessor(config=_make_config())
        with patch.object(Path, "stat", return_value=Mock(st_size=1024)):
            with patch.object(preprocessor, "_build_npz") as mock_build:
                preprocessor.preprocess()
        mock_build.assert_not_called()


# ============================================================================
# GROUP 9: preprocess - Full Pipeline Success (5 tests)
# ============================================================================


class TestPreprocessFullPipeline(unittest.TestCase):
    """Test preprocess() full pipeline execution with mocked dependencies."""

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_full_pipeline_returns_output_path(self, mock_parse, mock_build):
        """Full pipeline returns the configured output_npz_path."""
        config = _make_config()
        _, result = _create_and_run_pipeline(config, mock_parse, mock_build)
        self.assertEqual(result, Path(config["output_npz_path"]))

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_parse_called_with_h5_path(self, mock_parse, mock_build):
        """Step 2: _parse_ani2x_h5 called with correct HDF5 path.

        Evidence: ani2x.py line 289 -- h5_path comes from _get_h5_path.
        For .h5 files, _get_h5_path returns as-is (line 364).
        """
        config = _make_config()
        _create_and_run_pipeline(config, mock_parse, mock_build)
        mock_parse.assert_called_once()
        call_kwargs = mock_parse.call_args.kwargs
        self.assertEqual(call_kwargs.get("h5_path"), Path(config["raw_archive_path"]))

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_parse_called_with_property_keys(self, mock_parse, mock_build):
        """Step 2: _parse_ani2x_h5 called with correct property_keys."""
        config = _make_config()
        _create_and_run_pipeline(config, mock_parse, mock_build)
        call_kwargs = mock_parse.call_args.kwargs
        self.assertEqual(call_kwargs.get("property_keys"), ANI2xPreprocessor.DEFAULT_PROPERTY_KEYS)

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_parse_called_with_max_conformers(self, mock_parse, mock_build):
        """Step 2: num_molecules config passed as max_conformers."""
        config = _make_config(num_molecules=200)
        _create_and_run_pipeline(config, mock_parse, mock_build)
        call_kwargs = mock_parse.call_args.kwargs
        self.assertEqual(call_kwargs.get("max_conformers"), 200)

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_build_npz_called_with_features_and_metadata(self, mock_parse, mock_build):
        """Step 3: _build_npz called with features from parse and comprehensive metadata."""
        features, parse_metadata = _make_mock_features_and_metadata()
        _create_and_run_pipeline(
            _make_config(), mock_parse, mock_build, parse_return=(features, parse_metadata)
        )

        mock_build.assert_called_once()
        kw = mock_build.call_args.kwargs
        self.assertIs(kw.get("features"), features)
        metadata = kw.get("metadata")
        self.assertEqual(metadata.get("version"), "1.0")
        self.assertEqual(metadata.get("dataset_name"), "ANI2x")


# ============================================================================
# GROUP 10: preprocess - Error Wrapping (5 tests)
# ============================================================================


class TestPreprocessErrorWrapping(unittest.TestCase):
    """Test preprocess() wraps all exceptions in DataProcessingError."""

    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_parse_error_wrapped(self, mock_parse):
        """Parsing RuntimeError wrapped in DataProcessingError."""
        config = _make_config()
        mock_parse.side_effect = RuntimeError("HDF5 corrupt")

        exists_fn = _path_exists_factory(config["raw_archive_path"], config["output_npz_path"])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError) as ctx:
                preprocessor.preprocess()
        self.assertIn("ANI-2x preprocessing failed", str(ctx.exception))

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_build_error_wrapped(self, mock_parse, mock_build):
        """_build_npz error wrapped in DataProcessingError."""
        config = _make_config()
        mock_parse.return_value = _make_mock_features_and_metadata()
        mock_build.side_effect = OSError("Disk full")

        exists_fn = _path_exists_factory(config["raw_archive_path"], config["output_npz_path"])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError):
                preprocessor.preprocess()

    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_wrapped_error_preserves_cause(self, mock_parse):
        """DataProcessingError preserves original exception as __cause__."""
        config = _make_config()
        original_error = RuntimeError("Original error")
        mock_parse.side_effect = original_error

        exists_fn = _path_exists_factory(config["raw_archive_path"], config["output_npz_path"])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError) as ctx:
                preprocessor.preprocess()
        self.assertIs(ctx.exception.__cause__, original_error)

    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_wrapped_error_mentions_ani2x(self, mock_parse):
        """DataProcessingError message includes ANI-2x context."""
        config = _make_config()
        mock_parse.side_effect = RuntimeError("fail")

        exists_fn = _path_exists_factory(config["raw_archive_path"], config["output_npz_path"])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError) as ctx:
                preprocessor.preprocess()
        self.assertIn("ANI-2x", str(ctx.exception))

    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_value_error_wrapped_as_data_processing_error(self, mock_parse):
        """ValueError (e.g., bad data shape) also wrapped in DataProcessingError."""
        config = _make_config()
        mock_parse.side_effect = ValueError("Invalid shape")

        exists_fn = _path_exists_factory(config["raw_archive_path"], config["output_npz_path"])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError):
                preprocessor.preprocess()


# ============================================================================
# GROUP 11: preprocess - Metadata Construction (9 tests)
# ============================================================================


class TestPreprocessMetadata(unittest.TestCase):
    """Test preprocess() constructs correct metadata for NPZ.

    Evidence: ani2x.py lines 318-333.
    """

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_includes_version(self, mock_parse, mock_build):
        """NPZ metadata includes version='1.0'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(mock_build.call_args.kwargs["metadata"]["version"], "1.0")

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_includes_dataset_name(self, mock_parse, mock_build):
        """NPZ metadata includes dataset_name='ANI2x'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(mock_build.call_args.kwargs["metadata"]["dataset_name"], "ANI2x")

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_includes_parser(self, mock_parse, mock_build):
        """NPZ metadata includes parser='ANI2xPreprocessor'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(mock_build.call_args.kwargs["metadata"]["parser"], "ANI2xPreprocessor")

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_includes_source_url(self, mock_parse, mock_build):
        """NPZ metadata includes Zenodo source URL.

        Evidence: ani2x.py line 323.
        """
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs["metadata"]
        self.assertEqual(
            metadata["source_url"],
            "https://zenodo.org/records/10108942/files/ANI-2x-wB97X-631Gd.tar.gz",
        )

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_includes_coordinate_and_energy_units(self, mock_parse, mock_build):
        """NPZ metadata includes coordinate_units='angstrom' and energy_units='hartree'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs["metadata"]
        self.assertEqual(metadata["coordinate_units"], "angstrom")
        self.assertEqual(metadata["energy_units"], "hartree")

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_includes_force_units(self, mock_parse, mock_build):
        """NPZ metadata includes force_units='hartree/angstrom'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs["metadata"]
        self.assertEqual(metadata["force_units"], "hartree/angstrom")

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_includes_doi_and_reference(self, mock_parse, mock_build):
        """NPZ metadata includes DOI and reference to Devereux et al."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs["metadata"]
        self.assertEqual(metadata["doi"], "10.1021/acs.jctc.0c00121")
        self.assertIn("Devereux", metadata["reference"])

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_merges_parse_metadata(self, mock_parse, mock_build):
        """NPZ metadata merges parse_metadata from _parse_ani2x_h5."""
        pm = {
            "total_conformers": 42,
            "skipped_nan": 3,
            "skipped_unknown_element": 1,
            "mean_atoms": 8.5,
            "max_atoms": 15,
            "min_atoms": 3,
            "properties_extracted": ["atoms", "coordinates", "energy"],
        }
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build, parse_return=({}, pm))
        metadata = mock_build.call_args.kwargs["metadata"]
        self.assertEqual(metadata["total_conformers"], 42)
        self.assertEqual(metadata["skipped_nan"], 3)
        self.assertEqual(metadata["skipped_unknown_element"], 1)

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_supported_elements(self, mock_parse, mock_build):
        """NPZ metadata includes supported_elements string."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs["metadata"]
        self.assertEqual(metadata["supported_elements"], "H, C, N, O, S, F, Cl")


# ============================================================================
# GROUP 12: preprocess - Default Values (5 tests)
# ============================================================================


class TestPreprocessDefaults(unittest.TestCase):
    """Test preprocess() uses correct defaults for optional config keys."""

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_default_num_molecules_is_none(self, mock_parse, mock_build):
        """Default num_molecules is None (extract all conformers)."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertIsNone(mock_parse.call_args.kwargs.get("max_conformers"))

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_default_property_keys(self, mock_parse, mock_build):
        """Default property_keys uses ANI2xPreprocessor.DEFAULT_PROPERTY_KEYS."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(
            mock_parse.call_args.kwargs.get("property_keys"),
            ANI2xPreprocessor.DEFAULT_PROPERTY_KEYS,
        )

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_custom_property_keys_passed_through(self, mock_parse, mock_build):
        """Custom property_keys config is passed to _parse_ani2x_h5."""
        custom_keys = ["energies", "forces"]
        config = _make_config(property_keys=custom_keys)
        _create_and_run_pipeline(config, mock_parse, mock_build)
        self.assertEqual(mock_parse.call_args.kwargs.get("property_keys"), custom_keys)

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_source_from_archive_name(self, mock_parse, mock_build):
        """Metadata 'source' field uses archive file name."""
        _create_and_run_pipeline(
            _make_config(raw_archive_path="/data/raw/ANI-2x-wB97X-631Gd.h5"), mock_parse, mock_build
        )
        self.assertEqual(mock_build.call_args.kwargs["metadata"]["source"], "ANI-2x-wB97X-631Gd.h5")

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_default_num_molecules_none_passed_to_parser(self, mock_parse, mock_build):
        """Default num_molecules=None is passed to _parse_ani2x_h5 as max_conformers."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertIsNone(mock_parse.call_args.kwargs.get("max_conformers"))


# ============================================================================
# GROUP 13: preprocess - Pipeline Step Ordering (2 tests)
# ============================================================================


class TestPreprocessStepOrdering(unittest.TestCase):
    """Test preprocess() executes pipeline steps in correct order."""

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_steps_execute_in_order(self, mock_parse, mock_build):
        """Steps execute in order: parse_h5 -> build_npz."""
        config = _make_config()
        call_order = []

        def track_parse(**kw):
            call_order.append("parse")
            return _make_mock_features_and_metadata()

        def track_build(**kw):
            call_order.append("build")

        mock_parse.side_effect = track_parse
        mock_build.side_effect = track_build

        exists_fn = _path_exists_factory(config["raw_archive_path"], config["output_npz_path"])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            preprocessor.preprocess()

        self.assertEqual(call_order, ["parse", "build"])

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_build_receives_parse_output(self, mock_parse, mock_build):
        """Step 3 receives features and metadata from Step 2."""
        expected_features = {"atoms": np.array([6, 1], dtype=np.uint8)}
        expected_meta = {"total_conformers": 2}
        _create_and_run_pipeline(
            _make_config(), mock_parse, mock_build, parse_return=(expected_features, expected_meta)
        )
        self.assertIs(mock_build.call_args.kwargs.get("features"), expected_features)


# ============================================================================
# GROUP 14: BasePreprocessor Integration - run() Method (5 tests)
# ============================================================================


class TestBasePreprocessorRunIntegration(unittest.TestCase):
    """Test ANI2xPreprocessor works with BasePreprocessor.run() method."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_run_calls_preprocess_and_validate_output(self, mock_exists):
        """run() calls preprocess() then _validate_output() in sequence."""
        preprocessor = _make_preprocessor(config=_make_config())
        call_order = []

        def mock_preprocess():
            call_order.append("preprocess")
            return Path("/tmp/test_output.npz")

        def mock_validate_output(path):
            call_order.append("validate_output")

        with patch.object(preprocessor, "preprocess", side_effect=mock_preprocess):
            with patch.object(preprocessor, "_validate_output", side_effect=mock_validate_output):
                preprocessor.run()

        self.assertEqual(call_order, ["preprocess", "validate_output"])

    def test_run_raises_on_invalid_config(self):
        """Construction raises ConfigurationError when config is invalid."""
        with self.assertRaises(ConfigurationError):
            ANI2xPreprocessor(config={}, logger=_make_logger())

    @patch("pathlib.Path.exists", return_value=True)
    def test_run_calls_preprocess(self, mock_exists):
        """run() calls preprocess after validation."""
        preprocessor = _make_preprocessor(config=_make_config())
        with patch.object(Path, "stat", return_value=Mock(st_size=1024)):
            with patch.object(preprocessor, "preprocess", wraps=preprocessor.preprocess) as mock_pp:
                with contextlib.suppress(Exception):
                    preprocessor.run()
                mock_pp.assert_called_once()

    def test_has_run_method_from_base(self):
        """ANI2xPreprocessor inherits run() from BasePreprocessor."""
        self.assertTrue(hasattr(ANI2xPreprocessor, "run"))

    def test_has_validate_output_from_base(self):
        """ANI2xPreprocessor inherits _validate_output() from BasePreprocessor."""
        self.assertTrue(hasattr(ANI2xPreprocessor, "_validate_output"))


# ============================================================================
# GROUP 15: iter_data_buckets_ani2x - Module-Level Generator (12 tests)
# ============================================================================


class TestIterDataBucketsAni2x(unittest.TestCase):
    """Test iter_data_buckets_ani2x() module-level generator function.

    CRITICAL MOCKING NOTE:
    iter_data_buckets_ani2x() uses a LOCAL import: ``import h5py`` inside the
    function body (ani2x.py line 84). This means:
    - @patch("module.h5py", create=True) does NOT work
    - The correct approach: use ``patch.dict('sys.modules', {'h5py': mock_h5py})``
    - This is safe from mock pollution because patch.dict restores original state.
    """

    def test_yields_conformer_dicts_with_species_key(self):
        """iter_data_buckets_ani2x yields dict with expected keys when using 'species'.

        Evidence: ani2x.py lines 95-96 -- tries 'species' first (ANI-2x convention).
        """
        mock_h5py = _make_h5_mock_for_iter(
            {
                "C1H4": {
                    "species": np.array([[6, 1, 1, 1, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 5, 3).astype(np.float32),
                    "energies": np.array([-40.518], dtype=np.float64),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5", keys=["energies"]))

        self.assertEqual(len(results), 1)
        self.assertIn("atomic_numbers", results[0])
        self.assertIn("coordinates", results[0])
        self.assertIn("molecule_id", results[0])
        self.assertIn("energies", results[0])

    def test_falls_back_to_atomic_numbers_key(self):
        """iter_data_buckets_ani2x falls back to 'atomic_numbers' if 'species' absent.

        Evidence: ani2x.py lines 97-98.
        """
        mock_h5py = _make_h5_mock_for_iter(
            {
                "C1H4": {
                    "atomic_numbers": np.array([[6, 1, 1, 1, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 5, 3).astype(np.float32),
                    "energies": np.array([-40.518], dtype=np.float64),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5"))

        self.assertEqual(len(results), 1)
        np.testing.assert_array_equal(
            results[0]["atomic_numbers"], np.array([6, 1, 1, 1, 1], dtype=np.uint8)
        )

    def test_skips_groups_without_species_or_atomic_numbers(self):
        """iter_data_buckets_ani2x skips groups without species/atomic_numbers key.

        Evidence: ani2x.py lines 99-101.
        """
        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_valid": {
                    "species": np.array([[6, 1, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 3, 3).astype(np.float32),
                    "energies": np.array([-40.5], dtype=np.float64),
                },
                "mol_missing": {
                    "coordinates": np.random.randn(1, 3, 3).astype(np.float32),
                    "energies": np.array([-76.0], dtype=np.float64),
                },
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["molecule_id"], "mol_valid")

    def test_default_keys_is_energies(self):
        """iter_data_buckets_ani2x default keys is ['energies'].

        Evidence: ani2x.py lines 86-87.
        """
        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_a": {
                    "species": np.array([[6, 1, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 3, 3).astype(np.float32),
                    "energies": np.array([-40.518], dtype=np.float64),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5"))

        self.assertEqual(len(results), 1)
        self.assertIn("energies", results[0])

    def test_filters_nan_conformers(self):
        """iter_data_buckets_ani2x skips conformers with NaN energy values."""
        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_a": {
                    "species": np.array([[6, 1, 1, 1]] * 3, dtype=np.uint8),
                    "coordinates": np.random.randn(3, 4, 3).astype(np.float32),
                    "energies": np.array([-40.5, np.nan, -38.2], dtype=np.float64),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5"))

        self.assertEqual(len(results), 2)

    def test_filters_zero_padding_atoms(self):
        """iter_data_buckets_ani2x filters out padding atoms (Z=0).

        Evidence: ani2x.py lines 149-152.
        """
        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_padded": {
                    "species": np.array([[6, 1, 1, 0, 0]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 5, 3).astype(np.float32),
                    "energies": np.array([-40.5], dtype=np.float64),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5"))

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]["atomic_numbers"]), 3)
        self.assertEqual(results[0]["coordinates"].shape[0], 3)

    def test_molecule_id_set_from_group_name(self):
        """iter_data_buckets_ani2x sets molecule_id from HDF5 group name."""
        mock_h5py = _make_h5_mock_for_iter(
            {
                "my_molecule_group": {
                    "species": np.array([[6, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 2, 3).astype(np.float32),
                    "energies": np.array([-10.0], dtype=np.float64),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5"))

        self.assertEqual(results[0]["molecule_id"], "my_molecule_group")

    def test_handles_1d_atomic_numbers(self):
        """iter_data_buckets_ani2x handles 1D atomic_numbers (no Nc dim).

        Evidence: ani2x.py lines 144-147.
        """
        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_1d": {
                    "species": np.array([6, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(2, 3, 3).astype(np.float32),
                    "energies": np.array([-40.5, -40.6], dtype=np.float64),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5"))

        self.assertEqual(len(results), 2)
        np.testing.assert_array_equal(
            results[0]["atomic_numbers"], np.array([6, 1, 1], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            results[1]["atomic_numbers"], np.array([6, 1, 1], dtype=np.uint8)
        )

    def test_multiple_molecule_groups(self):
        """iter_data_buckets_ani2x iterates over all molecular groups."""
        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_A": {
                    "species": np.array([[6, 1, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 3, 3).astype(np.float32),
                    "energies": np.array([-10.0], dtype=np.float64),
                },
                "mol_B": {
                    "species": np.array([[8, 1, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 3, 3).astype(np.float32),
                    "energies": np.array([-20.0], dtype=np.float64),
                },
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5"))

        self.assertEqual(len(results), 2)
        mol_ids = {r["molecule_id"] for r in results}
        self.assertEqual(mol_ids, {"mol_A", "mol_B"})

    def test_per_atom_property_filtered_by_non_zero_mask(self):
        """Per-atom properties (forces) are filtered by same non_zero_mask.

        Evidence: ani2x.py lines 166-167.
        """
        mock_h5py = _make_h5_mock_for_iter(
            {
                "padded_mol": {
                    "species": np.array([[6, 1, 0]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 3, 3).astype(np.float32),
                    "energies": np.array([-40.5], dtype=np.float64),
                    "forces": np.array(
                        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]], dtype=np.float32
                    ),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5", keys=["energies", "forces"]))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["forces"].shape[0], 2)

    def test_nan_in_multidimensional_forces(self):
        """iter_data_buckets_ani2x detects NaN in multi-dimensional property arrays.

        Evidence: ani2x.py lines 134-135.
        """
        forces_data = np.random.randn(2, 3, 3).astype(np.float32)
        forces_data[1, 0, 0] = np.nan

        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_a": {
                    "species": np.array([[6, 1, 1]] * 2, dtype=np.uint8),
                    "coordinates": np.random.randn(2, 3, 3).astype(np.float32),
                    "energies": np.array([-40.5, -38.2], dtype=np.float64),
                    "forces": forces_data,
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5", keys=["energies", "forces"]))

        self.assertEqual(len(results), 1)

    def test_dtype_enforcement_forces_to_float32(self):
        """Forces are explicitly cast to float32.

        Evidence: ani2x.py line 116: 'forces': np.float32.
        """
        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_a": {
                    "species": np.array([[6, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 2, 3).astype(np.float32),
                    "energies": np.array([-40.5], dtype=np.float64),
                    "forces": np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float64),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(iter_data_buckets_ani2x("/fake/path.h5", keys=["energies", "forces"]))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["forces"].dtype, np.float32)


# ============================================================================
# GROUP 16: _parse_ani2x_h5 - Internal Method Logic (7 tests)
# ============================================================================


class TestParseAni2xH5(unittest.TestCase):
    """Test _parse_ani2x_h5 internal method for HDF5 parsing logic."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_parse_returns_features_and_metadata(self, mock_iter, mock_exists):
        """_parse_ani2x_h5 returns (features_dict, metadata_dict) tuple."""
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(3, 3).astype(np.float32),
                    "molecule_id": "mol_A",
                    "energies": np.float64(-40.5),
                }
            ]
        )
        preprocessor = _make_preprocessor()
        features, metadata = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=None
        )
        self.assertIn("atoms", features)
        self.assertIn("coordinates", features)
        self.assertIn("energy", features)
        self.assertIn("molecule_id", features)
        self.assertIn("total_conformers", metadata)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_parse_respects_max_conformers(self, mock_iter, mock_exists):
        """_parse_ani2x_h5 stops after max_conformers."""
        conformers = []
        for i in range(5):
            conformers.append(
                {
                    "atomic_numbers": np.array([6, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(2, 3).astype(np.float32),
                    "molecule_id": f"mol_{i}",
                    "energies": np.float64(-40.0 + i),
                }
            )
        mock_iter.return_value = iter(conformers)
        preprocessor = _make_preprocessor()
        features, metadata = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=2
        )
        self.assertEqual(metadata["total_conformers"], 2)
        self.assertEqual(len(features["energy"]), 2)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_parse_preserves_uint8_atom_dtype(self, mock_iter, mock_exists):
        """_parse_ani2x_h5 preserves uint8 dtype for atomic numbers."""
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(3, 3).astype(np.float32),
                    "molecule_id": "mol_A",
                    "energies": np.float64(-40.5),
                }
            ]
        )
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=None
        )
        self.assertEqual(features["atoms"][0].dtype, np.uint8)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_parse_stores_optional_forces(self, mock_iter, mock_exists):
        """_parse_ani2x_h5 stores forces when present in conformer data."""
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(2, 3).astype(np.float32),
                    "molecule_id": "mol_A",
                    "energies": np.float64(-40.5),
                    "forces": np.random.randn(2, 3).astype(np.float32),
                }
            ]
        )
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies", "forces"], max_conformers=None
        )
        self.assertIn("forces", features)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_parse_metadata_atom_statistics(self, mock_iter, mock_exists):
        """_parse_ani2x_h5 computes correct atom statistics in metadata."""
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(3, 3).astype(np.float32),
                    "molecule_id": "mol_A",
                    "energies": np.float64(-40.5),
                },
                {
                    "atomic_numbers": np.array([8, 1, 1, 1, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(6, 3).astype(np.float32),
                    "molecule_id": "mol_B",
                    "energies": np.float64(-76.4),
                },
            ]
        )
        preprocessor = _make_preprocessor()
        _, metadata = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=None
        )
        self.assertEqual(metadata["min_atoms"], 3)
        self.assertEqual(metadata["max_atoms"], 6)
        self.assertAlmostEqual(metadata["mean_atoms"], 4.5)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_parse_skips_unsupported_elements(self, mock_iter, mock_exists):
        """_parse_ani2x_h5 skips conformers with elements not in ANI2X_SUPPORTED_ELEMENTS.

        Element 35 (Bromine) is NOT in {1, 6, 7, 8, 9, 16, 17}.
        """
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(3, 3).astype(np.float32),
                    "molecule_id": "mol_valid",
                    "energies": np.float64(-40.5),
                },
                {
                    "atomic_numbers": np.array([35, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(3, 3).astype(np.float32),
                    "molecule_id": "mol_unsupported",
                    "energies": np.float64(-50.0),
                },
            ]
        )
        preprocessor = _make_preprocessor()
        features, metadata = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=None
        )
        self.assertEqual(metadata["total_conformers"], 1)
        self.assertEqual(metadata["skipped_unknown_element"], 1)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_parse_energy_stored_as_float64(self, mock_iter, mock_exists):
        """Energy values stored as float64 in features['energy']."""
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(2, 3).astype(np.float32),
                    "molecule_id": "mol_A",
                    "energies": np.float64(-40.5),
                }
            ]
        )
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=None
        )
        self.assertEqual(features["energy"].dtype, np.float64)


# ============================================================================
# GROUP 17: _build_npz - Internal Method Logic (4 tests)
# ============================================================================


class TestBuildNpz(unittest.TestCase):
    """Test _build_npz internal method for NPZ file construction."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_build_npz_creates_file(self, mock_exists):
        """_build_npz creates a compressed NPZ file at the specified path."""
        preprocessor = _make_preprocessor()
        features, metadata = _make_mock_features_and_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.npz"
            preprocessor._build_npz(features=features, metadata=metadata, output_path=output_path)
            self.assertTrue(output_path.exists())

    @patch("pathlib.Path.exists", return_value=True)
    def test_build_npz_includes_metadata_key(self, mock_exists):
        """_build_npz includes _metadata key in saved NPZ."""
        preprocessor = _make_preprocessor()
        features, metadata = _make_mock_features_and_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.npz"
            preprocessor._build_npz(features=features, metadata=metadata, output_path=output_path)
            loaded = np.load(str(output_path), allow_pickle=True)
            self.assertIn("_metadata", loaded.files)

    @patch("pathlib.Path.exists", return_value=True)
    def test_build_npz_creates_parent_directory(self, mock_exists):
        """_build_npz creates parent directories if they don't exist."""
        preprocessor = _make_preprocessor()
        features, metadata = _make_mock_features_and_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "deep" / "test_output.npz"
            preprocessor._build_npz(features=features, metadata=metadata, output_path=output_path)
            self.assertTrue(output_path.exists())

    @patch("pathlib.Path.exists", return_value=True)
    def test_build_npz_preserves_feature_keys(self, mock_exists):
        """_build_npz preserves all feature keys in the output NPZ."""
        preprocessor = _make_preprocessor()
        features, metadata = _make_mock_features_and_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.npz"
            preprocessor._build_npz(features=features, metadata=metadata, output_path=output_path)
            loaded = np.load(str(output_path), allow_pickle=True)
            for key in features:
                self.assertIn(key, loaded.files)


# ============================================================================
# GROUP 18: DEFAULT_PROPERTY_KEYS Class Attribute (4 tests)
# ============================================================================


class TestDefaultPropertyKeys(unittest.TestCase):
    """Test ANI2xPreprocessor.DEFAULT_PROPERTY_KEYS class attribute."""

    def test_default_property_keys_exists(self):
        """ANI2xPreprocessor has DEFAULT_PROPERTY_KEYS class attribute."""
        self.assertTrue(hasattr(ANI2xPreprocessor, "DEFAULT_PROPERTY_KEYS"))

    def test_default_property_keys_contains_energies(self):
        """DEFAULT_PROPERTY_KEYS includes 'energies'."""
        self.assertIn("energies", ANI2xPreprocessor.DEFAULT_PROPERTY_KEYS)

    def test_default_property_keys_contains_forces(self):
        """DEFAULT_PROPERTY_KEYS includes 'forces'."""
        self.assertIn("forces", ANI2xPreprocessor.DEFAULT_PROPERTY_KEYS)

    def test_default_property_keys_is_list(self):
        """DEFAULT_PROPERTY_KEYS is a list."""
        self.assertIsInstance(ANI2xPreprocessor.DEFAULT_PROPERTY_KEYS, list)


# ============================================================================
# GROUP 19: _get_h5_path - HDF5 Extraction Logic (5 tests)
# ============================================================================


class TestGetH5Path(unittest.TestCase):
    """Test _get_h5_path method for HDF5 file resolution.

    Evidence: ani2x.py lines 353-397.
    """

    @patch("pathlib.Path.exists", return_value=True)
    def test_h5_file_returned_as_is(self, mock_exists):
        """_get_h5_path returns .h5 path as-is without extraction."""
        preprocessor = _make_preprocessor()
        result = preprocessor._get_h5_path(Path("/data/raw/ANI-2x.h5"))
        self.assertEqual(result, Path("/data/raw/ANI-2x.h5"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_hdf5_file_returned_as_is(self, mock_exists):
        """_get_h5_path returns .hdf5 path as-is without extraction."""
        preprocessor = _make_preprocessor()
        result = preprocessor._get_h5_path(Path("/data/raw/ANI-2x.hdf5"))
        self.assertEqual(result, Path("/data/raw/ANI-2x.hdf5"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_h5_extension_case_insensitive(self, mock_exists):
        """_get_h5_path detects .H5 extension case-insensitively."""
        preprocessor = _make_preprocessor()
        result = preprocessor._get_h5_path(Path("/data/raw/ANI-2x.H5"))
        self.assertEqual(result, Path("/data/raw/ANI-2x.H5"))

    def test_tar_gz_triggers_extraction(self):
        """_get_h5_path extracts HDF5 from tar.gz archive."""
        # Use a single Path.exists patch for both constructor validation and
        # the h5_path.exists() check inside _get_h5_path (ani2x.py line 390).
        with patch("pathlib.Path.exists", autospec=True, return_value=True):
            preprocessor = _make_preprocessor()

            mock_member = Mock()
            mock_member.name = "ANI-2x-wB97X-631Gd.h5"

            mock_tar = MagicMock()
            mock_tar.__enter__ = Mock(return_value=mock_tar)
            mock_tar.__exit__ = Mock(return_value=False)
            mock_tar.getmembers.return_value = [mock_member]

            archive_path = Path("/data/raw/ANI-2x-wB97X-631Gd.tar.gz")
            expected_h5 = archive_path.parent / "ani2x_extracted" / mock_member.name

            with patch("tarfile.open", return_value=mock_tar):
                with patch("pathlib.Path.mkdir"):
                    result = preprocessor._get_h5_path(archive_path)

        self.assertEqual(result, expected_h5)

    @patch("pathlib.Path.exists", return_value=True)
    def test_tar_gz_no_h5_raises(self, mock_exists):
        """_get_h5_path raises DataProcessingError when archive has no HDF5 file.

        Evidence: ani2x.py lines 390-394.
        """
        preprocessor = _make_preprocessor()

        mock_member = Mock()
        mock_member.name = "readme.txt"

        mock_tar = MagicMock()
        mock_tar.__enter__ = Mock(return_value=mock_tar)
        mock_tar.__exit__ = Mock(return_value=False)
        mock_tar.getmembers.return_value = [mock_member]

        archive_path = Path("/data/raw/ANI-2x-wB97X-631Gd.tar.gz")

        with patch("tarfile.open", return_value=mock_tar), patch("pathlib.Path.mkdir"):
            with self.assertRaises(DataProcessingError) as ctx:
                preprocessor._get_h5_path(archive_path)
        self.assertIn("No HDF5 file found", str(ctx.exception))


# ============================================================================
# GROUP 20: _parse_ani2x_h5 - Object Array Construction (3 tests)
# ============================================================================


class TestParseObjectArrayConstruction(unittest.TestCase):
    """Test _parse_ani2x_h5 builds object arrays correctly for ragged data.

    Evidence: ani2x.py lines 479-484 -- _build_object_array uses np.empty()+assignment
    to preserve inner dtypes.
    """

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_atoms_array_is_object_dtype(self, mock_iter, mock_exists):
        """features['atoms'] is object dtype for ragged arrays."""
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(3, 3).astype(np.float32),
                    "molecule_id": "mol_A",
                    "energies": np.float64(-40.5),
                }
            ]
        )
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=None
        )
        self.assertEqual(features["atoms"].dtype, object)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_coordinates_array_is_object_dtype(self, mock_iter, mock_exists):
        """features['coordinates'] is object dtype for ragged arrays."""
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(3, 3).astype(np.float32),
                    "molecule_id": "mol_A",
                    "energies": np.float64(-40.5),
                }
            ]
        )
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=None
        )
        self.assertEqual(features["coordinates"].dtype, object)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_inner_atom_dtype_preserved_as_uint8(self, mock_iter, mock_exists):
        """Inner arrays in features['atoms'] preserve uint8 dtype."""
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(3, 3).astype(np.float32),
                    "molecule_id": "mol_A",
                    "energies": np.float64(-40.5),
                }
            ]
        )
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=None
        )
        self.assertEqual(features["atoms"][0].dtype, np.uint8)


# ============================================================================
# GROUP 21: Edge Cases and Robustness (7 tests)
# ============================================================================


class TestEdgeCasesAndRobustness(unittest.TestCase):
    """Test edge cases and robustness scenarios."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_large_num_molecules_valid(self, mock_exists):
        """Very large num_molecules is valid."""
        _make_preprocessor(config=_make_config(num_molecules=5_000_000))

    @patch("pathlib.Path.exists", return_value=True)
    def test_num_molecules_one_is_valid(self, mock_exists):
        """num_molecules=1 is the minimum valid value."""
        _make_preprocessor(config=_make_config(num_molecules=1))

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_preprocess_with_all_config_options(self, mock_parse, mock_build):
        """Pipeline works with all optional config options specified."""
        config = _make_config(num_molecules=50, property_keys=["energies", "forces"])
        _, result = _create_and_run_pipeline(config, mock_parse, mock_build)
        self.assertEqual(result, Path(config["output_npz_path"]))

    @patch("pathlib.Path.exists", return_value=True)
    def test_config_with_extra_unknown_keys_still_valid(self, mock_exists):
        """Config with extra unknown keys does not cause validation errors."""
        config = _make_config()
        config["extra_key"] = "extra_value"
        _make_preprocessor(config=config)

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_file_format(self, mock_parse, mock_build):
        """Metadata includes file_format='.h5 (HDF5 ANI-2x format)'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(
            mock_build.call_args.kwargs["metadata"]["file_format"], ".h5 (HDF5 ANI-2x format)"
        )

    @patch.object(ANI2xPreprocessor, "_build_npz")
    @patch.object(ANI2xPreprocessor, "_parse_ani2x_h5")
    def test_metadata_preprocessing_version(self, mock_parse, mock_build):
        """Metadata includes preprocessing_version='1.0'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(mock_build.call_args.kwargs["metadata"]["preprocessing_version"], "1.0")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani2x.iter_data_buckets_ani2x")
    def test_parse_energy_nan_appended_when_missing(self, mock_iter, mock_exists):
        """_parse_ani2x_h5 appends np.nan for energy when 'energies' key absent.

        Evidence: ani2x.py lines 454-455.
        """
        mock_iter.return_value = iter(
            [
                {
                    "atomic_numbers": np.array([6, 1], dtype=np.uint8),
                    "coordinates": np.random.randn(2, 3).astype(np.float32),
                    "molecule_id": "mol_A",
                }
            ]
        )
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani2x_h5(
            h5_path=Path("/fake/path.h5"), property_keys=["energies"], max_conformers=None
        )
        self.assertTrue(np.isnan(features["energy"][0]))


# ============================================================================
# GROUP 22: iter_data_buckets_ani2x - Unknown Property Dtype (2 tests)
# ============================================================================


class TestIterUnknownPropertyDtype(unittest.TestCase):
    """Test iter_data_buckets_ani2x handles unknown properties gracefully.

    Evidence: ani2x.py lines 125-126 -- unknown properties use np.array() without explicit dtype.
    """

    def test_unknown_property_preserved(self):
        """Properties not in property_dtypes dict are still extracted."""
        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_a": {
                    "species": np.array([[6, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 2, 3).astype(np.float32),
                    "energies": np.array([-40.5], dtype=np.float64),
                    "custom_prop": np.array([42], dtype=np.int32),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(
                iter_data_buckets_ani2x("/fake/path.h5", keys=["energies", "custom_prop"])
            )

        self.assertEqual(len(results), 1)
        self.assertIn("custom_prop", results[0])

    def test_missing_requested_property_skipped(self):
        """Properties requested but not present in group are silently skipped."""
        mock_h5py = _make_h5_mock_for_iter(
            {
                "mol_a": {
                    "species": np.array([[6, 1]], dtype=np.uint8),
                    "coordinates": np.random.randn(1, 2, 3).astype(np.float32),
                    "energies": np.array([-40.5], dtype=np.float64),
                }
            }
        )
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            results = list(
                iter_data_buckets_ani2x("/fake/path.h5", keys=["energies", "nonexistent_prop"])
            )

        self.assertEqual(len(results), 1)
        self.assertNotIn("nonexistent_prop", results[0])


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestANI2xPreprocessorIdentity,  # GROUP 1:   6 tests
        TestANI2xSupportedElements,  # GROUP 2:   4 tests
        TestValidateConfigSuccess,  # GROUP 3:   4 tests
        TestValidateConfigMissingKeys,  # GROUP 4:   4 tests
        TestValidateConfigPathValidation,  # GROUP 5:   3 tests
        TestValidateConfigExtension,  # GROUP 6:   6 tests
        TestValidateConfigNumMolecules,  # GROUP 7:   5 tests
        TestPreprocessOutputExists,  # GROUP 8:   3 tests
        TestPreprocessFullPipeline,  # GROUP 9:   5 tests
        TestPreprocessErrorWrapping,  # GROUP 10:  5 tests
        TestPreprocessMetadata,  # GROUP 11:  9 tests
        TestPreprocessDefaults,  # GROUP 12:  5 tests
        TestPreprocessStepOrdering,  # GROUP 13:  2 tests
        TestBasePreprocessorRunIntegration,  # GROUP 14:  5 tests
        TestIterDataBucketsAni2x,  # GROUP 15: 12 tests
        TestParseAni2xH5,  # GROUP 16:  7 tests
        TestBuildNpz,  # GROUP 17:  4 tests
        TestDefaultPropertyKeys,  # GROUP 18:  4 tests
        TestGetH5Path,  # GROUP 19:  5 tests
        TestParseObjectArrayConstruction,  # GROUP 20:  3 tests
        TestEdgeCasesAndRobustness,  # GROUP 21:  7 tests
        TestIterUnknownPropertyDtype,  # GROUP 22:  2 tests
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS - preprocessing/preprocessors/ani2x.py")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"\nTest Groups: {len(test_classes)}")

    if result.wasSuccessful():
        print("\n\u2705 ALL TESTS PASSED - PRODUCTION-READY")
        return 0
    else:
        print("\n\u274c SOME TESTS FAILED - REVIEW REQUIRED")
        return 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        pass
    else:
        sys.exit(run_comprehensive_suite())
