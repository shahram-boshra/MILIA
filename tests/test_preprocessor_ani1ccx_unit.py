#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/preprocessing/preprocessors/ani1ccx.py

Module under test: ani1ccx.py
- iter_data_buckets_ccx: Module-level generator yielding conformer dicts from HDF5
  - CRITICAL DIFFERENCE from ANI-1x: REQUIRES ccsd(t)_cbs.energy present per group
  - Auto-prepends 'ccsd(t)_cbs.energy' to keys if not present
  - Skips molecular groups without ccsd(t)_cbs.energy
- ANI1ccxPreprocessor: Preprocessor for ANI-1ccx quantum chemistry dataset (HDF5 format)
  - Inherits BasePreprocessor ABC (2 abstract methods: _validate_config, preprocess)
  - Registered via @PreprocessorRegistry.register("ANI1ccx")
  - CRITICAL: BasePreprocessor.__init__() calls self._validate_config() during construction
  - Pipeline: Parse HDF5 -> Build .npz (2-step, no extraction/cleanup)
  - Config keys: raw_archive_path, output_npz_path, num_molecules, property_keys
  - Supports: .h5, .hdf5 file extensions (warning on unrecognized extension)
  - Auto-skip if output .npz already exists
  - Wraps all errors in DataProcessingError (operation="ani1ccx_preprocessing")
  - Private methods: _parse_ani1ccx_h5, _build_npz
  - preprocess() auto-prepends 'ccsd(t)_cbs.energy' to property_keys if absent

Test path on local machine: ~/ml_projects/milia/tests/test_preprocessor_ani1ccx_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/preprocessing/preprocessor/ani1ccx.py

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution
- iter_data_buckets_ccx h5py mocking uses patch.dict('sys.modules', ...) at test level
  to intercept the LOCAL ``import h5py`` inside the function body (ani1ccx.py line 98).
  patch.dict restores original sys.modules state on context manager exit.

NPZ file paths (mocked, never downloaded):
- ~/Chem_Data/MILIA_PyG_Dataset/raw/ani1x-release.h5

Updated: February 2026 - Production-ready comprehensive test coverage
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call
import logging
import tempfile
import numpy as np
from typing import Dict, Any, List

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.preprocessing.preprocessors.ani1ccx import (
    ANI1ccxPreprocessor,
    iter_data_buckets_ccx,
)
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry
from milia_pipeline.exceptions import ConfigurationError, DataProcessingError


# ============================================================================
# HELPERS: Build realistic config and mock objects
# ============================================================================

def _make_config(**overrides):
    """
    Build a minimal config dict for ANI1ccxPreprocessor tests.

    Based on ANI1ccxPreprocessor._validate_config requirements:
    - Required: 'raw_archive_path', 'output_npz_path'
    - Optional: 'num_molecules', 'property_keys'
    """
    config = {
        'raw_archive_path': overrides.get('raw_archive_path', '/tmp/test_data/raw/ani1x-release.h5'),
        'output_npz_path': overrides.get('output_npz_path', '/tmp/test_data/processed/ani1ccx.npz'),
    }
    for key in ['num_molecules', 'property_keys']:
        if key in overrides:
            config[key] = overrides[key]
    for key in list(config.keys()):
        if overrides.get(f'_remove_{key}', False):
            del config[key]
    return config


def _make_logger():
    """Build a test logger."""
    return logging.getLogger("test.preprocessor.ani1ccx")


def _make_preprocessor(config=None, logger=None):
    """
    Build an ANI1ccxPreprocessor instance with configurable mocks.

    CRITICAL: BasePreprocessor.__init__() calls self._validate_config() during
    construction. Therefore Path.exists MUST be patched BEFORE calling this
    helper (for valid configs), or the constructor will raise ConfigurationError.
    """
    if config is None:
        config = _make_config()
    if logger is None:
        logger = _make_logger()
    return ANI1ccxPreprocessor(config=config, logger=logger)


def _path_exists_factory(h5_path_str, output_path_str):
    """
    Create a Path.exists side_effect that controls which paths 'exist'.

    Returns True for h5_path (so __init__ validation passes),
    False for output_path (so preprocess doesn't skip).
    """
    h5_p = Path(h5_path_str)
    output_p = Path(output_path_str)

    def exists_side_effect(self_path):
        if self_path == h5_p:
            return True
        if self_path == output_p:
            return False
        return False

    return exists_side_effect


def _make_mock_features_and_metadata():
    """
    Build realistic mock features and metadata dicts as returned by _parse_ani1ccx_h5.
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
        'atoms': atoms_arr,
        'coordinates': coords_arr,
        'ccsd_energy': np.array([-40.5, -76.4], dtype=np.float64),
        'molecule_id': mol_id_arr,
    }

    metadata = {
        'total_conformers': 2,
        'skipped_nan': 0,
        'skipped_no_cc': 0,
        'mean_atoms': 4.0,
        'max_atoms': 5,
        'min_atoms': 3,
        'properties_extracted': ['atoms', 'coordinates', 'ccsd_energy', 'molecule_id'],
        'primary_energy_key': 'ccsd_energy',
        'extraction_filter': 'ccsd(t)_cbs.energy present',
    }

    return features, metadata


def _create_and_run_pipeline(config, mock_parse, mock_build, parse_return=None):
    """
    Helper: create preprocessor with proper Path.exists handling and run preprocess.
    """
    mock_parse.return_value = parse_return or _make_mock_features_and_metadata()

    exists_fn = _path_exists_factory(
        config['raw_archive_path'], config['output_npz_path'])

    with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
        preprocessor = _make_preprocessor(config=config)
        result = preprocessor.preprocess()

    return preprocessor, result


# ============================================================================
# GROUP 1: ANI1ccxPreprocessor - Identity and Registration (6 tests)
# ============================================================================

class TestANI1ccxPreprocessorIdentity(unittest.TestCase):
    """Test ANI1ccxPreprocessor identity, registration, and basic attributes."""

    def test_is_subclass_of_base_preprocessor(self):
        """ANI1ccxPreprocessor is a proper BasePreprocessor subclass."""
        self.assertTrue(issubclass(ANI1ccxPreprocessor, BasePreprocessor))

    def test_registered_in_preprocessor_registry(self):
        """ANI1ccxPreprocessor is registered as 'ANI1ccx' in PreprocessorRegistry."""
        self.assertTrue(PreprocessorRegistry.supports_preprocessing("ANI1ccx"))

    def test_registry_returns_correct_class(self):
        """PreprocessorRegistry.get_preprocessor('ANI1ccx') returns ANI1ccxPreprocessor."""
        cls = PreprocessorRegistry.get_preprocessor("ANI1ccx")
        self.assertIs(cls, ANI1ccxPreprocessor)

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
        preprocessor = ANI1ccxPreprocessor(config=_make_config(), logger=logger)
        self.assertIs(preprocessor.logger, logger)

    def test_ani1ccx_in_list_preprocessors(self):
        """'ANI1ccx' appears in PreprocessorRegistry.list_preprocessors()."""
        available = PreprocessorRegistry.list_preprocessors()
        self.assertIn("ANI1ccx", available)


# ============================================================================
# GROUP 2: _validate_config - Success Paths (4 tests)
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
        _make_preprocessor(config=_make_config(
            property_keys=['ccsd(t)_cbs.energy', 'wb97x_dz.energy']))

    @patch("pathlib.Path.exists", return_value=True)
    def test_valid_config_with_all_optional_keys(self, mock_exists):
        """Config with all optional keys passes validation."""
        _make_preprocessor(config=_make_config(
            num_molecules=500,
            property_keys=['ccsd(t)_cbs.energy', 'wb97x_dz.energy']))


# ============================================================================
# GROUP 3: _validate_config - Missing Required Keys (4 tests)
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
# GROUP 4: _validate_config - Path Validation (3 tests)
# ============================================================================

class TestValidateConfigPathValidation(unittest.TestCase):
    """Test _validate_config error paths for invalid file paths."""

    @patch("pathlib.Path.exists", return_value=False)
    def test_nonexistent_h5_path_raises(self, mock_exists):
        """Non-existent raw_archive_path raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(config=_make_config())
        self.assertIn("not found", str(ctx.exception).lower())

    @patch("pathlib.Path.exists", return_value=False)
    def test_nonexistent_h5_path_mentions_path(self, mock_exists):
        """Error for non-existent path mentions the actual path."""
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(config=_make_config())
        self.assertIn("ani1x-release.h5", str(ctx.exception))

    @patch("pathlib.Path.exists", return_value=False)
    def test_nonexistent_path_error_type(self, mock_exists):
        """Path validation error is ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            _make_preprocessor(config=_make_config())


# ============================================================================
# GROUP 5: _validate_config - HDF5 Extension Validation (4 tests)
# ============================================================================

class TestValidateConfigHDF5Extension(unittest.TestCase):
    """Test _validate_config behavior for various HDF5 file extensions."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_h5_extension_accepted(self, mock_exists):
        """Archive with .h5 extension passes validation without warning."""
        _make_preprocessor(config=_make_config(
            raw_archive_path='/tmp/data/ani1x-release.h5'))

    @patch("pathlib.Path.exists", return_value=True)
    def test_hdf5_extension_accepted(self, mock_exists):
        """Archive with .hdf5 extension passes validation without warning."""
        _make_preprocessor(config=_make_config(
            raw_archive_path='/tmp/data/ani1x-release.hdf5'))

    @patch("pathlib.Path.exists", return_value=True)
    def test_uppercase_h5_extension_accepted(self, mock_exists):
        """Archive with .H5 extension (case insensitive) passes validation.

        Evidence: ani1ccx.py line 290: str(h5_path).lower().endswith(('.h5', '.hdf5'))
        """
        _make_preprocessor(config=_make_config(
            raw_archive_path='/tmp/data/ani1x-release.H5'))

    @patch("pathlib.Path.exists", return_value=True)
    def test_unrecognized_extension_logs_warning(self, mock_exists):
        """Unrecognized HDF5 extension logs warning but does not raise.

        Evidence: ani1ccx.py lines 290-294 - warns but proceeds.
        """
        logger = _make_logger()
        with patch.object(logger, 'warning') as mock_warn:
            _make_preprocessor(
                config=_make_config(raw_archive_path='/tmp/data/ani1x-release.dat'),
                logger=logger)
            mock_warn.assert_called_once()
            self.assertIn("not recognized", mock_warn.call_args[0][0].lower())


# ============================================================================
# GROUP 6: _validate_config - num_molecules Validation (5 tests)
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
        """num_molecules=None is valid (extract all ~500k conformers)."""
        _make_preprocessor(config=_make_config(num_molecules=None))


# ============================================================================
# GROUP 7: preprocess - Output Already Exists (Early Return) (3 tests)
# ============================================================================

class TestPreprocessOutputExists(unittest.TestCase):
    """Test preprocess() early return when output .npz already exists."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_existing_output_returns_path_without_processing(self, mock_exists):
        """When output .npz exists, returns path immediately without processing."""
        config = _make_config()
        preprocessor = _make_preprocessor(config=config)

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=1024 * 1024 * 50)
            result = preprocessor.preprocess()

        self.assertEqual(result, Path(config['output_npz_path']))

    @patch("pathlib.Path.exists", return_value=True)
    def test_existing_output_skips_h5_parsing(self, mock_exists):
        """When output .npz exists, HDF5 parsing is never called."""
        preprocessor = _make_preprocessor(config=_make_config())
        with patch.object(Path, 'stat', return_value=Mock(st_size=1024)):
            with patch.object(preprocessor, '_parse_ani1ccx_h5') as mock_parse:
                preprocessor.preprocess()
        mock_parse.assert_not_called()

    @patch("pathlib.Path.exists", return_value=True)
    def test_existing_output_skips_npz_build(self, mock_exists):
        """When output .npz exists, NPZ building is never called."""
        preprocessor = _make_preprocessor(config=_make_config())
        with patch.object(Path, 'stat', return_value=Mock(st_size=1024)):
            with patch.object(preprocessor, '_build_npz') as mock_build:
                preprocessor.preprocess()
        mock_build.assert_not_called()


# ============================================================================
# GROUP 8: preprocess - Full Pipeline Success (5 tests)
# ============================================================================

class TestPreprocessFullPipeline(unittest.TestCase):
    """Test preprocess() full pipeline execution with mocked dependencies."""

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_full_pipeline_returns_output_path(self, mock_parse, mock_build):
        """Full pipeline returns the configured output_npz_path."""
        config = _make_config()
        _, result = _create_and_run_pipeline(config, mock_parse, mock_build)
        self.assertEqual(result, Path(config['output_npz_path']))

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_parse_called_with_h5_path(self, mock_parse, mock_build):
        """Step 1: _parse_ani1ccx_h5 called with correct HDF5 path."""
        config = _make_config()
        _create_and_run_pipeline(config, mock_parse, mock_build)
        mock_parse.assert_called_once()
        call_kwargs = mock_parse.call_args.kwargs
        self.assertEqual(call_kwargs.get('h5_path'), Path(config['raw_archive_path']))

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_parse_called_with_property_keys(self, mock_parse, mock_build):
        """Step 1: _parse_ani1ccx_h5 called with correct property_keys."""
        config = _make_config()
        _create_and_run_pipeline(config, mock_parse, mock_build)
        call_kwargs = mock_parse.call_args.kwargs
        self.assertEqual(
            call_kwargs.get('property_keys'),
            ANI1ccxPreprocessor.DEFAULT_PROPERTY_KEYS
        )

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_parse_called_with_max_conformers(self, mock_parse, mock_build):
        """Step 1: num_molecules config passed as max_conformers."""
        config = _make_config(num_molecules=200)
        _create_and_run_pipeline(config, mock_parse, mock_build)
        call_kwargs = mock_parse.call_args.kwargs
        self.assertEqual(call_kwargs.get('max_conformers'), 200)

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_build_npz_called_with_features_and_metadata(self, mock_parse, mock_build):
        """Step 2: _build_npz called with features from parse and comprehensive metadata."""
        features, parse_metadata = _make_mock_features_and_metadata()
        _create_and_run_pipeline(
            _make_config(), mock_parse, mock_build,
            parse_return=(features, parse_metadata))

        mock_build.assert_called_once()
        kw = mock_build.call_args.kwargs
        self.assertIs(kw.get('features'), features)
        metadata = kw.get('metadata')
        self.assertEqual(metadata.get('version'), '1.0')
        self.assertEqual(metadata.get('dataset_name'), 'ANI1ccx')


# ============================================================================
# GROUP 9: preprocess - Error Wrapping (5 tests)
# ============================================================================

class TestPreprocessErrorWrapping(unittest.TestCase):
    """Test preprocess() wraps all exceptions in DataProcessingError."""

    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_parse_error_wrapped(self, mock_parse):
        """Parsing RuntimeError wrapped in DataProcessingError."""
        config = _make_config()
        mock_parse.side_effect = RuntimeError("HDF5 corrupt")

        exists_fn = _path_exists_factory(config['raw_archive_path'], config['output_npz_path'])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError) as ctx:
                preprocessor.preprocess()
        self.assertIn("ANI-1ccx preprocessing failed", str(ctx.exception))

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_build_error_wrapped(self, mock_parse, mock_build):
        """_build_npz error wrapped in DataProcessingError."""
        config = _make_config()
        mock_parse.return_value = _make_mock_features_and_metadata()
        mock_build.side_effect = IOError("Disk full")

        exists_fn = _path_exists_factory(config['raw_archive_path'], config['output_npz_path'])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError):
                preprocessor.preprocess()

    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_wrapped_error_preserves_cause(self, mock_parse):
        """DataProcessingError preserves original exception as __cause__."""
        config = _make_config()
        original_error = RuntimeError("Original error")
        mock_parse.side_effect = original_error

        exists_fn = _path_exists_factory(config['raw_archive_path'], config['output_npz_path'])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError) as ctx:
                preprocessor.preprocess()
        self.assertIs(ctx.exception.__cause__, original_error)

    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_wrapped_error_mentions_ani1ccx(self, mock_parse):
        """DataProcessingError message includes ANI-1ccx context."""
        config = _make_config()
        mock_parse.side_effect = RuntimeError("fail")

        exists_fn = _path_exists_factory(config['raw_archive_path'], config['output_npz_path'])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError) as ctx:
                preprocessor.preprocess()
        self.assertIn("ANI-1ccx", str(ctx.exception))

    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_value_error_wrapped_as_data_processing_error(self, mock_parse):
        """ValueError (e.g., bad data shape) also wrapped in DataProcessingError."""
        config = _make_config()
        mock_parse.side_effect = ValueError("Invalid shape")

        exists_fn = _path_exists_factory(config['raw_archive_path'], config['output_npz_path'])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError):
                preprocessor.preprocess()


# ============================================================================
# GROUP 10: preprocess - Metadata Construction (9 tests)
# ============================================================================

class TestPreprocessMetadata(unittest.TestCase):
    """Test preprocess() constructs correct metadata for NPZ."""

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_includes_version(self, mock_parse, mock_build):
        """NPZ metadata includes version='1.0'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(mock_build.call_args.kwargs['metadata']['version'], '1.0')

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_includes_dataset_name(self, mock_parse, mock_build):
        """NPZ metadata includes dataset_name='ANI1ccx'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(mock_build.call_args.kwargs['metadata']['dataset_name'], 'ANI1ccx')

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_includes_parser(self, mock_parse, mock_build):
        """NPZ metadata includes parser='ANI1ccxPreprocessor'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(
            mock_build.call_args.kwargs['metadata']['parser'], 'ANI1ccxPreprocessor')

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_includes_source_url(self, mock_parse, mock_build):
        """NPZ metadata includes Figshare source URL."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs['metadata']
        self.assertEqual(metadata['source_url'], 'https://figshare.com/ndownloader/files/18112775')

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_includes_coordinate_and_energy_units(self, mock_parse, mock_build):
        """NPZ metadata includes coordinate_units='angstrom' and energy_units='hartree'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs['metadata']
        self.assertEqual(metadata['coordinate_units'], 'angstrom')
        self.assertEqual(metadata['energy_units'], 'hartree')

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_includes_doi_and_reference(self, mock_parse, mock_build):
        """NPZ metadata includes DOI and reference to Smith et al."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs['metadata']
        self.assertEqual(metadata['doi'], '10.1038/s41597-020-0473-z')
        self.assertIn('Smith', metadata['reference'])

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_merges_parse_metadata(self, mock_parse, mock_build):
        """NPZ metadata merges parse_metadata from _parse_ani1ccx_h5."""
        pm = {'total_conformers': 42, 'skipped_nan': 3, 'skipped_no_cc': 1,
              'mean_atoms': 8.5, 'max_atoms': 15, 'min_atoms': 3,
              'properties_extracted': ['atoms', 'coordinates', 'ccsd_energy'],
              'primary_energy_key': 'ccsd_energy',
              'extraction_filter': 'ccsd(t)_cbs.energy present'}
        _create_and_run_pipeline(
            _make_config(), mock_parse, mock_build,
            parse_return=({}, pm))
        metadata = mock_build.call_args.kwargs['metadata']
        self.assertEqual(metadata['total_conformers'], 42)
        self.assertEqual(metadata['skipped_nan'], 3)

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_primary_energy(self, mock_parse, mock_build):
        """NPZ metadata includes primary_energy referencing CCSD(T)/CBS.

        Evidence: ani1ccx.py line 377: 'primary_energy': 'ccsd(t)_cbs.energy (CCSD(T)/CBS)'
        """
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs['metadata']
        self.assertEqual(metadata['primary_energy'], 'ccsd(t)_cbs.energy (CCSD(T)/CBS)')

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_secondary_energy(self, mock_parse, mock_build):
        """NPZ metadata includes secondary_energy referencing DFT.

        Evidence: ani1ccx.py line 378.
        """
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        metadata = mock_build.call_args.kwargs['metadata']
        self.assertIn('wb97x_dz.energy', metadata['secondary_energy'])


# ============================================================================
# GROUP 11: preprocess - Default Values (5 tests)
# ============================================================================

class TestPreprocessDefaults(unittest.TestCase):
    """Test preprocess() uses correct defaults for optional config keys."""

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_default_num_molecules_is_none(self, mock_parse, mock_build):
        """Default num_molecules is None (extract all ~500k conformers)."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertIsNone(mock_parse.call_args.kwargs.get('max_conformers'))

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_default_property_keys(self, mock_parse, mock_build):
        """Default property_keys uses ANI1ccxPreprocessor.DEFAULT_PROPERTY_KEYS."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(
            mock_parse.call_args.kwargs.get('property_keys'),
            ANI1ccxPreprocessor.DEFAULT_PROPERTY_KEYS
        )

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_custom_property_keys_passed_through(self, mock_parse, mock_build):
        """Custom property_keys config is passed to _parse_ani1ccx_h5."""
        custom_keys = ['ccsd(t)_cbs.energy', 'wb97x_dz.energy', 'wb97x_dz.forces']
        config = _make_config(property_keys=custom_keys)
        _create_and_run_pipeline(config, mock_parse, mock_build)
        self.assertEqual(
            mock_parse.call_args.kwargs.get('property_keys'), custom_keys)

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_source_from_h5_name(self, mock_parse, mock_build):
        """Metadata 'source' field uses HDF5 file name."""
        _create_and_run_pipeline(
            _make_config(raw_archive_path='/data/raw/ani1x-release.h5'),
            mock_parse, mock_build)
        self.assertEqual(
            mock_build.call_args.kwargs['metadata']['source'], 'ani1x-release.h5')

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_default_num_molecules_none_passed_to_parser(self, mock_parse, mock_build):
        """Default num_molecules=None is passed to _parse_ani1ccx_h5 as max_conformers."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertIsNone(mock_parse.call_args.kwargs.get('max_conformers'))


# ============================================================================
# GROUP 12: preprocess - Pipeline Step Ordering (2 tests)
# ============================================================================

class TestPreprocessStepOrdering(unittest.TestCase):
    """Test preprocess() executes pipeline steps in correct order."""

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_steps_execute_in_order(self, mock_parse, mock_build):
        """Steps execute in order: parse_h5 -> build_npz."""
        config = _make_config()
        call_order = []

        def track_parse(**kw):
            call_order.append('parse')
            return _make_mock_features_and_metadata()

        def track_build(**kw):
            call_order.append('build')

        mock_parse.side_effect = track_parse
        mock_build.side_effect = track_build

        exists_fn = _path_exists_factory(config['raw_archive_path'], config['output_npz_path'])
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            preprocessor = _make_preprocessor(config=config)
            preprocessor.preprocess()

        self.assertEqual(call_order, ['parse', 'build'])

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_build_receives_parse_output(self, mock_parse, mock_build):
        """Step 2 receives features and metadata from Step 1."""
        expected_features = {"atoms": np.array([6, 1], dtype=np.uint8)}
        expected_meta = {"total_conformers": 2}
        _create_and_run_pipeline(
            _make_config(), mock_parse, mock_build,
            parse_return=(expected_features, expected_meta))
        self.assertIs(mock_build.call_args.kwargs.get('features'), expected_features)


# ============================================================================
# GROUP 13: BasePreprocessor Integration - run() Method (5 tests)
# ============================================================================

class TestBasePreprocessorRunIntegration(unittest.TestCase):
    """Test ANI1ccxPreprocessor works with BasePreprocessor.run() method."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_run_calls_preprocess_and_validate_output(self, mock_exists):
        """run() calls preprocess() then _validate_output() in sequence.

        Based on actual BasePreprocessor.run() implementation:
        - _validate_config() is called in __init__, NOT in run()
        - run() calls self.preprocess() then self._validate_output(output_path)
        """
        preprocessor = _make_preprocessor(config=_make_config())
        call_order = []

        def mock_preprocess():
            call_order.append('preprocess')
            return Path("/tmp/test_output.npz")

        def mock_validate_output(path):
            call_order.append('validate_output')

        with patch.object(preprocessor, 'preprocess', side_effect=mock_preprocess):
            with patch.object(preprocessor, '_validate_output', side_effect=mock_validate_output):
                preprocessor.run()

        self.assertEqual(call_order, ['preprocess', 'validate_output'])

    def test_run_raises_on_invalid_config(self):
        """Construction raises ConfigurationError when config is invalid."""
        with self.assertRaises(ConfigurationError):
            ANI1ccxPreprocessor(config={}, logger=_make_logger())

    @patch("pathlib.Path.exists", return_value=True)
    def test_run_calls_preprocess(self, mock_exists):
        """run() calls preprocess after validation."""
        preprocessor = _make_preprocessor(config=_make_config())
        with patch.object(Path, 'stat', return_value=Mock(st_size=1024)):
            with patch.object(preprocessor, 'preprocess',
                              wraps=preprocessor.preprocess) as mock_pp:
                try:
                    preprocessor.run()
                except Exception:
                    pass
                mock_pp.assert_called_once()

    def test_has_run_method_from_base(self):
        """ANI1ccxPreprocessor inherits run() from BasePreprocessor."""
        self.assertTrue(hasattr(ANI1ccxPreprocessor, 'run'))

    def test_has_validate_output_from_base(self):
        """ANI1ccxPreprocessor inherits _validate_output() from BasePreprocessor."""
        self.assertTrue(hasattr(ANI1ccxPreprocessor, '_validate_output'))


# ============================================================================
# GROUP 14: iter_data_buckets_ccx - Module-Level Generator (10 tests)
# ============================================================================

class TestIterDataBucketsCcx(unittest.TestCase):
    """Test iter_data_buckets_ccx() module-level generator function.

    CRITICAL MOCKING NOTE:
    iter_data_buckets_ccx() uses a LOCAL import: ``import h5py`` inside the
    function body (ani1ccx.py line 98). This means:
    - @patch("module.h5py", create=True) does NOT work
    - The correct approach: use ``patch.dict('sys.modules', {'h5py': mock_h5py})``
    - This is safe from mock pollution because patch.dict restores original state.

    CRITICAL DIFFERENCE FROM ANI-1x iter_data_buckets:
    - iter_data_buckets_ccx REQUIRES ccsd(t)_cbs.energy in every molecular group.
    - Groups WITHOUT ccsd(t)_cbs.energy are SKIPPED entirely.
    - ccsd(t)_cbs.energy is auto-prepended to keys if not present.
    """

    def _make_h5_mock(self, mol_groups):
        """
        Build a mock h5py module + File context manager from mol_groups.

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

    def test_yields_conformer_dicts(self):
        """iter_data_buckets_ccx yields dict with expected keys for each conformer."""
        mock_h5py = self._make_h5_mock({
            'C1H4': {
                'atomic_numbers': np.array([[6, 1, 1, 1, 1]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 5, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-40.518], dtype=np.float64),
                'wb97x_dz.energy': np.array([-40.5], dtype=np.float64),
            }
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx(
                "/fake/path.h5",
                keys=['ccsd(t)_cbs.energy', 'wb97x_dz.energy']))

        self.assertEqual(len(results), 1)
        self.assertIn('atomic_numbers', results[0])
        self.assertIn('coordinates', results[0])
        self.assertIn('molecule_id', results[0])
        self.assertIn('ccsd(t)_cbs.energy', results[0])
        self.assertIn('wb97x_dz.energy', results[0])

    def test_skips_groups_without_cc_energy(self):
        """iter_data_buckets_ccx skips molecular groups without ccsd(t)_cbs.energy.

        Evidence: ani1ccx.py lines 119-122 - CRITICAL: Skip groups without CC energy.
        """
        mock_h5py = self._make_h5_mock({
            'mol_with_cc': {
                'atomic_numbers': np.array([[6, 1, 1]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 3, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-40.5], dtype=np.float64),
                'wb97x_dz.energy': np.array([-40.0], dtype=np.float64),
            },
            'mol_without_cc': {
                'atomic_numbers': np.array([[8, 1, 1]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 3, 3).astype(np.float32),
                'wb97x_dz.energy': np.array([-76.0], dtype=np.float64),
            },
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx("/fake/path.h5"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['molecule_id'], 'mol_with_cc')

    def test_auto_prepends_cc_energy_to_keys(self):
        """iter_data_buckets_ccx auto-prepends ccsd(t)_cbs.energy if not in keys.

        Evidence: ani1ccx.py lines 112-113.
        """
        mock_h5py = self._make_h5_mock({
            'mol_a': {
                'atomic_numbers': np.array([[6, 1]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 2, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-40.5], dtype=np.float64),
                'wb97x_dz.energy': np.array([-40.0], dtype=np.float64),
            }
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx(
                "/fake/path.h5", keys=['wb97x_dz.energy']))

        self.assertEqual(len(results), 1)
        self.assertIn('ccsd(t)_cbs.energy', results[0])

    def test_default_keys_includes_cc_energy(self):
        """iter_data_buckets_ccx default keys include ccsd(t)_cbs.energy.

        Evidence: ani1ccx.py lines 101-109.
        """
        mock_h5py = self._make_h5_mock({
            'mol_a': {
                'atomic_numbers': np.array([[6, 1, 1]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 3, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-40.518], dtype=np.float64),
                'wb97x_dz.energy': np.array([-40.5], dtype=np.float64),
            }
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx("/fake/path.h5"))

        self.assertEqual(len(results), 1)
        self.assertIn('ccsd(t)_cbs.energy', results[0])

    def test_filters_nan_conformers(self):
        """iter_data_buckets_ccx skips conformers with NaN CC energy values."""
        mock_h5py = self._make_h5_mock({
            'mol_a': {
                'atomic_numbers': np.array([[6, 1, 1, 1]] * 3, dtype=np.uint8),
                'coordinates': np.random.randn(3, 4, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-40.5, np.nan, -38.2], dtype=np.float64),
                'wb97x_dz.energy': np.array([-40.0, -39.0, -38.0], dtype=np.float64),
            }
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx("/fake/path.h5"))

        self.assertEqual(len(results), 2)

    def test_filters_zero_padding_atoms(self):
        """iter_data_buckets_ccx filters out padding atoms (Z=0)."""
        mock_h5py = self._make_h5_mock({
            'mol_padded': {
                'atomic_numbers': np.array([[6, 1, 1, 0, 0]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 5, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-40.5], dtype=np.float64),
            }
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx("/fake/path.h5"))

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]['atomic_numbers']), 3)
        self.assertEqual(results[0]['coordinates'].shape[0], 3)

    def test_molecule_id_set_from_group_name(self):
        """iter_data_buckets_ccx sets molecule_id from HDF5 group name."""
        mock_h5py = self._make_h5_mock({
            'my_molecule_group': {
                'atomic_numbers': np.array([[6, 1]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 2, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-10.0], dtype=np.float64),
            }
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx("/fake/path.h5"))

        self.assertEqual(results[0]['molecule_id'], 'my_molecule_group')

    def test_handles_1d_atomic_numbers(self):
        """iter_data_buckets_ccx handles 1D atomic_numbers (no Nc dim).

        Evidence: ani1ccx.py lines 186-189.
        """
        mock_h5py = self._make_h5_mock({
            'mol_1d': {
                'atomic_numbers': np.array([6, 1, 1], dtype=np.uint8),
                'coordinates': np.random.randn(2, 3, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-40.5, -40.6], dtype=np.float64),
            }
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx("/fake/path.h5"))

        self.assertEqual(len(results), 2)
        np.testing.assert_array_equal(
            results[0]['atomic_numbers'], np.array([6, 1, 1], dtype=np.uint8))
        np.testing.assert_array_equal(
            results[1]['atomic_numbers'], np.array([6, 1, 1], dtype=np.uint8))

    def test_multiple_molecule_groups(self):
        """iter_data_buckets_ccx iterates over all molecular groups with CC energy."""
        mock_h5py = self._make_h5_mock({
            'mol_A': {
                'atomic_numbers': np.array([[6, 1, 1]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 3, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-10.0], dtype=np.float64),
            },
            'mol_B': {
                'atomic_numbers': np.array([[8, 1, 1]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 3, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-20.0], dtype=np.float64),
            },
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx("/fake/path.h5"))

        self.assertEqual(len(results), 2)
        mol_ids = {r['molecule_id'] for r in results}
        self.assertEqual(mol_ids, {'mol_A', 'mol_B'})

    def test_per_atom_property_filtered_by_non_zero_mask(self):
        """Per-atom properties (forces) are filtered by same non_zero_mask.

        Evidence: ani1ccx.py lines 207-211.
        """
        mock_h5py = self._make_h5_mock({
            'padded_mol': {
                'atomic_numbers': np.array([[6, 1, 0]], dtype=np.uint8),
                'coordinates': np.random.randn(1, 3, 3).astype(np.float32),
                'ccsd(t)_cbs.energy': np.array([-40.5], dtype=np.float64),
                'wb97x_dz.forces': np.array(
                    [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
                    dtype=np.float32),
            }
        })
        with patch.dict('sys.modules', {'h5py': mock_h5py}):
            results = list(iter_data_buckets_ccx(
                "/fake/path.h5", keys=['ccsd(t)_cbs.energy', 'wb97x_dz.forces']))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['wb97x_dz.forces'].shape[0], 2)



# ============================================================================
# GROUP 15: _parse_ani1ccx_h5 - Internal Method Logic (5 tests)
# ============================================================================

class TestParseAni1ccxH5(unittest.TestCase):
    """Test _parse_ani1ccx_h5 internal method for HDF5 parsing logic."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_parse_returns_features_and_metadata(self, mock_iter, mock_exists):
        """_parse_ani1ccx_h5 returns (features_dict, metadata_dict) tuple."""
        mock_iter.return_value = iter([
            {
                'atomic_numbers': np.array([6, 1, 1], dtype=np.uint8),
                'coordinates': np.random.randn(3, 3).astype(np.float32),
                'molecule_id': 'mol_A',
                'ccsd(t)_cbs.energy': np.float64(-40.5),
                'wb97x_dz.energy': np.float64(-40.0),
            }
        ])
        preprocessor = _make_preprocessor()
        features, metadata = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy', 'wb97x_dz.energy'],
            max_conformers=None
        )
        self.assertIn('atoms', features)
        self.assertIn('coordinates', features)
        self.assertIn('ccsd_energy', features)
        self.assertIn('molecule_id', features)
        self.assertIn('total_conformers', metadata)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_parse_respects_max_conformers(self, mock_iter, mock_exists):
        """_parse_ani1ccx_h5 stops after max_conformers."""
        conformers = []
        for i in range(5):
            conformers.append({
                'atomic_numbers': np.array([6, 1], dtype=np.uint8),
                'coordinates': np.random.randn(2, 3).astype(np.float32),
                'molecule_id': f'mol_{i}',
                'ccsd(t)_cbs.energy': np.float64(-40.0 + i),
            })
        mock_iter.return_value = iter(conformers)
        preprocessor = _make_preprocessor()
        features, metadata = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy'],
            max_conformers=2
        )
        self.assertEqual(metadata['total_conformers'], 2)
        self.assertEqual(len(features['ccsd_energy']), 2)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_parse_preserves_uint8_atom_dtype(self, mock_iter, mock_exists):
        """_parse_ani1ccx_h5 preserves uint8 dtype for atomic numbers."""
        mock_iter.return_value = iter([
            {
                'atomic_numbers': np.array([6, 1, 1], dtype=np.uint8),
                'coordinates': np.random.randn(3, 3).astype(np.float32),
                'molecule_id': 'mol_A',
                'ccsd(t)_cbs.energy': np.float64(-40.5),
            }
        ])
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy'],
            max_conformers=None
        )
        self.assertEqual(features['atoms'][0].dtype, np.uint8)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_parse_stores_optional_forces(self, mock_iter, mock_exists):
        """_parse_ani1ccx_h5 stores forces when present in conformer data."""
        mock_iter.return_value = iter([
            {
                'atomic_numbers': np.array([6, 1], dtype=np.uint8),
                'coordinates': np.random.randn(2, 3).astype(np.float32),
                'molecule_id': 'mol_A',
                'ccsd(t)_cbs.energy': np.float64(-40.5),
                'wb97x_dz.forces': np.random.randn(2, 3).astype(np.float32),
            }
        ])
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy', 'wb97x_dz.forces'],
            max_conformers=None
        )
        self.assertIn('forces', features)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_parse_metadata_atom_statistics(self, mock_iter, mock_exists):
        """_parse_ani1ccx_h5 computes correct atom statistics in metadata."""
        mock_iter.return_value = iter([
            {
                'atomic_numbers': np.array([6, 1, 1], dtype=np.uint8),
                'coordinates': np.random.randn(3, 3).astype(np.float32),
                'molecule_id': 'mol_A',
                'ccsd(t)_cbs.energy': np.float64(-40.5),
            },
            {
                'atomic_numbers': np.array([8, 1, 1, 1, 1, 1], dtype=np.uint8),
                'coordinates': np.random.randn(6, 3).astype(np.float32),
                'molecule_id': 'mol_B',
                'ccsd(t)_cbs.energy': np.float64(-76.4),
            },
        ])
        preprocessor = _make_preprocessor()
        _, metadata = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy'],
            max_conformers=None
        )
        self.assertEqual(metadata['min_atoms'], 3)
        self.assertEqual(metadata['max_atoms'], 6)
        self.assertAlmostEqual(metadata['mean_atoms'], 4.5)


# ============================================================================
# GROUP 16: _build_npz - Internal Method Logic (4 tests)
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
            self.assertIn('_metadata', loaded.files)

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
# GROUP 17: DEFAULT_PROPERTY_KEYS Class Attribute (4 tests)
# ============================================================================

class TestDefaultPropertyKeys(unittest.TestCase):
    """Test ANI1ccxPreprocessor.DEFAULT_PROPERTY_KEYS class attribute."""

    def test_default_property_keys_exists(self):
        """ANI1ccxPreprocessor has DEFAULT_PROPERTY_KEYS class attribute."""
        self.assertTrue(hasattr(ANI1ccxPreprocessor, 'DEFAULT_PROPERTY_KEYS'))

    def test_default_property_keys_contains_cc_energy(self):
        """DEFAULT_PROPERTY_KEYS includes ccsd(t)_cbs.energy as REQUIRED key."""
        self.assertIn('ccsd(t)_cbs.energy', ANI1ccxPreprocessor.DEFAULT_PROPERTY_KEYS)

    def test_default_property_keys_cc_energy_is_first(self):
        """DEFAULT_PROPERTY_KEYS has ccsd(t)_cbs.energy as first element."""
        self.assertEqual(ANI1ccxPreprocessor.DEFAULT_PROPERTY_KEYS[0], 'ccsd(t)_cbs.energy')

    def test_default_property_keys_is_list(self):
        """DEFAULT_PROPERTY_KEYS is a list."""
        self.assertIsInstance(ANI1ccxPreprocessor.DEFAULT_PROPERTY_KEYS, list)


# ============================================================================
# GROUP 18: preprocess - ccsd(t)_cbs.energy Auto-Prepend Logic (3 tests)
# ============================================================================

class TestPreprocessCCEnergyAutoPrepend(unittest.TestCase):
    """Test preprocess() auto-prepends ccsd(t)_cbs.energy to property_keys."""

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_cc_energy_prepended_when_missing(self, mock_parse, mock_build):
        """Property keys without ccsd(t)_cbs.energy get it auto-prepended."""
        config = _make_config(property_keys=['wb97x_dz.energy', 'wb97x_dz.forces'])
        _create_and_run_pipeline(config, mock_parse, mock_build)
        passed_keys = mock_parse.call_args.kwargs.get('property_keys')
        self.assertEqual(passed_keys[0], 'ccsd(t)_cbs.energy')
        self.assertIn('wb97x_dz.energy', passed_keys)
        self.assertIn('wb97x_dz.forces', passed_keys)

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_cc_energy_not_duplicated_when_present(self, mock_parse, mock_build):
        """Property keys already containing ccsd(t)_cbs.energy don't get duplicate."""
        config = _make_config(property_keys=['ccsd(t)_cbs.energy', 'wb97x_dz.energy'])
        _create_and_run_pipeline(config, mock_parse, mock_build)
        passed_keys = mock_parse.call_args.kwargs.get('property_keys')
        count = passed_keys.count('ccsd(t)_cbs.energy')
        self.assertEqual(count, 1)

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_default_keys_already_include_cc_energy(self, mock_parse, mock_build):
        """Default property_keys already include ccsd(t)_cbs.energy."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        passed_keys = mock_parse.call_args.kwargs.get('property_keys')
        self.assertIn('ccsd(t)_cbs.energy', passed_keys)


# ============================================================================
# GROUP 19: _parse_ani1ccx_h5 - ANI-1ccx-Specific Features (5 tests)
# ============================================================================

class TestParseAni1ccxSpecific(unittest.TestCase):
    """Test _parse_ani1ccx_h5 ANI-1ccx-specific features."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_ccsd_energy_stored_as_float64(self, mock_iter, mock_exists):
        """CC energy stored as float64 in features['ccsd_energy']."""
        mock_iter.return_value = iter([{
            'atomic_numbers': np.array([6, 1], dtype=np.uint8),
            'coordinates': np.random.randn(2, 3).astype(np.float32),
            'molecule_id': 'mol_A',
            'ccsd(t)_cbs.energy': np.float64(-40.5),
        }])
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy'], max_conformers=None)
        self.assertEqual(features['ccsd_energy'].dtype, np.float64)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_dft_energy_stored_when_present(self, mock_iter, mock_exists):
        """DFT energy stored in features['dft_energy'] when present."""
        mock_iter.return_value = iter([{
            'atomic_numbers': np.array([6, 1], dtype=np.uint8),
            'coordinates': np.random.randn(2, 3).astype(np.float32),
            'molecule_id': 'mol_A',
            'ccsd(t)_cbs.energy': np.float64(-40.5),
            'wb97x_dz.energy': np.float64(-40.0),
        }])
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy', 'wb97x_dz.energy'], max_conformers=None)
        self.assertIn('dft_energy', features)
        self.assertEqual(features['dft_energy'].dtype, np.float64)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_metadata_primary_energy_key(self, mock_iter, mock_exists):
        """Metadata includes primary_energy_key='ccsd_energy'."""
        mock_iter.return_value = iter([{
            'atomic_numbers': np.array([6, 1], dtype=np.uint8),
            'coordinates': np.random.randn(2, 3).astype(np.float32),
            'molecule_id': 'mol_A',
            'ccsd(t)_cbs.energy': np.float64(-40.5),
        }])
        preprocessor = _make_preprocessor()
        _, metadata = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy'], max_conformers=None)
        self.assertEqual(metadata['primary_energy_key'], 'ccsd_energy')

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_metadata_extraction_filter(self, mock_iter, mock_exists):
        """Metadata includes extraction_filter describing CC energy requirement."""
        mock_iter.return_value = iter([{
            'atomic_numbers': np.array([6, 1], dtype=np.uint8),
            'coordinates': np.random.randn(2, 3).astype(np.float32),
            'molecule_id': 'mol_A',
            'ccsd(t)_cbs.energy': np.float64(-40.5),
        }])
        preprocessor = _make_preprocessor()
        _, metadata = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy'], max_conformers=None)
        self.assertEqual(metadata['extraction_filter'], 'ccsd(t)_cbs.energy present')

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_optional_charges_stored(self, mock_iter, mock_exists):
        """Optional Hirshfeld and CM5 charges stored when present."""
        mock_iter.return_value = iter([{
            'atomic_numbers': np.array([6, 1], dtype=np.uint8),
            'coordinates': np.random.randn(2, 3).astype(np.float32),
            'molecule_id': 'mol_A',
            'ccsd(t)_cbs.energy': np.float64(-40.5),
            'wb97x_dz.hirshfeld_charges': np.array([0.1, -0.1], dtype=np.float32),
            'wb97x_dz.cm5_charges': np.array([0.2, -0.2], dtype=np.float32),
        }])
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy', 'wb97x_dz.hirshfeld_charges',
                           'wb97x_dz.cm5_charges'], max_conformers=None)
        self.assertIn('hirshfeld_charges', features)
        self.assertIn('cm5_charges', features)


# ============================================================================
# GROUP 20: Edge Cases and Robustness (7 tests)
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

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_preprocess_with_all_config_options(self, mock_parse, mock_build):
        """Pipeline works with all optional config options specified."""
        config = _make_config(
            num_molecules=50,
            property_keys=['ccsd(t)_cbs.energy', 'wb97x_dz.energy', 'wb97x_dz.forces'])
        _, result = _create_and_run_pipeline(config, mock_parse, mock_build)
        self.assertEqual(result, Path(config['output_npz_path']))

    @patch("pathlib.Path.exists", return_value=True)
    def test_config_with_extra_unknown_keys_still_valid(self, mock_exists):
        """Config with extra unknown keys does not cause validation errors."""
        config = _make_config()
        config['extra_key'] = 'extra_value'
        _make_preprocessor(config=config)

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_file_format_is_h5_ani1ccx_subset(self, mock_parse, mock_build):
        """Metadata includes file_format='.h5 (HDF5 ANI-1x format, ANI-1ccx subset)'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(
            mock_build.call_args.kwargs['metadata']['file_format'],
            '.h5 (HDF5 ANI-1x format, ANI-1ccx subset)')

    @patch.object(ANI1ccxPreprocessor, '_build_npz')
    @patch.object(ANI1ccxPreprocessor, '_parse_ani1ccx_h5')
    def test_metadata_preprocessing_version(self, mock_parse, mock_build):
        """Metadata includes preprocessing_version='1.0'."""
        _create_and_run_pipeline(_make_config(), mock_parse, mock_build)
        self.assertEqual(
            mock_build.call_args.kwargs['metadata']['preprocessing_version'], '1.0')

    @patch("pathlib.Path.exists", return_value=True)
    @patch("milia_pipeline.preprocessing.preprocessors.ani1ccx.iter_data_buckets_ccx")
    def test_parse_stores_optional_dipole(self, mock_iter, mock_exists):
        """_parse_ani1ccx_h5 stores dipole vectors when present."""
        mock_iter.return_value = iter([{
            'atomic_numbers': np.array([6, 1], dtype=np.uint8),
            'coordinates': np.random.randn(2, 3).astype(np.float32),
            'molecule_id': 'mol_A',
            'ccsd(t)_cbs.energy': np.float64(-40.5),
            'wb97x_dz.dipole': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        }])
        preprocessor = _make_preprocessor()
        features, _ = preprocessor._parse_ani1ccx_h5(
            h5_path=Path("/fake/path.h5"),
            property_keys=['ccsd(t)_cbs.energy', 'wb97x_dz.dipole'], max_conformers=None)
        self.assertIn('dipole', features)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestANI1ccxPreprocessorIdentity,        # GROUP 1:   6 tests
        TestValidateConfigSuccess,               # GROUP 2:   4 tests
        TestValidateConfigMissingKeys,           # GROUP 3:   4 tests
        TestValidateConfigPathValidation,        # GROUP 4:   3 tests
        TestValidateConfigHDF5Extension,         # GROUP 5:   4 tests
        TestValidateConfigNumMolecules,          # GROUP 6:   5 tests
        TestPreprocessOutputExists,              # GROUP 7:   3 tests
        TestPreprocessFullPipeline,              # GROUP 8:   5 tests
        TestPreprocessErrorWrapping,             # GROUP 9:   5 tests
        TestPreprocessMetadata,                  # GROUP 10:  9 tests
        TestPreprocessDefaults,                  # GROUP 11:  5 tests
        TestPreprocessStepOrdering,              # GROUP 12:  2 tests
        TestBasePreprocessorRunIntegration,      # GROUP 13:  5 tests
        TestIterDataBucketsCcx,                  # GROUP 14: 10 tests
        TestParseAni1ccxH5,                      # GROUP 15:  5 tests
        TestBuildNpz,                            # GROUP 16:  4 tests
        TestDefaultPropertyKeys,                 # GROUP 17:  4 tests
        TestPreprocessCCEnergyAutoPrepend,       # GROUP 18:  3 tests
        TestParseAni1ccxSpecific,                # GROUP 19:  5 tests
        TestEdgeCasesAndRobustness,              # GROUP 20:  7 tests
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS - preprocessing/preprocessors/ani1ccx.py")
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
