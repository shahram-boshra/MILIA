#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/preprocessing/preprocessors/qm40.py

Module under test: qm40.py
- QM40Preprocessor: Preprocessor for the QM40 dataset (ZIP archive of 3 CSV files)
  - Inherits BasePreprocessor ABC (2 abstract methods: _validate_config, preprocess)
  - Registered via @PreprocessorRegistry.register("QM40")
  - CRITICAL: BasePreprocessor.__init__() calls self._validate_config() during construction
  - Pipeline: _extract_archive (stdlib zipfile) -> parse_qm40_csv_files -> build_npz -> cleanup
  - Required config keys: raw_archive_path, output_npz_path
  - Optional config keys: num_molecules, include_bond_data, include_initial_coordinates, cleanup_temp
  - Auto-skip if output .npz already exists
  - _validate_config raises ConfigurationError(config_key=<missing key>) and also
    requires the archive to exist; a non-.zip name is a warning only
  - build_npz is called POSITIONALLY: build_npz(features, metadata, output_npz, logger)
  - parse_qm40_csv_files is called with KWARGS: csv_dir, max_molecules,
    include_bond_data, include_initial_coordinates, logger
  - Wraps pipeline errors in DataProcessingError(operation="qm40_preprocessing")
  - _extract_archive raises DataProcessingError(operation="archive_extraction") on BadZipFile

Test path on local machine: ~/ml_projects/milia/tests/test_preprocessor_qm40_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/preprocessing/preprocessors/qm40.py

NOTE: This test suite runs inside Docker at /app/milia

MOCK POLLUTION PREVENTION:
- NO sys.modules injection at module level
- All mocking via @patch decorators or context managers (test-level only)
- No teardown_module needed since no global mock pollution
- No file downloads — archives/NPZ are mocked, or tiny real zips are built in tempdirs

Updated: June 2026 - Production-ready comprehensive test coverage
"""

import logging
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.preprocessors.qm40 import QM40Preprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry

# Fully-qualified patch targets (names imported INTO the qm40 module namespace)
_BUILD = "milia_pipeline.preprocessing.preprocessors.qm40.build_npz"
_PARSE = "milia_pipeline.preprocessing.preprocessors.qm40.parse_qm40_csv_files"


# ============================================================================
# HELPERS
# ============================================================================


def _make_config(**overrides):
    """
    Build a minimal config dict for QM40Preprocessor tests.

    Required: raw_archive_path, output_npz_path
    Optional: num_molecules, include_bond_data, include_initial_coordinates, cleanup_temp
    Use _remove_<key>=True to drop a required key (error-path testing).
    """
    config = {
        "raw_archive_path": overrides.get(
            "raw_archive_path", "/tmp/test_data/raw/QM40_dataset.zip"
        ),
        "output_npz_path": overrides.get("output_npz_path", "/tmp/test_data/processed/qm40.npz"),
    }
    for key in [
        "num_molecules",
        "include_bond_data",
        "include_initial_coordinates",
        "cleanup_temp",
    ]:
        if key in overrides:
            config[key] = overrides[key]
    for key in list(config.keys()):
        if overrides.get(f"_remove_{key}", False):
            del config[key]
    return config


def _make_logger():
    return logging.getLogger("test.preprocessor.qm40")


def _make_preprocessor(config=None, logger=None):
    """
    Build a QM40Preprocessor instance.

    CRITICAL: BasePreprocessor.__init__() calls self._validate_config() during
    construction, and QM40._validate_config requires the archive to EXIST.
    Therefore Path.exists must be patched True before calling this for configs
    that do not point at a real file.
    """
    if config is None:
        config = _make_config()
    if logger is None:
        logger = _make_logger()
    return QM40Preprocessor(config=config, logger=logger)


def _path_exists_factory(archive_path_str, output_path_str, temp_dir=None, output_exists=False):
    """
    Build a Path.exists side_effect (autospec receives the Path instance).

    archive_path -> True  (so __init__ validation passes)
    output_path  -> output_exists  (default False so preprocess runs the pipeline)
    temp_dir     -> True  (so cleanup proceeds)
    """
    archive_p = Path(archive_path_str)
    output_p = Path(output_path_str)

    def exists_side_effect(self_path):
        if self_path == archive_p:
            return True
        if self_path == output_p:
            return output_exists
        return bool(temp_dir and self_path == temp_dir)

    return exists_side_effect


def _make_features_and_metadata():
    """Mock (features, metadata) as returned by parse_qm40_csv_files."""
    features = {"compounds": ["ZINC1"], "atoms": [[6]], "coordinates": [[[0.0, 0.0, 0.0]]]}
    metadata = {
        "num_molecules_parsed": 1,
        "num_molecules_failed": 0,
        "source_format": "qm40_csv",
        "coordinate_units": "angstrom",
        "energy_units": "hartree",
        "level_of_theory": "B3LYP/6-31G(2df,p)",
    }
    return features, metadata


def _create_and_run_pipeline(
    config,
    mock_extract,
    mock_parse,
    mock_build,
    mock_rmtree,
    extract_return=None,
    parse_return=None,
    output_exists=False,
):
    """Create a preprocessor with controlled Path.exists and run preprocess()."""
    temp_dir = extract_return or Path("/tmp/qm40_extract_fake")
    mock_extract.return_value = temp_dir
    mock_parse.return_value = parse_return or _make_features_and_metadata()

    exists_fn = _path_exists_factory(
        config["raw_archive_path"], config["output_npz_path"], temp_dir, output_exists=output_exists
    )
    # Path.stat is only reached on the early-return branch (output exists); patch
    # it so stat().st_size works on the fake output path used in unit tests.
    with (
        patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn),
        patch("pathlib.Path.stat", autospec=True, return_value=MagicMock(st_size=1024)),
    ):
        preprocessor = _make_preprocessor(config=config)
        result = preprocessor.preprocess()
    return preprocessor, result


def _write_real_qm40_zip(base_dir: Path, name: str = "QM40_dataset.zip") -> Path:
    """Create a tiny but real QM40_dataset.zip with the 3 CSVs under 'QM40 dataset/'."""
    zip_path = base_dir / name
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "QM40 dataset/QM40_main.csv",
            "Zinc_id,smile,Internal_E(0K),HOMO,LUMO,HL_gap,Polarizability,spatial extent,"
            "dipol_mom,ZPE,rot1,rot2,rot3,Inter_E(298),Enthalpy,Free_E,CV,Entropy\n"
            "ZINC1,C,-40.4,-0.38,0.11,-0.5,13.2,35.3,0.0,27.7,157.7,157.7,157.7,"
            "-40.47,-40.47,-40.49,6.4,44.0\n",
        )
        zf.writestr(
            "QM40 dataset/QM40_xyz.csv",
            "Zinc_id,smile,atom,init_x,init_y,init_z,final_x,final_y,final_z,charge\n"
            "ZINC1,C,C,0,0,0,0,0,0,-0.5\n",
        )
        zf.writestr(
            "QM40 dataset/QM40_bond.csv",
            "Zinc_id,smile,atom1,atom2,bond,tag,lmod\n",
        )
    return zip_path


# ============================================================================
# GROUP 1: Identity and Registration
# ============================================================================


class TestIdentityAndRegistration(unittest.TestCase):
    def test_is_subclass_of_base_preprocessor(self):
        self.assertTrue(issubclass(QM40Preprocessor, BasePreprocessor))

    def test_registered_in_preprocessor_registry(self):
        self.assertTrue(PreprocessorRegistry.supports_preprocessing("QM40"))

    def test_registry_returns_correct_class(self):
        self.assertIs(PreprocessorRegistry.get_preprocessor("QM40"), QM40Preprocessor)

    @patch("pathlib.Path.exists", return_value=True)
    def test_stores_config(self, _mock_exists):
        config = _make_config()
        p = _make_preprocessor(config=config)
        self.assertIs(p.config, config)

    @patch("pathlib.Path.exists", return_value=True)
    def test_has_logger(self, _mock_exists):
        p = _make_preprocessor()
        self.assertTrue(hasattr(p, "logger"))


# ============================================================================
# GROUP 2: _validate_config — success paths
# ============================================================================


class TestValidateConfigSuccess(unittest.TestCase):
    @patch("pathlib.Path.exists", return_value=True)
    def test_valid_config_constructs(self, _mock_exists):
        # Should not raise
        _make_preprocessor(_make_config())

    @patch("pathlib.Path.exists", return_value=True)
    def test_extra_unknown_keys_ok(self, _mock_exists):
        config = _make_config()
        config["unexpected"] = "value"
        _make_preprocessor(config)

    @patch("pathlib.Path.exists", return_value=True)
    def test_non_zip_extension_is_warning_not_error(self, _mock_exists):
        config = _make_config(raw_archive_path="/tmp/test_data/raw/QM40_dataset.tar")
        logger = _make_logger()
        with patch.object(logger, "warning") as mock_warn:
            QM40Preprocessor(config=config, logger=logger)
            self.assertTrue(mock_warn.called)


# ============================================================================
# GROUP 3: _validate_config — missing required keys / missing archive
# ============================================================================


class TestValidateConfigErrors(unittest.TestCase):
    def test_missing_raw_archive_path_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(_make_config(_remove_raw_archive_path=True))
        self.assertEqual(ctx.exception.config_key, "raw_archive_path")

    def test_missing_output_npz_path_raises(self):
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(_make_config(_remove_output_npz_path=True))
        self.assertEqual(ctx.exception.config_key, "output_npz_path")

    def test_empty_config_raises_configuration_error(self):
        with self.assertRaises(ConfigurationError):
            _make_preprocessor(config={})

    @patch("pathlib.Path.exists", return_value=False)
    def test_nonexistent_archive_raises(self, _mock_exists):
        with self.assertRaises(ConfigurationError) as ctx:
            _make_preprocessor(_make_config())
        self.assertEqual(ctx.exception.config_key, "raw_archive_path")


# ============================================================================
# GROUP 4: preprocess — output already exists (early return)
# ============================================================================


class TestPreprocessOutputExists(unittest.TestCase):
    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_returns_output_without_extracting(
        self, mock_rmtree, mock_extract, mock_parse, mock_build
    ):
        config = _make_config()
        _, result = _create_and_run_pipeline(
            config, mock_extract, mock_parse, mock_build, mock_rmtree, output_exists=True
        )
        self.assertEqual(result, Path(config["output_npz_path"]))
        mock_extract.assert_not_called()
        mock_parse.assert_not_called()
        mock_build.assert_not_called()


# ============================================================================
# GROUP 5: preprocess — full pipeline success + step ordering
# ============================================================================


class TestPreprocessFullPipeline(unittest.TestCase):
    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_returns_output_path(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        config = _make_config()
        _, result = _create_and_run_pipeline(
            config, mock_extract, mock_parse, mock_build, mock_rmtree
        )
        self.assertEqual(result, Path(config["output_npz_path"]))

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_all_steps_called_once(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        _create_and_run_pipeline(_make_config(), mock_extract, mock_parse, mock_build, mock_rmtree)
        mock_extract.assert_called_once()
        mock_parse.assert_called_once()
        mock_build.assert_called_once()

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_extract_receives_archive_path(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        config = _make_config()
        _create_and_run_pipeline(config, mock_extract, mock_parse, mock_build, mock_rmtree)
        self.assertEqual(mock_extract.call_args[0][0], Path(config["raw_archive_path"]))


# ============================================================================
# GROUP 6: preprocess — parse_qm40_csv_files call contract (kwargs)
# ============================================================================


class TestParseCallContract(unittest.TestCase):
    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_parse_csv_dir_is_extracted_dir(
        self, mock_rmtree, mock_extract, mock_parse, mock_build
    ):
        temp_dir = Path("/tmp/qm40_extract_unit")
        _create_and_run_pipeline(
            _make_config(),
            mock_extract,
            mock_parse,
            mock_build,
            mock_rmtree,
            extract_return=temp_dir,
        )
        self.assertEqual(mock_parse.call_args.kwargs["csv_dir"], temp_dir)

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_parse_receives_logger(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        p, _ = _create_and_run_pipeline(
            _make_config(), mock_extract, mock_parse, mock_build, mock_rmtree
        )
        self.assertIs(mock_parse.call_args.kwargs["logger"], p.logger)

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_parse_defaults(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        """Defaults: max_molecules=None, include_bond_data=True, include_initial_coordinates=True."""
        _create_and_run_pipeline(_make_config(), mock_extract, mock_parse, mock_build, mock_rmtree)
        kw = mock_parse.call_args.kwargs
        self.assertIsNone(kw["max_molecules"])
        self.assertIs(kw["include_bond_data"], True)
        self.assertIs(kw["include_initial_coordinates"], True)

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_num_molecules_passed_as_max_molecules(
        self, mock_rmtree, mock_extract, mock_parse, mock_build
    ):
        _create_and_run_pipeline(
            _make_config(num_molecules=42), mock_extract, mock_parse, mock_build, mock_rmtree
        )
        self.assertEqual(mock_parse.call_args.kwargs["max_molecules"], 42)

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_include_bond_data_false_passed(
        self, mock_rmtree, mock_extract, mock_parse, mock_build
    ):
        _create_and_run_pipeline(
            _make_config(include_bond_data=False), mock_extract, mock_parse, mock_build, mock_rmtree
        )
        self.assertIs(mock_parse.call_args.kwargs["include_bond_data"], False)

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_include_initial_coordinates_false_passed(
        self, mock_rmtree, mock_extract, mock_parse, mock_build
    ):
        _create_and_run_pipeline(
            _make_config(include_initial_coordinates=False),
            mock_extract,
            mock_parse,
            mock_build,
            mock_rmtree,
        )
        self.assertIs(mock_parse.call_args.kwargs["include_initial_coordinates"], False)


# ============================================================================
# GROUP 7: preprocess — build_npz call contract (positional) + metadata
# ============================================================================


class TestBuildCallContract(unittest.TestCase):
    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_build_positional_output_path_and_logger(
        self, mock_rmtree, mock_extract, mock_parse, mock_build
    ):
        config = _make_config()
        p, _ = _create_and_run_pipeline(config, mock_extract, mock_parse, mock_build, mock_rmtree)
        args = mock_build.call_args[0]  # (features, metadata, output_npz, logger)
        self.assertEqual(args[2], Path(config["output_npz_path"]))
        self.assertIs(args[3], p.logger)

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_metadata_enrichment(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        config = _make_config()
        _create_and_run_pipeline(config, mock_extract, mock_parse, mock_build, mock_rmtree)
        metadata = mock_build.call_args[0][1]
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertEqual(metadata["dataset_name"], "QM40")
        self.assertEqual(metadata["source"], "QM40_dataset.zip")
        self.assertEqual(metadata["doi"], "10.1038/s41597-024-04206-y")
        self.assertEqual(metadata["figshare_doi"], "10.6084/m9.figshare.25993060")

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_parser_metadata_preserved(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        """Parser-provided metadata keys survive the enrichment merge."""
        _create_and_run_pipeline(_make_config(), mock_extract, mock_parse, mock_build, mock_rmtree)
        metadata = mock_build.call_args[0][1]
        self.assertEqual(metadata["source_format"], "qm40_csv")
        self.assertEqual(metadata["coordinate_units"], "angstrom")

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_build_receives_parsed_features(
        self, mock_rmtree, mock_extract, mock_parse, mock_build
    ):
        features, metadata = _make_features_and_metadata()
        _create_and_run_pipeline(
            _make_config(),
            mock_extract,
            mock_parse,
            mock_build,
            mock_rmtree,
            parse_return=(features, metadata),
        )
        self.assertIs(mock_build.call_args[0][0], features)


# ============================================================================
# GROUP 8: preprocess — cleanup behavior
# ============================================================================


class TestPreprocessCleanup(unittest.TestCase):
    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_cleanup_default_removes_temp(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        temp_dir = Path("/tmp/qm40_extract_cleanup")
        _create_and_run_pipeline(
            _make_config(),
            mock_extract,
            mock_parse,
            mock_build,
            mock_rmtree,
            extract_return=temp_dir,
        )
        mock_rmtree.assert_called_once_with(temp_dir)

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_cleanup_disabled_keeps_temp(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        _create_and_run_pipeline(
            _make_config(cleanup_temp=False), mock_extract, mock_parse, mock_build, mock_rmtree
        )
        mock_rmtree.assert_not_called()

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_cleanup_runs_on_parse_error(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        """The finally-block cleanup runs even when parsing raises."""
        config = _make_config()
        temp_dir = Path("/tmp/qm40_extract_err")
        mock_extract.return_value = temp_dir
        mock_parse.side_effect = ValueError("boom")
        exists_fn = _path_exists_factory(
            config["raw_archive_path"], config["output_npz_path"], temp_dir
        )
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            p = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError):
                p.preprocess()
        mock_rmtree.assert_called_once_with(temp_dir)


# ============================================================================
# GROUP 9: preprocess — error wrapping
# ============================================================================


class TestPreprocessErrorWrapping(unittest.TestCase):
    def _run_expecting_error(self, config, mock_extract, mock_parse, mock_build):
        temp_dir = Path("/tmp/qm40_extract_wrap")
        mock_extract.return_value = temp_dir
        mock_parse.return_value = _make_features_and_metadata()
        exists_fn = _path_exists_factory(
            config["raw_archive_path"], config["output_npz_path"], temp_dir
        )
        with patch("pathlib.Path.exists", autospec=True, side_effect=exists_fn):
            p = _make_preprocessor(config=config)
            with self.assertRaises(DataProcessingError) as ctx:
                p.preprocess()
        return ctx.exception

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_parse_error_wrapped(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        mock_parse.side_effect = ValueError("parse failed")
        exc = self._run_expecting_error(_make_config(), mock_extract, mock_parse, mock_build)
        self.assertEqual(exc.operation, "qm40_preprocessing")
        self.assertIsInstance(exc.__cause__, ValueError)

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_build_error_wrapped(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        mock_build.side_effect = OSError("disk full")
        exc = self._run_expecting_error(_make_config(), mock_extract, mock_parse, mock_build)
        self.assertEqual(exc.operation, "qm40_preprocessing")

    @patch(_BUILD)
    @patch(_PARSE)
    @patch.object(QM40Preprocessor, "_extract_archive")
    @patch("shutil.rmtree")
    def test_extract_error_wrapped(self, mock_rmtree, mock_extract, mock_parse, mock_build):
        mock_extract.side_effect = RuntimeError("bad archive")
        exc = self._run_expecting_error(_make_config(), mock_extract, mock_parse, mock_build)
        self.assertEqual(exc.operation, "qm40_preprocessing")
        # _extract_archive failed before the inner try/finally -> no cleanup attempt
        mock_rmtree.assert_not_called()


# ============================================================================
# GROUP 10: _extract_archive — real extraction + error handling
# ============================================================================


class TestExtractArchive(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_qm40_extract_")
        self._base = Path(self._tmpdir)
        self._created_dirs = []

    def tearDown(self):
        for d in self._created_dirs:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_extracts_real_zip_and_returns_dir(self):
        zip_path = _write_real_qm40_zip(self._base)
        config = _make_config(
            raw_archive_path=str(zip_path), output_npz_path=str(self._base / "out.npz")
        )
        # Real archive exists -> __init__ validation passes without patching
        p = _make_preprocessor(config=config)
        extracted = p._extract_archive(zip_path)
        self._created_dirs.append(extracted)
        self.assertTrue(extracted.exists())
        csvs = list(extracted.rglob("*.csv"))
        self.assertGreaterEqual(len(csvs), 3)

    def test_extracted_dir_contains_spaced_subfolder(self):
        zip_path = _write_real_qm40_zip(self._base)
        config = _make_config(
            raw_archive_path=str(zip_path), output_npz_path=str(self._base / "out.npz")
        )
        p = _make_preprocessor(config=config)
        extracted = p._extract_archive(zip_path)
        self._created_dirs.append(extracted)
        self.assertTrue((extracted / "QM40 dataset").exists())

    def test_bad_zip_raises_data_processing_error(self):
        bad_zip = self._base / "QM40_dataset.zip"
        bad_zip.write_bytes(b"this is not a zip file")
        config = _make_config(
            raw_archive_path=str(bad_zip), output_npz_path=str(self._base / "out.npz")
        )
        p = _make_preprocessor(config=config)
        with self.assertRaises(DataProcessingError) as ctx:
            p._extract_archive(bad_zip)
        self.assertEqual(ctx.exception.operation, "archive_extraction")


# ============================================================================
# GROUP 11: BasePreprocessor integration surface
# ============================================================================


class TestBasePreprocessorSurface(unittest.TestCase):
    @patch("pathlib.Path.exists", return_value=True)
    def test_has_run_method(self, _mock_exists):
        p = _make_preprocessor()
        self.assertTrue(hasattr(p, "run") and callable(p.run))

    @patch("pathlib.Path.exists", return_value=True)
    def test_has_validate_output_method(self, _mock_exists):
        p = _make_preprocessor()
        self.assertTrue(hasattr(p, "_validate_output"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_preprocess_is_callable(self, _mock_exists):
        p = _make_preprocessor()
        self.assertTrue(callable(p.preprocess))


# ============================================================================
# Comprehensive runner (pytest + unittest compatible)
# ============================================================================


def run_comprehensive_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestIdentityAndRegistration,
        TestValidateConfigSuccess,
        TestValidateConfigErrors,
        TestPreprocessOutputExists,
        TestPreprocessFullPipeline,
        TestParseCallContract,
        TestBuildCallContract,
        TestPreprocessCleanup,
        TestPreprocessErrorWrapping,
        TestExtractArchive,
        TestBasePreprocessorSurface,
    ]
    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — preprocessing/preprocessors/qm40.py")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "pytest" in sys.modules:
        pass
    else:
        sys.exit(run_comprehensive_suite())
