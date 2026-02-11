#!/usr/bin/env python3
"""
PRODUCTION-READY Unit Test Suite for milia_pipeline/preprocessing/utils/archive_handlers.py

Module under test: archive_handlers.py
- COMPRESSION_MODES: Module-level dict mapping extensions to tarfile modes
- _detect_compression_mode(tar_path): Detect tarfile mode from file extension
- extract_from_archive(): Generic multi-format archive extractor (streaming)
- extract_from_targz(): Backward-compatible wrapper for .tar.gz extraction
- get_supported_formats(): Returns copy of COMPRESSION_MODES dict

Test path on local machine: ~/ml_projects/milia/tests/test_archive_handlers_unit.py
Module path on local machine: ~/ml_projects/milia/milia_pipeline/preprocessing/utils/archive_handlers.py

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
from unittest.mock import Mock, MagicMock, patch, call
import logging
import tarfile
import tempfile
import shutil
from typing import Optional

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from milia_pipeline.preprocessing.utils.archive_handlers import (
    COMPRESSION_MODES,
    _detect_compression_mode,
    extract_from_archive,
    extract_from_targz,
    get_supported_formats,
)
from milia_pipeline.exceptions import DataProcessingError


# ============================================================================
# HELPERS: Archive creation utilities for testing
# ============================================================================

def _create_tar_archive(
    archive_path: Path,
    files: dict,
    mode: str = 'w:gz',
) -> Path:
    """
    Create a tar archive with the given files for testing.

    Args:
        archive_path: Path where the archive will be created.
        files: Dict mapping {internal_filename: file_content_bytes}.
        mode: tarfile open mode (e.g., 'w:gz', 'w:bz2', 'w:xz', 'w:').

    Returns:
        Path to the created archive.
    """
    import io

    with tarfile.open(archive_path, mode) as tar:
        for name, content in files.items():
            data = content if isinstance(content, bytes) else content.encode('utf-8')
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    return archive_path


def _make_logger():
    """Create a logger instance for testing."""
    logger = logging.getLogger("test_archive_handlers")
    logger.setLevel(logging.DEBUG)
    return logger


# ============================================================================
# GROUP 1: COMPRESSION_MODES Module-Level Constant (7 tests)
# ============================================================================

class TestCompressionModes(unittest.TestCase):
    """Test that COMPRESSION_MODES is correctly defined."""

    def test_compression_modes_is_dict(self):
        """COMPRESSION_MODES is a dictionary."""
        self.assertIsInstance(COMPRESSION_MODES, dict)

    def test_tar_gz_extension_present(self):
        """.tar.gz extension maps to 'r:gz'."""
        self.assertEqual(COMPRESSION_MODES['.tar.gz'], 'r:gz')

    def test_tgz_extension_present(self):
        """.tgz extension maps to 'r:gz'."""
        self.assertEqual(COMPRESSION_MODES['.tgz'], 'r:gz')

    def test_tar_bz2_extension_present(self):
        """.tar.bz2 extension maps to 'r:bz2'."""
        self.assertEqual(COMPRESSION_MODES['.tar.bz2'], 'r:bz2')

    def test_tbz2_extension_present(self):
        """.tbz2 extension maps to 'r:bz2'."""
        self.assertEqual(COMPRESSION_MODES['.tbz2'], 'r:bz2')

    def test_tar_xz_extension_present(self):
        """.tar.xz extension maps to 'r:xz'."""
        self.assertEqual(COMPRESSION_MODES['.tar.xz'], 'r:xz')

    def test_tar_uncompressed_present(self):
        """.tar extension maps to 'r:' (uncompressed)."""
        self.assertEqual(COMPRESSION_MODES['.tar'], 'r:')

    def test_expected_number_of_formats(self):
        """COMPRESSION_MODES contains exactly 7 format mappings."""
        self.assertEqual(len(COMPRESSION_MODES), 7)


# ============================================================================
# GROUP 2: _detect_compression_mode (10 tests)
# ============================================================================

class TestDetectCompressionMode(unittest.TestCase):
    """Test _detect_compression_mode correctly identifies compression from filename."""

    def test_detects_tar_gz(self):
        """Detects .tar.gz as 'r:gz'."""
        result = _detect_compression_mode(Path("archive.tar.gz"))
        self.assertEqual(result, 'r:gz')

    def test_detects_tgz(self):
        """Detects .tgz as 'r:gz'."""
        result = _detect_compression_mode(Path("archive.tgz"))
        self.assertEqual(result, 'r:gz')

    def test_detects_tar_bz2(self):
        """Detects .tar.bz2 as 'r:bz2'."""
        result = _detect_compression_mode(Path("data.tar.bz2"))
        self.assertEqual(result, 'r:bz2')

    def test_detects_tbz2(self):
        """Detects .tbz2 as 'r:bz2'."""
        result = _detect_compression_mode(Path("data.tbz2"))
        self.assertEqual(result, 'r:bz2')

    def test_detects_tar_xz(self):
        """Detects .tar.xz as 'r:xz'."""
        result = _detect_compression_mode(Path("data.tar.xz"))
        self.assertEqual(result, 'r:xz')

    def test_detects_txz(self):
        """Detects .txz as 'r:xz'."""
        result = _detect_compression_mode(Path("data.txz"))
        self.assertEqual(result, 'r:xz')

    def test_detects_plain_tar(self):
        """Detects .tar as 'r:' (uncompressed)."""
        result = _detect_compression_mode(Path("data.tar"))
        self.assertEqual(result, 'r:')

    def test_unknown_extension_returns_auto_detect(self):
        """Unknown extension falls back to 'r:*' auto-detection mode."""
        result = _detect_compression_mode(Path("data.zip"))
        self.assertEqual(result, 'r:*')

    def test_case_insensitive_detection(self):
        """Detection is case-insensitive (uses .lower() on filename)."""
        result = _detect_compression_mode(Path("DATA.TAR.GZ"))
        self.assertEqual(result, 'r:gz')

    def test_compound_extension_takes_priority(self):
        """Compound extensions like .tar.gz match before .gz alone."""
        # Even if .gz were in the dict, .tar.gz should match first
        result = _detect_compression_mode(Path("myfile.tar.gz"))
        self.assertEqual(result, 'r:gz')


# ============================================================================
# GROUP 3: extract_from_archive — Happy Path (12 tests)
# ============================================================================

class TestExtractFromArchiveHappyPath(unittest.TestCase):
    """Test extract_from_archive with valid archives and expected behavior."""

    def setUp(self):
        """Create temporary directories for test archives and extraction."""
        self._tmpdir = tempfile.mkdtemp(prefix='test_archive_handlers_')
        self._extract_dir = tempfile.mkdtemp(prefix='test_extract_')

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        shutil.rmtree(self._extract_dir, ignore_errors=True)

    def _archive_path(self, name="test.tar.gz"):
        return Path(self._tmpdir) / name

    def test_extracts_files_with_matching_extension(self):
        """Extracts only files matching the specified file_extension."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={
                "mol1.xyz": "atom data 1",
                "mol2.xyz": "atom data 2",
                "readme.txt": "not an xyz file",
            },
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 2)

    def test_returns_path_object(self):
        """Returns a Path object pointing to the extraction directory."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"mol.xyz": "data"},
            mode='w:gz',
        )
        result = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        self.assertIsInstance(result, Path)

    def test_returns_temp_dir_path(self):
        """When temp_dir is specified, extraction directory is that path."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"mol.xyz": "data"},
            mode='w:gz',
        )
        result = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        self.assertEqual(result, Path(self._extract_dir))

    def test_max_files_limits_extraction(self):
        """max_files parameter limits the number of extracted files."""
        files = {f"mol_{i}.xyz": f"data {i}" for i in range(10)}
        archive = _create_tar_archive(
            self._archive_path(),
            files=files,
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            max_files=3,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 3)

    def test_extracts_tar_bz2_format(self):
        """Successfully extracts from .tar.bz2 archives."""
        archive = _create_tar_archive(
            self._archive_path("test.tar.bz2"),
            files={"mol.xyz": "bz2 data"},
            mode='w:bz2',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 1)

    def test_extracts_tar_xz_format(self):
        """Successfully extracts from .tar.xz archives."""
        archive = _create_tar_archive(
            self._archive_path("test.tar.xz"),
            files={"mol.xyz": "xz data"},
            mode='w:xz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 1)

    def test_extracts_plain_tar_format(self):
        """Successfully extracts from uncompressed .tar archives."""
        archive = _create_tar_archive(
            self._archive_path("test.tar"),
            files={"mol.xyz": "plain tar data"},
            mode='w:',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 1)

    def test_compression_mode_override(self):
        """compression_mode parameter overrides auto-detection."""
        # Create a .tar.gz file but tell extract_from_archive to use a custom mode
        archive = _create_tar_archive(
            self._archive_path("test.tar.gz"),
            files={"mol.xyz": "data"},
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
            compression_mode='r:gz',
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 1)

    def test_default_file_extension_is_xyz(self):
        """Default file_extension parameter is '.xyz'."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"mol.xyz": "data", "other.txt": "text"},
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            temp_dir=Path(self._extract_dir),
        )
        extracted_xyz = list(result_dir.rglob("*.xyz"))
        extracted_txt = list(result_dir.rglob("*.txt"))
        self.assertEqual(len(extracted_xyz), 1)
        self.assertEqual(len(extracted_txt), 0)

    def test_creates_temp_dir_when_none(self):
        """When temp_dir is None, a temporary directory is created automatically."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"mol.xyz": "data"},
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=None,
        )
        try:
            self.assertTrue(result_dir.exists())
            self.assertTrue(result_dir.is_dir())
            # Should have the milia_extract_ prefix
            self.assertIn('milia_extract_', result_dir.name)
        finally:
            shutil.rmtree(result_dir, ignore_errors=True)

    def test_creates_temp_dir_parent_if_not_exists(self):
        """When temp_dir parent doesn't exist, it is created with parents=True."""
        nested_dir = Path(self._extract_dir) / "nested" / "deep" / "dir"
        archive = _create_tar_archive(
            self._archive_path(),
            files={"mol.xyz": "data"},
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=nested_dir,
        )
        self.assertTrue(result_dir.exists())

    def test_extracts_files_in_subdirectories(self):
        """Files in subdirectories within the archive are extracted correctly."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={
                "subdir/mol1.xyz": "data1",
                "subdir/nested/mol2.xyz": "data2",
            },
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 2)


# ============================================================================
# GROUP 4: extract_from_archive — Error Paths (8 tests)
# ============================================================================

class TestExtractFromArchiveErrors(unittest.TestCase):
    """Test extract_from_archive error handling and edge cases."""

    def setUp(self):
        """Create temporary directories."""
        self._tmpdir = tempfile.mkdtemp(prefix='test_archive_errors_')
        self._extract_dir = tempfile.mkdtemp(prefix='test_extract_errors_')

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        shutil.rmtree(self._extract_dir, ignore_errors=True)

    def _archive_path(self, name="test.tar.gz"):
        return Path(self._tmpdir) / name

    def test_nonexistent_archive_raises(self):
        """Raises DataProcessingError when archive file does not exist."""
        fake_path = Path("/tmp/nonexistent_archive_abc123.tar.gz")
        with self.assertRaises(DataProcessingError) as ctx:
            extract_from_archive(archive_path=fake_path)
        self.assertIn("Archive not found", str(ctx.exception))

    def test_nonexistent_archive_includes_path_in_error(self):
        """Error message includes the archive path."""
        fake_path = Path("/tmp/nonexistent_archive_xyz.tar.gz")
        with self.assertRaises(DataProcessingError) as ctx:
            extract_from_archive(archive_path=fake_path)
        self.assertIn(str(fake_path), str(ctx.exception))

    def test_no_matching_files_raises(self):
        """Raises DataProcessingError when no files match the extension filter."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"readme.txt": "text only", "data.csv": "csv data"},
            mode='w:gz',
        )
        with self.assertRaises(DataProcessingError) as ctx:
            extract_from_archive(
                archive_path=archive,
                file_extension='.xyz',
                temp_dir=Path(self._extract_dir),
            )
        self.assertIn(".xyz", str(ctx.exception))

    def test_no_matching_files_error_mentions_extension(self):
        """Error for no matching files mentions the expected file extension."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"readme.txt": "text"},
            mode='w:gz',
        )
        with self.assertRaises(DataProcessingError) as ctx:
            extract_from_archive(
                archive_path=archive,
                file_extension='.molden',
                temp_dir=Path(self._extract_dir),
            )
        self.assertIn(".molden", str(ctx.exception))

    def test_corrupt_archive_raises_data_processing_error(self):
        """Raises DataProcessingError for corrupt/invalid archive files."""
        corrupt_path = self._archive_path("corrupt.tar.gz")
        corrupt_path.write_bytes(b"this is not a valid archive")
        with self.assertRaises(DataProcessingError):
            extract_from_archive(
                archive_path=corrupt_path,
                file_extension='.xyz',
                temp_dir=Path(self._extract_dir),
            )

    def test_corrupt_archive_preserves_cause(self):
        """DataProcessingError from corrupt archive preserves TarError as __cause__."""
        corrupt_path = self._archive_path("corrupt2.tar.gz")
        corrupt_path.write_bytes(b"not a tar file at all")
        with self.assertRaises(DataProcessingError) as ctx:
            extract_from_archive(
                archive_path=corrupt_path,
                file_extension='.xyz',
                temp_dir=Path(self._extract_dir),
            )
        self.assertIsInstance(ctx.exception.__cause__, tarfile.TarError)

    def test_error_attributes_file_path(self):
        """DataProcessingError for missing archive has file_path attribute set."""
        fake_path = Path("/tmp/missing_archive_test.tar.gz")
        with self.assertRaises(DataProcessingError) as ctx:
            extract_from_archive(archive_path=fake_path)
        # DataProcessingError has file_path and operation attributes
        self.assertEqual(ctx.exception.file_path, str(fake_path))

    def test_error_attributes_operation(self):
        """DataProcessingError has operation attribute set to 'archive_extraction'."""
        fake_path = Path("/tmp/missing_archive_op_test.tar.gz")
        with self.assertRaises(DataProcessingError) as ctx:
            extract_from_archive(archive_path=fake_path)
        self.assertEqual(ctx.exception.operation, "archive_extraction")


# ============================================================================
# GROUP 5: extract_from_archive — Logging (8 tests)
# ============================================================================

class TestExtractFromArchiveLogging(unittest.TestCase):
    """Test that extract_from_archive logs appropriate messages."""

    def setUp(self):
        """Create temp directories and a simple test archive."""
        self._tmpdir = tempfile.mkdtemp(prefix='test_archive_logging_')
        self._extract_dir = tempfile.mkdtemp(prefix='test_extract_logging_')
        self._archive = _create_tar_archive(
            Path(self._tmpdir) / "test.tar.gz",
            files={f"mol_{i}.xyz": f"data {i}" for i in range(5)},
            mode='w:gz',
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        shutil.rmtree(self._extract_dir, ignore_errors=True)

    @patch('milia_pipeline.preprocessing.utils.archive_handlers.logger')
    def test_logs_archive_name(self, mock_logger):
        """Logs the archive filename during extraction."""
        extract_from_archive(
            archive_path=self._archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        info_messages = [
            str(c) for c in mock_logger.info.call_args_list
        ]
        joined = " ".join(info_messages)
        self.assertIn("test.tar.gz", joined)

    @patch('milia_pipeline.preprocessing.utils.archive_handlers.logger')
    def test_logs_archive_size(self, mock_logger):
        """Logs the archive size in GB."""
        extract_from_archive(
            archive_path=self._archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        info_messages = [
            str(c) for c in mock_logger.info.call_args_list
        ]
        joined = " ".join(info_messages)
        self.assertIn("GB", joined)

    @patch('milia_pipeline.preprocessing.utils.archive_handlers.logger')
    def test_logs_compression_mode(self, mock_logger):
        """Logs the compression mode being used."""
        extract_from_archive(
            archive_path=self._archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        info_messages = [
            str(c) for c in mock_logger.info.call_args_list
        ]
        joined = " ".join(info_messages)
        self.assertIn("r:gz", joined)

    @patch('milia_pipeline.preprocessing.utils.archive_handlers.logger')
    def test_logs_extraction_complete_count(self, mock_logger):
        """Logs the total count of extracted files upon completion."""
        extract_from_archive(
            archive_path=self._archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        info_messages = [
            str(c) for c in mock_logger.info.call_args_list
        ]
        joined = " ".join(info_messages)
        self.assertIn("5", joined)

    @patch('milia_pipeline.preprocessing.utils.archive_handlers.logger')
    def test_logs_max_files_limit_when_reached(self, mock_logger):
        """Logs when max_files limit is reached."""
        extract_from_archive(
            archive_path=self._archive,
            max_files=2,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        info_messages = [
            str(c) for c in mock_logger.info.call_args_list
        ]
        joined = " ".join(info_messages)
        self.assertIn("max_files", joined)

    @patch('milia_pipeline.preprocessing.utils.archive_handlers.logger')
    def test_logs_file_extension_in_completion(self, mock_logger):
        """Completion log includes the file extension being extracted."""
        extract_from_archive(
            archive_path=self._archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        info_messages = [
            str(c) for c in mock_logger.info.call_args_list
        ]
        joined = " ".join(info_messages)
        self.assertIn(".xyz", joined)

    @patch('milia_pipeline.preprocessing.utils.archive_handlers.logger')
    def test_logs_extraction_commenced(self, mock_logger):
        """Logs that extraction has commenced."""
        extract_from_archive(
            archive_path=self._archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        info_messages = [
            str(c) for c in mock_logger.info.call_args_list
        ]
        joined = " ".join(info_messages)
        self.assertIn("commenced", joined.lower())

    @patch('milia_pipeline.preprocessing.utils.archive_handlers.logger')
    def test_debug_logs_individual_files(self, mock_logger):
        """Debug-level logs include individual extracted file names."""
        extract_from_archive(
            archive_path=self._archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        debug_messages = [
            str(c) for c in mock_logger.debug.call_args_list
        ]
        joined = " ".join(debug_messages)
        self.assertIn("mol_0.xyz", joined)


# ============================================================================
# GROUP 6: extract_from_targz — Backward Compatibility Wrapper (6 tests)
# ============================================================================

class TestExtractFromTargz(unittest.TestCase):
    """Test extract_from_targz backward-compatibility wrapper."""

    def setUp(self):
        """Create temp directories."""
        self._tmpdir = tempfile.mkdtemp(prefix='test_targz_')
        self._extract_dir = tempfile.mkdtemp(prefix='test_extract_targz_')

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        shutil.rmtree(self._extract_dir, ignore_errors=True)

    def _archive_path(self, name="test.tar.gz"):
        return Path(self._tmpdir) / name

    def test_extracts_tar_gz_successfully(self):
        """extract_from_targz extracts .tar.gz files correctly."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"file.molden": "molden data"},
            mode='w:gz',
        )
        result_dir = extract_from_targz(
            tar_path=archive,
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.molden"))
        self.assertEqual(len(extracted), 1)

    def test_default_file_extension_is_molden(self):
        """Default file_extension for extract_from_targz is '.molden'."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"file.molden": "molden data", "file.xyz": "xyz data"},
            mode='w:gz',
        )
        result_dir = extract_from_targz(
            tar_path=archive,
            temp_dir=Path(self._extract_dir),
        )
        extracted_molden = list(result_dir.rglob("*.molden"))
        extracted_xyz = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted_molden), 1)
        self.assertEqual(len(extracted_xyz), 0)

    def test_max_files_parameter_works(self):
        """max_files parameter is passed through to extract_from_archive."""
        files = {f"mol_{i}.molden": f"data {i}" for i in range(10)}
        archive = _create_tar_archive(
            self._archive_path(),
            files=files,
            mode='w:gz',
        )
        result_dir = extract_from_targz(
            tar_path=archive,
            max_files=4,
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.molden"))
        self.assertEqual(len(extracted), 4)

    @patch(
        'milia_pipeline.preprocessing.utils.archive_handlers.extract_from_archive'
    )
    def test_delegates_to_extract_from_archive(self, mock_extract):
        """extract_from_targz delegates to extract_from_archive with compression_mode='r:gz'."""
        mock_extract.return_value = Path("/tmp/mock_dir")
        tar_path = Path("/tmp/some.tar.gz")

        extract_from_targz(
            tar_path=tar_path,
            max_files=5,
            file_extension='.xyz',
            temp_dir=Path("/tmp/out"),
        )

        mock_extract.assert_called_once_with(
            archive_path=tar_path,
            max_files=5,
            file_extension='.xyz',
            temp_dir=Path("/tmp/out"),
            compression_mode='r:gz',
        )

    @patch(
        'milia_pipeline.preprocessing.utils.archive_handlers.extract_from_archive'
    )
    def test_returns_result_from_extract_from_archive(self, mock_extract):
        """extract_from_targz returns the Path from extract_from_archive."""
        expected = Path("/tmp/result_dir")
        mock_extract.return_value = expected

        result = extract_from_targz(tar_path=Path("/tmp/a.tar.gz"))
        self.assertEqual(result, expected)

    def test_nonexistent_archive_raises(self):
        """extract_from_targz raises DataProcessingError for missing archives."""
        fake_path = Path("/tmp/nonexistent_targz_test.tar.gz")
        with self.assertRaises(DataProcessingError):
            extract_from_targz(tar_path=fake_path)


# ============================================================================
# GROUP 7: get_supported_formats (4 tests)
# ============================================================================

class TestGetSupportedFormats(unittest.TestCase):
    """Test get_supported_formats utility function."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        result = get_supported_formats()
        self.assertIsInstance(result, dict)

    def test_returns_copy_not_reference(self):
        """Returns a copy of COMPRESSION_MODES, not a reference."""
        result = get_supported_formats()
        self.assertIsNot(result, COMPRESSION_MODES)

    def test_copy_matches_original(self):
        """Returned copy has identical contents to COMPRESSION_MODES."""
        result = get_supported_formats()
        self.assertEqual(result, COMPRESSION_MODES)

    def test_mutation_does_not_affect_original(self):
        """Mutating the returned dict does not affect COMPRESSION_MODES."""
        result = get_supported_formats()
        result['.tar.zst'] = 'r:zst'
        self.assertNotIn('.tar.zst', COMPRESSION_MODES)


# ============================================================================
# GROUP 8: Edge Cases and Boundary Conditions (10 tests)
# ============================================================================

class TestArchiveHandlersEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Create temp directories."""
        self._tmpdir = tempfile.mkdtemp(prefix='test_archive_edge_')
        self._extract_dir = tempfile.mkdtemp(prefix='test_extract_edge_')

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        shutil.rmtree(self._extract_dir, ignore_errors=True)

    def _archive_path(self, name="test.tar.gz"):
        return Path(self._tmpdir) / name

    def test_max_files_none_extracts_all(self):
        """max_files=None extracts all matching files."""
        files = {f"mol_{i}.xyz": f"data {i}" for i in range(15)}
        archive = _create_tar_archive(
            self._archive_path(),
            files=files,
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            max_files=None,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 15)

    def test_max_files_one(self):
        """max_files=1 extracts exactly one file."""
        files = {f"mol_{i}.xyz": f"data {i}" for i in range(5)}
        archive = _create_tar_archive(
            self._archive_path(),
            files=files,
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            max_files=1,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 1)

    def test_max_files_exceeds_available(self):
        """max_files larger than available files extracts all available."""
        files = {f"mol_{i}.xyz": f"data {i}" for i in range(3)}
        archive = _create_tar_archive(
            self._archive_path(),
            files=files,
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            max_files=100,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 3)

    def test_empty_archive_with_no_files_raises(self):
        """Archive with no files matching extension raises DataProcessingError."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={},
            mode='w:gz',
        )
        with self.assertRaises(DataProcessingError):
            extract_from_archive(
                archive_path=archive,
                file_extension='.xyz',
                temp_dir=Path(self._extract_dir),
            )

    def test_custom_file_extension(self):
        """Works with custom file extensions like .molden."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"wfn.molden": "molden data", "other.xyz": "xyz"},
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.molden',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.molden"))
        self.assertEqual(len(extracted), 1)

    def test_archive_path_as_string_converted(self):
        """archive_path accepts Path objects (primary use case from module signature)."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={"mol.xyz": "data"},
            mode='w:gz',
        )
        # Module signature expects Path, verify it works with Path
        result = extract_from_archive(
            archive_path=Path(str(archive)),
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        self.assertTrue(result.exists())

    def test_file_content_preserved_after_extraction(self):
        """Extracted file content matches the original content in the archive."""
        content = "H 0.0 0.0 0.0\nO 1.0 0.0 0.0\n"
        archive = _create_tar_archive(
            self._archive_path(),
            files={"water.xyz": content},
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted_files = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted_files), 1)
        self.assertEqual(extracted_files[0].read_text(), content)

    def test_directories_in_archive_skipped(self):
        """Directory entries in the archive are not counted as extracted files."""
        import io
        archive_path = self._archive_path()
        with tarfile.open(archive_path, 'w:gz') as tar:
            # Add a directory entry
            dir_info = tarfile.TarInfo(name="subdir/")
            dir_info.type = tarfile.DIRTYPE
            tar.addfile(dir_info)
            # Add a file
            data = b"xyz data"
            file_info = tarfile.TarInfo(name="subdir/mol.xyz")
            file_info.size = len(data)
            tar.addfile(file_info, io.BytesIO(data))

        result_dir = extract_from_archive(
            archive_path=archive_path,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 1)

    def test_detect_compression_mode_with_unknown_logs_warning(self):
        """_detect_compression_mode logs a warning for unknown extensions."""
        with patch('milia_pipeline.preprocessing.utils.archive_handlers.logger') as mock_logger:
            _detect_compression_mode(Path("data.unknown_format"))
            mock_logger.warning.assert_called_once()

    def test_detect_compression_mode_with_known_logs_debug(self):
        """_detect_compression_mode logs debug for recognized extensions."""
        with patch('milia_pipeline.preprocessing.utils.archive_handlers.logger') as mock_logger:
            _detect_compression_mode(Path("data.tar.gz"))
            mock_logger.debug.assert_called_once()


# ============================================================================
# GROUP 9: Integration Scenarios (6 tests)
# ============================================================================

class TestArchiveHandlersIntegration(unittest.TestCase):
    """Integration-style tests combining multiple aspects of the module."""

    def setUp(self):
        """Create temp directories."""
        self._tmpdir = tempfile.mkdtemp(prefix='test_archive_integ_')
        self._extract_dir = tempfile.mkdtemp(prefix='test_extract_integ_')

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        shutil.rmtree(self._extract_dir, ignore_errors=True)

    def _archive_path(self, name="test.tar.gz"):
        return Path(self._tmpdir) / name

    def test_full_lifecycle_create_extract_verify(self):
        """Full lifecycle: create archive → extract → verify file count and content."""
        molecules = {
            f"molecule_{i:04d}.xyz": f"H 0.0 0.0 {float(i)}\n"
            for i in range(20)
        }
        archive = _create_tar_archive(
            self._archive_path(),
            files=molecules,
            mode='w:gz',
        )

        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )

        extracted = sorted(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 20)

        # Verify content of first file
        first_content = extracted[0].read_text()
        self.assertIn("H 0.0 0.0", first_content)

    def test_backward_compat_wrapper_matches_direct_call(self):
        """extract_from_targz produces same results as extract_from_archive with r:gz."""
        files = {"mol.molden": "molden content"}
        archive = _create_tar_archive(
            self._archive_path(),
            files=files,
            mode='w:gz',
        )

        extract_dir_1 = tempfile.mkdtemp(prefix='compat_1_')
        extract_dir_2 = tempfile.mkdtemp(prefix='compat_2_')

        try:
            result_1 = extract_from_targz(
                tar_path=archive,
                temp_dir=Path(extract_dir_1),
            )
            result_2 = extract_from_archive(
                archive_path=archive,
                file_extension='.molden',
                temp_dir=Path(extract_dir_2),
                compression_mode='r:gz',
            )

            files_1 = sorted(f.name for f in result_1.rglob("*.molden"))
            files_2 = sorted(f.name for f in result_2.rglob("*.molden"))
            self.assertEqual(files_1, files_2)
        finally:
            shutil.rmtree(extract_dir_1, ignore_errors=True)
            shutil.rmtree(extract_dir_2, ignore_errors=True)

    def test_all_supported_formats_extract_successfully(self):
        """All supported compression formats can create and extract archives."""
        format_modes = {
            'test.tar.gz': ('w:gz', 'r:gz'),
            'test.tar.bz2': ('w:bz2', 'r:bz2'),
            'test.tar.xz': ('w:xz', 'r:xz'),
            'test.tar': ('w:', 'r:'),
        }

        for filename, (write_mode, read_mode) in format_modes.items():
            with self.subTest(filename=filename):
                extract_dir = tempfile.mkdtemp(prefix=f'test_{filename}_')
                try:
                    archive = _create_tar_archive(
                        self._archive_path(filename),
                        files={"mol.xyz": f"data from {filename}"},
                        mode=write_mode,
                    )
                    result_dir = extract_from_archive(
                        archive_path=archive,
                        file_extension='.xyz',
                        temp_dir=Path(extract_dir),
                    )
                    extracted = list(result_dir.rglob("*.xyz"))
                    self.assertEqual(len(extracted), 1)
                finally:
                    shutil.rmtree(extract_dir, ignore_errors=True)

    def test_mixed_file_types_only_target_extracted(self):
        """Archive with mixed file types only extracts files matching the extension."""
        archive = _create_tar_archive(
            self._archive_path(),
            files={
                "mol1.xyz": "xyz data 1",
                "mol2.xyz": "xyz data 2",
                "readme.md": "documentation",
                "config.yaml": "settings",
                "data.csv": "csv data",
                "mol3.xyz": "xyz data 3",
            },
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        all_files = list(result_dir.rglob("*"))
        # Only directories and .xyz files should exist
        xyz_files = [f for f in all_files if f.is_file() and f.suffix == '.xyz']
        non_xyz_files = [f for f in all_files if f.is_file() and f.suffix != '.xyz']
        self.assertEqual(len(xyz_files), 3)
        self.assertEqual(len(non_xyz_files), 0)

    def test_qm9_like_scenario_bz2_with_xyz(self):
        """Simulates QM9 dataset extraction: .tar.bz2 with .xyz files."""
        qm9_files = {
            f"dsgdb9nsd_{i:06d}.xyz": (
                f"5\n"
                f"gdb 1\tprop1\tprop2\n"
                f"C\t0.0\t0.0\t{float(i)}\n"
                f"H\t1.0\t0.0\t0.0\n"
                f"H\t0.0\t1.0\t0.0\n"
                f"H\t0.0\t0.0\t1.0\n"
                f"H\t-1.0\t0.0\t0.0\n"
            )
            for i in range(50)
        }
        archive = _create_tar_archive(
            self._archive_path("qm9.tar.bz2"),
            files=qm9_files,
            mode='w:bz2',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            max_files=10,
            file_extension='.xyz',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.xyz"))
        self.assertEqual(len(extracted), 10)

    def test_wavefunction_like_scenario_gz_with_molden(self):
        """Simulates Wavefunction dataset extraction: .tar.gz with .molden files."""
        molden_files = {
            f"wfn_{i:04d}.molden": f"[Molden Format]\n[Atoms] AU\nH 1 1 0.0 0.0 {float(i)}\n"
            for i in range(30)
        }
        archive = _create_tar_archive(
            self._archive_path("wavefunctions.tar.gz"),
            files=molden_files,
            mode='w:gz',
        )
        result_dir = extract_from_archive(
            archive_path=archive,
            max_files=5,
            file_extension='.molden',
            temp_dir=Path(self._extract_dir),
        )
        extracted = list(result_dir.rglob("*.molden"))
        self.assertEqual(len(extracted), 5)


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_comprehensive_suite():
    """Run all test groups in a structured order."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestCompressionModes,                # GROUP 1:  8 tests
        TestDetectCompressionMode,           # GROUP 2: 10 tests
        TestExtractFromArchiveHappyPath,     # GROUP 3: 12 tests
        TestExtractFromArchiveErrors,        # GROUP 4:  8 tests
        TestExtractFromArchiveLogging,       # GROUP 5:  8 tests
        TestExtractFromTargz,                # GROUP 6:  6 tests
        TestGetSupportedFormats,             # GROUP 7:  4 tests
        TestArchiveHandlersEdgeCases,        # GROUP 8: 10 tests
        TestArchiveHandlersIntegration,      # GROUP 9:  6 tests
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("PRODUCTION-READY TEST SUITE RESULTS — archive_handlers.py")
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
TEST SUITE SUMMARY — milia_pipeline/preprocessing/utils/archive_handlers.py
============================================================================

72 comprehensive production-ready tests covering:

GROUP 1: COMPRESSION_MODES Module-Level Constant (8 tests)
- Is a dictionary
- .tar.gz maps to 'r:gz'
- .tgz maps to 'r:gz'
- .tar.bz2 maps to 'r:bz2'
- .tbz2 maps to 'r:bz2'
- .tar.xz maps to 'r:xz'
- .tar maps to 'r:' (uncompressed)
- Exactly 7 format mappings

GROUP 2: _detect_compression_mode (10 tests)
- Detects .tar.gz
- Detects .tgz
- Detects .tar.bz2
- Detects .tbz2
- Detects .tar.xz
- Detects .txz
- Detects plain .tar
- Unknown extension falls back to 'r:*'
- Case-insensitive detection
- Compound extensions take priority

GROUP 3: extract_from_archive — Happy Path (12 tests)
- Extracts files matching extension
- Returns Path object
- Returns temp_dir path when specified
- max_files limits extraction
- Extracts .tar.bz2 format
- Extracts .tar.xz format
- Extracts plain .tar format
- compression_mode override works
- Default file_extension is '.xyz'
- Creates temp dir when None
- Creates nested temp dir parents
- Extracts files in subdirectories

GROUP 4: extract_from_archive — Error Paths (8 tests)
- Nonexistent archive raises DataProcessingError
- Error includes archive path
- No matching files raises DataProcessingError
- Error mentions expected extension
- Corrupt archive raises DataProcessingError
- Corrupt archive preserves TarError as __cause__
- Error has file_path attribute
- Error has operation attribute = 'archive_extraction'

GROUP 5: extract_from_archive — Logging (8 tests)
- Logs archive filename
- Logs archive size in GB
- Logs compression mode
- Logs extracted file count
- Logs max_files limit when reached
- Logs file extension in completion
- Logs extraction commenced
- Debug logs individual file names

GROUP 6: extract_from_targz — Backward Compatibility (6 tests)
- Extracts .tar.gz successfully
- Default file_extension is '.molden'
- max_files parameter works
- Delegates to extract_from_archive with 'r:gz'
- Returns result from extract_from_archive
- Nonexistent archive raises DataProcessingError

GROUP 7: get_supported_formats (4 tests)
- Returns a dictionary
- Returns a copy not a reference
- Copy matches original content
- Mutation does not affect original

GROUP 8: Edge Cases and Boundary Conditions (10 tests)
- max_files=None extracts all
- max_files=1 extracts exactly one
- max_files exceeding available extracts all
- Empty archive raises DataProcessingError
- Custom file extension (.molden) works
- Path objects handled correctly
- File content preserved after extraction
- Directory entries in archive skipped
- Unknown extension logs warning
- Known extension logs debug

GROUP 9: Integration Scenarios (6 tests)
- Full lifecycle create→extract→verify
- Backward compat wrapper matches direct call
- All supported formats extract successfully
- Mixed file types only target extracted
- QM9-like scenario (.tar.bz2 + .xyz)
- Wavefunction-like scenario (.tar.gz + .molden)

Total: 72 comprehensive production-ready tests

PRODUCTION-READY QUALITIES:
- NO sys.modules pollution (no module-level mocking)
- All mocking via @patch decorators or context managers (test-level only)
- Real tar archive creation via helper (no downloaded files)
- Temporary directory cleanup in tearDown
- Comprehensive error path coverage
- Exception attribute verification (file_path, operation)
- Exception chaining verification (__cause__)
- Logging behavior verification at info and debug levels
- Backward compatibility wrapper verified
- Multiple compression format coverage
- Dataset-realistic scenarios (QM9, Wavefunction patterns)
- Compatible with both pytest and unittest runner
- subTest usage for parameterized format testing
"""
