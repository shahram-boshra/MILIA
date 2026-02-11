"""
Archive Handlers - Compressed Archive Extraction Utilities
──────────────────────────────────────────────────────────

Memory-efficient streaming extraction for large compressed archives.
Supports multiple compression formats (gzip, bzip2, xz, uncompressed).

Author: milia Pipeline Team
Version: 1.2 (Enhanced - Dynamic compression format support)
Date: December 2025

Version History:
- 1.0: Initial implementation with tar.gz support
- 1.1: Fixed to return directory path
- 1.2: Added dynamic compression format detection (tar.gz, tar.bz2, tar.xz, tar)
"""

import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

from milia_pipeline.exceptions import DataProcessingError


logger = logging.getLogger(__name__)


# Mapping of file extensions to tarfile modes
COMPRESSION_MODES = {
    '.tar.gz': 'r:gz',
    '.tgz': 'r:gz',
    '.tar.bz2': 'r:bz2',
    '.tbz2': 'r:bz2',
    '.tar.xz': 'r:xz',
    '.txz': 'r:xz',
    '.tar': 'r:',
}


def _detect_compression_mode(tar_path: Path) -> str:
    """
    Detect the appropriate tarfile mode based on file extension.
    
    Args:
        tar_path: Path to the archive file
        
    Returns:
        Tarfile mode string (e.g., 'r:gz', 'r:bz2', 'r:xz', 'r:')
        
    Raises:
        DataProcessingError: If file extension is not recognized
    """
    filename = tar_path.name.lower()
    
    # Check for compound extensions first (e.g., .tar.gz, .tar.bz2)
    for ext, mode in COMPRESSION_MODES.items():
        if filename.endswith(ext):
            logger.debug(f"Detected compression format: {ext} -> mode {mode}")
            return mode
    
    # If no match found, try auto-detection as fallback
    logger.warning(
        f"Unknown archive extension for {tar_path.name}, "
        "attempting auto-detection with 'r:*' mode"
    )
    return 'r:*'


def extract_from_archive(
    archive_path: Path,
    max_files: Optional[int] = None,
    file_extension: str = '.xyz',
    temp_dir: Optional[Path] = None,
    compression_mode: Optional[str] = None
) -> Path:
    """
    Extract files from compressed tar archive using streaming (memory-efficient).
    
    This function extracts files WITHOUT decompressing the entire archive
    into memory, making it suitable for very large archives (e.g., 100+ GB).
    
    Supports multiple compression formats:
    - .tar.gz / .tgz (gzip)
    - .tar.bz2 / .tbz2 (bzip2)
    - .tar.xz / .txz (xz/lzma)
    - .tar (uncompressed)
    
    Args:
        archive_path: Path to compressed archive (.tar.gz, .tar.bz2, etc.)
        max_files: Maximum number of files to extract (None = all)
        file_extension: Only extract files with this extension
        temp_dir: Directory for extraction (None = system temp)
        compression_mode: Override auto-detection with specific mode
        
    Returns:
        Path to extraction directory containing extracted files
        
    Raises:
        DataProcessingError: If extraction fails
    """
    if not archive_path.exists():
        raise DataProcessingError(
            f"Archive not found: {archive_path}",
            file_path=str(archive_path),
            operation="archive_extraction"
        )
    
    # Determine compression mode
    if compression_mode is None:
        compression_mode = _detect_compression_mode(archive_path)
    
    # Create temporary extraction directory
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix='milia_extract_'))
    else:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    archive_size_gb = archive_path.stat().st_size / (1024**3)
    logger.info(f"Extracting from {archive_path.name} to {temp_dir}")
    logger.info(f"Archive size: {archive_size_gb:.2f} GB")
    logger.info(f"Compression mode: {compression_mode}")
    logger.info(f"Extraction commenced for {file_extension} files...")
    
    extracted_count = 0
    
    try:
        with tarfile.open(archive_path, compression_mode) as tar:
            for member in tar:
                if member.name.endswith(file_extension) and member.isfile():
                    tar.extract(member, path=temp_dir)
                    extracted_count += 1
                    
                    # Progress updates at DEBUG level to avoid log clutter for large archives
                    if extracted_count % 1000 == 0:
                        logger.debug(f"Extraction progress: {extracted_count} files extracted...")
                    
                    logger.debug(f"Extracted [{extracted_count}]: {member.name}")
                    
                    if max_files and extracted_count >= max_files:
                        logger.info(f"Reached max_files limit ({max_files})")
                        break
        
        if extracted_count == 0:
            raise DataProcessingError(
                f"No {file_extension} files found in archive",
                file_path=str(archive_path),
                operation="archive_extraction"
            )
        
        logger.info(f"✓ Extraction complete: {extracted_count} {file_extension} files extracted")
        return temp_dir
        
    except tarfile.TarError as e:
        raise DataProcessingError(
            f"Failed to extract archive: {e}",
            file_path=str(archive_path),
            operation="archive_extraction"
        ) from e


def get_supported_formats() -> dict:
    """Get dictionary of supported archive formats and their tarfile modes."""
    return COMPRESSION_MODES.copy()


def extract_from_targz(
    tar_path: Path,
    max_files: Optional[int] = None,
    file_extension: str = '.molden',
    temp_dir: Optional[Path] = None
) -> Path:
    """
    Extract files from tar.gz archive using streaming (memory-efficient).
    
    BACKWARD COMPATIBILITY WRAPPER: This function maintains backward compatibility
    with existing code. New code should use extract_from_archive() which supports
    multiple compression formats.
    
    Args:
        tar_path: Path to .tar.gz archive
        max_files: Maximum number of files to extract (None = all)
        file_extension: Only extract files with this extension
        temp_dir: Directory for extraction (None = system temp)
        
    Returns:
        Path to extraction directory containing extracted files
        
    Raises:
        DataProcessingError: If extraction fails
    """
    return extract_from_archive(
        archive_path=tar_path,
        max_files=max_files,
        file_extension=file_extension,
        temp_dir=temp_dir,
        compression_mode='r:gz'
    )
