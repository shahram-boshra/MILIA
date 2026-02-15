# milia_pipeline/preprocessing/utils/__init__.py

"""
Preprocessing Utilities
=======================

Shared utility functions for preprocessing operations.

Modules:
--------
- archive_handlers: Compressed archive extraction (tar.gz, tar.bz2, tar.xz, tar)
- format_parsers: Molecular file parsing (.molden)
- qm9_xyz_parser: QM9 extended XYZ file parsing
- npz_builders: NPZ file creation and validation

Author: milia Pipeline Team
Version: 1.2
Date: December 2025
"""

from milia_pipeline.preprocessing.utils.archive_handlers import (
    extract_from_archive,
    extract_from_targz,
    get_supported_formats,
)
from milia_pipeline.preprocessing.utils.format_parsers import parse_molden_files
from milia_pipeline.preprocessing.utils.npz_builders import build_npz, validate_npz_structure
from milia_pipeline.preprocessing.utils.qm9_xyz_parser import (
    get_qm9_property_info,
    parse_qm9_xyz_files,
)

__all__ = [
    # Archive handlers
    "extract_from_targz",
    "extract_from_archive",
    "get_supported_formats",
    # Format parsers
    "parse_molden_files",
    "parse_qm9_xyz_files",
    "get_qm9_property_info",
    # NPZ builders
    "build_npz",
    "validate_npz_structure",
]
