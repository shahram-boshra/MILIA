# milia_pipeline/preprocessing/preprocessors/qm9.py

"""
QM9 Preprocessor
================

Preprocessor for QM9 quantum chemistry dataset (tar.bz2 archive of XYZ files).

Extracts XYZ files from tar.bz2 archive, parses QM9's extended XYZ format,
and creates .npz file compatible with miliaDataset.

QM9 Dataset Information:
------------------------
- Source: Figshare (https://figshare.com/ndownloader/files/3195389)
- Archive: dsgdb9nsd.xyz.tar.bz2 (bzip2 compressed)
- Contents: 133,885 individual .xyz files
- Format: Extended XYZ with properties in comment line

Author: milia Pipeline Team
Version: 1.0
Date: December 2025
"""

import logging
import shutil
from pathlib import Path

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry
from milia_pipeline.preprocessing.utils.archive_handlers import extract_from_archive
from milia_pipeline.preprocessing.utils.npz_builders import build_npz
from milia_pipeline.preprocessing.utils.qm9_xyz_parser import parse_qm9_xyz_files

logger = logging.getLogger(__name__)


@PreprocessorRegistry.register("QM9")
class QM9Preprocessor(BasePreprocessor):
    """
    Preprocessor for QM9 quantum chemistry dataset.

    Pipeline:
    ---------
    1. Extract .xyz files from tar.bz2 archive (streaming, memory-efficient)
    2. Parse QM9 extended XYZ files (extract all 15 properties + atoms + coords)
    3. Build .npz file (compressed format compatible with miliaDataset)
    4. Cleanup temporary files

    Configuration:
    --------------
    Required keys:
        - raw_archive_path: Path to dsgdb9nsd.xyz.tar.bz2
        - output_npz_path: Path for output .npz file

    Optional keys:
        - num_molecules: Number of molecules to extract (None = all 133,885)
        - cleanup_temp: Remove temporary files after processing (default: True)

    Example:
    --------
    >>> config = {
    ...     'raw_archive_path': 'raw/dsgdb9nsd.xyz.tar.bz2',
    ...     'output_npz_path': 'processed/qm9.npz',
    ...     'num_molecules': 1000,  # For testing
    ... }
    >>> preprocessor = QM9Preprocessor(config, logger)
    >>> output_path = preprocessor.run()

    Notes:
    ------
    - The QM9 archive is approximately 200 MB compressed
    - Full preprocessing takes approximately 5-10 minutes for all 133,885 molecules
    - Use num_molecules for testing with a subset
    - Output NPZ will be ~500 MB for the full dataset
    """

    def _validate_config(self) -> None:
        """Validate QM9-specific configuration."""
        # Check required keys
        required_keys = ["raw_archive_path", "output_npz_path"]
        missing = [k for k in required_keys if k not in self.config]

        if missing:
            raise ConfigurationError(
                f"Missing required configuration keys: {missing}", config_key=", ".join(missing)
            )

        # Validate archive path exists
        archive_path = Path(self.config["raw_archive_path"])
        if not archive_path.exists():
            raise ConfigurationError(
                f"QM9 archive file not found: {archive_path}",
                config_key="raw_archive_path",
                actual_value=str(archive_path),
            )

        # Validate archive extension (should be .tar.bz2 or similar)
        valid_extensions = [".tar.bz2", ".tbz2", ".tar.gz", ".tgz"]
        if not any(str(archive_path).lower().endswith(ext) for ext in valid_extensions):
            self.logger.warning(
                f"Archive extension not recognized: {archive_path.suffix}. "
                f"Expected one of {valid_extensions}. Proceeding with auto-detection."
            )

        # Validate num_molecules if specified
        num_molecules = self.config.get("num_molecules")
        if num_molecules is not None and (not isinstance(num_molecules, int) or num_molecules < 1):
            raise ConfigurationError(
                f"num_molecules must be positive integer, got {num_molecules}",
                config_key="num_molecules",
                actual_value=num_molecules,
            )

        self.logger.debug("QM9 configuration validation passed")

    def preprocess(self) -> Path:
        """
        Execute QM9 preprocessing pipeline.

        Returns:
            Path to generated .npz file
        """
        archive_path = Path(self.config["raw_archive_path"])
        output_npz = Path(self.config["output_npz_path"])
        num_molecules = self.config.get("num_molecules", None)
        cleanup_temp = self.config.get("cleanup_temp", True)

        # Check if output already exists (skip if so)
        if output_npz.exists():
            self.logger.info("=" * 70)
            self.logger.info("EXISTING .NPZ FILE DETECTED")
            self.logger.info("=" * 70)
            size_mb = output_npz.stat().st_size / (1024**2)
            self.logger.info(f"Found: {output_npz.name} ({size_mb:.2f} MB)")
            self.logger.info("Skipping preprocessing - file already exists")
            self.logger.info("Delete the file to regenerate, or use a different output path")
            self.logger.info("=" * 70)
            return output_npz

        temp_dir = None

        try:
            # ================================================================
            # STEP 1: Extract .xyz files from archive
            # ================================================================
            self.logger.info("=" * 70)
            self.logger.info("STEP 1: Extracting .xyz files from archive")
            self.logger.info("=" * 70)

            # Use the generic archive extractor which auto-detects compression
            temp_dir = extract_from_archive(
                archive_path=archive_path, max_files=num_molecules, file_extension=".xyz"
            )

            # ================================================================
            # STEP 2: Parse QM9 XYZ files
            # ================================================================
            self.logger.info("=" * 70)
            self.logger.info("STEP 2: Parsing QM9 XYZ files")
            self.logger.info("=" * 70)

            features, parse_metadata = parse_qm9_xyz_files(
                xyz_dir=temp_dir, max_molecules=num_molecules, logger=self.logger
            )

            # ================================================================
            # STEP 3: Build .npz file
            # ================================================================
            self.logger.info("=" * 70)
            self.logger.info("STEP 3: Building .npz file")
            self.logger.info("=" * 70)

            # Prepare comprehensive metadata
            npz_metadata = {
                "version": "1.0",
                "dataset_name": "QM9",
                "source": archive_path.name,
                "source_url": "https://figshare.com/ndownloader/files/3195389",
                "reference": "Ramakrishnan et al., Scientific Data 1, 140022 (2014)",
                "doi": "10.1038/sdata.2014.22",
                "file_format": ".xyz (extended QM9 format)",
                "parser": "qm9_xyz_parser",
                "preprocessing_version": "1.0",
                "coordinate_units": "angstrom",
                "energy_units": "hartree",
                **parse_metadata,  # Include parsing statistics
            }

            build_npz(
                features=features, metadata=npz_metadata, output_path=output_npz, logger=self.logger
            )

            # ================================================================
            # STEP 4: Cleanup
            # ================================================================
            if cleanup_temp and temp_dir and temp_dir.exists():
                self.logger.info("=" * 70)
                self.logger.info("STEP 4: Cleaning up temporary files")
                self.logger.info("=" * 70)

                shutil.rmtree(temp_dir)
                self.logger.info(f"✓ Removed temporary directory: {temp_dir}")

            self.logger.info("=" * 70)
            self.logger.info("QM9 PREPROCESSING COMPLETE")
            self.logger.info("=" * 70)

            return output_npz

        except Exception as e:
            # Cleanup on error
            if cleanup_temp and temp_dir and Path(temp_dir).exists():
                self.logger.warning(f"Cleaning up after error: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    self.logger.error(f"Cleanup failed: {cleanup_error}")

            raise DataProcessingError(
                f"QM9 preprocessing failed: {e}", operation="qm9_preprocessing"
            ) from e
