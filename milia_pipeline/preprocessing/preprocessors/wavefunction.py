"""
Wavefunction Preprocessor
=========================

Preprocessor for milia Wavefunction dataset (.molden files).

Extracts wavefunction files from tar.gz archive, parses with IOData,
and creates .npz file compatible with miliaDataset.

Author: milia Pipeline Team
Version: 1.1
Date: November 2025
"""

import logging
import shutil
from pathlib import Path

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry
from milia_pipeline.preprocessing.utils.archive_handlers import extract_from_targz
from milia_pipeline.preprocessing.utils.format_parsers import parse_molden_files
from milia_pipeline.preprocessing.utils.npz_builders import build_npz

logger = logging.getLogger(__name__)


@PreprocessorRegistry.register("Wavefunction")
class WavefunctionPreprocessor(BasePreprocessor):
    """
    Preprocessor for milia Wavefunction dataset.

    Pipeline:
    ---------
    1. Extract .molden files from tar.gz (streaming, memory-efficient)
    2. Parse .molden files with IOData (extract features)
    3. Build .npz file (compressed format)
    4. Cleanup temporary files

    Configuration:
    --------------
    Required keys:
        - raw_tar_path: Path to wavefunctions.tar.gz
        - output_npz_path: Path for output .npz file

    Optional keys:
        - num_molecules: Number of molecules to extract (None = all)
        - feature_tier: 'basic', 'standard', or 'complete'
        - cleanup_temp: Remove temporary files after processing

    Example:
    --------
    >>> config = {
    ...     'raw_tar_path': 'raw/wavefunctions.tar.gz',
    ...     'output_npz_path': 'processed/wavefunctions.npz',
    ...     'num_molecules': 100,
    ...     'feature_tier': 'standard'
    ... }
    >>> preprocessor = WavefunctionPreprocessor(config, logger)
    >>> output_path = preprocessor.run()
    """

    def _validate_config(self) -> None:
        """Validate wavefunction-specific configuration."""
        # Check required keys
        required_keys = ["raw_tar_path", "output_npz_path"]
        missing = [k for k in required_keys if k not in self.config]

        if missing:
            raise ConfigurationError(
                f"Missing required configuration keys: {missing}", config_key=", ".join(missing)
            )

        # Validate paths
        tar_path = Path(self.config["raw_tar_path"])
        if not tar_path.exists():
            raise ConfigurationError(
                f"Raw tar.gz file not found: {tar_path}",
                config_key="raw_tar_path",
                actual_value=str(tar_path),
            )

        # Validate feature tier if specified
        feature_tier = self.config.get("feature_tier", "standard")
        valid_tiers = ["basic", "standard", "complete"]
        if feature_tier not in valid_tiers:
            raise ConfigurationError(
                f"Invalid feature_tier '{feature_tier}'. Must be one of: {valid_tiers}",
                config_key="feature_tier",
                actual_value=feature_tier,
                expected_value=valid_tiers,
            )

        # Validate num_molecules if specified
        num_molecules = self.config.get("num_molecules")
        if num_molecules is not None and (not isinstance(num_molecules, int) or num_molecules < 1):
            raise ConfigurationError(
                f"num_molecules must be positive integer, got {num_molecules}",
                config_key="num_molecules",
                actual_value=num_molecules,
            )

        self.logger.debug("Configuration validation passed")

    def preprocess(self) -> Path:
        """
        Execute wavefunction preprocessing pipeline.

        Returns:
            Path to generated .npz file
        """
        tar_path = Path(self.config["raw_tar_path"])
        output_npz = Path(self.config["output_npz_path"])
        num_molecules = self.config.get("num_molecules", None)
        feature_tier = self.config.get("feature_tier", "standard")
        cleanup_temp = self.config.get("cleanup_temp", True)

        # CHECK IF OUTPUT ALREADY EXISTS (ADD THIS BLOCK)
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
            # Step 1: Extract .molden files
            self.logger.info("=" * 70)
            self.logger.info("STEP 1: Extracting .molden files from archive")
            self.logger.info("=" * 70)

            temp_dir = extract_from_targz(
                tar_path=tar_path, max_files=num_molecules, file_extension=".molden"
            )

            # Step 2: Parse .molden files
            self.logger.info("=" * 70)
            self.logger.info("STEP 2: Parsing .molden files with IOData")
            self.logger.info("=" * 70)

            features, parse_metadata = parse_molden_files(
                molden_dir=temp_dir, feature_tier=feature_tier, logger=self.logger
            )

            # Step 3: Build .npz file
            self.logger.info("=" * 70)
            self.logger.info("STEP 3: Building .npz file")
            self.logger.info("=" * 70)

            # Prepare comprehensive metadata
            npz_metadata = {
                "version": "1.1",
                "dataset_name": "milia_Wavefunction",
                "source": tar_path.name,
                "feature_tier": feature_tier,
                "file_format": ".molden",
                "parser": "IOData",
                "preprocessing_version": "1.1",
                **parse_metadata,  # Include parsing statistics
            }

            build_npz(
                features=features, metadata=npz_metadata, output_path=output_npz, logger=self.logger
            )

            # Step 4: Cleanup
            if cleanup_temp and temp_dir and temp_dir.exists():
                self.logger.info("=" * 70)
                self.logger.info("STEP 4: Cleaning up temporary files")
                self.logger.info("=" * 70)

                shutil.rmtree(temp_dir)
                self.logger.info(f"✓ Removed temporary directory: {temp_dir}")

            self.logger.info("=" * 70)
            self.logger.info("PREPROCESSING COMPLETE")
            self.logger.info("=" * 70)

            return output_npz

        except Exception as e:
            # Cleanup on error
            if cleanup_temp and temp_dir and temp_dir.exists():
                self.logger.warning(f"Cleaning up after error: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    self.logger.error(f"Cleanup failed: {cleanup_error}")

            raise DataProcessingError(
                f"Wavefunction preprocessing failed: {e}", operation="wavefunction_preprocessing"
            ) from e
