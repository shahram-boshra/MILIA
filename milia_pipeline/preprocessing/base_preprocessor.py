# milia_pipeline/preprocessing/base_preprocessor.py

"""
Base Preprocessor Abstract Class
=================================

Defines the contract for all dataset preprocessors in the milia Pipeline.
All preprocessors must inherit from this base class and implement required methods.

Author: milia Pipeline Team
Version: 1.1
Date: November 2025
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from milia_pipeline.exceptions import DataProcessingError


class BasePreprocessor(ABC):
    """
    Abstract base class for dataset preprocessors.

    Preprocessors handle one-time transformation of raw data files
    into the .npz format expected by miliaDataset. This is an OFFLINE
    operation that happens before dataset creation.

    Architecture Pattern:
    --------------------
    1. Validate inputs (raw files exist, config valid)
    2. Execute preprocessing (dataset-specific transformation)
    3. Validate output (.npz file structure and content)
    4. Return path to generated .npz file

    Usage:
    ------
    >>> from milia_pipeline.preprocessing.registry import PreprocessorRegistry
    >>> PreprocessorClass = PreprocessorRegistry.get_preprocessor("Wavefunction")
    >>> preprocessor = PreprocessorClass(config=config, logger=logger)
    >>> output_path = preprocessor.run()
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Preprocessing configuration dictionary
            logger: Logger instance for output
        """
        self.config = config
        self.logger = logger
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate preprocessor-specific configuration.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass

    @abstractmethod
    def preprocess(self) -> Path:
        """
        Execute preprocessing logic.

        Returns:
            Path to generated .npz file

        Raises:
            DataProcessingError: If preprocessing fails
        """
        pass

    def run(self) -> Path:
        """
        Execute full preprocessing pipeline with validation.

        Returns:
            Path to validated .npz output file
        """
        start_time = time.time()
        self.logger.info(f"Starting {self.__class__.__name__}")

        try:
            # Execute preprocessing
            output_path = self.preprocess()

            # Validate output
            self._validate_output(output_path)

            elapsed = time.time() - start_time
            self.logger.info(f"Preprocessing complete: {output_path} ({elapsed:.2f}s)")

            return output_path

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise DataProcessingError(f"Preprocessing error: {e}") from e

    def _validate_output(self, output_path: Path) -> None:
        """
        Validate generated .npz file structure.

        Args:
            output_path: Path to .npz file

        Raises:
            DataProcessingError: If output validation fails
        """
        if not output_path.exists():
            raise DataProcessingError(f"Output file not created: {output_path}")

        # Load and validate .npz structure
        try:
            data = np.load(output_path, allow_pickle=True)

            # Check required keys exist
            required_keys = ["compounds", "metadata"]
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise DataProcessingError(f"Missing required keys in .npz: {missing}")

            self.logger.info(f"✓ Output validated: {len(data['compounds'])} molecules")

        except Exception as e:
            raise DataProcessingError(f"Output validation failed: {e}") from e
