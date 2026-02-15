# milia_pipeline/preprocessing/registry.py

"""
Preprocessor Registry
=====================

Auto-discovery and management of dataset preprocessors.

Author: milia Pipeline Team
Version: 1.1
Date: November 2025
"""

import logging

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class PreprocessorRegistry:
    """
    Registry for dataset preprocessors with auto-discovery.

    Uses decorator pattern for automatic registration of preprocessor classes.
    Similar to the transform registry system used elsewhere in Milia Pipeline.

    Usage:
    ------
    # Register a preprocessor
    >>> @PreprocessorRegistry.register("Wavefunction")
    ... class WavefunctionPreprocessor(BasePreprocessor):
    ...     pass

    # Get a registered preprocessor
    >>> PreprocessorClass = PreprocessorRegistry.get_preprocessor("Wavefunction")
    >>> preprocessor = PreprocessorClass(config, logger)
    """

    _preprocessors: dict[str, type[BasePreprocessor]] = {}

    @classmethod
    def register(cls, dataset_type: str):
        """
        Decorator to register a preprocessor class.

        Args:
            dataset_type: Name/type of dataset this preprocessor handles

        Returns:
            Decorator function

        Example:
            >>> @PreprocessorRegistry.register("Wavefunction")
            ... class WavefunctionPreprocessor(BasePreprocessor):
            ...     pass
        """

        def decorator(preprocessor_class: type[BasePreprocessor]):
            if not issubclass(preprocessor_class, BasePreprocessor):
                raise ConfigurationError(
                    f"Preprocessor {preprocessor_class.__name__} must inherit from BasePreprocessor"
                )

            if dataset_type in cls._preprocessors:
                logger.warning(
                    f"Preprocessor '{dataset_type}' already registered, "
                    f"overwriting with {preprocessor_class.__name__}"
                )

            cls._preprocessors[dataset_type] = preprocessor_class
            logger.debug(
                f"Registered preprocessor: {dataset_type} -> {preprocessor_class.__name__}"
            )

            return preprocessor_class

        return decorator

    @classmethod
    def get_preprocessor(cls, dataset_type: str) -> type[BasePreprocessor]:
        """
        Get a registered preprocessor class by dataset type.

        Args:
            dataset_type: Name/type of dataset

        Returns:
            Preprocessor class

        Raises:
            DataProcessingError: If preprocessor not found

        PHASE 6.2 SIMPLIFICATION: Primary callers now receive normalized dataset_type
        from config_loader.py. Case-insensitive fallback kept as defensive measure
        for direct API calls that may bypass config loading.
        """
        # Primary path: exact match (config_loader normalizes at load time)
        if dataset_type in cls._preprocessors:
            return cls._preprocessors[dataset_type]

        # Defensive fallback: case-insensitive match for direct API calls
        dataset_type_upper = dataset_type.upper()
        for key, preprocessor_class in cls._preprocessors.items():
            if key.upper() == dataset_type_upper:
                logger.debug(
                    f"Preprocessor lookup used case-insensitive fallback: '{dataset_type}' -> '{key}'"
                )
                return preprocessor_class

        # Not found
        available = list(cls._preprocessors.keys())
        raise DataProcessingError(
            f"No preprocessor registered for dataset type '{dataset_type}'. Available: {available}"
        )

    @classmethod
    def list_preprocessors(cls) -> list[str]:
        """
        List all registered preprocessor types.

        Returns:
            List of registered dataset types
        """
        return list(cls._preprocessors.keys())

    @classmethod
    def supports_preprocessing(cls, dataset_type: str) -> bool:
        """
        Check if preprocessing is supported for a dataset type.

        Args:
            dataset_type: Name/type of dataset

        Returns:
            True if preprocessor exists for this dataset type

        PHASE 6.2 SIMPLIFICATION: Primary callers now receive normalized dataset_type
        from config_loader.py. Case-insensitive fallback kept as defensive measure
        for direct API calls that may bypass config loading.
        """
        # Primary path: exact match (config_loader normalizes at load time)
        if dataset_type in cls._preprocessors:
            return True
        # Defensive fallback: case-insensitive match for direct API calls
        dataset_type_upper = dataset_type.upper()
        return any(k.upper() == dataset_type_upper for k in cls._preprocessors.keys())

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered preprocessors.

        Useful for testing purposes.
        """
        cls._preprocessors.clear()
        logger.debug("Preprocessor registry cleared")
