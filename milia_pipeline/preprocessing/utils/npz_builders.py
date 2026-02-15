"""
NPZ Builders - .npz File Creation Utilities
===========================================

Build compressed .npz files compatible with miliaDataset format.

Author: milia Pipeline Team
Version: 1.1 (FIXED - Accepts separate features and metadata)
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from milia_pipeline.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


def build_npz(
    features: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output_path: Path,
    logger: logging.Logger | None = None,
) -> None:
    """
    Build compressed .npz file from features and metadata.

    Creates a .npz file compatible with miliaDataset expectations:
    - Required keys: 'compounds', 'atoms', 'coordinates', 'metadata'
    - Metadata stored as numpy array of dict
    - Uses compressed format for efficiency

    Args:
        features: Dictionary mapping feature names to numpy arrays
        metadata: Dictionary with metadata about the dataset
        output_path: Path where .npz file will be created
        logger: Logger instance (uses module logger if None)

    Raises:
        DataProcessingError: If .npz creation fails

    Example:
        >>> features = {
        ...     'compounds': np.array(['mol1', 'mol2'], dtype=object),
        ...     'atoms': np.array([...], dtype=object),
        ...     'coordinates': np.array([...], dtype=object)
        ... }
        >>> metadata = {'version': '1.0', 'num_molecules': 2}
        >>> build_npz(features, metadata, Path('output.npz'))
    """
    if logger is None:
        logger = globals()["logger"]

    # Validate features
    if not features:
        raise DataProcessingError(
            "Cannot create .npz with empty features", operation="npz_creation"
        )

    # Check required keys
    required_keys = ["compounds", "atoms", "coordinates"]
    missing_keys = [key for key in required_keys if key not in features]

    if missing_keys:
        raise DataProcessingError(
            f"Missing required keys for .npz: {missing_keys}",
            operation="npz_creation",
            details=f"Available keys: {list(features.keys())}",
        )

    # Validate shapes are consistent
    num_molecules = len(features["compounds"])
    logger.info(f"Building .npz with {num_molecules} molecules")

    for key, array in features.items():
        if key in ["compounds", "atoms", "coordinates"] and len(array) != num_molecules:
            raise DataProcessingError(
                f"Shape mismatch: '{key}' has {len(array)} entries, expected {num_molecules}",
                operation="npz_creation",
            )

    # Prepare data dictionary
    npz_data = features.copy()

    # Add metadata as numpy array
    enhanced_metadata = metadata.copy()
    enhanced_metadata["num_molecules"] = num_molecules
    enhanced_metadata["feature_keys"] = list(features.keys())

    npz_data["metadata"] = np.array([enhanced_metadata], dtype=object)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with compression
    try:
        logger.info(f"Saving to: {output_path}")
        np.savez_compressed(output_path, **npz_data)

        # Verify file was created
        if not output_path.exists():
            raise DataProcessingError(f"File not created: {output_path}", operation="npz_creation")

        file_size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"✓ Created {output_path.name} ({file_size_mb:.2f} MB)")
        logger.info(f"  Molecules: {num_molecules}")
        logger.info(f"  Features: {len(features)} arrays")

        # Log feature summary
        logger.debug("Feature summary:")
        for key, array in features.items():
            if hasattr(array, "dtype"):
                logger.debug(f"  {key}: shape={array.shape}, dtype={array.dtype}")
            else:
                logger.debug(f"  {key}: length={len(array)}")

    except Exception as e:
        raise DataProcessingError(
            f"Failed to create .npz file: {e}", file_path=str(output_path), operation="npz_creation"
        ) from e


def validate_npz_structure(npz_path: Path, logger: logging.Logger | None = None) -> dict[str, Any]:
    """
    Validate .npz file structure and return summary.

    Args:
        npz_path: Path to .npz file to validate
        logger: Logger instance (uses module logger if None)

    Returns:
        Dictionary with validation results and file summary

    Raises:
        DataProcessingError: If validation fails
    """
    if logger is None:
        logger = globals()["logger"]

    if not npz_path.exists():
        raise DataProcessingError(
            f"NPZ file not found: {npz_path}", file_path=str(npz_path), operation="npz_validation"
        )

    try:
        data = np.load(npz_path, allow_pickle=True)

        # Check required keys
        required_keys = ["compounds", "metadata"]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            raise DataProcessingError(
                f"Missing required keys: {missing_keys}",
                file_path=str(npz_path),
                operation="npz_validation",
            )

        # Extract metadata
        metadata = data["metadata"][0] if len(data["metadata"]) > 0 else {}

        # Build summary
        summary = {
            "valid": True,
            "path": str(npz_path),
            "file_size_mb": npz_path.stat().st_size / (1024**2),
            "num_molecules": len(data["compounds"]),
            "num_features": len(data.files),
            "feature_keys": list(data.files),
            "metadata": dict(metadata) if metadata else {},
        }

        logger.info(f"✓ Validation passed: {npz_path.name}")
        logger.info(f"  Molecules: {summary['num_molecules']}")
        logger.info(f"  Features: {summary['num_features']}")
        logger.info(f"  Size: {summary['file_size_mb']:.2f} MB")

        return summary

    except Exception as e:
        raise DataProcessingError(
            f"NPZ validation failed: {e}", file_path=str(npz_path), operation="npz_validation"
        ) from e
