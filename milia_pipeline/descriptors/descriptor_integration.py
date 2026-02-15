# milia_pipeline/descriptors/descriptor_integration.py

"""
Integration helpers for adding descriptors to PyG Data objects.

This module provides utilities for converting descriptor dictionaries to tensors,
adding them to PyG Data objects, and merging with existing features.
"""

import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def descriptors_to_tensor(
    descriptor_dict: dict[str, float],
    descriptor_order: list[str] | None = None,
    fill_missing: float = 0.0,
) -> torch.Tensor:
    """
    Convert descriptor dictionary to PyTorch tensor.

    Args:
        descriptor_dict: Dictionary mapping descriptor names to values
        descriptor_order: Optional list defining descriptor order in tensor
        fill_missing: Value to use for missing descriptors

    Returns:
        1D tensor of descriptor values

    Example:
        >>> desc_dict = {'MolWt': 180.16, 'TPSA': 40.46, 'LogP': 2.5}
        >>> tensor = descriptors_to_tensor(desc_dict)
        >>> tensor.shape
        torch.Size([3])
    """
    if not descriptor_dict:
        return torch.tensor([], dtype=torch.float32)

    if descriptor_order is None:
        # Use alphabetical order for consistency
        descriptor_order = sorted(descriptor_dict.keys())

    # Build tensor with specified order
    values = []
    for desc_name in descriptor_order:
        value = descriptor_dict.get(desc_name, fill_missing)
        values.append(float(value))

    return torch.tensor(values, dtype=torch.float32)


def add_descriptors_to_pyg_data(
    data: Data,
    descriptors: dict[str, float],
    prefix: str = "desc_",
    as_dict: bool = False,
    create_feature_vector: bool = True,
) -> Data:
    """
    Add descriptors to PyG Data object.

    Args:
        data: PyG Data object
        descriptors: Dictionary of descriptor name -> value
        prefix: Prefix for descriptor attribute names
        as_dict: Store as dictionary instead of individual attributes
        create_feature_vector: Create unified descriptor feature vector

    Returns:
        Modified PyG Data object with descriptors

    Example:
        >>> data = Data(x=node_features, edge_index=edges)
        >>> desc = {'MolWt': 180.16, 'TPSA': 40.46}
        >>> data = add_descriptors_to_pyg_data(data, desc)
        >>> hasattr(data, 'desc_MolWt')
        True
        >>> hasattr(data, 'descriptor_features')
        True
    """
    if not descriptors:
        logger.warning("No descriptors provided to add to PyG Data")
        return data

    # Store as dictionary if requested
    if as_dict:
        data.descriptors = descriptors
    else:
        # Store individual descriptors as attributes
        for desc_name, value in descriptors.items():
            attr_name = f"{prefix}{desc_name}"
            setattr(data, attr_name, torch.tensor([value], dtype=torch.float32))

    # Create unified feature vector
    if create_feature_vector:
        descriptor_order = sorted(descriptors.keys())
        descriptor_tensor = descriptors_to_tensor(descriptors, descriptor_order)
        data.descriptor_features = descriptor_tensor
        data.descriptor_names = descriptor_order  # Track order

    # Add metadata
    data.num_descriptors = len(descriptors)

    return data


def merge_descriptors_with_features(
    data: Data,
    descriptors: dict[str, float],
    feature_attr: str = "x",
    descriptor_order: list[str] | None = None,
) -> Data:
    """
    Merge descriptors with existing node/graph features.

    This function broadcasts graph-level descriptors to all nodes and
    concatenates them with existing node features.

    Args:
        data: PyG Data object with existing features
        descriptors: Dictionary of descriptor name -> value
        feature_attr: Attribute name of features to merge with (default: 'x')
        descriptor_order: Optional list defining descriptor order

    Returns:
        Modified PyG Data object with merged features

    Example:
        >>> data = Data(x=torch.randn(10, 5))  # 10 nodes, 5 features each
        >>> desc = {'MolWt': 180.16, 'TPSA': 40.46}
        >>> data = merge_descriptors_with_features(data, desc)
        >>> data.x.shape
        torch.Size([10, 7])  # 5 original + 2 descriptor features
    """
    if not hasattr(data, feature_attr):
        logger.warning(f"Data object has no '{feature_attr}' attribute to merge with")
        return add_descriptors_to_pyg_data(data, descriptors)

    if not descriptors:
        logger.warning("No descriptors to merge")
        return data

    # Get existing features
    existing_features = getattr(data, feature_attr)
    num_nodes = existing_features.shape[0]

    # Convert descriptors to tensor
    desc_tensor = descriptors_to_tensor(descriptors, descriptor_order)

    # Broadcast descriptor tensor to all nodes
    # Shape: (num_nodes, num_descriptors)
    desc_broadcasted = desc_tensor.unsqueeze(0).expand(num_nodes, -1)

    # Concatenate with existing features
    merged_features = torch.cat([existing_features, desc_broadcasted], dim=1)

    # Update data object
    setattr(data, feature_attr, merged_features)

    # Track descriptor contribution
    data.descriptor_dim = desc_tensor.shape[0]
    if descriptor_order is None:
        descriptor_order = sorted(descriptors.keys())
    data.descriptor_names_merged = descriptor_order

    return data


def extract_descriptors_from_pyg_data(
    data: Data, prefix: str = "desc_", from_dict: bool = False
) -> dict[str, float]:
    """
    Extract descriptors from PyG Data object.

    Args:
        data: PyG Data object with descriptors
        prefix: Prefix used when storing descriptors
        from_dict: Extract from 'descriptors' dict attribute

    Returns:
        Dictionary of descriptor name -> value

    Example:
        >>> # Assuming data has descriptors
        >>> desc = extract_descriptors_from_pyg_data(data)
        >>> print(desc)
        {'MolWt': 180.16, 'TPSA': 40.46, 'LogP': 2.5}
    """
    if from_dict and hasattr(data, "descriptors"):
        return data.descriptors.copy()

    # Extract from individual attributes
    descriptors = {}
    for attr_name in dir(data):
        if attr_name.startswith(prefix):
            desc_name = attr_name[len(prefix) :]
            value = getattr(data, attr_name)
            if isinstance(value, torch.Tensor):
                value = value.item()
            descriptors[desc_name] = float(value)

    return descriptors


def validate_descriptor_integration(
    data: Data, expected_descriptors: list[str] | None = None, check_feature_vector: bool = True
) -> tuple[bool, list[str]]:
    """
    Validate that descriptors were properly integrated into PyG Data.

    Args:
        data: PyG Data object to validate
        expected_descriptors: Optional list of expected descriptor names
        check_feature_vector: Verify unified feature vector exists

    Returns:
        Tuple of (is_valid, list_of_issues)

    Example:
        >>> is_valid, issues = validate_descriptor_integration(data, ['MolWt', 'TPSA'])
        >>> if not is_valid:
        ...     print("Validation issues:", issues)
    """
    issues = []

    # Check if descriptors exist
    if not hasattr(data, "num_descriptors") and not hasattr(data, "descriptors"):
        issues.append("No descriptors found in Data object")
        return False, issues

    # Check feature vector if requested
    if check_feature_vector:
        if not hasattr(data, "descriptor_features"):
            issues.append("Missing 'descriptor_features' attribute")
        elif not hasattr(data, "descriptor_names"):
            issues.append("Missing 'descriptor_names' attribute")

    # Check expected descriptors
    if expected_descriptors:
        if hasattr(data, "descriptors"):
            actual_descriptors = set(data.descriptors.keys())
        else:
            actual_descriptors = set()
            for attr in dir(data):
                if attr.startswith("desc_"):
                    actual_descriptors.add(attr[5:])

        expected_set = set(expected_descriptors)
        missing = expected_set - actual_descriptors
        extra = actual_descriptors - expected_set

        if missing:
            issues.append(f"Missing expected descriptors: {sorted(missing)}")
        if extra:
            issues.append(f"Unexpected descriptors found: {sorted(extra)}")

    is_valid = len(issues) == 0
    return is_valid, issues


def get_descriptor_statistics(data_list: list[Data]) -> dict[str, Any]:
    """
    Compute statistics about descriptors across a dataset.

    Args:
        data_list: List of PyG Data objects with descriptors

    Returns:
        Dictionary with descriptor statistics

    Example:
        >>> stats = get_descriptor_statistics(dataset)
        >>> print(stats['total_molecules'])
        1000
        >>> print(stats['avg_descriptors_per_molecule'])
        25.5
    """
    total_molecules = len(data_list)
    descriptor_counts = []
    all_descriptor_names = set()

    for data in data_list:
        if hasattr(data, "num_descriptors"):
            descriptor_counts.append(data.num_descriptors)

        if hasattr(data, "descriptor_names"):
            all_descriptor_names.update(data.descriptor_names)
        elif hasattr(data, "descriptors"):
            all_descriptor_names.update(data.descriptors.keys())

    if not descriptor_counts:
        return {
            "total_molecules": total_molecules,
            "molecules_with_descriptors": 0,
            "avg_descriptors_per_molecule": 0.0,
            "unique_descriptors": 0,
        }

    return {
        "total_molecules": total_molecules,
        "molecules_with_descriptors": len(descriptor_counts),
        "avg_descriptors_per_molecule": np.mean(descriptor_counts),
        "min_descriptors": min(descriptor_counts),
        "max_descriptors": max(descriptor_counts),
        "unique_descriptors": len(all_descriptor_names),
        "descriptor_names": sorted(all_descriptor_names),
    }
