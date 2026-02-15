"""
Example Descriptor Plugin for milia Pipeline

Demonstrates best practices for implementing custom molecular descriptors.

Author: Milia Team
License: MIT
"""

import logging

from rdkit import Chem

logger = logging.getLogger(__name__)


def calculate_aromatic_ratio(mol: Chem.Mol) -> float:
    """
    Calculate the ratio of aromatic atoms to total atoms.

    This descriptor measures the degree of aromaticity in the molecule.
    A value of 0.0 means no aromatic atoms, 1.0 means all atoms are aromatic.

    Args:
        mol: RDKit Mol object

    Returns:
        float: Ratio of aromatic atoms to total atoms (0.0 to 1.0)
               Returns NaN if calculation fails

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        >>> ratio = calculate_aromatic_ratio(mol)
        >>> print(f"{ratio:.2f}")  # Should be 1.0
        1.00

        >>> mol = Chem.MolFromSmiles("CCO")  # Ethanol
        >>> ratio = calculate_aromatic_ratio(mol)
        >>> print(f"{ratio:.2f}")  # Should be 0.0
        0.00
    """
    try:
        if mol is None:
            return float("nan")

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return 0.0

        aromatic_count = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

        return aromatic_count / num_atoms

    except Exception as e:
        logger.warning(f"Failed to calculate aromatic ratio: {e}")
        return float("nan")


def calculate_heteroatom_ratio(mol: Chem.Mol) -> float:
    """
    Calculate the ratio of heteroatoms to total atoms.

    Heteroatoms are non-carbon, non-hydrogen atoms (N, O, S, P, etc.).
    This descriptor is useful for assessing functional group content.

    Args:
        mol: RDKit Mol object

    Returns:
        float: Ratio of heteroatoms to total atoms (0.0 to 1.0)
               Returns NaN if calculation fails

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")  # Ethanol
        >>> ratio = calculate_heteroatom_ratio(mol)
        >>> # 1 oxygen out of 3 heavy atoms = 0.33
    """
    try:
        if mol is None:
            return float("nan")

        num_heavy_atoms = mol.GetNumHeavyAtoms()
        if num_heavy_atoms == 0:
            return 0.0

        heteroatom_count = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() not in [1, 6]  # Not H or C
        )

        return heteroatom_count / num_heavy_atoms

    except Exception as e:
        logger.warning(f"Failed to calculate heteroatom ratio: {e}")
        return float("nan")


def calculate_chain_length(mol: Chem.Mol) -> float:
    """
    Calculate the length of the longest carbon chain.

    Useful for distinguishing linear from branched structures.

    Args:
        mol: RDKit Mol object

    Returns:
        float: Length of longest carbon chain
               Returns NaN if calculation fails

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCCCCC")  # Hexane
        >>> length = calculate_chain_length(mol)
        >>> print(int(length))  # Should be 6
        6
    """
    try:
        if mol is None:
            return float("nan")

        # Find all carbon atoms
        carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]

        if not carbon_atoms:
            return 0.0

        # Find longest path between carbon atoms
        max_length = 0
        for start_idx in carbon_atoms:
            distances = Chem.GetDistanceMatrix(mol)[start_idx]
            for end_idx in carbon_atoms:
                if start_idx != end_idx:
                    max_length = max(max_length, int(distances[end_idx]))

        return float(max_length + 1)  # +1 because distance is edges, we want nodes

    except Exception as e:
        logger.warning(f"Failed to calculate chain length: {e}")
        return float("nan")


# Bonus descriptor not declared in plugin.yaml (will be auto-discovered)
def calculate_ring_complexity(mol: Chem.Mol) -> float:
    """
    Calculate ring system complexity.

    This is a bonus descriptor that will be discovered even though
    it's not declared in plugin.yaml.

    Args:
        mol: RDKit Mol object

    Returns:
        float: Ring complexity score
    """
    try:
        if mol is None:
            return float("nan")

        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()

        if num_rings == 0:
            return 0.0

        # Simple complexity: weighted sum of ring sizes
        complexity = 0.0
        for ring in ring_info.AtomRings():
            ring_size = len(ring)
            # Larger rings contribute more to complexity
            complexity += ring_size * 0.5

        return complexity

    except Exception as e:
        logger.warning(f"Failed to calculate ring complexity: {e}")
        return float("nan")
