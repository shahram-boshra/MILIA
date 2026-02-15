# mol_structural_features.py

"""
Molecular Structural Feature Extraction for PyTorch Geometric

This module provides functionality to extract and encode structural features from RDKit
molecule objects and integrate them into PyTorch Geometric Data objects for graph neural
network applications.

Key Features:
- Atom-level features: degree, hybridization, valence, aromaticity, ring membership,
  partial charges, chirality, and connectivity patterns
- Bond-level features: bond type, conjugation, aromaticity, ring membership, stereochemistry,
  and 3D geometric properties (bond lengths) for QM-optimized structures
- One-hot encoding for categorical features
- 3D-aware features leveraging QM-optimized coordinates (milia compatible)
- Robust error handling with Handler-Based Pattern Development enhanced custom exceptions
- Configurable feature selection via dictionary specification

The main entry point is `add_structural_features()` which enriches PyG Data objects
with computed molecular features based on the provided configuration.

Handler-Based Pattern Development Compatibility:
- Enhanced exception handling with handler-aware error reporting
- Full compatibility with dataset handler strategy pattern
- Dataset-agnostic feature extraction suitable for any handler type
- Improved error context and recovery suggestions

Dependencies:
    - RDKit: Molecular structure processing and feature extraction
    - PyTorch: Tensor operations and data structures
    - torch_geometric: Graph neural network data format
    - NumPy: Numerical operations for 3D geometry calculations
"""

import logging
import math
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BondType, HybridizationType, rdPartialCharges
from torch_geometric.data import Data

from milia_pipeline.exceptions import (
    MoleculeProcessingError,
    PyGDataCreationError,
    StructuralFeatureError,
)

logger = logging.getLogger(__name__)


def _one_hot_encoding(value, choices: list) -> list[int]:
    """
    Generates a one-hot encoding for a given value based on a list of choices.

    Args:
        value: The value to encode.
        choices (List): A list of possible values that `value` can take.

    Returns:
        List[int]: A list of integers (0s and 1s) representing the one-hot encoding.
                   Returns all zeros if the value is not found in choices.
    """
    encoding = [0] * len(choices)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        # Value not in choices, all zeros. This can be desirable for 'unknown' or 'other'
        pass
    return encoding


# --- Preprocessing Functions for milia Integration ---
def _ensure_conformer_and_charges(
    mol: Chem.Mol, coordinates: np.ndarray | None = None, mulliken_charges: np.ndarray | None = None
) -> None:
    """
    Ensures molecule has conformer data and charges for milia integration.

    Args:
        mol: RDKit molecule
        coordinates: QM-optimized coordinates from milia (shape: [n_atoms, 3])
        mulliken_charges: Precomputed Mulliken charges from milia
    """
    # Add conformer if coordinates provided
    if coordinates is not None and len(coordinates) == mol.GetNumAtoms():
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(coordinates):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        mol.AddConformer(conf)

    # Add Mulliken charges if available (preferred over Gasteiger for milia)
    if mulliken_charges is not None and len(mulliken_charges) == mol.GetNumAtoms():
        for i, charge in enumerate(mulliken_charges):
            atom = mol.GetAtomWithIdx(i)
            atom.SetDoubleProp("_MullikenCharge", float(charge))
    else:
        # Fallback to Gasteiger charges
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            pass


# --- Atom Feature Calculation Functions ---
def _get_atom_degree(atom: Chem.Atom) -> int:
    """
    Returns the number of directly bonded neighbors (excluding implicit/explicit Hs).

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: The degree of the atom.
    """
    return atom.GetDegree()


def _get_atom_total_degree(atom: Chem.Atom) -> int:
    """
    Returns the total number of bonds to an atom (including implicit/explicit Hs).

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: The total degree of the atom.
    """
    return atom.GetTotalDegree()


def _get_atom_hybridization_feature(atom: Chem.Atom) -> list[int]:
    """
    Returns a one-hot encoding of the atom's hybridization state.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        List[int]: A one-hot encoded list representing the hybridization type.
    """
    # Common RDKit Hybridization types
    hybridization_choices = [
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        HybridizationType.UNSPECIFIED,  # Handle cases where hybridization isn't clearly defined
    ]
    return _one_hot_encoding(atom.GetHybridization(), hybridization_choices)


def _get_atom_total_valence(atom: Chem.Atom) -> int:
    """
    Returns the total valence of the atom.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: The total valence.
    """
    return atom.GetTotalValence()


def _is_atom_aromatic(atom: Chem.Atom) -> int:
    """
    Checks if the atom is aromatic.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: 1 if the atom is aromatic, 0 otherwise.
    """
    return int(atom.GetIsAromatic())


def _is_atom_in_ring(atom: Chem.Atom) -> int:
    """
    Checks if the atom is part of any ring.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: 1 if the atom is in a ring, 0 otherwise.
    """
    return int(atom.IsInRing())


def _get_atom_partial_charge(atom: Chem.Atom) -> float:
    """
    Returns the partial charge of the atom (Gasteiger charges).

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        float: The partial charge, or raises exception if invalid.

    Note:
        Requires Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        to be called on the molecule first.
    """
    if atom.HasProp("_GasteigerCharge"):
        charge = atom.GetDoubleProp("_GasteigerCharge")
        # Raise exception if invalid - will trigger feature exclusion
        if math.isnan(charge) or math.isinf(charge):
            raise ValueError(f"Invalid partial charge (NaN/Inf) for atom {atom.GetIdx()}")
        return charge
    return 0.0


def _get_atom_mulliken_charge(atom: Chem.Atom) -> float:
    """
    Returns precomputed Mulliken charge if available (preferred for milia).

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        float: The Mulliken charge, or 0.0 if not available.
    """
    if atom.HasProp("_MullikenCharge"):
        return atom.GetDoubleProp("_MullikenCharge")
    return 0.0


def _get_num_aromatic_bonds_to_atom(atom: Chem.Atom) -> int:
    """
    Returns the number of aromatic bonds connected to this atom.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        int: Number of aromatic bonds connected to the atom.
    """
    return sum(1 for bond in atom.GetBonds() if bond.GetIsAromatic())


def _get_atom_chirality_feature(atom: Chem.Atom) -> list[int]:
    """
    Returns a one-hot encoding of the atom's chirality.
    Important for pharmaceutical applications.

    Args:
        atom (Chem.Atom): The RDKit atom object.

    Returns:
        List[int]: A one-hot encoded list representing chirality type.
    """
    chirality_choices = [
        Chem.ChiralType.CHI_UNSPECIFIED,  # No chirality
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,  # R configuration
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,  # S configuration
        Chem.ChiralType.CHI_OTHER,  # Other chirality types
    ]
    return _one_hot_encoding(atom.GetChiralTag(), chirality_choices)


# --- Bond Feature Calculation Functions ---
def _get_bond_type_feature(bond: Chem.Bond) -> list[int]:
    """
    Returns a one-hot encoding of the bond's type.

    Args:
        bond (Chem.Bond): The RDKit bond object.

    Returns:
        List[int]: A one-hot encoded list representing the bond type.
    """
    # Common RDKit Bond types
    bond_type_choices = [
        BondType.SINGLE,
        BondType.DOUBLE,
        BondType.TRIPLE,
        BondType.AROMATIC,  # Aromatic bonds are often treated as a separate type
    ]
    return _one_hot_encoding(bond.GetBondType(), bond_type_choices)


def _is_bond_conjugated(bond: Chem.Bond) -> int:
    """
    Checks if the bond is conjugated.

    Args:
        bond (Chem.Bond): The RDKit bond object.

    Returns:
        int: 1 if the bond is conjugated, 0 otherwise.
    """
    return int(bond.GetIsConjugated())


def _is_bond_aromatic(bond: Chem.Bond) -> int:
    """
    Checks if the bond is aromatic.

    Args:
        bond (Chem.Bond): The RDKit bond object.

    Returns:
        int: 1 if the bond is aromatic, 0 otherwise.
    """
    return int(bond.GetIsAromatic())


def _is_bond_in_any_ring(bond: Chem.Bond) -> int:
    """
    Checks if the bond is part of any ring.

    Args:
        bond (Chem.Bond): The RDKit bond object.

    Returns:
        int: 1 if the bond is in a ring, 0 otherwise.
    """
    return int(bond.IsInRing())


def _get_bond_stereo_feature(bond: Chem.Bond) -> list[int]:
    """
    Returns a one-hot encoding of the bond's stereochemistry.

    Args:
        bond (Chem.Bond): The RDKit bond object.

    Returns:
        List[int]: A one-hot encoded list representing bond stereochemistry.
    """
    stereo_choices = [
        Chem.BondStereo.STEREONONE,  # No stereochemistry
        Chem.BondStereo.STEREOANY,  # Unknown stereochemistry
        Chem.BondStereo.STEREOZ,  # Z (cis) stereochemistry
        Chem.BondStereo.STEREOE,  # E (trans) stereochemistry
    ]
    return _one_hot_encoding(bond.GetStereo(), stereo_choices)


def _get_bond_length_3d(bond: Chem.Bond, conformer_id: int = 0) -> float:
    """
    Returns the 3D bond length using QM-optimized coordinates.
    Perfect for milia dataset with precomputed 3D structures.

    Args:
        bond (Chem.Bond): The RDKit bond object.
        conformer_id (int): ID of the conformer to use. Defaults to 0.

    Returns:
        float: Bond length in Angstroms, or 0.0 if coordinates unavailable.
    """
    mol = bond.GetOwningMol()

    # Check if conformer exists
    if mol.GetNumConformers() == 0:
        return 0.0  # No conformer available

    try:
        conf = mol.GetConformer(conformer_id)
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        pos1 = conf.GetAtomPosition(atom1_idx)
        pos2 = conf.GetAtomPosition(atom2_idx)

        # Calculate Euclidean distance
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    except Exception:
        return 0.0  # Default for missing conformer data


def _get_bond_length_binned(
    bond: Chem.Bond, conformer_id: int = 0, bin_edges: list[float] | None = None
) -> list[int]:
    """
    Returns binned bond length as one-hot encoding.
    Useful for capturing bond length ranges in a discrete manner.

    Args:
        bond (Chem.Bond): The RDKit bond object.
        conformer_id (int): ID of the conformer to use. Defaults to 0.
        bin_edges (Optional[List[float]]): Bin edges for length discretization.

    Returns:
        List[int]: One-hot encoded bond length bin.
    """
    if bin_edges is None:
        # Default bins for common bond lengths (in Angstroms)
        bin_edges = [0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, float("inf")]

    bond_length = _get_bond_length_3d(bond, conformer_id)

    # Find appropriate bin
    for i, edge in enumerate(bin_edges[1:], 1):
        if bond_length <= edge:
            bin_vector = [0] * (len(bin_edges) - 1)
            bin_vector[i - 1] = 1
            return bin_vector

    # Default to last bin if nothing matches
    bin_vector = [0] * (len(bin_edges) - 1)
    bin_vector[-1] = 1
    return bin_vector


# --- Aggregation functions for atom and bond features ---
def _calculate_atom_features_tensor(
    rdkit_mol: Chem.Mol,
    selected_features: list[str],
    molecule_index: int | None = None,
    inchi: str | None = None,
) -> torch.Tensor:
    """
    Calculates atom features based on a list of selected feature names and returns them as a PyTorch tensor.

    Args:
        rdkit_mol (Chem.Mol): The RDKit molecule object.
        selected_features (List[str]): A list of string names for the atom features to calculate.
        molecule_index (Optional[int]): The index of the molecule in the dataset, for error context. Defaults to None.
        inchi (Optional[str]): The InChI string of the molecule, for error context. Defaults to None.

    Returns:
        torch.Tensor: A tensor of atom features, where each row corresponds to an atom
                      and columns are the concatenated features. Shape: (num_atoms, feature_dim).
                      Returns an empty tensor if no atoms or no features are selected.

    Raises:
        StructuralFeatureError: If an unsupported atom feature is requested,
                                if an error occurs during an RDKit atom feature calculation,
                                or if the final tensor conversion fails.
    """


# ---
def _calculate_atom_features_tensor(
    rdkit_mol: Chem.Mol,
    selected_features: list[str],
    molecule_index: int | None = None,
    inchi: str | None = None,
) -> torch.Tensor:
    """
    Calculates atom features based on a list of selected feature names and returns them as a PyTorch tensor.

    Args:
        rdkit_mol (Chem.Mol): The RDKit molecule object.
        selected_features (List[str]): A list of string names for the atom features to calculate.
        molecule_index (Optional[int]): The index of the molecule in the dataset, for error context. Defaults to None.
        inchi (Optional[str]): The InChI string of the molecule, for error context. Defaults to None.

    Returns:
        torch.Tensor: A tensor of atom features, where each row corresponds to an atom
                      and columns are the concatenated features. Shape: (num_atoms, feature_dim).
                      Returns an empty tensor if no atoms or no features are selected.

    Raises:
        StructuralFeatureError: If an unsupported atom feature is requested,
                                if an error occurs during an RDKit atom feature calculation,
                                or if the final tensor conversion fails.
    """
    # Ensure Gasteiger charges are computed if partial_charge is requested
    if "partial_charge" in selected_features:
        first_atom = rdkit_mol.GetAtomWithIdx(0) if rdkit_mol.GetNumAtoms() > 0 else None
        if first_atom and not first_atom.HasProp("_GasteigerCharge"):
            try:
                rdPartialCharges.ComputeGasteigerCharges(rdkit_mol)
            except Exception as e:
                raise StructuralFeatureError(
                    message="Failed to compute Gasteiger charges for partial_charge feature.",
                    molecule_index=molecule_index,
                    inchi=inchi,
                    feature_type="atom",
                    feature_name="partial_charge",
                    reason="Gasteiger charge computation failed.",
                    detail=str(e),
                ) from e

    atom_feature_map = {
        "degree": _get_atom_degree,
        "total_degree": _get_atom_total_degree,
        "hybridization": _get_atom_hybridization_feature,
        "total_valence": _get_atom_total_valence,
        "is_aromatic": _is_atom_aromatic,
        "is_in_ring": _is_atom_in_ring,
        "partial_charge": _get_atom_partial_charge,
        "mulliken_charge": _get_atom_mulliken_charge,
        "num_aromatic_bonds": _get_num_aromatic_bonds_to_atom,
        "chirality": _get_atom_chirality_feature,
    }

    features_list = []
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        atom_feature_vector = []
        for feature_name in selected_features:
            if feature_name in atom_feature_map:
                try:
                    feature_value = atom_feature_map[feature_name](atom)

                    if isinstance(feature_value, list):
                        atom_feature_vector.extend(feature_value)
                    else:
                        atom_feature_vector.append(feature_value)

                except Exception as e:
                    # Catch errors from RDKit atom methods
                    raise StructuralFeatureError(
                        message=f"Error calculating '{feature_name}' for atom {i}.",
                        molecule_index=molecule_index,
                        inchi=inchi,
                        feature_type="atom",
                        feature_name=feature_name,
                        reason=f"Failed to retrieve feature value for atom {i}.",
                        detail=str(e),
                    ) from e

            else:
                # If an unknown feature is requested, we raise an error with suggestions
                available = list(atom_feature_map.keys())
                suggestions = [
                    f
                    for f in available
                    if feature_name.lower() in f.lower() or f.lower() in feature_name.lower()
                ]
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""

                raise StructuralFeatureError(
                    message=f"Unsupported atom feature requested: '{feature_name}'.{suggestion_text}",
                    molecule_index=molecule_index,
                    inchi=inchi,
                    feature_type="atom",
                    feature_name=feature_name,
                    reason="Invalid feature configuration.",
                    detail=f"Available atom features: {', '.join(available)}",
                )

        features_list.append(atom_feature_vector)

    if not features_list:
        # Handle case with no atoms or no selected features.
        return torch.empty(0, 0, dtype=torch.float)

    # Ensure all atom feature vectors have the same length (important for torch.tensor)
    max_len = max(len(vec) for vec in features_list)
    padded_features_list = [vec + [0] * (max_len - len(vec)) for vec in features_list]

    try:
        return torch.tensor(padded_features_list, dtype=torch.float)
    except Exception as e:
        raise StructuralFeatureError(
            message="Failed to convert atom features to a PyTorch tensor.",
            molecule_index=molecule_index,
            inchi=inchi,
            feature_type="atom",
            reason="Inconsistent feature vector lengths or invalid data.",
            detail=str(e),
        ) from e


def _calculate_bond_features_tensor(
    rdkit_mol: Chem.Mol,
    pyg_edge_index: torch.Tensor,
    selected_features: list[str],
    molecule_index: int | None = None,
    inchi: str | None = None,
    logger_instance: logging.Logger | None = None,
) -> torch.Tensor:
    """
    Calculates bond features based on a list of selected feature names and returns them as a PyTorch tensor.
    Ensures features align with the provided PyTorch Geometric edge_index (bidirectional).

    Args:
        rdkit_mol (Chem.Mol): The RDKit molecule object.
        pyg_edge_index (torch.Tensor): The PyTorch Geometric edge_index tensor (shape [2, num_edges]),
                                         representing graph connectivity.
        selected_features (List[str]): A list of string names for the bond features to calculate.
        molecule_index (Optional[int]): The index of the molecule in the dataset, for error context. Defaults to None.
        inchi (Optional[str]): The InChI string of the molecule, for error context. Defaults to None.
        logger_instance (Optional[logging.Logger]): Logger instance for this specific call.

    Returns:
        torch.Tensor: A tensor of bond features, where each row corresponds to an edge in `pyg_edge_index`
                      and columns are the concatenated features. Shape: (num_edges, feature_dim).
                      Returns an empty tensor if no edges or no features are selected.
                      Assigns zeros to features for PyG edges that do not correspond to an explicit RDKit bond.

    Raises:
        StructuralFeatureError: If an unsupported bond feature is requested,
                                if an error occurs during an RDKit bond feature calculation,
                                if the dummy bond creation for feature length fails,
                                or if the final tensor conversion fails.
    """
    bond_feature_map = {
        "bond_type": _get_bond_type_feature,
        "is_conjugated": _is_bond_conjugated,
        "is_aromatic": _is_bond_aromatic,
        "is_in_any_ring": _is_bond_in_any_ring,
        "stereo": _get_bond_stereo_feature,
        "bond_length": _get_bond_length_3d,
        "bond_length_binned": _get_bond_length_binned,
    }

    # Check for 3D features requiring conformer
    requires_3d = any(feat in selected_features for feat in ["bond_length", "bond_length_binned"])
    if requires_3d and rdkit_mol.GetNumConformers() == 0:
        raise StructuralFeatureError(
            message="3D bond features requested but molecule has no conformer.",
            molecule_index=molecule_index,
            inchi=inchi,
            feature_type="bond",
            reason="Missing conformer data for 3D features",
            detail="Features 'bond_length' and 'bond_length_binned' require 3D coordinates. "
            "Ensure _ensure_conformer_and_charges() was called with coordinates.",
        )

    # Validate selected features first before processing
    for feature_name in selected_features:
        if feature_name not in bond_feature_map:
            available = list(bond_feature_map.keys())
            suggestions = [
                f
                for f in available
                if feature_name.lower() in f.lower() or f.lower() in feature_name.lower()
            ]
            suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""

            raise StructuralFeatureError(
                message=f"Unsupported bond feature requested: '{feature_name}'.{suggestion_text}",
                molecule_index=molecule_index,
                inchi=inchi,
                feature_type="bond",
                feature_name=feature_name,
                reason="Invalid feature configuration.",
                detail=f"Available bond features: {', '.join(available)}",
            )

    # Determine the length of a single bond feature vector for padding
    single_bond_feature_length = 0
    if selected_features:
        try:
            # A dummy bond to get feature length. Ensure its creation is robust.
            dummy_mol = Chem.MolFromSmiles("C-C")
            if dummy_mol is None or dummy_mol.GetNumBonds() == 0:
                raise ValueError(
                    "Could not create dummy molecule for bond feature length determination."
                )

            # Add conformer to dummy if 3D features are requested
            requires_3d = any(
                feat in selected_features for feat in ["bond_length", "bond_length_binned"]
            )
            if requires_3d:
                conf = Chem.Conformer(2)
                conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
                conf.SetAtomPosition(1, (1.5, 0.0, 0.0))
                dummy_mol.AddConformer(conf)

            dummy_bond = dummy_mol.GetBondWithIdx(0)

            for feature_name in selected_features:
                val = bond_feature_map[feature_name](dummy_bond)
                single_bond_feature_length += len(val) if isinstance(val, list) else 1

                if single_bond_feature_length == 0:
                    if logger_instance:
                        logger_instance.warning(
                            f"[{molecule_index}, '{inchi}'] No valid bond features selected or they have 0 length. Defaulting to 1 for dummy feature length."
                        )
                    else:
                        logger.warning(
                            f"[{molecule_index}, '{inchi}'] No valid bond features selected or they have 0 length. Defaulting to 1 for dummy feature length."
                        )
                    single_bond_feature_length = 1

        except Exception as e:
            raise StructuralFeatureError(
                message="Failed to determine expected bond feature vector length.",
                molecule_index=molecule_index,
                inchi=inchi,
                feature_type="bond",
                reason="Error during dummy bond processing or feature map lookup.",
                detail=str(e),
            ) from e
    else:  # No bond features selected
        single_bond_feature_length = 0  # No features means 0 length

    # Create a mapping from RDKit bond (represented by sorted atom indices) to its features
    rdkit_bond_features_dict: dict[tuple[int, int], list[float]] = {}

    for i, bond in enumerate(rdkit_mol.GetBonds()):
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        bond_feature_vector = []
        for feature_name in selected_features:
            try:
                feature_value = bond_feature_map[feature_name](bond)
                if isinstance(feature_value, list):
                    bond_feature_vector.extend(feature_value)
                else:
                    bond_feature_vector.append(feature_value)
            except ValueError as e:
                # Catch conformer-related errors and re-raise with context
                if "conformer" in str(e).lower():
                    raise StructuralFeatureError(
                        message=f"3D feature '{feature_name}' failed: missing conformer.",
                        molecule_index=molecule_index,
                        inchi=inchi,
                        feature_type="bond",
                        feature_name=feature_name,
                        reason="No conformer available for 3D bond feature computation.",
                        detail=str(e),
                    ) from e
                raise
            except Exception as e:
                # Catch errors from RDKit bond methods
                raise StructuralFeatureError(
                    message=f"Error calculating '{feature_name}' for bond between atoms {u}-{v}.",
                    molecule_index=molecule_index,
                    inchi=inchi,
                    feature_type="bond",
                    feature_name=feature_name,
                    reason=f"Failed to retrieve feature value for bond {i}.",
                    detail=str(e),
                ) from e

        # Ensure the feature vector has consistent length, even if some features were skipped
        bond_feature_vector_padded = bond_feature_vector + [0] * (
            single_bond_feature_length - len(bond_feature_vector)
        )

        # Store features for both directions for easy lookup
        rdkit_bond_features_dict[(u, v)] = bond_feature_vector_padded
        rdkit_bond_features_dict[(v, u)] = (
            bond_feature_vector_padded  # For reverse direction in PyG
        )

    # Validate edge_index exists
    if pyg_edge_index is None:
        raise StructuralFeatureError(
            message="Cannot compute bond features: edge_index is None.",
            molecule_index=molecule_index,
            inchi=inchi,
            feature_type="bond",
            reason="Missing edge_index in PyG Data object",
            detail="Bond features require edge_index to map bonds to edges.",
        )

    # Now, construct the PyG edge_attr tensor using pyg_edge_index
    num_edges = pyg_edge_index.size(1)

    if num_edges == 0:
        return torch.empty(0, single_bond_feature_length, dtype=torch.float)

    # Validate bidirectional edges: for each edge (u,v), (v,u) should also exist
    edge_set = set()
    for i in range(num_edges):
        u, v = pyg_edge_index[0, i].item(), pyg_edge_index[1, i].item()
        edge_set.add((u, v))

    for u, v in list(edge_set):
        if (v, u) not in edge_set:
            raise StructuralFeatureError(
                message=f"Edge_index is not bidirectional: found ({u},{v}) but not ({v},{u}).",
                molecule_index=molecule_index,
                inchi=inchi,
                feature_type="bond",
                reason="PyG requires bidirectional edges for undirected graphs.",
                detail="Ensure edge_index contains both (u,v) and (v,u) for each bond.",
            )

    edge_features_list = []
    for i in range(num_edges):
        u, v = pyg_edge_index[0, i].item(), pyg_edge_index[1, i].item()
        features = rdkit_bond_features_dict.get((u, v))
        if features is None:
            if logger_instance:
                logger_instance.warning(
                    f"[{molecule_index}, '{inchi}'] No RDKit bond found for PyG edge ({u}, {v}). Assigning zeros to features."
                )
            else:
                logger.warning(
                    f"[{molecule_index}, '{inchi}'] No RDKit bond found for PyG edge ({u}, {v}). Assigning zeros to features."
                )
            edge_features_list.append([0] * single_bond_feature_length)
        else:
            edge_features_list.append(features)

    try:
        return torch.tensor(edge_features_list, dtype=torch.float)
    except Exception as e:
        raise StructuralFeatureError(
            message="Failed to convert bond features to a PyTorch tensor.",
            molecule_index=molecule_index,
            inchi=inchi,
            feature_type="bond",
            reason="Inconsistent feature vector lengths or invalid data.",
            detail=str(e),
        ) from e


# --- Helper function to get available features ---
def get_available_features() -> dict[str, list[str]]:
    """
    Returns a dictionary of all available atom and bond features.

    Returns:
        Dict[str, List[str]]: Dictionary with 'atom' and 'bond' keys containing
                             lists of available feature names.
    """
    return {
        "atom": [
            "degree",
            "total_degree",
            "hybridization",
            "total_valence",
            "is_aromatic",
            "is_in_ring",
            "partial_charge",
            "mulliken_charge",
            "num_aromatic_bonds",
            "chirality",
        ],
        "bond": [
            "bond_type",
            "is_conjugated",
            "is_aromatic",
            "is_in_any_ring",
            "stereo",
            "bond_length",
            "bond_length_binned",
        ],
    }


# --- Main function to add structural features to PyG Data object ---
def add_structural_features(
    rdkit_mol: Chem.Mol,
    pyg_data: Data,
    feature_config: dict[str, Any] | None,
    logger: logging.Logger,
    molecule_index: int | None = None,
    inchi: str | None = None,
    coordinates: np.ndarray | None = None,
    mulliken_charges: np.ndarray | None = None,
) -> Data:
    """
    Adds atom-level (pyg_data.x) and bond-level (pyg_data.edge_attr) structural features
    to a PyTorch Geometric Data object based on a provided feature configuration.

    This function is fully compatible with the dataset handler strategy pattern
    and provides dataset-agnostic feature extraction suitable for any handler type.

    Args:
        rdkit_mol (Chem.Mol): The RDKit molecule object from which to extract features.
        pyg_data (Data): The PyTorch Geometric Data object to enrich with features.
                         Must contain 'edge_index' if bond features are requested.
        feature_config (Dict): A dictionary specifying which features to add.
                               Expected keys: "atom" and "bond", each with a list of feature names (str).
                               Example: {"atom": ["degree", "hybridization"], "bond": ["bond_type"]}
        logger (logging.Logger): A logger instance for logging informational messages and warnings.
        molecule_index (Optional[int]): The index of the molecule in the dataset. Used for detailed error reporting.
                                        Defaults to None.
        inchi (Optional[str]): The InChI string of the molecule. Used for detailed error reporting.
                               Defaults to None.
        coordinates (Optional[np.ndarray]): QM-optimized 3D coordinates for milia integration.
                                           Shape: [n_atoms, 3]. Defaults to None.
        mulliken_charges (Optional[np.ndarray]): Precomputed Mulliken charges from milia.
                                                Shape: [n_atoms]. Defaults to None.

    Returns:
        Data: The modified PyTorch Geometric Data object with 'x' (atom features)
              and 'edge_attr' (bond features) tensors populated.
              'x' or 'edge_attr' will be None if no corresponding features are configured or applicable.

    Raises:
        MoleculeProcessingError: If the input `rdkit_mol` is None.
        PyGDataCreationError: If the input `pyg_data` is None.
        StructuralFeatureError: If any error occurs during the calculation, encoding,
                                or assignment of atom or bond features (e.g., unsupported feature names,
                                RDKit errors during feature extraction, tensor conversion issues).
                                This acts as a wrapper for more specific issues from helper functions.
    """
    if feature_config is None:
        logger.info(f"No structural features configured. Skipping for molecule {molecule_index}.")
        return pyg_data

    log_prefix = (
        f"[Mol Index: {molecule_index}, InChI: '{inchi}']" if molecule_index is not None else ""
    )

    # Validate inputs
    if rdkit_mol is None:
        raise MoleculeProcessingError(
            message="RDKit molecule object is None.",
            molecule_index=molecule_index,
            inchi=inchi,
            reason="Input 'rdkit_mol' is invalid.",
            detail="Cannot extract structural features from a non-existent molecule.",
        )
    if pyg_data is None:
        raise PyGDataCreationError(
            message="PyTorch Geometric Data object is None.",
            molecule_index=molecule_index,
            inchi=inchi,
            reason="Input 'pyg_data' is invalid.",
            detail="Cannot add structural features to a non-existent PyG Data object.",
        )

    # Ensure conformer and charges are available for milia integration
    try:
        _ensure_conformer_and_charges(rdkit_mol, coordinates, mulliken_charges)
    except Exception as e:
        logger.warning(f"{log_prefix} Failed to set up conformer or charges: {str(e)}")

    selected_atom_features = feature_config.get("atom", [])
    selected_bond_features = feature_config.get("bond", [])

    try:
        # Calculate and assign atom features (pyg_data.x)
        if selected_atom_features:
            if rdkit_mol.GetNumAtoms() == 0:
                logger.warning(
                    f"{log_prefix} RDKit molecule has no atoms. Atom features will be empty."
                )
                try:
                    dummy_mol_for_dim = Chem.MolFromSmiles("C")
                    if dummy_mol_for_dim and dummy_mol_for_dim.GetNumAtoms() > 0:
                        # Call with no context for dummy to avoid recursive context passing
                        dummy_x = _calculate_atom_features_tensor(
                            dummy_mol_for_dim, selected_atom_features
                        )
                        atom_feature_dim = dummy_x.shape[1] if dummy_x.numel() > 0 else 0
                    else:
                        atom_feature_dim = 0
                        logger.warning(
                            f"{log_prefix} Could not determine atom feature dimension from dummy molecule. Setting dimension to 0."
                        )
                    pyg_data.x = torch.empty(0, atom_feature_dim, dtype=torch.float)
                except Exception as e:
                    raise StructuralFeatureError(
                        message="Failed to determine atom feature dimension for empty molecule.",
                        molecule_index=molecule_index,
                        inchi=inchi,
                        feature_type="atom",
                        reason="Could not get feature dimension from dummy atom for empty molecule.",
                        detail=str(e),
                    ) from e
            else:
                pyg_data.x = _calculate_atom_features_tensor(
                    rdkit_mol, selected_atom_features, molecule_index=molecule_index, inchi=inchi
                )
        else:
            logger.info(
                f"{log_prefix} No atom features configured to be added. Setting pyg_data.x to None."
            )
            pyg_data.x = None

        # Calculate and assign bond features (pyg_data.edge_attr)
        if (
            selected_bond_features
            and hasattr(pyg_data, "edge_index")
            and pyg_data.edge_index is not None
            and pyg_data.edge_index.size(1) > 0
        ):
            pyg_data.edge_attr = _calculate_bond_features_tensor(
                rdkit_mol,
                pyg_data.edge_index,
                selected_bond_features,
                molecule_index=molecule_index,
                inchi=inchi,
                logger_instance=logger,
            )
        else:
            logger.info(
                f"{log_prefix} No bond features configured, no edge_index present, or no edges. Setting pyg_data.edge_attr to None."
            )
            pyg_data.edge_attr = None

    except StructuralFeatureError:
        # Re-raise custom StructuralFeatureError directly as it already contains context
        raise
    except Exception as e:
        # Catch any other unexpected exceptions and wrap them in a StructuralFeatureError
        # to provide consistent error reporting for this module.
        raise StructuralFeatureError(
            message="An unexpected error occurred while adding structural features.",
            molecule_index=molecule_index,
            inchi=inchi,
            reason="Unhandled exception during feature computation.",
            detail=str(e),
        ) from e

    return pyg_data
