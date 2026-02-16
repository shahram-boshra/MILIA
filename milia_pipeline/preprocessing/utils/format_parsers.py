"""
Format Parsers - Molecular File Parsing Utilities
=================================================

Parse quantum chemistry file formats (.molden) and extract features.

Author: milia Pipeline Team
Version: 1.1 (FIXED - Returns tuple of (features, metadata))
Date: November 2025
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Try to import LoadWarning, fallback to generic Warning
try:
    from iodata.api import LoadWarning
except ImportError:
    LoadWarning = Warning  # Fallback to generic Warning class

from milia_pipeline.exceptions import DataProcessingError, MissingDependencyError

logger = logging.getLogger(__name__)


# Feature tier definitions - MUST match actual features extracted in _extract_molecule_features()
# These definitions are the single source of truth for tier-aware validation across the pipeline.
FEATURE_TIERS = {
    "basic": [
        # Always extracted (core molecular data)
        "compounds",
        "atoms",
        "coordinates",
        "n_atoms",
        "n_electrons",
        # MO data (extracted for ALL tiers when available)
        "mo_energies",
        "mo_occupations",
        # HOMO/LUMO features (extracted for ALL tiers)
        "homo_energy_eV",
        "homo_index",
        "lumo_energy_eV",
        "lumo_index",
        "homo_lumo_gap_eV",
    ],
    "standard": [
        # All basic tier features
        "compounds",
        "atoms",
        "coordinates",
        "n_atoms",
        "n_electrons",
        "mo_energies",
        "mo_occupations",
        "homo_energy_eV",
        "homo_index",
        "lumo_energy_eV",
        "lumo_index",
        "homo_lumo_gap_eV",
        # Standard tier additions
        "total_energy_eV",
        "total_energy_Hartree",
        "molecular_formula",
        "molecular_weight",
    ],
    "complete": [
        # All standard tier features
        "compounds",
        "atoms",
        "coordinates",
        "n_atoms",
        "n_electrons",
        "mo_energies",
        "mo_occupations",
        "homo_energy_eV",
        "homo_index",
        "lumo_energy_eV",
        "lumo_index",
        "homo_lumo_gap_eV",
        "total_energy_eV",
        "total_energy_Hartree",
        "molecular_formula",
        "molecular_weight",
        # Complete tier additions - MO coefficients and statistics
        "mo_coefficients",
        "mo_kind",
        "n_basis_functions",
        "mo_energy_mean_eV",
        "mo_energy_std_eV",
        "mo_energy_min_eV",
        "mo_energy_max_eV",
        "n_occupied_orbitals",
        "n_virtual_orbitals",
        "n_shells",
        # Complete tier additions - Derived quantum descriptors
        "ionization_potential_eV",
        "electron_affinity_eV",
        "chemical_hardness_eV",
        "chemical_potential_eV",
        "electrophilicity_eV",
    ],
}


def parse_molden_files(
    molden_dir: Path, feature_tier: str = "standard", logger: logging.Logger | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Parse all .molden files in a directory and extract features.

    Args:
        molden_dir: Directory containing .molden files
        feature_tier: Feature extraction level ('basic', 'standard', 'complete')
        logger: Logger instance (uses module logger if None)

    Returns:
        Tuple of (features_dict, metadata_dict) where:
            - features_dict: Dict mapping feature names to numpy arrays
            - metadata_dict: Dict with parsing metadata

    Raises:
        DataProcessingError: If parsing fails
        MissingDependencyError: If iodata not available

    Example:
        >>> molden_dir = Path("/tmp/molden_files")
        >>> features, metadata = parse_molden_files(molden_dir, 'standard')
        >>> print(features.keys())
        dict_keys(['compounds', 'atoms', 'coordinates', ...])
    """
    if logger is None:
        logger = globals()["logger"]

    # Validate feature tier
    if feature_tier not in FEATURE_TIERS:
        raise DataProcessingError(
            f"Invalid feature tier '{feature_tier}'. Available: {list(FEATURE_TIERS.keys())}"
        )

    # Check for iodata
    try:
        from iodata import load_one
    except ImportError as e:
        raise MissingDependencyError(
            "iodata library required for .molden parsing",
            missing_dependency="iodata>=1.0.0",
            install_command="pip install iodata",
        ) from e

    # Find all .molden files
    molden_files = sorted(molden_dir.rglob("*.molden"))

    if not molden_files:
        raise DataProcessingError(
            f"No .molden files found in {molden_dir}",
            file_path=str(molden_dir),
            operation="file_discovery",
        )

    logger.info(f"Found {len(molden_files)} .molden files")
    logger.info(f"Feature extraction tier: {feature_tier}")

    # ---
    # Initialize feature storage with only essential features
    features = {"compounds": [], "atoms": [], "coordinates": [], "n_atoms": [], "n_electrons": []}

    # Parsing statistics
    parsed_count = 0
    failed_count = 0
    errors = []

    # Parse each file
    for molden_file in molden_files:
        try:
            compound_name = molden_file.stem

            # Parse with IOData
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LoadWarning)
                mol_data = load_one(str(molden_file))

            # Extract features based on tier
            mol_features = _extract_molecule_features(mol_data, compound_name, feature_tier)

            # Store features - dynamically add ANY feature extracted
            for key, value in mol_features.items():
                if key not in features:
                    features[key] = []  # Initialize storage for new features
                features[key].append(value)

            parsed_count += 1
            logger.debug(f"Parsed [{parsed_count}]: {compound_name}")

        except Exception as e:
            failed_count += 1
            error_msg = f"{molden_file.name}: {str(e)}"
            errors.append(error_msg)
            logger.warning(f"Failed to parse {molden_file.name}: {e}")

            if failed_count > len(molden_files) * 0.5:
                raise DataProcessingError(
                    f"Too many parsing failures ({failed_count}/{len(molden_files)})",
                    operation="molden_parsing",
                    details=f"Recent errors: {errors[-5:]}",
                ) from e

    if parsed_count == 0:
        raise DataProcessingError(
            "No .molden files successfully parsed",
            operation="molden_parsing",
            details=f"Errors: {errors[:10]}",
        )

    # Convert lists to numpy arrays with proper dtypes
    numpy_features = _convert_to_numpy_arrays(features)

    # Create metadata
    metadata = {
        "total_files": len(molden_files),
        "parsed_successfully": parsed_count,
        "failed_to_parse": failed_count,
        "feature_tier": feature_tier,
        "feature_count": len(numpy_features),
        "errors": errors if failed_count > 0 else [],
    }

    logger.info(f"✓ Parsed {parsed_count}/{len(molden_files)} files successfully")
    if failed_count > 0:
        logger.warning(f"⚠ {failed_count} files failed to parse")

    return numpy_features, metadata


def _extract_molecule_features(
    mol_data: Any, compound_name: str, feature_tier: str
) -> dict[str, Any]:
    """
    Extract features from IOData molecule object.

    Args:
        mol_data: IOData molecule object
        compound_name: Name/identifier for molecule
        feature_tier: Feature extraction level

    Returns:
        Dictionary of extracted features
    """
    features = {}

    # Always extract basic features
    features["compounds"] = compound_name
    features["atoms"] = mol_data.atnums
    features["coordinates"] = mol_data.atcoords
    features["n_atoms"] = len(mol_data.atnums)
    features["n_electrons"] = int(np.sum(mol_data.atnums))

    # Extract electronic structure features
    if hasattr(mol_data, "mo") and mol_data.mo is not None:
        mo = mol_data.mo

        # MO energies and occupations
        if hasattr(mo, "energies") and mo.energies is not None:
            features["mo_energies"] = mo.energies

            # COMPLETE TIER: Add MO statistics
            if feature_tier == "complete":
                mo_energies_eV = mo.energies * 27.211386
                features["mo_energy_mean_eV"] = float(np.mean(mo_energies_eV))
                features["mo_energy_std_eV"] = float(np.std(mo_energies_eV))
                features["mo_energy_min_eV"] = float(np.min(mo_energies_eV))
                features["mo_energy_max_eV"] = float(np.max(mo_energies_eV))

            # Find HOMO/LUMO
            if hasattr(mo, "occs") and mo.occs is not None:
                features["mo_occupations"] = mo.occs

                # COMPLETE TIER: Add occupation statistics
                if feature_tier == "complete":
                    features["n_occupied_orbitals"] = int(np.sum(mo.occs > 0.5))
                    features["n_virtual_orbitals"] = int(np.sum(mo.occs <= 0.5))

                # HOMO: highest occupied
                occupied = mo.occs > 0.5
                if np.any(occupied):
                    homo_idx = np.where(occupied)[0][-1]
                    features["homo_energy_eV"] = mo.energies[homo_idx] * 27.211386
                    features["homo_index"] = int(homo_idx)

                    # LUMO: lowest unoccupied
                    if homo_idx + 1 < len(mo.energies):
                        lumo_idx = homo_idx + 1
                        features["lumo_energy_eV"] = mo.energies[lumo_idx] * 27.211386
                        features["lumo_index"] = int(lumo_idx)
                        features["homo_lumo_gap_eV"] = (
                            features["lumo_energy_eV"] - features["homo_energy_eV"]
                        )

        # MO coefficients and additional info for complete tier
        if feature_tier == "complete":
            if hasattr(mo, "coeffs") and mo.coeffs is not None:
                features["mo_coefficients"] = mo.coeffs

            # MO kind (restricted/unrestricted)
            if hasattr(mo, "kind"):
                features["mo_kind"] = str(mo.kind)

            # Number of basis functions
            if hasattr(mo, "coeffs") and mo.coeffs is not None:
                features["n_basis_functions"] = int(mo.coeffs.shape[0])

    # Total energy
    if hasattr(mol_data, "energy") and mol_data.energy is not None:
        features["total_energy_eV"] = mol_data.energy * 27.211386
        features["total_energy_Hartree"] = mol_data.energy
    else:
        # If energy not available, try to compute from other sources or set to NaN
        if feature_tier in ["standard", "complete"]:
            features["total_energy_eV"] = np.nan
            features["total_energy_Hartree"] = np.nan

    # COMPLETE TIER: Number of shells
    if feature_tier == "complete":
        if hasattr(mol_data, "obasis") and mol_data.obasis is not None:
            if hasattr(mol_data.obasis, "nbasis"):
                features["n_shells"] = int(mol_data.obasis.nbasis)

    # Standard and complete tier features
    if feature_tier in ["standard", "complete"]:
        # Molecular formula and weight
        features["molecular_formula"] = _get_molecular_formula(mol_data.atnums)
        features["molecular_weight"] = _get_molecular_weight(mol_data.atnums)

    # Complete tier: derived quantum descriptors
    if feature_tier == "complete":
        if "homo_energy_eV" in features and "lumo_energy_eV" in features:
            homo = features["homo_energy_eV"]
            lumo = features["lumo_energy_eV"]

            # Ionization potential ≈ -HOMO
            features["ionization_potential_eV"] = -homo

            # Electron affinity ≈ -LUMO
            features["electron_affinity_eV"] = -lumo

            # Chemical hardness = (IP - EA) / 2
            features["chemical_hardness_eV"] = (
                features["ionization_potential_eV"] - features["electron_affinity_eV"]
            ) / 2.0

            # Chemical potential = -(IP + EA) / 2
            features["chemical_potential_eV"] = (
                -(features["ionization_potential_eV"] + features["electron_affinity_eV"]) / 2.0
            )

            # Electrophilicity index = μ²/(2η)
            if features["chemical_hardness_eV"] != 0:
                features["electrophilicity_eV"] = features["chemical_potential_eV"] ** 2 / (
                    2 * features["chemical_hardness_eV"]
                )

    return features


def _get_molecular_formula(atnums: np.ndarray) -> str:
    """Generate molecular formula from atomic numbers."""
    from collections import Counter

    # Atomic number to symbol mapping (subset)
    ATOMIC_SYMBOLS = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        15: "P",
        16: "S",
        17: "Cl",
        35: "Br",
        53: "I",
    }

    atom_counts = Counter(atnums)
    formula_parts = []

    for atnum in sorted(atom_counts.keys()):
        symbol = ATOMIC_SYMBOLS.get(atnum, f"X{atnum}")
        count = atom_counts[atnum]
        if count == 1:
            formula_parts.append(symbol)
        else:
            formula_parts.append(f"{symbol}{count}")

    return "".join(formula_parts)


def _get_molecular_weight(atnums: np.ndarray) -> float:
    """Calculate molecular weight from atomic numbers."""
    # Approximate atomic masses
    ATOMIC_MASSES = {
        1: 1.008,
        6: 12.011,
        7: 14.007,
        8: 15.999,
        9: 18.998,
        15: 30.974,
        16: 32.06,
        17: 35.45,
        35: 79.904,
        53: 126.90,
    }

    total_mass = sum(ATOMIC_MASSES.get(atnum, atnum) for atnum in atnums)
    return float(total_mass)


def _convert_to_numpy_arrays(features: dict[str, list]) -> dict[str, np.ndarray]:
    """
    Convert feature lists to numpy arrays with proper dtypes.

    Ensures compatibility with miliaDataset expectations.
    """
    numpy_features = {}

    # Define feature categories
    OBJECT_DTYPE_FEATURES = {
        "compounds",
        "atoms",
        "coordinates",
        "n_atoms",
        "n_electrons",
        "mo_energies",
        "mo_occupations",
        "mo_coefficients",
        "molecular_formula",
        "mo_kind",
    }

    FLOAT64_FEATURES = {
        "homo_energy_eV",
        "lumo_energy_eV",
        "homo_lumo_gap_eV",
        "total_energy_eV",
        "total_energy_Hartree",
        "molecular_weight",
        "ionization_potential_eV",
        "electron_affinity_eV",
        "chemical_hardness_eV",
        "chemical_potential_eV",
        "electrophilicity_eV",
        "mo_energy_mean_eV",
        "mo_energy_std_eV",
        "mo_energy_min_eV",
        "mo_energy_max_eV",
    }

    INT64_FEATURES = {
        "homo_index",
        "lumo_index",
        "n_occupied_orbitals",
        "n_virtual_orbitals",
        "n_basis_functions",
        "n_shells",
    }

    for key, value_list in features.items():
        if not value_list:
            continue

        try:
            if key in OBJECT_DTYPE_FEATURES:
                # Object dtype for variable-length or string data
                numpy_features[key] = np.array(value_list, dtype=object)
            elif key in FLOAT64_FEATURES:
                # Float64 for numeric features
                numpy_features[key] = np.array(value_list, dtype=np.float64)
            elif key in INT64_FEATURES:
                # Int64 for integer features
                numpy_features[key] = np.array(value_list, dtype=np.int64)
            else:
                # Default: try float64, fallback to object
                try:
                    numpy_features[key] = np.array(value_list, dtype=np.float64)
                except (ValueError, TypeError):
                    numpy_features[key] = np.array(value_list, dtype=object)
        except Exception:
            # If all else fails, use object dtype
            numpy_features[key] = np.array(value_list, dtype=object)

    return numpy_features
