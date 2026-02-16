"""
QM9 XYZ Format Parser
─────────────────────

Parser for the QM9 dataset's extended XYZ file format.

The QM9 dataset uses a custom XYZ-like format with additional property
information in the comment line. This parser extracts all molecular
properties and atomic data from the QM9 XYZ files.

QM9 XYZ File Format (from readme.txt):
──────────────────────────────────────
Line    Content
----    -------
1       Number of atoms (na)
2       Properties 1-17 (tab-separated): tag, index, A, B, C, mu, alpha,
        homo, lumo, gap, r2, zpve, U0, U, H, G, Cv
3..na+2 Element type, x, y, z coordinates (Angstrom), Mulliken charge (e)
na+3    Harmonic vibrational frequencies (3na-5 or 3na-6 values, cm⁻¹)
na+4    SMILES strings (GDB-17 and relaxed geometry)
na+5    InChI strings (GDB-17 and relaxed geometry)

Property Units (from QM9 readme.txt):
───────────────────────────────────────────────────────────────────────────
I   Property    Unit            Description
───────────────────────────────────────────────────────────────────────────
1   tag         -               "gdb9" constant string
2   index       -               Consecutive 1-based integer identifier
3   A           GHz             Rotational constant A
4   B           GHz             Rotational constant B
5   C           GHz             Rotational constant C
6   mu          Debye           Dipole moment
7   alpha       Bohr³           Isotropic polarizability
8   homo        Hartree         Energy of HOMO
9   lumo        Hartree         Energy of LUMO
10  gap         Hartree         LUMO - HOMO gap
11  r2          Bohr²           Electronic spatial extent
12  zpve        Hartree         Zero point vibrational energy
13  U0          Hartree         Internal energy at 0K
14  U           Hartree         Internal energy at 298.15K
15  H           Hartree         Enthalpy at 298.15K
16  G           Hartree         Free energy at 298.15K
17  Cv          cal/(mol·K)     Heat capacity at 298.15K

Author: milia Pipeline Team
Version: 1.0
Date: December 2025
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from milia_pipeline.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


# Atomic number lookup table
ELEMENT_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Br": 35,
    "I": 53,
}

# QM9 property names in order (matching readme.txt)
QM9_PROPERTY_NAMES = [
    "tag",  # 0: "gdb9" string constant
    "index",  # 1: molecule index (1-based)
    "A",  # 2: Rotational constant A (GHz)
    "B",  # 3: Rotational constant B (GHz)
    "C",  # 4: Rotational constant C (GHz)
    "mu",  # 5: Dipole moment (Debye)
    "alpha",  # 6: Isotropic polarizability (Bohr³)
    "homo",  # 7: HOMO energy (Hartree)
    "lumo",  # 8: LUMO energy (Hartree)
    "gap",  # 9: HOMO-LUMO gap (Hartree)
    "r2",  # 10: Electronic spatial extent (Bohr²)
    "zpve",  # 11: Zero point vibrational energy (Hartree)
    "U0",  # 12: Internal energy at 0K (Hartree)
    "U",  # 13: Internal energy at 298.15K (Hartree)
    "H",  # 14: Enthalpy at 298.15K (Hartree)
    "G",  # 15: Free energy at 298.15K (Hartree)
    "Cv",  # 16: Heat capacity at 298.15K (cal/(mol·K))
]


def _parse_scientific_notation(value_str: str) -> float:
    """
    Parse QM9's scientific notation which uses '*^' instead of 'e'.

    QM9 XYZ files use a non-standard format like '1.234*^-5' instead of '1.234e-5'.

    Args:
        value_str: String value potentially containing '*^' notation

    Returns:
        Parsed float value
    """
    # Replace QM9's '*^' notation with standard 'e' notation
    normalized = value_str.replace("*^", "e")
    return float(normalized)


def _parse_qm9_xyz_file(file_path: Path) -> dict[str, Any]:
    """
    Parse a single QM9 XYZ file and extract all properties.

    Args:
        file_path: Path to the .xyz file

    Returns:
        Dictionary containing all parsed molecular data:
        - 'num_atoms': int
        - 'atoms': np.ndarray of atomic numbers
        - 'coordinates': np.ndarray of shape (num_atoms, 3)
        - 'Qmulliken': np.ndarray of Mulliken charges
        - 'freqs': np.ndarray of vibrational frequencies
        - 'smiles': str (original SMILES)
        - 'smiles_relaxed': str (relaxed geometry SMILES)
        - 'inchi': str (original InChI)
        - 'inchi_relaxed': str (relaxed geometry InChI)
        - All 15 scalar properties (A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv)

    Raises:
        DataProcessingError: If parsing fails
    """
    try:
        with open(file_path) as f:
            lines = [line.strip() for line in f.readlines()]

        if len(lines) < 3:
            raise DataProcessingError(
                f"QM9 XYZ file too short: {file_path}",
                file_path=str(file_path),
                operation="qm9_xyz_parsing",
            )

        # Line 1: Number of atoms
        num_atoms = int(lines[0])

        # Line 2: Properties (tab-separated)
        # Format: "gdb9 index A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv"
        # Note: The first element may include "gdb" prefix before the space
        prop_line = lines[1]

        # Split by tabs and/or spaces, handling the "gdb 9" -> "gdb9" issue
        # The line format is: "gdb 123\tA\tB\tC\t..." where the first part has space
        parts = prop_line.split("\t")

        # First part contains "gdb INDEX" - split by space
        first_parts = parts[0].split()
        tag = first_parts[0]  # "gdb"
        mol_index = int(first_parts[1])  # molecule index

        # Remaining parts are the numeric properties
        numeric_props = [_parse_scientific_notation(p) for p in parts[1:]]

        # Build properties dictionary
        result = {
            "tag": tag,
            "index": mol_index,
        }

        # Map numeric properties to names (A, B, C, mu, alpha, ...)
        prop_names_numeric = QM9_PROPERTY_NAMES[2:]  # Skip 'tag' and 'index'
        for i, prop_name in enumerate(prop_names_numeric):
            if i < len(numeric_props):
                result[prop_name] = numeric_props[i]

        # Lines 3 to num_atoms+2: Atomic coordinates and charges
        atoms = []
        coordinates = []
        mulliken_charges = []

        for i in range(num_atoms):
            atom_line = lines[2 + i]
            # Format: "Element X Y Z Charge" (tab-separated)
            # Note: QM9 uses '*^' for scientific notation
            atom_parts = atom_line.split("\t")

            element = atom_parts[0]
            x = _parse_scientific_notation(atom_parts[1])
            y = _parse_scientific_notation(atom_parts[2])
            z = _parse_scientific_notation(atom_parts[3])
            charge = _parse_scientific_notation(atom_parts[4])

            # Convert element symbol to atomic number
            atomic_num = ELEMENT_TO_Z.get(element)
            if atomic_num is None:
                raise DataProcessingError(
                    f"Unknown element '{element}' in {file_path}",
                    file_path=str(file_path),
                    operation="qm9_xyz_parsing",
                )

            atoms.append(atomic_num)
            coordinates.append([x, y, z])
            mulliken_charges.append(charge)

        result["num_atoms"] = num_atoms
        result["atoms"] = np.array(atoms, dtype=np.int32)
        result["coordinates"] = np.array(coordinates, dtype=np.float64)
        result["Qmulliken"] = np.array(mulliken_charges, dtype=np.float64)

        # Line num_atoms+3: Vibrational frequencies (tab-separated)
        freq_line_idx = 2 + num_atoms
        if freq_line_idx < len(lines) and lines[freq_line_idx]:
            freq_parts = lines[freq_line_idx].split("\t")
            freqs = [float(f) for f in freq_parts if f]
            result["freqs"] = np.array(freqs, dtype=np.float64)
        else:
            result["freqs"] = np.array([], dtype=np.float64)

        # Line num_atoms+4: SMILES strings (original and relaxed, tab-separated)
        smiles_line_idx = 3 + num_atoms
        if smiles_line_idx < len(lines) and lines[smiles_line_idx]:
            smiles_parts = lines[smiles_line_idx].split("\t")
            result["smiles"] = smiles_parts[0] if len(smiles_parts) > 0 else ""
            result["smiles_relaxed"] = smiles_parts[1] if len(smiles_parts) > 1 else ""
        else:
            result["smiles"] = ""
            result["smiles_relaxed"] = ""

        # Line num_atoms+5: InChI strings (original and relaxed, tab-separated)
        inchi_line_idx = 4 + num_atoms
        if inchi_line_idx < len(lines) and lines[inchi_line_idx]:
            inchi_parts = lines[inchi_line_idx].split("\t")
            result["inchi"] = inchi_parts[0] if len(inchi_parts) > 0 else ""
            result["inchi_relaxed"] = inchi_parts[1] if len(inchi_parts) > 1 else ""
        else:
            result["inchi"] = ""
            result["inchi_relaxed"] = ""

        return result

    except Exception as e:
        if isinstance(e, DataProcessingError):
            raise
        raise DataProcessingError(
            f"Failed to parse QM9 XYZ file: {e}",
            file_path=str(file_path),
            operation="qm9_xyz_parsing",
        ) from e


def parse_qm9_xyz_files(
    xyz_dir: Path, max_molecules: int | None = None, logger: logging.Logger | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Parse all QM9 XYZ files in a directory and build feature arrays.

    This is the main entry point for QM9 preprocessing. It reads all .xyz
    files from the extraction directory and builds numpy arrays suitable
    for NPZ file creation.

    Args:
        xyz_dir: Directory containing extracted .xyz files
        max_molecules: Maximum number of molecules to parse (None = all)
        logger: Logger instance (uses module logger if None)

    Returns:
        Tuple of (features_dict, metadata_dict):
        - features_dict: Dictionary of numpy arrays for NPZ file
        - metadata_dict: Dictionary with parsing statistics

    Raises:
        DataProcessingError: If parsing fails

    Example:
        >>> features, metadata = parse_qm9_xyz_files(Path("/tmp/qm9_xyz"))
        >>> print(f"Parsed {metadata['num_molecules_parsed']} molecules")
    """
    if logger is None:
        logger = globals()["logger"]

    # Find all .xyz files
    xyz_files = sorted(xyz_dir.rglob("*.xyz"))

    if not xyz_files:
        raise DataProcessingError(
            f"No .xyz files found in {xyz_dir}", file_path=str(xyz_dir), operation="qm9_xyz_parsing"
        )

    total_files = len(xyz_files)
    if max_molecules is not None:
        xyz_files = xyz_files[:max_molecules]

    logger.info(f"Found {total_files} .xyz files, processing {len(xyz_files)}")
    logger.info("Parsing commenced...")

    # Initialize collection lists
    compounds = []
    atoms_list = []
    coordinates_list = []
    mulliken_charges_list = []
    freqs_list = []
    smiles_list = []
    smiles_relaxed_list = []
    inchi_list = []
    inchi_relaxed_list = []

    # Scalar property arrays
    scalar_props = {name: [] for name in QM9_PROPERTY_NAMES[2:]}  # Skip tag, index

    # Parsing statistics
    parsed_count = 0
    failed_count = 0
    failed_files = []

    for i, xyz_file in enumerate(xyz_files):
        # Progress updates at DEBUG level to avoid log clutter for large datasets
        if (i + 1) % 10000 == 0:
            logger.debug(f"Parsing progress: {i + 1}/{len(xyz_files)} molecules parsed")

        try:
            mol_data = _parse_qm9_xyz_file(xyz_file)

            # Use file stem as compound identifier (e.g., "dsgdb9nsd_000001")
            compound_id = xyz_file.stem
            compounds.append(compound_id)

            atoms_list.append(mol_data["atoms"])
            coordinates_list.append(mol_data["coordinates"])
            mulliken_charges_list.append(mol_data["Qmulliken"])
            freqs_list.append(mol_data["freqs"])

            smiles_list.append(mol_data["smiles"])
            smiles_relaxed_list.append(mol_data["smiles_relaxed"])
            inchi_list.append(mol_data["inchi"])
            inchi_relaxed_list.append(mol_data["inchi_relaxed"])

            # Collect scalar properties
            for prop_name in scalar_props:
                scalar_props[prop_name].append(mol_data.get(prop_name, np.nan))

            parsed_count += 1

        except Exception as e:
            failed_count += 1
            failed_files.append(str(xyz_file))
            logger.warning(f"Failed to parse {xyz_file.name}: {e}")

            # Stop if too many failures
            if failed_count > 100 and failed_count > parsed_count * 0.1:
                raise DataProcessingError(
                    f"Too many parsing failures ({failed_count}/{parsed_count + failed_count})",
                    operation="qm9_xyz_parsing",
                    details=f"First failures: {failed_files[:10]}",
                ) from e

    if parsed_count == 0:
        raise DataProcessingError(
            "No molecules successfully parsed", file_path=str(xyz_dir), operation="qm9_xyz_parsing"
        )

    logger.info(f"✓ Parsing complete: {parsed_count} molecules parsed ({failed_count} failures)")

    # Build feature dictionary with numpy arrays
    features = {
        # Required core features
        "compounds": np.array(compounds, dtype=object),
        "atoms": np.array(atoms_list, dtype=object),
        "coordinates": np.array(coordinates_list, dtype=object),
        # Per-atom features
        "Qmulliken": np.array(mulliken_charges_list, dtype=object),
        # Variable-length features
        "freqs": np.array(freqs_list, dtype=object),
        # String identifiers
        "smiles": np.array(smiles_list, dtype=object),
        "smiles_relaxed": np.array(smiles_relaxed_list, dtype=object),
        "inchi": np.array(inchi_list, dtype=object),
        "inchi_relaxed": np.array(inchi_relaxed_list, dtype=object),
    }

    # Add scalar properties as separate arrays
    for prop_name, values in scalar_props.items():
        features[prop_name] = np.array(values, dtype=np.float64)

    # Build metadata dictionary
    metadata = {
        "num_molecules_parsed": parsed_count,
        "num_molecules_failed": failed_count,
        "total_files_found": total_files,
        "source_format": "qm9_xyz",
        "property_names": list(scalar_props.keys()),
        "has_mulliken_charges": True,
        "has_frequencies": True,
        "coordinate_units": "angstrom",
        "energy_units": "hartree",
    }

    if failed_files:
        metadata["failed_files_sample"] = failed_files[:10]

    return features, metadata


def get_qm9_property_info() -> dict[str, dict[str, str]]:
    """
    Get information about QM9 properties including units and descriptions.

    Returns:
        Dictionary mapping property names to their metadata

    Example:
        >>> info = get_qm9_property_info()
        >>> print(info['U0']['unit'])
        'Hartree'
    """
    return {
        "A": {"unit": "GHz", "description": "Rotational constant A"},
        "B": {"unit": "GHz", "description": "Rotational constant B"},
        "C": {"unit": "GHz", "description": "Rotational constant C"},
        "mu": {"unit": "Debye", "description": "Dipole moment"},
        "alpha": {"unit": "Bohr³", "description": "Isotropic polarizability"},
        "homo": {"unit": "Hartree", "description": "HOMO energy"},
        "lumo": {"unit": "Hartree", "description": "LUMO energy"},
        "gap": {"unit": "Hartree", "description": "HOMO-LUMO gap"},
        "r2": {"unit": "Bohr²", "description": "Electronic spatial extent"},
        "zpve": {"unit": "Hartree", "description": "Zero point vibrational energy"},
        "U0": {"unit": "Hartree", "description": "Internal energy at 0K"},
        "U": {"unit": "Hartree", "description": "Internal energy at 298.15K"},
        "H": {"unit": "Hartree", "description": "Enthalpy at 298.15K"},
        "G": {"unit": "Hartree", "description": "Free energy at 298.15K"},
        "Cv": {"unit": "cal/(mol·K)", "description": "Heat capacity at 298.15K"},
    }
