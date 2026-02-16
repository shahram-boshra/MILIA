# milia_pipeline/preprocessing/preprocessors/qdpi.py

"""
QDπ Preprocessor
================

Preprocessor for QDπ (Quantum Deep Potential Interaction) dataset.

Parses QDπ HDF5 files in DeePMD-kit format, extracts molecular data from
chemical formula groups, handles unit conversions, and creates .npz file
compatible with miliaDataset.

QDπ Dataset Information:
------------------------
- Source: Zenodo (DOI: 10.5281/zenodo.14970869)
- Archive: QDpiDataset-main.tar.gz
- Reference: Zeng et al., Scientific Data 12, 693 (2025)
- Contents: ~1.6 million structures for drug discovery ML potentials
- Method: ωB97M-D3(BJ)/def2-TZVPPD (highly accurate DFT functional)

CRITICAL: QDπ Contains BOTH Neutral AND Charged Molecules
----------------------------------------------------------
Unlike ANI-2x (neutral-only), QDπ dataset is partitioned into:
- data/neutral/ : Neutral molecules (molecular_charge = 0)
- data/charged/ : Charged molecules (ions, protonated/deprotonated species)

The molecular charge is NOT stored in the HDF5 files! It must be:
1. Determined from the file path during preprocessing
2. Stored in the NPZ as 'molecular_charge' property
3. Retrieved by the handler's get_molecular_charge() method

This is ESSENTIAL for correct bond order determination via rdDetermineBonds.

HDF5 Structure (DeePMD-kit format):
-----------------------------------------------------------
According to DeePMD-kit documentation and dpdata source code:
https://docs.deepmodeling.com/projects/deepmd/en/master/data/data-conv.html

Each HDF5 file contains groups organized by chemical formula.
Each formula group contains:
- 'type.raw': Atom type indices (integers), shape (Na,)
- 'type_map.raw': Element symbol mapping (byte strings), shape (Nt,)
- 'set.XXX/coord.npy': Coordinates in Angstrom, shape (Nc, Na*3) flattened
- 'set.XXX/energy.npy': Total energies in eV, shape (Nc,)
- 'set.XXX/force.npy': Atomic forces in eV/Å, shape (Nc, Na*3) flattened

Where: Nc = number of conformers, Na = number of atoms, Nt = number of element types

Unit Conversions Applied:
- Energy: eV → Hartree (1 Ha = 27.211386245988 eV)
- Forces: eV/Å → Hartree/Å
- Coordinates: Angstrom (no conversion)

Source Dataset Subsets:
- SPICE: dipeptides, solvated amino acids, ion pairs, PubChem molecules
- ANI: neutral organic molecules
- GEOM: drug-like molecules
- FreeSolv: solvation free energy molecules
- RE: relative energy benchmarks (tautomers, protonation states)
- COMP6: validation benchmarks
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry

logger = logging.getLogger(__name__)


# Unit conversion constants
EV_TO_HARTREE = 1.0 / 27.211386245988  # 1 Hartree = 27.211386245988 eV

# QDπ supports 13 elements: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I
QDPI_SUPPORTED_ELEMENTS = {1, 3, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53}

# Element symbol to atomic number mapping
ELEMENT_TO_Z = {
    "H": 1,
    "Li": 3,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Br": 35,
    "I": 53,
}


def iter_data_buckets_qdpi(
    h5filename: str, keys: list[str] = None, charge_type: str = "neutral"
) -> dict[str, Any]:
    """
    Iterate over buckets of data in QDπ HDF5 file (DeePMD-kit format).

    This function iterates over formula groups in the HDF5 file, yielding
    dictionaries with atomic numbers, coordinates, and requested properties
    for each conformer, filtering out entries with NaN values.

    CRITICAL: The charge_type parameter determines molecular_charge for all
    conformers in this file. This is essential for correct molecule creation.

    DeePMD-kit HDF5 Format:
    -----------------------
    According to DeePMD-kit documentation, each formula group contains:
    - 'type.raw': Atom type indices (integers), shape (Na,)
    - 'type_map.raw': Element symbol mapping (byte strings), shape (Nt,)
    - 'set.XXX/coord.npy': Coordinates, shape (Nc, Na*3) - flattened
    - 'set.XXX/energy.npy': Energies in eV, shape (Nc,)
    - 'set.XXX/force.npy': Forces in eV/Å, shape (Nc, Na*3) - flattened

    Where: Nc = number of conformers, Na = number of atoms, Nt = number of types

    Reference: https://docs.deepmodeling.com/projects/deepmd/en/master/data/data-conv.html

    Args:
        h5filename: Path to QDπ HDF5 file
        keys: List of property keys to load (default: ['energies'])
        charge_type: 'neutral' or 'charged' (from directory structure)

    Yields:
        Dict with:
            - 'atomic_numbers': Shape (Na,) - atomic numbers for this conformer
            - 'coordinates': Shape (Na, 3) - positions for this conformer (Angstrom)
            - 'formula': String identifier for the chemical formula group
            - 'molecular_charge': int - charge based on charge_type and inference
            - 'charge_type': 'neutral' or 'charged'
            - Plus any requested property keys (energies → energy in Hartree)
    """
    import h5py

    if keys is None:
        keys = ["energies"]

    with h5py.File(h5filename, "r") as f:
        for formula_name in f:
            formula_group = f[formula_name]

            # Skip non-group entries (e.g., attributes at root level)
            if not isinstance(formula_group, h5py.Group):
                continue

            # ============================================================
            # DeePMD-kit HDF5 Format: type.raw contains atom type indices
            # ============================================================
            if "type.raw" not in formula_group:
                logger.warning(
                    f"Skipping {formula_name}: no 'type.raw' key found (not DeePMD-kit format)"
                )
                continue

            # type.raw: integer indices into type_map.raw
            atom_types = np.array(formula_group["type.raw"][:])
            n_atoms = atom_types.size
            n_types = np.max(atom_types) + 1

            # ============================================================
            # type_map.raw: element symbol mapping (byte strings)
            # ============================================================
            if "type_map.raw" in formula_group:
                type_map_raw = formula_group["type_map.raw"][:]
                # Decode byte strings to Python strings
                if type_map_raw.dtype.kind == "S":  # byte string
                    type_map = [s.decode("utf-8") for s in type_map_raw]
                else:
                    type_map = list(type_map_raw)
            else:
                # Fallback: create artificial type names
                type_map = [f"Type_{i}" for i in range(n_types)]
                logger.debug(f"No type_map.raw for {formula_name}, using artificial names")

            # Convert type indices to atomic numbers
            # type_map contains element symbols like ['H', 'C', 'N', 'O']
            # atom_types contains indices like [0, 0, 1, 1, 2, 3, 0, 0]
            atomic_numbers = np.array(
                [ELEMENT_TO_Z.get(type_map[int(at)], 0) for at in atom_types], dtype=np.uint8
            )

            # Get list of elements for charge inference
            elements = [type_map[int(at)] for at in sorted(set(atom_types))]

            # ============================================================
            # Find set.XXX directories containing actual data
            # ============================================================
            set_dirs = sorted([k for k in formula_group if k.startswith("set.")])

            if not set_dirs:
                logger.warning(f"Skipping {formula_name}: no set.XXX directories found")
                continue

            # ============================================================
            # Determine molecular charge
            # ============================================================
            if charge_type == "neutral":
                base_charge = 0
            else:
                base_charge = _infer_charge_from_formula(formula_name, elements)

            # ============================================================
            # Process each set directory
            # ============================================================
            for set_dir in set_dirs:
                set_group = formula_group[set_dir]

                # coord.npy: coordinates, shape (Nc, Na*3) flattened
                if "coord.npy" not in set_group:
                    logger.warning(f"Skipping {formula_name}/{set_dir}: no coord.npy")
                    continue

                coords_flat = np.array(set_group["coord.npy"][:], dtype=np.float32)
                n_conformers = coords_flat.shape[0]

                # Reshape from (Nc, Na*3) to (Nc, Na, 3)
                coordinates_all = coords_flat.reshape(n_conformers, n_atoms, 3)

                # Load requested properties
                properties = {}
                valid_mask = np.ones(n_conformers, dtype=bool)

                for key in keys:
                    # Map property names to DeePMD-kit format
                    if key == "energies":
                        npy_key = "energy.npy"
                    elif key == "forces":
                        npy_key = "force.npy"
                    else:
                        npy_key = f"{key}.npy"

                    if npy_key in set_group:
                        prop_data = np.array(set_group[npy_key])
                        properties[key] = prop_data

                        # Check for NaN values
                        if np.issubdtype(prop_data.dtype, np.floating):
                            if prop_data.ndim == 1:
                                valid_mask &= ~np.isnan(prop_data)
                            else:
                                valid_mask &= ~np.any(
                                    np.isnan(prop_data.reshape(n_conformers, -1)), axis=1
                                )

                # ============================================================
                # Yield data for each valid conformer
                # ============================================================
                for conf_idx in range(n_conformers):
                    if not valid_mask[conf_idx]:
                        continue

                    # Filter out padding zeros (atoms with Z=0)
                    non_zero_mask = atomic_numbers > 0
                    conf_atomic_numbers = atomic_numbers[non_zero_mask]
                    conf_coordinates = coordinates_all[conf_idx][non_zero_mask]

                    result = {
                        "atomic_numbers": conf_atomic_numbers,
                        "coordinates": conf_coordinates,
                        "formula": formula_name,
                        "molecular_charge": base_charge,
                        "charge_type": charge_type,
                    }

                    # Add properties with unit conversions
                    for key, prop_data in properties.items():
                        if key == "energies":
                            # Convert eV to Hartree
                            energy_ev = prop_data[conf_idx]
                            result["energy"] = float(energy_ev * EV_TO_HARTREE)
                        elif key == "forces":
                            # Forces are stored flattened (Nc, Na*3) → reshape to (Na, 3)
                            forces_flat = prop_data[conf_idx]
                            forces_ev = forces_flat.reshape(n_atoms, 3)
                            # Apply same mask as coordinates
                            forces_ev = forces_ev[non_zero_mask]
                            # Convert eV/Angstrom to Hartree/Angstrom
                            result["forces"] = (forces_ev * EV_TO_HARTREE).astype(np.float32)
                        else:
                            if prop_data.ndim == 1:
                                result[key] = prop_data[conf_idx]
                            else:
                                result[key] = prop_data[conf_idx]

                    yield result


def _infer_charge_from_formula(formula: str, elements: list[str]) -> int:
    """
    Infer molecular charge from chemical formula and elements.

    This is a heuristic for charged molecules where exact charge isn't stored.
    Identifies common ionic species in QDπ charged subset.

    Common patterns in QDπ charged subset:
    - Single alkali cations: Li+, Na+, K+ → charge = +1
    - Single halide anions: F-, Cl-, Br-, I- → charge = -1
    - Protonated amino acids: typically +1 (e.g., LYS-H+)
    - Deprotonated carboxylic acids: typically -1 (e.g., ASP-H-)

    Args:
        formula: Chemical formula group name
        elements: List of element symbols in this formula

    Returns:
        int: Inferred molecular charge (defaults to 0 if uncertain)
    """
    # Single atom ions
    if len(elements) == 1:
        elem = elements[0]
        if elem in ["Li", "Na", "K"]:
            return 1  # Alkali cation
        if elem in ["F", "Cl", "Br", "I"]:
            return -1  # Halide anion

    # Check formula name for charge hints (some datasets encode this)
    formula_lower = formula.lower()
    if "+1" in formula or "_pos" in formula_lower or "_cation" in formula_lower:
        return 1
    if "-1" in formula or "_neg" in formula_lower or "_anion" in formula_lower:
        return -1
    if "+2" in formula:
        return 2
    if "-2" in formula:
        return -2

    # Count element types for heuristics
    has_alkali = any(e in ["Li", "Na", "K"] for e in elements)
    has_halide = any(e in ["F", "Cl", "Br", "I"] for e in elements)

    # Ion pair (e.g., NaCl) - typically neutral overall
    if has_alkali and has_halide and len(elements) == 2:
        return 0

    # For more complex molecules, cannot reliably determine
    # Log that we're defaulting to 0 for tracking
    logger.debug(f"Cannot infer charge for {formula} with elements {elements}, defaulting to 0")
    return 0


@PreprocessorRegistry.register("QDPi")
class QDPiPreprocessor(BasePreprocessor):
    """
    Preprocessor for QDπ (Quantum Deep Potential Interaction) dataset.

    Pipeline:
    ---------
    1. Extract HDF5 files from tar.gz archive (if needed)
    2. Identify neutral/ and charged/ subdirectories
    3. Process each HDF5 file, tracking charge_type from directory
    4. Extract conformer data with unit conversions (eV→Hartree)
    5. Store molecular_charge for each conformer
    6. Build .npz file (compressed format compatible with miliaDataset)

    CRITICAL: Charge Tracking
    -------------------------
    The QDπ dataset is partitioned into neutral and charged subsets:
    - data/neutral/*.hdf5 → all molecules have molecular_charge = 0
    - data/charged/*.hdf5 → molecules have non-zero charges

    The molecular_charge is NOT stored in the HDF5 files and must be
    determined from the directory structure during preprocessing.

    Configuration:
    --------------
    Required keys:
        - raw_archive_path: Path to QDpiDataset-main.tar.gz or extracted directory
        - output_npz_path: Path for output .npz file

    Optional keys:
        - num_molecules: Number of conformers to extract (None = all)
        - property_keys: List of properties to extract (default: ['energies', 'forces'])
        - include_charged: Whether to include charged molecules (default: True)
        - include_neutral: Whether to include neutral molecules (default: True)

    Example:
    --------
    >>> config = {
    ...     'raw_archive_path': 'raw/QDpiDataset-main.tar.gz',
    ...     'output_npz_path': 'processed/qdpi.npz',
    ...     'num_molecules': 10000,  # For testing
    ... }
    >>> preprocessor = QDPiPreprocessor(config, logger)
    >>> output_path = preprocessor.run()
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger):
        """
        Initialize QDπ preprocessor.

        Args:
            config: Configuration dictionary with:
                - raw_archive_path: Path to tar.gz or extracted directory
                - output_npz_path: Path for output .npz file
                - preprocessing: Dict with optional settings
            logger: Logger instance for output messages
        """
        super().__init__(config, logger)
        # Note: self.logger and self.config are set by BasePreprocessor.__init__

    def _validate_config(self) -> None:
        """Validate QDπ-specific configuration."""
        required_keys = ["raw_archive_path", "output_npz_path"]
        missing = [k for k in required_keys if k not in self.config]

        if missing:
            raise ConfigurationError(
                f"QDπ preprocessor missing required config keys: {missing}",
                config_key="qdpi_config",
            )

    def preprocess(self) -> Path:
        """
        Run QDπ preprocessing pipeline.

        Returns:
            Path to created .npz file
        """
        self._validate_config()

        archive_path = Path(self.config["raw_archive_path"])
        output_npz = Path(self.config["output_npz_path"])

        # Get preprocessing options
        # Config structure matches ANI-2x: all preprocessing params are at root level of self.config
        # This is because the main CLI flattens processing_config.preprocessing into the config
        # passed to the preprocessor
        max_conformers = self.config.get("num_molecules", None)
        property_keys = self.config.get("property_keys", ["energies", "forces"])
        include_charged = self.config.get("include_charged", True)
        include_neutral = self.config.get("include_neutral", True)

        # Ensure we have at least energies
        if "energies" not in property_keys:
            property_keys.insert(0, "energies")
        # Map 'energy' to 'energies' and 'force' to 'forces' for HDF5 keys
        hdf5_keys = []
        for k in property_keys:
            if k == "energy":
                hdf5_keys.append("energies")
            elif k == "force":
                hdf5_keys.append("forces")
            else:
                hdf5_keys.append(k)

        self.logger.info("=" * 70)
        self.logger.info("QDπ PREPROCESSING PIPELINE")
        self.logger.info("=" * 70)
        self.logger.info(f"Archive/Directory: {archive_path}")
        self.logger.info(f"Output NPZ: {output_npz}")
        self.logger.info(f"Max conformers: {max_conformers or 'ALL'}")
        self.logger.info(f"Property keys: {hdf5_keys}")
        self.logger.info(f"Include neutral: {include_neutral}")
        self.logger.info(f"Include charged: {include_charged}")

        try:
            # Step 1: Get path to extracted data
            data_path = self._get_data_path(archive_path)

            # Step 2: Find HDF5 files in neutral/ and charged/ subdirectories
            h5_files = self._find_h5_files(data_path, include_neutral, include_charged)

            if not h5_files:
                raise DataProcessingError(
                    "No HDF5 files found in QDπ dataset", operation="find_h5_files"
                )

            self.logger.info(f"Found {len(h5_files)} HDF5 files to process")

            # Step 3: Parse all HDF5 files and extract data
            features, metadata = self._parse_qdpi_h5_files(h5_files, hdf5_keys, max_conformers)

            # Step 4: Build NPZ file
            self._build_npz(features, metadata, output_npz)

            self.logger.info("=" * 70)
            self.logger.info("QDπ PREPROCESSING COMPLETE")
            self.logger.info("=" * 70)

            return output_npz

        except Exception as e:
            raise DataProcessingError(
                f"QDπ preprocessing failed: {e}", operation="qdpi_preprocessing"
            ) from e

    def _get_data_path(self, archive_path: Path) -> Path:
        """
        Get path to QDπ data directory, extracting from archive if needed.

        Args:
            archive_path: Path to archive (.tar.gz) or extracted directory

        Returns:
            Path to data directory containing neutral/ and charged/ subdirs
        """
        # If already a directory, return as-is
        if archive_path.is_dir():
            self.logger.info(f"Using directory directly: {archive_path}")
            # Look for data/ subdirectory
            data_dir = archive_path / "data"
            if data_dir.exists():
                return data_dir
            return archive_path

        # Extract from tar.gz archive
        self.logger.info("=" * 70)
        self.logger.info("STEP 1: Extracting from tar.gz archive")
        self.logger.info("=" * 70)
        self.logger.info(f"Archive: {archive_path}")

        import tarfile

        # Extract to directory alongside archive
        extract_dir = archive_path.parent / "qdpi_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "r:gz") as tar:
            self.logger.info("Extracting archive contents...")
            tar.extractall(extract_dir)

        # Find the data directory (may be nested)
        # QDπ archive structure: QDpiDataset-main/data/neutral/*.hdf5
        for candidate in [
            extract_dir / "QDpiDataset-main" / "data",
            extract_dir / "data",
            extract_dir,
        ]:
            if candidate.exists() and (
                (candidate / "neutral").exists() or (candidate / "charged").exists()
            ):
                self.logger.info(f"Found data directory: {candidate}")
                return candidate

        # If no structured directory, return extract_dir
        self.logger.info(f"Using extracted directory: {extract_dir}")
        return extract_dir

    def _find_h5_files(
        self, data_path: Path, include_neutral: bool, include_charged: bool
    ) -> list[tuple[Path, str]]:
        """
        Find all HDF5 files in QDπ data directory.

        Args:
            data_path: Path to data directory
            include_neutral: Whether to include neutral molecules
            include_charged: Whether to include charged molecules

        Returns:
            List of (h5_path, charge_type) tuples
        """
        h5_files = []

        # Look for neutral/ subdirectory
        neutral_dir = data_path / "neutral"
        if include_neutral and neutral_dir.exists():
            for h5_file in neutral_dir.glob("*.hdf5"):
                h5_files.append((h5_file, "neutral"))
            for h5_file in neutral_dir.glob("*.h5"):
                h5_files.append((h5_file, "neutral"))
            self.logger.info(
                f"Found {len([f for f, t in h5_files if t == 'neutral'])} neutral HDF5 files"
            )

        # Look for charged/ subdirectory
        charged_dir = data_path / "charged"
        if include_charged and charged_dir.exists():
            charged_count_before = len(h5_files)
            for h5_file in charged_dir.glob("*.hdf5"):
                h5_files.append((h5_file, "charged"))
            for h5_file in charged_dir.glob("*.h5"):
                h5_files.append((h5_file, "charged"))
            charged_count = len(h5_files) - charged_count_before
            self.logger.info(f"Found {charged_count} charged HDF5 files")

        # If no subdirectories, look for HDF5 files directly (assume neutral)
        if not h5_files:
            for h5_file in data_path.glob("*.hdf5"):
                h5_files.append((h5_file, "neutral"))
            for h5_file in data_path.glob("*.h5"):
                h5_files.append((h5_file, "neutral"))
            if h5_files:
                self.logger.warning(
                    f"No neutral/charged subdirectories found. "
                    f"Treating {len(h5_files)} HDF5 files as neutral."
                )

        return h5_files

    def _parse_qdpi_h5_files(
        self,
        h5_files: list[tuple[Path, str]],
        property_keys: list[str],
        max_conformers: int | None = None,
    ) -> tuple[dict[str, list], dict[str, Any]]:
        """
        Parse QDπ HDF5 files and extract molecular data.

        Args:
            h5_files: List of (h5_path, charge_type) tuples
            property_keys: List of property keys to extract
            max_conformers: Maximum number of conformers to extract (None = all)

        Returns:
            Tuple of (features_dict, metadata_dict)
        """
        # Initialize storage for ragged arrays (variable-length per molecule)
        atoms_list = []  # List of atomic number arrays (integers)
        coordinates_list = []  # List of coordinate arrays
        energy_list = []  # List of energy values (Hartree)
        forces_list = []  # List of force arrays (Hartree/Angstrom)
        formula_list = []  # List of formula identifiers
        molecular_charge_list = []  # CRITICAL: List of molecular charges
        charge_type_list = []  # List of charge types ('neutral'/'charged')
        subset_list = []  # List of source subset identifiers

        conformer_count = 0
        skipped_nan = 0
        skipped_unknown_element = 0
        neutral_count = 0
        charged_count = 0

        self.logger.info("=" * 70)
        self.logger.info("STEP 2: Parsing HDF5 files")
        self.logger.info("=" * 70)

        for h5_path, charge_type in h5_files:
            if max_conformers is not None and conformer_count >= max_conformers:
                break

            # Determine subset from filename
            subset = h5_path.stem  # e.g., 'spice', 'ani', 'geom'

            self.logger.info(f"Processing: {h5_path.name} ({charge_type})")

            file_conformer_count = 0

            for data in iter_data_buckets_qdpi(
                str(h5_path), keys=property_keys, charge_type=charge_type
            ):
                # Check if we've reached the limit
                if max_conformers is not None and conformer_count >= max_conformers:
                    break

                # Validate elements are in QDπ supported set
                atomic_numbers = np.ascontiguousarray(data["atomic_numbers"], dtype=np.uint8)
                unique_elements = set(atomic_numbers.tolist())
                if not unique_elements.issubset(QDPI_SUPPORTED_ELEMENTS):
                    unsupported = unique_elements - QDPI_SUPPORTED_ELEMENTS
                    self.logger.debug(
                        f"Skipping conformer with unsupported elements: {unsupported}"
                    )
                    skipped_unknown_element += 1
                    continue

                # Ensure coordinates are explicit contiguous float32 arrays
                coordinates = np.ascontiguousarray(data["coordinates"], dtype=np.float32)

                # Store data
                atoms_list.append(atomic_numbers)
                coordinates_list.append(coordinates)
                formula_list.append(data["formula"])
                subset_list.append(subset)

                # CRITICAL: Store molecular charge
                molecular_charge = data.get("molecular_charge", 0)
                molecular_charge_list.append(molecular_charge)
                charge_type_list.append(charge_type)

                # Count based on charge_type from file path, NOT molecular_charge value
                # This is because _infer_charge_from_formula() returns 0 for complex molecules
                # where we cannot reliably determine charge from formula alone
                if charge_type == "neutral":
                    neutral_count += 1
                else:
                    charged_count += 1

                # Energy (required) - already converted to Hartree
                if "energy" in data:
                    energy_list.append(data["energy"])
                else:
                    energy_list.append(np.nan)

                # Forces (optional) - already converted to Hartree/Angstrom
                if "forces" in data and data["forces"] is not None:
                    forces_list.append(np.ascontiguousarray(data["forces"], dtype=np.float32))
                else:
                    forces_list.append(None)

                conformer_count += 1
                file_conformer_count += 1

            self.logger.info(
                f"  → Extracted {file_conformer_count:,} conformers from {h5_path.name}"
            )

        self.logger.info("=" * 70)
        self.logger.info(f"Total conformers extracted: {conformer_count:,}")
        self.logger.info(f"  Neutral molecules: {neutral_count:,}")
        self.logger.info(f"  Charged molecules: {charged_count:,}")
        if skipped_nan > 0:
            self.logger.info(f"  Skipped (NaN values): {skipped_nan:,}")
        if skipped_unknown_element > 0:
            self.logger.info(f"  Skipped (unsupported elements): {skipped_unknown_element:,}")

        # Build features dictionary with object arrays for ragged data
        # CRITICAL: Use np.empty() + element assignment instead of np.array(list, dtype=object)
        # Evidence: np.array(list, dtype=object) corrupts inner array dtypes to object

        def _build_object_array(items: list) -> np.ndarray:
            """Build object array while preserving inner array dtypes."""
            arr = np.empty(len(items), dtype=object)
            for i, item in enumerate(items):
                arr[i] = item
            return arr

        features = {
            "atoms": _build_object_array(atoms_list),
            "coordinates": _build_object_array(coordinates_list),
            "energy": np.array(energy_list, dtype=np.float64),
            "formula": _build_object_array(formula_list),
            "molecular_charge": np.array(molecular_charge_list, dtype=np.int32),  # CRITICAL
            "charge_type": _build_object_array(charge_type_list),
            "subset": _build_object_array(subset_list),
        }

        # Add optional properties if any were extracted
        if any(f is not None for f in forces_list):
            features["forces"] = _build_object_array(forces_list)

        # Compute metadata
        atom_counts = [len(a) for a in atoms_list]
        metadata = {
            "total_conformers": conformer_count,
            "neutral_count": neutral_count,
            "charged_count": charged_count,
            "skipped_nan": skipped_nan,
            "skipped_unknown_element": skipped_unknown_element,
            "mean_atoms": np.mean(atom_counts) if atom_counts else 0,
            "max_atoms": max(atom_counts) if atom_counts else 0,
            "min_atoms": min(atom_counts) if atom_counts else 0,
            "properties_extracted": list(features.keys()),
            "has_forces": any(f is not None for f in forces_list),
            "energy_units": "hartree",
            "force_units": "hartree/angstrom",
            "coordinate_units": "angstrom",
        }

        return features, metadata

    def _build_npz(
        self, features: dict[str, np.ndarray], metadata: dict[str, Any], output_path: Path
    ) -> None:
        """
        Build compressed .npz file from extracted features.

        Args:
            features: Dictionary of feature arrays
            metadata: Dictionary of metadata
            output_path: Path for output .npz file
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata to features
        features["_metadata"] = np.array([str(metadata)])

        # Save as compressed NPZ
        self.logger.info("=" * 70)
        self.logger.info("STEP 3: Building NPZ file")
        self.logger.info("=" * 70)
        self.logger.info(f"Saving to: {output_path}")
        np.savez_compressed(str(output_path), **features)

        # Log file size
        size_mb = output_path.stat().st_size / (1024**2)
        self.logger.info(f"✓ Created {output_path.name} ({size_mb:.2f} MB)")
        self.logger.info(f"  Total conformers: {metadata.get('total_conformers', 'N/A'):,}")
        self.logger.info(f"  Neutral: {metadata.get('neutral_count', 'N/A'):,}")
        self.logger.info(f"  Charged: {metadata.get('charged_count', 'N/A'):,}")
        self.logger.info(f"  Properties: {metadata.get('properties_extracted', [])}")
