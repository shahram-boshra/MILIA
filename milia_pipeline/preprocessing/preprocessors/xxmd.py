# milia_pipeline/preprocessing/preprocessors/xxmd.py

"""
xxMD (Extended Excited-state Molecular Dynamics) Preprocessor
=============================================================

Preprocessor for xxMD quantum chemistry dataset (ZIP archive with extended XYZ files).

Extracts the xxMD-main.zip archive, parses extended XYZ files using ASE (Atomic
Simulation Environment), converts units from eV to Hartree, and creates a unified
.npz file compatible with miliaDataset.

xxMD Dataset Information:
-------------------------
- Source: Zenodo (DOI: 10.5281/zenodo.10393859)
- GitHub: https://github.com/zpengmei/xxMD
- Download URL: https://zenodo.org/api/records/10393859/files/xxMD-main.zip/content
- File: xxMD-main.zip (~XXX MB compressed)
- Contents: Nonadiabatic dynamics trajectories for 4 photochemically active molecules
- Format: ZIP archive containing extended XYZ files (processed with ASE)
- Method (xxMD-DFT): M06 exchange-correlation functional (spin-polarized KS-DFT)

Archive Structure:
------------------
xxMD-main.zip
└── xxMD-main/
    └── xxMD-DFT/
        ├── azo/
        │   └── azo.zip          ← Nested ZIP containing XYZ files
        │       ├── train.xyz
        │       ├── val.xyz
        │       └── test.xyz
        ├── dia/
        │   └── dia.zip
        ├── mal/
        │   └── mal.zip
        └── sti/
            └── sti.zip

Note: Each molecule directory contains a nested ZIP file that must be extracted
to access the actual extended XYZ data files.

Extended XYZ Format (ASE):
--------------------------
Each frame contains:
- Lattice: Unit cell (not used for molecules)
- Properties: Column definitions (species:S:1:pos:R:3:forces:R:3)
- energy: Total DFT energy (eV, ASE default units)
- forces: Atomic forces (eV/Angstrom, ASE default units)
- Per-atom columns: species, positions, forces

CRITICAL: Unit Conversion
-------------------------
xxMD stores energies in eV and forces in eV/Angstrom (ASE default units).
MILIA pipeline standardizes to Hartree and Hartree/Angstrom.
Conversion: 1 eV = 0.0367493 Hartree (1/27.211386245988)

Molecules (4 total):
--------------------
- azobenzene (C12H10N2, 24 atoms)
- dithiophene (C8H6S2, 16 atoms)
- malonaldehyde (C3H4O2, 9 atoms)
- stilbene (C14H12, 26 atoms)

Reference: Pengmei, Z., Liu, J. & Shu, Y. Beyond MD17: the reactive xxMD dataset.
           Sci Data 11, 222 (2024). https://doi.org/10.1038/s41597-024-03019-3

Author: MILIA Pipeline Team
Version: 1.0
Date: January 2026
"""

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry

logger = logging.getLogger(__name__)


# Unit conversion constant: 1 eV = 0.0367493 Hartree (1/27.211386245988)
# Reference: NIST CODATA 2018 recommended values
# This is used to convert xxMD energies (eV) to Hartree for MILIA standardization
EV_TO_HARTREE = 1.0 / 27.211386245988  # 0.0367493


# List of molecules in xxMD-DFT dataset
# CRITICAL: These are the ACTUAL directory names in the xxMD-main.zip archive
# The archive uses abbreviated names, NOT full molecule names:
#   azo = azobenzene (C12H10N2, 24 atoms)
#   dia = dithiophene (C8H6S2, 16 atoms) - note: 'dia' not 'dit'
#   mal = malonaldehyde (C3H4O2, 9 atoms)
#   sti = stilbene (C14H12, 26 atoms)
# Evidence: ls -la xxMD-main/xxMD-DFT/ shows: azo, dia, mal, sti
XXMD_MOLECULES = [
    "azo",
    "dia",
    "mal",
    "sti",
]

# Mapping from abbreviated names to full molecule names (for metadata/display)
XXMD_MOLECULE_FULL_NAMES = {
    "azo": "azobenzene",
    "dia": "dithiophene",
    "mal": "malonaldehyde",
    "sti": "stilbene",
}


# Split file names (temporal splits)
XXMD_SPLITS = ["train", "val", "test"]


def _build_object_array(items: list) -> np.ndarray:
    """
    Build object array while preserving inner array dtypes.

    CRITICAL: Using np.array(list, dtype=object) corrupts inner array dtypes.
    Evidence: np.array([arr_uint8], dtype=object)[0].dtype == object (WRONG)
    Fix: Use np.empty() + element assignment to preserve inner dtypes.

    Args:
        items: List of arrays or values to store

    Returns:
        Object array with preserved inner dtypes
    """
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _parse_extended_xyz_with_ase(xyz_path: Path) -> list[dict[str, Any]]:
    """
    Parse extended XYZ file using ASE.

    Args:
        xyz_path: Path to extended XYZ file

    Returns:
        List of dictionaries with atomic data for each frame:
        - 'atomic_numbers': numpy array of atomic numbers
        - 'positions': numpy array of positions (Angstrom)
        - 'energy': energy value (eV)
        - 'forces': numpy array of forces (eV/Angstrom) or None

    Raises:
        DataProcessingError: If ASE parsing fails
    """
    try:
        from ase.io import read as ase_read
    except ImportError as e:
        raise DataProcessingError(
            "ASE (Atomic Simulation Environment) is required for xxMD preprocessing. "
            "Install with: pip install ase",
            operation="ase_import",
        ) from e

    try:
        # Read all frames from extended XYZ file
        # ASE automatically handles extended XYZ format
        frames = ase_read(str(xyz_path), index=":", format="extxyz")

        # Handle single frame case
        if not isinstance(frames, list):
            frames = [frames]

        parsed_data = []
        for frame_idx, atoms in enumerate(frames):
            # Extract atomic numbers
            atomic_numbers = atoms.get_atomic_numbers()

            # Extract positions (always in Angstrom for ASE)
            positions = atoms.get_positions()

            # Extract energy from atoms.info or calculator
            energy = None
            if hasattr(atoms, "info") and "energy" in atoms.info:
                energy = atoms.info["energy"]
            elif atoms.calc is not None:
                try:
                    energy = atoms.get_potential_energy()
                except Exception:
                    pass

            # Extract forces from atoms.arrays or calculator
            forces = None
            if hasattr(atoms, "arrays") and "forces" in atoms.arrays:
                forces = atoms.arrays["forces"]
            elif atoms.calc is not None:
                try:
                    forces = atoms.get_forces()
                except Exception:
                    pass

            parsed_data.append(
                {
                    "atomic_numbers": np.array(atomic_numbers, dtype=np.uint8),
                    "positions": np.array(positions, dtype=np.float32),
                    "energy": energy,
                    "forces": np.array(forces, dtype=np.float32) if forces is not None else None,
                }
            )

        return parsed_data

    except Exception as e:
        raise DataProcessingError(
            f"Failed to parse extended XYZ file with ASE: {xyz_path}. Error: {e}",
            file_path=str(xyz_path),
            operation="ase_xyz_parsing",
        ) from e


@PreprocessorRegistry.register("XXMD")
class XXMDPreprocessor(BasePreprocessor):
    """
    Preprocessor for xxMD (Extended Excited-state Molecular Dynamics) dataset.

    Pipeline:
    ---------
    1. Extract ZIP archive to temporary directory
    2. Locate xxMD-DFT subdirectory with molecule folders
    3. Parse extended XYZ files using ASE for each molecule
    4. Convert units: eV → Hartree, eV/Å → Hartree/Å
    5. Combine all conformers into unified NPZ file
    6. Clean up temporary extraction directory

    Configuration:
    --------------
    Required keys:
        - raw_archive_path: Path to xxMD-main.zip
        - output_npz_path: Path for output .npz file

    Optional keys:
        - molecules_to_include: List of molecule names to include (default: all)
        - max_conformers_per_molecule: Max conformers per molecule (default: all)
        - include_splits: Whether to include split info (train/val/test) (default: True)
        - cleanup_temp: Whether to clean up temp extraction dir (default: True)
    """

    def _validate_config(self) -> None:
        """Validate xxMD-specific configuration requirements."""
        required_keys = ["raw_archive_path", "output_npz_path"]

        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(
                    f"xxMD preprocessor missing required config key: {key}", config_key=key
                )

        # Validate archive exists
        archive_path = Path(self.config["raw_archive_path"])
        if not archive_path.exists():
            raise ConfigurationError(
                f"xxMD archive not found: {archive_path}", config_key="raw_archive_path"
            )

        # Validate archive is a ZIP file
        if not archive_path.name.endswith(".zip"):
            self.logger.warning(f"xxMD archive does not have .zip extension: {archive_path.name}")

        # Validate molecules_to_include if specified
        if "molecules_to_include" in self.config:
            molecules = self.config["molecules_to_include"]
            if molecules is not None:
                invalid_molecules = [m for m in molecules if m not in XXMD_MOLECULES]
                if invalid_molecules:
                    valid_with_names = [
                        f"{abbr} ({XXMD_MOLECULE_FULL_NAMES[abbr]})" for abbr in XXMD_MOLECULES
                    ]
                    raise ConfigurationError(
                        f"Unknown xxMD molecules specified: {invalid_molecules}. "
                        f"Valid molecules (use abbreviated names): {valid_with_names}",
                        config_key="molecules_to_include",
                    )

    def preprocess(self) -> Path:
        """
        Main preprocessing pipeline for xxMD dataset.

        Returns:
            Path to the created .npz file

        Raises:
            DataProcessingError: If preprocessing fails
        """
        try:
            archive_path = Path(self.config["raw_archive_path"])
            output_npz = Path(self.config["output_npz_path"])

            # Get optional configuration
            molecules_to_include = self.config.get("molecules_to_include", None)
            max_conformers = self.config.get("max_conformers_per_molecule", None)
            include_splits = self.config.get("include_splits", True)
            cleanup_temp = self.config.get("cleanup_temp", True)

            self.logger.info("=" * 60)
            self.logger.info("xxMD Preprocessing Pipeline")
            self.logger.info("=" * 60)
            self.logger.info(f"Archive: {archive_path}")
            self.logger.info(f"Output: {output_npz}")
            if molecules_to_include:
                self.logger.info(f"Molecules: {molecules_to_include}")
            else:
                self.logger.info(f"Molecules: all ({len(XXMD_MOLECULES)})")
            if max_conformers:
                self.logger.info(f"Max conformers per molecule: {max_conformers}")
            self.logger.info(f"Include split info: {include_splits}")
            self.logger.info("Unit conversion: eV → Hartree")
            self.logger.info("=" * 60)

            # Check if output already exists
            if output_npz.exists():
                self.logger.info("=" * 60)
                self.logger.info("EXISTING .NPZ FILE DETECTED")
                self.logger.info("=" * 60)
                size_mb = output_npz.stat().st_size / (1024**2)
                self.logger.info(f"Found: {output_npz.name} ({size_mb:.2f} MB)")
                self.logger.info("Skipping preprocessing - file already exists")
                self.logger.info("Delete the file to regenerate, or use a different output path")
                self.logger.info("=" * 60)
                return output_npz

            # Step 1: Extract archive
            self.logger.info("Step 1: Extracting ZIP archive...")
            extracted_dir = self._extract_archive(archive_path)

            try:
                # Step 2: Parse XYZ files and combine
                self.logger.info("Step 2: Parsing extended XYZ files with ASE...")
                features, metadata = self._parse_xxmd_xyz_files(
                    extracted_dir=extracted_dir,
                    molecules_to_include=molecules_to_include,
                    max_conformers=max_conformers,
                    include_splits=include_splits,
                )

                # Step 3: Build unified NPZ file
                self.logger.info("Step 3: Building unified NPZ file...")
                self._build_npz(features, metadata, output_npz)

            finally:
                # Step 4: Cleanup
                if cleanup_temp and extracted_dir.exists():
                    self.logger.info("Step 4: Cleaning up temporary files...")
                    shutil.rmtree(extracted_dir)
                    self.logger.info(f"  Removed: {extracted_dir}")

            self.logger.info("=" * 60)
            self.logger.info("xxMD Preprocessing Complete!")
            self.logger.info("=" * 60)

            return output_npz

        except Exception as e:
            raise DataProcessingError(
                f"xxMD preprocessing failed: {e}", operation="xxmd_preprocessing"
            ) from e

    def _extract_archive(self, archive_path: Path) -> Path:
        """
        Extract xxMD ZIP archive to temporary directory.

        Args:
            archive_path: Path to xxMD-main.zip

        Returns:
            Path to directory containing xxMD-DFT folder with molecule subdirectories

        Note:
            The archive structure is:
            xxMD-main.zip -> xxMD-main/xxMD-DFT/{molecule}/{split}.xyz
        """
        # Create temporary directory for extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="xxmd_extract_"))

        self.logger.info(f"  Extracting to: {temp_dir}")

        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                # Get list of members
                members = zf.namelist()
                self.logger.info(f"  Archive contains {len(members)} items")

                # Extract all
                zf.extractall(temp_dir)

            # Find the xxMD-DFT directory
            xxmd_dft_dir = None

            # Priority 1: Standard structure (xxMD-main/xxMD-DFT/)
            standard_path = temp_dir / "xxMD-main" / "xxMD-DFT"
            if standard_path.exists():
                xxmd_dft_dir = standard_path

            # Priority 2: Alternative structure (xxMD-DFT/ directly)
            if not xxmd_dft_dir:
                alt_path = temp_dir / "xxMD-DFT"
                if alt_path.exists():
                    xxmd_dft_dir = alt_path

            # Priority 3: Search recursively
            if not xxmd_dft_dir:
                for path in temp_dir.rglob("xxMD-DFT"):
                    if path.is_dir():
                        xxmd_dft_dir = path
                        break

            if not xxmd_dft_dir:
                raise DataProcessingError(
                    "Could not find xxMD-DFT directory in archive", operation="archive_extraction"
                )

            # Validate molecule directories exist
            found_molecules = []

            # Debug: List actual contents of xxMD-DFT directory
            if xxmd_dft_dir.exists():
                actual_contents = list(xxmd_dft_dir.iterdir())
                self.logger.info(
                    f"  xxMD-DFT directory contents: {[p.name for p in actual_contents]}"
                )

                # Also check for case-insensitive matches
                actual_dirs = {p.name.lower(): p.name for p in actual_contents if p.is_dir()}
                self.logger.info(
                    f"  Available subdirectories (lowercase): {list(actual_dirs.keys())}"
                )

            for mol in XXMD_MOLECULES:
                mol_dir = xxmd_dft_dir / mol
                if mol_dir.exists():
                    found_molecules.append(mol)
                else:
                    # Try case-insensitive match
                    if mol.lower() in actual_dirs:
                        actual_name = actual_dirs[mol.lower()]
                        self.logger.info(f"  Found case-insensitive match: {mol} -> {actual_name}")
                        found_molecules.append(actual_name)

            self.logger.info(f"  Found xxMD-DFT directory: {xxmd_dft_dir}")
            self.logger.info(f"  Found molecules: {found_molecules}")

            if not found_molecules:
                raise DataProcessingError(
                    f"No molecule directories found in {xxmd_dft_dir}",
                    operation="archive_extraction",
                )

            # Extract nested ZIP files if present
            # xxMD archive structure: xxMD-main/xxMD-DFT/{mol}/{mol}.zip containing train.xyz, val.xyz, test.xyz
            for mol in found_molecules:
                mol_dir = xxmd_dft_dir / mol
                inner_zip = mol_dir / f"{mol}.zip"

                if inner_zip.exists():
                    self.logger.info(f"  Extracting nested archive: {inner_zip.name}")
                    try:
                        with zipfile.ZipFile(inner_zip, "r") as zf:
                            zf.extractall(mol_dir)
                        self.logger.info(f"    Extracted to: {mol_dir}")
                        # List extracted contents for debugging
                        extracted_files = [
                            f.name for f in mol_dir.iterdir() if f.is_file() and f.suffix == ".xyz"
                        ]
                        self.logger.info(f"    XYZ files found: {extracted_files}")
                    except zipfile.BadZipFile as e:
                        self.logger.warning(f"    Failed to extract {inner_zip.name}: {e}")

            return xxmd_dft_dir

        except zipfile.BadZipFile as e:
            # Cleanup on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise DataProcessingError(
                f"Failed to extract xxMD archive: {e}", operation="archive_extraction"
            ) from e

    def _parse_xxmd_xyz_files(
        self,
        extracted_dir: Path,
        molecules_to_include: list[str] | None = None,
        max_conformers: int | None = None,
        include_splits: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Parse xxMD extended XYZ files and combine into unified feature dictionary.

        Args:
            extracted_dir: Path to directory containing molecule folders
            molecules_to_include: List of molecule names to include (None = all)
            max_conformers: Max conformers per molecule (None = all)
            include_splits: Whether to include split information

        Returns:
            Tuple of (features_dict, metadata_dict)
        """
        # Determine which molecules to process
        if molecules_to_include is None:
            molecules_to_include = XXMD_MOLECULES.copy()

        # Initialize storage lists
        atoms_list = []  # List of atomic number arrays
        coordinates_list = []  # List of coordinate arrays
        energies_list = []  # List of energy values (converted to Hartree)
        forces_list = []  # List of force arrays (converted to Hartree/Å)
        molecule_name_list = []  # List of molecule names
        split_list = []  # List of split labels (train/val/test)

        total_conformers = 0
        molecule_counts = {}
        split_counts = {"train": 0, "val": 0, "test": 0}
        skipped_no_energy = 0

        for molecule in molecules_to_include:
            mol_dir = extracted_dir / molecule

            if not mol_dir.exists():
                self.logger.warning(f"  Molecule directory not found: {mol_dir}")
                continue

            self.logger.info(f"  Processing {molecule}...")
            mol_conformer_count = 0

            # Process each split file
            # xxMD XYZ files follow pattern: {mol}_{split}_uks.xyz (e.g., azo_train_uks.xyz)
            for split in XXMD_SPLITS:
                # Try multiple filename patterns for robustness
                possible_patterns = [
                    f"{molecule}_{split}_uks.xyz",  # Primary: azo_train_uks.xyz
                    f"{split}.xyz",  # Fallback: train.xyz
                    f"{molecule}_{split}.xyz",  # Alternative: azo_train.xyz
                ]

                xyz_path = None
                for pattern in possible_patterns:
                    candidate = mol_dir / pattern
                    if candidate.exists():
                        xyz_path = candidate
                        break

                if xyz_path is None:
                    self.logger.debug(
                        f"    Split file not found for {split}, tried: {possible_patterns}"
                    )
                    continue

                self.logger.info(f"    Parsing {xyz_path.name}...")

                try:
                    frames = _parse_extended_xyz_with_ase(xyz_path)

                    for frame_data in frames:
                        # Check conformer limit per molecule
                        if max_conformers is not None and mol_conformer_count >= max_conformers:
                            break

                        # Skip frames without energy
                        if frame_data["energy"] is None:
                            skipped_no_energy += 1
                            continue

                        # Store atomic numbers (ensure contiguous uint8)
                        atoms_list.append(
                            np.ascontiguousarray(frame_data["atomic_numbers"], dtype=np.uint8)
                        )

                        # Store coordinates (ensure contiguous float32)
                        coordinates_list.append(
                            np.ascontiguousarray(frame_data["positions"], dtype=np.float32)
                        )

                        # Convert energy from eV to Hartree
                        energy_hartree = frame_data["energy"] * EV_TO_HARTREE
                        energies_list.append(energy_hartree)

                        # Convert forces from eV/Å to Hartree/Å (if available)
                        if frame_data["forces"] is not None:
                            forces_hartree = frame_data["forces"] * EV_TO_HARTREE
                            forces_list.append(
                                np.ascontiguousarray(forces_hartree, dtype=np.float32)
                            )
                        else:
                            forces_list.append(None)

                        # Store molecule name (use full name for readability)
                        full_name = XXMD_MOLECULE_FULL_NAMES.get(molecule, molecule)
                        molecule_name_list.append(full_name)

                        # Store split info
                        if include_splits:
                            split_list.append(split)
                            split_counts[split] += 1

                        mol_conformer_count += 1
                        total_conformers += 1

                    self.logger.info(f"      Parsed {len(frames)} frames from {split}.xyz")

                except Exception as e:
                    self.logger.error(f"    Error parsing {xyz_path}: {e}")
                    raise

                # Check if we've reached the limit
                if max_conformers is not None and mol_conformer_count >= max_conformers:
                    self.logger.info(f"    Reached max conformers limit ({max_conformers})")
                    break

            molecule_counts[molecule] = mol_conformer_count
            self.logger.info(f"    Total conformers for {molecule}: {mol_conformer_count}")

        self.logger.info(f"  Total conformers extracted: {total_conformers:,}")
        if skipped_no_energy > 0:
            self.logger.info(f"  Skipped (no energy): {skipped_no_energy:,}")

        # Build features dictionary with object arrays for ragged data
        features = {
            "atoms": _build_object_array(atoms_list),
            "coordinates": _build_object_array(coordinates_list),
            "energy": np.array(energies_list, dtype=np.float64),
            "molecule_name": _build_object_array(molecule_name_list),
        }

        # Add forces if any were extracted
        if any(f is not None for f in forces_list):
            features["forces"] = _build_object_array(forces_list)

        # Add split info if requested
        if include_splits and split_list:
            features["split"] = _build_object_array(split_list)

        # Compute metadata
        atom_counts = [len(a) for a in atoms_list]
        metadata = {
            "version": "1.0",
            "dataset_name": "XXMD",
            "subset": "xxMD-DFT",
            "total_conformers": total_conformers,
            "molecules_included": molecules_to_include,
            "molecule_counts": molecule_counts,
            "split_counts": split_counts if include_splits else {},
            "skipped_no_energy": skipped_no_energy,
            "mean_atoms": np.mean(atom_counts) if atom_counts else 0,
            "max_atoms": max(atom_counts) if atom_counts else 0,
            "min_atoms": min(atom_counts) if atom_counts else 0,
            "properties_extracted": list(features.keys()),
            "energy_units": "hartree",
            "force_units": "hartree/angstrom",
            "original_energy_units": "eV",
            "original_force_units": "eV/angstrom",
            "conversion_factor": EV_TO_HARTREE,
            "level_of_theory": "M06 (spin-polarized KS-DFT)",
            "coordinate_units": "angstrom",
            "source": "xxMD (Pengmei, Liu, Shu. Sci Data 2024)",
            "doi": "10.1038/s41597-024-03019-3",
            "zenodo_doi": "10.5281/zenodo.10393859",
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
        self.logger.info(f"  Saving to: {output_path}")
        np.savez_compressed(str(output_path), **features)

        # Log file size
        size_mb = output_path.stat().st_size / (1024**2)
        self.logger.info(f"  ✓ Created {output_path.name} ({size_mb:.2f} MB)")
        self.logger.info(f"    Total conformers: {metadata.get('total_conformers', 'N/A'):,}")
        self.logger.info(f"    Molecules: {metadata.get('molecules_included', [])}")
        self.logger.info(f"    Energy units: {metadata.get('energy_units', 'N/A')}")
        self.logger.info(f"    Properties: {metadata.get('properties_extracted', [])}")
