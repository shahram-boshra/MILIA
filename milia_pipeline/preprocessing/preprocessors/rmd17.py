# milia_pipeline/preprocessing/preprocessors/rmd17.py

"""
rMD17 (Revised MD17) Preprocessor
=================================

Preprocessor for rMD17 quantum chemistry dataset (tar.bz2 archive with NPZ files).

Extracts the rMD17 tar.bz2 archive, parses individual molecular NPZ files,
converts units from kcal/mol to Hartree, and creates a unified .npz file
compatible with miliaDataset.

rMD17 Dataset Information:
--------------------------
- Source: Materials Cloud Archive (DOI: 10.24435/materialscloud:wy-kn)
- Download URL: https://archive.materialscloud.org/record/file?filename=rmd17.tar.bz2&record_id=466
- File: rmd17.tar.bz2 (~100 MB compressed)
- Contents: ~100,000 conformations for each of 10 small organic molecules
- Format: tar.bz2 archive containing 10 NPZ files (one per molecule)
- Method: PBE/def2-SVP with very tight SCF convergence (ORCA)

Archive Structure:
------------------
rmd17.tar.bz2
└── rmd17/
    ├── npz_data/
    │   ├── rmd17_aspirin.npz
    │   ├── rmd17_azobenzene.npz
    │   ├── rmd17_benzene.npz
    │   ├── rmd17_ethanol.npz
    │   ├── rmd17_malonaldehyde.npz
    │   ├── rmd17_naphthalene.npz
    │   ├── rmd17_paracetamol.npz
    │   ├── rmd17_salicylic.npz
    │   ├── rmd17_toluene.npz
    │   └── rmd17_uracil.npz
    └── splits/
        └── (CSV files for train/test splits)

NPZ File Structure (per molecule):
----------------------------------
Each rmd17_{molecule}.npz contains:
- 'nuclear_charges': Shape (N_atoms,) - atomic numbers [uint8]
- 'coords': Shape (N_conf, N_atoms, 3) - positions [Angstrom, float32]
- 'energies': Shape (N_conf,) - total energies [kcal/mol, float64]
- 'forces': Shape (N_conf, N_atoms, 3) - atomic forces [kcal/mol/Angstrom, float32]
- 'old_indices': Shape (N_conf,) - indices in original MD17 [int]
- 'old_energies': Shape (N_conf,) - original MD17 energies [kcal/mol, float64]
- 'old_forces': Shape (N_conf, N_atoms, 3) - original MD17 forces [kcal/mol/Angstrom, float32]

CRITICAL: Unit Conversion
-------------------------
rMD17 stores energies in kcal/mol and forces in kcal/mol/Angstrom.
MILIA pipeline standardizes to Hartree and Hartree/Angstrom.
Conversion: 1 kcal/mol = 0.00159360143764 Hartree (NIST CODATA 2018)

CRITICAL: Training Warning from Paper
--------------------------------------
- DO NOT train on more than 1000 samples due to autocorrelation in MD time-series
- Data from MD simulations at 500K are NOT independent samples
- Original MD17 data published with 50K samples is meaningless

Molecules (10 total):
---------------------
- aspirin (C9H8O4, 21 atoms, 100,000 conformations)
- azobenzene (C12H10N2, 24 atoms, 99,988 conformations - 11 failed DFT)
- benzene (C6H6, 12 atoms, 100,000 conformations)
- ethanol (C2H6O, 9 atoms, 100,000 conformations)
- malonaldehyde (C3H4O2, 9 atoms, 100,000 conformations)
- naphthalene (C10H8, 18 atoms, 100,000 conformations)
- paracetamol (C8H9NO2, 20 atoms, 100,000 conformations)
- salicylic (C7H6O3, 16 atoms, 100,000 conformations)
- toluene (C7H8, 15 atoms, 100,000 conformations)
- uracil (C4H4N2O2, 12 atoms, 100,000 conformations)

Reference: Christensen, A.S. & von Lilienfeld, O.A. (2020). On the role of
           gradients for machine learning of molecular energies and forces.
           Mach. Learn.: Sci. Technol. 1, 045018.

Author: MILIA Pipeline Team
Version: 1.0
Date: January 2026
"""

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry

logger = logging.getLogger(__name__)


# Unit conversion constant: 1 kcal/mol = 0.00159360143764 Hartree
# Reference: NIST CODATA 2018 recommended values
# This is used to convert rMD17 energies (kcal/mol) to Hartree for MILIA standardization
KCAL_MOL_TO_HARTREE = 0.00159360143764


# List of molecules in rMD17 dataset (alphabetical order)
RMD17_MOLECULES = [
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
]


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


@PreprocessorRegistry.register("RMD17")
class RMD17Preprocessor(BasePreprocessor):
    """
    Preprocessor for rMD17 (Revised MD17) quantum chemistry dataset.

    Pipeline:
    ---------
    1. Extract tar.bz2 archive to temporary directory
    2. Load each molecular NPZ file (10 molecules)
    3. Convert units: kcal/mol → Hartree, kcal/mol/Å → Hartree/Å
    4. Combine all conformers into unified NPZ file
    5. Clean up temporary extraction directory

    Configuration:
    --------------
    Required keys:
        - raw_archive_path: Path to rmd17.tar.bz2
        - output_npz_path: Path for output .npz file

    Optional keys:
        - molecules_to_include: List of molecule names to include (default: all)
        - max_conformers_per_molecule: Max conformers per molecule (default: all)
        - include_old_data: Whether to include old_energies/old_forces (default: False)
        - cleanup_temp: Whether to clean up temp extraction dir (default: True)
    """

    def _validate_config(self) -> None:
        """Validate rMD17-specific configuration requirements."""
        required_keys = ["raw_archive_path", "output_npz_path"]

        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(
                    f"rMD17 preprocessor missing required config key: {key}", config_key=key
                )

        # Validate archive exists
        archive_path = Path(self.config["raw_archive_path"])
        if not archive_path.exists():
            raise ConfigurationError(
                f"rMD17 archive not found: {archive_path}", config_key="raw_archive_path"
            )

        # Validate archive is a tar.bz2 file
        if not archive_path.name.endswith(".tar.bz2"):
            self.logger.warning(
                f"rMD17 archive does not have .tar.bz2 extension: {archive_path.name}"
            )

        # Validate molecules_to_include if specified
        if "molecules_to_include" in self.config:
            molecules = self.config["molecules_to_include"]
            if molecules is not None:
                invalid_molecules = [m for m in molecules if m not in RMD17_MOLECULES]
                if invalid_molecules:
                    raise ConfigurationError(
                        f"Unknown rMD17 molecules specified: {invalid_molecules}. "
                        f"Valid molecules: {RMD17_MOLECULES}",
                        config_key="molecules_to_include",
                    )

    def preprocess(self) -> Path:
        """
        Main preprocessing pipeline for rMD17 dataset.

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
            include_old_data = self.config.get("include_old_data", False)
            cleanup_temp = self.config.get("cleanup_temp", True)

            self.logger.info("=" * 60)
            self.logger.info("rMD17 Preprocessing Pipeline")
            self.logger.info("=" * 60)
            self.logger.info(f"Archive: {archive_path}")
            self.logger.info(f"Output: {output_npz}")
            if molecules_to_include:
                self.logger.info(f"Molecules: {molecules_to_include}")
            else:
                self.logger.info(f"Molecules: all ({len(RMD17_MOLECULES)})")
            if max_conformers:
                self.logger.info(f"Max conformers per molecule: {max_conformers}")
            self.logger.info(f"Include old MD17 data: {include_old_data}")
            self.logger.info("Unit conversion: kcal/mol → Hartree")
            self.logger.info("=" * 60)

            # Step 1: Extract archive
            self.logger.info("Step 1: Extracting tar.bz2 archive...")
            extracted_dir = self._extract_archive(archive_path)

            try:
                # Step 2: Parse NPZ files and combine
                self.logger.info("Step 2: Parsing molecular NPZ files...")
                features, metadata = self._parse_rmd17_npz_files(
                    extracted_dir=extracted_dir,
                    molecules_to_include=molecules_to_include,
                    max_conformers=max_conformers,
                    include_old_data=include_old_data,
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
            self.logger.info("rMD17 Preprocessing Complete!")
            self.logger.info("=" * 60)

            return output_npz

        except Exception as e:
            raise DataProcessingError(
                f"rMD17 preprocessing failed: {e}", operation="rmd17_preprocessing"
            ) from e

    def _extract_archive(self, archive_path: Path) -> Path:
        """
        Extract rMD17 tar.bz2 archive to temporary directory.

        Args:
            archive_path: Path to rmd17.tar.bz2

        Returns:
            Path to directory containing rmd17_*.npz files

        Note:
            The Materials Cloud archive structure is:
            rmd17.tar.bz2 -> rmd17/npz_data/rmd17_*.npz
                          -> rmd17/splits/*.csv
            This method handles multiple possible structures for robustness.
        """
        # Create temporary directory for extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="rmd17_extract_"))

        self.logger.info(f"  Extracting to: {temp_dir}")

        try:
            with tarfile.open(archive_path, "r:bz2") as tar:
                # Get list of members to extract
                members = tar.getmembers()
                self.logger.info(f"  Archive contains {len(members)} items")

                # Extract all
                tar.extractall(temp_dir)

            # The archive extracts to rmd17/ subdirectory with npz_data/ inside
            # Structure: rmd17.tar.bz2 -> rmd17/npz_data/rmd17_*.npz
            # Reference: kgcnn MD17RevisedDataset uses rmd17/npz_data/ path
            rmd17_dir = None
            npz_files = []

            # Priority 1: Check standard structure (rmd17/npz_data/)
            npz_data_dir = temp_dir / "rmd17" / "npz_data"
            if npz_data_dir.exists():
                npz_files = list(npz_data_dir.glob("rmd17_*.npz"))
                if npz_files:
                    rmd17_dir = npz_data_dir

            # Priority 2: Check rmd17/ directly (alternative archive structure)
            if not npz_files:
                rmd17_subdir = temp_dir / "rmd17"
                if rmd17_subdir.exists():
                    npz_files = list(rmd17_subdir.glob("rmd17_*.npz"))
                    if npz_files:
                        rmd17_dir = rmd17_subdir

            # Priority 3: Check temp_dir directly (flat extraction)
            if not npz_files:
                npz_files = list(temp_dir.glob("rmd17_*.npz"))
                if npz_files:
                    rmd17_dir = temp_dir

            # Priority 4: Recursive search for any directory containing rmd17_*.npz
            if not npz_files:
                for npz_file in temp_dir.rglob("rmd17_*.npz"):
                    rmd17_dir = npz_file.parent
                    npz_files = list(rmd17_dir.glob("rmd17_*.npz"))
                    break

            # Final validation
            self.logger.info(f"  Found {len(npz_files)} NPZ files in {rmd17_dir}")

            if len(npz_files) == 0:
                raise DataProcessingError(
                    "No rmd17_*.npz files found after extraction", operation="archive_extraction"
                )

            return rmd17_dir

        except tarfile.TarError as e:
            # Cleanup on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise DataProcessingError(
                f"Failed to extract rMD17 archive: {e}", operation="archive_extraction"
            ) from e

    def _parse_rmd17_npz_files(
        self,
        extracted_dir: Path,
        molecules_to_include: list[str] | None = None,
        max_conformers: int | None = None,
        include_old_data: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Parse rMD17 NPZ files and combine into unified feature dictionary.

        Args:
            extracted_dir: Path to directory containing rmd17_*.npz files
            molecules_to_include: List of molecule names to include (None = all)
            max_conformers: Max conformers per molecule (None = all)
            include_old_data: Whether to include old_energies/old_forces

        Returns:
            Tuple of (features_dict, metadata_dict)
        """
        # Determine which molecules to process
        if molecules_to_include is None:
            molecules_to_include = RMD17_MOLECULES.copy()

        # Initialize storage lists
        atoms_list = []  # List of atomic number arrays
        coordinates_list = []  # List of coordinate arrays
        energies_list = []  # List of energy values (will be converted to Hartree)
        forces_list = []  # List of force arrays (will be converted to Hartree/Å)
        molecule_name_list = []  # List of molecule names

        # Optional old data lists
        old_energies_list = [] if include_old_data else None
        old_forces_list = [] if include_old_data else None
        old_indices_list = [] if include_old_data else None

        total_conformers = 0
        molecule_counts = {}

        for molecule in molecules_to_include:
            npz_path = extracted_dir / f"rmd17_{molecule}.npz"

            if not npz_path.exists():
                self.logger.warning(f"  NPZ file not found for {molecule}: {npz_path}")
                continue

            self.logger.info(f"  Processing {molecule}...")

            try:
                data = np.load(npz_path)

                # Get basic info
                nuclear_charges = data["nuclear_charges"]  # Shape (N_atoms,) - constant
                coords = data["coords"]  # Shape (N_conf, N_atoms, 3)
                energies = data["energies"]  # Shape (N_conf,) in kcal/mol
                forces = data["forces"]  # Shape (N_conf, N_atoms, 3) in kcal/mol/Å

                n_conformers = coords.shape[0]
                n_atoms = len(nuclear_charges)

                # Apply conformer limit if specified
                if max_conformers is not None:
                    n_conformers = min(n_conformers, max_conformers)

                self.logger.info(f"    Atoms: {n_atoms}, Conformers: {n_conformers}")

                # Convert units: kcal/mol → Hartree
                energies_hartree = energies[:n_conformers] * KCAL_MOL_TO_HARTREE
                forces_hartree = forces[:n_conformers] * KCAL_MOL_TO_HARTREE

                # Load old data if requested
                if include_old_data:
                    old_energies = data.get("old_energies", None)
                    old_forces = data.get("old_forces", None)
                    old_indices = data.get("old_indices", None)

                # Process each conformer
                for conf_idx in range(n_conformers):
                    # Store atomic numbers (constant for all conformers of this molecule)
                    # CRITICAL: Explicitly cast to uint8 for proper dtype preservation
                    atoms_list.append(np.ascontiguousarray(nuclear_charges, dtype=np.uint8))

                    # Store coordinates (ensure contiguous float32)
                    coordinates_list.append(
                        np.ascontiguousarray(coords[conf_idx], dtype=np.float32)
                    )

                    # Store energy (already converted to Hartree)
                    energies_list.append(energies_hartree[conf_idx])

                    # Store forces (ensure contiguous float32, already in Hartree/Å)
                    forces_list.append(
                        np.ascontiguousarray(forces_hartree[conf_idx], dtype=np.float32)
                    )

                    # Store molecule name
                    molecule_name_list.append(molecule)

                    # Store old data if requested
                    if include_old_data:
                        if old_energies is not None:
                            old_energies_list.append(old_energies[conf_idx])
                        else:
                            old_energies_list.append(np.nan)

                        if old_forces is not None:
                            old_forces_list.append(
                                np.ascontiguousarray(old_forces[conf_idx], dtype=np.float32)
                            )
                        else:
                            old_forces_list.append(None)

                        if old_indices is not None:
                            old_indices_list.append(int(old_indices[conf_idx]))
                        else:
                            old_indices_list.append(-1)

                total_conformers += n_conformers
                molecule_counts[molecule] = n_conformers

                data.close()

            except Exception as e:
                self.logger.error(f"  Error processing {molecule}: {e}")
                raise

        self.logger.info(f"  Total conformers extracted: {total_conformers:,}")

        # Build features dictionary with object arrays for ragged data
        features = {
            "atoms": _build_object_array(atoms_list),
            "coordinates": _build_object_array(coordinates_list),
            "energies": np.array(energies_list, dtype=np.float64),
            "forces": _build_object_array(forces_list),
            "molecule_name": _build_object_array(molecule_name_list),
        }

        # Add old data if requested
        if include_old_data:
            if old_energies_list:
                features["old_energies"] = np.array(old_energies_list, dtype=np.float64)
            if old_forces_list and any(f is not None for f in old_forces_list):
                features["old_forces"] = _build_object_array(old_forces_list)
            if old_indices_list:
                features["old_indices"] = np.array(old_indices_list, dtype=np.int64)

        # Compute metadata
        atom_counts = [len(a) for a in atoms_list]
        metadata = {
            "total_conformers": total_conformers,
            "molecules_included": molecules_to_include,
            "molecule_counts": molecule_counts,
            "mean_atoms": np.mean(atom_counts) if atom_counts else 0,
            "max_atoms": max(atom_counts) if atom_counts else 0,
            "min_atoms": min(atom_counts) if atom_counts else 0,
            "properties_extracted": list(features.keys()),
            "energy_units": "hartree",
            "force_units": "hartree/angstrom",
            "original_energy_units": "kcal/mol",
            "conversion_factor": KCAL_MOL_TO_HARTREE,
            "level_of_theory": "PBE/def2-SVP",
            "source": "rMD17 (Christensen & von Lilienfeld, 2020)",
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
