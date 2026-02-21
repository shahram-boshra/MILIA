# milia_pipeline/preprocessing/preprocessors/ani2x.py

"""
ANI-2x Preprocessor
===================

Preprocessor for ANI-2x quantum chemistry dataset (HDF5 format).

Parses ANI-2x HDF5 file structure, extracts molecular data from chemical isomer
groups, and creates .npz file compatible with miliaDataset.

ANI-2x Dataset Information:
---------------------------
- Source: Zenodo (https://zenodo.org/records/10108942)
- File: ANI-2x-wB97X-631Gd.tar.gz (contains .h5 file)
- Contents: DFT conformations for organic molecules (H, C, N, O, S, F, Cl)
- Format: HDF5 with molecular groups containing conformations
- Method: Active learning with ωB97X/6-31G(d) DFT

HDF5 Structure (based on ANI family format):
--------------------------------------------
Each molecular group (chemical isomer) contains:
- 'species' or 'atomic_numbers': Atomic numbers [uint8]
- 'coordinates': Positions [Angstrom, float32]
- 'energies': DFT energies [Hartree, float64]
- 'forces': Atomic forces [Hartree/Angstrom, float32] (if available)

Where: Nc = number of conformations, Na = number of atoms

Key Differences from ANI-1x:
- Extended elements: H, C, N, O, S, F, Cl (vs H, C, N, O for ANI-1x)
- Torsional refinement training applied
- Same HDF5 structure pattern as ANI-1x

Reference: Devereux, C., Smith, J.S., et al. Extending the Applicability of the
           ANI Deep Learning Molecular Potential to Sulfur and Halogens.
           J. Chem. Theory Comput. 2020, 16, 7, 4192-4202.

Author: MILIA Pipeline Team
Version: 1.0
Date: January 2026
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry

logger = logging.getLogger(__name__)


# ANI-2x supports 7 elements: H, C, N, O, S, F, Cl
# Atomic numbers for validation
ANI2X_SUPPORTED_ELEMENTS = {1, 6, 7, 8, 9, 16, 17}  # H, C, N, O, F, S, Cl


def iter_data_buckets_ani2x(h5filename: str, keys: list[str] = None) -> dict[str, Any]:
    """
    Iterate over buckets of data in ANI-2x HDF5 file.

    This function implements the same pattern as ANI-1x dataloader,
    yielding dictionaries with atomic numbers, coordinates, and requested properties
    for each conformer, filtering out entries with NaN values.

    Args:
        h5filename: Path to ANI-2x HDF5 file
        keys: List of property keys to load (default: ['energies'])

    Yields:
        Dict with:
            - 'atomic_numbers': Shape (Na,) - atomic numbers for this conformer
            - 'coordinates': Shape (Na, 3) - positions for this conformer
            - 'molecule_id': String identifier for the molecular group
            - Plus any requested property keys

    Note: ANI-2x may use 'species' or 'atomic_numbers' key, and 'energies' key
          instead of ANI-1x's 'wb97x_dz.energy' pattern.
    """
    import h5py

    if keys is None:
        keys = ["energies"]

    with h5py.File(h5filename, "r") as f:
        for mol_name in f:
            mol_group = f[mol_name]

            # Get atomic numbers - ANI-2x may use 'species' or 'atomic_numbers'
            # Try 'species' first (ANI-2x convention), then 'atomic_numbers' (ANI-1x convention)
            if "species" in mol_group:
                atomic_numbers_all = np.array(mol_group["species"], dtype=np.uint8)
            elif "atomic_numbers" in mol_group:
                atomic_numbers_all = np.array(mol_group["atomic_numbers"], dtype=np.uint8)
            else:
                logger.warning(f"Skipping molecule {mol_name}: no species/atomic_numbers key found")
                continue

            # Get coordinates (always present)
            coordinates_all = np.array(mol_group["coordinates"], dtype=np.float32)

            # Get number of conformations
            n_conformers = coordinates_all.shape[0]

            # Load requested properties
            properties = {}
            valid_mask = np.ones(n_conformers, dtype=bool)

            # Define expected dtypes for each property based on ANI-2x HDF5 specification
            property_dtypes = {
                "energies": np.float64,  # Hartree, high precision
                "forces": np.float32,  # Hartree/Angstrom
            }

            for key in keys:
                if key in mol_group:
                    # Use explicit dtype if known, otherwise let numpy infer
                    expected_dtype = property_dtypes.get(key)
                    if expected_dtype is not None:
                        prop_data = np.array(mol_group[key], dtype=expected_dtype)
                    else:
                        prop_data = np.array(mol_group[key])
                    properties[key] = prop_data

                    # Check for NaN values and update mask
                    if np.issubdtype(prop_data.dtype, np.floating):
                        if prop_data.ndim == 1:
                            valid_mask &= ~np.isnan(prop_data)
                        else:
                            # For multi-dimensional arrays, check if any NaN in each conformer
                            valid_mask &= ~np.any(
                                np.isnan(prop_data.reshape(n_conformers, -1)), axis=1
                            )

            # Yield data for each valid conformer
            for conf_idx in range(n_conformers):
                if not valid_mask[conf_idx]:
                    continue

                # Get atomic numbers for this conformer
                # Note: atomic_numbers may be (Nc, Na) or just (Na,) depending on file version
                if atomic_numbers_all.ndim == 2:
                    atomic_numbers = atomic_numbers_all[conf_idx]
                else:
                    atomic_numbers = atomic_numbers_all

                # Filter out padding zeros (atoms with Z=0)
                non_zero_mask = atomic_numbers > 0
                atomic_numbers = atomic_numbers[non_zero_mask]
                coordinates = coordinates_all[conf_idx][non_zero_mask]

                result = {
                    "atomic_numbers": atomic_numbers,
                    "coordinates": coordinates,
                    "molecule_id": mol_name,
                }

                # Add properties
                for key, prop_data in properties.items():
                    if prop_data.ndim == 1:
                        result[key] = prop_data[conf_idx]
                    else:
                        # For per-atom properties, filter by non_zero_mask
                        if prop_data.shape[1] == len(non_zero_mask):
                            result[key] = prop_data[conf_idx][non_zero_mask]
                        else:
                            result[key] = prop_data[conf_idx]

                yield result


@PreprocessorRegistry.register("ANI2x")
class ANI2xPreprocessor(BasePreprocessor):
    """
    Preprocessor for ANI-2x quantum chemistry dataset.

    Pipeline:
    ---------
    1. Extract HDF5 file from tar.gz archive (if needed)
    2. Open HDF5 file and iterate over molecular groups
    3. Extract conformer data (atomic_numbers, coordinates, properties)
    4. Filter out entries with NaN values
    5. Build .npz file (compressed format compatible with miliaDataset)

    Configuration:
    --------------
    Required keys:
        - raw_archive_path: Path to ANI-2x-wB97X-631Gd.tar.gz or .h5 file
        - output_npz_path: Path for output .npz file

    Optional keys:
        - num_molecules: Number of conformers to extract (None = all)
        - property_keys: List of properties to extract (default: ['energies'])

    Example:
    --------
    >>> config = {
    ...     'raw_archive_path': 'raw/ANI-2x-wB97X-631Gd.tar.gz',
    ...     'output_npz_path': 'processed/ani2x.npz',
    ...     'num_molecules': 10000,  # For testing
    ... }
    >>> preprocessor = ANI2xPreprocessor(config, logger)
    >>> output_path = preprocessor.run()

    Notes:
    ------
    - The ANI-2x tar.gz archive contains an HDF5 file
    - Full preprocessing may take significant time for large datasets
    - Use num_molecules for testing with a subset
    """

    # Default property keys to extract from HDF5
    DEFAULT_PROPERTY_KEYS = [
        "energies",
        "forces",
    ]

    def _validate_config(self) -> None:
        """Validate ANI-2x-specific configuration."""
        # Check required keys
        required_keys = ["raw_archive_path", "output_npz_path"]
        missing = [k for k in required_keys if k not in self.config]

        if missing:
            raise ConfigurationError(
                f"Missing required configuration keys: {missing}", config_key=", ".join(missing)
            )

        # Validate archive/H5 path exists
        archive_path = Path(self.config["raw_archive_path"])
        if not archive_path.exists():
            raise ConfigurationError(
                f"ANI-2x archive/HDF5 file not found: {archive_path}",
                config_key="raw_archive_path",
                actual_value=str(archive_path),
            )

        # Validate file extension
        valid_extensions = (".h5", ".hdf5", ".tar.gz", ".tgz")
        if not any(str(archive_path).lower().endswith(ext) for ext in valid_extensions):
            self.logger.warning(
                f"File extension not recognized: {archive_path.suffix}. "
                f"Expected one of {valid_extensions}. Proceeding anyway."
            )

        # Validate num_molecules if specified
        num_molecules = self.config.get("num_molecules")
        if num_molecules is not None and (not isinstance(num_molecules, int) or num_molecules < 1):
            raise ConfigurationError(
                f"num_molecules must be positive integer, got {num_molecules}",
                config_key="num_molecules",
                actual_value=num_molecules,
            )

        self.logger.debug("ANI-2x configuration validation passed")

    def preprocess(self) -> Path:
        """
        Execute ANI-2x preprocessing pipeline.

        Returns:
            Path to generated .npz file
        """
        archive_path = Path(self.config["raw_archive_path"])
        output_npz = Path(self.config["output_npz_path"])
        num_molecules = self.config.get("num_molecules", None)
        property_keys = self.config.get("property_keys", self.DEFAULT_PROPERTY_KEYS)

        # Check if output already exists (skip if so)
        if output_npz.exists():
            self.logger.info("=" * 70)
            self.logger.info("EXISTING .NPZ FILE DETECTED")
            self.logger.info("=" * 70)
            size_mb = output_npz.stat().st_size / (1024**2)
            self.logger.info(f"Found: {output_npz.name} ({size_mb:.2f} MB)")
            self.logger.info("Skipping preprocessing - file already exists")
            self.logger.info("Delete the file to regenerate, or use a different output path")
            self.logger.info("=" * 70)
            return output_npz

        try:
            # ================================================================
            # STEP 1: Extract HDF5 from archive if needed
            # ================================================================
            h5_path = self._get_h5_path(archive_path)

            # ================================================================
            # STEP 2: Parse HDF5 file and collect data
            # ================================================================
            self.logger.info("=" * 70)
            self.logger.info("STEP 2: Parsing ANI-2x HDF5 file")
            self.logger.info("=" * 70)
            self.logger.info(f"Source file: {h5_path}")
            self.logger.info(f"Properties to extract: {property_keys}")
            if num_molecules:
                self.logger.info(f"Maximum conformers: {num_molecules}")
            else:
                self.logger.info("Maximum conformers: ALL")

            features, parse_metadata = self._parse_ani2x_h5(
                h5_path=h5_path, property_keys=property_keys, max_conformers=num_molecules
            )

            # ================================================================
            # STEP 3: Build .npz file
            # ================================================================
            self.logger.info("=" * 70)
            self.logger.info("STEP 3: Building .npz file")
            self.logger.info("=" * 70)

            # Prepare comprehensive metadata
            npz_metadata = {
                "version": "1.0",
                "dataset_name": "ANI2x",
                "source": archive_path.name,
                "source_url": "https://zenodo.org/records/10108942/files/ANI-2x-wB97X-631Gd.tar.gz",
                "reference": "Devereux et al., J. Chem. Theory Comput. 2020, 16, 4192-4202",
                "doi": "10.1021/acs.jctc.0c00121",
                "file_format": ".h5 (HDF5 ANI-2x format)",
                "parser": "ANI2xPreprocessor",
                "preprocessing_version": "1.0",
                "coordinate_units": "angstrom",
                "energy_units": "hartree",
                "force_units": "hartree/angstrom",
                "supported_elements": "H, C, N, O, S, F, Cl",
                **parse_metadata,  # Include parsing statistics
            }

            self._build_npz(features=features, metadata=npz_metadata, output_path=output_npz)

            self.logger.info("=" * 70)
            self.logger.info("ANI-2x PREPROCESSING COMPLETE")
            self.logger.info("=" * 70)

            return output_npz

        except Exception as e:
            raise DataProcessingError(
                f"ANI-2x preprocessing failed: {e}", operation="ani2x_preprocessing"
            ) from e

    def _get_h5_path(self, archive_path: Path) -> Path:
        """
        Get path to HDF5 file, extracting from archive if needed.

        Args:
            archive_path: Path to archive (.tar.gz) or HDF5 file (.h5)

        Returns:
            Path to HDF5 file
        """
        # If already an HDF5 file, return as-is
        if str(archive_path).lower().endswith((".h5", ".hdf5")):
            self.logger.info(f"Using HDF5 file directly: {archive_path}")
            return archive_path

        # Extract from tar.gz archive
        self.logger.info("=" * 70)
        self.logger.info("STEP 1: Extracting HDF5 from tar.gz archive")
        self.logger.info("=" * 70)
        self.logger.info(f"Archive: {archive_path}")

        import tarfile

        # Extract to temp directory or alongside archive
        extract_dir = archive_path.parent / "ani2x_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        h5_path = None
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith((".h5", ".hdf5")):
                    self.logger.info(f"Extracting: {member.name}")
                    tar.extract(member, extract_dir)
                    h5_path = extract_dir / member.name
                    break

        if h5_path is None or not h5_path.exists():
            raise DataProcessingError(
                "No HDF5 file found in archive", operation="archive_extraction"
            )

        self.logger.info(f"Extracted HDF5: {h5_path}")
        return h5_path

    def _parse_ani2x_h5(
        self, h5_path: Path, property_keys: list[str], max_conformers: int | None = None
    ) -> tuple[dict[str, list], dict[str, Any]]:
        """
        Parse ANI-2x HDF5 file and extract molecular data.

        Args:
            h5_path: Path to ANI-2x HDF5 file
            property_keys: List of property keys to extract
            max_conformers: Maximum number of conformers to extract (None = all)

        Returns:
            Tuple of (features_dict, metadata_dict)
        """
        # Initialize storage for ragged arrays (variable-length per molecule)
        atoms_list = []  # List of atomic number arrays (integers)
        coordinates_list = []  # List of coordinate arrays
        energy_list = []  # List of energy values
        forces_list = []  # List of force arrays
        molecule_id_list = []  # List of molecule identifiers

        conformer_count = 0
        skipped_nan = 0
        skipped_unknown_element = 0

        self.logger.info("Starting HDF5 iteration...")

        for data in iter_data_buckets_ani2x(str(h5_path), keys=property_keys):
            # Check if we've reached the limit
            if max_conformers is not None and conformer_count >= max_conformers:
                break

            # Validate elements are in ANI-2x supported set
            atomic_numbers = np.ascontiguousarray(data["atomic_numbers"], dtype=np.uint8)
            unique_elements = set(atomic_numbers.tolist())
            if not unique_elements.issubset(ANI2X_SUPPORTED_ELEMENTS):
                unsupported = unique_elements - ANI2X_SUPPORTED_ELEMENTS
                self.logger.debug(f"Skipping conformer with unsupported elements: {unsupported}")
                skipped_unknown_element += 1
                continue

            # Ensure coordinates are explicit contiguous float32 arrays
            coordinates = np.ascontiguousarray(data["coordinates"], dtype=np.float32)

            # Store data
            atoms_list.append(atomic_numbers)
            coordinates_list.append(coordinates)
            molecule_id_list.append(data["molecule_id"])

            # Energy (required) - ANI-2x uses 'energies' key
            if "energies" in data:
                energy_list.append(data["energies"])
            else:
                energy_list.append(np.nan)

            # Forces (optional) - ensure contiguous float32 for proper NPZ serialization
            if "forces" in data and data["forces"] is not None:
                forces_list.append(np.ascontiguousarray(data["forces"], dtype=np.float32))
            else:
                forces_list.append(None)

            conformer_count += 1

            # Progress logging
            if conformer_count % 100000 == 0:
                self.logger.info(f"Processed {conformer_count:,} conformers...")

        self.logger.info(f"Finished parsing: {conformer_count:,} conformers extracted")
        if skipped_nan > 0:
            self.logger.info(f"Skipped due to NaN values: {skipped_nan:,}")
        if skipped_unknown_element > 0:
            self.logger.info(f"Skipped due to unsupported elements: {skipped_unknown_element:,}")

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
            "molecule_id": _build_object_array(molecule_id_list),
        }

        # Add optional properties if any were extracted
        if any(f is not None for f in forces_list):
            features["forces"] = _build_object_array(forces_list)

        # Compute metadata
        atom_counts = [len(a) for a in atoms_list]
        metadata = {
            "total_conformers": conformer_count,
            "skipped_nan": skipped_nan,
            "skipped_unknown_element": skipped_unknown_element,
            "mean_atoms": np.mean(atom_counts) if atom_counts else 0,
            "max_atoms": max(atom_counts) if atom_counts else 0,
            "min_atoms": min(atom_counts) if atom_counts else 0,
            "properties_extracted": list(features.keys()),
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
        self.logger.info(f"Saving to: {output_path}")
        np.savez_compressed(str(output_path), **features)

        # Log file size
        size_mb = output_path.stat().st_size / (1024**2)
        self.logger.info(f"✓ Created {output_path.name} ({size_mb:.2f} MB)")
        self.logger.info(f"  Total conformers: {metadata.get('total_conformers', 'N/A'):,}")
        self.logger.info(f"  Properties: {metadata.get('properties_extracted', [])}")
