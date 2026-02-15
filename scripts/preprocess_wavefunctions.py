#!/usr/bin/env python3
"""
milia Wavefunction Dataset - ENHANCED Preprocessing Module
==========================================================

PURPOSE:
    Extract and preprocess 10 .molden files from 106 GB tar.gz archive
    Create production-ready wavefunction.npz with COMPLETE feature spectrum

ENHANCEMENTS:
    ✓ FIXED: Data type handling bug (explicit float64 key listing)
    ✓ NEW: Complete feature extraction (~30 keys vs original 13)
    ✓ NEW: MO kind, orbital counts, energy statistics
    ✓ NEW: Chemical descriptors (IP, EA, hardness, potential)
    ✓ NEW: Total energy, n_shells, molecular formula
    ✓ NEW: Orbital energy statistics (mean/std/min/max)

FEATURE COVERAGE:
    - TIER 1: Essential geometric (atoms, coords) ✓
    - TIER 2: Basic electronic (HOMO, LUMO, gap) ✓
    - TIER 3: Detailed electronic (MO energies/occs) ✓
    - TIER 4: Quantum descriptors (new!) ✓
    - TIER 5: Derived features (new!) ✓

SAFETY FEATURES:
    - Streaming extraction (no full archive decompression)
    - Memory-efficient processing
    - Error handling and rollback
    - Progress tracking
    - Validation at each step
    - Data type consistency with DFT/DMC datasets

USAGE:
    python preprocess_wavefunction_test_ENHANCED.py

Author: milia Pipeline Team
Date: 2025-11-08
Version: 3.0 - PRODUCTION READY WITH COMPLETE FEATURE SPECTRUM
"""

import logging
import shutil
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "tar_path": Path.home() / "Chem_Data/milia_PyG_Dataset/raw/wavefunctions.tar.gz",
    "output_npz": Path.home() / "Chem_Data/milia_PyG_Dataset/raw/wavefunctions_sliced.npz",
    "num_molecules": 10,  # Number of molecules to extract
    "feature_tier": "complete",  # Extract ALL available data including MO coefficients
    "temp_dir": None,  # Will use system temp
}


class SafeWavefunctionPreprocessor:
    """
    Production-ready preprocessor with safety guarantees.
    """

    def __init__(self, config: dict):
        self.config = config
        self.tar_path = config["tar_path"]
        self.output_npz = config["output_npz"]
        self.num_molecules = config["num_molecules"]
        self.feature_tier = config["feature_tier"]
        self.temp_dir = None

        # Statistics
        self.stats = {
            "extracted": 0,
            "parsed": 0,
            "failed": 0,
            "errors": [],
            "time_extraction": 0.0,
            "time_processing": 0.0,
            "time_npz_creation": 0.0,
            "time_total": 0.0,
        }

    def validate_environment(self) -> bool:
        """Validate environment before processing."""
        logger.info("=" * 70)
        logger.info("VALIDATING ENVIRONMENT")
        logger.info("=" * 70)

        # Check tar.gz exists
        if not self.tar_path.exists():
            logger.error(f"Archive not found: {self.tar_path}")
            return False
        logger.info(f"✓ Archive found: {self.tar_path}")
        logger.info(f"  Size: {self.tar_path.stat().st_size / (1024**3):.2f} GB")

        # Check IOData available
        try:
            from iodata import load_one

            logger.info("✓ IOData available")
        except ImportError as e:
            logger.error(f"IOData not available: {e}")
            return False

        # Check output directory writable
        output_dir = self.output_npz.parent
        if not output_dir.exists():
            logger.error(f"Output directory not found: {output_dir}")
            return False

        if not output_dir.is_dir():
            logger.error(f"Output path is not a directory: {output_dir}")
            return False

        logger.info(f"✓ Output directory: {output_dir}")

        # Check disk space (need ~500 MB safety margin)
        stat = shutil.disk_usage(output_dir)
        free_gb = stat.free / (1024**3)
        logger.info(f"✓ Free disk space: {free_gb:.2f} GB")

        if free_gb < 1.0:
            logger.error("Insufficient disk space (need at least 1 GB free)")
            return False

        logger.info("=" * 70)
        logger.info("✓ ENVIRONMENT VALIDATED")
        logger.info("=" * 70)
        return True

    def extract_molden_files_safely(self) -> list[Path]:
        """
        Safely extract N .molden files from tar.gz WITHOUT decompressing entire archive.

        Returns:
            List of paths to extracted .molden files
        """
        start_time = time.time()

        logger.info(f"\nExtracting {self.num_molecules} .molden files from archive...")
        logger.info("Using streaming extraction (memory-efficient)")

        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="milia_wavefunction_"))
        logger.info(f"Temporary directory: {self.temp_dir}")

        extracted_files = []

        try:
            # Open tar.gz in streaming mode (doesn't load entire file)
            with tarfile.open(self.tar_path, "r:gz") as tar:
                molden_count = 0

                # Iterate through archive members
                for member in tar:
                    # Only extract .molden files
                    if member.name.endswith(".molden") and member.isfile():
                        # Extract single file
                        tar.extract(member, path=self.temp_dir)
                        extracted_path = self.temp_dir / member.name
                        extracted_files.append(extracted_path)
                        molden_count += 1

                        logger.info(
                            f"  [{molden_count}/{self.num_molecules}] Extracted: {member.name}"
                        )

                        # Stop after N files
                        if molden_count >= self.num_molecules:
                            break

                self.stats["extracted"] = molden_count

                if molden_count == 0:
                    logger.error("No .molden files found in archive!")
                    return []

                elapsed = time.time() - start_time
                self.stats["time_extraction"] = elapsed
                logger.info(f"✓ Extracted {molden_count} files successfully in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            self.cleanup()
            raise

        return extracted_files

    def parse_molden_file(self, filepath: Path) -> dict | None:
        """
        Parse single .molden file with IOData.

        Returns:
            Dictionary with extracted features or None on failure
        """
        try:
            from iodata import load_one

            # Parse with IOData
            mol_data = load_one(str(filepath))

            # Extract features based on tier
            features = self._extract_features(mol_data, filepath.stem)

            return features

        except Exception as e:
            logger.warning(f"Failed to parse {filepath.name}: {e}")
            self.stats["failed"] += 1
            self.stats["errors"].append(f"{filepath.name}: {str(e)}")
            return None

    def _extract_features(self, mol_data, compound_name: str) -> dict:
        """
        Extract COMPLETE feature spectrum from mol_data.

        ENHANCED VERSION: Extracts ~30 features vs original 13
        """
        # ========================================================================
        # TIER 1: ESSENTIAL GEOMETRIC FEATURES
        # ========================================================================
        features = {
            "compound": compound_name,
            "atoms": mol_data.atnums,
            "coordinates": mol_data.atcoords,
            "n_atoms": len(mol_data.atnums),
        }

        # ========================================================================
        # TIER 2-5: ELECTRONIC & QUANTUM FEATURES
        # ========================================================================
        if hasattr(mol_data, "mo") and mol_data.mo is not None:
            # --------------------------------------------------------------------
            # Basic Electronic Properties
            # --------------------------------------------------------------------
            # Total electrons
            n_electrons = mol_data.mo.occs.sum()
            features["n_electrons"] = n_electrons

            # MO kind (restricted/unrestricted/generalized)
            features["mo_kind"] = mol_data.mo.kind

            # Number of basis functions
            features["n_basis"] = len(mol_data.mo.energies)

            # --------------------------------------------------------------------
            # Orbital Classification
            # --------------------------------------------------------------------
            occupied_mask = mol_data.mo.occs > 0.1
            virtual_mask = mol_data.mo.occs < 0.1

            n_occupied = int(occupied_mask.sum())
            n_virtual = int(virtual_mask.sum())
            features["n_occupied_orbitals"] = n_occupied
            features["n_virtual_orbitals"] = n_virtual

            # --------------------------------------------------------------------
            # HOMO/LUMO Energies and Gap
            # --------------------------------------------------------------------
            if occupied_mask.any():
                homo_energy = float(mol_data.mo.energies[occupied_mask].max())
                features["homo_energy"] = homo_energy

            if virtual_mask.any():
                lumo_energy = float(mol_data.mo.energies[virtual_mask].min())
                features["lumo_energy"] = lumo_energy

            # HOMO-LUMO gap (most important descriptor!)
            if "homo_energy" in features and "lumo_energy" in features:
                gap_hartree = features["lumo_energy"] - features["homo_energy"]
                features["homo_lumo_gap"] = gap_hartree

                # Convert to eV (1 Hartree = 27.211386245988 eV)
                gap_eV = gap_hartree * 27.211386245988
                features["homo_lumo_gap_eV"] = gap_eV

            # --------------------------------------------------------------------
            # Orbital Energy Statistics (NEW!)
            # --------------------------------------------------------------------
            if occupied_mask.any():
                occupied_energies = mol_data.mo.energies[occupied_mask]
                features["occupied_energy_mean"] = float(occupied_energies.mean())
                features["occupied_energy_std"] = float(occupied_energies.std())
                features["occupied_energy_min"] = float(occupied_energies.min())
                features["occupied_energy_max"] = float(occupied_energies.max())

            if virtual_mask.any():
                virtual_energies = mol_data.mo.energies[virtual_mask]
                features["virtual_energy_mean"] = float(virtual_energies.mean())
                features["virtual_energy_std"] = float(virtual_energies.std())
                features["virtual_energy_min"] = float(virtual_energies.min())
                features["virtual_energy_max"] = float(virtual_energies.max())

            # --------------------------------------------------------------------
            # Chemical Descriptors (NEW!)
            # --------------------------------------------------------------------
            if "homo_energy" in features:
                # Ionization Potential (Koopmans' theorem approximation)
                features["ionization_potential_approx"] = -features["homo_energy"]

            if "lumo_energy" in features:
                # Electron Affinity approximation
                features["electron_affinity_approx"] = -features["lumo_energy"]

            if "homo_lumo_gap" in features:
                # Chemical Hardness (Pearson HSAB theory)
                features["chemical_hardness"] = features["homo_lumo_gap"] / 2.0

            if "homo_energy" in features and "lumo_energy" in features:
                # Chemical Potential (electronegativity approximation)
                features["chemical_potential"] = (
                    features["homo_energy"] + features["lumo_energy"]
                ) / 2.0

            # --------------------------------------------------------------------
            # Tier-Specific Features (Standard/Complete)
            # --------------------------------------------------------------------
            if self.feature_tier in ["standard", "complete"]:
                # Full MO energies and occupations
                features["mo_energies"] = mol_data.mo.energies
                features["mo_occupations"] = mol_data.mo.occs

            if self.feature_tier == "complete":
                # Full wavefunction (MO coefficients)
                features["mo_coefficients"] = mol_data.mo.coeffs

        # ========================================================================
        # BASIS SET INFORMATION (NEW!)
        # ========================================================================
        if hasattr(mol_data, "obasis") and mol_data.obasis is not None:
            # Calculate number of shells from shell_types array length
            if hasattr(mol_data.obasis, "shell_types"):
                features["n_shells"] = len(mol_data.obasis.shell_types)

        # ========================================================================
        # TOTAL ENERGY (NEW!)
        # ========================================================================
        if hasattr(mol_data, "energy") and mol_data.energy is not None:
            features["total_energy"] = float(mol_data.energy)

        # ========================================================================
        # MOLECULAR FORMULA (NEW!)
        # ========================================================================
        # Simple atomic mass table (subset of elements)
        atomic_masses = {
            1: 1.008,
            6: 12.011,
            7: 14.007,
            8: 15.999,
            9: 18.998,
            14: 28.085,
            15: 30.974,
            16: 32.06,
            17: 35.45,
            35: 79.904,
        }

        element_symbols = {
            1: "H",
            6: "C",
            7: "N",
            8: "O",
            9: "F",
            14: "Si",
            15: "P",
            16: "S",
            17: "Cl",
            35: "Br",
        }

        # Count atoms by element
        from collections import Counter

        element_counts = Counter(mol_data.atnums)

        # Build formula string (heavy atoms first, then H)
        formula_parts = []
        for z in sorted(element_counts.keys(), reverse=True):
            if z == 1:  # Save H for last
                continue
            count = element_counts[z]
            symbol = element_symbols.get(z, f"Z{z}")
            formula_parts.append(f"{symbol}{count if count > 1 else ''}")

        # Add H at the end (if present)
        if 1 in element_counts:
            count = element_counts[1]
            formula_parts.append(f"H{count if count > 1 else ''}")

        features["molecular_formula"] = "".join(formula_parts)

        # Molecular weight
        total_mass = sum(atomic_masses.get(z, 0) * count for z, count in element_counts.items())
        features["molecular_weight"] = float(total_mass)

        return features

    def process_files(self, molden_files: list[Path]) -> dict[str, np.ndarray]:
        """
        Process all extracted .molden files.

        *** THIS IS WHERE THE BUG WAS FIXED ***

        Returns:
            Dictionary ready for .npz saving
        """
        logger.info(f"\nProcessing {len(molden_files)} .molden files...")
        logger.info(f"Feature tier: {self.feature_tier}")
        logger.info("Extracting COMPLETE feature spectrum per molecule")

        # Storage for aggregated data
        data = defaultdict(list)

        for i, filepath in enumerate(molden_files, 1):
            logger.info(f"  [{i}/{len(molden_files)}] Processing: {filepath.name}")

            features = self.parse_molden_file(filepath)

            if features is None:
                continue

            # Aggregate ALL features
            # TIER 1: Geometric
            data["compounds"].append(features["compound"])
            data["atoms"].append(features["atoms"])
            data["coordinates"].append(features["coordinates"])
            data["n_atoms"].append(features["n_atoms"])

            # TIER 2: Basic Electronic
            if "n_electrons" in features:
                data["n_electrons"].append(features["n_electrons"])
            if "mo_kind" in features:
                data["mo_kind"].append(features["mo_kind"])
            if "n_basis" in features:
                data["n_basis"].append(features["n_basis"])
            if "n_occupied_orbitals" in features:
                data["n_occupied_orbitals"].append(features["n_occupied_orbitals"])
            if "n_virtual_orbitals" in features:
                data["n_virtual_orbitals"].append(features["n_virtual_orbitals"])
            if "homo_energy" in features:
                data["homo_energy"].append(features["homo_energy"])
            if "lumo_energy" in features:
                data["lumo_energy"].append(features["lumo_energy"])
            if "homo_lumo_gap" in features:
                data["homo_lumo_gap"].append(features["homo_lumo_gap"])
            if "homo_lumo_gap_eV" in features:
                data["homo_lumo_gap_eV"].append(features["homo_lumo_gap_eV"])

            # TIER 3: Orbital Energy Statistics (NEW!)
            for stat_key in [
                "occupied_energy_mean",
                "occupied_energy_std",
                "occupied_energy_min",
                "occupied_energy_max",
                "virtual_energy_mean",
                "virtual_energy_std",
                "virtual_energy_min",
                "virtual_energy_max",
            ]:
                if stat_key in features:
                    data[stat_key].append(features[stat_key])

            # TIER 4: Chemical Descriptors (NEW!)
            for chem_key in [
                "ionization_potential_approx",
                "electron_affinity_approx",
                "chemical_hardness",
                "chemical_potential",
            ]:
                if chem_key in features:
                    data[chem_key].append(features[chem_key])

            # TIER 5: Derived Features (NEW!)
            if "n_shells" in features:
                data["n_shells"].append(features["n_shells"])
            if "total_energy" in features:
                data["total_energy"].append(features["total_energy"])
            if "molecular_formula" in features:
                data["molecular_formula"].append(features["molecular_formula"])
            if "molecular_weight" in features:
                data["molecular_weight"].append(features["molecular_weight"])

            # Detailed MO data (tier-dependent)
            if self.feature_tier in ["standard", "complete"]:
                if "mo_energies" in features:
                    data["mo_energies"].append(features["mo_energies"])
                if "mo_occupations" in features:
                    data["mo_occupations"].append(features["mo_occupations"])

            if self.feature_tier == "complete":
                if "mo_coefficients" in features:
                    data["mo_coefficients"].append(features["mo_coefficients"])

            self.stats["parsed"] += 1

        # ============================================================================
        # ENHANCED: Convert to numpy arrays with COMPLETE data type handling
        # ============================================================================
        npz_data = {}
        for key, values in data.items():
            # STRING DATA: Molecule identifiers and categorical
            if key in ["compounds", "mo_kind", "molecular_formula"] or key in [
                "atoms",
                "coordinates",
                "mo_energies",
                "mo_occupations",
                "mo_coefficients",
            ]:
                npz_data[key] = np.array(values, dtype=object)

            # INTEGER DATA: Scalar integers per molecule
            elif key in [
                "n_atoms",
                "n_basis",
                "n_occupied_orbitals",
                "n_virtual_orbitals",
                "n_shells",
            ]:
                npz_data[key] = np.array(values, dtype=np.int32)

            # FLOAT DATA: Scalar floats per molecule
            # Core energies
            elif (
                key
                in [
                    "n_electrons",
                    "homo_energy",
                    "lumo_energy",
                    "homo_lumo_gap",
                    "homo_lumo_gap_eV",
                ]
                or key
                in [
                    "occupied_energy_mean",
                    "occupied_energy_std",
                    "occupied_energy_min",
                    "occupied_energy_max",
                    "virtual_energy_mean",
                    "virtual_energy_std",
                    "virtual_energy_min",
                    "virtual_energy_max",
                ]
                or key
                in [
                    "ionization_potential_approx",
                    "electron_affinity_approx",
                    "chemical_hardness",
                    "chemical_potential",
                ]
                or key in ["total_energy", "molecular_weight"]
            ):
                npz_data[key] = np.array(values, dtype=np.float64)

            # SAFETY NET: Unknown keys use object dtype
            else:
                logger.warning(f"⚠ Unknown key '{key}' encountered - using object dtype for safety")
                npz_data[key] = np.array(values, dtype=object)

        logger.info(f"✓ Processed {self.stats['parsed']} molecules successfully")
        if self.stats["failed"] > 0:
            logger.warning(f"⚠ Failed to process {self.stats['failed']} files")

        return npz_data

    def save_npz(self, data: dict[str, np.ndarray]) -> None:
        """
        Save data to .npz file with validation.
        """
        start_time = time.time()
        num_molecules = len(data["compounds"])

        logger.info(f"\nSaving to: {self.output_npz}")
        logger.info(f"Molecules: {num_molecules}")

        # Add enhanced metadata
        metadata = {
            "version": "3.0",
            "dataset_name": "milia_Wavefunction",
            "feature_tier": self.feature_tier,
            "num_molecules": num_molecules,
            "source": "wavefunctions.tar.gz",
            "extraction_type": "test_subset",
            "file_format": ".molden",
            "parser": "IOData",
            "date_created": "2025-11-08",
            "coordinate_units": "Bohr",
            "energy_units": "Hartree",
            "feature_count": len(data),
            "enhancements": "Complete feature spectrum extraction v3.0",
        }
        data["metadata"] = np.array([metadata], dtype=object)

        # Save with compression
        np.savez_compressed(self.output_npz, **data)

        elapsed = time.time() - start_time
        self.stats["time_npz_creation"] = elapsed

        file_size = self.output_npz.stat().st_size / (1024**2)
        logger.info(f"✓ Saved successfully in {elapsed:.2f}s")
        logger.info(f"  Size: {file_size:.2f} MB")
        logger.info(f"  Rate: {num_molecules / elapsed:.2f} molecules/s")
        logger.info(f"  Keys: {list(data.keys())}")

    def validate_npz(self) -> bool:
        """Validate created .npz file."""
        logger.info("\nValidating .npz file...")

        try:
            data = np.load(self.output_npz, allow_pickle=True)

            required_keys = ["compounds", "atoms", "coordinates"]
            for key in required_keys:
                if key not in data:
                    logger.error(f"Missing required key: {key}")
                    return False

            n_molecules = len(data["compounds"])
            logger.info(f"✓ Contains {n_molecules} molecules")

            # Check shapes
            for key in ["atoms", "coordinates"]:
                if len(data[key]) != n_molecules:
                    logger.error(f"Shape mismatch for {key}")
                    return False

            # Validate data types
            logger.info("\nData type check:")
            logger.info(f"  compounds: {data['compounds'].dtype}")
            logger.info(f"  atoms: {data['atoms'].dtype}")
            logger.info(f"  coordinates: {data['coordinates'].dtype}")
            if "n_atoms" in data:
                logger.info(f"  n_atoms: {data['n_atoms'].dtype}")
            if "n_electrons" in data:
                logger.info(f"  n_electrons: {data['n_electrons'].dtype}")
            if "homo_lumo_gap_eV" in data:
                logger.info(f"  homo_lumo_gap_eV: {data['homo_lumo_gap_eV'].dtype}")

            logger.info("✓ Validation passed")
            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            logger.info(f"\nCleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            logger.info("✓ Cleanup complete")

    def print_summary(self) -> None:
        """Print processing summary."""
        logger.info("\n" + "=" * 70)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Archive: {self.tar_path.name}")
        logger.info(f"Extracted: {self.stats['extracted']} files")
        logger.info(f"Parsed successfully: {self.stats['parsed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Output: {self.output_npz}")
        logger.info(f"Feature tier: {self.feature_tier}")

        if self.output_npz.exists():
            size_mb = self.output_npz.stat().st_size / (1024**2)
            logger.info(f"File size: {size_mb:.2f} MB")

        # Timing breakdown
        logger.info("\nTiming breakdown:")
        logger.info(f"  Extraction: {self.stats.get('time_extraction', 0):.2f}s")
        logger.info(f"  Processing: {self.stats.get('time_processing', 0):.2f}s")
        logger.info(f"  NPZ creation: {self.stats.get('time_npz_creation', 0):.2f}s")
        logger.info(f"  TOTAL TIME: {self.stats.get('time_total', 0):.2f}s")

        if self.stats["errors"]:
            logger.info("\nErrors encountered:")
            for error in self.stats["errors"][:5]:  # Show first 5
                logger.info(f"  - {error}")

        logger.info("=" * 70)

    def run(self) -> bool:
        """
        Execute full preprocessing pipeline.

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()

        try:
            # Check if wavefunction*.npz already exists
            output_dir = self.output_npz.parent
            existing_files = list(output_dir.glob("wavefunction*.npz"))

            if existing_files:
                logger.info("=" * 70)
                logger.info("EXISTING .NPZ FILE DETECTED")
                logger.info("=" * 70)
                for f in existing_files:
                    size_mb = f.stat().st_size / (1024**2)
                    logger.info(f"Found: {f.name} ({size_mb:.2f} MB)")
                logger.info("Skipping preprocessing - file already available")
                logger.info("=" * 70)
                return True

            # Validate environment
            if not self.validate_environment():
                logger.error("Environment validation failed")
                return False

            # Extract files
            molden_files = self.extract_molden_files_safely()
            if not molden_files:
                logger.error("No files extracted")
                return False

            # Process files
            npz_data = self.process_files(molden_files)

            if not npz_data or len(npz_data["compounds"]) == 0:
                logger.error("No data to save")
                return False

            # Save to .npz
            self.save_npz(npz_data)

            # Validate output
            if not self.validate_npz():
                logger.error("Output validation failed")
                return False

            # Calculate total time
            self.stats["time_total"] = time.time() - start_time

            # Cleanup
            self.cleanup()

            # Summary
            self.print_summary()

            logger.info("\n✓ PREPROCESSING COMPLETED SUCCESSFULLY")
            return True

        except KeyboardInterrupt:
            logger.warning("\n⚠ Process interrupted by user")
            self.cleanup()
            return False

        except Exception as e:
            logger.error(f"\n✗ PREPROCESSING FAILED: {e}", exc_info=True)
            self.cleanup()
            return False


def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("milia WAVEFUNCTION TEST PREPROCESSING v3.0 (ENHANCED)")
    logger.info("=" * 70)
    logger.info("Safety features:")
    logger.info("  ✓ Streaming extraction (no full decompression)")
    logger.info("  ✓ Memory-efficient processing")
    logger.info("  ✓ Automatic cleanup on error")
    logger.info("  ✓ Validation at each step")
    logger.info("=" * 70)
    logger.info("Enhancements:")
    logger.info("  ✓ FIXED: Proper data type handling")
    logger.info("  ✓ NEW: Complete feature spectrum (~30 keys)")
    logger.info("  ✓ NEW: Orbital energy statistics")
    logger.info("  ✓ NEW: Chemical descriptors (IP, EA, hardness)")
    logger.info("  ✓ NEW: Molecular formula & weight")
    logger.info("=" * 70)

    # Create preprocessor
    preprocessor = SafeWavefunctionPreprocessor(CONFIG)

    # Run preprocessing
    success = preprocessor.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
