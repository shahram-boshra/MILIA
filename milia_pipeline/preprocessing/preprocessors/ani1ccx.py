# milia_pipeline/preprocessing/preprocessors/ani1ccx.py

"""
ANI-1ccx Preprocessor
=====================

Preprocessor for ANI-1ccx quantum chemistry dataset (HDF5 format).

Parses ANI-1x HDF5 file structure (same file as ANI-1x), extracts molecular data 
for conformers that have coupled-cluster energies, and creates .npz file 
compatible with miliaDataset.

ANI-1ccx Dataset Information:
-----------------------------
- Source: Figshare (https://figshare.com/ndownloader/files/18112775)
- File: ani1x-release.h5 (~5.21 GB) - SAME file as ANI-1x
- Contents: ~500k conformations with CCSD(T)/CBS energies (subset of ANI-1x)
- Format: HDF5 with molecular groups containing conformations
- Method: CCSD(T)/CBS extrapolation using transfer learning

CRITICAL DIFFERENCE FROM ANI-1x PREPROCESSOR:
- ANI-1x: Extracts ALL ~5M conformers (only requires DFT properties)
- ANI-1ccx: Extracts ~500k conformers that HAVE ccsd(t)_cbs.energy

The key difference is in iter_data_buckets_ccx() which SKIPS conformers 
without ccsd(t)_cbs.energy, matching the official loader behavior:
> "if the 'data_keys' list contains 'ccsd(t)_cbs.energy', then only 
>  conformers that share [this] will be loaded"

HDF5 Structure (from Scientific Data Table 1):
----------------------------------------------
Each molecular group (chemical isomer) contains:
- 'atomic_numbers': Shape (Nc, Na) - atomic numbers [uint8]
- 'coordinates': Shape (Nc, Na, 3) - positions [Angstrom, float32]
- 'ccsd(t)_cbs.energy': Shape (Nc,) - CC energies [Hartree, float64] - REQUIRED
- 'wb97x_dz.energy': Shape (Nc,) - DFT energies [Hartree, float64]
- 'wb97x_dz.forces': Shape (Nc, Na, 3) - atomic forces [Hartree/Angstrom, float32]
- 'wb97x_dz.hirshfeld_charges': Shape (Nc, Na) - Hirshfeld charges [e, float32]
- 'wb97x_dz.cm5_charges': Shape (Nc, Na) - CM5 charges [e, float32]
- 'wb97x_dz.dipole': Shape (Nc, 3) - molecular dipole [Debye, float32]

Where: Nc = number of conformations, Na = number of atoms

Reference: Smith, J.S., et al. The ANI-1ccx and ANI-1x data sets, 
           coupled-cluster and density functional theory properties 
           for molecules. Sci Data 7, 134 (2020).

Author: MILIA Pipeline Team
Version: 1.0
Date: January 2026
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry


logger = logging.getLogger(__name__)


# Note: ANI-1ccx HDF5 contains atomic numbers as uint8 integers.
# We store them directly (consistent with Wavefunction preprocessor using mol_data.atnums)
# and compatible with validate_molecular_structure() which expects numeric arrays.


def iter_data_buckets_ccx(h5filename: str, keys: List[str] = None) -> Dict[str, Any]:
    """
    Iterate over buckets of data in ANI-1x HDF5 file for ANI-1ccx extraction.
    
    CRITICAL DIFFERENCE FROM ANI-1x iter_data_buckets:
    This function REQUIRES ccsd(t)_cbs.energy to be present. Conformers without
    coupled-cluster energy are SKIPPED entirely.
    
    This matches the official loader behavior from the Nature paper:
    > "if the 'data_keys' list contains 'ccsd(t)_cbs.energy', then only 
    >  conformers that share [this] will be loaded, approximately 500k structures"
    
    Args:
        h5filename: Path to ANI-1x HDF5 file (same file contains both datasets)
        keys: List of property keys to load (default includes ccsd(t)_cbs.energy)
        
    Yields:
        Dict with:
            - 'atomic_numbers': Shape (Na,) - atomic numbers for this conformer
            - 'coordinates': Shape (Na, 3) - positions for this conformer
            - 'molecule_id': String identifier for the molecular group
            - 'ccsd(t)_cbs.energy': Coupled-cluster energy (REQUIRED)
            - Plus any other requested property keys
            
    Evidence: aiqm/ANI1x_datasets/dataloader.py iter_data_buckets function
              Modified to REQUIRE ccsd(t)_cbs.energy for ANI-1ccx extraction
    """
    import h5py
    
    # Default keys for ANI-1ccx - MUST include ccsd(t)_cbs.energy
    if keys is None:
        keys = [
            'ccsd(t)_cbs.energy',  # REQUIRED - this filters to ANI-1ccx subset
            'wb97x_dz.energy',
            'wb97x_dz.forces',
            'wb97x_dz.hirshfeld_charges',
            'wb97x_dz.cm5_charges',
            'wb97x_dz.dipole',
        ]
    
    # Ensure ccsd(t)_cbs.energy is in the keys list
    if 'ccsd(t)_cbs.energy' not in keys:
        keys = ['ccsd(t)_cbs.energy'] + list(keys)
    
    with h5py.File(h5filename, 'r') as f:
        for mol_name in f.keys():
            mol_group = f[mol_name]
            
            # CRITICAL: Skip molecular groups that don't have ccsd(t)_cbs.energy
            # This is the key difference from ANI-1x preprocessor
            if 'ccsd(t)_cbs.energy' not in mol_group:
                continue
            
            # Get atomic numbers and coordinates (always present)
            # CRITICAL: Explicitly specify dtype to ensure proper type through slicing
            # Evidence: NPZ inspection showed dtype issues when not explicitly specified
            atomic_numbers_all = np.array(mol_group['atomic_numbers'], dtype=np.uint8)
            coordinates_all = np.array(mol_group['coordinates'], dtype=np.float32)
            
            # Get CC energy array to determine valid conformers
            cc_energy_all = np.array(mol_group['ccsd(t)_cbs.energy'], dtype=np.float64)
            
            # Get number of conformations
            n_conformers = coordinates_all.shape[0]
            
            # Load requested properties
            properties = {}
            valid_mask = np.ones(n_conformers, dtype=bool)
            
            # Define expected dtypes for each property based on ANI-1x/1ccx HDF5 specification
            # Reference: Scientific Data Table 1 (Smith et al., 2020)
            property_dtypes = {
                'ccsd(t)_cbs.energy': np.float64,    # Hartree, high precision (CC energy)
                'wb97x_dz.energy': np.float64,       # Hartree, high precision (DFT energy)
                'wb97x_dz.forces': np.float32,       # Hartree/Angstrom
                'wb97x_dz.hirshfeld_charges': np.float32,  # electrons
                'wb97x_dz.cm5_charges': np.float32,  # electrons
                'wb97x_dz.dipole': np.float32,       # Debye
            }
            
            # First, check CC energy validity - REQUIRED property
            if np.issubdtype(cc_energy_all.dtype, np.floating):
                valid_mask &= ~np.isnan(cc_energy_all)
            
            properties['ccsd(t)_cbs.energy'] = cc_energy_all
            
            # Load other requested properties
            for key in keys:
                if key == 'ccsd(t)_cbs.energy':
                    continue  # Already loaded
                    
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
                            valid_mask &= ~np.any(np.isnan(prop_data.reshape(n_conformers, -1)), axis=1)
            
            # Yield data for each valid conformer (those with valid CC energy)
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
                    'atomic_numbers': atomic_numbers,
                    'coordinates': coordinates,
                    'molecule_id': mol_name,
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


@PreprocessorRegistry.register("ANI1ccx")
class ANI1ccxPreprocessor(BasePreprocessor):
    """
    Preprocessor for ANI-1ccx quantum chemistry dataset.
    
    Pipeline:
    ---------
    1. Open HDF5 file and iterate over molecular groups
    2. FILTER to conformers that have ccsd(t)_cbs.energy (ANI-1ccx subset)
    3. Extract conformer data (atomic_numbers, coordinates, CC energy, DFT properties)
    4. Filter out entries with NaN values
    5. Build .npz file (compressed format compatible with miliaDataset)
    
    Configuration:
    --------------
    Required keys:
        - raw_archive_path: Path to ani1x-release.h5 (same file as ANI-1x)
        - output_npz_path: Path for output .npz file
        
    Optional keys:
        - num_molecules: Number of conformers to extract (None = all ~500k)
        - property_keys: List of properties to extract (default: all available)
        
    Example:
    --------
    >>> config = {
    ...     'raw_archive_path': 'raw/ani1x-release.h5',
    ...     'output_npz_path': 'processed/ani1ccx.npz',
    ...     'num_molecules': 10000,  # For testing
    ... }
    >>> preprocessor = ANI1ccxPreprocessor(config, logger)
    >>> output_path = preprocessor.run()
    
    Notes:
    ------
    - Uses the SAME HDF5 file as ANI-1x (ani1x-release.h5, ~5.21 GB)
    - Filters to ~500k conformers with ccsd(t)_cbs.energy
    - Full preprocessing takes approximately 10-20 minutes
    - Output NPZ will be ~500MB for the full dataset
    """
    
    # Default property keys to extract from HDF5
    # CRITICAL: ccsd(t)_cbs.energy MUST be first - it's the REQUIRED filter key
    DEFAULT_PROPERTY_KEYS = [
        'ccsd(t)_cbs.energy',        # REQUIRED - coupled-cluster energy (primary target)
        'wb97x_dz.energy',           # DFT energy for comparison
        'wb97x_dz.forces',
        'wb97x_dz.hirshfeld_charges',
        'wb97x_dz.cm5_charges',
        'wb97x_dz.dipole',
    ]
    
    def _validate_config(self) -> None:
        """Validate ANI-1ccx-specific configuration."""
        # Check required keys
        required_keys = ['raw_archive_path', 'output_npz_path']
        missing = [k for k in required_keys if k not in self.config]
        
        if missing:
            raise ConfigurationError(
                f"Missing required configuration keys: {missing}",
                config_key=', '.join(missing)
            )
        
        # Validate H5 path exists
        h5_path = Path(self.config['raw_archive_path'])
        if not h5_path.exists():
            raise ConfigurationError(
                f"ANI-1x/ANI-1ccx HDF5 file not found: {h5_path}",
                config_key='raw_archive_path',
                actual_value=str(h5_path)
            )
        
        # Validate file extension
        if not str(h5_path).lower().endswith(('.h5', '.hdf5')):
            self.logger.warning(
                f"File extension not recognized as HDF5: {h5_path.suffix}. "
                f"Expected .h5 or .hdf5. Proceeding anyway."
            )
        
        # Validate num_molecules if specified
        num_molecules = self.config.get('num_molecules')
        if num_molecules is not None:
            if not isinstance(num_molecules, int) or num_molecules < 1:
                raise ConfigurationError(
                    f"num_molecules must be positive integer, got {num_molecules}",
                    config_key='num_molecules',
                    actual_value=num_molecules
                )
        
        self.logger.debug("ANI-1ccx configuration validation passed")
    
    def preprocess(self) -> Path:
        """
        Execute ANI-1ccx preprocessing pipeline.
        
        Returns:
            Path to generated .npz file
        """
        h5_path = Path(self.config['raw_archive_path'])
        output_npz = Path(self.config['output_npz_path'])
        num_molecules = self.config.get('num_molecules', None)
        property_keys = self.config.get('property_keys', self.DEFAULT_PROPERTY_KEYS)
        
        # Ensure ccsd(t)_cbs.energy is in property_keys
        if 'ccsd(t)_cbs.energy' not in property_keys:
            property_keys = ['ccsd(t)_cbs.energy'] + list(property_keys)
        
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
            # STEP 1: Parse HDF5 file and collect data (ANI-1ccx subset)
            # ================================================================
            self.logger.info("=" * 70)
            self.logger.info("STEP 1: Parsing ANI-1ccx from HDF5 file")
            self.logger.info("=" * 70)
            self.logger.info(f"Source file: {h5_path}")
            self.logger.info(f"NOTE: Same HDF5 file as ANI-1x, filtering to CC subset")
            self.logger.info(f"Properties to extract: {property_keys}")
            if num_molecules:
                self.logger.info(f"Maximum conformers: {num_molecules}")
            else:
                self.logger.info("Maximum conformers: ALL (~500k with CC energy)")
            
            features, parse_metadata = self._parse_ani1ccx_h5(
                h5_path=h5_path,
                property_keys=property_keys,
                max_conformers=num_molecules
            )
            
            # ================================================================
            # STEP 2: Build .npz file
            # ================================================================
            self.logger.info("=" * 70)
            self.logger.info("STEP 2: Building .npz file")
            self.logger.info("=" * 70)
            
            # Prepare comprehensive metadata
            npz_metadata = {
                'version': '1.0',
                'dataset_name': 'ANI1ccx',
                'source': h5_path.name,
                'source_url': 'https://figshare.com/ndownloader/files/18112775',
                'reference': 'Smith et al., Scientific Data 7, 134 (2020)',
                'doi': '10.1038/s41597-020-0473-z',
                'file_format': '.h5 (HDF5 ANI-1x format, ANI-1ccx subset)',
                'parser': 'ANI1ccxPreprocessor',
                'preprocessing_version': '1.0',
                'coordinate_units': 'angstrom',
                'energy_units': 'hartree',
                'primary_energy': 'ccsd(t)_cbs.energy (CCSD(T)/CBS)',
                'secondary_energy': 'wb97x_dz.energy (ωB97x/6-31G* DFT)',
                **parse_metadata  # Include parsing statistics
            }
            
            self._build_npz(
                features=features,
                metadata=npz_metadata,
                output_path=output_npz
            )
            
            self.logger.info("=" * 70)
            self.logger.info("ANI-1ccx PREPROCESSING COMPLETE")
            self.logger.info("=" * 70)
            
            return output_npz
            
        except Exception as e:
            raise DataProcessingError(
                f"ANI-1ccx preprocessing failed: {e}",
                operation="ani1ccx_preprocessing"
            ) from e
    
    def _parse_ani1ccx_h5(
        self, 
        h5_path: Path, 
        property_keys: List[str],
        max_conformers: Optional[int] = None
    ) -> Tuple[Dict[str, List], Dict[str, Any]]:
        """
        Parse ANI-1x HDF5 file and extract ANI-1ccx molecular data.
        
        CRITICAL: Only extracts conformers with ccsd(t)_cbs.energy present.
        This is the defining characteristic of ANI-1ccx (~500k conformers).
        
        Args:
            h5_path: Path to ANI-1x HDF5 file (contains both ANI-1x and ANI-1ccx)
            property_keys: List of property keys to extract
            max_conformers: Maximum number of conformers to extract (None = all)
            
        Returns:
            Tuple of (features_dict, metadata_dict)
        """
        # Initialize storage for ragged arrays (variable-length per molecule)
        atoms_list = []              # List of atomic number arrays (integers)
        coordinates_list = []        # List of coordinate arrays
        ccsd_energy_list = []        # List of CC energy values (PRIMARY TARGET)
        dft_energy_list = []         # List of DFT energy values (for comparison)
        forces_list = []             # List of force arrays
        hirshfeld_charges_list = []  # List of charge arrays
        cm5_charges_list = []        # List of charge arrays
        dipole_list = []             # List of dipole vectors
        molecule_id_list = []        # List of molecule identifiers
        
        conformer_count = 0
        skipped_nan = 0
        skipped_no_cc = 0
        
        self.logger.info("Starting HDF5 iteration (ANI-1ccx extraction)...")
        self.logger.info("Filtering to conformers with ccsd(t)_cbs.energy...")
        
        for data in iter_data_buckets_ccx(str(h5_path), keys=property_keys):
            # Check if we've reached the limit
            if max_conformers is not None and conformer_count >= max_conformers:
                break
            
            # Store atomic numbers directly as integers (same as Wavefunction preprocessor)
            # HDF5 contains: data['atomic_numbers'] as uint8 integers [6, 1, 8, 7]
            # Store atomic numbers directly as integers (same as Wavefunction preprocessor)
            # HDF5 contains: data['atomic_numbers'] as uint8 integers [6, 1, 8, 7]
            # CRITICAL: Explicitly cast to uint8 to ensure proper dtype through NPZ save/load cycle
            # Evidence: NPZ inspection showed dtype issues when not explicitly specified
            atomic_numbers = np.ascontiguousarray(data['atomic_numbers'], dtype=np.uint8)
            
            # Ensure coordinates are explicit contiguous float32 arrays
            coordinates = np.ascontiguousarray(data['coordinates'], dtype=np.float32)
            
            # Store data
            atoms_list.append(atomic_numbers)
            coordinates_list.append(coordinates)
            molecule_id_list.append(data['molecule_id'])
            
            # CC Energy (REQUIRED - primary target)
            if 'ccsd(t)_cbs.energy' in data:
                ccsd_energy_list.append(data['ccsd(t)_cbs.energy'])
            else:
                # This should not happen due to iter_data_buckets_ccx filtering
                ccsd_energy_list.append(np.nan)
                skipped_no_cc += 1
            
            # DFT Energy (optional - for comparison)
            if 'wb97x_dz.energy' in data:
                dft_energy_list.append(data['wb97x_dz.energy'])
            else:
                dft_energy_list.append(None)
            
            # Forces (optional) - ensure contiguous float32 for proper NPZ serialization
            if 'wb97x_dz.forces' in data:
                forces_list.append(np.ascontiguousarray(data['wb97x_dz.forces'], dtype=np.float32))
            else:
                forces_list.append(None)
            
            # Hirshfeld charges (optional) - ensure contiguous float32
            if 'wb97x_dz.hirshfeld_charges' in data:
                hirshfeld_charges_list.append(np.ascontiguousarray(data['wb97x_dz.hirshfeld_charges'], dtype=np.float32))
            else:
                hirshfeld_charges_list.append(None)
            
            # CM5 charges (optional) - ensure contiguous float32
            if 'wb97x_dz.cm5_charges' in data:
                cm5_charges_list.append(np.ascontiguousarray(data['wb97x_dz.cm5_charges'], dtype=np.float32))
            else:
                cm5_charges_list.append(None)
            
            # Dipole (optional) - ensure contiguous float32
            if 'wb97x_dz.dipole' in data:
                dipole_list.append(np.ascontiguousarray(data['wb97x_dz.dipole'], dtype=np.float32))
            else:
                dipole_list.append(None)
            
            conformer_count += 1
            
            # Progress logging
            if conformer_count % 50000 == 0:
                self.logger.info(f"Processed {conformer_count:,} ANI-1ccx conformers...")
        
        self.logger.info(f"Finished parsing: {conformer_count:,} ANI-1ccx conformers extracted")
        if skipped_nan > 0:
            self.logger.info(f"Skipped due to NaN values: {skipped_nan:,}")
        if skipped_no_cc > 0:
            self.logger.warning(f"Skipped due to missing CC energy: {skipped_no_cc:,}")
        
        # Build features dictionary with object arrays for ragged data
        # CRITICAL: Use np.empty() + element assignment instead of np.array(list, dtype=object)
        # Evidence: np.array(list, dtype=object) corrupts inner array dtypes to object
        # Test: np.array([arr_uint8], dtype=object)[0].dtype == object (WRONG)
        # Fix: obj_arr = np.empty(n, dtype=object); obj_arr[i] = arr → preserves inner dtype
        
        def _build_object_array(items: list) -> np.ndarray:
            """Build object array while preserving inner array dtypes."""
            arr = np.empty(len(items), dtype=object)
            for i, item in enumerate(items):
                arr[i] = item
            return arr
        
        features = {
            'atoms': _build_object_array(atoms_list),
            'coordinates': _build_object_array(coordinates_list),
            'ccsd_energy': np.array(ccsd_energy_list, dtype=np.float64),  # PRIMARY TARGET
            'molecule_id': _build_object_array(molecule_id_list),
        }
        
        # Add DFT energy if any were extracted
        if any(e is not None for e in dft_energy_list):
            # Convert None to np.nan for uniform array
            dft_energy_array = np.array([e if e is not None else np.nan for e in dft_energy_list], dtype=np.float64)
            features['dft_energy'] = dft_energy_array
        
        # Add optional properties if any were extracted
        if any(f is not None for f in forces_list):
            features['forces'] = _build_object_array(forces_list)
        
        if any(c is not None for c in hirshfeld_charges_list):
            features['hirshfeld_charges'] = _build_object_array(hirshfeld_charges_list)
        
        if any(c is not None for c in cm5_charges_list):
            features['cm5_charges'] = _build_object_array(cm5_charges_list)
        
        if any(d is not None for d in dipole_list):
            features['dipole'] = _build_object_array(dipole_list)
        
        # Compute metadata
        atom_counts = [len(a) for a in atoms_list]
        metadata = {
            'total_conformers': conformer_count,
            'skipped_nan': skipped_nan,
            'skipped_no_cc': skipped_no_cc,
            'mean_atoms': np.mean(atom_counts) if atom_counts else 0,
            'max_atoms': max(atom_counts) if atom_counts else 0,
            'min_atoms': min(atom_counts) if atom_counts else 0,
            'properties_extracted': list(features.keys()),
            'primary_energy_key': 'ccsd_energy',
            'extraction_filter': 'ccsd(t)_cbs.energy present',
        }
        
        return features, metadata
    
    def _build_npz(
        self, 
        features: Dict[str, np.ndarray], 
        metadata: Dict[str, Any],
        output_path: Path
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
        features['_metadata'] = np.array([str(metadata)])
        
        # Save as compressed NPZ
        self.logger.info(f"Saving to: {output_path}")
        np.savez_compressed(str(output_path), **features)
        
        # Log file size
        size_mb = output_path.stat().st_size / (1024**2)
        self.logger.info(f"✓ Created {output_path.name} ({size_mb:.2f} MB)")
        self.logger.info(f"  Total ANI-1ccx conformers: {metadata.get('total_conformers', 'N/A'):,}")
        self.logger.info(f"  Primary energy: ccsd_energy (CCSD(T)/CBS)")
        self.logger.info(f"  Properties: {metadata.get('properties_extracted', [])}")
