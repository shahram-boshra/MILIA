# milia_pipeline/preprocessing/preprocessors/ani1x.py

"""
ANI-1x Preprocessor
===================

Preprocessor for ANI-1x quantum chemistry dataset (HDF5 format).

Parses ANI-1x HDF5 file structure, extracts molecular data from chemical isomer
groups, and creates .npz file compatible with miliaDataset.

ANI-1x Dataset Information:
---------------------------
- Source: Figshare (https://figshare.com/ndownloader/files/18112775)
- File: ani1x-release.h5 (~5.21 GB)
- Contents: ~5 million DFT conformations for organic molecules (CHNO)
- Format: HDF5 with molecular groups containing conformations
- Method: Active learning with ωB97x/6-31G* DFT

HDF5 Structure (from Scientific Data Table 1):
----------------------------------------------
Each molecular group (chemical isomer) contains:
- 'atomic_numbers': Shape (Nc, Na) - atomic numbers [uint8]
- 'coordinates': Shape (Nc, Na, 3) - positions [Angstrom, float32]
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
Date: December 2025
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry


logger = logging.getLogger(__name__)


# Note: ANI-1x HDF5 contains atomic numbers as uint8 integers.
# We store them directly (consistent with Wavefunction preprocessor using mol_data.atnums)
# and compatible with validate_molecular_structure() which expects numeric arrays.


def iter_data_buckets(h5filename: str, keys: List[str] = None) -> Dict[str, Any]:
    """
    Iterate over buckets of data in ANI-1x HDF5 file.
    
    This function implements the same pattern as aiqm/ANI1x_datasets dataloader.py,
    yielding dictionaries with atomic numbers, coordinates, and requested properties
    for each conformer, filtering out entries with NaN values.
    
    Args:
        h5filename: Path to ANI-1x HDF5 file
        keys: List of property keys to load (default: ['wb97x_dz.energy'])
        
    Yields:
        Dict with:
            - 'atomic_numbers': Shape (Na,) - atomic numbers for this conformer
            - 'coordinates': Shape (Na, 3) - positions for this conformer
            - 'molecule_id': String identifier for the molecular group
            - Plus any requested property keys
            
    Evidence: aiqm/ANI1x_datasets/dataloader.py iter_data_buckets function
    """
    import h5py
    
    if keys is None:
        keys = ['wb97x_dz.energy']
    
    with h5py.File(h5filename, 'r') as f:
        for mol_name in f.keys():
            mol_group = f[mol_name]
            
            # Get atomic numbers and coordinates (always present)
            # CRITICAL: Explicitly specify dtype to ensure proper type through slicing
            # Evidence: NPZ inspection showed arr[0] dtype: object when dtype not specified
            atomic_numbers_all = np.array(mol_group['atomic_numbers'], dtype=np.uint8)
            coordinates_all = np.array(mol_group['coordinates'], dtype=np.float32)
            
            # Get number of conformations
            n_conformers = coordinates_all.shape[0]
            
            # Load requested properties
            properties = {}
            valid_mask = np.ones(n_conformers, dtype=bool)
            
            # Define expected dtypes for each property based on ANI-1x HDF5 specification
            # Reference: Scientific Data Table 1 (Smith et al., 2020)
            property_dtypes = {
                'wb97x_dz.energy': np.float64,       # Hartree, high precision
                'wb97x_dz.forces': np.float32,       # Hartree/Angstrom
                'wb97x_dz.hirshfeld_charges': np.float32,  # electrons
                'wb97x_dz.cm5_charges': np.float32,  # electrons
                'wb97x_dz.dipole': np.float32,       # Debye
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
                            valid_mask &= ~np.any(np.isnan(prop_data.reshape(n_conformers, -1)), axis=1)
            
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


@PreprocessorRegistry.register("ANI1x")
class ANI1xPreprocessor(BasePreprocessor):
    """
    Preprocessor for ANI-1x quantum chemistry dataset.
    
    Pipeline:
    ---------
    1. Open HDF5 file and iterate over molecular groups
    2. Extract conformer data (atomic_numbers, coordinates, properties)
    3. Filter out entries with NaN values
    4. Build .npz file (compressed format compatible with miliaDataset)
    
    Configuration:
    --------------
    Required keys:
        - raw_archive_path: Path to ani1x-release.h5
        - output_npz_path: Path for output .npz file
        
    Optional keys:
        - num_molecules: Number of conformers to extract (None = all ~5M)
        - property_keys: List of properties to extract (default: all available)
        
    Example:
    --------
    >>> config = {
    ...     'raw_archive_path': 'raw/ani1x-release.h5',
    ...     'output_npz_path': 'processed/ani1x.npz',
    ...     'num_molecules': 10000,  # For testing
    ... }
    >>> preprocessor = ANI1xPreprocessor(config, logger)
    >>> output_path = preprocessor.run()
    
    Notes:
    ------
    - The ANI-1x HDF5 file is approximately 5.21 GB
    - Full preprocessing takes approximately 30-60 minutes for all ~5M conformers
    - Use num_molecules for testing with a subset
    - Output NPZ will be ~2-3 GB for the full dataset
    """
    
    # Default property keys to extract from HDF5
    DEFAULT_PROPERTY_KEYS = [
        'wb97x_dz.energy',
        'wb97x_dz.forces',
        'wb97x_dz.hirshfeld_charges',
        'wb97x_dz.cm5_charges',
        'wb97x_dz.dipole',
    ]
    
    def _validate_config(self) -> None:
        """Validate ANI-1x-specific configuration."""
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
                f"ANI-1x HDF5 file not found: {h5_path}",
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
        
        self.logger.debug("ANI-1x configuration validation passed")
    
    def preprocess(self) -> Path:
        """
        Execute ANI-1x preprocessing pipeline.
        
        Returns:
            Path to generated .npz file
        """
        h5_path = Path(self.config['raw_archive_path'])
        output_npz = Path(self.config['output_npz_path'])
        num_molecules = self.config.get('num_molecules', None)
        property_keys = self.config.get('property_keys', self.DEFAULT_PROPERTY_KEYS)
        
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
            # STEP 1: Parse HDF5 file and collect data
            # ================================================================
            self.logger.info("=" * 70)
            self.logger.info("STEP 1: Parsing ANI-1x HDF5 file")
            self.logger.info("=" * 70)
            self.logger.info(f"Source file: {h5_path}")
            self.logger.info(f"Properties to extract: {property_keys}")
            if num_molecules:
                self.logger.info(f"Maximum conformers: {num_molecules}")
            else:
                self.logger.info("Maximum conformers: ALL (~5 million)")
            
            features, parse_metadata = self._parse_ani1x_h5(
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
                'dataset_name': 'ANI1x',
                'source': h5_path.name,
                'source_url': 'https://figshare.com/ndownloader/files/18112775',
                'reference': 'Smith et al., Scientific Data 7, 134 (2020)',
                'doi': '10.1038/s41597-020-0473-z',
                'file_format': '.h5 (HDF5 ANI-1x format)',
                'parser': 'ANI1xPreprocessor',
                'preprocessing_version': '1.0',
                'coordinate_units': 'angstrom',
                'energy_units': 'hartree',
                'force_units': 'hartree/angstrom',
                **parse_metadata  # Include parsing statistics
            }
            
            self._build_npz(
                features=features,
                metadata=npz_metadata,
                output_path=output_npz
            )
            
            self.logger.info("=" * 70)
            self.logger.info("ANI-1x PREPROCESSING COMPLETE")
            self.logger.info("=" * 70)
            
            return output_npz
            
        except Exception as e:
            raise DataProcessingError(
                f"ANI-1x preprocessing failed: {e}",
                operation="ani1x_preprocessing"
            ) from e
    
    def _parse_ani1x_h5(
        self, 
        h5_path: Path, 
        property_keys: List[str],
        max_conformers: Optional[int] = None
    ) -> Tuple[Dict[str, List], Dict[str, Any]]:
        """
        Parse ANI-1x HDF5 file and extract molecular data.
        
        Args:
            h5_path: Path to ANI-1x HDF5 file
            property_keys: List of property keys to extract
            max_conformers: Maximum number of conformers to extract (None = all)
            
        Returns:
            Tuple of (features_dict, metadata_dict)
        """
        # Initialize storage for ragged arrays (variable-length per molecule)
        atoms_list = []          # List of atomic number arrays (integers)
        coordinates_list = []    # List of coordinate arrays
        energy_list = []         # List of energy values
        forces_list = []         # List of force arrays
        hirshfeld_charges_list = []  # List of charge arrays
        cm5_charges_list = []    # List of charge arrays
        dipole_list = []         # List of dipole vectors
        molecule_id_list = []    # List of molecule identifiers
        
        conformer_count = 0
        skipped_nan = 0
        skipped_unknown_element = 0
        
        self.logger.info("Starting HDF5 iteration...")
        
        for data in iter_data_buckets(str(h5_path), keys=property_keys):
            # Check if we've reached the limit
            if max_conformers is not None and conformer_count >= max_conformers:
                break
            
            # Store atomic numbers directly as integers (same as Wavefunction preprocessor)
            # HDF5 contains: data['atomic_numbers'] as uint8 integers [6, 1, 8, 7]
            # This is consistent with format_parsers.py line 216: features['atoms'] = mol_data.atnums
            # CRITICAL: Explicitly cast to uint8 to ensure proper dtype through NPZ save/load cycle
            # Evidence: NPZ inspection showed arr[0] dtype: object with Python int scalars
            # when dtype is not explicitly specified
            atomic_numbers = np.ascontiguousarray(data['atomic_numbers'], dtype=np.uint8)
            
            # Ensure coordinates are explicit contiguous float32 arrays
            # This ensures proper dtype preservation through NPZ save/load cycle
            # and compatibility with is_value_valid_and_not_nan() validation
            coordinates = np.ascontiguousarray(data['coordinates'], dtype=np.float32)
            
            # Store data
            atoms_list.append(atomic_numbers)
            coordinates_list.append(coordinates)
            molecule_id_list.append(data['molecule_id'])
            
            # Energy (required)
            if 'wb97x_dz.energy' in data:
                energy_list.append(data['wb97x_dz.energy'])
            else:
                energy_list.append(np.nan)
            
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
            if conformer_count % 100000 == 0:
                self.logger.info(f"Processed {conformer_count:,} conformers...")
        
        self.logger.info(f"Finished parsing: {conformer_count:,} conformers extracted")
        if skipped_nan > 0:
            self.logger.info(f"Skipped due to NaN values: {skipped_nan:,}")
        if skipped_unknown_element > 0:
            self.logger.info(f"Skipped due to unknown elements: {skipped_unknown_element:,}")
        
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
            'energy': np.array(energy_list, dtype=np.float64),
            'molecule_id': _build_object_array(molecule_id_list),
        }
        
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
            'skipped_unknown_element': skipped_unknown_element,
            'mean_atoms': np.mean(atom_counts) if atom_counts else 0,
            'max_atoms': max(atom_counts) if atom_counts else 0,
            'min_atoms': min(atom_counts) if atom_counts else 0,
            'properties_extracted': list(features.keys()),
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
        self.logger.info(f"  Total conformers: {metadata.get('total_conformers', 'N/A'):,}")
        self.logger.info(f"  Properties: {metadata.get('properties_extracted', [])}")
