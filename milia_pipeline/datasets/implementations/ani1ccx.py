# milia_pipeline/datasets/implementations/ani1ccx.py

"""
ANI-1ccx dataset implementation.

This module provides the ANI1ccxDataset class which encapsulates all ANI-1ccx-specific
metadata and configuration for the ANI-1ccx quantum chemistry dataset.

ANI-1ccx Dataset Information:
-----------------------------
- Source: ANI-1ccx dataset with ~500k coupled-cluster conformations
- Reference: Smith et al., Scientific Data 7, 134 (2020)
- DOI: 10.1038/s41597-020-0473-z
- Figshare: https://doi.org/10.6084/m9.figshare.10047041
- Contains: ~500k conformations with CCSD(T)/CBS energies (subset of ANI-1x)
- Properties: Coupled-cluster energies + DFT properties from ωB97x/6-31G*
- Method: CCSD(T)/CBS extrapolation on selected ANI-1x conformations

CRITICAL: ANI-1ccx uses coordinate_based molecule creation strategy.
The HDF5 file structure contains only atomic_numbers and coordinates - 
NO parseable chemical identifiers (InChI/SMILES) are available.

KEY DIFFERENCE FROM ANI-1x:
- ANI-1x: ~5M conformers with DFT (ωB97x/6-31G*) properties only
- ANI-1ccx: ~500k conformers with BOTH DFT AND coupled-cluster (CCSD(T)/CBS) energies

The preprocessor filters to only include conformers that have ccsd(t)_cbs.energy.

Evidence sources:
- Scientific Data paper Table 1 (data layout)
- GitHub aiqm/ANI1x_datasets (dataloader.py implementation)
- MILIA_Adding_New_Datasets_Implementation_Blueprint.md (decision tree analysis)
"""

from typing import Dict, List, Tuple

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


# NOTE: ANI1ccxDatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_handler() method uses lazy import to
# instantiate the handler when needed. This follows the registry-based architecture
# pattern established in XXMD and documented in the Handler Migration Tracker.


@register
class ANI1ccxDataset(BaseDataset):
    """
    ANI-1ccx coupled-cluster quantum chemistry dataset implementation.
    
    ANI-1ccx contains ~500k conformations for organic molecules containing
    H, C, N, O atoms. This is a high-accuracy subset of ANI-1x with coupled-cluster
    energies computed at CCSD(T)/CBS level of theory using transfer learning.
    
    Properties in ANI-1ccx (from Scientific Data Table 1):
    ------------------------------------------------------
    - coordinates: Atomic coordinates (Nc, Na, 3) [Angstrom]
    - atomic_numbers: Atomic numbers (Nc, Na) [uint8]
    - ccsd(t)_cbs.energy: Coupled-cluster energy (Nc,) [Hartree] - PRIMARY TARGET
    - wb97x_dz.energy: DFT energy (Nc,) [Hartree] - for comparison
    - wb97x_dz.forces: Atomic forces (Nc, Na, 3) [Hartree/Angstrom]
    - wb97x_dz.hirshfeld_charges: Hirshfeld charges (Nc, Na) [e]
    - wb97x_dz.cm5_charges: CM5 charges (Nc, Na) [e]
    - wb97x_dz.dipole: Molecular dipole (Nc, 3) [Debye]
    
    Where: Nc = number of conformations (~500k), Na = number of atoms
    
    CRITICAL: ANI-1ccx is a SUBSET of ANI-1x
    - Both datasets are in the SAME HDF5 file (ani1x-release.h5)
    - ANI-1ccx conformers are those with ccsd(t)_cbs.energy present
    - The preprocessor filters conformers that have CC energy
    
    CRITICAL DIFFERENCES FROM QM9/DFT:
    
    1. Molecule Creation Strategy: 'coordinate_based'
       - ANI-1ccx HDF5 structure contains NO parseable chemical identifiers
       - Only atomic_numbers and coordinates are available per molecule
       - Molecular connectivity inferred from 3D coordinates using rdDetermineBonds
       - All molecules are neutral (no charge calculation needed)
    
    2. Data Format: HDF5 (same file as ANI-1x)
       - Requires custom preprocessor for HDF5 → NPZ conversion
       - Filters to conformers with ccsd(t)_cbs.energy present
    
    3. Identifier Keys: Empty tuple
       - Like ANI-1x, ANI-1ccx has no identifier keys
    
    Reference: Smith, J.S., et al. The ANI-1ccx and ANI-1x data sets, 
               coupled-cluster and density functional theory properties 
               for molecules. Sci Data 7, 134 (2020).
    """
    
    metadata = DatasetMetadata(
        name="ANI1ccx",
        version="1.0.0",
        description=(
            "ANI-1ccx dataset with ~500k coupled-cluster conformations for organic "
            "molecules (CHNO). Properties computed at CCSD(T)/CBS level using transfer "
            "learning from ωB97x/6-31G* DFT calculations."
        ),
        author="Smith, Nebgen, Lubbers, Isayev, Roitberg",
        license="CC0",
    )
    
    schema = DatasetSchema(
        # Required properties that must be present for every molecule
        # Based on ANI-1ccx HDF5 structure - atoms and coordinates always present
        # ccsd(t)_cbs.energy is the primary energy target (mapped to 'ccsd_energy' in NPZ)
        required_properties=('ccsd_energy', 'atoms', 'coordinates'),
        
        # Optional properties available in ANI-1ccx
        # All properties from the HDF5 file (keys mapped during preprocessing)
        optional_properties=(
            'dft_energy',            # wb97x_dz.energy (Hartree) - for comparison
            'forces',                # wb97x_dz.forces (Hartree/Angstrom)
            'hirshfeld_charges',     # wb97x_dz.hirshfeld_charges (e)
            'cm5_charges',           # wb97x_dz.cm5_charges (e)
            'dipole',                # wb97x_dz.dipole (Debye)
            'molecule_id',           # Molecule group identifier from HDF5
        ),
        
        # CRITICAL: Empty tuple - ANI-1ccx has NO parseable chemical identifiers
        # The HDF5 structure contains only atomic_numbers and coordinates
        # Molecule connectivity must be inferred from 3D coordinates
        identifier_keys=(),
        
        # ANI-1ccx coordinates are in Angstrom (DFT-optimized geometries)
        coordinate_units='angstrom',
        
        # ANI-1ccx energies are in Hartree (standard quantum chemistry output)
        energy_units='hartree',
    )
    
    features = DatasetFeatures(
        # ANI-1ccx does not have vibrational frequencies
        vibrational_analysis=False,
        
        # ANI-1ccx is deterministic coupled-cluster - no statistical uncertainties
        uncertainty_handling=False,
        
        # Atomization energies can be calculated from ccsd(t)_cbs.energy
        atomization_energy=True,
        
        # ANI-1ccx does not have rotational constants
        rotational_constants=False,
        
        # ANI-1ccx does not have frequency analysis
        frequency_analysis=False,
        
        # ANI-1ccx does not have orbital analysis
        orbital_analysis=False,
        
        # ANI-1ccx does not have HOMO-LUMO gap
        homo_lumo_gap=False,
        
        # ANI-1ccx does not have MO energies
        mo_energies=False,
    )
    
    # Configuration key matching config.yaml section
    config_key = "ani1ccx_config"
    
    # NOTE: handler_class is intentionally NOT set here.
    # ANI1ccxDatasetHandler is registered via @register_handler decorator and
    # discovered dynamically through the HandlerRegistry by create_dataset_handler().
    # Setting handler_class = None (default from BaseDataset) is correct.
    # The factory pattern handles handler instantiation via registry lookup.
    # We override create_handler() to use lazy import to avoid circular dependency.
    
    @classmethod
    def create_handler(
        cls,
        dataset_config,
        filter_config,
        processing_config,
        logger,
        experimental_setup=None
    ):
        """
        Factory method to create ANI1ccxDatasetHandler instance.
        
        Uses lazy import to avoid circular dependency between
        datasets/implementations/ani1ccx.py and handlers/implementations/ani1ccx.py.
        
        This pattern breaks the circular import chain:
            datasets/implementations/ani1ccx.py
                → handlers/dataset_handlers.py (module-level)
                    → config containers → dataset registry → ani1ccx.py (CYCLE!)
        
        By importing inside the method, the import only happens at runtime
        when create_handler() is called, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.ani1ccx import ANI1ccxDatasetHandler
        
        return ANI1ccxDatasetHandler(
            dataset_config,
            filter_config,
            processing_config,
            logger,
            experimental_setup
        )
    
    @classmethod
    def get_required_properties(cls) -> List[str]:
        """
        Return list of required properties for ANI-1ccx dataset.
        
        These are the minimum properties that must be present for
        each molecule in the preprocessed NPZ file.
        
        Note: Property names are mapped from HDF5 keys during preprocessing:
        - 'ccsd(t)_cbs.energy' → 'ccsd_energy'
        - 'atomic_numbers' → 'atoms' (converted to symbols)
        - 'coordinates' → 'coordinates'
        """
        return list(cls.schema.required_properties)
    
    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        """
        Return feature support dictionary for ANI-1ccx dataset.
        
        This indicates which analysis features are available for ANI-1ccx:
        - vibrational_analysis: False (no frequencies)
        - uncertainty_handling: False (CC is deterministic)
        - atomization_energy: True (can compute from ccsd_energy)
        - rotational_constants: False (not available)
        - frequency_analysis: False (no frequencies)
        - orbital_analysis: False (no MO data)
        - homo_lumo_gap: False (not available)
        - mo_energies: False (no MO data)
        """
        return cls.features.to_dict()
    
    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        """
        ANI-1ccx datasets use coordinate_based strategy.
        
        CRITICAL: ANI-1ccx HDF5 structure contains NO parseable chemical identifiers.
        The data structure (from Scientific Data Table 1):
        - 'atomic_numbers': Shape (Nc, Na) - atomic numbers only
        - 'coordinates': Shape (Nc, Na, 3) - 3D positions in Angstrom
        - No SMILES, no InChI, no compound labels
        
        Therefore, molecular connectivity must be inferred directly from 
        3D atomic coordinates using the rdDetermineBonds algorithm.
        
        Data requirements:
            - Atomic numbers (for atom types)
            - Coordinates in Angstrom (for 3D geometry and bond inference)
            - Molecular charge = 0 (ANI-1ccx contains only neutral molecules)
        
        Evidence: 
        - aiqm/ANI1x_datasets GitHub repository structure
        - Scientific Data paper Table 1 (no identifier columns)
        - MILIA blueprint decision tree for coordinate_based
        
        Returns:
            str: 'coordinate_based'
        """
        return 'coordinate_based'
