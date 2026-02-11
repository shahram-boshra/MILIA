# milia_pipeline/datasets/implementations/xxmd.py

"""
xxMD (Extended Excited-state Molecular Dynamics) Dataset Implementation.

This module provides the XXMDDataset class which encapsulates all xxMD-specific
metadata and configuration for the xxMD quantum chemistry dataset.

xxMD Dataset Information:
-------------------------
- Source: Zenodo (DOI: 10.5281/zenodo.10393859)
- GitHub: https://github.com/zpengmei/xxMD
- Reference: Pengmei, Liu, Shu. Scientific Data 11, 222 (2024)
- DOI: 10.1038/s41597-024-03019-3
- Contains: Nonadiabatic dynamics trajectories for 4 photochemically active molecules
- Molecules: azobenzene, dithiophene, malonaldehyde, stilbene
- Two subsets: xxMD-CASSCF (multi-state) and xxMD-DFT (ground state only)
- Format: Extended XYZ files processed with ASE (Atomic Simulation Environment)
- Method (xxMD-DFT): M06 exchange-correlation functional (spin-polarized KS-DFT)

xxMD-DFT Dataset Structure:
---------------------------
xxMD-main.zip
└── xxMD-DFT/
    ├── azobenzene/
    │   ├── train.xyz
    │   ├── val.xyz
    │   └── test.xyz
    ├── dithiophene/
    │   └── ...
    ├── malonaldehyde/
    │   └── ...
    └── stilbene/
        └── ...

Extended XYZ Format (ASE):
--------------------------
Each frame contains:
- energy: Total DFT energy (eV, ASE default units)
- forces: Atomic forces (eV/Angstrom, ASE default units)
- positions: Atomic coordinates (Angstrom)
- species: Element symbols

CRITICAL: xxMD uses coordinate_based molecule creation strategy.
The extended XYZ format contains only atomic positions and species -
NO parseable chemical identifiers (InChI/SMILES) are available.

Evidence sources:
- Scientific Data paper (structure description, Table 2)
- GitHub repository (xxMD-DFT directory structure)
- ASE documentation (extended XYZ format, eV/Angstrom units)

Author: MILIA Pipeline Team
Version: 1.0
Date: January 2026
"""

from typing import Dict, List

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


# NOTE: XXMDDatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_dataset_handler() factory function will
# find it when needed. This follows the registry-based architecture pattern.


@register
class XXMDDataset(BaseDataset):
    """
    xxMD (Extended Excited-state Molecular Dynamics) dataset implementation.
    
    xxMD contains nonadiabatic dynamics trajectories for photochemically active
    molecules computed at SA-CASSCF and M06 DFT levels of theory. This implementation
    focuses on the xxMD-DFT subset (ground state, M06 functional).
    
    Key Characteristics:
    --------------------
    - Contains 4 molecules: azobenzene, dithiophene, malonaldehyde, stilbene
    - Trajectories from nonadiabatic dynamics (larger configuration space)
    - Includes regions near conical intersections
    - Samples reactants, transition states, products
    - More challenging benchmark than MD17/rMD17
    
    Properties in xxMD-DFT (from ASE extended XYZ):
    -----------------------------------------------
    - positions: Atomic coordinates (Angstrom)
    - species/numbers: Atomic numbers/symbols
    - energy: M06 DFT total energy (eV, ASE default)
    - forces: Atomic forces (eV/Angstrom, ASE default)
    
    CRITICAL DIFFERENCES FROM QM9/DFT:
    
    1. Molecule Creation Strategy: 'coordinate_based'
       - xxMD extended XYZ contains NO parseable chemical identifiers
       - Only atomic positions and species are available
       - Molecular connectivity inferred from 3D coordinates using rdDetermineBonds
       - All molecules are neutral (no charge calculation needed)
    
    2. Data Format: Extended XYZ (zip archive with .xyz files)
       - Requires custom preprocessor for XYZ → NPZ conversion
       - Uses ASE for extended XYZ parsing
       - Pre-split into train/val/test based on temporal information
    
    3. Energy Units: eV (ASE default)
       - Unlike Hartree used in most quantum chemistry datasets
       - Will be converted to Hartree in preprocessor for consistency
    
    4. Identifier Keys: Empty tuple
       - Unlike QM9 (InChI/SMILES), xxMD has no identifier keys
    
    Reference: Pengmei, Z., Liu, J. & Shu, Y. Beyond MD17: the reactive xxMD dataset.
               Sci Data 11, 222 (2024). https://doi.org/10.1038/s41597-024-03019-3
    """
    
    metadata = DatasetMetadata(
        name="XXMD",
        version="1.0.0",
        description=(
            "xxMD (Extended Excited-state Molecular Dynamics) dataset containing "
            "nonadiabatic dynamics trajectories for 4 photochemically active molecules "
            "(azobenzene, dithiophene, malonaldehyde, stilbene). Properties computed at "
            "M06 DFT level. Samples larger configuration space including transition states "
            "and conical intersections."
        ),
        author="Pengmei, Liu, Shu",
        license="CC0",
    )
    
    schema = DatasetSchema(
        # Required properties that must be present for every molecule
        # Based on xxMD extended XYZ structure
        required_properties=('energy', 'atoms', 'coordinates'),
        
        # Optional properties available in xxMD
        optional_properties=(
            'forces',           # Atomic forces (eV/Angstrom → Hartree/Angstrom)
            'molecule_name',    # Molecule identifier (azobenzene, dithiophene, etc.)
            'split',            # train/val/test split indicator
        ),
        
        # CRITICAL: Empty tuple - xxMD has NO parseable chemical identifiers
        # Extended XYZ format contains only atomic positions and species
        # Molecule connectivity must be inferred from 3D coordinates
        identifier_keys=(),
        
        # xxMD coordinates are in Angstrom (ASE default)
        coordinate_units='angstrom',
        
        # xxMD energies are in Hartree (converted from eV during preprocessing)
        # IMPORTANT: Preprocessor converts eV → Hartree for MILIA standardization
        # The schema reflects the POST-PREPROCESSING units in the NPZ file
        energy_units='hartree',
    )
    
    features = DatasetFeatures(
        # xxMD does not have vibrational frequencies
        vibrational_analysis=False,
        
        # xxMD is deterministic DFT - no statistical uncertainties
        uncertainty_handling=False,
        
        # Atomization energies can be calculated from total energy
        atomization_energy=True,
        
        # xxMD does not have rotational constants
        rotational_constants=False,
        
        # xxMD does not have frequency analysis
        frequency_analysis=False,
        
        # xxMD does not have orbital analysis
        orbital_analysis=False,
        
        # xxMD does not have HOMO-LUMO gap
        homo_lumo_gap=False,
        
        # xxMD does not have MO energies
        mo_energies=False,
    )
    
    # Configuration key matching config.yaml section
    config_key = "xxmd_config"
    
    # NOTE: handler_class is intentionally NOT set here.
    # XXMDDatasetHandler is registered via @register_handler decorator and
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
        Factory method to create XXMDDatasetHandler instance.
        
        Uses lazy import to avoid circular dependency between
        datasets/implementations/xxmd.py and handlers/implementations/xxmd.py.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.xxmd import XXMDDatasetHandler
        
        return XXMDDatasetHandler(
            dataset_config,
            filter_config,
            processing_config,
            logger,
            experimental_setup
        )
    
    @classmethod
    def get_required_properties(cls) -> List[str]:
        """
        Return list of required properties for xxMD dataset.
        
        These are the minimum properties that must be present for
        each molecule in the preprocessed NPZ file.
        
        Note: Property names are mapped from extended XYZ keys during preprocessing:
        - 'energy' → 'energy' (converted from eV to Hartree)
        - Atomic numbers → 'atoms'
        - 'positions' → 'coordinates'
        """
        return list(cls.schema.required_properties)
    
    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        """
        Return feature support dictionary for xxMD dataset.
        
        This indicates which analysis features are available for xxMD:
        - vibrational_analysis: False (no frequencies)
        - uncertainty_handling: False (DFT is deterministic)
        - atomization_energy: True (can compute from energy)
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
        xxMD datasets use coordinate_based strategy.
        
        CRITICAL: xxMD extended XYZ format contains NO parseable chemical identifiers.
        The data structure only contains:
        - Atomic species/numbers
        - 3D atomic positions (coordinates)
        - Energy and forces as properties
        - No SMILES, no InChI, no compound labels
        
        Therefore, molecular connectivity must be inferred directly from 
        3D atomic coordinates using the rdDetermineBonds algorithm.
        
        Data requirements:
            - Atomic numbers (for atom types)
            - Coordinates in Angstrom (for 3D geometry and bond inference)
            - Molecular charge = 0 (xxMD contains only neutral molecules)
        
        Evidence: 
        - xxMD GitHub repository structure (extended XYZ files)
        - Scientific Data paper (data description)
        - MILIA blueprint decision tree for coordinate_based
        
        Returns:
            str: 'coordinate_based'
        """
        return 'coordinate_based'
