# milia_pipeline/datasets/implementations/rmd17.py

"""
rMD17 (Revised MD17) dataset implementation.

This module provides the RMD17Dataset class which encapsulates all rMD17-specific
metadata and configuration for the revised MD17 quantum chemistry dataset.

rMD17 Dataset Information:
--------------------------
- Source: Materials Cloud Archive (DOI: 10.24435/materialscloud:wy-kn)
- Download URL: https://archive.materialscloud.org/record/file?filename=rmd17.tar.bz2&record_id=466
- Reference: Christensen & von Lilienfeld, Mach. Learn.: Sci. Technol. 1, 045018 (2020)
- DOI: 10.1088/2632-2153/abba6f
- arXiv: 2007.09593
- Contains: ~100,000 conformations for each of 10 small organic molecules
- Properties: Total energies and forces computed at PBE/def2-SVP level
- Software: ORCA with very tight SCF convergence and very dense DFT integration grid
- Result: Dataset is practically free from numerical noise

CRITICAL: rMD17 uses coordinate_based molecule creation strategy.
The NPZ file structure contains only nuclear_charges and coordinates - 
NO parseable chemical identifiers (InChI/SMILES) are available.

IMPORTANT TRAINING WARNING FROM PAPER:
- Dataset sampled from MD trajectories at 500K (time-series, NOT independent samples)
- DO NOT train on more than 1000 samples due to autocorrelation issues
- Data published with 50K samples on original MD17 should be considered meaningless

Archive Structure:
------------------
- rmd17.tar.bz2 contains folder 'rmd17/' with 10 NPZ files
- Each NPZ file: rmd17_{molecule}.npz (e.g., rmd17_aspirin.npz)
- NPZ keys per file:
  - 'nuclear_charges': Atomic numbers (constant for all conformers)
  - 'coords': Coordinates in Angstrom (N_conf × N_atoms × 3)
  - 'energies': Total energy in kcal/mol (N_conf,)
  - 'forces': Cartesian forces in kcal/mol/Angstrom (N_conf × N_atoms × 3)
  - 'old_indices': Indices in original MD17 dataset
  - 'old_energies': Energies from original MD17 (kcal/mol)
  - 'old_forces': Forces from original MD17 (kcal/mol/Angstrom)

Molecules (10 total):
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

Evidence sources:
- Christensen & von Lilienfeld paper (2020) - methodology and data description
- Materials Cloud archive structure inspection
- Figshare dataset description (DOI: 10.6084/m9.figshare.12672038)
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


# NOTE: RMD17DatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_handler() method uses lazy import to
# instantiate the handler when needed. This follows the registry-based architecture
# pattern established in XXMD and documented in the Handler Migration Tracker.


# Unit conversion constant: 1 kcal/mol = 0.00159360144 Hartree
# Reference: NIST CODATA 2018
KCAL_MOL_TO_HARTREE = 0.00159360143764


@register
class RMD17Dataset(BaseDataset):
    """
    rMD17 (Revised MD17) quantum chemistry dataset implementation.
    
    rMD17 contains ~100,000 conformations for each of 10 small organic molecules.
    This is a revised version of the original MD17 dataset with recalculated
    energies and forces at PBE/def2-SVP level using very tight SCF convergence
    and dense DFT grid, making it practically noise-free.
    
    Properties in rMD17 (from Materials Cloud archive):
    ---------------------------------------------------
    - nuclear_charges: Atomic numbers (constant per molecule)
    - coords: Atomic coordinates (N_conf, N_atoms, 3) [Angstrom]
    - energies: Total DFT energy (N_conf,) [kcal/mol]
    - forces: Atomic forces (N_conf, N_atoms, 3) [kcal/mol/Angstrom]
    - old_indices: Reference to original MD17 conformer indices
    - old_energies: Original MD17 energies for comparison [kcal/mol]
    - old_forces: Original MD17 forces for comparison [kcal/mol/Angstrom]
    
    CRITICAL DIFFERENCES FROM ANI-1x/QM9:
    
    1. Molecule Creation Strategy: 'coordinate_based'
       - rMD17 NPZ structure contains NO parseable chemical identifiers
       - Only nuclear_charges (atomic numbers) and coordinates are available
       - Molecular connectivity inferred from 3D coordinates using rdDetermineBonds
       - All molecules are neutral (no charge calculation needed)
    
    2. Energy Units: kcal/mol (NOT Hartree)
       - rMD17 stores energies in kcal/mol
       - Preprocessor converts to Hartree for consistency with other MILIA datasets
       - Conversion: 1 kcal/mol = 0.00159360144 Hartree
    
    3. Data Format: tar.bz2 archive with multiple NPZ files
       - Requires custom preprocessor for archive extraction and NPZ merging
       - Each molecule in separate NPZ file, preprocessor combines all
    
    4. Identifier Keys: Empty tuple
       - Like ANI-1x, rMD17 has no identifier keys
       - Molecules identified by molecule_name + conformation_id compound identifier
    
    5. Training Warning:
       - DO NOT train on more than 1000 samples (autocorrelation in MD time-series)
       - This is documented in the dataset description
    
    Reference: Christensen, A.S. & von Lilienfeld, O.A. (2020). On the role of 
               gradients for machine learning of molecular energies and forces.
               Mach. Learn.: Sci. Technol. 1, 045018.
    """
    
    # List of molecules in rMD17 dataset
    MOLECULES = [
        'aspirin',        # C9H8O4, 21 atoms
        'azobenzene',     # C12H10N2, 24 atoms (99,988 conformations)
        'benzene',        # C6H6, 12 atoms
        'ethanol',        # C2H6O, 9 atoms
        'malonaldehyde',  # C3H4O2, 9 atoms
        'naphthalene',    # C10H8, 18 atoms
        'paracetamol',    # C8H9NO2, 20 atoms
        'salicylic',      # C7H6O3, 16 atoms
        'toluene',        # C7H8, 15 atoms
        'uracil',         # C4H4N2O2, 12 atoms
    ]
    
    metadata = DatasetMetadata(
        name="RMD17",
        version="1.0.0",
        description=(
            "Revised MD17 dataset with ~100,000 conformations for 10 small organic "
            "molecules. Energies and forces computed at PBE/def2-SVP level using "
            "ORCA with very tight SCF convergence and dense integration grid, "
            "making it practically noise-free. WARNING: DO NOT train on >1000 samples "
            "due to autocorrelation in MD time-series data."
        ),
        author="Anders S. Christensen, O. Anatole von Lilienfeld",
        license="CC BY 4.0",
    )
    
    schema = DatasetSchema(
        # Required properties that must be present for every molecule
        # Based on rMD17 NPZ structure - atoms and coordinates always present
        # energies is the primary energy target (in kcal/mol, converted to Hartree)
        required_properties=('energies', 'atoms', 'coordinates'),
        
        # Optional properties available in rMD17
        # Forces are the key feature of rMD17 for force field training
        optional_properties=(
            'forces',           # Atomic forces (kcal/mol/Angstrom -> Hartree/Angstrom)
            'molecule_name',    # Molecule type identifier (aspirin, benzene, etc.)
            'old_indices',      # Reference to original MD17 conformer indices
            'old_energies',     # Original MD17 energies for comparison (kcal/mol)
            'old_forces',       # Original MD17 forces for comparison
        ),
        
        # CRITICAL: Empty tuple - rMD17 has NO parseable chemical identifiers
        # The NPZ structure contains only nuclear_charges and coordinates
        # Molecule connectivity must be inferred from 3D coordinates
        identifier_keys=(),
        
        # rMD17 coordinates are in Angstrom (standard DFT output)
        coordinate_units='angstrom',
        
        # rMD17 energies are in kcal/mol (converted to Hartree during preprocessing)
        # Note: Original files use kcal/mol, but MILIA standardizes to Hartree
        energy_units='hartree',
    )
    
    features = DatasetFeatures(
        # rMD17 does not have vibrational frequencies
        vibrational_analysis=False,
        
        # rMD17 is deterministic DFT - no statistical uncertainties
        uncertainty_handling=False,
        
        # Atomization energies can be calculated from total energies
        atomization_energy=True,
        
        # rMD17 does not have rotational constants
        rotational_constants=False,
        
        # rMD17 does not have frequency analysis
        frequency_analysis=False,
        
        # rMD17 does not have orbital analysis
        orbital_analysis=False,
        
        # rMD17 does not have HOMO-LUMO gap
        homo_lumo_gap=False,
        
        # rMD17 does not have MO energies
        mo_energies=False,
    )
    
    # Configuration key matching config.yaml section
    config_key = "rmd17_config"
    
    # NOTE: handler_class is intentionally NOT set here.
    # RMD17DatasetHandler is registered via @register_handler decorator and
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
        Factory method to create RMD17DatasetHandler instance.
        
        Uses lazy import to avoid circular dependency between
        datasets/implementations/rmd17.py and handlers/implementations/rmd17.py.
        
        This pattern breaks the circular import chain:
            datasets/implementations/rmd17.py
                → handlers/dataset_handlers.py (module-level)
                    → config containers → dataset registry → rmd17.py (CYCLE!)
        
        By importing inside the method, the import only happens at runtime
        when create_handler() is called, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.rmd17 import RMD17DatasetHandler
        
        return RMD17DatasetHandler(
            dataset_config,
            filter_config,
            processing_config,
            logger,
            experimental_setup
        )
    
    @classmethod
    def get_required_properties(cls) -> List[str]:
        """
        Return list of required properties for rMD17 dataset.
        
        These are the minimum properties that must be present for
        each molecule in the preprocessed NPZ file.
        
        Note: Property names are mapped from source NPZ keys during preprocessing:
        - 'energies' -> 'energies' (converted from kcal/mol to Hartree)
        - 'nuclear_charges' -> 'atoms'
        - 'coords' -> 'coordinates'
        """
        return list(cls.schema.required_properties)
    
    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        """
        Return feature support dictionary for rMD17 dataset.
        
        This indicates which analysis features are available for rMD17:
        - vibrational_analysis: False (no frequencies)
        - uncertainty_handling: False (DFT is deterministic)
        - atomization_energy: True (can compute from energies)
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
        rMD17 dataset uses coordinate_based strategy.
        
        CRITICAL: rMD17 NPZ structure contains NO parseable chemical identifiers.
        The data structure contains:
        - 'nuclear_charges': Atomic numbers (constant per molecule)
        - 'coords': Shape (N_conf, N_atoms, 3) - 3D positions in Angstrom
        - No SMILES, no InChI, no compound labels
        
        Therefore, molecular connectivity must be inferred directly from 
        3D atomic coordinates using the rdDetermineBonds algorithm.
        
        Data requirements:
            - Atomic numbers (for atom types)
            - Coordinates in Angstrom (for 3D geometry and bond inference)
            - Molecular charge = 0 (all rMD17 molecules are neutral)
        
        Evidence: 
        - Materials Cloud archive structure (NPZ keys)
        - Figshare dataset description
        - MILIA blueprint decision tree for coordinate_based
        
        Returns:
            str: 'coordinate_based'
        """
        return 'coordinate_based'
    
    @classmethod
    def get_molecules(cls) -> List[str]:
        """
        Return list of molecules available in rMD17 dataset.
        
        Returns:
            List of molecule names: aspirin, azobenzene, benzene, ethanol,
            malonaldehyde, naphthalene, paracetamol, salicylic, toluene, uracil
        """
        return cls.MOLECULES.copy()
