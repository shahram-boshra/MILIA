# milia_pipeline/datasets/implementations/wavefunction.py

"""
Wavefunction dataset implementation.

This module provides the WavefunctionDataset class which encapsulates all
Wavefunction-specific metadata and configuration.

CRITICAL DIFFERENCES FROM DFT/DMC:
1. Uses 'coordinate_based' strategy (not 'identifier_coordinate_based')
2. Coordinates are in Bohr (automatic conversion to Angstrom required)
3. Charge calculated from n_electrons, not InChI
4. Compound IDs are labels only, not parseable identifiers

Evidence sources:
- config_constants.py lines 209-284 (hardcoded handler dictionaries)
- dataset_handlers.py lines 2960-3800 (WavefunctionDatasetHandler class)
- dataset_handlers.py lines 3071-3091 (get_molecule_creation_strategy)
- dataset_handlers.py lines 3036-3069 (get_molecular_charge)
"""

from typing import Dict, List

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


# NOTE: WavefunctionDatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_handler() method uses lazy import to
# instantiate the handler when needed. This follows the registry-based architecture
# pattern established in XXMD and documented in the Handler Migration Tracker.


@register
class WavefunctionDataset(BaseDataset):
    """
    Wavefunction dataset implementation.
    
    Wavefunction datasets contain quantum mechanical wavefunction data including:
    - Molecular orbital energies and occupations
    - HOMO-LUMO gap
    - High-quality 3D coordinates from QM calculations
    
    CRITICAL DIFFERENCES FROM DFT/DMC:
    
    1. Molecule Creation Strategy: 'coordinate_based'
       - Compound IDs (e.g., 'BrCPxSiSxH4_331') are NOT parseable chemical identifiers
       - Molecular connectivity inferred from 3D coordinates using rdDetermineBonds
       - Molecular charge REQUIRED for accurate bond order assignment
    
    2. Coordinate Units: Bohr
       - Wavefunction data uses atomic units (Bohr)
       - Automatic conversion to Angstrom required during processing
    
    3. Charge Determination: From n_electrons
       - charge = n_electrons - sum(atomic_numbers)
       - NOT from InChI /q layer (compound IDs not parseable)
    
    4. Identifier Keys: compound_id (label only)
       - Used for tracking/logging only
       - NOT parsed for molecular structure
    
    Evidence: config_constants.py lines 235-244, 251, 258, 274-276, 283
    Evidence: dataset_handlers.py lines 3036-3091
    """
    
    metadata = DatasetMetadata(
        name="Wavefunction",
        version="1.0.0",
        description="Quantum mechanical wavefunction dataset from .molden files with orbital analysis",
        author="MILIA Pipeline Team",
    )
    
    schema = DatasetSchema(
        required_properties=('atoms', 'coordinates', 'compounds'),
        optional_properties=('mo_energies', 'mo_occupations', 'homo_lumo_gap_eV',
                            'total_energy'),
        identifier_keys=(('compounds', 'compound_id'),),
        coordinate_units='bohr',
        energy_units='eV',
    )
    
    features = DatasetFeatures(
        vibrational_analysis=False,
        uncertainty_handling=False,
        atomization_energy=False,
        rotational_constants=False,
        frequency_analysis=False,
        orbital_analysis=True,
        homo_lumo_gap=True,
        mo_energies=True,
    )
    
    config_key = "wavefunction_config"
    
    # NOTE: handler_class is intentionally NOT set here.
    # WavefunctionDatasetHandler is registered via @register_handler decorator and
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
        Factory method to create WavefunctionDatasetHandler instance.
        
        Uses lazy import to avoid circular dependency between
        datasets/implementations/wavefunction.py and handlers/implementations/wavefunction.py.
        
        This pattern breaks the circular import chain:
            datasets/implementations/wavefunction.py
                → handlers/dataset_handlers.py (module-level)
                    → config containers → dataset registry → wavefunction.py (CYCLE!)
        
        By importing inside the method, the import only happens at runtime
        when create_handler() is called, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.wavefunction import WavefunctionDatasetHandler
        
        return WavefunctionDatasetHandler(
            dataset_config,
            filter_config,
            processing_config,
            logger,
            experimental_setup
        )
    
    @classmethod
    def get_required_properties(cls) -> List[str]:
        """
        Return list of required properties for Wavefunction datasets.
        
        Evidence: config_constants.py line 251
        HANDLER_REQUIRED_PROPERTIES['Wavefunction'] = ['atoms', 'coordinates', 'compounds']
        
        Note: Unlike DFT/DMC, Wavefunction does NOT require 'Etot' as a core property.
        Energy information comes from orbital analysis (mo_energies, total_energy).
        """
        return list(cls.schema.required_properties)
    
    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        """
        Return feature support dictionary for Wavefunction datasets.
        
        Evidence: config_constants.py lines 235-244
        HANDLER_FEATURE_SUPPORT['Wavefunction'] = {
            'vibrational_analysis': False,
            'uncertainty_handling': False,
            'atomization_energy': False,
            'rotational_constants': False,
            'frequency_analysis': False,
            'orbital_analysis': True,
            'homo_lumo_gap': True,
            'mo_energies': True
        }
        """
        return cls.features.to_dict()
    
    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        """
        Wavefunction datasets use coordinate_based strategy.
        
        CRITICAL: Unlike DFT/DMC, Wavefunction compound IDs are NOT parseable
        chemical identifiers. Molecular connectivity must be inferred directly
        from 3D atomic coordinates using rdDetermineBonds algorithm.
        
        Molecular charge (calculated from n_electrons - sum(atomic_numbers)) is
        REQUIRED for accurate bond order assignment in rdDetermineBonds.
        
        Data requirements:
            - Compound label/identifier (for tracking only, not parsed)
            - Atomic numbers (for atom types)
            - Coordinates in Bohr (automatically converted to Angstrom)
            - Molecular charge from n_electrons (REQUIRED for rdDetermineBonds)
        
        Evidence: dataset_handlers.py lines 3071-3091
        WavefunctionDatasetHandler.get_molecule_creation_strategy() returns 'coordinate_based'
        
        Returns:
            str: 'coordinate_based'
        """
        return 'coordinate_based'
