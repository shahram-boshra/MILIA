# milia_pipeline/datasets/implementations/dft.py

"""
DFT (Density Functional Theory) dataset implementation.

This module provides the DFTDataset class which encapsulates all DFT-specific
metadata and configuration previously hardcoded in config_constants.py.

Evidence sources:
- config_constants.py lines 209-284 (hardcoded handler dictionaries)
- dataset_handlers.py lines 832-2039 (DFTDatasetHandler class)
- dataset_handlers.py lines 972-989 (get_molecule_creation_strategy)
"""

from typing import Dict, List

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


# NOTE: DFTDatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_handler() method uses lazy import to
# instantiate the handler when needed. This follows the registry-based architecture
# pattern established in XXMD and documented in the Handler Migration Tracker.


@register
class DFTDataset(BaseDataset):
    """
    DFT (Density Functional Theory) dataset implementation.
    
    DFT datasets contain quantum chemistry calculations including:
    - Total energies (Etot, U0, zpves)
    - Vibrational frequencies and modes
    - Rotational constants
    - Mulliken charges
    - Optimized 3D molecular geometries
    
    Molecule creation uses identifier_coordinate_based strategy:
    - InChI identifiers parsed for molecular connectivity
    - QM-optimized coordinates assigned for 3D geometry
    - Charge extracted from InChI /q layer
    
    Evidence: config_constants.py lines 220-227, 248-249, 255-256, 265-269, 280-281
    """
    
    metadata = DatasetMetadata(
        name="DFT",
        version="1.0.0",
        description="DFT quantum chemistry dataset with vibrational analysis and thermodynamic properties",
        author="MILIA Pipeline Team",
    )
    
    schema = DatasetSchema(
        required_properties=('Etot', 'atoms', 'coordinates'),
        optional_properties=('freqs', 'vibmodes', 'rots', 'dipoles'),
        identifier_keys=(('inchi', 'inchi'), ('graphs', 'smiles')),
        coordinate_units='angstrom',
        energy_units='hartree',
    )
    
    features = DatasetFeatures(
        vibrational_analysis=True,
        uncertainty_handling=False,
        atomization_energy=True,
        rotational_constants=True,
        frequency_analysis=True,
        orbital_analysis=False,
        homo_lumo_gap=False,
        mo_energies=False,
    )
    
    config_key = "dft_config"
    
    # NOTE: handler_class is intentionally NOT set here.
    # DFTDatasetHandler is registered via @register_handler decorator and
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
        Factory method to create DFTDatasetHandler instance.
        
        Uses lazy import to avoid circular dependency between
        datasets/implementations/dft.py and handlers/implementations/dft.py.
        
        This pattern breaks the circular import chain:
            datasets/implementations/dft.py
                → handlers/dataset_handlers.py (module-level)
                    → config containers → dataset registry → dft.py (CYCLE!)
        
        By importing inside the method, the import only happens at runtime
        when create_handler() is called, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.dft import DFTDatasetHandler
        
        return DFTDatasetHandler(
            dataset_config,
            filter_config,
            processing_config,
            logger,
            experimental_setup
        )
    
    @classmethod
    def get_required_properties(cls) -> List[str]:
        """
        Return list of required properties for DFT datasets.
        
        Evidence: config_constants.py line 249
        HANDLER_REQUIRED_PROPERTIES['DFT'] = ['Etot', 'atoms', 'coordinates']
        """
        return list(cls.schema.required_properties)
    
    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        """
        Return feature support dictionary for DFT datasets.
        
        Evidence: config_constants.py lines 221-227
        HANDLER_FEATURE_SUPPORT['DFT'] = {
            'vibrational_analysis': True,
            'uncertainty_handling': False,
            'atomization_energy': True,
            'rotational_constants': True,
            'frequency_analysis': True
        }
        """
        return cls.features.to_dict()
    
    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        """
        DFT datasets use identifier_coordinate_based strategy.
        
        DFT molecular data contains InChI identifiers which encode molecular
        connectivity and bonding. These are parsed to create the molecular graph,
        then QM-optimized coordinates are assigned to preserve exact 3D geometry.
        
        Evidence: dataset_handlers.py lines 972-989
        DFTDatasetHandler.get_molecule_creation_strategy() returns 'identifier_coordinate_based'
        
        Returns:
            str: 'identifier_coordinate_based'
        """
        return 'identifier_coordinate_based'
