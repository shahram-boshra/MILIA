# milia_pipeline/datasets/implementations/dmc.py

"""
DMC (Diffusion Monte Carlo) dataset implementation.

This module provides the DMCDataset class which encapsulates all DMC-specific
metadata and configuration previously hardcoded in config_constants.py.

Evidence sources:
- config_constants.py lines 209-284 (hardcoded handler dictionaries)
- dataset_handlers.py lines 2045-2955 (DMCDatasetHandler class)
- dataset_handlers.py lines 2190-2207 (get_molecule_creation_strategy)
"""

from typing import Dict, List

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


# NOTE: DMCDatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_handler() method uses lazy import to
# instantiate the handler when needed. This follows the registry-based architecture
# pattern established in XXMD and documented in the Handler Migration Tracker.


@register
class DMCDataset(BaseDataset):
    """
    DMC (Diffusion Monte Carlo) dataset implementation.
    
    DMC datasets contain quantum Monte Carlo calculations including:
    - Total energies with statistical uncertainties (Etot, std)
    - QMC-specific statistics and correlation data
    - Molecular structures from QMC-optimized geometries
    
    Key difference from DFT: DMC includes uncertainty/std values and
    supports uncertainty-aware processing and weighting.
    
    Molecule creation uses identifier_coordinate_based strategy:
    - InChI identifiers parsed for molecular connectivity
    - QMC-optimized coordinates assigned for 3D geometry
    - Charge extracted from InChI /q layer
    
    Evidence: config_constants.py lines 228-234, 250, 257, 270-273, 282
    """
    
    metadata = DatasetMetadata(
        name="DMC",
        version="1.0.0",
        description="DMC quantum Monte Carlo dataset with uncertainty handling",
        author="MILIA Pipeline Team",
    )
    
    schema = DatasetSchema(
        required_properties=('Etot', 'std', 'atoms', 'coordinates'),
        optional_properties=('qmc_stats', 'correlation_data'),
        identifier_keys=(('inchi', 'inchi'), ('graphs', 'smiles')),
        coordinate_units='angstrom',
        energy_units='hartree',
    )
    
    features = DatasetFeatures(
        vibrational_analysis=False,
        uncertainty_handling=True,
        atomization_energy=False,
        rotational_constants=False,
        frequency_analysis=False,
        orbital_analysis=False,
        homo_lumo_gap=False,
        mo_energies=False,
    )
    
    config_key = "dmc_config"
    
    # NOTE: handler_class is intentionally NOT set here.
    # DMCDatasetHandler is registered via @register_handler decorator and
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
        Factory method to create DMCDatasetHandler instance.
        
        Uses lazy import to avoid circular dependency between
        datasets/implementations/dmc.py and handlers/implementations/dmc.py.
        
        This pattern breaks the circular import chain:
            datasets/implementations/dmc.py
                → handlers/dataset_handlers.py (module-level)
                    → config containers → dataset registry → dmc.py (CYCLE!)
        
        By importing inside the method, the import only happens at runtime
        when create_handler() is called, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.dmc import DMCDatasetHandler
        
        return DMCDatasetHandler(
            dataset_config,
            filter_config,
            processing_config,
            logger,
            experimental_setup
        )
    
    @classmethod
    def get_required_properties(cls) -> List[str]:
        """
        Return list of required properties for DMC datasets.
        
        Evidence: config_constants.py line 250
        HANDLER_REQUIRED_PROPERTIES['DMC'] = ['Etot', 'std', 'atoms', 'coordinates']
        """
        return list(cls.schema.required_properties)
    
    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        """
        Return feature support dictionary for DMC datasets.
        
        Evidence: config_constants.py lines 228-234
        HANDLER_FEATURE_SUPPORT['DMC'] = {
            'vibrational_analysis': False,
            'uncertainty_handling': True,
            'atomization_energy': False,
            'rotational_constants': False,
            'frequency_analysis': False
        }
        """
        return cls.features.to_dict()
    
    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        """
        DMC datasets use identifier_coordinate_based strategy.
        
        DMC molecular data contains InChI identifiers which encode molecular
        connectivity and bonding. These are parsed to create the molecular graph,
        then QMC-optimized coordinates are assigned to preserve exact 3D geometry.
        
        Evidence: dataset_handlers.py lines 2190-2207
        DMCDatasetHandler.get_molecule_creation_strategy() returns 'identifier_coordinate_based'
        
        Returns:
            str: 'identifier_coordinate_based'
        """
        return 'identifier_coordinate_based'
