# milia_pipeline/datasets/implementations/ani2x.py

"""
ANI-2x dataset implementation.

This module provides the ANI2xDataset class which encapsulates all ANI-2x-specific
metadata and configuration for the ANI-2x quantum chemistry dataset.

ANI-2x Dataset Information:
---------------------------
- Source: ANI-2x dataset with DFT conformations for organic molecules
- Reference: Devereux et al., J. Chem. Theory Comput. 16, 4192-4202 (2020)
- DOI: 10.1021/acs.jctc.0c00121
- Zenodo: https://zenodo.org/records/10108942
- Contains: DFT conformations for organic molecules containing H, C, N, O, S, F, Cl
- Properties: DFT energies, forces computed at ωB97X/6-31G(d) level
- Method: Active learning sampling extending ANI-1x to include sulfur and halogens

CRITICAL: ANI-2x uses coordinate_based molecule creation strategy.
The HDF5 file structure contains only atomic_numbers (species) and coordinates -
NO parseable chemical identifiers (InChI/SMILES) are available.

Key Differences from ANI-1x:
- ANI-1x: H, C, N, O (4 elements)
- ANI-2x: H, C, N, O, S, F, Cl (7 elements - ~90% of drug-like molecules)
- ANI-2x has torsional refinement training for better torsion profiles

Evidence sources:
- Devereux et al., J. Chem. Theory Comput. 2020
- Zenodo record 10108942 (dataset release)
- MILIA_Adding_New_Datasets_Implementation_Blueprint.md (decision tree analysis)
"""

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetFeatures,
    DatasetMetadata,
    DatasetSchema,
)
from milia_pipeline.datasets.registry import register

# NOTE: ANI2xDatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_handler() method uses lazy import to
# instantiate the handler when needed. This follows the registry-based architecture
# pattern established in XXMD and documented in the Handler Migration Tracker.


@register
class ANI2xDataset(BaseDataset):
    """
    ANI-2x quantum chemistry dataset implementation.

    ANI-2x contains DFT conformations for organic molecules containing
    H, C, N, O, S, F, Cl atoms. Properties were computed at ωB97X/6-31G(d) level
    of theory using active learning sampling, extending ANI-1x to sulfur and halogens.

    Properties in ANI-2x (from Zenodo supplementary information):
    -------------------------------------------------------------
    - coordinates: Atomic coordinates (Nc, Na, 3) [Angstrom]
    - species/atomic_numbers: Atomic numbers (Nc, Na) or (Na,)
    - energies: DFT energy (Nc,) [Hartree]
    - forces: Atomic forces (Nc, Na, 3) [Hartree/Angstrom] (if available)

    Where: Nc = number of conformations, Na = number of atoms

    CRITICAL DIFFERENCES FROM ANI-1x:

    1. Extended Element Coverage: H, C, N, O, S, F, Cl (7 elements)
       - ANI-1x only covers H, C, N, O (4 elements)
       - These 7 elements make up ~90% of drug-like molecules

    2. Torsional Refinement: ANI-2x underwent additional torsional refinement
       training for better prediction of molecular torsion profiles

    3. Same Molecule Creation Strategy: 'coordinate_based'
       - ANI-2x HDF5 structure contains NO parseable chemical identifiers
       - Only species (atomic_numbers) and coordinates are available per molecule
       - Molecular connectivity inferred from 3D coordinates using rdDetermineBonds
       - All molecules are neutral (no charge calculation needed)

    Reference: Devereux, C., Smith, J.S., et al. Extending the Applicability of the
               ANI Deep Learning Molecular Potential to Sulfur and Halogens.
               J. Chem. Theory Comput. 2020, 16, 7, 4192-4202.
               DOI: 10.1021/acs.jctc.0c00121
    """

    metadata = DatasetMetadata(
        name="ANI2x",
        version="1.0.0",
        description=(
            "ANI-2x dataset with DFT conformations for organic molecules "
            "(H, C, N, O, S, F, Cl). Properties computed at ωB97X/6-31G(d) level "
            "using active learning, extending ANI-1x to sulfur and halogens."
        ),
        author="Devereux, Smith, Huddleston, Barros, Zubatyuk, Isayev, Roitberg",
        license="CC-BY-4.0",
    )

    schema = DatasetSchema(
        # Required properties that must be present for every molecule
        # Based on ANI-2x HDF5 structure - atoms and coordinates always present
        # energies is the primary energy target (mapped to 'energy' in NPZ)
        required_properties=("energy", "atoms", "coordinates"),
        # Optional properties available in ANI-2x
        # Forces may or may not be present depending on the HDF5 file variant
        optional_properties=(
            "forces",  # Atomic forces (Hartree/Angstrom)
            "molecule_id",  # Molecule group identifier from HDF5
        ),
        # CRITICAL: Empty tuple - ANI-2x has NO parseable chemical identifiers
        # The HDF5 structure contains only species (atomic_numbers) and coordinates
        # Molecule connectivity must be inferred from 3D coordinates
        identifier_keys=(),
        # ANI-2x coordinates are in Angstrom (DFT calculations)
        coordinate_units="angstrom",
        # ANI-2x energies are in Hartree (standard DFT output)
        energy_units="hartree",
    )

    features = DatasetFeatures(
        # ANI-2x does not have vibrational frequencies
        vibrational_analysis=False,
        # ANI-2x is deterministic DFT - no statistical uncertainties
        uncertainty_handling=False,
        # Atomization energies can be calculated from total energy
        atomization_energy=True,
        # ANI-2x does not have rotational constants
        rotational_constants=False,
        # ANI-2x does not have frequency analysis
        frequency_analysis=False,
        # ANI-2x does not have orbital analysis
        orbital_analysis=False,
        # ANI-2x does not have HOMO-LUMO gap
        homo_lumo_gap=False,
        # ANI-2x does not have MO energies
        mo_energies=False,
    )

    # Configuration key matching config.yaml section
    config_key = "ani2x_config"

    # NOTE: handler_class is intentionally NOT set here.
    # ANI2xDatasetHandler is registered via @register_handler decorator and
    # discovered dynamically through the HandlerRegistry by create_dataset_handler().
    # Setting handler_class = None (default from BaseDataset) is correct.
    # The factory pattern handles handler instantiation via registry lookup.
    # We override create_handler() to use lazy import to avoid circular dependency.

    @classmethod
    def create_handler(
        cls, dataset_config, filter_config, processing_config, logger, experimental_setup=None
    ):
        """
        Factory method to create ANI2xDatasetHandler instance.

        Uses lazy import to avoid circular dependency between
        datasets/implementations/ani2x.py and handlers/implementations/ani2x.py.

        This pattern breaks the circular import chain:
            datasets/implementations/ani2x.py
                → handlers/dataset_handlers.py (module-level)
                    → config containers → dataset registry → ani2x.py (CYCLE!)

        By importing inside the method, the import only happens at runtime
        when create_handler() is called, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.ani2x import ANI2xDatasetHandler

        return ANI2xDatasetHandler(
            dataset_config, filter_config, processing_config, logger, experimental_setup
        )

    @classmethod
    def get_required_properties(cls) -> list[str]:
        """
        Return list of required properties for ANI-2x dataset.

        These are the minimum properties that must be present for
        each molecule in the preprocessed NPZ file.

        Note: Property names are mapped from HDF5 keys during preprocessing:
        - 'energies' → 'energy'
        - 'species'/'atomic_numbers' → 'atoms'
        - 'coordinates' → 'coordinates'
        """
        return list(cls.schema.required_properties)

    @classmethod
    def get_feature_support(cls) -> dict[str, bool]:
        """
        Return feature support dictionary for ANI-2x dataset.

        This indicates which analysis features are available for ANI-2x:
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
        ANI-2x datasets use coordinate_based strategy.

        CRITICAL: ANI-2x HDF5 structure contains NO parseable chemical identifiers.
        The data structure contains:
        - 'species' or 'atomic_numbers': Atomic numbers
        - 'coordinates': Shape (Nc, Na, 3) - 3D positions in Angstrom
        - No SMILES, no InChI, no compound labels

        Therefore, molecular connectivity must be inferred directly from
        3D atomic coordinates using the rdDetermineBonds algorithm.

        Data requirements:
            - Atomic numbers (for atom types)
            - Coordinates in Angstrom (for 3D geometry and bond inference)
            - Molecular charge = 0 (ANI-2x contains only neutral molecules)

        Evidence:
        - Zenodo record 10108942 dataset structure
        - ANI-2x paper (Devereux et al., 2020)
        - MILIA blueprint decision tree for coordinate_based

        Returns:
            str: 'coordinate_based'
        """
        return "coordinate_based"
