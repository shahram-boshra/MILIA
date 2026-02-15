# milia_pipeline/datasets/implementations/ani1x.py

"""
ANI-1x dataset implementation.

This module provides the ANI1xDataset class which encapsulates all ANI-1x-specific
metadata and configuration for the ANI-1x quantum chemistry dataset.

ANI-1x Dataset Information:
---------------------------
- Source: ANI-1x dataset with ~5 million DFT conformations
- Reference: Smith et al., Scientific Data 7, 134 (2020)
- DOI: 10.1038/s41597-020-0473-z
- Figshare: https://doi.org/10.6084/m9.figshare.10047041
- Contains: ~5 million non-equilibrium conformations for organic molecules (CHNO)
- Properties: DFT energies, forces, charges, dipoles computed at ωB97x/6-31G* level
- Method: Active learning sampling with ωB97x/6-31G* DFT calculations

CRITICAL: ANI-1x uses coordinate_based molecule creation strategy.
The HDF5 file structure contains only atomic_numbers and coordinates -
NO parseable chemical identifiers (InChI/SMILES) are available.

Evidence sources:
- Scientific Data paper Table 1 (data layout)
- GitHub aiqm/ANI1x_datasets (dataloader.py implementation)
- MILIA_Adding_New_Datasets_Implementation_Blueprint.md (decision tree analysis)
"""

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetFeatures,
    DatasetMetadata,
    DatasetSchema,
)
from milia_pipeline.datasets.registry import register

# NOTE: ANI1xDatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_handler() method uses lazy import to
# instantiate the handler when needed. This follows the registry-based architecture
# pattern established in XXMD and documented in the Handler Migration Tracker.


@register
class ANI1xDataset(BaseDataset):
    """
    ANI-1x quantum chemistry dataset implementation.

    ANI-1x contains ~5 million DFT conformations for organic molecules containing
    H, C, N, O atoms. Properties were computed at ωB97x/6-31G* level of theory
    using active learning sampling.

    Properties in ANI-1x (from Scientific Data Table 1):
    ----------------------------------------------------
    - coordinates: Atomic coordinates (Nc, Na, 3) [Angstrom]
    - atomic_numbers: Atomic numbers (Nc, Na) [uint8]
    - wb97x_dz.energy: DFT energy (Nc,) [Hartree]
    - wb97x_dz.forces: Atomic forces (Nc, Na, 3) [Hartree/Angstrom]
    - wb97x_dz.hirshfeld_charges: Hirshfeld charges (Nc, Na) [e]
    - wb97x_dz.cm5_charges: CM5 charges (Nc, Na) [e]
    - wb97x_dz.dipole: Molecular dipole (Nc, 3) [Debye]

    Where: Nc = number of conformations, Na = number of atoms

    CRITICAL DIFFERENCES FROM QM9/DFT:

    1. Molecule Creation Strategy: 'coordinate_based'
       - ANI-1x HDF5 structure contains NO parseable chemical identifiers
       - Only atomic_numbers and coordinates are available per molecule
       - Molecular connectivity inferred from 3D coordinates using rdDetermineBonds
       - All molecules are neutral (no charge calculation needed)

    2. Data Format: HDF5 (not NPZ or tar.bz2)
       - Requires custom preprocessor for HDF5 → NPZ conversion
       - Uses iter_data_buckets pattern from aiqm/ANI1x_datasets

    3. Identifier Keys: Empty tuple
       - Unlike QM9 (InChI/SMILES) or Wavefunction (compound_id),
         ANI-1x has no identifier keys at all

    Reference: Smith, J.S., et al. The ANI-1ccx and ANI-1x data sets,
               coupled-cluster and density functional theory properties
               for molecules. Sci Data 7, 134 (2020).
    """

    metadata = DatasetMetadata(
        name="ANI1x",
        version="1.0.0",
        description=(
            "ANI-1x dataset with ~5 million DFT conformations for organic molecules "
            "(CHNO). Properties computed at ωB97x/6-31G* level using active learning."
        ),
        author="Smith, Nebgen, Lubbers, Isayev, Roitberg",
        license="CC0",
    )

    schema = DatasetSchema(
        # Required properties that must be present for every molecule
        # Based on ANI-1x HDF5 structure - atoms and coordinates always present
        # wb97x_dz.energy is the primary energy target (mapped to 'energy' in NPZ)
        required_properties=("energy", "atoms", "coordinates"),
        # Optional properties available in ANI-1x
        # All properties from the HDF5 file (keys mapped during preprocessing)
        optional_properties=(
            "forces",  # wb97x_dz.forces (Hartree/Angstrom)
            "hirshfeld_charges",  # wb97x_dz.hirshfeld_charges (e)
            "cm5_charges",  # wb97x_dz.cm5_charges (e)
            "dipole",  # wb97x_dz.dipole (Debye)
            "molecule_id",  # Molecule group identifier from HDF5
        ),
        # CRITICAL: Empty tuple - ANI-1x has NO parseable chemical identifiers
        # The HDF5 structure contains only atomic_numbers and coordinates
        # Molecule connectivity must be inferred from 3D coordinates
        identifier_keys=(),
        # ANI-1x coordinates are in Angstrom (DFT calculations)
        coordinate_units="angstrom",
        # ANI-1x energies are in Hartree (standard DFT output)
        energy_units="hartree",
    )

    features = DatasetFeatures(
        # ANI-1x does not have vibrational frequencies
        vibrational_analysis=False,
        # ANI-1x is deterministic DFT - no statistical uncertainties
        uncertainty_handling=False,
        # Atomization energies can be calculated from wb97x_dz.energy
        atomization_energy=True,
        # ANI-1x does not have rotational constants
        rotational_constants=False,
        # ANI-1x does not have frequency analysis
        frequency_analysis=False,
        # ANI-1x does not have orbital analysis
        orbital_analysis=False,
        # ANI-1x does not have HOMO-LUMO gap
        homo_lumo_gap=False,
        # ANI-1x does not have MO energies
        mo_energies=False,
    )

    # Configuration key matching config.yaml section
    config_key = "ani1x_config"

    # NOTE: handler_class is intentionally NOT set here.
    # ANI1xDatasetHandler is registered via @register_handler decorator and
    # discovered dynamically through the HandlerRegistry by create_dataset_handler().
    # Setting handler_class = None (default from BaseDataset) is correct.
    # The factory pattern handles handler instantiation via registry lookup.
    # We override create_handler() to use lazy import to avoid circular dependency.

    @classmethod
    def create_handler(
        cls, dataset_config, filter_config, processing_config, logger, experimental_setup=None
    ):
        """
        Factory method to create ANI1xDatasetHandler instance.

        Uses lazy import to avoid circular dependency between
        datasets/implementations/ani1x.py and handlers/implementations/ani1x.py.

        This pattern breaks the circular import chain:
            datasets/implementations/ani1x.py
                → handlers/dataset_handlers.py (module-level)
                    → config containers → dataset registry → ani1x.py (CYCLE!)

        By importing inside the method, the import only happens at runtime
        when create_handler() is called, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.ani1x import ANI1xDatasetHandler

        return ANI1xDatasetHandler(
            dataset_config, filter_config, processing_config, logger, experimental_setup
        )

    @classmethod
    def get_required_properties(cls) -> list[str]:
        """
        Return list of required properties for ANI-1x dataset.

        These are the minimum properties that must be present for
        each molecule in the preprocessed NPZ file.

        Note: Property names are mapped from HDF5 keys during preprocessing:
        - 'wb97x_dz.energy' → 'energy'
        - 'atomic_numbers' → 'atoms' (converted to symbols)
        - 'coordinates' → 'coordinates'
        """
        return list(cls.schema.required_properties)

    @classmethod
    def get_feature_support(cls) -> dict[str, bool]:
        """
        Return feature support dictionary for ANI-1x dataset.

        This indicates which analysis features are available for ANI-1x:
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
        ANI-1x datasets use coordinate_based strategy.

        CRITICAL: ANI-1x HDF5 structure contains NO parseable chemical identifiers.
        The data structure (from Scientific Data Table 1):
        - 'atomic_numbers': Shape (Nc, Na) - atomic numbers only
        - 'coordinates': Shape (Nc, Na, 3) - 3D positions in Angstrom
        - No SMILES, no InChI, no compound labels

        Therefore, molecular connectivity must be inferred directly from
        3D atomic coordinates using the rdDetermineBonds algorithm.

        Data requirements:
            - Atomic numbers (for atom types)
            - Coordinates in Angstrom (for 3D geometry and bond inference)
            - Molecular charge = 0 (ANI-1x contains only neutral molecules)

        Evidence:
        - aiqm/ANI1x_datasets GitHub repository structure
        - Scientific Data paper Table 1 (no identifier columns)
        - MILIA blueprint decision tree for coordinate_based

        Returns:
            str: 'coordinate_based'
        """
        return "coordinate_based"
