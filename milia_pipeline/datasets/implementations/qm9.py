# milia_pipeline/datasets/implementations/qm9.py

"""
QM9 dataset implementation.

This module provides the QM9Dataset class which encapsulates all QM9-specific
metadata and configuration for the QM9 quantum chemistry dataset.

QM9 Dataset Information:
------------------------
- Source: Quantum chemistry structures and properties of 134k molecules
- Reference: Ramakrishnan et al., Scientific Data 1, 140022 (2014)
- DOI: 10.1038/sdata.2014.22
- Figshare: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
- Contains: 133,885 stable small organic molecules (CHONF) with up to 9 heavy atoms
- Properties: Geometric, energetic, electronic, and thermodynamic properties
- Method: B3LYP/6-31G(2df,p) level of DFT

Evidence sources:
- QM9 readme.txt from Figshare (property definitions and units)
- Scientific Data publication (methodology and data description)
- TensorFlow Datasets QM9 documentation (property list)
"""

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetFeatures,
    DatasetMetadata,
    DatasetSchema,
)
from milia_pipeline.datasets.registry import register

# NOTE: QM9DatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_handler() method uses lazy import to
# instantiate the handler when needed. This follows the registry-based architecture
# pattern established in XXMD and documented in the Handler Migration Tracker.


@register
class QM9Dataset(BaseDataset):
    """
    QM9 quantum chemistry dataset implementation.

    QM9 contains computed geometric, energetic, electronic, and thermodynamic
    properties for 133,885 stable small organic molecules made up of C, H, O, N, F.
    All properties were calculated at B3LYP/6-31G(2df,p) level of DFT.

    Properties in QM9 (from readme.txt):
    ------------------------------------
    - A, B, C: Rotational constants (GHz)
    - mu: Dipole moment (Debye)
    - alpha: Isotropic polarizability (Bohr³)
    - homo: HOMO energy (Hartree)
    - lumo: LUMO energy (Hartree)
    - gap: HOMO-LUMO gap (Hartree)
    - r2: Electronic spatial extent (Bohr²)
    - zpve: Zero point vibrational energy (Hartree)
    - U0: Internal energy at 0K (Hartree)
    - U: Internal energy at 298.15K (Hartree)
    - H: Enthalpy at 298.15K (Hartree)
    - G: Free energy at 298.15K (Hartree)
    - Cv: Heat capacity at 298.15K (cal/mol·K)
    - Mulliken charges: Per-atom partial charges (e)

    Molecule creation uses identifier_coordinate_based strategy:
    - SMILES identifiers encode molecular connectivity
    - B3LYP-optimized coordinates provide 3D geometry
    - Coordinates are in Angstrom
    - Energies are in Hartree

    Note: QM9 does NOT have a dedicated handler class because it follows
    the standard DFT-like processing pattern. The generic handler will be used.
    """

    metadata = DatasetMetadata(
        name="QM9",
        version="1.0.0",
        description=(
            "QM9 quantum chemistry dataset with 133,885 stable small organic molecules "
            "(CHONF, up to 9 heavy atoms). Properties computed at B3LYP/6-31G(2df,p) level."
        ),
        author="Ramakrishnan, Dral, Rupp, von Lilienfeld",
        license="CC0",
    )

    schema = DatasetSchema(
        # Required properties that must be present for every molecule
        # Based on QM9 XYZ file format - atoms and coordinates always present
        # U0 (internal energy at 0K) is the primary energy target
        required_properties=("U0", "atoms", "coordinates"),
        # Optional properties available in QM9
        # All 15 scalar properties from the QM9 readme.txt
        optional_properties=(
            "A",
            "B",
            "C",  # Rotational constants (GHz)
            "mu",  # Dipole moment (Debye)
            "alpha",  # Isotropic polarizability (Bohr³)
            "homo",
            "lumo",
            "gap",  # Orbital energies (Hartree)
            "r2",  # Electronic spatial extent (Bohr²)
            "zpve",  # Zero point vibrational energy (Hartree)
            "U",
            "H",
            "G",  # Thermodynamic energies (Hartree)
            "Cv",  # Heat capacity (cal/mol·K)
            "Qmulliken",  # Mulliken partial charges (per-atom)
            "freqs",  # Vibrational frequencies
            "smiles_relaxed",  # SMILES after geometry relaxation
            "inchi_relaxed",  # InChI after geometry relaxation
        ),
        # Identifier key mappings: (npz_key, identifier_type)
        # QM9 provides both SMILES and InChI identifiers
        # InChI is tried FIRST as it is MILIA's primary molecular scheme:
        # - InChI encodes complete hydrogen information explicitly
        # - InChI format: InChI=1S/CH4/h1H4 shows all H atoms
        # - No AddHs() call needed, avoiding RDKit sanitization issues
        # - SMILES is fallback only
        identifier_keys=(
            ("inchi", "inchi"),
            ("smiles", "smiles"),
        ),
        # QM9 coordinates are in Angstrom (from B3LYP optimization)
        coordinate_units="angstrom",
        # QM9 energies are in Hartree (standard QC output)
        energy_units="hartree",
    )

    features = DatasetFeatures(
        # QM9 has vibrational frequencies
        vibrational_analysis=True,
        # QM9 is deterministic DFT - no statistical uncertainties
        uncertainty_handling=False,
        # Atomization energies can be calculated from U0 and atomic references
        atomization_energy=True,
        # QM9 has rotational constants A, B, C
        rotational_constants=True,
        # QM9 has vibrational frequencies
        frequency_analysis=True,
        # QM9 does not have full orbital analysis (only HOMO/LUMO)
        orbital_analysis=False,
        # QM9 has HOMO-LUMO gap
        homo_lumo_gap=True,
        # QM9 has HOMO and LUMO energies but not full MO spectrum
        mo_energies=False,
    )

    # Configuration key matching config.yaml section
    config_key = "qm9_config"

    # NOTE: handler_class is intentionally NOT set here.
    # QM9DatasetHandler is registered via @register_handler decorator and
    # discovered dynamically through the HandlerRegistry by create_dataset_handler().
    # Setting handler_class = None (default from BaseDataset) is correct.
    # The factory pattern handles handler instantiation via registry lookup.
    # We override create_handler() to use lazy import to avoid circular dependency.

    @classmethod
    def create_handler(
        cls, dataset_config, filter_config, processing_config, logger, experimental_setup=None
    ):
        """
        Factory method to create QM9DatasetHandler instance.

        Uses lazy import to avoid circular dependency between
        datasets/implementations/qm9.py and handlers/implementations/qm9.py.

        This pattern breaks the circular import chain:
            datasets/implementations/qm9.py
                → handlers/dataset_handlers.py (module-level)
                    → config containers → dataset registry → qm9.py (CYCLE!)

        By importing inside the method, the import only happens at runtime
        when create_handler() is called, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.qm9 import QM9DatasetHandler

        return QM9DatasetHandler(
            dataset_config, filter_config, processing_config, logger, experimental_setup
        )

    @classmethod
    def get_required_properties(cls) -> list[str]:
        """
        Return list of required properties for QM9 dataset.

        These are the minimum properties that must be present for
        each molecule in the preprocessed NPZ file.
        """
        return list(cls.schema.required_properties)

    @classmethod
    def get_feature_support(cls) -> dict[str, bool]:
        """
        Return feature support dictionary for QM9 dataset.

        This indicates which analysis features are available for QM9:
        - vibrational_analysis: True (frequencies available)
        - uncertainty_handling: False (DFT is deterministic)
        - atomization_energy: True (can compute from U0)
        - rotational_constants: True (A, B, C available)
        - frequency_analysis: True (frequencies available)
        - orbital_analysis: False (only HOMO/LUMO, not full spectrum)
        - homo_lumo_gap: True (gap property available)
        - mo_energies: False (only HOMO/LUMO energies)
        """
        return cls.features.to_dict()

    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        """
        QM9 datasets use identifier_coordinate_based strategy.

        QM9 provides SMILES strings that encode complete molecular
        connectivity and bonding information. These are parsed to create
        the molecular graph, then B3LYP-optimized Cartesian coordinates
        are assigned to provide the exact 3D geometry.

        This is the same strategy used by DFT datasets because:
        1. SMILES/InChI identifiers are chemically parseable
        2. They encode complete bond topology
        3. 3D coordinates come from quantum chemistry optimization

        Returns:
            str: 'identifier_coordinate_based'
        """
        return "identifier_coordinate_based"
