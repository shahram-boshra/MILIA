# milia_pipeline/datasets/implementations/qdpi.py

"""
QDπ Dataset Implementation
==========================

Dataset class for QDπ (Quantum Deep Potential Interaction) dataset.

QDπ Dataset Information:
------------------------
- Source: Zenodo (DOI: 10.5281/zenodo.14970869)
- Reference: Zeng et al., Scientific Data 12, 693 (2025)
- DOI: 10.1038/s41597-025-04972-3
- Contents: ~1.6 million structures for drug discovery ML potentials
- Method: ωB97M-D3(BJ)/def2-TZVPPD (highly accurate DFT functional)
- Elements: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I (13 elements)
- Format: DeePMD-kit HDF5 format

CRITICAL: QDπ Contains Both Neutral AND Charged Molecules
----------------------------------------------------------
Unlike ANI-2x (neutral-only), QDπ dataset is partitioned into:
- data/neutral/ : Neutral molecules (charge = 0)
- data/charged/ : Charged molecules (ions, protonated/deprotonated species)

The charge is NOT stored explicitly in the HDF5 format. It must be:
1. Tracked during preprocessing based on file path
2. Stored in the NPZ as 'molecular_charge' property
3. Retrieved by the handler's get_molecular_charge() method

This is essential for correct bond order determination via rdDetermineBonds.

Source datasets included in QDπ:
- SPICE: dipeptides, solvated amino acids, ion pairs, PubChem molecules
- ANI: neutral organic molecules (H, C, N, O, S, F, Cl)
- GEOM: drug-like molecules
- FreeSolv: solvation free energy molecules
- RE: relative energy benchmark sets (tautomers, protonation states)
- COMP6: benchmark validation sets

HDF5 Structure (DeePMD-kit format, Table 5 from paper):
-------------------------------------------------------
Groups organized by chemical formula, each containing:
- 'elements': Element symbols array (e.g., ['H', 'C', 'N', 'O'])
- 'atomic_types': Integer indices into elements array
- 'energies': Total energies (eV) - converted to Hartree
- 'coordinates': Atomic positions (Angstrom)
- 'forces': Atomic forces (eV/Angstrom) - converted to Hartree/Angstrom
"""

from typing import ClassVar

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetFeatures,
    DatasetMetadata,
    DatasetSchema,
)
from milia_pipeline.datasets.registry import register


@register
class QDPiDataset(BaseDataset):
    """
    QDπ (Quantum Deep Potential Interaction) Dataset for drug discovery.

    QDπ is designed for training universal machine learning potentials for
    drug-like molecules and biopolymer fragments. It contains ~1.6 million
    structures selected via query-by-committee active learning from diverse
    source datasets.

    CRITICAL DIFFERENCES FROM ANI-2x:
    - QDπ contains BOTH neutral AND charged molecules
    - Charge must be tracked from preprocessing (not stored in HDF5)
    - 13 elements vs ANI-2x's 7 elements
    - Higher accuracy DFT: ωB97M-D3(BJ)/def2-TZVPPD vs ωB97X/6-31G(d)

    Molecule Creation Strategy: 'coordinate_based'
    - No parseable chemical identifiers (InChI/SMILES) available
    - Molecular connectivity inferred from 3D coordinates via rdDetermineBonds
    - REQUIRES correct molecular_charge for accurate bond order determination
    """

    metadata: ClassVar[DatasetMetadata] = DatasetMetadata(
        name="QDPi",
        version="1.0.0",
        description=(
            "Quantum Deep Potential Interaction dataset for drug discovery. "
            "Contains ~1.6 million structures of drug-like molecules and biopolymer "
            "fragments at ωB97M-D3(BJ)/def2-TZVPPD level. Includes both neutral and "
            "charged molecules for comprehensive protonation state coverage."
        ),
        author="Zeng, Giese, Götz, York (Rutgers LBSR)",
        license="CC BY 4.0",
    )

    schema: ClassVar[DatasetSchema] = DatasetSchema(
        # Required properties for QDπ
        required_properties=(
            "atoms",  # Atomic numbers (from atomic_types + elements)
            "coordinates",  # Cartesian coordinates in Angstrom
            "energy",  # Total energy in Hartree (converted from eV)
        ),
        # Optional properties
        optional_properties=(
            "forces",  # Atomic forces in Hartree/Angstrom (converted from eV/Angstrom)
            "formula",  # Chemical formula group identifier from HDF5
            "molecular_charge",  # CRITICAL: Molecular charge for charged molecules
            "subset",  # Source subset identifier (spice, ani, geom, etc.)
            "charge_type",  # 'neutral' or 'charged' (from directory structure)
        ),
        # QDπ has NO parseable identifiers - uses coordinate_based strategy
        identifier_keys=(),
        # Coordinates are in Angstrom
        coordinate_units="angstrom",
        # Energies are in Hartree (converted from eV during preprocessing)
        energy_units="hartree",
    )

    features: ClassVar[DatasetFeatures] = DatasetFeatures(
        vibrational_analysis=False,  # No frequency data
        uncertainty_handling=False,  # DFT is deterministic
        atomization_energy=True,  # Can calculate from total energy
        rotational_constants=False,  # Not available
        frequency_analysis=False,  # Not available
        orbital_analysis=False,  # Not available
        homo_lumo_gap=False,  # Not available
        mo_energies=False,  # Not available
    )

    # Config key for YAML configuration
    config_key: ClassVar[str] = "qdpi_config"

    # NOTE: handler_class is intentionally NOT set here.
    # QDPiDatasetHandler is registered via @register_handler decorator and
    # discovered dynamically through the HandlerRegistry by create_dataset_handler().
    # Setting handler_class = None (default from BaseDataset) is correct.
    # The factory pattern handles handler instantiation via registry lookup.
    # We override create_handler() to use lazy import to avoid circular dependency.

    @classmethod
    def create_handler(
        cls, dataset_config, filter_config, processing_config, logger, experimental_setup=None
    ):
        """
        Factory method to create QDPiDatasetHandler instance.

        Uses lazy import to avoid circular dependency between
        datasets/implementations/qdpi.py and handlers/implementations/qdpi.py.

        This pattern breaks the circular import chain:
            datasets/implementations/qdpi.py
                → handlers/dataset_handlers.py (module-level)
                    → config containers → dataset registry → qdpi.py (CYCLE!)

        By importing inside the method, the import only happens at runtime
        when create_handler() is called, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.qdpi import QDPiDatasetHandler

        return QDPiDatasetHandler(
            dataset_config, filter_config, processing_config, logger, experimental_setup
        )

    @classmethod
    def get_required_properties(cls) -> list[str]:
        """Return list of required properties for QDπ dataset."""
        return list(cls.schema.required_properties)

    @classmethod
    def get_feature_support(cls) -> dict[str, bool]:
        """Return feature support dictionary."""
        return cls.features.to_dict()

    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        """
        QDπ uses coordinate_based strategy.

        CRITICAL: QDπ HDF5 structure contains NO parseable chemical identifiers.
        The data structure only contains:
        - 'elements': Element symbols
        - 'atomic_types': Integer indices into elements
        - 'coordinates': 3D positions
        - 'energies': DFT energies
        - 'forces': Atomic forces (optional)

        Molecular connectivity must be inferred from 3D coordinates using
        the rdDetermineBonds algorithm.

        IMPORTANT: Unlike ANI-2x (neutral-only), QDπ contains charged molecules.
        The molecular_charge parameter is CRITICAL for correct bond order
        determination in charged species.

        Returns:
            str: 'coordinate_based'
        """
        return "coordinate_based"

    @classmethod
    def get_supported_elements(cls) -> list[int]:
        """
        Return atomic numbers of elements supported in QDπ.

        QDπ supports 13 elements:
        H(1), Li(3), C(6), N(7), O(8), F(9), Na(11), P(15), S(16), Cl(17), K(19), Br(35), I(53)

        This is broader than ANI-2x (7 elements) to cover drug-like molecules.
        """
        return [1, 3, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53]

    @classmethod
    def get_supported_element_symbols(cls) -> list[str]:
        """Return element symbols supported in QDπ."""
        return ["H", "Li", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I"]

    @classmethod
    def supports_charged_molecules(cls) -> bool:
        """
        QDπ supports both neutral AND charged molecules.

        This is a CRITICAL difference from ANI-2x (neutral-only).
        The charged subset includes:
        - Ion pairs (Li+, Na+, K+, Cl-, Br-, I-)
        - Protonated amino acids (e.g., LYS with +1 charge)
        - Deprotonated species (e.g., ASP, GLU with -1 charge)
        - Tautomers and protonation states from RE dataset

        Returns:
            bool: True (QDπ supports charged molecules)
        """
        return True

    @classmethod
    def get_source_subsets(cls) -> list[str]:
        """
        Return list of source dataset subsets included in QDπ.

        Each subset was selected via query-by-committee active learning
        to maximize chemical diversity while minimizing redundancy.
        """
        return [
            "spice",  # SPICE dataset (dipeptides, amino acids, ion pairs, PubChem)
            "ani",  # ANI-1x/2x datasets (neutral organics)
            "geom",  # GEOM AICures subset (drug-like molecules)
            "freesolvmd",  # FreeSolv with MD sampling
            "re",  # Relative energy benchmarks
            "remd",  # RE with MD sampling
            "comp6",  # COMP6 validation benchmarks
        ]
