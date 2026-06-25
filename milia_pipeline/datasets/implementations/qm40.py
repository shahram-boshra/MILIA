# milia_pipeline/datasets/implementations/qm40.py

"""
QM40 Dataset Implementation
===========================

Dataset class for the QM40 quantum chemistry dataset.

QM40 Dataset Information:
-------------------------
- Reference: Madushanka, Moura Jr. & Kraka. Sci Data 11, 1376 (2024).
- DOI: 10.1038/s41597-024-04206-y
- Figshare DOI: 10.6084/m9.figshare.25993060
- Download URL: https://ndownloader.figshare.com/files/47535647 (QM40_dataset.zip)
- Contents: 162,954 drug-like ZINC molecules (10-40 heavy atoms).
- Method: B3LYP/6-31G(2df,p) (Gaussian16) — identical level of theory to QM9.
- Elements: C, O, N, S, F, Cl (+ H) — 7 elements.
- Charge state: NEUTRAL ONLY. The dataset explicitly excludes anions and
  cations; every structure is a charge-neutral singlet ground state.

Provided per molecule (joined from three CSV files by Zinc_id during
preprocessing — see preprocessing/utils/qm40_csv_parser.py):
- Optimized Cartesian geometry + element symbols (QM40_xyz.csv)
- Per-atom Mulliken charges (QM40_xyz.csv)
- 16 scalar QM properties (QM40_main.csv)
- Per-bond local vibrational mode force constants (QM40_bond.csv)

Molecule Creation Strategy: 'coordinate_based'
----------------------------------------------
QM40 ships SMILES strings but NO InChI. MILIA's identifier scheme is
InChI-first precisely because SMILES-first parsing is unreliable (implicit
hydrogens / atom-count mismatch against per-atom coordinate arrays). Because
QM40 provides full optimized coordinates with explicit per-atom symbols and
all molecules are neutral closed-shell singlets, molecular connectivity is
inferred directly from the 3D geometry via rdDetermineBonds (the same strategy
as Wavefunction / ANI-1x / QDπ). SMILES is retained only as a tracking label,
never for graph construction.
"""

from typing import ClassVar

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetFeatures,
    DatasetMetadata,
    DatasetSchema,
)
from milia_pipeline.datasets.registry import register

# NOTE: QM40DatasetHandler is intentionally NOT imported at module level to
# avoid a circular import. The handler is registered via @register_handler and
# discovered through the HandlerRegistry; create_handler() uses a lazy import
# to instantiate it at runtime. This follows the pattern established by QM9/QDπ.


@register
class QM40Dataset(BaseDataset):
    """
    QM40 quantum chemistry dataset implementation.

    QM40 contains 162,954 neutral drug-like organic molecules (C, O, N, S, F,
    Cl + H; 10-40 heavy atoms) computed at B3LYP/6-31G(2df,p). It provides
    optimized geometries, Mulliken charges, 16 scalar quantum-mechanical
    properties, and per-bond local vibrational mode force constants.

    Properties (QM40_main.csv, sanitized NPZ keys):
    -----------------------------------------------
    - Internal_E_0K: Internal energy at 0 K (Hartree) — primary energy target
    - HOMO, LUMO: Orbital energies (Hartree)
    - HL_gap: HOMO - LUMO (Hartree; may be negative)
    - Polarizability: Isotropic polarizability (Bohr^3)
    - spatial_extent: Electronic spatial extent (Bohr^2)
    - dipol_mom: Dipole moment (Debye)
    - ZPE: Zero point energy (kcal/mol)
    - rot1, rot2, rot3: Rotational constants (GHz)
    - Inter_E_298, Enthalpy, Free_E: Thermodynamic energies at 298.15 K (Hartree)
    - CV, Entropy: Heat capacity / entropy (cal/mol*K)
    - Qmulliken: Per-atom Mulliken partial charges

    CONTRAST WITH QDπ: QM40 is neutral-only, so no molecular_charge tracking is
    needed (supports_charged_molecules() returns False).
    """

    metadata: ClassVar[DatasetMetadata] = DatasetMetadata(
        name="QM40",
        version="1.0.0",
        description=(
            "QM40 quantum chemistry dataset: 162,954 neutral drug-like ZINC molecules "
            "(C, O, N, S, F, Cl + H; 10-40 heavy atoms) at B3LYP/6-31G(2df,p). Provides "
            "optimized geometries, Mulliken charges, 16 scalar QM properties, and per-bond "
            "local vibrational mode force constants."
        ),
        author="Madushanka, Moura Jr., Kraka (SMU CATCO)",
        license="CC BY 4.0",
    )

    schema: ClassVar[DatasetSchema] = DatasetSchema(
        # Required for every molecule. Internal_E_0K is the primary energy target
        # (mirrors QM9's use of U0 as the leading required property).
        required_properties=(
            "Internal_E_0K",
            "atoms",
            "coordinates",
        ),
        optional_properties=(
            # Scalar QM properties (QM40_main.csv)
            "HOMO",
            "LUMO",
            "HL_gap",  # HOMO - LUMO (Hartree; may be negative)
            "Polarizability",  # Bohr^3
            "spatial_extent",  # Bohr^2 (CSV header is "spatial extent")
            "dipol_mom",  # Debye
            "ZPE",  # kcal/mol
            "rot1",
            "rot2",
            "rot3",  # GHz
            "Inter_E_298",  # Hartree (298.15 K)
            "Enthalpy",  # Hartree (298.15 K)
            "Free_E",  # Hartree (298.15 K)
            "CV",  # cal/(mol*K)
            "Entropy",  # cal/(mol*K)
            # Per-atom node feature
            "Qmulliken",  # Mulliken partial charges (per atom)
            # Tracking label (coordinate_based: NOT used for graph construction)
            "smiles",
            # Pre-optimization geometry
            "initial_coordinates",
            # Per-bond local vibrational mode data (QM40-specific)
            "bond_atom1_idx",  # 0-based atom index
            "bond_atom2_idx",  # 0-based atom index
            "bond_tag",  # bond element pair, e.g. "CC"
            "bond_lmod_ka",  # local mode force constant (mDyn/Angstrom)
        ),
        # QM40 ships SMILES but NO InChI; the dataset uses coordinate_based
        # (see get_molecule_creation_strategy), so no identifier keys are declared.
        identifier_keys=(),
        # Optimized Cartesian coordinates are in Angstrom.
        coordinate_units="angstrom",
        # Energies are in Hartree (standard QC output).
        energy_units="hartree",
    )

    features: ClassVar[DatasetFeatures] = DatasetFeatures(
        # Only scalar ZPE + per-bond local modes are released, not per-mode
        # frequency arrays, so vibrational/frequency analysis is not advertised.
        vibrational_analysis=False,
        # Deterministic DFT — no statistical uncertainties.
        uncertainty_handling=False,
        # Atomization energy computable from Internal_E_0K; the global
        # atomic_energies_hartree table covers all QM40 elements (H, C, N, O, F, Cl, S).
        atomization_energy=True,
        # Rotational constants rot1/rot2/rot3 are present.
        rotational_constants=True,
        # No frequency arrays released.
        frequency_analysis=False,
        # No MO coefficients/energies (only HOMO/LUMO scalars).
        orbital_analysis=False,
        # HOMO, LUMO and HL_gap are present.
        homo_lumo_gap=True,
        # Only HOMO/LUMO scalars, not the full MO spectrum.
        mo_energies=False,
    )

    # Configuration key matching configs/datasets/qm40.yaml
    config_key: ClassVar[str] = "qm40_config"

    # NOTE: handler_class is intentionally NOT set here (default None is correct).
    # We override create_handler() with a lazy import to avoid the circular
    # dependency datasets/implementations/qm40.py -> handlers -> registry -> qm40.py.

    @classmethod
    def create_handler(
        cls, dataset_config, filter_config, processing_config, logger, experimental_setup=None
    ):
        """
        Factory method to create a QM40DatasetHandler instance.

        Uses a lazy import to avoid the circular dependency between
        datasets/implementations/qm40.py and handlers/implementations/qm40.py.
        The import happens at call time, after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.qm40 import QM40DatasetHandler

        return QM40DatasetHandler(
            dataset_config, filter_config, processing_config, logger, experimental_setup
        )

    @classmethod
    def get_required_properties(cls) -> list[str]:
        """Return list of required properties for the QM40 dataset."""
        return list(cls.schema.required_properties)

    @classmethod
    def get_feature_support(cls) -> dict[str, bool]:
        """Return the feature support dictionary for QM40."""
        return cls.features.to_dict()

    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        """
        QM40 uses the 'coordinate_based' strategy.

        QM40 provides SMILES but NO InChI. Since SMILES-first parsing is
        unreliable (implicit hydrogens vs. explicit per-atom coordinate arrays)
        and QM40 supplies full optimized geometries for neutral closed-shell
        singlets, molecular connectivity is inferred from the 3D coordinates via
        rdDetermineBonds — identical to Wavefunction / ANI-1x / QDπ.

        Returns:
            str: 'coordinate_based'
        """
        return "coordinate_based"

    @classmethod
    def get_supported_elements(cls) -> list[int]:
        """
        Return atomic numbers of elements supported in QM40.

        QM40 supports 7 elements: H(1), C(6), N(7), O(8), F(9), S(16), Cl(17).
        """
        return [1, 6, 7, 8, 9, 16, 17]

    @classmethod
    def get_supported_element_symbols(cls) -> list[str]:
        """Return element symbols supported in QM40."""
        return ["H", "C", "N", "O", "F", "S", "Cl"]

    @classmethod
    def supports_charged_molecules(cls) -> bool:
        """
        QM40 is NEUTRAL-ONLY.

        The dataset explicitly excludes anions and cations; all structures are
        charge-neutral singlet ground states. This is the inverse of QDπ and
        means no molecular_charge tracking is required (the handler returns 0).

        Returns:
            bool: False
        """
        return False
