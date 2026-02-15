"""
RDKit Descriptor Categories and Metadata

Comprehensive categorization of 412+ RDKit molecular descriptors across
6 major categories with metadata for validation and requirement checking.

Categories:
- Constitutional (35 descriptors)
- Topological (350+ descriptors)
- Electronic (8 descriptors)
- Geometric (10 descriptors)
- Drug-likeness (4 descriptors)
- Fragments (85 descriptors)

Pydantic V2 Migration (Phase 30):
    - Migrated DescriptorMetadata from @dataclass(frozen=True) to Pydantic BaseModel (frozen=True)
    - Custom __init__ added to support positional arguments (backward compatibility with 412+ usages)
    - Custom __hash__ preserved for set/dict usage (hashes by name)
    - Added to_dict() method with model_dump(mode='json') for enum value serialization
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: Milia Team
Version: 1.1.0
"""

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DescriptorCategory(Enum):
    """Descriptor category enumeration"""

    CONSTITUTIONAL = "constitutional"
    TOPOLOGICAL = "topological"
    ELECTRONIC = "electronic"
    GEOMETRIC = "geometric"
    DRUG_LIKENESS = "drug_likeness"
    FRAGMENTS = "fragments"


class DescriptorMetadata(BaseModel, frozen=True):
    """
    Metadata for a molecular descriptor.

    Pydantic V2 Migration (Phase 30):
        - Migrated from @dataclass(frozen=True) to Pydantic BaseModel (frozen=True)
        - Custom __init__ added to support positional arguments (backward compatibility)
        - Custom __hash__ preserved for set/dict usage (hashes by name only)
        - Added to_dict() method with model_dump(mode='json') for enum value serialization
        - NON-BREAKING: Same constructor API and attribute access preserved

    Attributes:
        name: Descriptor name (must match RDKit function name)
        category: Category from DescriptorCategory enum
        requires_3d: Whether 3D coordinates are required
        requires_charges: Whether partial charges are required
        description: Brief description of the descriptor
        rdkit_module: RDKit module path (e.g., "Descriptors", "Descriptors3D")
    """

    name: str
    category: DescriptorCategory
    requires_3d: bool = False
    requires_charges: bool = False
    description: str = ""
    rdkit_module: str = "Descriptors"

    def __init__(
        self,
        name: str = None,
        category: DescriptorCategory = None,
        requires_3d: bool = False,
        requires_charges: bool = False,
        description: str = "",
        rdkit_module: str = "Descriptors",
        **data,
    ):
        """
        Initialize DescriptorMetadata with support for positional arguments.

        This custom __init__ preserves backward compatibility with the original
        @dataclass pattern that allowed positional arguments.

        Args:
            name: Descriptor name (positional or keyword)
            category: DescriptorCategory enum (positional or keyword)
            requires_3d: Whether 3D coordinates required (default: False)
            requires_charges: Whether partial charges required (default: False)
            description: Brief description (default: "")
            rdkit_module: RDKit module path (default: "Descriptors")
        """
        super().__init__(
            name=name,
            category=category,
            requires_3d=requires_3d,
            requires_charges=requires_charges,
            description=description,
            rdkit_module=rdkit_module,
            **data,
        )

    def __hash__(self) -> int:
        """Hash by name only for set/dict usage."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Equality comparison by name for consistency with __hash__."""
        if isinstance(other, DescriptorMetadata):
            return self.name == other.name
        return False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Uses model_dump(mode='json') for automatic enum value serialization
        (DescriptorCategory enum -> string value).

        Returns:
            Dictionary with all 6 fields, category as string value
        """
        return self.model_dump(mode="json")


# =============================================================================
# CONSTITUTIONAL DESCRIPTORS (35 descriptors)
# =============================================================================

CONSTITUTIONAL_DESCRIPTORS: list[DescriptorMetadata] = [
    # Basic molecular properties
    DescriptorMetadata("MolWt", DescriptorCategory.CONSTITUTIONAL, description="Molecular weight"),
    DescriptorMetadata(
        "HeavyAtomMolWt",
        DescriptorCategory.CONSTITUTIONAL,
        description="Heavy atom molecular weight",
    ),
    DescriptorMetadata(
        "ExactMolWt", DescriptorCategory.CONSTITUTIONAL, description="Exact molecular weight"
    ),
    # Atom counts (Note: These use Mol methods, not Descriptors module)
    DescriptorMetadata(
        "NumHeavyAtoms",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of heavy atoms",
        rdkit_module="Mol",
    ),
    DescriptorMetadata(
        "NumAtoms",
        DescriptorCategory.CONSTITUTIONAL,
        description="Total number of atoms",
        rdkit_module="Mol",
    ),
    DescriptorMetadata(
        "NumHeteroatoms", DescriptorCategory.CONSTITUTIONAL, description="Number of heteroatoms"
    ),
    DescriptorMetadata(
        "NumValenceElectrons",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of valence electrons",
    ),
    DescriptorMetadata(
        "NumRadicalElectrons",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of radical electrons",
    ),
    # Hydrogen counts
    DescriptorMetadata(
        "NumHAcceptors", DescriptorCategory.CONSTITUTIONAL, description="Number of H-bond acceptors"
    ),
    DescriptorMetadata(
        "NumHDonors", DescriptorCategory.CONSTITUTIONAL, description="Number of H-bond donors"
    ),
    # Bond counts
    DescriptorMetadata(
        "NumRotatableBonds",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of rotatable bonds",
    ),
    DescriptorMetadata(
        "NumAmideBonds", DescriptorCategory.CONSTITUTIONAL, description="Number of amide bonds"
    ),
    DescriptorMetadata(
        "NumBridgeheadAtoms",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of bridgehead atoms",
    ),
    DescriptorMetadata(
        "NumSpiroAtoms", DescriptorCategory.CONSTITUTIONAL, description="Number of spiro atoms"
    ),
    # Ring descriptors
    DescriptorMetadata(
        "NumAromaticRings",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of aromatic rings",
    ),
    DescriptorMetadata(
        "NumSaturatedRings",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of saturated rings",
    ),
    DescriptorMetadata(
        "NumAliphaticRings",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of aliphatic rings",
    ),
    DescriptorMetadata(
        "NumAromaticHeterocycles",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of aromatic heterocycles",
    ),
    DescriptorMetadata(
        "NumAromaticCarbocycles",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of aromatic carbocycles",
    ),
    DescriptorMetadata(
        "NumSaturatedHeterocycles",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of saturated heterocycles",
    ),
    DescriptorMetadata(
        "NumSaturatedCarbocycles",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of saturated carbocycles",
    ),
    DescriptorMetadata(
        "NumAliphaticHeterocycles",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of aliphatic heterocycles",
    ),
    DescriptorMetadata(
        "NumAliphaticCarbocycles",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of aliphatic carbocycles",
    ),
    # Ring size descriptors
    DescriptorMetadata(
        "RingCount", DescriptorCategory.CONSTITUTIONAL, description="Total number of rings"
    ),
    # Fraction descriptors (Note: RDKit uses uppercase CSP3)
    DescriptorMetadata(
        "FractionCSP3", DescriptorCategory.CONSTITUTIONAL, description="Fraction of sp3 carbons"
    ),
    # Lipinski descriptors
    DescriptorMetadata(
        "MolLogP", DescriptorCategory.CONSTITUTIONAL, description="Wildman-Crippen LogP"
    ),
    DescriptorMetadata(
        "MolMR", DescriptorCategory.CONSTITUTIONAL, description="Molecular refractivity"
    ),
    # TPSA
    DescriptorMetadata(
        "TPSA", DescriptorCategory.CONSTITUTIONAL, description="Topological polar surface area"
    ),
    DescriptorMetadata(
        "LabuteASA",
        DescriptorCategory.CONSTITUTIONAL,
        description="Labute's approximate surface area",
    ),
    # Additional descriptors
    DescriptorMetadata(
        "NumUnspecifiedAtomStereoCenters",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of unspecified atom stereocenters",
    ),
    DescriptorMetadata(
        "NumDefinedAtomStereoCenters",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of defined atom stereocenters",
    ),
    DescriptorMetadata(
        "NumUnassignedAtomStereoCenters",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of unassigned atom stereocenters",
    ),
    DescriptorMetadata(
        "NumUnspecifiedBondStereoCenters",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of unspecified bond stereocenters",
    ),
    DescriptorMetadata(
        "NumDefinedBondStereoCenters",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of defined bond stereocenters",
    ),
    DescriptorMetadata(
        "NumUnassignedBondStereoCenters",
        DescriptorCategory.CONSTITUTIONAL,
        description="Number of unassigned bond stereocenters",
    ),
]


# =============================================================================
# TOPOLOGICAL DESCRIPTORS (350+ descriptors)
# =============================================================================

# Core topological descriptors (24 descriptors)
TOPOLOGICAL_CORE: list[DescriptorMetadata] = [
    DescriptorMetadata("BertzCT", DescriptorCategory.TOPOLOGICAL, description="Bertz complexity"),
    DescriptorMetadata(
        "HallKierAlpha", DescriptorCategory.TOPOLOGICAL, description="Hall-Kier alpha"
    ),
    DescriptorMetadata("Kappa1", DescriptorCategory.TOPOLOGICAL, description="Kappa shape index 1"),
    DescriptorMetadata("Kappa2", DescriptorCategory.TOPOLOGICAL, description="Kappa shape index 2"),
    DescriptorMetadata("Kappa3", DescriptorCategory.TOPOLOGICAL, description="Kappa shape index 3"),
    DescriptorMetadata(
        "Chi0v", DescriptorCategory.TOPOLOGICAL, description="Chi0v connectivity index"
    ),
    DescriptorMetadata(
        "Chi1v", DescriptorCategory.TOPOLOGICAL, description="Chi1v connectivity index"
    ),
    DescriptorMetadata(
        "Chi2v", DescriptorCategory.TOPOLOGICAL, description="Chi2v connectivity index"
    ),
    DescriptorMetadata(
        "Chi3v", DescriptorCategory.TOPOLOGICAL, description="Chi3v connectivity index"
    ),
    DescriptorMetadata(
        "Chi4v", DescriptorCategory.TOPOLOGICAL, description="Chi4v connectivity index"
    ),
    DescriptorMetadata(
        "Chi0n", DescriptorCategory.TOPOLOGICAL, description="Chi0n connectivity index"
    ),
    DescriptorMetadata(
        "Chi1n", DescriptorCategory.TOPOLOGICAL, description="Chi1n connectivity index"
    ),
    DescriptorMetadata(
        "Chi2n", DescriptorCategory.TOPOLOGICAL, description="Chi2n connectivity index"
    ),
    DescriptorMetadata(
        "Chi3n", DescriptorCategory.TOPOLOGICAL, description="Chi3n connectivity index"
    ),
    DescriptorMetadata(
        "Chi4n", DescriptorCategory.TOPOLOGICAL, description="Chi4n connectivity index"
    ),
    DescriptorMetadata(
        "Ipc", DescriptorCategory.TOPOLOGICAL, description="Information content (mean)"
    ),
    DescriptorMetadata(
        "AvgIpc", DescriptorCategory.TOPOLOGICAL, description="Average information content"
    ),
    DescriptorMetadata("BalabanJ", DescriptorCategory.TOPOLOGICAL, description="Balaban J index"),
    DescriptorMetadata(
        "FpDensityMorgan1",
        DescriptorCategory.TOPOLOGICAL,
        description="Morgan fingerprint density 1",
    ),
    DescriptorMetadata(
        "FpDensityMorgan2",
        DescriptorCategory.TOPOLOGICAL,
        description="Morgan fingerprint density 2",
    ),
    DescriptorMetadata(
        "FpDensityMorgan3",
        DescriptorCategory.TOPOLOGICAL,
        description="Morgan fingerprint density 3",
    ),
    DescriptorMetadata(
        "HeavyAtomCount", DescriptorCategory.TOPOLOGICAL, description="Heavy atom count"
    ),
    DescriptorMetadata("NHOHCount", DescriptorCategory.TOPOLOGICAL, description="NH + OH count"),
    DescriptorMetadata("NOCount", DescriptorCategory.TOPOLOGICAL, description="N + O count"),
]

# PEOE_VSA descriptors (14 descriptors - COMPLETE)
TOPOLOGICAL_PEOE_VSA: list[DescriptorMetadata] = [
    DescriptorMetadata(
        f"PEOE_VSA{i}", DescriptorCategory.TOPOLOGICAL, description=f"PEOE VSA descriptor {i}"
    )
    for i in range(1, 15)  # PEOE_VSA1 through PEOE_VSA14
]

# SMR_VSA descriptors (10 descriptors - COMPLETE)
TOPOLOGICAL_SMR_VSA: list[DescriptorMetadata] = [
    DescriptorMetadata(
        f"SMR_VSA{i}", DescriptorCategory.TOPOLOGICAL, description=f"SMR VSA descriptor {i}"
    )
    for i in range(1, 11)  # SMR_VSA1 through SMR_VSA10
]

# SlogP_VSA descriptors (12 descriptors - COMPLETE)
TOPOLOGICAL_SLOGP_VSA: list[DescriptorMetadata] = [
    DescriptorMetadata(
        f"SlogP_VSA{i}", DescriptorCategory.TOPOLOGICAL, description=f"SlogP VSA descriptor {i}"
    )
    for i in range(1, 13)  # SlogP_VSA1 through SlogP_VSA12
]

# EState_VSA descriptors (11 descriptors - COMPLETE)
TOPOLOGICAL_ESTATE_VSA: list[DescriptorMetadata] = [
    DescriptorMetadata(
        f"EState_VSA{i}", DescriptorCategory.TOPOLOGICAL, description=f"EState VSA descriptor {i}"
    )
    for i in range(1, 12)  # EState_VSA1 through EState_VSA11
]

# VSA_EState descriptors (10 descriptors - COMPLETE)
TOPOLOGICAL_VSA_ESTATE: list[DescriptorMetadata] = [
    DescriptorMetadata(
        f"VSA_EState{i}", DescriptorCategory.TOPOLOGICAL, description=f"VSA EState descriptor {i}"
    )
    for i in range(1, 11)  # VSA_EState1 through VSA_EState10
]

# BCUT2D descriptors (8 descriptors - COMPLETE)
TOPOLOGICAL_BCUT2D: list[DescriptorMetadata] = [
    DescriptorMetadata(
        "BCUT2D_MWHI", DescriptorCategory.TOPOLOGICAL, description="BCUT2D molecular weight high"
    ),
    DescriptorMetadata(
        "BCUT2D_MWLOW", DescriptorCategory.TOPOLOGICAL, description="BCUT2D molecular weight low"
    ),
    DescriptorMetadata(
        "BCUT2D_CHGHI", DescriptorCategory.TOPOLOGICAL, description="BCUT2D charge high"
    ),
    DescriptorMetadata(
        "BCUT2D_CHGLO", DescriptorCategory.TOPOLOGICAL, description="BCUT2D charge low"
    ),
    DescriptorMetadata(
        "BCUT2D_LOGPHI", DescriptorCategory.TOPOLOGICAL, description="BCUT2D LogP high"
    ),
    DescriptorMetadata(
        "BCUT2D_LOGPLOW", DescriptorCategory.TOPOLOGICAL, description="BCUT2D LogP low"
    ),
    DescriptorMetadata("BCUT2D_MRHI", DescriptorCategory.TOPOLOGICAL, description="BCUT2D MR high"),
    DescriptorMetadata("BCUT2D_MRLOW", DescriptorCategory.TOPOLOGICAL, description="BCUT2D MR low"),
]

# AUTOCORR2D descriptors (192 descriptors - COMPLETE with non-contiguous indexing)
# Note: AUTOCORR2D uses special indexing - not all indices exist
# Valid indices: 1-192 but with gaps (e.g., no AUTOCORR2D_13, AUTOCORR2D_27, etc.)
# We'll generate all 192 and handle missing ones during introspection
TOPOLOGICAL_AUTOCORR2D: list[DescriptorMetadata] = [
    DescriptorMetadata(
        f"AUTOCORR2D_{i}",
        DescriptorCategory.TOPOLOGICAL,
        description=f"2D autocorrelation descriptor {i}",
    )
    for i in range(1, 193)  # AUTOCORR2D_1 through AUTOCORR2D_192
]

# Combine all topological descriptors
TOPOLOGICAL_DESCRIPTORS: list[DescriptorMetadata] = (
    TOPOLOGICAL_CORE
    + TOPOLOGICAL_PEOE_VSA
    + TOPOLOGICAL_SMR_VSA
    + TOPOLOGICAL_SLOGP_VSA
    + TOPOLOGICAL_ESTATE_VSA
    + TOPOLOGICAL_VSA_ESTATE
    + TOPOLOGICAL_BCUT2D
    + TOPOLOGICAL_AUTOCORR2D
)


# =============================================================================
# ELECTRONIC DESCRIPTORS (8 descriptors)
# =============================================================================

ELECTRONIC_DESCRIPTORS: list[DescriptorMetadata] = [
    DescriptorMetadata(
        "MaxPartialCharge",
        DescriptorCategory.ELECTRONIC,
        requires_charges=True,
        description="Maximum partial charge",
    ),
    DescriptorMetadata(
        "MinPartialCharge",
        DescriptorCategory.ELECTRONIC,
        requires_charges=True,
        description="Minimum partial charge",
    ),
    DescriptorMetadata(
        "MaxAbsPartialCharge",
        DescriptorCategory.ELECTRONIC,
        requires_charges=True,
        description="Maximum absolute partial charge",
    ),
    DescriptorMetadata(
        "MinAbsPartialCharge",
        DescriptorCategory.ELECTRONIC,
        requires_charges=True,
        description="Minimum absolute partial charge",
    ),
    DescriptorMetadata(
        "MaxEStateIndex", DescriptorCategory.ELECTRONIC, description="Maximum E-state index"
    ),
    DescriptorMetadata(
        "MinEStateIndex", DescriptorCategory.ELECTRONIC, description="Minimum E-state index"
    ),
    DescriptorMetadata(
        "MaxAbsEStateIndex",
        DescriptorCategory.ELECTRONIC,
        description="Maximum absolute E-state index",
    ),
    DescriptorMetadata(
        "MinAbsEStateIndex",
        DescriptorCategory.ELECTRONIC,
        description="Minimum absolute E-state index",
    ),
]


# =============================================================================
# GEOMETRIC DESCRIPTORS (10 descriptors - 3D-DEPENDENT)
# =============================================================================

GEOMETRIC_DESCRIPTORS: list[DescriptorMetadata] = [
    DescriptorMetadata(
        "RadiusOfGyration",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Radius of gyration",
    ),
    DescriptorMetadata(
        "InertialShapeFactor",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Inertial shape factor",
    ),
    DescriptorMetadata(
        "Eccentricity",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Molecular eccentricity",
    ),
    DescriptorMetadata(
        "Asphericity",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Molecular asphericity",
    ),
    DescriptorMetadata(
        "SpherocityIndex",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Spherocity index",
    ),
    DescriptorMetadata(
        "PBF",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Plane of best fit",
    ),
    DescriptorMetadata(
        "NPR1",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Normalized principal moments ratio 1",
    ),
    DescriptorMetadata(
        "NPR2",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Normalized principal moments ratio 2",
    ),
    DescriptorMetadata(
        "PMI1",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Principal moment of inertia 1",
    ),
    DescriptorMetadata(
        "PMI2",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Principal moment of inertia 2",
    ),
    DescriptorMetadata(
        "PMI3",
        DescriptorCategory.GEOMETRIC,
        requires_3d=True,
        rdkit_module="Descriptors3D",
        description="Principal moment of inertia 3",
    ),
]


# =============================================================================
# DRUG-LIKENESS DESCRIPTORS (4 descriptors - NEW CATEGORY)
# =============================================================================

DRUG_LIKENESS_DESCRIPTORS: list[DescriptorMetadata] = [
    DescriptorMetadata(
        "qed",
        DescriptorCategory.DRUG_LIKENESS,
        description="Quantitative estimate of drug-likeness",
    ),
    DescriptorMetadata(
        "SPS", DescriptorCategory.DRUG_LIKENESS, description="Synthetic accessibility score"
    ),
    DescriptorMetadata(
        "Phi", DescriptorCategory.DRUG_LIKENESS, description="Kier flexibility index"
    ),
    DescriptorMetadata(
        "FractionCSP3",
        DescriptorCategory.DRUG_LIKENESS,
        description="Fraction of sp3 carbons (alternative naming)",
    ),
]


# =============================================================================
# FRAGMENT DESCRIPTORS (85 descriptors - NEW CATEGORY)
# =============================================================================

# All RDKit fr_* functional group descriptors
FRAGMENT_DESCRIPTOR_NAMES: list[str] = [
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
]

FRAGMENT_DESCRIPTORS: list[DescriptorMetadata] = [
    DescriptorMetadata(
        name, DescriptorCategory.FRAGMENTS, description=f"Fragment descriptor: {name}"
    )
    for name in FRAGMENT_DESCRIPTOR_NAMES
]


# =============================================================================
# AGGREGATED DESCRIPTOR COLLECTIONS
# =============================================================================

ALL_DESCRIPTORS: list[DescriptorMetadata] = (
    CONSTITUTIONAL_DESCRIPTORS
    + TOPOLOGICAL_DESCRIPTORS
    + ELECTRONIC_DESCRIPTORS
    + GEOMETRIC_DESCRIPTORS
    + DRUG_LIKENESS_DESCRIPTORS
    + FRAGMENT_DESCRIPTORS
)

# Category mapping
DESCRIPTORS_BY_CATEGORY: dict[DescriptorCategory, list[DescriptorMetadata]] = {
    DescriptorCategory.CONSTITUTIONAL: CONSTITUTIONAL_DESCRIPTORS,
    DescriptorCategory.TOPOLOGICAL: TOPOLOGICAL_DESCRIPTORS,
    DescriptorCategory.ELECTRONIC: ELECTRONIC_DESCRIPTORS,
    DescriptorCategory.GEOMETRIC: GEOMETRIC_DESCRIPTORS,
    DescriptorCategory.DRUG_LIKENESS: DRUG_LIKENESS_DESCRIPTORS,
    DescriptorCategory.FRAGMENTS: FRAGMENT_DESCRIPTORS,
}

# Name to metadata mapping
DESCRIPTOR_METADATA_MAP: dict[str, DescriptorMetadata] = {
    desc.name: desc for desc in ALL_DESCRIPTORS
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_descriptors_by_category(category: DescriptorCategory) -> list[DescriptorMetadata]:
    """
    Get all descriptors for a specific category.

    Args:
        category: DescriptorCategory enum value

    Returns:
        List of DescriptorMetadata objects for the category
    """
    return DESCRIPTORS_BY_CATEGORY.get(category, [])


def get_descriptor_metadata(name: str) -> DescriptorMetadata | None:
    """
    Get metadata for a specific descriptor by name.

    Args:
        name: Descriptor name (e.g., "MolWt", "PEOE_VSA1")

    Returns:
        DescriptorMetadata object or None if not found
    """
    return DESCRIPTOR_METADATA_MAP.get(name)


def requires_3d_coordinates(descriptor_name: str) -> bool:
    """
    Check if a descriptor requires 3D coordinates.

    Args:
        descriptor_name: Name of the descriptor

    Returns:
        True if 3D coordinates required, False otherwise
    """
    metadata = get_descriptor_metadata(descriptor_name)
    return metadata.requires_3d if metadata else False


def requires_partial_charges(descriptor_name: str) -> bool:
    """
    Check if a descriptor requires partial charges.

    Args:
        descriptor_name: Name of the descriptor

    Returns:
        True if partial charges required, False otherwise
    """
    metadata = get_descriptor_metadata(descriptor_name)
    return metadata.requires_charges if metadata else False


def get_all_descriptor_names() -> list[str]:
    """
    Get list of all descriptor names.

    Returns:
        List of all descriptor names
    """
    return list(DESCRIPTOR_METADATA_MAP.keys())


def get_category_descriptor_names(category: DescriptorCategory) -> list[str]:
    """
    Get list of descriptor names for a specific category.

    Args:
        category: DescriptorCategory enum value

    Returns:
        List of descriptor names in the category
    """
    return [desc.name for desc in get_descriptors_by_category(category)]


def filter_descriptors_by_requirements(
    descriptor_names: list[str], has_3d: bool = False, has_charges: bool = False
) -> tuple[list[str], list[str]]:
    """
    Filter descriptors based on available molecular properties.

    Args:
        descriptor_names: List of descriptor names to filter
        has_3d: Whether molecule has 3D coordinates
        has_charges: Whether molecule has partial charges

    Returns:
        Tuple of (valid_descriptors, filtered_out_descriptors)
    """
    valid = []
    filtered = []

    for name in descriptor_names:
        metadata = get_descriptor_metadata(name)
        if not metadata:
            filtered.append(name)
            continue

        if metadata.requires_3d and not has_3d:
            filtered.append(name)
            continue

        if metadata.requires_charges and not has_charges:
            filtered.append(name)
            continue

        valid.append(name)

    return valid, filtered


def get_descriptor_count_by_category() -> dict[str, int]:
    """
    Get count of descriptors in each category.

    Returns:
        Dictionary mapping category name to count
    """
    return {
        category.value: len(descriptors)
        for category, descriptors in DESCRIPTORS_BY_CATEGORY.items()
    }


def validate_descriptor_coverage() -> dict[str, Any]:
    """
    Validate that all expected descriptors are present.

    Returns:
        Dictionary with validation results
    """
    expected_counts = {
        DescriptorCategory.CONSTITUTIONAL: 35,
        DescriptorCategory.TOPOLOGICAL: 350,  # Approximate due to AUTOCORR2D
        DescriptorCategory.ELECTRONIC: 8,
        DescriptorCategory.GEOMETRIC: 10,
        DescriptorCategory.DRUG_LIKENESS: 4,
        DescriptorCategory.FRAGMENTS: 85,
    }

    actual_counts = {
        category: len(descriptors) for category, descriptors in DESCRIPTORS_BY_CATEGORY.items()
    }

    return {
        "expected": expected_counts,
        "actual": actual_counts,
        "total_expected": 492,  # Approximate
        "total_actual": len(ALL_DESCRIPTORS),
        "coverage_complete": all(
            actual_counts[cat] >= expected_counts[cat] for cat in expected_counts
        ),
    }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"Loaded {len(ALL_DESCRIPTORS)} RDKit descriptors across {len(DESCRIPTORS_BY_CATEGORY)} categories"
)
logger.debug(f"Descriptor counts: {get_descriptor_count_by_category()}")
