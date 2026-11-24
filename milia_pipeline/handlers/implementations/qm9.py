# milia_pipeline/handlers/implementations/qm9.py

"""
QM9 Dataset Handler
===================

Handler for QM9 quantum chemistry datasets with exception integration
and transformation system support.

Extracted from dataset_handlers.py as part of the Handler Module Refactoring.

Key Features:
- Uses identifier_coordinate_based strategy (SMILES/InChI identifiers)
- Coordinates in Angstrom (B3LYP-optimized)
- Energies in Hartree
- Primary energy target: U0 (internal energy at 0K)
- Has vibrational frequencies and Mulliken charges

QM9 is a quantum chemistry dataset containing 133,885 stable small organic molecules
(CHONF, up to 9 heavy atoms) with properties computed at B3LYP/6-31G(2df,p) level.

Reference: Ramakrishnan et al., Scientific Data 1, 140022 (2014)
"""

import logging
from typing import Any

import numpy as np

from milia_pipeline.config.validators import (
    is_value_valid_and_not_nan,
    validate_molecular_structure,
)
from milia_pipeline.exceptions import (
    DatasetSpecificHandlerError,
    HandlerError,
    HandlerValidationError,
    MoleculeProcessingError,
)

# Import from refactored base handler
from milia_pipeline.handlers.base_handler import DatasetHandler
from milia_pipeline.handlers.handler_registry import register_handler

logger = logging.getLogger(__name__)


@register_handler
class QM9DatasetHandler(DatasetHandler):
    """
    Handler for QM9 datasets with exception integration and transformation system support.

    QM9 is a quantum chemistry dataset containing 133,885 stable small organic molecules
    (CHONF, up to 9 heavy atoms) with properties computed at B3LYP/6-31G(2df,p) level.

    Key characteristics:
    - Uses identifier_coordinate_based strategy (SMILES/InChI identifiers)
    - Coordinates in Angstrom (B3LYP-optimized)
    - Energies in Hartree
    - Primary energy target: U0 (internal energy at 0K)
    - Has vibrational frequencies and Mulliken charges

    Reference: Ramakrishnan et al., Scientific Data 1, 140022 (2014)
    """

    def get_dataset_type(self) -> str:
        return "QM9"

    def validate_molecule_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str = "N/A"
    ) -> None:
        """Validate QM9-specific molecular data with exception handling."""
        try:
            # Validate essential QM9 properties
            essential_props = ["U0", "atoms", "coordinates"]
            missing_props = []

            for prop in essential_props:
                if not self._is_valid_property(raw_properties_dict.get(prop)):
                    missing_props.append(prop)

            if missing_props:
                raise HandlerValidationError(
                    message=f"Missing required QM9 properties: {missing_props}",
                    handler_type="QM9",
                    validation_type="essential_properties",
                    failed_validations=[f"Missing {prop}" for prop in missing_props],
                    molecule_index=molecule_index,
                    details="QM9 molecules must have U0 energy, atoms, and coordinates",
                )

            # Validate structural consistency
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(atoms, coordinates, molecule_index, identifier)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="QM9",
                        message=f"QM9 molecular structure validation failed for molecule {molecule_index}",
                        operation="structure_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"Identifier: {identifier}, Atoms: {len(atoms) if atoms else 0}, "
                        f"Coords: {len(coordinates) if coordinates else 0}, "
                        f"Error: {str(e)}",
                    ) from e

            # Validate energy ranges (QM9 energies typically negative in Hartree)
            u0 = raw_properties_dict.get("U0")
            if u0 is not None and isinstance(u0, (int, float, np.number)) and u0 > 0:
                self.logger.warning(f"QM9 molecule {molecule_index} has positive U0 energy: {u0}")

        except (HandlerError, DatasetSpecificHandlerError):
            # Re-raise handler-specific errors
            raise
        except MoleculeProcessingError as e:
            # Convert molecule processing errors to QM9 handler validation errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM9",
                message=f"QM9 validation failed for molecule {molecule_index}: {e.message}",
                operation="molecule_validation",
                molecule_index=molecule_index,
                identifier=identifier,
                details=f"Identifier: {identifier}, Underlying error: {str(e)}",
            ) from e
        except Exception as e:
            # Convert unexpected errors to QM9 handler errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM9",
                message=f"Unexpected error during QM9 validation: {str(e)}",
                operation="molecule_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def get_required_properties(self) -> list[str]:
        """Get QM9-specific required properties."""
        required = self.get_common_required_properties()
        required.extend(["U0", "atoms", "coordinates"])  # Core QM9 properties

        # Add properties from processing config
        if self.processing_config:
            required.extend(self.processing_config.scalar_graph_targets)
            required.extend(self.processing_config.node_features)
            required.extend(self.processing_config.vector_graph_properties)
            required.extend(self.processing_config.variable_len_graph_properties)

            # Add atomization energy base if configured
            if self.processing_config.calculate_atomization_energy_from:
                required.append(self.processing_config.calculate_atomization_energy_from)

        return list(set(required))

    def get_identifier_keys(self) -> list[tuple[str, str]]:
        """
        Get QM9 identifier keys for molecule creation.

        QM9 datasets provide both SMILES and InChI identifiers.
        InChI is tried first as it is MILIA's primary molecular scheme - InChI
        encodes complete hydrogen information and molecular connectivity, ensuring
        exact 3D structure preservation with QM-optimized coordinates.
        SMILES is used as fallback.

        Returns:
            List of (npz_key, identifier_type) tuples
        """
        return [("inchi", "inchi"), ("smiles", "smiles")]

    def get_molecular_charge(
        self,
        raw_properties_dict: dict[str, Any],
        atomic_numbers: np.ndarray,
        mol_identifier: str | None = None,
    ) -> int:
        """
        Extract charge from SMILES or InChI string.

        QM9 molecules are predominantly neutral. The dataset provides both
        SMILES and InChI identifiers. Try InChI first (has explicit /q layer),
        then fall back to SMILES parsing if needed.

        Args:
            raw_properties_dict: Raw molecule data from NPZ
            atomic_numbers: Array of atomic numbers (not used for QM9)
            mol_identifier: Molecular identifier (not used, get from raw data)

        Returns:
            int: Molecular charge (0 for most QM9 molecules)
        """
        # Try InChI first (has explicit charge layer)
        inchi = raw_properties_dict.get("inchi")
        if inchi:
            charge = self._extract_charge_from_inchi(inchi)
            if charge != 0:
                self.logger.debug(f"Extracted charge from InChI: {charge}")
            return charge

        # Try relaxed InChI
        inchi_relaxed = raw_properties_dict.get("inchi_relaxed")
        if inchi_relaxed:
            charge = self._extract_charge_from_inchi(inchi_relaxed)
            if charge != 0:
                self.logger.debug(f"Extracted charge from relaxed InChI: {charge}")
            return charge

        # QM9 molecules are predominantly neutral
        self.logger.debug("No InChI found, assuming neutral molecule (typical for QM9)")
        return 0

    def get_molecule_creation_strategy(self) -> str:
        """
        QM9 datasets use identifier_coordinate_based strategy.

        QM9 molecular data contains SMILES/InChI identifiers which encode molecular
        connectivity and bonding. These are parsed to create the molecular graph,
        then B3LYP-optimized coordinates are assigned to preserve exact 3D geometry.

        Data requirements:
            - SMILES or InChI identifier (for connectivity/bonds)
            - Atomic numbers (for atom types)
            - Coordinates in Angstrom (for 3D geometry)
            - Molecular charge (typically 0 for QM9)

        Returns:
            str: 'identifier_coordinate_based'
        """
        return "identifier_coordinate_based"

    def process_property_value(
        self, key: str, value: Any, molecule_index: int, identifier: str = "N/A"
    ) -> Any:
        """Process QM9-specific property values with exception handling."""
        try:
            # Handle vibrational frequencies
            if key == "freqs" and value is not None and not is_value_valid_and_not_nan(value):
                self.logger.warning(f"QM9 molecule {molecule_index} has invalid frequency data")
                return None

            # Handle Mulliken charges
            if key == "Qmulliken" and value is not None and not is_value_valid_and_not_nan(value):
                self.logger.warning(f"QM9 molecule {molecule_index} has invalid Mulliken charges")
                return None

            # Handle rotational constants (A, B, C)
            if (
                key in ["A", "B", "C"]
                and value is not None
                and not is_value_valid_and_not_nan(value)
            ):
                self.logger.warning(
                    f"QM9 molecule {molecule_index} has invalid rotational constant {key}"
                )
                return None

            return value

        except DatasetSpecificHandlerError:
            # Re-raise QM9 handler errors
            raise
        except Exception as e:
            # Convert unexpected property processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM9",
                message=f"Unexpected error processing QM9 property '{key}': {str(e)}",
                operation="property_processing",
                property_name=key,
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _is_valid_property(self, value: Any) -> bool:
        """Check if a property value is valid for QM9."""
        if value is None:
            return False
        if isinstance(value, str) and value.lower() in ["missing", "invalid", "", "nan"]:
            return False
        return is_value_valid_and_not_nan(value)

    def get_processing_statistics(
        self, processed_molecules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate QM9-specific processing statistics."""
        stats = {
            "dataset_type": "QM9",
            "total_processed": len(processed_molecules),
            "experimental_setup": self.experimental_setup,
        }

        atomization_calculations = 0

        for mol_data in processed_molecules:
            if mol_data.get("atomization_energy_calculated"):
                atomization_calculations += 1

        if atomization_calculations > 0:
            stats["atomization_energy_calculations"] = atomization_calculations

        # Add transform usage information if experimental setup exists
        if self.experimental_setup:
            stats["transform_aware_processing"] = True
            stats["experimental_context"] = {
                "setup_name": self.experimental_setup,
                "dataset_type": "QM9",
                "transform_validation_performed": True,
            }

        return stats

    def get_supported_structural_features(self) -> dict[str, list[str]]:
        """
        QM9 datasets support ALL structural features.

        QM9 has optimized 3D geometries (B3LYP/6-31G(2df,p)) and Mulliken charges,
        enabling all structural feature calculations.
        """
        return {
            "atom": [
                # Basic connectivity
                "degree",
                "total_degree",
                # Hybridization and bonding
                "hybridization",
                "total_valence",
                "is_aromatic",
                "is_in_ring",
                "num_aromatic_bonds",
                # Chirality
                "chirality",
                # Partial charges
                # atomic_charge is decided by configuration (atomic_charges_property), not here
                "gasteiger_charge",  # Can be calculated from SMILES
            ],
            "bond": [
                # Bond types
                "bond_type",
                "is_conjugated",
                "is_aromatic",
                "is_in_any_ring",
                "stereo",
                # Geometric features (QM9 has 3D coordinates)
                "bond_length",
                "bond_length_binned",
            ],
        }

    def get_supported_descriptors(self) -> dict[str, list[str]]:
        """
        Get molecular descriptors supported by QM9 dataset.

        QM9 has optimized 3D geometries and can support ALL
        descriptor categories including geometric descriptors.
        """
        return {
            "categories": [
                "constitutional",
                "topological",
                "electronic",
                "geometric",  # QM9 has 3D coordinates
                "drug_likeness",
                "fragments",
            ],
            "excluded": [],  # QM9 supports all descriptors
            "recommended": [
                # Constitutional
                "MolWt",
                "NumRotatableBonds",
                "NumHDonors",
                "NumHAcceptors",
                # Topological
                "TPSA",
                "BertzCT",
                "Chi0v",
                "Chi1v",
                # Electronic
                "MaxPartialCharge",
                "MinPartialCharge",
                # Geometric (QM9 has optimized 3D structures)
                "RadiusOfGyration",
                "InertialShapeFactor",
                "Asphericity",
                # Drug-likeness
                "qed",
                "SPS",
            ],
            "requires_3d": True,  # QM9 provides optimized 3D structures
            "requires_charges": True,
        }

    def get_transform_recommendations(self) -> dict[str, list[str]]:
        """
        Get QM9-specific transform recommendations.

        Returns:
            Dict with recommended, avoid, and warning transforms
        """
        recommendations = {
            "recommended": [
                "GCNNorm - for message passing networks",
                "AddSelfLoops - required before GCNNorm",
                "NormalizeFeatures - for stable training",
                "RandomRotate - QM9 has 3D coordinates",
                "Distance - add distance-based edge features",
            ],
            "avoid": [],
            "warnings": [
                "VirtualNode may need careful handling with Mulliken charges",
            ],
        }

        return recommendations

    def _get_dataset_suitable_transforms(self, available_transforms: dict[str, Any]) -> list[str]:
        """
        QM9-suitable transforms based on structural and energetic properties.

        Args:
            available_transforms: Dict of all available transforms

        Returns:
            List of transform names suitable for QM9 datasets
        """
        suitable = []

        # Geometric transforms - QM9 has 3D coordinates
        geometric = ["RandomRotate", "RandomTranslate", "RandomScale"]
        suitable.extend([t for t in geometric if t in available_transforms])

        # Normalization
        normalization = ["GCNNorm", "NormalizeFeatures"]
        suitable.extend([t for t in normalization if t in available_transforms])

        # Graph structure
        structure = ["AddSelfLoops", "ToUndirected"]
        suitable.extend([t for t in structure if t in available_transforms])

        # Edge features
        edge_features = ["Distance", "Cartesian"]
        suitable.extend([t for t in edge_features if t in available_transforms])

        # Light augmentation
        augmentation = ["DropEdge", "MaskFeatures"]
        suitable.extend([t for t in augmentation if t in available_transforms])

        return suitable

    def _validate_dataset_specific_transforms(self, transform_names: list[str]) -> list[str]:
        """
        Validate transforms for QM9 dataset compatibility.

        Args:
            transform_names: List of transform class names

        Returns:
            List of warning messages
        """
        warnings = []

        # QM9 datasets have 3D coordinates - geometric transforms are relevant
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate", "RandomFlip"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            warnings.append(
                "QM9 dataset without geometric augmentation - consider adding RandomRotate for invariance"
            )

        # Vibrational data considerations
        if (
            hasattr(self.processing_config, "variable_len_graph_properties")
            and "freqs" in self.processing_config.variable_len_graph_properties
            and "RandomRotate" in transform_names
        ):
            warnings.append(
                "QM9 dataset has vibrational frequencies - geometric transforms may affect spectral properties"
            )

        # Distance-based transforms
        if "Distance" in transform_names or "Cartesian" in transform_names:
            warnings.append(
                "Distance/Cartesian transforms will add edge attributes - ensure model handles them"
            )

        return warnings

    def _check_transform_incompatibilities(self, transform_names: list[str]) -> list[str]:
        """
        Check for incompatible transform combinations for QM9.

        Args:
            transform_names: List of transform class names

        Returns:
            List of error messages (empty if all compatible)
        """
        # Per-atom dataset values vs VirtualNode: checked for every dataset (base handler)
        return []

    def _get_transform_recommendations(self, transform_names: list[str]) -> list[str]:
        """
        Get transform recommendations for QM9 datasets with specific suggestions.

        Args:
            transform_names: List of transform class names

        Returns:
            List of recommendation messages
        """
        recommendations = []

        # Recommend normalization if not present
        norm_transforms = ["Normalize", "GCNNorm", "NormalizeFeatures"]
        has_norm = any(t in transform_names for t in norm_transforms)

        if not has_norm:
            recommendations.append(
                "Consider adding GCNNorm for message passing neural networks. "
                "Use: transforms.GCNNorm(add_self_loops=False) if loops already added"
            )

        # Recommend self-loops for graph convolutions
        if "AddSelfLoops" not in transform_names and "GCNNorm" in transform_names:
            recommendations.append(
                "GCNNorm typically requires AddSelfLoops before it. "
                "Add: transforms.AddSelfLoops() before GCNNorm"
            )

        # Augmentation recommendations
        augmentation_transforms = ["DropEdge", "DropNode", "MaskFeatures"]
        has_augmentation = any(t in transform_names for t in augmentation_transforms)

        if not has_augmentation:
            recommendations.append(
                "Consider adding data augmentation to improve generalization. "
                "Suggestions: DropEdge(p=0.1) or MaskFeatures(p=0.1)"
            )

        # Geometric transform recommendations for QM9
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            recommendations.append(
                "QM9 3D structures benefit from geometric augmentation. "
                "Suggestion: RandomRotate() for rotational invariance testing"
            )

        # Distance/edge feature recommendations
        if (
            "Distance" not in transform_names
            and "Cartesian" not in transform_names
            and any(t in transform_names for t in ["GCNNorm", "GATConv", "SAGEConv"])
        ):
            recommendations.append(
                "Graph neural networks may benefit from edge features. "
                "Consider: Distance(norm=False, cat=True) or Cartesian(norm=False, cat=True)"
            )

        return recommendations
