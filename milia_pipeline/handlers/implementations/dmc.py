# milia_pipeline/handlers/implementations/dmc.py

"""
DMC Dataset Handler
===================

Handler for DMC (Diffusion Monte Carlo) datasets with exception integration
and transformation system support.

Extracted from dataset_handlers.py as part of the Handler Module Refactoring.

Key Features:
- DMC-specific transform compatibility validation
- Uncertainty-aware transform warnings
- Minimal transform recommendations for quantum structure preservation
- Inverse variance uncertainty weighting

Reference: milia Dataset Architecture Refactoring Plan v2.2.0
"""

import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from milia_pipeline.config.validators import (
    is_value_valid_and_not_nan,
    validate_molecular_structure,
    validate_uncertainty_data,
)
from milia_pipeline.exceptions import (
    DatasetSpecificHandlerError,
    HandlerConfigurationError,
    HandlerError,
    HandlerValidationError,
    MoleculeProcessingError,
    PropertyEnrichmentError,
)

# Import from refactored base handler
from milia_pipeline.handlers.base_handler import DatasetHandler, handle_transform_errors
from milia_pipeline.handlers.handler_registry import register_handler

logger = logging.getLogger(__name__)


@register_handler
class DMCDatasetHandler(DatasetHandler):
    """
    Handler for DMC datasets with exception integration and
    transformation system support.

    Enhancements:
    - DMC-specific transform compatibility validation
    - Uncertainty-aware transform warnings
    - Minimal transform recommendations for quantum structure preservation
    """

    def get_dataset_type(self) -> str:
        return "DMC"

    def validate_molecule_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str = "N/A"
    ) -> None:
        """Validate DMC-specific molecular data with exception handling."""
        try:
            # Validate core DMC properties
            essential_props = ["Etot", "atoms", "coordinates"]
            missing_props = []

            for prop in essential_props:
                if not self._is_valid_property(raw_properties_dict.get(prop)):
                    missing_props.append(prop)

            if missing_props:
                raise HandlerValidationError(
                    message=f"Missing required DMC properties: {missing_props}",
                    handler_type="DMC",
                    validation_type="essential_properties",
                    failed_validations=[f"Missing {prop}" for prop in missing_props],
                    molecule_index=molecule_index,
                    details="DMC molecules must have energy, atoms, and coordinates",
                )

            # Validate structural consistency
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(atoms, coordinates, molecule_index, identifier)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="DMC",
                        message=f"DMC molecular structure validation failed for molecule {molecule_index}",
                        operation="structure_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"InChI: {identifier}, Atoms: {len(atoms) if atoms else 0}, "
                        f"Coords: {len(coordinates) if coordinates else 0}, "
                        f"Error: {str(e)}",
                    ) from e

            # DMC-specific energy validation
            etot = raw_properties_dict.get("Etot")
            if etot is not None:
                try:
                    energy_val = float(etot)
                    # DMC energies should be reasonable
                    if abs(energy_val) > 10000:  # Hartree
                        self.logger.warning(
                            f"DMC molecule {molecule_index} has unusually large energy: {energy_val}"
                        )
                except (ValueError, TypeError):
                    raise DatasetSpecificHandlerError(
                        dataset_type="DMC",
                        message=f"DMC energy value cannot be converted to float for molecule {molecule_index}",
                        operation="energy_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"InChI: {identifier}, Energy value: {etot}, Type: {type(etot)}",
                    )
            # Validate uncertainty data if enabled
            if self.dataset_config.is_uncertainty_enabled:
                self._validate_uncertainty_data(raw_properties_dict, molecule_index, identifier)

        except (HandlerError, DatasetSpecificHandlerError):
            # Re-raise handler-specific errors
            raise
        except MoleculeProcessingError as e:
            # Convert molecule processing errors to DMC handler validation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DMC",
                message=f"DMC validation failed: {e.message}",
                operation="molecule_validation",
                details=f"Underlying error: {str(e)}",
            ) from e
        except Exception as e:
            # Convert unexpected errors to DMC handler errors
            raise DatasetSpecificHandlerError(
                dataset_type="DMC",
                message=f"Unexpected error during DMC validation: {str(e)}",
                operation="molecule_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def get_required_properties(self) -> list[str]:
        """Get DMC-specific required properties."""
        required = self.get_common_required_properties()
        required.extend(["Etot"])  # DMC primarily focuses on total energy
        required.append("inchi")  # Required for molecular charge determination

        # Add uncertainty field if enabled
        if self.dataset_config.is_uncertainty_enabled and self.dataset_config.uncertainty_config:
            uncertainty_field = self.dataset_config.uncertainty_config.get(
                "uncertainty_field_name", "std"
            )
            required.append(uncertainty_field)

        # Add properties from processing config (though DMC typically has fewer)
        if self.processing_config:
            required.extend(self.processing_config.scalar_graph_targets)
            required.extend(self.processing_config.node_features)

        return list(set(required))

    def get_identifier_keys(self) -> list[tuple[str, str]]:
        """
        Get DMC identifier keys for molecule creation.

        DMC datasets use InChI as primary identifier (same as DFT),
        with SMILES (from 'graphs' key) as fallback.

        Returns:
            List of (npz_key, identifier_type) tuples
        """
        return [("inchi", "inchi"), ("graphs", "smiles")]

    def get_molecular_charge(
        self,
        raw_properties_dict: dict[str, Any],
        atomic_numbers: np.ndarray,
        mol_identifier: str | None = None,
    ) -> int:
        """
        Extract charge from InChI string.

        DMC datasets contain InChI with charge information in /q layer.
        Same strategy as DFT since data format is similar.

        Args:
            raw_properties_dict: Raw molecule data from NPZ
            atomic_numbers: Array of atomic numbers (not used for DMC)
            mol_identifier: Molecular identifier (not used, InChI from raw data)

        Returns:
            int: Molecular charge from InChI (0 if cannot determine)
        """
        # Try to get InChI from raw data
        inchi = raw_properties_dict.get("inchi")

        if inchi:
            charge = self._extract_charge_from_inchi(inchi)
            if charge != 0:
                self.logger.debug(f"Extracted charge from InChI: {charge}")
            return charge

        # Fallback: assume neutral
        self.logger.debug("No InChI found, assuming neutral molecule")
        return 0

    def get_molecule_creation_strategy(self) -> str:
        """
        DMC datasets use identifier_coordinate_based strategy.

        DMC molecular data contains InChI identifiers which encode molecular
        connectivity and bonding. These are parsed to create the molecular graph,
        then QM-optimized coordinates are assigned to preserve exact 3D geometry.

        Data requirements:
            - InChI identifier (for connectivity/bonds)
            - Atomic numbers (for atom types)
            - Coordinates in Angstrom (for 3D geometry)
            - Molecular charge from InChI /q layer (for metadata)

        Returns:
            str: 'identifier_coordinate_based'
        """
        return "identifier_coordinate_based"

    def process_property_value(
        self, key: str, value: Any, molecule_index: int, identifier: str = "N/A"
    ) -> Any:
        """Process DMC-specific property values with exception handling."""
        try:
            # Handle uncertainty field
            if self.dataset_config.uncertainty_config:
                uncertainty_field = self.dataset_config.uncertainty_config.get(
                    "uncertainty_field_name", "std"
                )

                if key == uncertainty_field:
                    # Validate and convert uncertainty data
                    try:
                        validated_uncertainty = validate_uncertainty_data(
                            value,
                            molecule_index=molecule_index,
                            uncertainty_field_name=key,
                            require_positive=True,
                        )
                        return validated_uncertainty
                    except ValueError as e:
                        raise DatasetSpecificHandlerError(
                            dataset_type="DMC",
                            message=f"DMC uncertainty validation failed: {str(e)}",
                            operation="property_processing",
                            property_name=key,
                            details=str(e),
                        ) from e

            # Handle energy conversion for DMC
            if key == "Etot":
                if isinstance(value, (str, bytes, np.str_, np.bytes_)):
                    try:
                        return float(value)
                    except ValueError:
                        raise DatasetSpecificHandlerError(
                            dataset_type="DMC",
                            message=f"DMC energy is non-numeric string: '{value}'",
                            operation="property_processing",
                            details=f"Cannot convert '{value}' to float",
                        )

            return value

        except DatasetSpecificHandlerError:
            # Re-raise handler errors
            raise
        except Exception as e:
            # Convert unexpected property processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="DMC",
                message=f"Unexpected error processing DMC property '{key}': {str(e)}",
                operation="property_processing",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    # ========================================================================
    # DMC-Specific Transform Validation
    # ========================================================================

    def _validate_dataset_specific_transforms(self, transform_names: list[str]) -> list[str]:
        """
        Validate transforms for DMC dataset compatibility.

        Args:
            transform_names: List of transform class names

        Returns:
            List of warning messages
        """
        warnings = []

        # DMC datasets have uncertainty - augmentation affects it
        # Check both is_uncertainty_enabled AND if uncertainty_config exists
        has_uncertainty = (
            self.dataset_config.uncertainty_config is not None
            and len(self.dataset_config.uncertainty_config) > 0
        )

        if has_uncertainty:
            augmentation_transforms = ["DropEdge", "DropNode", "MaskFeatures"]
            has_augmentation = any(t in transform_names for t in augmentation_transforms)

            if has_augmentation:
                warnings.append(
                    "DMC dataset has uncertainties - data augmentation may require "
                    "uncertainty recalculation or special handling"
                )

        # Geometric transforms with DMC
        geometric_transforms = ["RandomRotate", "RandomScale"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if has_geometric:
            warnings.append(
                "DMC dataset with geometric augmentation - ensure this aligns with research goals"
            )

        return warnings

    def _check_transform_incompatibilities(self, transform_names: list[str]) -> list[str]:
        """
        Check for incompatible transform combinations for DMC.

        Args:
            transform_names: List of transform class names

        Returns:
            List of error messages (empty if all compatible)
        """
        errors = []

        # Check both uncertainty_config exists AND has content
        has_uncertainty = (
            self.dataset_config.uncertainty_config is not None
            and len(self.dataset_config.uncertainty_config) > 0
        )

        # VirtualNode incompatibility with uncertainty
        if "VirtualNode" in transform_names:
            if has_uncertainty:
                errors.append(
                    "VirtualNode incompatible with DMC uncertainties - "
                    "virtual node would need artificial uncertainty value"
                )

        # DropNode incompatibility with uncertainty weighting
        if "DropNode" in transform_names:
            if has_uncertainty:
                uncertainty_config = self.dataset_config.uncertainty_config
                if (
                    uncertainty_config
                    and uncertainty_config.get("uncertainty_weighting") == "inverse_variance"
                ):
                    errors.append(
                        "DropNode incompatible with inverse_variance uncertainty weighting - "
                        "node dropping changes graph structure and weight distribution"
                    )

        return errors

    def _get_transform_recommendations(self, transform_names: list[str]) -> list[str]:
        """
        Enhanced: Get transform recommendations for DMC datasets (conservative).

        Args:
            transform_names: List of transform class names

        Returns:
            List of recommendation messages
        """
        recommendations = []

        # DMC requires minimal transforms to preserve quantum structure
        if len(transform_names) > 3:
            recommendations.append(
                "DMC datasets typically use minimal transforms (≤3) to preserve quantum structure. "
                f"Current count: {len(transform_names)}. Consider reducing complexity."
            )

        # Structural transforms only
        structural_only = ["AddSelfLoops", "ToUndirected", "ToSparseTensor"]
        non_structural = [t for t in transform_names if t not in structural_only]

        if non_structural:
            recommendations.append(
                f"DMC dataset has non-structural transforms: {', '.join(non_structural)}. "
                "Verify these are compatible with DMC quantum calculations and uncertainties."
            )

        # Warn about any augmentation
        augmentation_transforms = ["DropEdge", "DropNode", "MaskFeatures", "RandomRotate"]
        has_augmentation = any(t in transform_names for t in augmentation_transforms)

        if has_augmentation:
            augmentation_used = [t for t in transform_names if t in augmentation_transforms]
            recommendations.append(
                f"Data augmentation detected: {', '.join(augmentation_used)}. "
                "DMC quantum structures should not be augmented. Remove these transforms "
                "unless you have specific research justification."
            )

        # Normalization recommendation (if none)
        if "NormalizeFeatures" not in transform_names:
            recommendations.append(
                "Consider adding NormalizeFeatures() for feature scaling. "
                "This is the ONLY recommended normalization for DMC datasets."
            )

        # Warn about edge features
        if "Distance" in transform_names or "Cartesian" in transform_names:
            recommendations.append(
                "Distance/Cartesian transforms add edge features to DMC quantum structures. "
                "Ensure your model architecture handles these appropriately."
            )

        return recommendations

    def get_transform_recommendations(self) -> dict[str, list[str]]:
        """
        Get DMC-specific transform recommendations.

        DMC datasets require conservative transforms to preserve quantum structure
        and uncertainty information.

        Returns:
            Dict with recommended, avoid, and warning transforms
        """
        recommendations = {"recommended": [], "avoid": [], "warnings": []}

        # Recommended minimal transforms for DMC
        recommendations["recommended"].extend(
            [
                "AddSelfLoops - structural connectivity only",
                "ToUndirected - if graph directionality not needed",
                "NormalizeFeatures - ONLY safe normalization for DMC",
            ]
        )

        # Avoid data augmentation for DMC
        recommendations["avoid"].extend(
            [
                "DropNode - breaks quantum structure",
                "DropEdge - disrupts molecular connectivity",
                "MaskFeatures - corrupts quantum properties",
                "RandomRotate - not applicable to DMC quantum calculations",
                "RandomScale - distorts quantum geometry",
                "VirtualNode - artificial nodes incompatible with DMC",
            ]
        )

        # Configuration-specific avoidances
        if "vibmodes" in getattr(self.processing_config, "variable_len_graph_properties", []):
            recommendations["avoid"].append("DropNode - incompatible with vibrational modes")

        # Check uncertainty configuration
        has_uncertainty = (
            self.dataset_config.uncertainty_config is not None
            and len(self.dataset_config.uncertainty_config) > 0
        )

        if has_uncertainty:
            recommendations["avoid"].extend(
                [
                    "VirtualNode - incompatible with DMC uncertainties",
                    "DropNode - incompatible with uncertainty weighting",
                ]
            )

        # Warnings
        recommendations["warnings"].extend(
            [
                "DMC datasets should use ≤3 transforms to preserve quantum structure",
                "Geometric transforms (Distance/Cartesian) require model support for edge features",
                "Any augmentation breaks DMC quantum uncertainty properties",
                "Normalization beyond NormalizeFeatures may corrupt uncertainties",
            ]
        )

        return recommendations

    def get_supported_descriptors(self) -> dict[str, list[str]]:
        """
        Get descriptors supported by DMC datasets.

        DMC datasets have 3D geometries and support most descriptors,
        similar to DFT datasets.
        """
        return {
            "categories": [
                "constitutional",
                "topological",
                "electronic",
                "geometric",  # DMC has 3D coordinates
                "drug_likeness",
                "fragments",
            ],
            "excluded": [],  # DMC supports all descriptors
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
                # Geometric (DMC-specific strength)
                "RadiusOfGyration",
                "InertialShapeFactor",
                # Drug-likeness
                "qed",
                "SPS",
            ],
            "requires_3d": True,  # DMC provides 3D structures
            "requires_charges": True,
        }

    def _get_dataset_suitable_transforms(self, available_transforms: dict[str, Any]) -> list[str]:
        """
        DMC-suitable transforms (minimal, structure-preserving).

        Args:
            available_transforms: Dict of all available transforms

        Returns:
            List of transform names suitable for DMC datasets
        """
        suitable = []

        # Minimal structural transforms only (preserve quantum calculations)
        structural = ["AddSelfLoops", "ToUndirected"]
        suitable.extend([t for t in structural if t in available_transforms])

        # Very light normalization only
        normalization = ["NormalizeFeatures"]
        suitable.extend([t for t in normalization if t in available_transforms])

        # NO aggressive augmentation for DMC (quantum structure must be preserved)
        # NO geometric transforms (coordinates are quantum-optimized)

        return suitable

    # ========================================================================
    # Self-contained scalar targets implementation
    # ========================================================================

    def _add_scalar_targets_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal DMC-specific implementation of scalar target addition with enhancements."""
        try:
            if not self.processing_config.scalar_graph_targets:
                return

            collected_targets = []

            for key in self.processing_config.scalar_graph_targets:
                try:
                    value = raw_properties_dict.get(key)

                    if value is None:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"Required DMC scalar target '{key}' is missing from raw data",
                            detail="Value retrieved was None",
                        )

                    # Convert to scalar float with DMC-specific handling
                    if isinstance(value, (int, float, np.number)):
                        val_to_add = float(value)
                    elif isinstance(value, np.ndarray):
                        if value.size == 1:
                            val_to_add = float(value.item())
                        else:
                            raise PropertyEnrichmentError(
                                molecule_index=molecule_index,
                                inchi=identifier,
                                property_name=key,
                                reason=f"DMC scalar target '{key}' has array shape {value.shape}, expected single scalar",
                                detail=f"Shape: {value.shape}, Size: {value.size}",
                            )
                    elif isinstance(value, (str, bytes, np.str_, np.bytes_)):
                        try:
                            val_to_add = float(value)
                        except ValueError:
                            raise PropertyEnrichmentError(
                                molecule_index=molecule_index,
                                inchi=identifier,
                                property_name=key,
                                reason=f"DMC scalar target '{key}' string cannot be converted to number",
                                detail=f"Value: '{value}'",
                            )
                    elif isinstance(value, (list, tuple)):
                        if len(value) == 1:
                            val_to_add = float(value[0])
                        else:
                            raise PropertyEnrichmentError(
                                molecule_index=molecule_index,
                                inchi=identifier,
                                property_name=key,
                                reason=f"DMC scalar target '{key}' is a list with {len(value)} elements, expected single scalar",
                                detail=f"Value: {value}",
                            )
                    else:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"DMC scalar target '{key}' has unexpected type {type(value)}",
                            detail=f"Value type: {type(value)}, Value: {value}",
                        )

                    # Validate the converted value
                    if not is_value_valid_and_not_nan(val_to_add):
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"DMC scalar target '{key}' has NaN, Inf, or None value after conversion",
                            detail=f"Converted value: {val_to_add}",
                        )

                    collected_targets.append(val_to_add)

                except PropertyEnrichmentError:
                    raise
                except Exception as e:
                    raise PropertyEnrichmentError(
                        molecule_index=molecule_index,
                        inchi=identifier,
                        property_name=key,
                        reason=f"Critical error processing DMC scalar target '{key}'",
                        detail=str(e),
                    ) from e

            if collected_targets:
                pyg_data.y = self._ensure_tensor(
                    collected_targets,
                    torch.float32,
                    "dmc_scalar_targets",
                    molecule_index,
                    identifier,
                )

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to DMC handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DMC",
                message=f"DMC scalar targets processing failed: {str(e)}",
                operation="add_scalar_targets",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    @handle_transform_errors("enrich_pyg_data")
    def enrich_pyg_data(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str = "N/A",
    ) -> Data:
        """DMC-specific PyG data enrichment with exception handling."""
        try:
            # Set dataset type
            pyg_data.dataset_type = "DMC"

            # Ensure num_nodes is set properly
            if not hasattr(pyg_data, "num_nodes") or pyg_data.num_nodes == 0:
                pyg_data.num_nodes = (
                    pyg_data.z.size(0) if hasattr(pyg_data, "z") and pyg_data.z is not None else 0
                )

            if pyg_data.num_nodes == 0:
                raise DatasetSpecificHandlerError(
                    dataset_type="DMC",
                    message="DMC molecule has 0 nodes, cannot proceed with enrichment",
                    operation="enrich_pyg_data",
                    details="No atoms available for processing",
                )

            # 1. Add scalar graph targets using internal implementation
            self._add_scalar_targets_internal(
                pyg_data, raw_properties_dict, molecule_index, identifier
            )

            # 2. Add DMC-specific uncertainty information if enabled
            if (
                self.dataset_config.is_uncertainty_enabled
                and self.dataset_config.uncertainty_config
            ):
                self._add_uncertainty_metadata_internal(
                    pyg_data, raw_properties_dict, molecule_index, identifier
                )

            self.logger.debug(f"DMC molecule {molecule_index}: Enrichment completed successfully")
            return pyg_data

        except (PropertyEnrichmentError, DatasetSpecificHandlerError):
            raise
        except Exception as e:
            # Convert unexpected errors to DMC handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DMC",
                message=f"DMC enrichment failed: {str(e)}",
                operation="enrich_pyg_data",
                details=f"Molecule {molecule_index}, Error during DMC-specific enrichment",
            ) from e

    def _add_uncertainty_metadata_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal DMC-specific uncertainty metadata implementation with exception handling."""
        try:
            uncertainty_config = self.dataset_config.uncertainty_config
            uncertainty_field = uncertainty_config.get("uncertainty_field_name", "std")

            uncertainty_value = raw_properties_dict.get(uncertainty_field)
            if uncertainty_value is not None:
                try:
                    uncertainty_scalar = validate_uncertainty_data(
                        uncertainty_value,  # ← First positional arg
                        molecule_index=molecule_index,
                        uncertainty_field_name=uncertainty_field,
                        require_positive=True,
                    )

                    if uncertainty_scalar is not None:
                        # Add uncertainty as tensor
                        pyg_data.uncertainty = self._ensure_tensor(
                            [uncertainty_scalar],
                            torch.float32,
                            "uncertainty",
                            molecule_index,
                            identifier,
                        )

                        # Calculate uncertainty weight
                        weighting_strategy = uncertainty_config.get(
                            "uncertainty_weighting", "inverse_variance"
                        )
                        if weighting_strategy == "inverse_variance":
                            epsilon = 1e-8
                            weight = 1.0 / (uncertainty_scalar**2 + epsilon)
                        else:
                            weight = 1.0

                        pyg_data.uncertainty_weight = self._ensure_tensor(
                            [weight],
                            torch.float32,
                            "uncertainty_weight",
                            molecule_index,
                            identifier,
                        )

                        # Calculate relative uncertainty if energy is available
                        if (
                            hasattr(pyg_data, "y")
                            and pyg_data.y is not None
                            and pyg_data.y.numel() > 0
                        ):
                            energy = pyg_data.y[0].item()

                            if energy != 0:
                                relative_uncertainty = abs(uncertainty_scalar / energy)
                                pyg_data.relative_uncertainty = self._ensure_tensor(
                                    [relative_uncertainty],
                                    torch.float32,
                                    "relative_uncertainty",
                                    molecule_index,
                                    identifier,
                                )

                                # Flag high uncertainty molecules
                                is_high_uncertainty = relative_uncertainty > 0.1  # 10% threshold
                                pyg_data.high_uncertainty = self._ensure_tensor(
                                    [is_high_uncertainty],
                                    torch.bool,
                                    "high_uncertainty",
                                    molecule_index,
                                    identifier,
                                )

                        self.logger.debug(
                            f"DMC molecule {molecule_index}: uncertainty={uncertainty_scalar:.6f}, weight={weight:.6f}"
                        )

                except ValueError as e:
                    self.logger.warning(
                        f"DMC molecule {molecule_index} uncertainty processing failed: {e}"
                    )

        except Exception as e:
            # Log uncertainty processing errors but don't fail enrichment
            self.logger.warning(
                f"DMC molecule {molecule_index} uncertainty metadata processing failed: {str(e)}"
            )

    def get_processing_statistics(
        self, processed_molecules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Enhanced: Generate DMC-specific processing statistics with transform info.

        Args:
            processed_molecules: List of processed molecule data dicts

        Returns:
            Dict containing comprehensive statistics
        """
        stats = {
            "dataset_type": "DMC",
            "total_processed": len(processed_molecules),
            "experimental_setup": self.experimental_setup,
        }

        # Existing uncertainty statistics
        if self.dataset_config.is_uncertainty_enabled:
            uncertainty_processed = 0
            high_uncertainty_count = 0

            for mol_data in processed_molecules:
                if mol_data.get("uncertainty_processed"):
                    uncertainty_processed += 1
                if mol_data.get("high_uncertainty"):
                    high_uncertainty_count += 1

            if uncertainty_processed > 0:
                stats["uncertainty_processing"] = {
                    "molecules_with_uncertainty": uncertainty_processed,
                    "high_uncertainty_molecules": high_uncertainty_count,
                }

        # Add transform usage information if experimental setup exists
        if self.experimental_setup:
            stats["transform_aware_processing"] = True
            stats["experimental_context"] = {
                "setup_name": self.experimental_setup,
                "dataset_type": "DMC",
                "transform_validation_performed": True,
                "uncertainty_aware": self.dataset_config.is_uncertainty_enabled,
            }

        return stats

    def get_supported_structural_features(self) -> dict[str, list[str]]:
        """
        DMC datasets support LIMITED structural features.

        DMC limitations:
        1. NO Mulliken charges (not computed in DMC calculations)
        2. NO partial charges (Gasteiger fails on DMC structures - causes NaN/Inf)
        3. May not have complete 3D coordinates (affects bond length calculations)

        This conservative feature set prevents NaN/Inf warnings and molecule
        rejections during DMC dataset processing.

        Returns:
            Dict containing only safe/supported atom and bond features
        """
        return {
            "atom": [
                # Topology features (always safe for DMC)
                "degree",
                "total_degree",
                "hybridization",
                "total_valence",
                "is_aromatic",
                "is_in_ring",
                "num_aromatic_bonds",
                "chirality",
                # EXCLUDED: 'partial_charge' - Gasteiger calculation fails on DMC
                # EXCLUDED: 'mulliken_charge' - not available in DMC data
            ],
            "bond": [
                # Bond topology features (always safe for DMC)
                "bond_type",
                "is_conjugated",
                "is_aromatic",
                "is_in_any_ring",
                "stereo",
                "bond_length",
                "bond_length_binned",
            ],
        }

    def validate_configuration(self) -> None:
        """Validate DMC-specific configuration with enhancements."""
        try:
            super().validate_configuration()

            # Validate uncertainty configuration if enabled
            if self.dataset_config.is_uncertainty_enabled:
                uncertainty_config = self.dataset_config.uncertainty_config
                if not uncertainty_config:
                    raise HandlerConfigurationError(
                        message="DMC uncertainty handling enabled but configuration missing",
                        handler_type="DMC",
                        config_validation_errors=["Missing uncertainty configuration"],
                        invalid_config_keys=["dmc_config.uncertainty_handling"],
                        details="Uncertainty enabled but no configuration provided",
                    )

                uncertainty_field = uncertainty_config.get("uncertainty_field_name")
                if not uncertainty_field:
                    raise HandlerConfigurationError(
                        message="DMC uncertainty field name not specified",
                        handler_type="DMC",
                        config_validation_errors=["Missing uncertainty field name"],
                        invalid_config_keys=[
                            "dmc_config.uncertainty_handling.uncertainty_field_name"
                        ],
                        details="Uncertainty field name required when uncertainty handling is enabled",
                    )

                max_threshold = uncertainty_config.get("max_uncertainty_threshold")
                if max_threshold is not None and max_threshold < 0:
                    raise HandlerConfigurationError(
                        message="DMC max uncertainty threshold must be non-negative",
                        handler_type="DMC",
                        config_validation_errors=[f"Invalid threshold: {max_threshold}"],
                        invalid_config_keys=[
                            "dmc_config.uncertainty_handling.max_uncertainty_threshold"
                        ],
                        details=f"Threshold value: {max_threshold}",
                    )

        except HandlerConfigurationError:
            raise
        except Exception as e:
            # Convert unexpected configuration validation errors
            raise HandlerConfigurationError(
                message=f"DMC configuration validation failed: {str(e)}",
                handler_type="DMC",
                details=f"Unexpected validation error: {type(e).__name__}: {str(e)}",
            ) from e

    def _is_valid_property(self, value: Any) -> bool:
        """Check if a property value is valid for DMC."""
        if value is None:
            return False

        # For string values, only reject obviously invalid ones
        if isinstance(value, str):
            value_str = value.strip().lower()
            if value_str in ["missing", "missing_etot", "invalid", "", "nan"]:
                return False
            # Don't reject numeric strings - let the specific validation handle conversion
            return True

        return is_value_valid_and_not_nan(value)

    def _validate_uncertainty_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str
    ) -> None:
        """Validate DMC uncertainty data with exception handling."""
        try:
            if not self.dataset_config.uncertainty_config:
                return

            uncertainty_field = self.dataset_config.uncertainty_config.get(
                "uncertainty_field_name", "std"
            )
            uncertainty_value = raw_properties_dict.get(uncertainty_field)

            if uncertainty_value is None:
                raise DatasetSpecificHandlerError(
                    dataset_type="DMC",
                    message=f"DMC uncertainty field '{uncertainty_field}' is missing but required",
                    operation="uncertainty_validation",
                    property_name=uncertainty_field,
                    details=f"Field '{uncertainty_field}' not found in molecular data",
                )

            # Validate uncertainty value
            try:
                validated_uncertainty = validate_uncertainty_data(
                    uncertainty_value,
                    molecule_index=molecule_index,
                    uncertainty_field_name=uncertainty_field,
                    require_positive=True,
                )

                if validated_uncertainty is None:
                    raise DatasetSpecificHandlerError(
                        dataset_type="DMC",
                        message=f"DMC uncertainty validation failed for field '{uncertainty_field}'",
                        operation="uncertainty_validation",
                        property_name=uncertainty_field,
                        details="Uncertainty validation returned None",
                    )

                # Check threshold if configured
                max_threshold = self.dataset_config.uncertainty_config.get(
                    "max_uncertainty_threshold"
                )
                if max_threshold is not None and validated_uncertainty > max_threshold:
                    raise DatasetSpecificHandlerError(
                        dataset_type="DMC",
                        message=f"DMC uncertainty {validated_uncertainty} exceeds threshold {max_threshold}",
                        operation="uncertainty_validation",
                        property_name=uncertainty_field,
                        details=f"Uncertainty: {validated_uncertainty}, Threshold: {max_threshold}",
                    )

            except ValueError as e:
                raise DatasetSpecificHandlerError(
                    dataset_type="DMC",
                    message=f"DMC uncertainty validation error: {str(e)}",
                    operation="uncertainty_validation",
                    property_name=uncertainty_field,
                    details=str(e),
                ) from e

        except DatasetSpecificHandlerError:
            raise
        except Exception as e:
            # Convert unexpected uncertainty validation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DMC",
                message=f"Unexpected error during DMC uncertainty validation: {str(e)}",
                operation="uncertainty_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e
