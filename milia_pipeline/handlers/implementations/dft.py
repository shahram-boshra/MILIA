# milia_pipeline/handlers/implementations/dft.py

"""
DFT Dataset Handler
===================

Handler for DFT (Density Functional Theory) quantum chemistry datasets with
exception integration and transformation system support.

Extracted from dataset_handlers.py as part of the Handler Module Refactoring.

Key Features:
- DFT-specific transform compatibility validation
- Geometric transform warnings for vibrational data
- Normalization and augmentation recommendations
- Vibrational data refinement with tolerance configuration
- Atomization energy calculation with unit safety

Reference: milia Dataset Architecture Refactoring Plan v2.2.0
"""

import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from milia_pipeline.config.config_constants import ATOMIC_ENERGIES_HARTREE, HAR2EV
from milia_pipeline.config.validators import (
    is_value_valid_and_not_nan,
    validate_molecular_structure,
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
class DFTDatasetHandler(DatasetHandler):
    """
    Handler for DFT datasets with exception integration and
    transformation system support.

    Enhancements:
    - DFT-specific transform compatibility validation
    - Geometric transform warnings for vibrational data
    - Normalization and augmentation recommendations
    """

    # Class-level tracking variables (shared across all instances)
    _vibrational_log_emitted = False
    _vibrational_error_count = 0

    def get_dataset_type(self) -> str:
        return "DFT"

    def validate_molecule_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str = "N/A"
    ) -> None:
        """Validate DFT-specific molecular data with exception handling."""
        try:
            # Validate essential DFT properties
            essential_props = ["Etot", "atoms", "coordinates"]
            missing_props = []

            for prop in essential_props:
                if not self._is_valid_property(raw_properties_dict.get(prop)):
                    missing_props.append(prop)

            if missing_props:
                raise HandlerValidationError(
                    message=f"Missing required DFT properties: {missing_props}",
                    handler_type="DFT",
                    validation_type="essential_properties",
                    failed_validations=[f"Missing {prop}" for prop in missing_props],
                    molecule_index=molecule_index,
                    details="DFT molecules must have energy, atoms, and coordinates",
                )

            # Validate structural consistency
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(atoms, coordinates, molecule_index, identifier)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="DFT",
                        message=f"DFT molecular structure validation failed for molecule {molecule_index}",
                        operation="structure_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"InChI: {identifier}, Atoms: {len(atoms) if atoms else 0}, "
                        f"Coords: {len(coordinates) if coordinates else 0}, "
                        f"Error: {str(e)}",
                    ) from e

            # Validate energy ranges (DFT energies typically more negative)
            etot = raw_properties_dict.get("Etot")
            if etot is not None and isinstance(etot, (int, float, np.number)):
                if etot > 0:
                    self.logger.warning(
                        f"DFT molecule {molecule_index} has positive total energy: {etot}"
                    )

            # Validate vibrational data if present
            self._validate_vibrational_data(raw_properties_dict, molecule_index, identifier)

        except (HandlerError, DatasetSpecificHandlerError):
            # Re-raise handler-specific errors
            raise
        except MoleculeProcessingError as e:
            # Convert molecule processing errors to DFT handler validation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message=f"DFT validation failed for molecule {molecule_index}: {e.message}",
                operation="molecule_validation",
                molecule_index=molecule_index,
                identifier=identifier,
                details=f"InChI: {identifier}, Underlying error: {str(e)}",
            ) from e
        except Exception as e:
            # Convert unexpected errors to DFT handler errors
            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message=f"Unexpected error during DFT validation: {str(e)}",
                operation="molecule_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def get_required_properties(self) -> list[str]:
        """Get DFT-specific required properties."""
        required = self.get_common_required_properties()
        required.extend(["Etot", "U0", "zpves"])  # Core DFT energies
        required.append("inchi")  # Required for molecular charge determination

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
        Get DFT identifier keys for molecule creation.

        DFT datasets use InChI as primary identifier (contains connectivity and charge info),
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

        DFT datasets contain InChI with charge information in /q layer.

        Args:
            raw_properties_dict: Raw molecule data from NPZ
            atomic_numbers: Array of atomic numbers (not used for DFT)
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
        DFT datasets use identifier_coordinate_based strategy.

        DFT molecular data contains InChI identifiers which encode molecular
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
        """Process DFT-specific property values with exception handling."""
        try:
            # Handle 'rots' property homogenization
            if key == "rots" and isinstance(value, list):
                try:
                    return np.array(value, dtype=float)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="DFT",
                        message=f"Failed to convert 'rots' list to numeric array for molecule {molecule_index}",
                        operation="property_processing",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        property_name=key,
                        details=f"InChI: {identifier}, Value type: {type(value)}, "
                        f"Length: {len(value)}, Conversion error: {str(e)}",
                    ) from e

            # Handle vibrational frequencies (can be complex)
            if key == "freqs" and value is not None:
                if not is_value_valid_and_not_nan(value):
                    raise DatasetSpecificHandlerError(
                        dataset_type="DFT",
                        message=f"Invalid vibrational frequency data for molecule {molecule_index}",
                        operation="property_processing",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        property_name=key,
                        details=f"InChI: {identifier}, Frequencies contain NaN or invalid values, "
                        f"Value type: {type(value)}",
                    )

            # Handle Mulliken charges
            if key == "Qmulliken" and value is not None:
                if not is_value_valid_and_not_nan(value):
                    self.logger.warning(
                        f"DFT molecule {molecule_index} has invalid Mulliken charges"
                    )
                    return None

            return value

        except DatasetSpecificHandlerError:
            # Re-raise DFT handler errors
            raise
        except Exception as e:
            # Convert unexpected property processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message=f"Unexpected error processing DFT property '{key}': {str(e)}",
                operation="property_processing",
                property_name=key,
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def get_transform_recommendations(self) -> dict[str, list[str]]:
        """
        Get DFT-specific transform recommendations.

        Returns:
            Dict with recommended, avoid, and warning transforms
        """
        recommendations = {"recommended": [], "avoid": [], "warnings": []}

        # Recommended transforms for DFT
        recommendations["recommended"].extend(
            [
                "GCNNorm - for message passing networks",
                "AddSelfLoops - required before GCNNorm",
                "RandomRotate - for 3D geometric augmentation",
                "Distance or Cartesian - for edge features",
            ]
        )

        # Avoid these in certain situations
        if "vibmodes" in getattr(self.processing_config, "variable_len_graph_properties", []):
            recommendations["avoid"].append(
                "DropNode - incompatible with vibrational modes (indexed by atom)"
            )

        if hasattr(self.processing_config, "node_features"):
            if "Qmulliken" in self.processing_config.node_features:
                recommendations["avoid"].append("VirtualNode - incompatible with Mulliken charges")

        # Warnings about transform combinations
        recommendations["warnings"].extend(
            [
                "RandomRotate may affect vibrational mode orientations",
                "Distance/Cartesian transforms add edge attributes - ensure model handles them",
            ]
        )

        return recommendations

    def get_supported_descriptors(self) -> dict[str, list[str]]:
        """
        Get descriptors supported by DFT datasets.

        DFT datasets have optimized 3D geometries and can support ALL
        descriptor categories including geometric descriptors.
        """
        return {
            "categories": [
                "constitutional",
                "topological",
                "electronic",
                "geometric",  # DFT has 3D coordinates
                "drug_likeness",
                "fragments",
            ],
            "excluded": [],  # DFT supports all descriptors
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
                # Geometric (DFT-specific strength)
                "RadiusOfGyration",
                "InertialShapeFactor",
                "Asphericity",
                # Drug-likeness
                "qed",
                "SPS",
            ],
            "requires_3d": True,  # DFT provides optimized 3D structures
            "requires_charges": True,
        }

    # ========================================================================
    # DFT-Specific Transform Validation
    # ========================================================================

    def _validate_dataset_specific_transforms(self, transform_names: list[str]) -> list[str]:
        """
        Validate transforms for DFT dataset compatibility.

        Args:
            transform_names: List of transform class names

        Returns:
            List of warning messages
        """
        warnings = []

        # DFT datasets have 3D coordinates - geometric transforms are relevant
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate", "RandomFlip"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            warnings.append(
                "DFT dataset without geometric augmentation - consider adding RandomRotate for invariance"
            )

        # Vibrational data considerations
        if hasattr(self.processing_config, "variable_len_graph_properties"):
            if "freqs" in self.processing_config.variable_len_graph_properties:
                if "RandomRotate" in transform_names:
                    warnings.append(
                        "DFT dataset has vibrational modes - RandomRotate may affect mode orientations"
                    )

        # Distance-based transforms
        if "Distance" in transform_names or "Cartesian" in transform_names:
            warnings.append(
                "Distance/Cartesian transforms will add edge attributes - ensure model handles them"
            )

        return warnings

    def _check_transform_incompatibilities(self, transform_names: list[str]) -> list[str]:
        """
        Check for incompatible transform combinations for DFT.

        Args:
            transform_names: List of transform class names

        Returns:
            List of error messages (empty if all compatible)
        """
        errors = []

        # VirtualNode incompatibility with certain DFT properties
        if "VirtualNode" in transform_names:
            if hasattr(self.processing_config, "node_features"):
                if "Qmulliken" in self.processing_config.node_features:
                    errors.append(
                        "VirtualNode incompatible with Mulliken charges - "
                        "virtual node would need artificial charge"
                    )

        # DropNode incompatibility with vibrational modes
        if "DropNode" in transform_names:
            if "vibmodes" in getattr(self.processing_config, "variable_len_graph_properties", []):
                errors.append(
                    "DropNode incompatible with vibrational modes - "
                    "modes are indexed by atom number"
                )

        return errors

    def _get_transform_recommendations(self, transform_names: list[str]) -> list[str]:
        """
        Enhanced: Get transform recommendations for DFT datasets with specific suggestions.

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
        if "AddSelfLoops" not in transform_names:
            if "GCNNorm" in transform_names:
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

        # Geometric transform recommendations for DFT
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            recommendations.append(
                "DFT 3D structures benefit from geometric augmentation. "
                "Suggestion: RandomRotate() for rotational invariance testing"
            )

        # Distance/edge feature recommendations
        if "Distance" not in transform_names and "Cartesian" not in transform_names:
            if any(t in transform_names for t in ["GCNNorm", "GATConv", "SAGEConv"]):
                recommendations.append(
                    "Graph neural networks may benefit from edge features. "
                    "Consider: Distance(norm=False, cat=True) or Cartesian(norm=False, cat=True)"
                )

        return recommendations

    def _get_dataset_suitable_transforms(self, available_transforms: dict[str, Any]) -> list[str]:
        """
        DFT-suitable transforms based on structural and energetic properties.

        Args:
            available_transforms: Dict of all available transforms

        Returns:
            List of transform names suitable for DFT datasets
        """
        suitable = []

        # Geometric transforms - DFT has 3D coordinates
        geometric = ["RandomRotate", "RandomTranslate", "RandomScale"]
        suitable.extend([t for t in geometric if t in available_transforms])

        # Graph structure transforms
        structural = ["AddSelfLoops", "ToUndirected", "Distance", "Cartesian"]
        suitable.extend([t for t in structural if t in available_transforms])

        # Normalization for neural networks
        normalization = ["GCNNorm", "NormalizeFeatures", "Normalize"]
        suitable.extend([t for t in normalization if t in available_transforms])

        # Light augmentation (not too aggressive for quantum data)
        augmentation = ["DropEdge", "MaskFeatures"]
        suitable.extend([t for t in augmentation if t in available_transforms])

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
        """Internal DFT-specific implementation of scalar target addition with enhancements."""
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
                            reason=f"Required DFT scalar target '{key}' is missing from raw data",
                            detail="Value retrieved was None",
                        )

                    # Convert to scalar float with DFT-specific handling
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
                                reason=f"DFT scalar target '{key}' has array shape {value.shape}, expected single scalar",
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
                                reason=f"DFT scalar target '{key}' string cannot be converted to number",
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
                                reason=f"DFT scalar target '{key}' is a list with {len(value)} elements, expected single scalar",
                                detail=f"Value: {value}",
                            )
                    else:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"DFT scalar target '{key}' has unexpected type {type(value)}",
                            detail=f"Value type: {type(value)}, Value: {value}",
                        )

                    # Validate the converted value
                    if not is_value_valid_and_not_nan(val_to_add):
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"DFT scalar target '{key}' has NaN, Inf, or None value after conversion",
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
                        reason=f"Critical error processing DFT scalar target '{key}'",
                        detail=str(e),
                    ) from e

            if collected_targets:
                pyg_data.y = self._ensure_tensor(
                    collected_targets,
                    torch.float32,
                    "dft_scalar_targets",
                    molecule_index,
                    identifier,
                )

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to DFT handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message=f"DFT scalar targets processing failed: {str(e)}",
                operation="add_scalar_targets",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    # ========================================================================
    # Self-contained variable-length properties
    # ========================================================================

    def _add_variable_length_properties_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal DFT-specific implementation of variable-length property addition with enhancements."""
        try:
            if not self.processing_config.variable_len_graph_properties:
                return

            if pyg_data.num_nodes == 0:
                raise DatasetSpecificHandlerError(
                    dataset_type="DFT",
                    message="No nodes found for DFT variable-length graph properties",
                    operation="add_variable_length_properties",
                    details="Required for DFT variable-length property processing",
                )

            # Process vibrational data with DFT-specific refinement
            if (
                "freqs" in self.processing_config.variable_len_graph_properties
                and "vibmodes" in self.processing_config.variable_len_graph_properties
            ):
                self._process_vibrational_data_internal(
                    pyg_data, raw_properties_dict, molecule_index, identifier
                )

            # Process other variable-length properties
            for key in self.processing_config.variable_len_graph_properties:
                if key in ["freqs", "vibmodes"]:
                    continue  # Already processed above

                try:
                    value = raw_properties_dict.get(key)

                    if not is_value_valid_and_not_nan(value):
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"Missing, invalid, or NaN DFT variable-length property '{key}'",
                            detail=f"Value: {value}",
                        )

                    # Convert to tensor with appropriate dtype
                    dtype = torch.complex64 if "freq" in key else torch.float32
                    property_tensor = self._ensure_tensor(
                        value, dtype, key, molecule_index, identifier
                    )
                    setattr(pyg_data, key, property_tensor)

                except PropertyEnrichmentError:
                    raise
                except Exception as e:
                    raise PropertyEnrichmentError(
                        molecule_index=molecule_index,
                        inchi=identifier,
                        property_name=key,
                        reason=f"Error processing DFT variable-length property '{key}'",
                        detail=str(e),
                    ) from e

        except (PropertyEnrichmentError, DatasetSpecificHandlerError):
            raise
        except Exception as e:
            # Convert unexpected errors to DFT handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message=f"DFT variable-length properties processing failed: {str(e)}",
                operation="add_variable_length_properties",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _process_vibrational_data_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal DFT-specific vibrational data processing with refinement and exception handling."""
        try:
            raw_freqs = raw_properties_dict.get("freqs")
            raw_vibmodes = raw_properties_dict.get("vibmodes")

            if raw_freqs is None or raw_vibmodes is None:
                self.logger.debug(
                    f"DFT molecule {molecule_index}: Skipping vibrational refinement - missing data"
                )
                return

            # Log once at the very first vibrational processing (class-level check)
            # NOTE: Logging moved to avoid tqdm progress bar interference
            # This log is now emitted before the processing loop starts in milia_dataset.py
            if not DFTDatasetHandler._vibrational_log_emitted:
                DFTDatasetHandler._vibrational_log_emitted = True
                self.logger.debug(
                    "DFT vibrational refinement processing active for dataset molecules"
                )

            # Get refinement tolerance from processing config
            refinement_tolerance = 1e-4  # Default
            if self.processing_config.vibration_refinement and isinstance(
                self.processing_config.vibration_refinement, dict
            ):
                refinement_tolerance = self.processing_config.vibration_refinement.get(
                    "comparison_tolerance", refinement_tolerance
                )

            try:
                # Import refinement function as internal utility
                from milia_pipeline.config.data_refining import refine_molecular_vibrations

                cleaned_freqs, cleaned_vibmodes, is_accepted = refine_molecular_vibrations(
                    freqs=raw_freqs,
                    vibmodes=raw_vibmodes,
                    comparison_tolerance=refinement_tolerance,
                    molecule_index=molecule_index,
                    dataset_config=self.dataset_config,
                )

                if not is_accepted:
                    # ERROR - log this specific failure
                    DFTDatasetHandler._vibrational_error_count += 1
                    self.logger.warning(
                        f"DFT molecule {molecule_index} ({identifier}): Vibrational refinement REJECTED - "
                        f"freqs={len(cleaned_freqs)}, modes={len(cleaned_vibmodes)}"
                    )
                    raise DatasetSpecificHandlerError(
                        dataset_type="DFT",
                        message=f"DFT vibrational data refinement rejected for molecule {molecule_index}",
                        operation="vibrational_refinement",
                        details=f"Freqs count: {len(cleaned_freqs)}, Vibmodes count: {len(cleaned_vibmodes)}",
                    )

                # Process frequencies
                freqs_dtype = (
                    torch.complex64
                    if hasattr(raw_freqs, "dtype") and np.iscomplexobj(raw_freqs)
                    else torch.float32
                )
                pyg_data.freqs = self._ensure_tensor(
                    np.asarray(cleaned_freqs), freqs_dtype, "freqs", molecule_index, identifier
                )

                # Process vibrational modes
                pyg_data.vibmodes = self._process_vibmodes_internal(
                    cleaned_vibmodes, pyg_data.num_nodes, molecule_index, identifier
                )

                # SUCCESS - no logging at INFO level, only DEBUG
                self.logger.debug(
                    f"DFT molecule {molecule_index}: Vibrational data processed "
                    f"(freqs: {len(cleaned_freqs)}, modes: {len(cleaned_vibmodes)})"
                )

            except DatasetSpecificHandlerError:
                # Already logged above, just re-raise
                raise
            except Exception as e:
                # ERROR - log this specific failure
                DFTDatasetHandler._vibrational_error_count += 1
                self.logger.warning(
                    f"DFT molecule {molecule_index} ({identifier}): Vibrational processing FAILED - {str(e)}"
                )
                raise DatasetSpecificHandlerError(
                    dataset_type="DFT",
                    message=f"DFT vibrational data processing failed: {str(e)}",
                    operation="vibrational_refinement",
                    details="Error during vibrational refinement",
                ) from e

        except DatasetSpecificHandlerError:
            raise
        except Exception as e:
            # Unexpected ERROR - log it
            DFTDatasetHandler._vibrational_error_count += 1
            self.logger.warning(
                f"DFT molecule {molecule_index} ({identifier}): Unexpected vibrational error - "
                f"{type(e).__name__}: {str(e)}"
            )
            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message=f"Unexpected error in vibrational data processing: {str(e)}",
                operation="vibrational_refinement",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _process_vibmodes_internal(
        self, vibmodes_data: np.ndarray | list, num_atoms: int, molecule_index: int, identifier: str
    ) -> list[torch.Tensor]:
        """Internal DFT-specific vibrational modes processing with exception handling."""
        try:
            if num_atoms == 0:
                raise DatasetSpecificHandlerError(
                    dataset_type="DFT",
                    message="Cannot process DFT vibmodes: num_nodes is 0",
                    operation="vibmodes_processing",
                    details="No atoms available for vibmode processing",
                )

            # Handle refined vibmodes (list of arrays)
            if isinstance(vibmodes_data, list) and all(
                isinstance(v, (np.ndarray, list)) for v in vibmodes_data
            ):
                tensor_list = []
                for i, mode_data in enumerate(vibmodes_data):
                    if isinstance(mode_data, list):
                        mode_data = np.asarray(mode_data, dtype=np.float32)

                    # Validate mode shape
                    if (
                        mode_data.ndim == 2
                        and mode_data.shape[0] == num_atoms
                        and mode_data.shape[1] == 3
                    ):
                        mode_tensor = self._ensure_tensor(
                            mode_data,
                            torch.float32,
                            f"vibmodes_mode_{i}",
                            molecule_index,
                            identifier,
                        )
                        tensor_list.append(mode_tensor)
                    else:
                        raise DatasetSpecificHandlerError(
                            dataset_type="DFT",
                            message=f"DFT vibmode {i} has invalid shape: expected ({num_atoms}, 3), got {mode_data.shape}",
                            operation="vibmodes_processing",
                            details=f"Mode {i} shape: {mode_data.shape}",
                        )
                return tensor_list

            # Handle other formats
            elif isinstance(vibmodes_data, np.ndarray):
                if vibmodes_data.ndim == 2 and vibmodes_data.shape[1] == 3:
                    # Reshape from (N*num_atoms, 3) to (N, num_atoms, 3)
                    if vibmodes_data.shape[0] % num_atoms != 0:
                        raise DatasetSpecificHandlerError(
                            dataset_type="DFT",
                            message=f"DFT vibmodes array dimension mismatch: {vibmodes_data.shape[0]} not divisible by {num_atoms}",
                            operation="vibmodes_processing",
                            details=f"Array shape: {vibmodes_data.shape}",
                        )

                    num_modes = vibmodes_data.shape[0] // num_atoms
                    reshaped_value = vibmodes_data.reshape(num_modes, num_atoms, 3)

                    tensor_list = []
                    for i, mode_data in enumerate(reshaped_value):
                        mode_tensor = self._ensure_tensor(
                            mode_data,
                            torch.float32,
                            f"vibmodes_mode_{i}",
                            molecule_index,
                            identifier,
                        )
                        tensor_list.append(mode_tensor)
                    return tensor_list

                elif (
                    vibmodes_data.ndim == 3
                    and vibmodes_data.shape[1] == num_atoms
                    and vibmodes_data.shape[2] == 3
                ):
                    # Already in correct format
                    tensor_list = []
                    for i, mode_data in enumerate(vibmodes_data):
                        mode_tensor = self._ensure_tensor(
                            mode_data,
                            torch.float32,
                            f"vibmodes_mode_{i}",
                            molecule_index,
                            identifier,
                        )
                        tensor_list.append(mode_tensor)
                    return tensor_list

            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message="DFT vibmodes has unexpected format",
                operation="vibmodes_processing",
                details=f"Type: {type(vibmodes_data)}, Shape: {vibmodes_data.shape if isinstance(vibmodes_data, np.ndarray) else 'N/A'}",
            )

        except DatasetSpecificHandlerError:
            raise
        except Exception as e:
            # Convert unexpected errors to DFT handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message=f"Unexpected error processing vibmodes: {str(e)}",
                operation="vibmodes_processing",
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
        """DFT-specific PyG data enrichment with exception handling."""
        try:
            # Set dataset type
            pyg_data.dataset_type = "DFT"

            # Ensure num_nodes is set properly
            if not hasattr(pyg_data, "num_nodes") or pyg_data.num_nodes == 0:
                pyg_data.num_nodes = (
                    pyg_data.z.size(0) if hasattr(pyg_data, "z") and pyg_data.z is not None else 0
                )

            if pyg_data.num_nodes == 0:
                raise DatasetSpecificHandlerError(
                    dataset_type="DFT",
                    message="DFT molecule has 0 nodes, cannot proceed with enrichment",
                    operation="enrich_pyg_data",
                    details="No atoms available for processing",
                )

            # 1. Add scalar graph targets using internal implementation
            self._add_scalar_targets_internal(
                pyg_data, raw_properties_dict, molecule_index, identifier
            )

            # 2. Add vector graph properties (simplified internal implementation)
            if self.processing_config.vector_graph_properties:
                self._add_vector_properties_internal(
                    pyg_data, raw_properties_dict, molecule_index, identifier
                )

            # 3. Add variable-length properties using internal implementation
            if self.processing_config.variable_len_graph_properties:
                self._add_variable_length_properties_internal(
                    pyg_data, raw_properties_dict, molecule_index, identifier
                )

            # 4. Add atomization energy if configured
            if (
                self.processing_config.calculate_atomization_energy_from
                and self.processing_config.atomization_energy_key_name
            ):
                atomization_energy = self._calculate_atomization_energy_internal(
                    raw_properties_dict, pyg_data, molecule_index, identifier
                )

                if atomization_energy is not None:
                    setattr(
                        pyg_data,
                        self.processing_config.atomization_energy_key_name,
                        self._ensure_tensor(
                            [atomization_energy],
                            torch.float32,
                            "atomization_energy",
                            molecule_index,
                            identifier,
                        ),
                    )

                    # Also add to targets if y exists
                    if hasattr(pyg_data, "y") and pyg_data.y is not None:
                        current_y = pyg_data.y
                        atomization_tensor = self._ensure_tensor(
                            [atomization_energy],
                            torch.float32,
                            "atomization_energy_target",
                            molecule_index,
                            identifier,
                        )
                        pyg_data.y = torch.cat([current_y, atomization_tensor])

            self.logger.debug(f"DFT molecule {molecule_index}: Enrichment completed")
            return pyg_data

        except (PropertyEnrichmentError, DatasetSpecificHandlerError):
            raise
        except Exception as e:
            # Convert unexpected errors to DFT handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message=f"DFT enrichment failed: {str(e)}",
                operation="enrich_pyg_data",
                details=f"Molecule {molecule_index}, Error during DFT-specific enrichment",
            ) from e

    def _add_vector_properties_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal DFT-specific vector properties implementation with exception handling."""
        try:
            for prop_key in self.processing_config.vector_graph_properties:
                try:
                    value = raw_properties_dict.get(prop_key)

                    if not is_value_valid_and_not_nan(value):
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=prop_key,
                            reason=f"Missing or invalid DFT vector property '{prop_key}'",
                            detail=f"Value: {value}",
                        )

                    # Convert to numpy array if needed
                    if isinstance(value, (list, tuple)):
                        value = np.asarray(value, dtype=np.float32)

                    if not isinstance(value, np.ndarray) or value.ndim != 1:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=prop_key,
                            reason=f"DFT vector property '{prop_key}' is not a 1D array",
                            detail=f"Type: {type(value)}, Dims: {getattr(value, 'ndim', 'N/A')}",
                        )

                    # Special handling for 'rots' - pad (2,) to (3,) for linear molecules
                    if prop_key == "rots":
                        if value.shape == (2,):
                            value = np.pad(value, (0, 1), "constant", constant_values=0.0)
                            self.logger.debug(
                                f"DFT molecule {molecule_index}: Padded 'rots' from (2,) to (3,)"
                            )
                        elif value.shape != (3,):
                            raise PropertyEnrichmentError(
                                molecule_index=molecule_index,
                                inchi=identifier,
                                property_name=prop_key,
                                reason=f"DFT vector property '{prop_key}' has unexpected shape {value.shape}",
                                detail="Expected (3,) or (2,) for 'rots'",
                            )

                    # Convert to tensor and set attribute
                    property_tensor = self._ensure_tensor(
                        value, torch.float32, prop_key, molecule_index, identifier
                    )
                    setattr(pyg_data, prop_key, property_tensor)

                except PropertyEnrichmentError:
                    raise
                except Exception as e:
                    raise PropertyEnrichmentError(
                        molecule_index=molecule_index,
                        inchi=identifier,
                        property_name=prop_key,
                        reason=f"Error processing DFT vector property '{prop_key}'",
                        detail=str(e),
                    ) from e

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to DFT handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="DFT",
                message=f"DFT vector properties processing failed: {str(e)}",
                operation="add_vector_properties",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def get_processing_statistics(
        self, processed_molecules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Enhanced: Generate DFT-specific processing statistics with transform info.

        Args:
            processed_molecules: List of processed molecule data dicts

        Returns:
            Dict containing comprehensive statistics
        """
        stats = {
            "dataset_type": "DFT",
            "total_processed": len(processed_molecules),
            "experimental_setup": self.experimental_setup,
        }

        # Existing vibrational refinement statistics
        vibrational_refinements = 0
        total_freq_reduction = 0
        atomization_calculations = 0

        for mol_data in processed_molecules:
            if mol_data.get("vibrational_refinement_performed"):
                vibrational_refinements += 1
                original_freqs = mol_data.get("original_freqs_count", 0)
                refined_freqs = mol_data.get("refined_freqs_count", 0)
                if original_freqs > 0:
                    reduction = (original_freqs - refined_freqs) / original_freqs
                    total_freq_reduction += reduction

            if mol_data.get("atomization_energy_calculated"):
                atomization_calculations += 1

        if vibrational_refinements > 0:
            stats["vibrational_refinement"] = {
                "molecules_refined": vibrational_refinements,
                "average_frequency_reduction": total_freq_reduction / vibrational_refinements,
            }

        if DFTDatasetHandler._vibrational_error_count > 0:
            stats["vibrational_errors"] = DFTDatasetHandler._vibrational_error_count

        if atomization_calculations > 0:
            stats["atomization_energy_calculations"] = atomization_calculations

        # Add transform usage information if experimental setup exists
        if self.experimental_setup:
            stats["transform_aware_processing"] = True
            stats["experimental_context"] = {
                "setup_name": self.experimental_setup,
                "dataset_type": "DFT",
                "transform_validation_performed": True,
            }

        return stats

    def get_supported_structural_features(self) -> dict[str, list[str]]:
        """
        DFT datasets support ALL structural features.

        DFT has complete quantum chemical data including:
        - Mulliken charges (directly from DFT calculations)
        - 3D coordinates (for bond lengths)
        - Full electronic structure information

        However, partial_charge (Gasteiger) fails for most DFT molecules,
        causing NaN/Inf values and molecule rejection.

        Returns:
            Dict containing all supported atom and bond features
        """
        return {
            "atom": [
                # Topology features (always available)
                "degree",
                "total_degree",
                "hybridization",
                "total_valence",
                "is_aromatic",
                "is_in_ring",
                "num_aromatic_bonds",
                "chirality",
                # Charge features (DFT has both)
                #'partial_charge',      # Gasteiger charges (computed from structure)
                "mulliken_charge",  # DFT Mulliken charges (from quantum calculation)
            ],
            "bond": [
                # Bond topology features (always available)
                "bond_type",
                "is_conjugated",
                "is_aromatic",
                "is_in_any_ring",
                "stereo",
                # Geometric features (DFT has 3D coordinates)
                "bond_length",
                "bond_length_binned",
            ],
        }

    def validate_configuration(self) -> None:
        """Validate DFT-specific configuration with enhancements."""
        try:
            super().validate_configuration()

            # Check atomization energy configuration
            if (
                self.processing_config.calculate_atomization_energy_from
                and not ATOMIC_ENERGIES_HARTREE
            ):
                raise HandlerConfigurationError(
                    message="DFT atomization energy calculation requested but atomic energies not available",
                    handler_type="DFT",
                    config_validation_errors=["Missing atomic energies reference data"],
                    details="ATOMIC_ENERGIES_HARTREE constant not available",
                )

            # Check vibrational refinement configuration
            if (
                "freqs" in self.processing_config.variable_len_graph_properties
                and "vibmodes" in self.processing_config.variable_len_graph_properties
            ):
                vibration_config = self.processing_config.vibration_refinement or {}
                tolerance = vibration_config.get("comparison_tolerance", 1e-4)
                if tolerance <= 0:
                    raise HandlerConfigurationError(
                        message="DFT vibrational refinement tolerance must be positive",
                        handler_type="DFT",
                        config_validation_errors=[f"Invalid tolerance: {tolerance}"],
                        invalid_config_keys=["vibration_refinement.comparison_tolerance"],
                        details=f"Tolerance value: {tolerance}",
                    )

        except HandlerConfigurationError:
            raise
        except Exception as e:
            # Convert unexpected configuration validation errors
            raise HandlerConfigurationError(
                message=f"DFT configuration validation failed: {str(e)}",
                handler_type="DFT",
                details=f"Unexpected validation error: {type(e).__name__}: {str(e)}",
            ) from e

    def _is_valid_property(self, value: Any) -> bool:
        """Check if a property value is valid for DFT."""
        if value is None:
            return False
        if isinstance(value, str) and value.lower() in ["missing", "invalid", "", "nan"]:
            return False
        return is_value_valid_and_not_nan(value)

    def _validate_vibrational_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str
    ) -> None:
        """
        Validate DFT vibrational data with exception handling.

        Skip detailed validation of vibrational data during initial
        validation phase, as the complex nested structures in milia require specialized
        processing before meaningful validation can occur. The refinement process in
        _process_vibrational_data_internal() includes comprehensive validation.
        """
        try:
            freqs = raw_properties_dict.get("freqs")
            vibmodes = raw_properties_dict.get("vibmodes")

            if freqs is not None and vibmodes is not None:
                # Only check for presence, not detailed structure
                # The complex nested structures in milia (object arrays with lists of np.float64,
                # mixed empty lists, variable nesting depths) require specialized processing
                # in the refinement phase before meaningful validation can occur.

                self.logger.debug(
                    f"DFT molecule {molecule_index}: Vibrational data present - will validate during refinement"
                )

                # Light validation - just ensure they're not obviously broken
                if freqs is None or vibmodes is None:
                    self.logger.warning(f"DFT molecule {molecule_index} missing vibrational data")
                elif not hasattr(freqs, "__len__") or not hasattr(vibmodes, "__len__"):
                    self.logger.warning(
                        f"DFT molecule {molecule_index} vibrational data not array-like"
                    )
                else:
                    # Skip detailed validation - let refinement handle the complex structures
                    self.logger.debug(
                        f"DFT molecule {molecule_index}: Vibrational data validation deferred to refinement phase"
                    )

        except Exception as e:
            # Log validation errors but don't fail the whole validation
            self.logger.warning(
                f"DFT molecule {molecule_index} vibrational validation warning: {str(e)}"
            )

    def _calculate_atomization_energy_internal(
        self,
        raw_properties_dict: dict[str, Any],
        pyg_data: Data,
        molecule_index: int,
        identifier: str,
    ) -> float | None:
        """
        Internal DFT-specific atomization energy calculation with exception handling.

        UNIT SAFETY: Always works with original Hartree values to ensure correct
        unit handling. Converts to eV only at the final step.
        """
        try:
            if not self.processing_config.calculate_atomization_energy_from:
                return None

            base_energy_key = self.processing_config.calculate_atomization_energy_from
            base_energy_raw = raw_properties_dict.get(base_energy_key)

            if not is_value_valid_and_not_nan(base_energy_raw):
                self.logger.warning(
                    f"DFT molecule {molecule_index} missing {base_energy_key} for atomization energy"
                )
                return None

            if not hasattr(pyg_data, "z") or pyg_data.z is None:
                self.logger.warning(
                    f"DFT molecule {molecule_index} missing atomic numbers for atomization energy"
                )
                return None

            # ✓ Convert to float - keep in original Hartree units
            if isinstance(base_energy_raw, (str, bytes)):
                base_energy_hartree = float(base_energy_raw)
            elif isinstance(base_energy_raw, np.ndarray) and base_energy_raw.size == 1:
                base_energy_hartree = float(base_energy_raw.item())
            elif isinstance(base_energy_raw, (int, float, np.number)):
                base_energy_hartree = float(base_energy_raw)
            else:
                self.logger.warning(
                    f"DFT molecule {molecule_index}: Cannot convert {base_energy_key} to float"
                )
                return None

            # Calculate atomization energy using internal logic
            if HAR2EV is None or not ATOMIC_ENERGIES_HARTREE:
                self.logger.warning(
                    f"DFT molecule {molecule_index}: Missing atomic energies for atomization calculation"
                )
                return None

            # ✓ Sum atomic energies (already in Hartree from constants)
            sum_atomic_energies_hartree = 0.0
            for atomic_num in pyg_data.z.tolist():
                atomic_energy = ATOMIC_ENERGIES_HARTREE.get(atomic_num)
                if atomic_energy is None:
                    self.logger.warning(
                        f"DFT molecule {molecule_index}: Missing atomic energy for element {atomic_num}"
                    )
                    return None
                sum_atomic_energies_hartree += atomic_energy

            # ✓ CORRECT - Do arithmetic in Hartree (same units!)
            atomization_energy_hartree = base_energy_hartree - sum_atomic_energies_hartree

            # ✓ CORRECT - Convert to eV only at the end
            atomization_energy_eV = atomization_energy_hartree * HAR2EV

            self.logger.debug(
                f"DFT molecule {molecule_index} atomization energy: {atomization_energy_eV:.4f} eV"
            )
            return atomization_energy_eV

        except Exception as e:
            # Log errors but don't fail processing
            self.logger.error(
                f"Error calculating DFT atomization energy for molecule {molecule_index}: {e}"
            )
            return None
