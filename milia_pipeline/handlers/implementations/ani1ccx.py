# milia_pipeline/handlers/implementations/ani1ccx.py

"""
ANI-1ccx Dataset Handler
========================

Handler for ANI-1ccx quantum chemistry datasets with exception integration
and transformation system support.

Extracted from dataset_handlers.py as part of the Handler Module Refactoring.

Key Features:
- Uses coordinate_based strategy (NO InChI/SMILES identifiers available)
- Coordinates in Angstrom
- Energies in Hartree
- Primary energy target: ccsd(t)_cbs.energy (mapped to 'energy' in NPZ)
- CCSD(T)/CBS accurate energies for a subset of ANI-1x
- All molecules are neutral (charge = 0)

ANI-1ccx is a subset of ANI-1x with CCSD(T)/CBS level energies for
high-accuracy energy predictions.

Reference: Smith et al., Nature Communications 10, 2903 (2019)
"""

import logging
from typing import Any

import numpy as np

from milia_pipeline.config.validators import (
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
class ANI1ccxDatasetHandler(DatasetHandler):
    """
    Handler for ANI-1ccx datasets with exception integration and transformation system support.

    ANI-1ccx is a quantum chemistry dataset containing ~500k coupled-cluster conformations
    for organic molecules (CHNO) computed at CCSD(T)/CBS level using transfer learning.

    Key characteristics:
    - Uses coordinate_based strategy (NO InChI/SMILES identifiers available)
    - Coordinates in Angstrom (DFT-optimized from ωB97x/6-31G*)
    - Primary energy: ccsd_energy (CCSD(T)/CBS extrapolation) in Hartree
    - Secondary energy: dft_energy (ωB97x/6-31G*) for comparison
    - Has forces, Hirshfeld charges, CM5 charges, and dipoles (from DFT)
    - All molecules are neutral (charge = 0)

    CRITICAL DIFFERENCES FROM ANI-1x:
    - ANI-1x: ~5M conformers with DFT energies only
    - ANI-1ccx: ~500k conformers with CCSD(T)/CBS energies (higher accuracy)
    - ANI-1ccx is a subset of ANI-1x with coupled-cluster properties

    Reference: Smith et al., Scientific Data 7, 134 (2020)
    """

    def get_dataset_type(self) -> str:
        return "ANI1ccx"

    def validate_molecule_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str = "N/A"
    ) -> None:
        """Validate ANI-1ccx-specific molecular data with exception handling."""
        try:
            # Validate essential ANI-1ccx properties
            # Note: 'ccsd_energy' is the primary target (mapped from ccsd(t)_cbs.energy)
            essential_props = ["ccsd_energy", "atoms", "coordinates"]
            missing_props = []

            for prop in essential_props:
                if not self._is_valid_property(raw_properties_dict.get(prop)):
                    missing_props.append(prop)

            if missing_props:
                raise HandlerValidationError(
                    message=f"Missing required ANI-1ccx properties: {missing_props}",
                    handler_type="ANI1ccx",
                    validation_type="essential_properties",
                    failed_validations=[f"Missing {prop}" for prop in missing_props],
                    molecule_index=molecule_index,
                    details="ANI-1ccx molecules must have ccsd_energy (CCSD(T)/CBS), atoms, and coordinates",
                )

            # Validate structural consistency
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(atoms, coordinates, molecule_index, identifier)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="ANI1ccx",
                        message=f"ANI-1ccx molecular structure validation failed for molecule {molecule_index}",
                        operation="structure_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"Identifier: {identifier}, Atoms: {len(atoms) if atoms else 0}, "
                        f"Coords: {len(coordinates) if coordinates else 0}, "
                        f"Error: {str(e)}",
                    ) from e

            # Validate CC energy (ANI-1ccx energies are typically negative in Hartree)
            ccsd_energy = raw_properties_dict.get("ccsd_energy")
            if (
                ccsd_energy is not None
                and isinstance(ccsd_energy, (int, float, np.number))
                and ccsd_energy > 0
            ):
                self.logger.warning(
                    f"ANI-1ccx molecule {molecule_index} has positive CC energy: {ccsd_energy}"
                )

        except (HandlerError, DatasetSpecificHandlerError):
            # Re-raise handler-specific errors
            raise
        except MoleculeProcessingError as e:
            # Convert molecule processing errors to ANI-1ccx handler validation errors
            raise DatasetSpecificHandlerError(
                dataset_type="ANI1ccx",
                message=f"ANI-1ccx validation failed for molecule {molecule_index}: {e.message}",
                operation="molecule_validation",
                molecule_index=molecule_index,
                identifier=identifier,
                details=f"Identifier: {identifier}, Underlying error: {str(e)}",
            ) from e
        except Exception as e:
            # Convert unexpected errors to ANI-1ccx handler errors
            raise DatasetSpecificHandlerError(
                dataset_type="ANI1ccx",
                message=f"Unexpected error during ANI-1ccx validation: {str(e)}",
                operation="molecule_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def get_required_properties(self) -> list[str]:
        """Get ANI-1ccx-specific required properties."""
        required = self.get_common_required_properties()
        required.extend(["ccsd_energy", "atoms", "coordinates"])  # Core ANI-1ccx properties

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
        Get ANI-1ccx identifier keys for molecule creation.

        CRITICAL: ANI-1ccx has NO parseable chemical identifiers.
        The HDF5 structure contains only atomic_numbers and coordinates.
        Returns empty list - coordinate_based strategy will be used.

        Returns:
            Empty list (no identifier keys available)
        """
        return []

    def get_molecular_charge(
        self,
        raw_properties_dict: dict[str, Any],
        atomic_numbers: np.ndarray,
        mol_identifier: str | None = None,
    ) -> int:
        """
        Return molecular charge for ANI-1ccx molecules.

        ANI-1ccx dataset contains only neutral molecules (CHNO organic compounds).
        This is documented in the original dataset publication.

        Args:
            raw_properties_dict: Raw molecule data from NPZ (not used)
            atomic_numbers: Array of atomic numbers (not used)
            mol_identifier: Molecular identifier (not used)

        Returns:
            int: 0 (all ANI-1ccx molecules are neutral)
        """
        # ANI-1ccx contains only neutral molecules
        return 0

    def get_molecule_creation_strategy(self) -> str:
        """
        ANI-1ccx datasets use coordinate_based strategy.

        CRITICAL: ANI-1ccx HDF5 structure contains NO parseable chemical identifiers.
        The data structure only contains:
        - 'atomic_numbers': Shape (Nc, Na) - atomic numbers
        - 'coordinates': Shape (Nc, Na, 3) - 3D positions

        Molecular connectivity must be inferred from 3D coordinates using
        the rdDetermineBonds algorithm. All molecules are neutral (charge=0).

        Returns:
            str: 'coordinate_based'
        """
        return "coordinate_based"

    def process_property_value(
        self, key: str, value: Any, molecule_index: int, identifier: str = "N/A"
    ) -> Any:
        """Process ANI-1ccx-specific property values with exception handling.

        CRITICAL: This method normalizes ALL array properties to ensure they have
        proper native numeric dtypes. When NPZ files store data in object arrays
        (dtype=object), the individual elements may maintain object dtype after
        indexing. This causes two types of downstream failures:

        1. validate_molecular_structure() (validators.py) fails when it calls
           is_value_valid_and_not_nan() which uses np.isfinite() on object arrays.

        2. _ensure_tensor() fails when PyTorch tries to convert object arrays
           to tensors with error: "can't convert np.ndarray of type numpy.object_"

        The fix ensures ALL properties are converted to native numeric dtypes:
        - atoms: int64 (standard atomic number dtype)
        - coordinates, forces, charges, dipole: float64/float32 (standard float dtypes)
        - ccsd_energy, dft_energy: float64 (already native, just validate)

        Evidence: ANI-1ccx preprocessor (ani1ccx_preprocessor.py) stores ragged array
        properties as dtype=object in the NPZ file (lines 489, 490, 503, 506, 509, 512).
        """
        try:
            # Return None values as-is
            if value is None:
                return None

            # =================================================================
            # ATOMS - Ensure native integer dtype
            # =================================================================
            # ANI-1ccx stores atomic numbers as uint8 integers from HDF5.
            # When stored in object arrays and loaded from NPZ, we need to
            # ensure the result has a proper native integer dtype for
            # downstream validation (validate_molecular_structure).
            if key == "atoms":
                arr = np.asarray(value)
                # Convert to int64 (standard atomic number dtype) if not already integer
                if not np.issubdtype(arr.dtype, np.integer):
                    try:
                        arr = arr.astype(np.int64)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"ANI-1ccx molecule {molecule_index}: Could not convert atoms to int64: {e}"
                        )
                        return value  # Return original, let downstream validation handle it
                elif arr.dtype != np.int64:
                    # Normalize to int64 for consistency (uint8 -> int64)
                    arr = arr.astype(np.int64)
                return arr

            # =================================================================
            # COORDINATES - Ensure native float dtype
            # =================================================================
            # ANI-1ccx stores coordinates as float32 from HDF5.
            # Convert to float64 for consistency with other datasets and
            # to ensure np.isfinite() works correctly in downstream validation.
            if key == "coordinates":
                arr = np.asarray(value)
                # Convert to float64 (standard coordinate dtype) if not already floating
                if not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float64)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"ANI-1ccx molecule {molecule_index}: Could not convert coordinates to float64: {e}"
                        )
                        return value
                elif arr.dtype != np.float64:
                    # Normalize to float64 for consistency (float32 -> float64)
                    arr = arr.astype(np.float64)
                return arr

            # =================================================================
            # FORCES - Per-atom property (N, 3) - Ensure native float dtype
            # =================================================================
            # ANI-1ccx stores forces as float32 from HDF5 in object arrays.
            # PyTorch tensor conversion requires native float dtype.
            if key == "forces":
                arr = np.asarray(value)
                # Ensure proper float dtype for tensor conversion
                if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"ANI-1ccx molecule {molecule_index}: Could not convert forces to float32: {e}"
                        )
                        return None
                # Validate after conversion
                if not np.all(np.isfinite(arr)):
                    self.logger.warning(
                        f"ANI-1ccx molecule {molecule_index} has non-finite forces data"
                    )
                    return None
                return arr

            # =================================================================
            # HIRSHFELD_CHARGES - Per-atom property (N,) - Ensure native float dtype
            # =================================================================
            # ANI-1ccx stores Hirshfeld charges as float32 from HDF5 in object arrays.
            if key == "hirshfeld_charges":
                arr = np.asarray(value)
                # Ensure proper float dtype for tensor conversion
                if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"ANI-1ccx molecule {molecule_index}: Could not convert hirshfeld_charges to float32: {e}"
                        )
                        return None
                # Validate after conversion
                if not np.all(np.isfinite(arr)):
                    self.logger.warning(
                        f"ANI-1ccx molecule {molecule_index} has non-finite Hirshfeld charges"
                    )
                    return None
                return arr

            # =================================================================
            # CM5_CHARGES - Per-atom property (N,) - Ensure native float dtype
            # =================================================================
            # ANI-1ccx stores CM5 charges as float32 from HDF5 in object arrays.
            if key == "cm5_charges":
                arr = np.asarray(value)
                # Ensure proper float dtype for tensor conversion
                if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"ANI-1ccx molecule {molecule_index}: Could not convert cm5_charges to float32: {e}"
                        )
                        return None
                # Validate after conversion
                if not np.all(np.isfinite(arr)):
                    self.logger.warning(
                        f"ANI-1ccx molecule {molecule_index} has non-finite CM5 charges"
                    )
                    return None
                return arr

            # =================================================================
            # DIPOLE - Molecular property (3,) - Ensure native float dtype
            # =================================================================
            # ANI-1ccx stores dipole as float32 from HDF5 in object arrays.
            # This property commonly fails tensor conversion with "can't convert
            # np.ndarray of type numpy.object_" error if not normalized.
            if key == "dipole":
                arr = np.asarray(value)
                # Ensure proper float dtype for tensor conversion
                if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"ANI-1ccx molecule {molecule_index}: Could not convert dipole to float32: {e}"
                        )
                        return None
                # Validate after conversion
                if not np.all(np.isfinite(arr)):
                    self.logger.warning(
                        f"ANI-1ccx molecule {molecule_index} has non-finite dipole data"
                    )
                    return None
                return arr

            # =================================================================
            # CCSD_ENERGY - Primary scalar target (already float64 in NPZ)
            # =================================================================
            # CCSD(T)/CBS energy is the primary target for ANI-1ccx.
            # Stored as float64 scalar, not in object array.
            if key == "ccsd_energy":
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = float(value.item())
                    # Check for NaN
                    if np.isnan(value) if isinstance(value, float) else np.any(np.isnan(value)):
                        self.logger.debug(f"ANI-1ccx molecule {molecule_index} has NaN ccsd_energy")
                        return None
                return value

            # =================================================================
            # DFT_ENERGY - Optional comparison property (already float64 in NPZ)
            # =================================================================
            # DFT energy (wb97x_dz.energy) is stored as float64 for comparison.
            if key == "dft_energy":
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = float(value.item())
                    # Check for NaN (DFT energy may be NaN for CC-only molecules)
                    if np.isnan(value) if isinstance(value, float) else np.any(np.isnan(value)):
                        self.logger.debug(
                            f"ANI-1ccx molecule {molecule_index} has NaN dft_energy (expected for some molecules)"
                        )
                        return None
                return value

            # All other properties - pass through unchanged
            return value

        except DatasetSpecificHandlerError:
            # Re-raise ANI-1ccx handler errors
            raise
        except Exception as e:
            # Convert unexpected property processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="ANI1ccx",
                message=f"Unexpected error processing ANI-1ccx property '{key}': {str(e)}",
                operation="property_processing",
                property_name=key,
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _is_valid_property(self, value: Any) -> bool:
        """
        Check if a property value is valid for ANI-1ccx.

        Following WavefunctionDatasetHandler pattern for coordinate_based datasets:
        - Simply checks for None and empty arrays
        - Does NOT call is_value_valid_and_not_nan() which fails on string arrays
        """
        if value is None:
            return False
        return not (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0)

    def get_processing_statistics(
        self, processed_molecules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate ANI-1ccx-specific processing statistics."""
        stats = {
            "dataset_type": "ANI1ccx",
            "total_processed": len(processed_molecules),
            "experimental_setup": self.experimental_setup,
            "primary_energy": "ccsd_energy (CCSD(T)/CBS)",
        }

        atomization_calculations = 0
        molecules_with_forces = 0
        molecules_with_charges = 0
        molecules_with_dft_energy = 0

        for mol_data in processed_molecules:
            if mol_data.get("atomization_energy_calculated"):
                atomization_calculations += 1
            if mol_data.get("forces") is not None:
                molecules_with_forces += 1
            if (
                mol_data.get("hirshfeld_charges") is not None
                or mol_data.get("cm5_charges") is not None
            ):
                molecules_with_charges += 1
            if mol_data.get("dft_energy") is not None:
                molecules_with_dft_energy += 1

        if atomization_calculations > 0:
            stats["atomization_energy_calculations"] = atomization_calculations

        stats["molecules_with_forces"] = molecules_with_forces
        stats["molecules_with_charges"] = molecules_with_charges
        stats["molecules_with_dft_energy"] = molecules_with_dft_energy

        # Add transform usage information if experimental setup exists
        if self.experimental_setup:
            stats["transform_aware_processing"] = True
            stats["experimental_context"] = {
                "setup_name": self.experimental_setup,
                "dataset_type": "ANI1ccx",
                "transform_validation_performed": True,
            }

        return stats

    def get_supported_structural_features(self) -> dict[str, list[str]]:
        """
        ANI-1ccx datasets support ALL structural features.

        ANI-1ccx has optimized 3D geometries (ωB97x/6-31G*) and precomputed charges
        (Hirshfeld and CM5), enabling all structural feature calculations.
        Same as ANI-1x since geometries are from DFT optimization.
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
                # Partial charges (ANI-1ccx has Hirshfeld and CM5 charges from DFT)
                "hirshfeld_charge",
                "cm5_charge",
                "gasteiger_charge",  # Can be calculated from structure
            ],
            "bond": [
                # Bond types
                "bond_type",
                "is_conjugated",
                "is_aromatic",
                "is_in_any_ring",
                "stereo",
                # Geometric features (ANI-1ccx has 3D coordinates)
                "bond_length",
                "bond_length_binned",
            ],
        }

    def get_supported_descriptors(self) -> dict[str, list[str]]:
        """
        Get molecular descriptors supported by ANI-1ccx dataset.

        ANI-1ccx has optimized 3D geometries and can support ALL
        descriptor categories including geometric descriptors.
        """
        return {
            "categories": [
                "constitutional",
                "topological",
                "electronic",
                "geometric",  # ANI-1ccx has 3D coordinates
                "drug_likeness",
                "fragments",
            ],
            "excluded": [],  # ANI-1ccx supports all descriptors
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
                # Geometric (ANI-1ccx has optimized 3D structures)
                "RadiusOfGyration",
                "InertialShapeFactor",
                "Asphericity",
                # Drug-likeness
                "qed",
                "SPS",
            ],
            "requires_3d": True,  # ANI-1ccx provides optimized 3D structures
            "requires_charges": True,
        }

    def get_transform_recommendations(self) -> dict[str, list[str]]:
        """
        Get ANI-1ccx-specific transform recommendations.

        Returns:
            Dict with recommended, avoid, and warning transforms
        """
        recommendations = {
            "recommended": [
                "GCNNorm - for message passing networks",
                "AddSelfLoops - required before GCNNorm",
                "NormalizeFeatures - for stable training",
                "RandomRotate - ANI-1ccx has 3D coordinates",
                "Distance - add distance-based edge features",
            ],
            "avoid": [],
            "warnings": [
                "VirtualNode may need careful handling with precomputed charges",
            ],
        }

        return recommendations

    def _get_dataset_suitable_transforms(self, available_transforms: dict[str, Any]) -> list[str]:
        """
        ANI-1ccx-suitable transforms based on structural and energetic properties.
        """
        suitable = []

        # Geometric transforms - ANI-1ccx has 3D coordinates
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
        Validate transforms for ANI-1ccx dataset compatibility.
        """
        warnings = []

        # ANI-1ccx datasets have 3D coordinates - geometric transforms are relevant
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate", "RandomFlip"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            warnings.append(
                "ANI-1ccx dataset without geometric augmentation - consider adding RandomRotate for invariance"
            )

        # Force data considerations
        if (
            hasattr(self.processing_config, "variable_len_graph_properties")
            and "forces" in self.processing_config.variable_len_graph_properties
            and "RandomRotate" in transform_names
        ):
            warnings.append(
                "ANI-1ccx dataset has forces - geometric transforms will require force rotation"
            )

        # Distance-based transforms
        if "Distance" in transform_names or "Cartesian" in transform_names:
            warnings.append(
                "Distance/Cartesian transforms will add edge attributes - ensure model handles them"
            )

        return warnings

    def _check_transform_incompatibilities(self, transform_names: list[str]) -> list[str]:
        """
        Check for incompatible transform combinations for ANI-1ccx.
        """
        # Per-atom dataset values vs VirtualNode: checked for every dataset (base handler)
        return []

    def _get_transform_recommendations(self, transform_names: list[str]) -> list[str]:
        """
        Get transform recommendations for ANI-1ccx datasets with specific suggestions.
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

        # Geometric transform recommendations for ANI-1ccx
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            recommendations.append(
                "ANI-1ccx 3D structures benefit from geometric augmentation. "
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


# ============================================================================
# + ANI-2x Dataset Handler
# ============================================================================
