# milia_pipeline/handlers/implementations/xxmd.py

"""
xxMD Dataset Handler
====================

Handler for xxMD (Extended Excited-state Molecular Dynamics) datasets with
exception integration and transformation system support.

Extracted following the Handler Module architecture pattern.

Key Features:
- Uses coordinate_based strategy (NO InChI/SMILES identifiers available)
- Coordinates in Angstrom
- Energies in Hartree (converted from eV during preprocessing)
- Forces in Hartree/Angstrom (converted from eV/Angstrom during preprocessing)
- Primary energy target: energy (M06 DFT)
- Has forces as per-atom properties
- All molecules are neutral (charge = 0)

xxMD is a quantum chemistry dataset containing nonadiabatic dynamics trajectories
for 4 photochemically active molecules computed at M06 DFT level of theory.

CRITICAL DIFFERENCE FROM QM9/DFT:
xxMD extended XYZ format contains NO parseable chemical identifiers.
Molecular connectivity must be inferred from 3D coordinates.

Reference: Pengmei, Liu, Shu. Scientific Data 11, 222 (2024)
           DOI: 10.1038/s41597-024-03019-3

Author: MILIA Pipeline Team
Version: 1.0
Date: January 2026
"""

import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from milia_pipeline.config.validators import (
    is_value_valid_and_not_nan,
    validate_molecular_structure,
)
from milia_pipeline.exceptions import (
    DatasetSpecificHandlerError,
    HandlerError,
    HandlerValidationError,
    MoleculeProcessingError,
    PropertyEnrichmentError,
)

# Import from refactored base handler
from milia_pipeline.handlers.base_handler import DatasetHandler
from milia_pipeline.handlers.handler_registry import register_handler

logger = logging.getLogger(__name__)


# Atomic energy references in Hartree for atomization energy calculation
# Reference: Standard isolated atom energies at M06/def2-TZVP level
# These values are approximate and may need refinement for exact M06 functional match
try:
    from milia_pipeline.constants import ATOMIC_ENERGIES_HARTREE, HAR2EV
except ImportError:
    # Fallback values if constants module not available
    ATOMIC_ENERGIES_HARTREE = {
        1: -0.500273,  # H
        6: -37.846772,  # C
        7: -54.583861,  # N
        8: -75.064579,  # O
        16: -398.100442,  # S
    }
    HAR2EV = 27.211386245988


@register_handler
class XXMDDatasetHandler(DatasetHandler):
    """
    Handler for xxMD datasets with exception integration and transformation system support.

    xxMD is a quantum chemistry dataset containing nonadiabatic dynamics trajectories
    for 4 photochemically active molecules:
    - azobenzene (C12H10N2)
    - dithiophene (C8H6S2)
    - malonaldehyde (C3H4O2)
    - stilbene (C14H12)

    Key characteristics:
    - Uses coordinate_based strategy (NO InChI/SMILES identifiers available)
    - Coordinates in Angstrom
    - Energies in Hartree (converted from eV during preprocessing)
    - Forces in Hartree/Angstrom (converted from eV/Angstrom during preprocessing)
    - Primary energy target: energy (M06 DFT)
    - Has forces as per-atom properties
    - All molecules are neutral (charge = 0)
    - Samples broader configuration space (transition states, conical intersections)

    CRITICAL DIFFERENCE FROM QM9/DFT:
    xxMD extended XYZ format contains NO parseable chemical identifiers.
    Molecular connectivity must be inferred from 3D coordinates.

    Reference: Pengmei, Liu, Shu. Scientific Data 11, 222 (2024)
    """

    def get_dataset_type(self) -> str:
        return "XXMD"

    def validate_molecule_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str = "N/A"
    ) -> None:
        """Validate xxMD-specific molecular data with exception handling."""
        try:
            # Validate essential xxMD properties
            essential_props = ["energy", "atoms", "coordinates"]
            missing_props = []

            for prop in essential_props:
                if not self._is_valid_property(raw_properties_dict.get(prop)):
                    missing_props.append(prop)

            if missing_props:
                raise HandlerValidationError(
                    message=f"Missing required xxMD properties: {missing_props}",
                    handler_type="XXMD",
                    validation_type="essential_properties",
                    failed_validations=[f"Missing {prop}" for prop in missing_props],
                    molecule_index=molecule_index,
                    details="xxMD molecules must have energy, atoms, and coordinates",
                )

            # Validate structural consistency
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(atoms, coordinates, molecule_index, identifier)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="XXMD",
                        message=f"xxMD molecular structure validation failed for molecule {molecule_index}",
                        operation="structure_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"Identifier: {identifier}, Atoms: {len(atoms) if atoms else 0}, "
                        f"Coords: {len(coordinates) if coordinates else 0}, "
                        f"Error: {str(e)}",
                    ) from e

            # Validate energy (xxMD energies are typically negative in Hartree after conversion)
            energy = raw_properties_dict.get("energy")
            if energy is not None and isinstance(energy, (int, float, np.number)):
                if energy > 0:
                    self.logger.warning(
                        f"xxMD molecule {molecule_index} has positive energy: {energy}"
                    )

        except (HandlerError, DatasetSpecificHandlerError):
            # Re-raise handler-specific errors
            raise
        except MoleculeProcessingError as e:
            # Convert molecule processing errors to xxMD handler validation errors
            raise DatasetSpecificHandlerError(
                dataset_type="XXMD",
                message=f"xxMD validation failed for molecule {molecule_index}: {e.message}",
                operation="molecule_validation",
                molecule_index=molecule_index,
                identifier=identifier,
                details=f"Identifier: {identifier}, Underlying error: {str(e)}",
            ) from e
        except Exception as e:
            # Convert unexpected errors to xxMD handler errors
            raise DatasetSpecificHandlerError(
                dataset_type="XXMD",
                message=f"Unexpected error during xxMD validation: {str(e)}",
                operation="molecule_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def get_required_properties(self) -> list[str]:
        """Get xxMD-specific required properties."""
        required = self.get_common_required_properties()
        required.extend(["energy", "atoms", "coordinates"])  # Core xxMD properties

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
        Get xxMD identifier keys for molecule creation.

        CRITICAL: xxMD has NO parseable chemical identifiers.
        The extended XYZ format contains only atomic positions and species.
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
        Return molecular charge for xxMD molecules.

        xxMD dataset contains only neutral molecules (photochemically active organic compounds).
        All four molecules (azobenzene, dithiophene, malonaldehyde, stilbene) are neutral.

        Args:
            raw_properties_dict: Raw molecule data from NPZ (not used)
            atomic_numbers: Array of atomic numbers (not used)
            mol_identifier: Molecular identifier (not used)

        Returns:
            int: 0 (all xxMD molecules are neutral)
        """
        # xxMD contains only neutral molecules
        return 0

    def get_molecule_creation_strategy(self) -> str:
        """
        xxMD datasets use coordinate_based strategy.

        CRITICAL: xxMD extended XYZ format contains NO parseable chemical identifiers.
        The data structure only contains:
        - Atomic species/numbers
        - 3D atomic positions (coordinates)
        - Energy and forces

        Molecular connectivity must be inferred from 3D coordinates using
        the rdDetermineBonds algorithm. All molecules are neutral (charge=0).

        Returns:
            str: 'coordinate_based'
        """
        return "coordinate_based"

    def process_property_value(
        self, key: str, value: Any, molecule_index: int, identifier: str = "N/A"
    ) -> Any:
        """Process xxMD-specific property values with exception handling.

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
        - coordinates, forces: float64/float32 (standard float dtypes)
        - energy: float64 (already native, just validate)

        Evidence: Preprocessor stores ragged array properties as dtype=object
        in the NPZ file (similar pattern to ANI-1x, rMD17).
        """
        try:
            # Return None values as-is
            if value is None:
                return None

            # =================================================================
            # ATOMS - Ensure native integer dtype
            # =================================================================
            # xxMD stores atomic numbers as integers from ASE.
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
                            f"xxMD molecule {molecule_index}: Could not convert atoms to int64: {e}"
                        )
                        return value  # Return original, let downstream validation handle it
                elif arr.dtype != np.int64:
                    # Normalize to int64 for consistency (uint8 -> int64)
                    arr = arr.astype(np.int64)
                return arr

            # =================================================================
            # COORDINATES - Ensure native float dtype
            # =================================================================
            # xxMD stores coordinates as float from ASE extended XYZ.
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
                            f"xxMD molecule {molecule_index}: Could not convert coordinates to float64: {e}"
                        )
                        return value
                elif arr.dtype != np.float64:
                    # Normalize to float64 for consistency (float32 -> float64)
                    arr = arr.astype(np.float64)
                return arr

            # =================================================================
            # FORCES - Per-atom property (N, 3) - Ensure native float dtype
            # =================================================================
            # xxMD stores forces as float from ASE extended XYZ in object arrays.
            # PyTorch tensor conversion requires native float dtype.
            if key == "forces":
                arr = np.asarray(value)
                # Ensure proper float dtype for tensor conversion
                if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"xxMD molecule {molecule_index}: Could not convert forces to float32: {e}"
                        )
                        return None
                # Validate after conversion
                if not np.all(np.isfinite(arr)):
                    self.logger.warning(
                        f"xxMD molecule {molecule_index} has non-finite forces data"
                    )
                    return None
                return arr

            # =================================================================
            # ENERGY - Scalar property - Pass through (already float64 in NPZ)
            # =================================================================
            # Energy is stored as float64 scalar, not in object array.
            # It doesn't need conversion but validate it.
            if key == "energy":
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = float(value.item())
                    # Check for NaN
                    if np.isnan(value) if isinstance(value, float) else np.any(np.isnan(value)):
                        self.logger.debug(f"xxMD molecule {molecule_index} has NaN energy")
                        return None
                return value

            # All other properties - pass through unchanged
            return value

        except DatasetSpecificHandlerError:
            # Re-raise xxMD handler errors
            raise
        except Exception as e:
            # Convert unexpected property processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="XXMD",
                message=f"Unexpected error processing xxMD property '{key}': {str(e)}",
                operation="property_processing",
                property_name=key,
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def enrich_pyg_data(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str = "N/A",
    ) -> Data:
        """xxMD-specific PyG data enrichment with exception handling."""
        try:
            # Set dataset type
            pyg_data.dataset_type = "XXMD"

            # Ensure num_nodes is set properly
            if not hasattr(pyg_data, "num_nodes") or pyg_data.num_nodes == 0:
                pyg_data.num_nodes = (
                    pyg_data.z.size(0) if hasattr(pyg_data, "z") and pyg_data.z is not None else 0
                )

            if pyg_data.num_nodes == 0:
                raise DatasetSpecificHandlerError(
                    dataset_type="XXMD",
                    message="xxMD molecule has 0 nodes, cannot proceed with enrichment",
                    operation="enrich_pyg_data",
                    details="No atoms available for processing",
                )

            # 1. Add scalar graph targets (energy)
            self._add_scalar_targets_internal(
                pyg_data, raw_properties_dict, molecule_index, identifier
            )

            # 2. Add vector graph properties (none for xxMD currently)
            if self.processing_config.vector_graph_properties:
                self._add_vector_properties_internal(
                    pyg_data, raw_properties_dict, molecule_index, identifier
                )

            # 3. Add per-atom properties (forces)
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

            self.logger.debug(f"xxMD molecule {molecule_index}: Enrichment completed")
            return pyg_data

        except (PropertyEnrichmentError, DatasetSpecificHandlerError):
            raise
        except Exception as e:
            # Convert unexpected errors to xxMD handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="XXMD",
                message=f"xxMD enrichment failed: {str(e)}",
                operation="enrich_pyg_data",
                details=f"Molecule {molecule_index}, Error during xxMD-specific enrichment",
            ) from e

    def _add_scalar_targets_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal xxMD-specific scalar targets implementation."""
        try:
            if not self.processing_config.scalar_graph_targets:
                return

            collected_targets = []
            for key in self.processing_config.scalar_graph_targets:
                try:
                    value = raw_properties_dict.get(key)

                    if not is_value_valid_and_not_nan(value):
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"Missing or invalid xxMD scalar target '{key}'",
                            detail=f"Value: {value}",
                        )

                    # Handle conversion to float
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            val_to_add = float(value.item())
                        else:
                            raise PropertyEnrichmentError(
                                molecule_index=molecule_index,
                                inchi=identifier,
                                property_name=key,
                                reason=f"xxMD scalar target '{key}' is not a single value",
                                detail=f"Shape: {value.shape}",
                            )
                    elif isinstance(value, (int, float, np.number)):
                        val_to_add = float(value)
                    else:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"xxMD scalar target '{key}' has unexpected type",
                            detail=f"Type: {type(value)}",
                        )

                    collected_targets.append(val_to_add)

                except PropertyEnrichmentError:
                    raise
                except Exception as e:
                    raise PropertyEnrichmentError(
                        molecule_index=molecule_index,
                        inchi=identifier,
                        property_name=key,
                        reason=f"Critical error processing xxMD scalar target '{key}'",
                        detail=str(e),
                    ) from e

            if collected_targets:
                pyg_data.y = self._ensure_tensor(
                    collected_targets,
                    torch.float32,
                    "xxmd_scalar_targets",
                    molecule_index,
                    identifier,
                )

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to xxMD handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="XXMD",
                message=f"xxMD scalar targets processing failed: {str(e)}",
                operation="add_scalar_targets",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _add_vector_properties_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal xxMD-specific vector properties implementation."""
        try:
            for prop_key in self.processing_config.vector_graph_properties:
                try:
                    value = raw_properties_dict.get(prop_key)

                    if not is_value_valid_and_not_nan(value):
                        self.logger.debug(
                            f"xxMD molecule {molecule_index}: Skipping vector property '{prop_key}' - not available"
                        )
                        continue

                    # Convert to numpy array if needed
                    if isinstance(value, (list, tuple)):
                        value = np.asarray(value, dtype=np.float32)

                    if not isinstance(value, np.ndarray) or value.ndim != 1:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=prop_key,
                            reason=f"xxMD vector property '{prop_key}' is not a 1D array",
                            detail=f"Type: {type(value)}, Dims: {getattr(value, 'ndim', 'N/A')}",
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
                        reason=f"Error processing xxMD vector property '{prop_key}'",
                        detail=str(e),
                    ) from e

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            raise DatasetSpecificHandlerError(
                dataset_type="XXMD",
                message=f"xxMD vector properties processing failed: {str(e)}",
                operation="add_vector_properties",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _add_variable_length_properties_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal xxMD-specific variable-length property implementation."""
        try:
            if not self.processing_config.variable_len_graph_properties:
                return

            for key in self.processing_config.variable_len_graph_properties:
                try:
                    value = raw_properties_dict.get(key)

                    if not is_value_valid_and_not_nan(value):
                        self.logger.debug(
                            f"xxMD molecule {molecule_index}: Skipping variable-length property '{key}' - not available"
                        )
                        continue

                    # Convert to tensor with appropriate dtype
                    dtype = torch.float32
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
                        reason=f"Error processing xxMD variable-length property '{key}'",
                        detail=str(e),
                    ) from e

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            raise DatasetSpecificHandlerError(
                dataset_type="XXMD",
                message=f"xxMD variable-length properties processing failed: {str(e)}",
                operation="add_variable_length_properties",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _calculate_atomization_energy_internal(
        self,
        raw_properties_dict: dict[str, Any],
        pyg_data: Data,
        molecule_index: int,
        identifier: str,
    ) -> float | None:
        """
        Internal xxMD-specific atomization energy calculation.

        Uses energy (M06 DFT, converted to Hartree) as base for atomization energy calculation.
        """
        try:
            if not self.processing_config.calculate_atomization_energy_from:
                return None

            base_energy_key = self.processing_config.calculate_atomization_energy_from
            base_energy_raw = raw_properties_dict.get(base_energy_key)

            if not is_value_valid_and_not_nan(base_energy_raw):
                self.logger.warning(
                    f"xxMD molecule {molecule_index} missing {base_energy_key} for atomization energy"
                )
                return None

            if not hasattr(pyg_data, "z") or pyg_data.z is None:
                self.logger.warning(
                    f"xxMD molecule {molecule_index} missing atomic numbers for atomization energy"
                )
                return None

            # Convert to float - energy is already in Hartree after preprocessing
            if isinstance(base_energy_raw, np.ndarray) and base_energy_raw.size == 1:
                base_energy_hartree = float(base_energy_raw.item())
            elif isinstance(base_energy_raw, (int, float, np.number)):
                base_energy_hartree = float(base_energy_raw)
            else:
                self.logger.warning(
                    f"xxMD molecule {molecule_index}: Cannot convert {base_energy_key} to float"
                )
                return None

            # Calculate atomization energy
            if HAR2EV is None or not ATOMIC_ENERGIES_HARTREE:
                self.logger.warning(
                    f"xxMD molecule {molecule_index}: Missing atomic energies for atomization calculation"
                )
                return None

            # Sum atomic energies (in Hartree)
            sum_atomic_energies_hartree = 0.0
            for atomic_num in pyg_data.z.tolist():
                atomic_energy = ATOMIC_ENERGIES_HARTREE.get(atomic_num)
                if atomic_energy is None:
                    self.logger.warning(
                        f"xxMD molecule {molecule_index}: Missing atomic energy for element {atomic_num}"
                    )
                    return None
                sum_atomic_energies_hartree += atomic_energy

            # Calculate atomization energy in Hartree, then convert to eV
            atomization_energy_hartree = base_energy_hartree - sum_atomic_energies_hartree
            atomization_energy_eV = atomization_energy_hartree * HAR2EV

            self.logger.debug(
                f"xxMD molecule {molecule_index} atomization energy: {atomization_energy_eV:.4f} eV"
            )
            return atomization_energy_eV

        except Exception as e:
            self.logger.error(
                f"Error calculating xxMD atomization energy for molecule {molecule_index}: {e}"
            )
            return None

    def _ensure_tensor(
        self, value: Any, dtype: torch.dtype, key: str, molecule_index: int, identifier: str
    ) -> torch.Tensor:
        """Ensure value is converted to a PyTorch tensor with proper dtype."""
        try:
            if isinstance(value, torch.Tensor):
                return value.to(dtype)
            elif isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
                return torch.tensor(value, dtype=dtype)
            elif isinstance(value, (int, float, np.number)):
                return torch.tensor([value], dtype=dtype)
            else:
                raise ValueError(f"Cannot convert {type(value)} to tensor")
        except Exception as e:
            raise DatasetSpecificHandlerError(
                dataset_type="XXMD",
                message=f"Failed to convert '{key}' to tensor: {str(e)}",
                operation="tensor_conversion",
                property_name=key,
                details=f"Molecule {molecule_index}, Value type: {type(value)}",
            ) from e

    def _is_valid_property(self, value: Any) -> bool:
        """
        Check if a property value is valid for xxMD.

        Following coordinate_based dataset patterns (ANI-1x, rMD17):
        - Simply checks for None and empty arrays
        - Does NOT call is_value_valid_and_not_nan() which fails on string arrays

        Evidence: ANI1xDatasetHandler._is_valid_property pattern
        """
        if value is None:
            return False
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0:
            return False
        return True

    def get_processing_statistics(
        self, processed_molecules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate xxMD-specific processing statistics."""
        stats = {
            "dataset_type": "XXMD",
            "total_processed": len(processed_molecules),
            "experimental_setup": self.experimental_setup,
        }

        atomization_calculations = 0
        molecules_with_forces = 0
        molecule_counts = {}

        for mol_data in processed_molecules:
            if mol_data.get("atomization_energy_calculated"):
                atomization_calculations += 1
            if mol_data.get("forces") is not None:
                molecules_with_forces += 1

            # Count by molecule type if available
            mol_name = mol_data.get("molecule_name", "unknown")
            molecule_counts[mol_name] = molecule_counts.get(mol_name, 0) + 1

        if atomization_calculations > 0:
            stats["atomization_energy_calculations"] = atomization_calculations

        stats["molecules_with_forces"] = molecules_with_forces
        stats["molecule_type_distribution"] = molecule_counts

        # Add transform usage information if experimental setup exists
        if self.experimental_setup:
            stats["transform_aware_processing"] = True
            stats["experimental_context"] = {
                "setup_name": self.experimental_setup,
                "dataset_type": "XXMD",
                "transform_validation_performed": True,
            }

        return stats

    def get_supported_structural_features(self) -> dict[str, list[str]]:
        """
        xxMD datasets support structural features derived from 3D coordinates.

        xxMD has optimized/sampled 3D geometries from nonadiabatic dynamics,
        enabling structural feature calculations from coordinates.
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
                # Charges (computed from structure, not precomputed)
                "gasteiger_charge",
            ],
            "bond": [
                # Bond types
                "bond_type",
                "is_conjugated",
                "is_aromatic",
                "is_in_any_ring",
                "stereo",
                # Geometric features (xxMD has 3D coordinates)
                "bond_length",
                "bond_length_binned",
            ],
        }

    def get_supported_descriptors(self) -> dict[str, list[str]]:
        """
        Get molecular descriptors supported by xxMD dataset.

        xxMD has 3D geometries and can support geometric descriptors.
        """
        return {
            "categories": [
                "constitutional",
                "topological",
                "electronic",
                "geometric",  # xxMD has 3D coordinates
                "drug_likeness",
                "fragments",
            ],
            "excluded": [],  # xxMD supports all descriptors
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
                # Geometric (xxMD has 3D structures from MD)
                "RadiusOfGyration",
                "InertialShapeFactor",
                "Asphericity",
                # Drug-likeness
                "qed",
                "SPS",
            ],
            "requires_3d": True,  # xxMD provides 3D structures
            "requires_charges": False,  # No precomputed charges
        }

    def get_transform_recommendations(self) -> dict[str, list[str]]:
        """
        Get xxMD-specific transform recommendations.

        Returns:
            Dict with recommended, avoid, and warning transforms
        """
        recommendations = {
            "recommended": [
                "GCNNorm - for message passing networks",
                "AddSelfLoops - required before GCNNorm",
                "NormalizeFeatures - for stable training",
                "RandomRotate - xxMD has 3D coordinates",
                "Distance - add distance-based edge features",
            ],
            "avoid": [],
            "warnings": [
                "xxMD samples broader configuration space (transition states) - "
                "model may need to handle diverse geometries",
            ],
        }

        return recommendations

    def _get_dataset_suitable_transforms(self, available_transforms: dict[str, Any]) -> list[str]:
        """
        xxMD-suitable transforms based on structural and energetic properties.

        Args:
            available_transforms: Dict of all available transforms

        Returns:
            List of transform names suitable for xxMD datasets
        """
        suitable = []

        # Geometric transforms - xxMD has 3D coordinates
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
        Validate transforms for xxMD dataset compatibility.

        Args:
            transform_names: List of transform class names

        Returns:
            List of warning messages
        """
        warnings = []

        # xxMD datasets have 3D coordinates - geometric transforms are relevant
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate", "RandomFlip"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            warnings.append(
                "xxMD dataset without geometric augmentation - consider adding RandomRotate for invariance"
            )

        # Force data considerations
        if hasattr(self.processing_config, "variable_len_graph_properties"):
            if "forces" in self.processing_config.variable_len_graph_properties:
                if "RandomRotate" in transform_names:
                    warnings.append(
                        "xxMD dataset has forces - geometric transforms will require force rotation"
                    )

        # Distance-based transforms
        if "Distance" in transform_names or "Cartesian" in transform_names:
            warnings.append(
                "Distance/Cartesian transforms will add edge attributes - ensure model handles them"
            )

        return warnings

    def _check_transform_incompatibilities(self, transform_names: list[str]) -> list[str]:
        """
        Check for incompatible transform combinations for xxMD.

        Args:
            transform_names: List of transform class names

        Returns:
            List of error messages (empty if all compatible)
        """
        errors = []

        # xxMD doesn't have precomputed charges, so VirtualNode is compatible
        # (unlike ANI-1x which has precomputed charges)

        return errors

    def _get_transform_recommendations(self, transform_names: list[str]) -> list[str]:
        """
        Get transform recommendations for xxMD datasets with specific suggestions.

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

        # Geometric transform recommendations for xxMD
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            recommendations.append(
                "xxMD 3D structures benefit from geometric augmentation. "
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
