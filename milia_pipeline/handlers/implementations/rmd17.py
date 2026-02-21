# milia_pipeline/handlers/implementations/rmd17.py

"""
rMD17 Dataset Handler
=====================

Handler for rMD17 (Revised MD17) datasets with exception integration
and transformation system support.

Extracted from dataset_handlers.py as part of the Handler Module Refactoring.

Key Features:
- Uses coordinate_based strategy (NO InChI/SMILES identifiers available)
- Coordinates in Angstrom
- Energies in kcal/mol (different from ANI/QM9 which use Hartree!)
- Primary energy target: energy
- Has forces (kcal/mol/Angstrom)
- Single molecule per dataset file (e.g., aspirin, benzene, ethanol)

rMD17 is a revised version of the MD17 dataset with corrected energies
and forces from DFT calculations.

CRITICAL DIFFERENCES:
- Energy units: kcal/mol (NOT Hartree like ANI/QM9/DFT)
- Force units: kcal/mol/Angstrom
- Single molecule type per NPZ file
- Molecular dynamics trajectory data (conformations of same molecule)

Reference: Christensen & von Lilienfeld, Machine Learning: Science and Technology 1, 045018 (2020)
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
    HandlerError,
    HandlerValidationError,
    MoleculeProcessingError,
    PropertyEnrichmentError,
)

# Import from refactored base handler
from milia_pipeline.handlers.base_handler import DatasetHandler
from milia_pipeline.handlers.handler_registry import register_handler

logger = logging.getLogger(__name__)


@register_handler
class RMD17DatasetHandler(DatasetHandler):
    """
    Handler for rMD17 (Revised MD17) datasets with exception integration and
    transformation system support.

    rMD17 is a quantum chemistry dataset containing ~100,000 conformations for
    each of 10 small organic molecules computed at PBE/def2-SVP level using ORCA
    with very tight SCF convergence and dense integration grid.

    Key characteristics:
    - Uses coordinate_based strategy (NO InChI/SMILES identifiers available)
    - Coordinates in Angstrom
    - Energies stored in kcal/mol (converted to Hartree during preprocessing)
    - Forces in kcal/mol/Angstrom (converted to Hartree/Angstrom during preprocessing)
    - All molecules are neutral (charge = 0)
    - Dataset is practically noise-free (unlike original MD17)

    CRITICAL TRAINING WARNING:
    - DO NOT train on more than 1000 samples due to autocorrelation in MD time-series
    - Data from MD simulations at 500K are NOT independent samples

    CRITICAL DIFFERENCE FROM ANI-1x:
    - rMD17: 10 specific molecules, ~100k conformations each, PBE/def2-SVP
    - ANI-1x: ~5M conformers across many molecules, ωB97x/6-31G*

    Reference: Christensen & von Lilienfeld, Mach. Learn.: Sci. Technol. 1, 045018 (2020)
    """

    def get_dataset_type(self) -> str:
        return "RMD17"

    def validate_molecule_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str = "N/A"
    ) -> None:
        """Validate rMD17-specific molecular data with exception handling."""
        try:
            # Validate essential rMD17 properties
            essential_props = ["energies", "atoms", "coordinates"]
            missing_props = []

            for prop in essential_props:
                if not self._is_valid_property(raw_properties_dict.get(prop)):
                    missing_props.append(prop)

            if missing_props:
                raise HandlerValidationError(
                    message=f"Missing required rMD17 properties: {missing_props}",
                    handler_type="RMD17",
                    validation_type="essential_properties",
                    failed_validations=[f"Missing {prop}" for prop in missing_props],
                    molecule_index=molecule_index,
                    details="rMD17 molecules must have energies, atoms, and coordinates",
                )

            # Validate structural consistency
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(atoms, coordinates, molecule_index, identifier)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="RMD17",
                        message=f"rMD17 molecular structure validation failed for molecule {molecule_index}",
                        operation="structure_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"Identifier: {identifier}, Atoms: {len(atoms) if atoms else 0}, "
                        f"Coords: {len(coordinates) if coordinates else 0}, "
                        f"Error: {str(e)}",
                    ) from e

            # Validate energy (rMD17 energies are in Hartree after preprocessing)
            energy = raw_properties_dict.get("energies")
            if energy is not None and isinstance(energy, (int, float, np.number)) and energy > 0:
                # rMD17 energies should be negative (total molecular energy)
                self.logger.warning(
                    f"rMD17 molecule {molecule_index} has positive energy: {energy}"
                )

        except (HandlerError, DatasetSpecificHandlerError):
            # Re-raise handler-specific errors
            raise
        except MoleculeProcessingError as e:
            # Convert molecule processing errors to rMD17 handler validation errors
            raise DatasetSpecificHandlerError(
                dataset_type="RMD17",
                message=f"rMD17 validation failed for molecule {molecule_index}: {e.message}",
                operation="molecule_validation",
                molecule_index=molecule_index,
                identifier=identifier,
                details=f"Identifier: {identifier}, Underlying error: {str(e)}",
            ) from e
        except Exception as e:
            # Convert unexpected errors to rMD17 handler errors
            raise DatasetSpecificHandlerError(
                dataset_type="RMD17",
                message=f"Unexpected error during rMD17 validation: {str(e)}",
                operation="molecule_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def get_required_properties(self) -> list[str]:
        """Get rMD17-specific required properties."""
        required = self.get_common_required_properties()
        required.extend(["energies", "atoms", "coordinates"])  # Core rMD17 properties

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
        Get rMD17 identifier keys for molecule creation.

        CRITICAL: rMD17 has NO parseable chemical identifiers.
        The NPZ structure contains only nuclear_charges and coordinates.
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
        Return molecular charge for rMD17 molecules.

        rMD17 dataset contains only neutral molecules (small organic molecules).
        This is documented in the original dataset publication.

        Args:
            raw_properties_dict: Raw molecule data from NPZ (not used)
            atomic_numbers: Array of atomic numbers (not used)
            mol_identifier: Molecular identifier (not used)

        Returns:
            int: 0 (all rMD17 molecules are neutral)
        """
        # rMD17 contains only neutral molecules
        return 0

    def get_molecule_creation_strategy(self) -> str:
        """
        rMD17 datasets use coordinate_based strategy.

        CRITICAL: rMD17 NPZ structure contains NO parseable chemical identifiers.
        The data structure only contains:
        - 'nuclear_charges': Atomic numbers (constant per molecule)
        - 'coords': Shape (N_conf, N_atoms, 3) - 3D positions

        Molecular connectivity must be inferred from 3D coordinates using
        the rdDetermineBonds algorithm. All molecules are neutral (charge=0).

        Returns:
            str: 'coordinate_based'
        """
        return "coordinate_based"

    def process_property_value(
        self, key: str, value: Any, molecule_index: int, identifier: str = "N/A"
    ) -> Any:
        """Process rMD17-specific property values with exception handling.

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
        - energies: float64 (already native, just validate)

        Evidence: ANI-1x preprocessor stores ragged array properties as dtype=object
        in NPZ files; same pattern applies to rMD17 preprocessor output.
        """
        try:
            # Return None values as-is
            if value is None:
                return None

            # =================================================================
            # ATOMS - Ensure native integer dtype
            # =================================================================
            if key == "atoms":
                arr = np.asarray(value)
                if not np.issubdtype(arr.dtype, np.integer):
                    try:
                        arr = arr.astype(np.int64)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"rMD17 molecule {molecule_index}: Could not convert atoms to int64: {e}"
                        )
                        return value
                elif arr.dtype != np.int64:
                    arr = arr.astype(np.int64)
                return arr

            # =================================================================
            # COORDINATES - Ensure native float dtype
            # =================================================================
            if key == "coordinates":
                arr = np.asarray(value)
                if not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float64)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"rMD17 molecule {molecule_index}: Could not convert coordinates to float64: {e}"
                        )
                        return value
                elif arr.dtype != np.float64:
                    arr = arr.astype(np.float64)
                return arr

            # =================================================================
            # FORCES - Per-atom property (N, 3) - Ensure native float dtype
            # =================================================================
            if key == "forces":
                arr = np.asarray(value)
                if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"rMD17 molecule {molecule_index}: Could not convert forces to float32: {e}"
                        )
                        return None
                if not np.all(np.isfinite(arr)):
                    self.logger.warning(
                        f"rMD17 molecule {molecule_index} has non-finite forces data"
                    )
                    return None
                return arr

            # =================================================================
            # ENERGIES - Scalar property
            # =================================================================
            if key == "energies":
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = float(value.item())
                    if np.isnan(value) if isinstance(value, float) else np.any(np.isnan(value)):
                        self.logger.debug(f"rMD17 molecule {molecule_index} has NaN energy")
                        return None
                return value

            # =================================================================
            # OLD_ENERGIES - Optional comparison property
            # =================================================================
            if key == "old_energies":
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = float(value.item())
                    if np.isnan(value) if isinstance(value, float) else np.any(np.isnan(value)):
                        self.logger.debug(f"rMD17 molecule {molecule_index} has NaN old_energies")
                        return None
                return value

            # =================================================================
            # OLD_FORCES - Optional comparison property from original MD17
            # =================================================================
            if key == "old_forces":
                arr = np.asarray(value)
                if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"rMD17 molecule {molecule_index}: Could not convert old_forces to float32: {e}"
                        )
                        return None
                if not np.all(np.isfinite(arr)):
                    self.logger.warning(
                        f"rMD17 molecule {molecule_index} has non-finite old_forces data"
                    )
                    return None
                return arr

            # =================================================================
            # MOLECULE_NAME - String property, pass through
            # =================================================================
            if key == "molecule_name":
                return value

            # All other properties - pass through unchanged
            return value

        except DatasetSpecificHandlerError:
            # Re-raise rMD17 handler errors
            raise
        except Exception as e:
            # Convert unexpected property processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="RMD17",
                message=f"Unexpected error processing rMD17 property '{key}': {str(e)}",
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
        """rMD17-specific PyG data enrichment with exception handling."""
        try:
            # Set dataset type
            pyg_data.dataset_type = "RMD17"

            # Ensure num_nodes is set properly
            if not hasattr(pyg_data, "num_nodes") or pyg_data.num_nodes == 0:
                pyg_data.num_nodes = (
                    pyg_data.z.size(0) if hasattr(pyg_data, "z") and pyg_data.z is not None else 0
                )

            if pyg_data.num_nodes == 0:
                raise DatasetSpecificHandlerError(
                    dataset_type="RMD17",
                    message="rMD17 molecule has 0 nodes, cannot proceed with enrichment",
                    operation="enrich_pyg_data",
                    details="No atoms available for processing",
                )

            # 1. Add scalar graph targets (energies)
            self._add_scalar_targets_internal(
                pyg_data, raw_properties_dict, molecule_index, identifier
            )

            # 2. Add vector graph properties (if any configured)
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

            self.logger.debug(f"rMD17 molecule {molecule_index}: Enrichment completed")
            return pyg_data

        except (PropertyEnrichmentError, DatasetSpecificHandlerError):
            raise
        except Exception as e:
            # Convert unexpected errors to rMD17 handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="RMD17",
                message=f"rMD17 enrichment failed: {str(e)}",
                operation="enrich_pyg_data",
                details=f"Molecule {molecule_index}, Error during rMD17-specific enrichment",
            ) from e

    def _add_scalar_targets_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal rMD17-specific scalar targets implementation."""
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
                            reason=f"Missing or invalid rMD17 scalar target '{key}'",
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
                                reason=f"rMD17 scalar target '{key}' is not a single value",
                                detail=f"Shape: {value.shape}",
                            )
                    elif isinstance(value, (int, float, np.number)):
                        val_to_add = float(value)
                    else:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"rMD17 scalar target '{key}' has unexpected type",
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
                        reason=f"Critical error processing rMD17 scalar target '{key}'",
                        detail=str(e),
                    ) from e

            if collected_targets:
                pyg_data.y = self._ensure_tensor(
                    collected_targets,
                    torch.float32,
                    "rmd17_scalar_targets",
                    molecule_index,
                    identifier,
                )

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to rMD17 handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="RMD17",
                message=f"rMD17 scalar targets processing failed: {str(e)}",
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
        """Internal rMD17-specific vector properties implementation."""
        try:
            for prop_key in self.processing_config.vector_graph_properties:
                try:
                    value = raw_properties_dict.get(prop_key)

                    if not is_value_valid_and_not_nan(value):
                        self.logger.debug(
                            f"rMD17 molecule {molecule_index}: Skipping vector property '{prop_key}' - not available"
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
                            reason=f"rMD17 vector property '{prop_key}' is not a 1D array",
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
                        reason=f"Error processing rMD17 vector property '{prop_key}'",
                        detail=str(e),
                    ) from e

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to rMD17 handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="RMD17",
                message=f"rMD17 vector properties processing failed: {str(e)}",
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
        """Internal rMD17-specific variable-length property implementation (forces)."""
        try:
            if not self.processing_config.variable_len_graph_properties:
                return

            for key in self.processing_config.variable_len_graph_properties:
                try:
                    value = raw_properties_dict.get(key)

                    if not is_value_valid_and_not_nan(value):
                        self.logger.debug(
                            f"rMD17 molecule {molecule_index}: Skipping variable-length property '{key}' - not available"
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
                        reason=f"Error processing rMD17 variable-length property '{key}'",
                        detail=str(e),
                    ) from e

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to rMD17 handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="RMD17",
                message=f"rMD17 variable-length properties processing failed: {str(e)}",
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
        Internal rMD17-specific atomization energy calculation.

        Uses energies (already in Hartree after preprocessing) as base for
        atomization energy calculation.
        """
        try:
            if not self.processing_config.calculate_atomization_energy_from:
                return None

            base_energy_key = self.processing_config.calculate_atomization_energy_from
            base_energy_raw = raw_properties_dict.get(base_energy_key)

            if not is_value_valid_and_not_nan(base_energy_raw):
                self.logger.warning(
                    f"rMD17 molecule {molecule_index} missing {base_energy_key} for atomization energy"
                )
                return None

            if not hasattr(pyg_data, "z") or pyg_data.z is None:
                self.logger.warning(
                    f"rMD17 molecule {molecule_index} missing atomic numbers for atomization energy"
                )
                return None

            # Convert to float - energies are already in Hartree after preprocessing
            if isinstance(base_energy_raw, np.ndarray) and base_energy_raw.size == 1:
                base_energy_hartree = float(base_energy_raw.item())
            elif isinstance(base_energy_raw, (int, float, np.number)):
                base_energy_hartree = float(base_energy_raw)
            else:
                self.logger.warning(
                    f"rMD17 molecule {molecule_index}: Cannot convert {base_energy_key} to float"
                )
                return None

            # Calculate atomization energy
            if HAR2EV is None or not ATOMIC_ENERGIES_HARTREE:
                self.logger.warning(
                    f"rMD17 molecule {molecule_index}: Missing atomic energies for atomization calculation"
                )
                return None

            # Sum atomic energies (in Hartree)
            sum_atomic_energies_hartree = 0.0
            for atomic_num in pyg_data.z.tolist():
                atomic_energy = ATOMIC_ENERGIES_HARTREE.get(atomic_num)
                if atomic_energy is None:
                    self.logger.warning(
                        f"rMD17 molecule {molecule_index}: Missing atomic energy for element {atomic_num}"
                    )
                    return None
                sum_atomic_energies_hartree += atomic_energy

            # Calculate atomization energy in Hartree, then convert to eV
            atomization_energy_hartree = base_energy_hartree - sum_atomic_energies_hartree
            atomization_energy_eV = atomization_energy_hartree * HAR2EV

            self.logger.debug(
                f"rMD17 molecule {molecule_index} atomization energy: {atomization_energy_eV:.4f} eV (from {base_energy_key})"
            )
            return atomization_energy_eV

        except Exception as e:
            self.logger.error(
                f"Error calculating rMD17 atomization energy for molecule {molecule_index}: {e}"
            )
            return None

    def _ensure_tensor(
        self, value: Any, dtype: torch.dtype, key: str, molecule_index: int, identifier: str
    ) -> torch.Tensor:
        """Ensure value is converted to a PyTorch tensor with proper dtype."""
        try:
            if isinstance(value, torch.Tensor):
                return value.to(dtype)
            elif isinstance(value, (np.ndarray, list, tuple)):
                return torch.tensor(value, dtype=dtype)
            elif isinstance(value, (int, float, np.number)):
                return torch.tensor([value], dtype=dtype)
            else:
                raise ValueError(f"Cannot convert {type(value)} to tensor")
        except Exception as e:
            raise DatasetSpecificHandlerError(
                dataset_type="RMD17",
                message=f"Failed to convert '{key}' to tensor: {str(e)}",
                operation="tensor_conversion",
                property_name=key,
                details=f"Molecule {molecule_index}, Value type: {type(value)}",
            ) from e

    def _is_valid_property(self, value: Any) -> bool:
        """
        Check if a property value is valid for rMD17.

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
        """Generate rMD17-specific processing statistics."""
        stats = {
            "dataset_type": "RMD17",
            "total_processed": len(processed_molecules),
            "experimental_setup": self.experimental_setup,
            "level_of_theory": "PBE/def2-SVP",
            "energy_units": "Hartree (converted from kcal/mol)",
        }

        atomization_calculations = 0
        molecules_with_forces = 0
        molecule_types = set()

        for mol_data in processed_molecules:
            if mol_data.get("atomization_energy_calculated"):
                atomization_calculations += 1
            if mol_data.get("forces") is not None:
                molecules_with_forces += 1
            if mol_data.get("molecule_name"):
                molecule_types.add(mol_data.get("molecule_name"))

        if atomization_calculations > 0:
            stats["atomization_energy_calculations"] = atomization_calculations

        stats["molecules_with_forces"] = molecules_with_forces
        stats["unique_molecule_types"] = list(molecule_types)

        # Add transform usage information if experimental setup exists
        if self.experimental_setup:
            stats["transform_aware_processing"] = True
            stats["experimental_context"] = {
                "setup_name": self.experimental_setup,
                "dataset_type": "RMD17",
                "transform_validation_performed": True,
            }

        return stats

    def get_supported_structural_features(self) -> dict[str, list[str]]:
        """
        rMD17 datasets support ALL structural features.

        rMD17 has optimized 3D geometries from MD trajectories at 500K,
        enabling all structural feature calculations. Similar support
        pattern to ANI-1x.
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
                # Partial charges (can be calculated from structure)
                "gasteiger_charge",
            ],
            "bond": [
                # Bond types
                "bond_type",
                "is_conjugated",
                "is_aromatic",
                "is_in_any_ring",
                "stereo",
                # Geometric features (rMD17 has 3D coordinates)
                "bond_length",
                "bond_length_binned",
            ],
        }

    def get_supported_descriptors(self) -> dict[str, list[str]]:
        """
        Get molecular descriptors supported by rMD17 dataset.

        rMD17 has 3D geometries from MD simulations and can support ALL
        descriptor categories including geometric descriptors.
        """
        return {
            "categories": [
                "constitutional",
                "topological",
                "electronic",
                "geometric",  # rMD17 has 3D coordinates
                "drug_likeness",
                "fragments",
            ],
            "excluded": [],  # rMD17 supports all descriptors
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
                # Geometric (rMD17 has 3D structures from MD)
                "RadiusOfGyration",
                "InertialShapeFactor",
                "Asphericity",
                # Drug-likeness
                "qed",
                "SPS",
            ],
            "requires_3d": True,  # rMD17 provides 3D structures from MD
            "requires_charges": False,  # No precomputed charges in rMD17
        }

    def get_transform_recommendations(self) -> dict[str, list[str]]:
        """
        Get rMD17-specific transform recommendations.

        Returns:
            Dict with recommended, avoid, and warning transforms
        """
        recommendations = {
            "recommended": [
                "GCNNorm - for message passing networks",
                "AddSelfLoops - required before GCNNorm",
                "NormalizeFeatures - for stable training",
                "RandomRotate - rMD17 has 3D coordinates from MD",
                "Distance - add distance-based edge features",
            ],
            "avoid": [],
            "warnings": [
                "Force data requires careful handling with geometric transforms",
                "DO NOT train on >1000 samples due to autocorrelation",
            ],
        }

        return recommendations

    def _get_dataset_suitable_transforms(self, available_transforms: dict[str, Any]) -> list[str]:
        """
        rMD17-suitable transforms based on structural and energetic properties.
        """
        suitable = []

        # Geometric transforms - rMD17 has 3D coordinates
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
        Validate transforms for rMD17 dataset compatibility.
        """
        warnings = []

        # rMD17 datasets have 3D coordinates - geometric transforms are relevant
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate", "RandomFlip"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            warnings.append(
                "rMD17 dataset without geometric augmentation - consider adding RandomRotate for invariance"
            )

        # Force data considerations
        if (
            hasattr(self.processing_config, "variable_len_graph_properties")
            and "forces" in self.processing_config.variable_len_graph_properties
            and "RandomRotate" in transform_names
        ):
            warnings.append(
                "rMD17 dataset has forces - geometric transforms will require force rotation"
            )

        # Distance-based transforms
        if "Distance" in transform_names or "Cartesian" in transform_names:
            warnings.append(
                "Distance/Cartesian transforms will add edge attributes - ensure model handles them"
            )

        return warnings

    def _check_transform_incompatibilities(self, transform_names: list[str]) -> list[str]:
        """
        Check for incompatible transform combinations for rMD17.
        """
        errors = []

        # rMD17 doesn't have precomputed charges, so VirtualNode is generally safe
        # No specific incompatibilities for rMD17

        return errors

    def _get_transform_recommendations(self, transform_names: list[str]) -> list[str]:
        """
        Get transform recommendations for rMD17 datasets with specific suggestions.
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

        # Geometric transform recommendations for rMD17
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            recommendations.append(
                "rMD17 3D structures benefit from geometric augmentation. "
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
# + Factory Functions
# ============================================================================
