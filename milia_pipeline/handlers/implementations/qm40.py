# milia_pipeline/handlers/implementations/qm40.py

"""
QM40 Dataset Handler
====================

Handler for the QM40 quantum chemistry dataset with exception integration and
transformation system support.

Key Features:
- Uses coordinate_based strategy (QM40 ships SMILES but NO InChI; connectivity
  is inferred from optimized 3D geometries via rdDetermineBonds).
- Coordinates in Angstrom (B3LYP-optimized).
- Energies in Hartree.
- Primary energy target: Internal_E_0K (internal energy at 0 K).
- Per-atom Mulliken charges (Qmulliken) and per-bond local vibrational mode
  force constants are available.

CRITICAL DIFFERENCE FROM QDπ:
- QDπ contains BOTH neutral and charged molecules and reads 'molecular_charge'.
- QM40 is NEUTRAL ONLY (anions and cations are excluded; charge-neutral singlet
  ground states), so get_molecular_charge() returns a constant 0 and no
  molecular-charge tracking is required.

QM40 Dataset Information:
-------------------------
- 7 elements: H, C, N, O, F, S, Cl
- 162,954 drug-like ZINC molecules (10-40 heavy atoms)
- Level of theory: B3LYP/6-31G(2df,p) (Gaussian16) — identical to QM9

Reference: Madushanka, Moura Jr. & Kraka, Scientific Data 11, 1376 (2024).
           DOI: 10.1038/s41597-024-04206-y
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


# QM40 supports 7 elements: H, C, N, O, F, S, Cl
QM40_SUPPORTED_ELEMENTS = {1, 6, 7, 8, 9, 16, 17}


@register_handler
class QM40DatasetHandler(DatasetHandler):
    """
    Handler for QM40 datasets with exception integration and transformation
    system support.

    Key characteristics:
    - Uses coordinate_based strategy (no InChI; SMILES kept only as a label)
    - Coordinates in Angstrom (B3LYP-optimized)
    - Energies in Hartree
    - Primary energy target: Internal_E_0K
    - NEUTRAL ONLY: get_molecular_charge() returns 0

    Reference: Madushanka, Moura Jr. & Kraka, Scientific Data 11, 1376 (2024).
    """

    def get_dataset_type(self) -> str:
        return "QM40"

    def validate_molecule_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str = "N/A"
    ) -> None:
        """Validate QM40-specific molecular data with exception handling."""
        try:
            # Validate essential QM40 properties
            essential_props = ["Internal_E_0K", "atoms", "coordinates"]
            missing_props = []

            for prop in essential_props:
                if not self._is_valid_property(raw_properties_dict.get(prop)):
                    missing_props.append(prop)

            if missing_props:
                raise HandlerValidationError(
                    message=f"Missing required QM40 properties: {missing_props}",
                    handler_type="QM40",
                    validation_type="essential_properties",
                    failed_validations=[f"Missing {prop}" for prop in missing_props],
                    molecule_index=molecule_index,
                    details="QM40 molecules must have Internal_E_0K energy, atoms, and coordinates",
                )

            # Validate structural consistency
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(atoms, coordinates, molecule_index, identifier)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="QM40",
                        message=f"QM40 molecular structure validation failed for molecule {molecule_index}",
                        operation="structure_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"Identifier: {identifier}, Atoms: {len(atoms) if atoms else 0}, "
                        f"Coords: {len(coordinates) if coordinates else 0}, "
                        f"Error: {str(e)}",
                    ) from e

            # Validate energy ranges (QM40 internal energies are negative in Hartree)
            energy = raw_properties_dict.get("Internal_E_0K")
            if energy is not None and isinstance(energy, (int, float, np.number)) and energy > 0:
                self.logger.warning(
                    f"QM40 molecule {molecule_index} has positive Internal_E_0K energy: {energy}"
                )

            # Validate elements are in the QM40 supported set
            if atoms is not None:
                atoms_array = np.asarray(atoms)
                unique_elements = set(atoms_array.flatten().tolist())
                unsupported = unique_elements - QM40_SUPPORTED_ELEMENTS
                if unsupported:
                    self.logger.warning(
                        f"QM40 molecule {molecule_index} has unsupported elements: {unsupported}"
                    )

        except (HandlerError, DatasetSpecificHandlerError):
            # Re-raise handler-specific errors
            raise
        except MoleculeProcessingError as e:
            # Convert molecule processing errors to QM40 handler validation errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM40",
                message=f"QM40 validation failed for molecule {molecule_index}: {e.message}",
                operation="molecule_validation",
                molecule_index=molecule_index,
                identifier=identifier,
                details=f"Identifier: {identifier}, Underlying error: {str(e)}",
            ) from e
        except Exception as e:
            # Convert unexpected errors to QM40 handler errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM40",
                message=f"Unexpected error during QM40 validation: {str(e)}",
                operation="molecule_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def get_required_properties(self) -> list[str]:
        """Get QM40-specific required properties."""
        required = self.get_common_required_properties()
        required.extend(["Internal_E_0K", "atoms", "coordinates"])  # Core QM40 properties

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
        Get QM40 identifier keys for molecule creation.

        QM40 has NO parseable chemical identifier suitable for graph construction
        (SMILES is present but kept only as a tracking label; no InChI). An empty
        list is returned so the coordinate_based strategy is used.

        Returns:
            Empty list (no identifier keys available).
        """
        return []

    def get_molecular_charge(
        self,
        raw_properties_dict: dict[str, Any],
        atomic_numbers: np.ndarray,
        mol_identifier: str | None = None,
    ) -> int:
        """
        Return molecular charge for QM40 molecules.

        QM40 is NEUTRAL ONLY: the dataset explicitly excludes anions and cations,
        so every structure is a charge-neutral singlet ground state. This is the
        inverse of QDπ (which reads a per-molecule 'molecular_charge'). The
        constant 0 is the correct, evidence-based charge for rdDetermineBonds
        bond-order determination on QM40 geometries.

        Args:
            raw_properties_dict: Raw molecule data from NPZ (unused).
            atomic_numbers: Array of atomic numbers (unused).
            mol_identifier: Molecular identifier (unused).

        Returns:
            int: 0 (always neutral).
        """
        return 0

    def get_molecule_creation_strategy(self) -> str:
        """
        QM40 datasets use the coordinate_based strategy.

        QM40 provides SMILES but NO InChI. Because SMILES-first parsing is
        unreliable (implicit hydrogens vs. explicit per-atom coordinate arrays)
        and QM40 supplies full optimized geometries for neutral closed-shell
        singlets, molecular connectivity is inferred from the 3D coordinates via
        rdDetermineBonds — identical to Wavefunction / ANI-1x / QDπ.

        Returns:
            str: 'coordinate_based'
        """
        return "coordinate_based"

    def process_property_value(
        self, key: str, value: Any, molecule_index: int, identifier: str = "N/A"
    ) -> Any:
        """Process QM40-specific property values with exception handling.

        Normalizes array properties to native numeric dtypes. NPZ files store
        ragged per-molecule data as object arrays; the individual elements must
        be coerced to native dtypes or downstream validate_molecular_structure()
        and _ensure_tensor() fail on object arrays (Pitfall 6). Mirrors the QDπ
        normalization pattern.
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
                            f"QM40 molecule {molecule_index}: Could not convert atoms to int64: {e}"
                        )
                        return value  # Return original, let downstream validation handle it
                elif arr.dtype != np.int64:
                    # Normalize to int64 for consistency (uint8 -> int64)
                    arr = arr.astype(np.int64)
                return arr

            # =================================================================
            # COORDINATES / INITIAL_COORDINATES - Ensure native float dtype
            # =================================================================
            if key in ("coordinates", "initial_coordinates"):
                arr = np.asarray(value)
                if not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float64)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"QM40 molecule {molecule_index}: Could not convert {key} to float64: {e}"
                        )
                        return value
                elif arr.dtype != np.float64:
                    # Normalize to float64 for consistency (float32 -> float64)
                    arr = arr.astype(np.float64)
                return arr

            # =================================================================
            # QMULLIKEN - Per-atom Mulliken charges - Ensure native float dtype
            # =================================================================
            if key == "Qmulliken":
                arr = np.asarray(value)
                if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"QM40 molecule {molecule_index}: Could not convert Qmulliken to float32: {e}"
                        )
                        return None
                if not np.all(np.isfinite(arr)):
                    self.logger.warning(
                        f"QM40 molecule {molecule_index} has non-finite Mulliken charges"
                    )
                    return None
                return arr

            # All other properties (scalar targets, bond_* arrays, smiles, ...)
            # are normalized at point of use (scalar/vector/variable-length
            # enrichers) — pass through unchanged here.
            return value

        except DatasetSpecificHandlerError:
            # Re-raise QM40 handler errors
            raise
        except Exception as e:
            # Convert unexpected property processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM40",
                message=f"Unexpected error processing QM40 property '{key}': {str(e)}",
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
        """QM40-specific PyG data enrichment with exception handling."""
        try:
            # Set dataset type
            pyg_data.dataset_type = "QM40"

            # Ensure num_nodes is set properly
            if not hasattr(pyg_data, "num_nodes") or pyg_data.num_nodes == 0:
                pyg_data.num_nodes = (
                    pyg_data.z.size(0) if hasattr(pyg_data, "z") and pyg_data.z is not None else 0
                )

            if pyg_data.num_nodes == 0:
                raise DatasetSpecificHandlerError(
                    dataset_type="QM40",
                    message="QM40 molecule has 0 nodes, cannot proceed with enrichment",
                    operation="enrich_pyg_data",
                    details="No atoms available for processing",
                )

            # 1. Add scalar graph targets (config-driven: e.g. Internal_E_0K, HOMO, ...)
            self._add_scalar_targets_internal(
                pyg_data, raw_properties_dict, molecule_index, identifier
            )

            # 2. Add vector graph properties
            if self.processing_config.vector_graph_properties:
                self._add_vector_properties_internal(
                    pyg_data, raw_properties_dict, molecule_index, identifier
                )

            # 3. Add variable-length properties (e.g. per-bond local-mode arrays)
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

            self.logger.debug(f"QM40 molecule {molecule_index}: Enrichment completed")
            return pyg_data

        except (PropertyEnrichmentError, DatasetSpecificHandlerError):
            raise
        except Exception as e:
            # Convert unexpected errors to QM40 handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM40",
                message=f"QM40 enrichment failed: {str(e)}",
                operation="enrich_pyg_data",
                details=f"Molecule {molecule_index}, Error during QM40-specific enrichment",
            ) from e

    def _add_scalar_targets_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """Internal QM40-specific scalar targets implementation."""
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
                            reason=f"Missing or invalid QM40 scalar target '{key}'",
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
                                reason=f"QM40 scalar target '{key}' is not a single value",
                                detail=f"Shape: {value.shape}",
                            )
                    elif isinstance(value, (int, float, np.number)):
                        val_to_add = float(value)
                    else:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"QM40 scalar target '{key}' has unexpected type",
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
                        reason=f"Critical error processing QM40 scalar target '{key}'",
                        detail=str(e),
                    ) from e

            if collected_targets:
                pyg_data.y = self._ensure_tensor(
                    collected_targets,
                    torch.float32,
                    "qm40_scalar_targets",
                    molecule_index,
                    identifier,
                )

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to QM40 handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM40",
                message=f"QM40 scalar targets processing failed: {str(e)}",
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
        """Internal QM40-specific vector properties implementation."""
        try:
            for prop_key in self.processing_config.vector_graph_properties:
                try:
                    value = raw_properties_dict.get(prop_key)

                    if not is_value_valid_and_not_nan(value):
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=prop_key,
                            reason=f"Missing or invalid QM40 vector property '{prop_key}'",
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
                            reason=f"QM40 vector property '{prop_key}' is not a 1D array",
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
                        reason=f"Error processing QM40 vector property '{prop_key}'",
                        detail=str(e),
                    ) from e

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to QM40 handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM40",
                message=f"QM40 vector properties processing failed: {str(e)}",
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
        """Internal QM40-specific variable-length property implementation.

        Used for per-bond local vibrational mode data (e.g. bond_lmod_ka,
        bond_atom1_idx, bond_atom2_idx). Properties not present for a molecule
        are skipped rather than treated as errors.
        """
        try:
            if not self.processing_config.variable_len_graph_properties:
                return

            for key in self.processing_config.variable_len_graph_properties:
                try:
                    value = raw_properties_dict.get(key)

                    if not is_value_valid_and_not_nan(value):
                        self.logger.debug(
                            f"QM40 molecule {molecule_index}: Skipping variable-length property "
                            f"'{key}' - not available"
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
                        reason=f"Error processing QM40 variable-length property '{key}'",
                        detail=str(e),
                    ) from e

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            # Convert unexpected errors to QM40 handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="QM40",
                message=f"QM40 variable-length properties processing failed: {str(e)}",
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
        Internal QM40-specific atomization energy calculation.

        Uses the configured base energy (Internal_E_0K, internal energy at 0 K)
        and the per-element reference energies in ATOMIC_ENERGIES_HARTREE. QM40's
        elements (H, C, N, O, F, S, Cl) must all be present in that table for the
        calculation to proceed; any missing element yields None (graceful skip).
        Returns the atomization energy in eV (consistent with the QM9 handler).
        """
        try:
            if not self.processing_config.calculate_atomization_energy_from:
                return None

            base_energy_key = self.processing_config.calculate_atomization_energy_from
            base_energy_raw = raw_properties_dict.get(base_energy_key)

            if not is_value_valid_and_not_nan(base_energy_raw):
                self.logger.warning(
                    f"QM40 molecule {molecule_index} missing {base_energy_key} for atomization energy"
                )
                return None

            if not hasattr(pyg_data, "z") or pyg_data.z is None:
                self.logger.warning(
                    f"QM40 molecule {molecule_index} missing atomic numbers for atomization energy"
                )
                return None

            # Convert to float - keep in original Hartree units
            if isinstance(base_energy_raw, np.ndarray) and base_energy_raw.size == 1:
                base_energy_hartree = float(base_energy_raw.item())
            elif isinstance(base_energy_raw, (int, float, np.number)):
                base_energy_hartree = float(base_energy_raw)
            else:
                self.logger.warning(
                    f"QM40 molecule {molecule_index}: Cannot convert {base_energy_key} to float"
                )
                return None

            # Calculate atomization energy
            if HAR2EV is None or not ATOMIC_ENERGIES_HARTREE:
                self.logger.warning(
                    f"QM40 molecule {molecule_index}: Missing atomic energies for atomization calculation"
                )
                return None

            # Sum atomic energies (in Hartree)
            sum_atomic_energies_hartree = 0.0
            for atomic_num in pyg_data.z.tolist():
                atomic_energy = ATOMIC_ENERGIES_HARTREE.get(atomic_num)
                if atomic_energy is None:
                    self.logger.warning(
                        f"QM40 molecule {molecule_index}: Missing atomic energy for element {atomic_num}"
                    )
                    return None
                sum_atomic_energies_hartree += atomic_energy

            # Calculate atomization energy in Hartree, then convert to eV
            atomization_energy_hartree = base_energy_hartree - sum_atomic_energies_hartree
            atomization_energy_eV = atomization_energy_hartree * HAR2EV

            self.logger.debug(
                f"QM40 molecule {molecule_index} atomization energy: {atomization_energy_eV:.4f} eV"
            )
            return atomization_energy_eV

        except Exception as e:
            self.logger.error(
                f"Error calculating QM40 atomization energy for molecule {molecule_index}: {e}"
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
                dataset_type="QM40",
                message=f"Failed to convert '{key}' to tensor: {str(e)}",
                operation="tensor_conversion",
                property_name=key,
                details=f"Molecule {molecule_index}, Value type: {type(value)}",
            ) from e

    def _is_valid_property(self, value: Any) -> bool:
        """Check if a property value is valid for QM40."""
        if value is None:
            return False
        if isinstance(value, str) and value.lower() in ["missing", "invalid", "", "nan"]:
            return False
        return is_value_valid_and_not_nan(value)

    def get_processing_statistics(
        self, processed_molecules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate QM40-specific processing statistics."""
        stats = {
            "dataset_type": "QM40",
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
                "dataset_type": "QM40",
                "transform_validation_performed": True,
            }

        return stats

    def get_supported_structural_features(self) -> dict[str, list[str]]:
        """
        QM40 datasets support ALL structural features.

        QM40 has optimized 3D geometries (B3LYP/6-31G(2df,p)) and Mulliken
        charges, enabling all structural feature calculations.
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
                # Partial charges (QM40 has Mulliken charges)
                "mulliken_charge",
                "gasteiger_charge",  # Can be calculated from structure
            ],
            "bond": [
                # Bond types
                "bond_type",
                "is_conjugated",
                "is_aromatic",
                "is_in_any_ring",
                "stereo",
                # Geometric features (QM40 has optimized 3D coordinates)
                "bond_length",
                "bond_length_binned",
            ],
        }

    def get_supported_descriptors(self) -> dict[str, list[str]]:
        """
        Get molecular descriptors supported by QM40 dataset.

        QM40 has optimized 3D geometries and Mulliken charges and can support ALL
        descriptor categories including geometric descriptors.
        """
        return {
            "categories": [
                "constitutional",
                "topological",
                "electronic",
                "geometric",  # QM40 has optimized 3D coordinates
                "drug_likeness",
                "fragments",
            ],
            "excluded": [],  # QM40 supports all descriptors
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
                # Geometric (QM40 has optimized 3D structures)
                "RadiusOfGyration",
                "InertialShapeFactor",
                "Asphericity",
                # Drug-likeness (QM40 is drug-like ZINC molecules)
                "qed",
                "SPS",
            ],
            "requires_3d": True,  # QM40 provides optimized 3D structures
            "requires_charges": True,  # QM40 has Mulliken charges
        }

    def get_transform_recommendations(self) -> dict[str, list[str]]:
        """
        Get QM40-specific transform recommendations.

        Returns:
            Dict with recommended, avoid, and warning transforms.
        """
        recommendations = {
            "recommended": [
                "GCNNorm - for message passing networks",
                "AddSelfLoops - required before GCNNorm",
                "NormalizeFeatures - for stable training",
                "RandomRotate - QM40 has 3D coordinates",
                "Distance - add distance-based edge features",
            ],
            "avoid": [],
            "warnings": [
                "VirtualNode may need careful handling with Mulliken charges",
            ],
        }

        return recommendations

    def _get_dataset_suitable_transforms(self, available_transforms: dict[str, Any]) -> list[str]:
        """QM40-suitable transforms based on structural and energetic properties."""
        suitable = []

        # Geometric transforms - QM40 has 3D coordinates
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
        """Validate transforms for QM40 dataset compatibility."""
        warnings = []

        # QM40 datasets have 3D coordinates - geometric transforms are relevant
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate", "RandomFlip"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            warnings.append(
                "QM40 dataset without geometric augmentation - consider adding RandomRotate "
                "for invariance"
            )

        # Distance-based transforms
        if "Distance" in transform_names or "Cartesian" in transform_names:
            warnings.append(
                "Distance/Cartesian transforms will add edge attributes - ensure model handles them"
            )

        return warnings

    def _check_transform_incompatibilities(self, transform_names: list[str]) -> list[str]:
        """Check for incompatible transform combinations for QM40."""
        errors = []

        # VirtualNode incompatibility with Mulliken charges
        if (
            "VirtualNode" in transform_names
            and hasattr(self.processing_config, "node_features")
            and "Qmulliken" in self.processing_config.node_features
        ):
            errors.append(
                "VirtualNode incompatible with Mulliken charges - "
                "virtual node would need artificial charge"
            )

        return errors

    def _get_transform_recommendations(self, transform_names: list[str]) -> list[str]:
        """Get transform recommendations for QM40 datasets with specific suggestions."""
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

        # Geometric transform recommendations for QM40
        geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate"]
        has_geometric = any(t in transform_names for t in geometric_transforms)

        if not has_geometric:
            recommendations.append(
                "QM40 3D structures benefit from geometric augmentation. "
                "Suggestion: RandomRotate() for rotational invariance testing"
            )

        return recommendations
