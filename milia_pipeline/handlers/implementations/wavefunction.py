# milia_pipeline/handlers/implementations/wavefunction.py

"""
Wavefunction Dataset Handler
============================

Handler for Wavefunction datasets with exception integration
and transformation system support.

Extracted from dataset_handlers.py as part of the Handler Module Refactoring.

Key Features:
- Validates wavefunction-specific features (MO energies, orbital data)
- Extracts HOMO/LUMO properties and gap calculations
- Handles molecular orbital occupation data
- Supports complete quantum mechanical feature spectrum
- Uses coordinate_based molecule creation strategy

Reference: milia Dataset Architecture Refactoring Plan v2.2.0
"""

import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from milia_pipeline.config.config_containers import DatasetConfig, FilterConfig, ProcessingConfig
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
class WavefunctionDatasetHandler(DatasetHandler):
    """
    Handler for milia Wavefunction dataset.

    This handler processes molecular wavefunction data from the milia dataset,
    which contains quantum mechanical descriptors extracted from .molden files.

    CRITICAL: This handler LOADS preprocessed .npz files created by the preprocessor.
    It does NOT create or preprocess data.

    WORKFLOW CLARIFICATION:
    ----------------------
    This handler assumes preprocessing has ALREADY been completed:

    1. Preprocessing (Phase 1-4):
       - Extracts .molden files from tar.gz archive
       - Parses wavefunction data with IOData
       - Creates .npz file with features

    2. Handler Loading (Phase 5 - THIS CLASS):
       - Reads existing .npz file
       - Validates wavefunction data
       - Extracts HOMO/LUMO properties
       - Creates PyG dataset objects

    Key Features:
    - Validates wavefunction-specific features (MO energies, orbital data)
    - Extracts HOMO/LUMO properties and gap calculations
    - Handles molecular orbital occupation data
    - Supports complete quantum mechanical feature spectrum

    Attributes:
        dataset_config: Dataset configuration
        filter_config: Filter configuration
        processing_config: Processing configuration
        logger: Logger instance
        experimental_setup: Optional experimental setup name
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        filter_config: FilterConfig,
        processing_config: ProcessingConfig,
        logger: logging.Logger,
        experimental_setup: str | None = None,
    ):
        """
        Initialize Wavefunction dataset handler.

        Args:
            dataset_config: Dataset configuration
            filter_config: Filter configuration
            processing_config: Processing configuration
            logger: Logger instance
            experimental_setup: Optional experimental setup name
        """
        super().__init__(
            dataset_config, filter_config, processing_config, logger, experimental_setup
        )

        self.logger.info("Initializing WavefunctionDatasetHandler")

    def get_dataset_type(self) -> str:
        """
        Get the dataset type identifier.

        Returns:
            "Wavefunction"
        """
        return "Wavefunction"

    def get_molecular_charge(
        self,
        raw_properties_dict: dict[str, Any],
        atomic_numbers: np.ndarray,
        mol_identifier: str | None = None,
    ) -> int:
        """
        Calculate charge from electronic structure data.

        Method: charge = n_electrons - sum(atomic_numbers)

        This is the most accurate method when electronic structure data is available.

        Args:
            raw_properties_dict: Raw molecule data from NPZ (must contain n_electrons)
            atomic_numbers: Array of atomic numbers (Z values)
            mol_identifier: Molecular identifier (not used for wavefunction)

        Returns:
            int: Molecular charge calculated from n_electrons (0 if cannot determine)
        """
        n_electrons = raw_properties_dict.get("n_electrons")

        if n_electrons is None:
            self.logger.debug("n_electrons not found, assuming neutral molecule")
            return 0

        try:
            charge = int(n_electrons) - int(np.sum(atomic_numbers))
            self.logger.debug(f"Calculated charge from n_electrons: {charge}")
            return charge
        except Exception as e:
            self.logger.warning(f"Failed to calculate charge from n_electrons: {e}")
            return 0

    def get_molecule_creation_strategy(self) -> str:
        """
        Wavefunction datasets use coordinate_based strategy.

        Wavefunction molecular data contains only compound labels (e.g., 'BrCPxSiSxH4_331')
        which are NOT parseable chemical identifiers. Molecular connectivity must be
        inferred directly from 3D atomic coordinates using rdDetermineBonds algorithm.

        Molecular charge (calculated from n_electrons - sum(atomic_numbers)) is
        CRITICAL for accurate bond order assignment in rdDetermineBonds.

        Data requirements:
            - Compound label/identifier (for tracking only, not parsed)
            - Atomic numbers (for atom types)
            - Coordinates in Bohr (automatically converted to Angstrom)
            - Molecular charge from n_electrons (REQUIRED for rdDetermineBonds)

        Returns:
            str: 'coordinate_based'
        """
        return "coordinate_based"

    def validate_molecule_data(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str = "N/A"
    ) -> None:
        """
        Validate wavefunction-specific molecular data.

        Validates presence and correctness of wavefunction features including
        orbital energies, occupations, and structural data.

        Args:
            raw_properties_dict: Dictionary of molecular properties
            molecule_index: Index of molecule being validated
            identifier: Molecule identifier (usually InChI)

        Raises:
            HandlerValidationError: If required properties are missing
            DatasetSpecificHandlerError: If dataset-specific validation fails
        """
        try:
            # Validate essential wavefunction properties
            essential_props = ["atoms", "coordinates"]
            missing_props = []

            for prop in essential_props:
                if not self._is_valid_property(raw_properties_dict.get(prop)):
                    missing_props.append(prop)

            if missing_props:
                raise HandlerValidationError(
                    message=f"Missing required wavefunction properties: {missing_props}",
                    handler_type="Wavefunction",
                    validation_type="essential_properties",
                    failed_validations=[f"Missing {prop}" for prop in missing_props],
                    molecule_index=molecule_index,
                    details="Wavefunction molecules must have atoms and coordinates",
                )

            # Validate structural consistency
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(atoms, coordinates, molecule_index, identifier)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="Wavefunction",
                        message=f"Wavefunction molecular structure validation failed for molecule {molecule_index}",
                        operation="structure_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"InChI: {identifier}, Atoms: {len(atoms) if atoms else 0}, "
                        f"Coords: {len(coordinates) if coordinates else 0}, "
                        f"Error: {str(e)}",
                    ) from e

            # Validate wavefunction-specific features
            self._validate_wavefunction_features(raw_properties_dict, molecule_index, identifier)

        except (HandlerError, DatasetSpecificHandlerError):
            # Re-raise handler-specific errors
            raise
        except MoleculeProcessingError as e:
            # Convert molecule processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="Wavefunction",
                message=f"Wavefunction validation failed for molecule {molecule_index}: {e.message}",
                operation="molecule_validation",
                molecule_index=molecule_index,
                identifier=identifier,
                details=f"InChI: {identifier}, Underlying error: {str(e)}",
            ) from e
        except Exception as e:
            # Convert unexpected errors
            raise DatasetSpecificHandlerError(
                dataset_type="Wavefunction",
                message=f"Unexpected error during wavefunction validation: {str(e)}",
                operation="molecule_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _validate_wavefunction_features(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str
    ) -> None:
        """
        Validate wavefunction-specific quantum mechanical features.

        Args:
            raw_properties_dict: Dictionary of molecular properties
            molecule_index: Index of molecule being validated
            identifier: Molecule identifier

        Raises:
            DatasetSpecificHandlerError: If wavefunction features are invalid
        """
        # Validate MO energies if present
        mo_energies = raw_properties_dict.get("mo_energies")
        if mo_energies is not None:
            if not isinstance(mo_energies, (list, np.ndarray)):
                raise DatasetSpecificHandlerError(
                    dataset_type="Wavefunction",
                    message=f"MO energies must be array-like for molecule {molecule_index}",
                    operation="mo_energies_validation",
                    property_name="mo_energies",
                    molecule_index=molecule_index,
                    details=f"Got type: {type(mo_energies)}",
                )

            # Check for valid energy values
            try:
                energies_array = np.array(mo_energies, dtype=float)
                if not np.all(np.isfinite(energies_array)):
                    self.logger.warning(f"Molecule {molecule_index} has non-finite MO energies")
            except (ValueError, TypeError) as e:
                raise DatasetSpecificHandlerError(
                    dataset_type="Wavefunction",
                    message=f"Invalid MO energy values for molecule {molecule_index}",
                    operation="mo_energies_validation",
                    property_name="mo_energies",
                    molecule_index=molecule_index,
                    details=str(e),
                ) from e

        # Validate MO occupations if present
        mo_occupations = raw_properties_dict.get("mo_occupations")
        if mo_occupations is not None:
            if not isinstance(mo_occupations, (list, np.ndarray)):
                raise DatasetSpecificHandlerError(
                    dataset_type="Wavefunction",
                    message=f"MO occupations must be array-like for molecule {molecule_index}",
                    operation="mo_occupations_validation",
                    property_name="mo_occupations",
                    molecule_index=molecule_index,
                    details=f"Got type: {type(mo_occupations)}",
                )

        # Validate HOMO-LUMO gap if present
        homo_lumo_gap = raw_properties_dict.get("homo_lumo_gap_eV")
        if homo_lumo_gap is not None:
            try:
                gap_value = float(homo_lumo_gap)
                if gap_value < 0:
                    self.logger.warning(
                        f"Molecule {molecule_index} has negative HOMO-LUMO gap: {gap_value} eV"
                    )
            except (ValueError, TypeError):
                self.logger.warning(f"Molecule {molecule_index} has invalid HOMO-LUMO gap value")

    def get_required_properties(self) -> list[str]:
        """
        Get wavefunction-specific required properties.

        Returns:
            List of required property names
        """
        required = self.get_common_required_properties()

        # Add wavefunction-specific properties
        # Note: We don't require mo_energies or homo_lumo_gap as required
        # since not all molecules may have complete orbital analysis

        # Add properties from processing config
        if self.processing_config:
            required.extend(self.processing_config.scalar_graph_targets)
            required.extend(self.processing_config.node_features)
            required.extend(self.processing_config.vector_graph_properties)
            required.extend(self.processing_config.variable_len_graph_properties)

        return list(set(required))

    def get_identifier_keys(self) -> list[tuple[str, str]]:
        """
        Get Wavefunction identifier keys for molecule creation.

        Wavefunction datasets use compound IDs as identifiers (not parseable
        molecular identifiers like InChI/SMILES). These are used for tracking
        but molecule creation uses coordinate-based approach.

        Returns:
            List of (npz_key, identifier_type) tuples
        """
        return [("compounds", "compound_id")]

    def process_property_value(
        self, key: str, value: Any, molecule_index: int, identifier: str = "N/A"
    ) -> Any:
        """
        Process wavefunction-specific property values.

        Args:
            key: Property key
            value: Property value
            molecule_index: Index of molecule
            identifier: Molecule identifier

        Returns:
            Processed property value

        Raises:
            DatasetSpecificHandlerError: If processing fails
        """
        try:
            # Handle MO energies
            if key == "mo_energies" and value is not None:
                if isinstance(value, (list, tuple)):
                    try:
                        return np.array(value, dtype=float)
                    except ValueError as e:
                        raise DatasetSpecificHandlerError(
                            dataset_type="Wavefunction",
                            message=f"Failed to convert MO energies for molecule {molecule_index}",
                            operation="property_processing",
                            property_name=key,
                            details=f"InChI: {identifier}, Value type: {type(value)}, "
                            f"Conversion error: {str(e)}",
                        ) from e

            # Handle MO occupations
            if key == "mo_occupations" and value is not None:
                if isinstance(value, (list, tuple)):
                    try:
                        return np.array(value, dtype=float)
                    except ValueError as e:
                        raise DatasetSpecificHandlerError(
                            dataset_type="Wavefunction",
                            message=f"Failed to convert MO occupations for molecule {molecule_index}",
                            operation="property_processing",
                            property_name=key,
                            details=f"InChI: {identifier}, Value type: {type(value)}, "
                            f"Conversion error: {str(e)}",
                        ) from e

            # Handle HOMO-LUMO gap
            if key == "homo_lumo_gap_eV" and value is not None:
                if not is_value_valid_and_not_nan(value):
                    self.logger.warning(
                        f"Wavefunction molecule {molecule_index} has invalid HOMO-LUMO gap"
                    )
                    return None

            return value

        except DatasetSpecificHandlerError:
            # Re-raise dataset-specific handler errors
            raise
        except Exception as e:
            # Convert unexpected property processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="Wavefunction",
                message=f"Unexpected error processing wavefunction property '{key}': {str(e)}",
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
        """
        Add wavefunction-specific enrichments to PyG Data object.

        Args:
            pyg_data: PyTorch Geometric Data object
            raw_properties_dict: Dictionary of molecular properties
            molecule_index: Index of molecule
            identifier: Molecule identifier

        Returns:
            Enriched PyG Data object

        Raises:
            PropertyEnrichmentError: If property enrichment fails
            DatasetSpecificHandlerError: If dataset-specific enrichment fails
        """
        try:
            # Add scalar targets (HOMO-LUMO gap, total energy, etc.)
            self._add_scalar_targets_internal(
                pyg_data, raw_properties_dict, molecule_index, identifier
            )

            # Add orbital properties if configured
            self._add_orbital_properties_internal(
                pyg_data, raw_properties_dict, molecule_index, identifier
            )

            # Add node features if configured
            if self.processing_config.node_features:
                self._add_node_features_internal(
                    pyg_data, raw_properties_dict, molecule_index, identifier
                )

            return pyg_data

        except (PropertyEnrichmentError, DatasetSpecificHandlerError):
            raise
        except Exception as e:
            raise DatasetSpecificHandlerError(
                dataset_type="Wavefunction",
                message=f"Wavefunction PyG enrichment failed: {str(e)}",
                operation="pyg_enrichment",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _add_scalar_targets_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """
        Internal wavefunction-specific implementation of scalar target addition.

        Handles HOMO-LUMO gap, total energy, and other scalar quantum properties.
        Now tier-aware: only requires targets available for the feature_tier.
        """
        try:
            if not self.processing_config.scalar_graph_targets:
                return

            collected_targets = []

            # PHASE 6 FIX: Tier-aware scalar target filtering
            # Get the feature_tier from raw_properties_dict (passed from NPZ metadata)
            feature_tier = raw_properties_dict.get("_feature_tier")

            # Determine which targets are available for this tier
            if feature_tier:
                try:
                    from milia_pipeline.preprocessing.utils.format_parsers import FEATURE_TIERS

                    tier_available_keys = set(FEATURE_TIERS.get(feature_tier, []))

                    # Filter scalar_graph_targets to only include tier-available keys
                    targets_to_process = [
                        key
                        for key in self.processing_config.scalar_graph_targets
                        if key in tier_available_keys
                    ]

                    # Log if some targets are being skipped due to tier
                    skipped_targets = [
                        key
                        for key in self.processing_config.scalar_graph_targets
                        if key not in tier_available_keys
                    ]
                    if skipped_targets:
                        self.logger.debug(
                            f"Skipping {len(skipped_targets)} scalar targets not available in "
                            f"'{feature_tier}' tier: {skipped_targets}"
                        )
                except ImportError:
                    # If FEATURE_TIERS import fails, use all configured targets
                    targets_to_process = self.processing_config.scalar_graph_targets
            else:
                # No tier info available - use all configured targets (backward compatibility)
                targets_to_process = self.processing_config.scalar_graph_targets

            for key in targets_to_process:
                try:
                    value = raw_properties_dict.get(key)

                    if value is None:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"Required wavefunction scalar target '{key}' is missing",
                            detail="Value retrieved was None",
                        )

                    # Convert to scalar float
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
                                reason=f"Wavefunction scalar target '{key}' has array shape {value.shape}",
                                detail=f"Expected single scalar, got shape: {value.shape}",
                            )
                    elif isinstance(value, (str, bytes, np.str_, np.bytes_)):
                        try:
                            val_to_add = float(value)
                        except ValueError:
                            raise PropertyEnrichmentError(
                                molecule_index=molecule_index,
                                inchi=identifier,
                                property_name=key,
                                reason=f"Wavefunction scalar target '{key}' string cannot be converted",
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
                                reason=f"Wavefunction scalar target '{key}' is a list with {len(value)} elements",
                                detail="Expected single scalar",
                            )
                    else:
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"Wavefunction scalar target '{key}' has unexpected type {type(value)}",
                            detail=f"Value type: {type(value)}",
                        )

                    # Validate the converted value
                    if not is_value_valid_and_not_nan(val_to_add):
                        raise PropertyEnrichmentError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            property_name=key,
                            reason=f"Wavefunction scalar target '{key}' has NaN or Inf value",
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
                        reason=f"Critical error processing wavefunction scalar target '{key}'",
                        detail=str(e),
                    ) from e

            if collected_targets:
                pyg_data.y = self._ensure_tensor(
                    collected_targets,
                    torch.float32,
                    "wavefunction_scalar_targets",
                    molecule_index,
                    identifier,
                )

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            raise DatasetSpecificHandlerError(
                dataset_type="Wavefunction",
                message=f"Wavefunction scalar targets processing failed: {str(e)}",
                operation="add_scalar_targets",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}",
            ) from e

    def _add_orbital_properties_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """
        Add molecular orbital properties to PyG data.

        Handles MO energies, occupations, and derived properties.
        """
        try:
            # Add MO energies if present and configured
            mo_energies = raw_properties_dict.get("mo_energies")
            if (
                mo_energies is not None
                and "mo_energies" in self.processing_config.variable_len_graph_properties
            ):
                mo_array = self._ensure_tensor(
                    mo_energies, torch.float32, "mo_energies", molecule_index, identifier
                )
                pyg_data.mo_energies = mo_array

            # Add MO occupations if present and configured
            mo_occupations = raw_properties_dict.get("mo_occupations")
            if (
                mo_occupations is not None
                and "mo_occupations" in self.processing_config.variable_len_graph_properties
            ):
                occ_array = self._ensure_tensor(
                    mo_occupations, torch.float32, "mo_occupations", molecule_index, identifier
                )
                pyg_data.mo_occupations = occ_array

        except Exception as e:
            # Non-critical - log warning but don't fail
            self.logger.warning(
                f"Could not add orbital properties for molecule {molecule_index}: {e}"
            )

    def _add_node_features_internal(
        self,
        pyg_data: Data,
        raw_properties_dict: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> None:
        """
        Add wavefunction-specific node features.

        Currently uses atom types as primary node feature.
        Future: Could add atomic orbital contributions, charges, etc.
        """
        try:
            atoms = raw_properties_dict.get("atoms")
            if atoms is None:
                raise PropertyEnrichmentError(
                    molecule_index=molecule_index,
                    inchi=identifier,
                    property_name="atoms",
                    reason="Atoms required for node features but not found",
                    detail="Cannot create node features without atomic information",
                )

            # Convert atoms to atomic numbers (node features)
            from milia_pipeline.config.config_constants import HEAVY_ATOM_SYMBOLS_TO_Z

            atomic_numbers = []
            for atom in atoms:
                if atom in HEAVY_ATOM_SYMBOLS_TO_Z:
                    atomic_numbers.append(HEAVY_ATOM_SYMBOLS_TO_Z[atom])
                else:
                    # Unknown atom - use 0 as placeholder
                    self.logger.warning(f"Unknown atom type '{atom}' in molecule {molecule_index}")
                    atomic_numbers.append(0)

            node_features = self._ensure_tensor(
                atomic_numbers, torch.long, "node_features", molecule_index, identifier
            )
            pyg_data.x = node_features.unsqueeze(-1) if node_features.dim() == 1 else node_features

        except PropertyEnrichmentError:
            raise
        except Exception as e:
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=identifier,
                property_name="node_features",
                reason=f"Failed to add node features: {str(e)}",
                detail=f"Error type: {type(e).__name__}",
            ) from e

    def get_processing_statistics(
        self, processed_molecules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Generate wavefunction-specific processing statistics.

        Args:
            processed_molecules: List of processed molecule dictionaries

        Returns:
            Dictionary of processing statistics
        """
        stats = {"total_molecules": len(processed_molecules), "dataset_type": "Wavefunction"}

        if not processed_molecules:
            return stats

        # Compute HOMO-LUMO gap statistics if available
        homo_lumo_gaps = []
        molecules_with_orbital_data = 0

        for mol in processed_molecules:
            gap = mol.get("homo_lumo_gap_eV")
            if gap is not None and is_value_valid_and_not_nan(gap):
                homo_lumo_gaps.append(float(gap))

            if mol.get("mo_energies") is not None:
                molecules_with_orbital_data += 1

        if homo_lumo_gaps:
            stats["homo_lumo_gap_stats"] = {
                "mean": np.mean(homo_lumo_gaps),
                "std": np.std(homo_lumo_gaps),
                "min": np.min(homo_lumo_gaps),
                "max": np.max(homo_lumo_gaps),
                "count": len(homo_lumo_gaps),
            }

        stats["molecules_with_orbital_data"] = molecules_with_orbital_data
        stats["orbital_data_percentage"] = (
            molecules_with_orbital_data / len(processed_molecules)
        ) * 100

        # Compute atom count statistics
        atom_counts = []
        for mol in processed_molecules:
            atoms = mol.get("atoms")
            if atoms is not None and (
                isinstance(atoms, np.ndarray)
                or isinstance(atoms, (list, tuple))
                and len(atoms) > 0
            ):
                atom_counts.append(len(atoms))

        if atom_counts:
            stats["atom_count_stats"] = {
                "mean": np.mean(atom_counts),
                "std": np.std(atom_counts),
                "min": int(np.min(atom_counts)),
                "max": int(np.max(atom_counts)),
            }

        return stats

    def get_supported_structural_features(self) -> dict[str, list[str]]:
        """
        Get structural features supported by wavefunction dataset.

        Wavefunction data has complete 3D coordinates and quantum chemical data,
        similar to DFT. However, some charge-based features may not be available.

        Returns:
            Dict with 'atom' and 'bond' keys containing supported feature lists
        """
        return {
            "atom": [
                # Topological features (always available)
                "degree",
                "total_degree",
                "num_h",
                "implicit_valence",
                "explicit_valence",
                "hybridization",
                "aromatic",
                "ring_size",
                "in_ring",
                # Charge features (may not be available for all molecules)
                # 'partial_charge',  # Excluded - not computed in wavefunction preprocessing
                # 'mulliken_charge',  # Excluded - not computed in wavefunction preprocessing
                # 3D features (available with coordinates)
                "chiral_tag",
            ],
            "bond": [
                # Topological features (always available)
                "bond_type",
                "is_conjugated",
                "in_ring",
                "stereo",
                # 3D features (available with coordinates)
                # 'bond_length',  # Conservative - may be computed but not validated
                # 'bond_length_binned',  # Conservative
            ],
        }

    def get_transform_recommendations(self) -> dict[str, list[str]]:
        """
        Get wavefunction-specific transform recommendations.

        Returns:
            Dict with recommended, avoid, and warning transforms
        """
        recommendations = {"recommended": [], "avoid": [], "warnings": []}

        # Recommended transforms for wavefunction data
        recommendations["recommended"].extend(
            [
                "Distance",  # Good for 3D molecular structures
                "Cartesian",  # 3D coordinate transformations
                "ToUndirected",  # Molecular graphs are undirected
                "AddSelfLoops",  # Helpful for message passing
                "NormalizeFeatures",  # Important for neural networks
            ]
        )

        # Transforms to avoid
        recommendations["avoid"].extend(
            [
                "VirtualNode",  # May interfere with molecular structure
                "RandomNodeSplit",  # Inappropriate for molecules
            ]
        )

        # Warnings for specific transforms
        recommendations["warnings"].extend(
            [
                "GCNNorm: Ensure graph connectivity is preserved",
                "DropEdge: Use carefully to maintain molecular validity",
                "MaskFeatures: May hide important quantum properties",
            ]
        )

        return recommendations

    def get_supported_descriptors(self) -> dict[str, list[str]]:
        """
        Get descriptors supported by Wavefunction datasets.

        Wavefunction datasets have high-quality 3D geometries from
        quantum mechanical calculations and can support ALL descriptor
        categories including geometric descriptors. This is the most
        complete dataset type for descriptor calculation.
        """
        return {
            "categories": [
                "constitutional",
                "topological",
                "electronic",
                "geometric",  # Wavefunction has high-quality 3D coordinates
                "drug_likeness",
                "fragments",
            ],
            "excluded": [],  # Wavefunction supports all descriptors
            "recommended": [
                # Constitutional
                "MolWt",
                "NumRotatableBonds",
                "NumHDonors",
                "NumHAcceptors",
                "NumAmideBonds",
                "NumBridgeheadAtoms",
                # Topological
                "TPSA",
                "BertzCT",
                "Chi0v",
                "Chi1v",
                "Chi2v",
                "AvgIpc",
                "HallKierAlpha",
                # Electronic (Wavefunction-specific strength)
                "MaxPartialCharge",
                "MinPartialCharge",
                "MaxAbsPartialCharge",
                "MinAbsPartialCharge",
                "MaxEStateIndex",
                "MinEStateIndex",
                # Geometric (Wavefunction-specific strength)
                "RadiusOfGyration",
                "InertialShapeFactor",
                "Asphericity",
                "Eccentricity",
                "SpherocityIndex",
                "PMI1",
                "PMI2",
                "PMI3",
                "NPR1",
                "NPR2",
                # Drug-likeness
                "qed",
                "SPS",
                "Phi",
                "FractionCSP3",
                # Fragments
                "fr_amide",
                "fr_ester",
                "fr_ether",
                "fr_phenol",
            ],
            "requires_3d": True,  # Wavefunction provides QM-quality 3D structures
            "requires_charges": True,  # Can compute from wavefunction data
        }

    # Helper methods from base class pattern
    def _is_valid_property(self, value: Any) -> bool:
        """Check if a property value is valid (not None, not empty)."""
        if value is None:
            return False
        return not (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0)

    def _ensure_tensor(
        self,
        value: Any,
        dtype: torch.dtype,
        property_name: str,
        molecule_index: int,
        identifier: str,
    ) -> torch.Tensor:
        """
        Convert value to PyTorch tensor with validation.

        Args:
            value: Value to convert
            dtype: Target dtype
            property_name: Name of property (for error messages)
            molecule_index: Molecule index
            identifier: Molecule identifier

        Returns:
            PyTorch tensor

        Raises:
            PropertyEnrichmentError: If conversion fails
        """
        try:
            if isinstance(value, torch.Tensor):
                return value.to(dtype)
            elif isinstance(value, np.ndarray):
                return torch.from_numpy(value).to(dtype)
            elif isinstance(value, (list, tuple)):
                return torch.tensor(value, dtype=dtype)
            else:
                return torch.tensor([value], dtype=dtype)
        except Exception as e:
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=identifier,
                property_name=property_name,
                reason=f"Failed to convert to tensor: {str(e)}",
                detail=f"Value type: {type(value)}, Target dtype: {dtype}",
            ) from e

    def _validate_dataset_specific_transforms(self, transform_names: list[str]) -> list[str]:
        """
        Validate dataset-specific transform compatibility.

        Args:
            transform_names: List of transform names

        Returns:
            List of warning messages
        """
        warnings = []

        # Warn about transforms that might affect quantum properties
        if "MaskFeatures" in transform_names:
            warnings.append(
                "MaskFeatures may hide important wavefunction properties. "
                "Use with caution for quantum mechanical data."
            )

        if "DropNode" in transform_names:
            warnings.append(
                "DropNode is inappropriate for molecular structures. "
                "Removing atoms changes the molecule identity."
            )

        return warnings

    def _check_transform_incompatibilities(self, transform_names: list[str]) -> list[str]:
        """
        Check for known transform incompatibilities.

        Args:
            transform_names: List of transform names

        Returns:
            List of error messages (empty if no incompatibilities)
        """
        errors = []

        # Check for incompatible transforms
        if "VirtualNode" in transform_names:
            errors.append(
                "VirtualNode is incompatible with molecular structures. "
                "Virtual nodes interfere with chemical graph semantics."
            )

        if "RandomNodeSplit" in transform_names:
            errors.append(
                "RandomNodeSplit is incompatible with molecular data. "
                "Atoms cannot be arbitrarily split."
            )

        return errors

    def _get_transform_recommendations(self, transform_names: list[str]) -> list[str]:
        """
        Get recommendations based on configured transforms.

        Args:
            transform_names: List of transform names

        Returns:
            List of recommendation messages
        """
        recommendations = []

        # Recommend Distance if not present
        if "Distance" not in transform_names:
            recommendations.append("Consider adding Distance transform for 3D molecular structures")

        # Recommend normalization
        if not any(t in transform_names for t in ["NormalizeFeatures", "Normalize", "GCNNorm"]):
            recommendations.append(
                "Consider adding normalization transform for neural network training"
            )

        return recommendations

    def _get_dataset_suitable_transforms(self, available_transforms: dict[str, Any]) -> list[str]:
        """
        Get transforms suitable for wavefunction dataset.

        Args:
            available_transforms: Dictionary of available transforms

        Returns:
            List of suitable transform names
        """
        suitable = []

        # Geometric transforms
        geometric = ["Distance", "Cartesian", "Polar", "Spherical"]
        suitable.extend([t for t in geometric if t in available_transforms])

        # Graph structure transforms
        structural = ["AddSelfLoops", "ToUndirected"]
        suitable.extend([t for t in structural if t in available_transforms])

        # Normalization
        normalization = ["GCNNorm", "NormalizeFeatures", "Normalize"]
        suitable.extend([t for t in normalization if t in available_transforms])

        # Light augmentation
        augmentation = ["DropEdge"]
        suitable.extend([t for t in augmentation if t in available_transforms])

        return suitable
