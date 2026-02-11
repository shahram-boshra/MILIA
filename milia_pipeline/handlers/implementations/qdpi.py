# milia_pipeline/handlers/implementations/qdpi.py

"""
QDπ Dataset Handler
===================

Handler for QDπ (Quantum Deep Potential Interaction) dataset with exception 
integration and transformation system support.

Extracted and adapted from ANI-2x handler as part of the Handler Module Refactoring.

Key Features:
- Uses coordinate_based strategy (NO InChI/SMILES identifiers available)
- Coordinates in Angstrom
- Energies in Hartree (converted from eV during preprocessing)
- Forces in Hartree/Angstrom (converted from eV/Angstrom during preprocessing)
- Primary energy target: energy (mapped from 'energies' in HDF5)
- CRITICAL: Supports BOTH neutral AND charged molecules

CRITICAL DIFFERENCE FROM ANI-2x:
- ANI-2x: All molecules are neutral (charge = 0 hardcoded)
- QDπ: Contains BOTH neutral AND charged molecules
       Charge must be retrieved from NPZ 'molecular_charge' property
       (determined during preprocessing from file path: neutral/ vs charged/)

QDπ Dataset Information:
------------------------
- 13 elements: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I
- ~1.6 million structures from SPICE, ANI, GEOM, FreeSolv, RE, COMP6
- Level of theory: ωB97M-D3(BJ)/def2-TZVPPD (highly accurate)
- Includes ion pairs, protonated/deprotonated amino acids, tautomers

Reference: Zeng et al., Scientific Data 12, 693 (2025)
           DOI: 10.1038/s41597-025-04972-3
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union

from torch_geometric.data import Data

from milia_pipeline.config.config_containers import (
    DatasetConfig, 
    FilterConfig, 
    ProcessingConfig
)
from milia_pipeline.config.config_constants import (
    ATOMIC_ENERGIES_HARTREE, 
    HAR2EV
)
from milia_pipeline.config.validators import (
    validate_molecular_structure,
    is_value_valid_and_not_nan
)
from milia_pipeline.exceptions import (
    PropertyEnrichmentError,
    MoleculeProcessingError,
    HandlerError,
    HandlerConfigurationError,
    HandlerValidationError,
    DatasetSpecificHandlerError,
)

# Import from refactored base handler
from milia_pipeline.handlers.base_handler import (
    DatasetHandler,
    handle_transform_errors
)
from milia_pipeline.handlers.handler_registry import register_handler

logger = logging.getLogger(__name__)


# QDπ supports 13 elements
QDPI_SUPPORTED_ELEMENTS = {1, 3, 6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53}
# H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I


@register_handler
class QDPiDatasetHandler(DatasetHandler):
    """
    Handler for QDπ datasets with exception integration and transformation system support.
    
    QDπ is a quantum chemistry dataset for drug discovery containing ~1.6 million
    structures of drug-like molecules and biopolymer fragments computed at the
    ωB97M-D3(BJ)/def2-TZVPPD level of theory.
    
    Key characteristics:
    - Uses coordinate_based strategy (NO InChI/SMILES identifiers available)
    - Coordinates in Angstrom
    - Energies in Hartree (converted from eV during preprocessing)
    - Forces in Hartree/Angstrom (converted from eV/Angstrom)
    - Primary energy target: energy
    - CRITICAL: Supports BOTH neutral AND charged molecules
    
    CRITICAL DIFFERENCES FROM ANI-2x:
    - ANI-2x: H, C, N, O, S, F, Cl (7 elements), neutral only
    - QDπ: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I (13 elements)
    - QDπ includes charged molecules (ions, protonated species)
    - QDπ uses more accurate DFT: ωB97M-D3(BJ)/def2-TZVPPD
    
    Charge Handling:
    - Neutral molecules: molecular_charge = 0
    - Charged molecules: molecular_charge retrieved from NPZ property
    - Charge is REQUIRED for correct rdDetermineBonds bond order determination
    
    Reference: Zeng et al., Scientific Data 12, 693 (2025)
    """

    def get_dataset_type(self) -> str:
        return "QDPi"
    
    def validate_molecule_data(self,
                              raw_properties_dict: Dict[str, Any],
                              molecule_index: int,
                              identifier: str = "N/A") -> None:
        """Validate QDπ-specific molecular data with exception handling."""
        try:
            # Validate essential QDπ properties
            essential_props = ['energy', 'atoms', 'coordinates']
            missing_props = []
            
            for prop in essential_props:
                if not self._is_valid_property(raw_properties_dict.get(prop)):
                    missing_props.append(prop)
            
            if missing_props:
                raise HandlerValidationError(
                    message=f"Missing required QDπ properties: {missing_props}",
                    handler_type="QDPi",
                    validation_type="essential_properties",
                    failed_validations=[f"Missing {prop}" for prop in missing_props],
                    molecule_index=molecule_index,
                    details=f"QDπ molecules must have energy, atoms, and coordinates"
                )
            
            # Validate structural consistency
            atoms = raw_properties_dict.get('atoms')
            coordinates = raw_properties_dict.get('coordinates')
            
            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(atoms, coordinates, molecule_index, identifier)
                except ValueError as e:
                    raise DatasetSpecificHandlerError(
                        dataset_type="QDPi",
                        message=f"QDπ molecular structure validation failed for molecule {molecule_index}",
                        operation="structure_validation",
                        molecule_index=molecule_index,
                        identifier=identifier,
                        details=f"Identifier: {identifier}, Atoms: {len(atoms) if atoms else 0}, "
                               f"Coords: {len(coordinates) if coordinates else 0}, "
                               f"Error: {str(e)}"
                    ) from e
            
            # Validate energy (QDπ energies are typically negative in Hartree)
            energy = raw_properties_dict.get('energy')
            if energy is not None and isinstance(energy, (int, float, np.number)):
                if energy > 0:
                    self.logger.warning(f"QDπ molecule {molecule_index} has positive energy: {energy}")
            
            # Validate elements are in QDπ supported set
            if atoms is not None:
                atoms_array = np.asarray(atoms)
                unique_elements = set(atoms_array.flatten().tolist())
                unsupported = unique_elements - QDPI_SUPPORTED_ELEMENTS
                if unsupported:
                    self.logger.warning(
                        f"QDπ molecule {molecule_index} has unsupported elements: {unsupported}"
                    )
            
        except (HandlerError, DatasetSpecificHandlerError):
            # Re-raise handler-specific errors
            raise
        except MoleculeProcessingError as e:
            # Convert molecule processing errors to QDπ handler validation errors
            raise DatasetSpecificHandlerError(
                dataset_type="QDPi",
                message=f"QDπ validation failed for molecule {molecule_index}: {e.message}",
                operation="molecule_validation",
                molecule_index=molecule_index,
                identifier=identifier,
                details=f"Identifier: {identifier}, Underlying error: {str(e)}"
            ) from e
        except Exception as e:
            # Convert unexpected errors to QDπ handler errors
            raise DatasetSpecificHandlerError(
                dataset_type="QDPi",
                message=f"Unexpected error during QDπ validation: {str(e)}",
                operation="molecule_validation",
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}"
            ) from e
    
    def get_required_properties(self) -> List[str]:
        """Get QDπ-specific required properties."""
        required = self.get_common_required_properties()
        required.extend(['energy', 'atoms', 'coordinates'])  # Core QDπ properties
        
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
    
    def get_identifier_keys(self) -> List[Tuple[str, str]]:
        """
        Get QDπ identifier keys for molecule creation.
        
        CRITICAL: QDπ has NO parseable chemical identifiers.
        The HDF5 structure contains only formula groups (for organization),
        atomic_types, and coordinates.
        Returns empty list - coordinate_based strategy will be used.
        
        Returns:
            Empty list (no identifier keys available)
        """
        return []
    
    def get_molecular_charge(
        self, 
        raw_properties_dict: Dict[str, Any],
        atomic_numbers: np.ndarray,
        mol_identifier: Optional[str] = None
    ) -> int:
        """
        Return molecular charge for QDπ molecules.
        
        CRITICAL: Unlike ANI-2x (neutral-only), QDπ contains BOTH neutral AND 
        charged molecules. The charge must be retrieved from the NPZ data, where
        it was determined during preprocessing based on the source directory:
        - data/neutral/ → molecular_charge = 0
        - data/charged/ → molecular_charge stored per-molecule
        
        The molecular_charge is ESSENTIAL for correct bond order determination
        when using rdDetermineBonds for coordinate-based molecule creation.
        Incorrect charge leads to incorrect bond orders, which breaks the
        molecular graph representation.
        
        Charge Inference Strategy (if not stored):
        1. Check 'molecular_charge' key in raw_properties_dict
        2. Check 'charge' key (alternative naming)
        3. Check 'charge_type' key ('neutral' → 0, 'charged' → infer from formula)
        4. Default to 0 (neutral) with warning for coordinate_based strategy
        
        Args:
            raw_properties_dict: Raw molecule data from NPZ
            atomic_numbers: Array of atomic numbers (Z values)
            mol_identifier: Molecular identifier (formula group name)
            
        Returns:
            int: Net molecular charge
            
        Note:
            For neutral molecules: returns 0
            For charged molecules: returns value from NPZ (typically ±1, ±2)
        """
        # Strategy 1: Check for explicit molecular_charge property
        if 'molecular_charge' in raw_properties_dict:
            charge_value = raw_properties_dict['molecular_charge']
            if charge_value is not None:
                try:
                    return int(charge_value)
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"QDπ molecule {mol_identifier}: Invalid molecular_charge value: {charge_value}"
                    )
        
        # Strategy 2: Check for 'charge' key (alternative naming)
        if 'charge' in raw_properties_dict:
            charge_value = raw_properties_dict['charge']
            if charge_value is not None:
                try:
                    return int(charge_value)
                except (ValueError, TypeError):
                    pass
        
        # Strategy 3: Check 'charge_type' metadata
        charge_type = raw_properties_dict.get('charge_type', None)
        if charge_type == 'neutral':
            return 0
        elif charge_type == 'charged':
            # Charged but specific value not stored - try to infer from formula
            # This is a fallback for cases where preprocessing didn't store exact charge
            inferred_charge = self._infer_charge_from_elements(atomic_numbers, mol_identifier)
            if inferred_charge != 0:
                self.logger.debug(
                    f"QDπ molecule {mol_identifier}: Inferred charge {inferred_charge} from elements"
                )
            return inferred_charge
        
        # Strategy 4: Default to neutral (with debug logging)
        self.logger.debug(
            f"QDπ molecule {mol_identifier}: No charge information found, assuming neutral (charge=0)"
        )
        return 0
    
    def _infer_charge_from_elements(
        self,
        atomic_numbers: np.ndarray,
        mol_identifier: Optional[str]
    ) -> int:
        """
        Attempt to infer molecular charge from atomic composition.
        
        This is a fallback mechanism for charged molecules where the exact
        charge wasn't stored during preprocessing. It identifies common
        ionic species in QDπ:
        - Alkali metal cations: Li+ (3), Na+ (11), K+ (19)
        - Halide anions: F- (9), Cl- (17), Br- (35), I- (53)
        
        For single-atom ions, the charge is unambiguous.
        For multi-atom molecules, this heuristic has limitations.
        
        Args:
            atomic_numbers: Array of atomic numbers
            mol_identifier: Identifier for logging
            
        Returns:
            int: Inferred charge (0 if cannot determine)
        """
        if atomic_numbers is None or len(atomic_numbers) == 0:
            return 0
        
        atoms = np.asarray(atomic_numbers).flatten()
        
        # Single atom - can be ion
        if len(atoms) == 1:
            z = int(atoms[0])
            # Alkali cations
            if z in [3, 11, 19]:  # Li, Na, K
                return 1
            # Halide anions (but F, Cl, Br, I can also be neutral in molecules)
            # Only return -1 if truly isolated (single atom)
            if z in [9, 17, 35, 53]:  # F, Cl, Br, I
                return -1
        
        # Multi-atom systems: look for patterns
        # Count alkali metals and halogens
        alkali_count = sum(1 for z in atoms if z in [3, 11, 19])
        
        # Simple ion pair detection: 2 atoms, one alkali + one halogen
        if len(atoms) == 2:
            has_alkali = any(z in [3, 11, 19] for z in atoms)
            has_halide = any(z in [9, 17, 35, 53] for z in atoms)
            if has_alkali and has_halide:
                # Ion pair: typically neutral overall (e.g., NaCl)
                return 0
        
        # For more complex molecules, cannot reliably determine charge
        # Return 0 and rely on the preprocessing to have stored correct charge
        return 0

    def get_molecule_creation_strategy(self) -> str:
        """
        QDπ datasets use coordinate_based strategy.
        
        CRITICAL: QDπ HDF5 structure contains NO parseable chemical identifiers.
        The data structure only contains:
        - 'elements': Element symbols (e.g., ['H', 'C', 'N', 'O'])
        - 'atomic_types': Integer indices into elements array
        - 'coordinates': 3D positions
        - 'energies': Total energies (eV)
        - 'forces': Atomic forces (eV/Angstrom)
        
        Molecular connectivity must be inferred from 3D coordinates using
        the rdDetermineBonds algorithm.
        
        IMPORTANT: Unlike ANI-2x, QDπ contains charged molecules.
        The molecular_charge parameter is CRITICAL for correct bond order
        determination in rdDetermineBonds.
        
        Returns:
            str: 'coordinate_based'
        """
        return 'coordinate_based'
    
    def process_property_value(self,
                              key: str,
                              value: Any,
                              molecule_index: int,
                              identifier: str = "N/A") -> Any:
        """Process QDπ-specific property values with exception handling.
        
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
        - molecular_charge: int64 (charge must be integer)
        """
        try:
            # Return None values as-is
            if value is None:
                return None
            
            # =================================================================
            # ATOMS - Ensure native integer dtype
            # =================================================================
            if key == 'atoms':
                arr = np.asarray(value)
                # Convert to int64 (standard atomic number dtype) if not already integer
                if not np.issubdtype(arr.dtype, np.integer):
                    try:
                        arr = arr.astype(np.int64)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"QDπ molecule {molecule_index}: Could not convert atoms to int64: {e}"
                        )
                        return value  # Return original, let downstream validation handle it
                elif arr.dtype != np.int64:
                    # Normalize to int64 for consistency (uint8 -> int64)
                    arr = arr.astype(np.int64)
                return arr
            
            # =================================================================
            # COORDINATES - Ensure native float dtype
            # =================================================================
            if key == 'coordinates':
                arr = np.asarray(value)
                # Convert to float64 (standard coordinate dtype) if not already floating
                if not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float64)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"QDπ molecule {molecule_index}: Could not convert coordinates to float64: {e}"
                        )
                        return value
                elif arr.dtype != np.float64:
                    # Normalize to float64 for consistency (float32 -> float64)
                    arr = arr.astype(np.float64)
                return arr
            
            # =================================================================
            # FORCES - Per-atom property (N, 3) - Ensure native float dtype
            # =================================================================
            if key == 'forces':
                arr = np.asarray(value)
                # Ensure proper float dtype for tensor conversion
                if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            f"QDπ molecule {molecule_index}: Could not convert forces to float32: {e}"
                        )
                        return None
                # Validate after conversion
                if not np.all(np.isfinite(arr)):
                    self.logger.warning(f"QDπ molecule {molecule_index} has non-finite forces data")
                    return None
                return arr
            
            # =================================================================
            # ENERGY - Scalar property - Pass through (already float64 in NPZ)
            # =================================================================
            if key == 'energy':
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = float(value.item())
                    # Check for NaN
                    if np.isnan(value) if isinstance(value, float) else np.any(np.isnan(value)):
                        self.logger.debug(f"QDπ molecule {molecule_index} has NaN energy")
                        return None
                return value
            
            # =================================================================
            # MOLECULAR_CHARGE - Ensure integer dtype
            # =================================================================
            if key in ('molecular_charge', 'charge'):
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = int(value.item())
                elif isinstance(value, (float, np.floating)):
                    value = int(round(value))
                return value
            
            # All other properties - pass through unchanged
            return value
            
        except DatasetSpecificHandlerError:
            # Re-raise QDπ handler errors
            raise
        except Exception as e:
            # Convert unexpected property processing errors
            raise DatasetSpecificHandlerError(
                dataset_type="QDPi",
                message=f"Unexpected error processing QDπ property '{key}': {str(e)}",
                operation="property_processing",
                property_name=key,
                details=f"Molecule {molecule_index}, Error: {type(e).__name__}: {str(e)}"
            ) from e

    def enrich_pyg_data(self,
                       pyg_data: Data,
                       raw_properties_dict: Dict[str, Any],
                       molecule_index: int,
                       identifier: str = "N/A") -> Data:
        """QDπ-specific PyG data enrichment with exception handling."""
        try:
            # Set dataset type
            pyg_data.dataset_type = "QDPi"
            
            # Ensure num_nodes is set properly
            if not hasattr(pyg_data, 'num_nodes') or pyg_data.num_nodes == 0:
                pyg_data.num_nodes = pyg_data.z.size(0) if hasattr(pyg_data, 'z') and pyg_data.z is not None else 0
            
            if pyg_data.num_nodes == 0:
                raise DatasetSpecificHandlerError(
                    dataset_type="QDPi",
                    message="QDπ molecule has 0 nodes, cannot proceed with enrichment",
                    operation="enrich_pyg_data",
                    molecule_index=molecule_index,
                    details=f"Identifier: {identifier}"
                )
            
            # Get processed coordinates  
            coordinates = raw_properties_dict.get('coordinates')
            if coordinates is not None:
                coords_array = np.asarray(coordinates, dtype=np.float32)
                if coords_array.shape[0] == pyg_data.num_nodes:
                    pyg_data.pos = torch.tensor(coords_array, dtype=torch.float32)
                else:
                    self.logger.warning(
                        f"QDπ molecule {molecule_index}: Coordinate count mismatch "
                        f"({coords_array.shape[0]} vs {pyg_data.num_nodes} nodes)"
                    )
            
            # Add energy as target (y)
            energy = raw_properties_dict.get('energy')
            if energy is not None:
                # Use _ensure_tensor to properly handle scalar -> (1,) shape conversion
                pyg_data.y = self._ensure_tensor(
                    energy, torch.float32, 'energy', molecule_index, identifier
                )
            
            # Add forces if available
            forces = raw_properties_dict.get('forces')
            if forces is not None:
                forces_array = np.asarray(forces, dtype=np.float32)
                if forces_array.shape[0] == pyg_data.num_nodes:
                    pyg_data.forces = torch.tensor(forces_array, dtype=torch.float32)
            
            # Store molecular charge if available (useful for downstream analysis)
            charge = raw_properties_dict.get('molecular_charge', 
                     raw_properties_dict.get('charge', 0))
            if charge is not None:
                pyg_data.molecular_charge = torch.tensor([int(charge)], dtype=torch.long)
            
            # Store formula/identifier if available
            formula = raw_properties_dict.get('formula')
            if formula is not None:
                pyg_data.formula = formula
            
            # Store subset info if available
            subset = raw_properties_dict.get('subset')
            if subset is not None:
                pyg_data.subset = subset
            
            return pyg_data
            
        except (PropertyEnrichmentError, DatasetSpecificHandlerError):
            raise
        except Exception as e:
            # Convert unexpected errors to QDπ handler operation errors
            raise DatasetSpecificHandlerError(
                dataset_type="QDPi",
                message=f"Unexpected error during QDπ PyG enrichment: {str(e)}",
                operation="pyg_enrichment",
                molecule_index=molecule_index,
                details=f"Identifier: {identifier}, Error: {type(e).__name__}: {str(e)}"
            ) from e
    
    def get_processing_statistics(self, processed_molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get QDπ-specific processing statistics.
        
        Args:
            processed_molecules: List of processed molecule data
            
        Returns:
            Dict with processing statistics
        """
        stats = {
            'dataset_type': 'QDPi',
            'total_processed': len(processed_molecules),
            'has_forces': 0,
            'has_charge_info': 0,
            'neutral_count': 0,
            'charged_count': 0,
        }
        
        for mol_data in processed_molecules:
            if mol_data.get('forces') is not None:
                stats['has_forces'] += 1
            
            charge = mol_data.get('molecular_charge', mol_data.get('charge'))
            if charge is not None:
                stats['has_charge_info'] += 1
                if int(charge) == 0:
                    stats['neutral_count'] += 1
                else:
                    stats['charged_count'] += 1
        
        # Add experimental setup context if available
        if self.experimental_setup:
            stats['experimental_context'] = {
                'setup_name': self.experimental_setup,
                'dataset_type': 'QDPi',
                'transform_validation_performed': True
            }
        
        return stats

    def get_supported_structural_features(self) -> Dict[str, List[str]]:
        """
        QDπ datasets support ALL structural features.
        
        QDπ has optimized 3D geometries (ωB97M-D3(BJ)/def2-TZVPPD) enabling all 
        structural feature calculations. This includes geometric descriptors
        that require accurate 3D coordinates.
        """
        return {
            'atom': [
                # Basic connectivity
                'degree',
                'total_degree',
                
                # Hybridization and bonding
                'hybridization',
                'total_valence',
                'is_aromatic',
                'is_in_ring',
                'num_aromatic_bonds',
                
                # Chirality
                'chirality',
                
                # Partial charges (can be calculated from structure)
                'gasteiger_charge',
            ],
            'bond': [
                # Bond types
                'bond_type',
                'is_conjugated',
                'is_aromatic',
                'is_in_any_ring',
                'stereo',
                
                # Geometric features (QDπ has optimized 3D coordinates)
                'bond_length',
                'bond_length_binned',
            ]
        }

    def get_supported_descriptors(self) -> Dict[str, List[str]]:
        """
        Get molecular descriptors supported by QDπ dataset.
        
        QDπ has optimized 3D geometries and can support ALL
        descriptor categories including geometric descriptors.
        """
        return {
            'categories': [
                'constitutional',
                'topological', 
                'electronic',
                'geometric',  # QDπ has optimized 3D coordinates
                'drug_likeness',
                'fragments'
            ],
            'excluded': [],  # QDπ supports all descriptors
            'recommended': [
                # Constitutional
                'MolWt', 'NumRotatableBonds', 'NumHDonors', 'NumHAcceptors',
                # Topological
                'TPSA', 'BertzCT', 'Chi0v', 'Chi1v',
                # Electronic
                'MaxPartialCharge', 'MinPartialCharge',
                # Geometric (QDπ has optimized 3D structures)
                'RadiusOfGyration', 'InertialShapeFactor', 'Asphericity',
                # Drug-likeness (QDπ is designed for drug discovery)
                'qed', 'SPS'
            ],
            'requires_3d': True,  # QDπ provides optimized 3D structures
            'requires_charges': False  # QDπ doesn't have precomputed partial charges
        }

    def get_transform_recommendations(self) -> Dict[str, List[str]]:
        """
        Get QDπ-specific transform recommendations.
        
        Returns:
            Dict with recommended, avoid, and warning transforms
        """
        recommendations = {
            'recommended': [
                'GCNNorm - for message passing networks',
                'AddSelfLoops - required before GCNNorm',
                'NormalizeFeatures - for stable training',
                'RandomRotate - QDπ has 3D coordinates',
                'Distance - add distance-based edge features',
            ],
            'avoid': [],
            'warnings': []
        }
        
        return recommendations

    def _get_dataset_suitable_transforms(self, 
                                        available_transforms: Dict[str, Any]) -> List[str]:
        """
        QDπ-suitable transforms based on structural and energetic properties.
        
        Args:
            available_transforms: Dict of all available transforms
            
        Returns:
            List of transform names suitable for QDπ datasets
        """
        suitable = []
        
        # Geometric transforms - QDπ has 3D coordinates
        geometric = ['RandomRotate', 'RandomTranslate', 'RandomScale']
        suitable.extend([t for t in geometric if t in available_transforms])
        
        # Normalization
        normalization = ['GCNNorm', 'NormalizeFeatures']
        suitable.extend([t for t in normalization if t in available_transforms])
        
        # Graph structure
        structure = ['AddSelfLoops', 'ToUndirected']
        suitable.extend([t for t in structure if t in available_transforms])
        
        # Edge features
        edge_features = ['Distance', 'Cartesian']
        suitable.extend([t for t in edge_features if t in available_transforms])
        
        # Light augmentation
        augmentation = ['DropEdge', 'MaskFeatures']
        suitable.extend([t for t in augmentation if t in available_transforms])
        
        return suitable

    def _validate_dataset_specific_transforms(self, transform_names: List[str]) -> List[str]:
        """
        Validate transforms for QDπ dataset compatibility.
        
        Args:
            transform_names: List of transform class names
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # QDπ datasets have 3D coordinates - geometric transforms are relevant
        geometric_transforms = ['RandomRotate', 'RandomScale', 'RandomTranslate', 'RandomFlip']
        has_geometric = any(t in transform_names for t in geometric_transforms)
        
        if not has_geometric:
            warnings.append("QDπ dataset without geometric augmentation - consider adding RandomRotate for invariance")
        
        # Force data considerations
        if hasattr(self.processing_config, 'variable_len_graph_properties'):
            if 'forces' in self.processing_config.variable_len_graph_properties:
                if 'RandomRotate' in transform_names:
                    warnings.append("QDπ dataset has forces - geometric transforms will require force rotation")
        
        # Distance-based transforms
        if 'Distance' in transform_names or 'Cartesian' in transform_names:
            warnings.append("Distance/Cartesian transforms will add edge attributes - ensure model handles them")
        
        return warnings
    
    def _check_transform_incompatibilities(self, transform_names: List[str]) -> List[str]:
        """
        Check for incompatible transform combinations for QDπ.
        
        Args:
            transform_names: List of transform class names
            
        Returns:
            List of error messages (empty if all compatible)
        """
        # QDπ doesn't have specific incompatibilities
        return []
    
    def _get_transform_recommendations(self, transform_names: List[str]) -> List[str]:
        """
        Provide specific transform recommendations for QDπ datasets.
        
        Args:
            transform_names: List of current transform class names
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if 'GCNNorm' in transform_names and 'AddSelfLoops' not in transform_names:
            recommendations.append("Consider adding AddSelfLoops before GCNNorm for better performance")
        
        if not any(t in transform_names for t in ['RandomRotate', 'RandomTranslate']):
            recommendations.append("QDπ has 3D coordinates - consider geometric augmentation")
        
        return recommendations
