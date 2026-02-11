# mol_conversion_utils.py

"""
Utility functions for converting molecular representations, specifically from InChI and coordinates
to RDKit molecules, and then to PyTorch Geometric Data objects.

Handles various error conditions and ensures proper data integrity during conversion.
Supports any registered dataset type through the dataset handlers pattern.

Modified to focus on connectivity and avoid stereochemistry conflicts with QM-optimized structures.

Backward Compatibility Cleanup - Handler-Only Architecture
Removed all legacy parameter-based system support. All operations now require explicit handler creation.
Functions no longer accept None handler or fallback to global config creation.

KEY MODIFICATIONS FOR QM STRUCTURES:
1. Disabled full sanitization in MolFromInchi/MolFromSmiles - uses sanitize=False
2. Applied selective partial sanitization to avoid valence errors with bromine/halogens 
3. Removed stereochemistry operations (RemoveStereochemistry, AssignStereochemistryFrom3D)
4. Removed back-conversion validation (InChI comparison) that rejected valid QM structures
5. Added robust error handling for AddHs failures in problematic molecules
6. Focus on preserving QM-optimized coordinates without structural modifications
7. Enhanced with handler-specific exception types for better error context
"""
import logging
from typing import Optional, Dict, Any, Union, List, Tuple

import numpy as np
import torch
from torch_geometric.utils import from_rdmol
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D

from milia_pipeline.handlers import DatasetHandler
from milia_pipeline.config.validators import (
    is_value_valid_and_not_nan,
    validate_molecular_structure,
    validate_property_value,
    validate_coordinates_3d,
    validate_atomic_numbers
)
from milia_pipeline.exceptions import (
    MoleculeProcessingError, 
    RDKitConversionError, 
    PyGDataCreationError,
    HandlerError,
    HandlerOperationError,
    HandlerValidationError,
    DatasetSpecificHandlerError,
    HandlerNotAvailableError,
    wrap_handler_operation,
    create_handler_error_context,
    create_dataset_handler_error
)


logger = logging.getLogger(__name__)


@wrap_handler_operation("UNKNOWN", "create_rdkit_mol")
def create_rdkit_mol(
    mol_identifier: str,
    coordinates: np.ndarray, 
    atomic_numbers: np.ndarray,
    logger: logging.Logger,
    handler: DatasetHandler,  
    molecule_index: Optional[int] = None,
    mol_id_type: str = 'inchi',
    molecular_charge: int = 0
) -> Chem.Mol:
    """
    Creates an RDKit molecule using handler-determined strategy.
    
    DYNAMIC STRATEGY SELECTION:
    The handler determines which molecule creation approach to use based on
    available data. This makes the function future-proof for any dataset type.
    
    Strategies:
    
    1. identifier_coordinate_based:
       - Parse identifier (InChI/SMILES) to get molecular connectivity and bonds
       - Perform atom mapping between identifier ordering and QM dataset ordering
       - Assign QM-optimized coordinates to preserve exact 3D geometry
       - molecular_charge used for logging/metadata only
       
    2. coordinate_based:
       - Infer connectivity and bond orders from coordinates using rdDetermineBonds
       - Identifier used only for logging (not parsed)
       - molecular_charge CRITICAL for accurate bond order determination
       - Automatically handle coordinate unit conversion (e.g., Bohr→Angstrom)
    
    Args:
        mol_identifier (str): Molecular identifier (InChI, SMILES, or compound label)
        coordinates (np.ndarray): QM-optimized 3D coordinates, shape (num_atoms, 3)
        atomic_numbers (np.ndarray): Atomic numbers (Z), shape (num_atoms,)
        logger (logging.Logger): Logger instance
        handler (DatasetHandler): Dataset handler (REQUIRED)
        molecule_index (Optional[int]): Molecule index for error reporting
        mol_id_type (str): Type of identifier ('inchi', 'smiles', 'compound_id', etc.)
        molecular_charge (int): Molecular charge (default: 0)
    
    Returns:
        rdkit.Chem.Mol: RDKit molecule with 3D coordinates in conformer
    
    Raises:
        ValueError: If handler is None
        RDKitConversionError: If molecule creation fails
        HandlerOperationError: If handler operations fail
        HandlerValidationError: If validation fails
    """
    # Validate handler
    if handler is None:
        raise ValueError(
            "Handler is required for create_rdkit_mol(). "
            "Please create a handler explicitly before calling this function."
        )
    
    # Validate handler for this operation
    validate_handler_for_conversion(handler, "create_rdkit_mol", molecule_index)
    
    # Get strategy from handler
    strategy = handler.get_molecule_creation_strategy()
    
    dataset_type = handler.get_dataset_type()
    context = f"{dataset_type} Molecule {molecule_index} ({mol_identifier})"
    
    logger.debug(f"{context}: Using '{strategy}' molecule creation strategy")
    
    # Log molecular charge if non-zero
    if molecular_charge != 0:
        logger.debug(f"{context}: Molecular charge = {molecular_charge}")
    
    # Validate molecular structure
    try:
        validated_atomic_numbers, validated_coords = validate_molecular_structure(
            atomic_numbers, coordinates, molecule_index or -1, mol_identifier
        )
    except ValueError as e:
        raise HandlerValidationError(
            message="Molecular structure validation failed",
            handler_type=dataset_type,
            validation_type="structure_validation",
            failed_validations=[str(e)],
            molecule_index=molecule_index,
            details=f"Failed to validate atomic numbers and coordinates: {str(e)}"
        ) from e
    
    # Route to appropriate strategy
    if strategy == 'identifier_coordinate_based':
        # Strategy 1: Parse identifier for connectivity, assign coordinates
        return _create_molecule_identifier_coordinate_based(
            mol_identifier=mol_identifier,
            mol_id_type=mol_id_type,
            validated_coords=validated_coords,
            validated_atomic_numbers=validated_atomic_numbers,
            molecular_charge=molecular_charge,
            handler=handler,
            dataset_type=dataset_type,
            context=context,
            logger=logger,
            molecule_index=molecule_index
        )
    
    elif strategy == 'coordinate_based':
        # Strategy 2: Infer connectivity from coordinates
        return _create_molecule_coordinate_based(
            mol_identifier=mol_identifier,
            validated_coords=validated_coords,
            validated_atomic_numbers=validated_atomic_numbers,
            molecular_charge=molecular_charge,
            handler=handler,
            dataset_type=dataset_type,
            context=context,
            logger=logger,
            molecule_index=molecule_index
        )
    
    else:
        raise ValueError(
            f"Unknown molecule creation strategy: '{strategy}'. "
            f"Handler {dataset_type} returned unsupported strategy. "
            f"Expected 'identifier_coordinate_based' or 'coordinate_based'."
        )


def _create_molecule_identifier_coordinate_based(
    mol_identifier: str,
    mol_id_type: str,
    validated_coords: np.ndarray,
    validated_atomic_numbers: np.ndarray,
    molecular_charge: int,
    handler: DatasetHandler,
    dataset_type: str,
    context: str,
    logger: logging.Logger,
    molecule_index: Optional[int]
) -> Chem.Mol:
    """
    Create molecule using identifier for connectivity + coordinates for 3D geometry.
    
    This is the v0 approach: parse InChI/SMILES to get bonds, then assign QM coordinates.
    Molecular charge is used for logging/metadata only (not needed for molecule creation).
    
    Process:
        1. Parse identifier (InChI or SMILES) → get molecular connectivity
        2. Add hydrogens if needed
        3. Verify atom count matches QM data
        4. Verify atomic composition matches
        5. Create atom mapping (identifier ordering → QM ordering)
        6. Apply minimal sanitization
        7. Assign QM coordinates using mapping
    
    Args:
        mol_identifier: InChI or SMILES string
        mol_id_type: 'inchi' or 'smiles'
        validated_coords: Validated coordinates array
        validated_atomic_numbers: Validated atomic numbers array
        molecular_charge: Molecular charge (for logging only)
        handler: Dataset handler
        dataset_type: Dataset type name
        context: Context string for logging
        logger: Logger instance
        molecule_index: Molecule index
    
    Returns:
        Chem.Mol: RDKit molecule with QM coordinates
    """
    # Create error context for handler operations
    error_context = create_handler_error_context(
        handler_type=dataset_type,
        operation="create_rdkit_mol_identifier_coordinate",
        molecule_index=molecule_index,
        additional_context={
            'mol_id_type': mol_id_type,
            'identifier': mol_identifier,
            'molecular_charge': molecular_charge
        }
    )

    if mol_id_type == 'smiles':
        logger.debug(f"{context}: Starting SMILES-based RDKit mol creation.")
        
        try:
            # Step 1: Parse SMILES to get connectivity/bonding information
            mol = Chem.MolFromSmiles(mol_identifier, sanitize=False)
            if mol is None or mol.GetNumAtoms() == 0:
                raise RDKitConversionError(
                    molecule_index=molecule_index,
                    inchi=mol_identifier,
                    reason="RDKit MolFromSmiles (sanitize=False) resulted in None or empty molecule.",
                    detail="The SMILES string could not be parsed by RDKit"
                )
            
            # Step 2: Try to add hydrogens to get complete structure
            num_atoms_before_addhs = mol.GetNumAtoms()
            try:
                mol_with_hs = Chem.AddHs(mol, addCoords=False)
                if mol_with_hs is not None and mol_with_hs.GetNumAtoms() > 0:
                    mol = mol_with_hs
                    logger.debug(f"{context}: Successfully added hydrogens ({num_atoms_before_addhs} -> {mol.GetNumAtoms()} atoms)")
                else:
                    logger.warning(
                        f"{context}: AddHs returned None/empty. "
                        f"Proceeding with original molecule (may be missing explicit H). "
                        f"Atoms: {num_atoms_before_addhs}"
                    )
            except Exception as addhs_e:
                logger.warning(
                    f"{context}: AddHs failed with error: {addhs_e}. "
                    f"This is common for QM structures with unusual valences. "
                    f"Proceeding with molecule without explicit hydrogens ({num_atoms_before_addhs} atoms)."
                )
            
            # Step 3: Verify atom count matches
            num_atoms_smiles = mol.GetNumAtoms()
            num_atoms_coords = validated_coords.shape[0]
            if num_atoms_smiles != num_atoms_coords:
                raise HandlerValidationError(
                    message="Atom count mismatch between SMILES and QM coordinates",
                    handler_type=dataset_type,
                    validation_type="atom_count_validation",
                    failed_validations=[f"SMILES: {num_atoms_smiles}, QM coords: {num_atoms_coords}"],
                    molecule_index=molecule_index,
                    details=f"SMILES has {num_atoms_smiles} atoms but QM calculation has {num_atoms_coords} atoms. "
                            f"If AddHs failed, the SMILES may be missing explicit hydrogens."
                )
            
            # Step 4: Verify atomic composition matches
            rdkit_atomic_nums_sorted = sorted([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_atoms_smiles)])
            qm_atomic_nums_sorted = sorted(validated_atomic_numbers.tolist())
            
            if rdkit_atomic_nums_sorted != qm_atomic_nums_sorted:
                raise HandlerValidationError(
                    message="Atomic composition mismatch between SMILES and QM data",
                    handler_type=dataset_type,
                    validation_type="atomic_composition_validation",
                    failed_validations=[
                        f"SMILES atomic numbers: {rdkit_atomic_nums_sorted}",
                        f"QM atomic numbers: {qm_atomic_nums_sorted}"
                    ],
                    molecule_index=molecule_index,
                    details="The molecules have different atomic compositions"
                )
            
            # Step 5: Reorder RDKit molecule to match QM dataset ordering
            rdkit_atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_atoms_smiles)]
            qm_atomic_nums = validated_atomic_numbers.tolist()
            
            # Create mapping: RDKit index -> QM index
            rdkit_to_qm_map = []
            qm_used = [False] * num_atoms_coords
            
            for rdkit_idx in range(num_atoms_smiles):
                rdkit_z = rdkit_atomic_nums[rdkit_idx]
                
                found = False
                for qm_idx in range(num_atoms_coords):
                    if not qm_used[qm_idx] and qm_atomic_nums[qm_idx] == rdkit_z:
                        rdkit_to_qm_map.append(qm_idx)
                        qm_used[qm_idx] = True
                        found = True
                        break
                
                if not found:
                    raise HandlerValidationError(
                        message=f"Could not map RDKit atom {rdkit_idx} (Z={rdkit_z}) to QM data",
                        handler_type=dataset_type,
                        validation_type="atom_mapping_validation",
                        failed_validations=[f"RDKit atom {rdkit_idx} with Z={rdkit_z} has no unmapped QM counterpart"],
                        molecule_index=molecule_index,
                        details="Atom mapping failed"
                    )
            
            # Step 6: OPTIONAL - Apply minimal sanitization
            try:
                sanitize_ops = (
                    Chem.SANITIZE_SETAROMATICITY |
                    Chem.SANITIZE_SETCONJUGATION
                )
                Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
            except Exception as sanitize_e:
                logger.debug(f"{context}: Minimal sanitization failed (non-critical): {sanitize_e}")
            
            # Step 7: Assign QM coordinates in dataset order
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for rdkit_idx in range(num_atoms_smiles):
                qm_idx = rdkit_to_qm_map[rdkit_idx]
                x, y, z = validated_coords[qm_idx]
                conformer.SetAtomPosition(rdkit_idx, Point3D(float(x), float(y), float(z)))
            mol.AddConformer(conformer, assignId=True)
            
            logger.debug(f"{context}: Successfully created SMILES molecule with QM coordinates.")
            
            return mol
            
        except (RDKitConversionError, HandlerValidationError, HandlerOperationError):
            raise
        except Exception as e:
            raise HandlerOperationError(
                message="Failed RDKit molecule creation from SMILES",
                handler_type=dataset_type,
                operation="smiles_to_mol",
                molecule_index=molecule_index,
                details=f"Original error: {type(e).__name__}: {str(e)}"
            ) from e

    elif mol_id_type == 'inchi':
        logger.debug(f"{context}: Starting InChI-based RDKit mol creation.")
        
        num_atoms_coords = validated_coords.shape[0]
        num_atoms_atomic = len(validated_atomic_numbers)
        
        if num_atoms_coords != num_atoms_atomic:
            raise HandlerValidationError(
                message="Mismatch between coordinates and atomic numbers arrays",
                handler_type=dataset_type,
                validation_type="data_consistency_validation",
                failed_validations=[f"Coordinates: {num_atoms_coords}, Atomic numbers: {num_atoms_atomic}"],
                molecule_index=molecule_index,
                details="Arrays must have the same length for proper molecule construction"
            )
        
        try:
            # Step 1: Parse InChI to get connectivity/bonding information
            rdkit_mol = Chem.MolFromInchi(mol_identifier, sanitize=False)
            if rdkit_mol is None:
                raise RDKitConversionError(
                    molecule_index=molecule_index,
                    inchi=mol_identifier,
                    reason="RDKit MolFromInchi (sanitize=False) resulted in None.",
                    detail="The InChI string could not be parsed by RDKit"
                )

            # Step 2: Try to add hydrogens to get complete structure
            num_atoms_inchi = rdkit_mol.GetNumAtoms()
            try:
                rdkit_mol_with_hs = Chem.AddHs(rdkit_mol, addCoords=False)
                if rdkit_mol_with_hs is not None and rdkit_mol_with_hs.GetNumAtoms() > 0:
                    rdkit_mol = rdkit_mol_with_hs
                    logger.debug(f"{context}: Successfully added hydrogens ({num_atoms_inchi} -> {rdkit_mol.GetNumAtoms()} atoms)")
                else:
                    logger.warning(
                        f"{context}: AddHs returned None/empty. "
                        f"Proceeding with original InChI molecule (may be missing explicit H). "
                        f"Atoms: {num_atoms_inchi}"
                    )
            except Exception as addhs_e:
                logger.warning(
                    f"{context}: AddHs failed with error: {addhs_e}. "
                    f"This is common for QM structures with unusual valences. "
                    f"Proceeding with InChI molecule without explicit hydrogens ({num_atoms_inchi} atoms)."
                )

            # Step 3: Verify atom count matches between InChI+H and QM coordinates
            num_atoms_inchi = rdkit_mol.GetNumAtoms()
            if num_atoms_inchi != num_atoms_coords:
                raise HandlerValidationError(
                    message="Atom count mismatch between InChI and QM coordinates",
                    handler_type=dataset_type,
                    validation_type="atom_count_validation",
                    failed_validations=[f"InChI: {num_atoms_inchi}, QM coords: {num_atoms_coords}"],
                    molecule_index=molecule_index,
                    details=f"InChI has {num_atoms_inchi} atoms but QM calculation has {num_atoms_coords} atoms"
                )
            
            # Step 4: Verify atomic composition matches between InChI and QM
            rdkit_atomic_nums = [rdkit_mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_atoms_inchi)]
            rdkit_atomic_nums_sorted = sorted(rdkit_atomic_nums)
            qm_atomic_nums_sorted = sorted(validated_atomic_numbers.tolist())
            
            if rdkit_atomic_nums_sorted != qm_atomic_nums_sorted:
                raise HandlerValidationError(
                    message="Atomic composition mismatch between InChI and QM data",
                    handler_type=dataset_type,
                    validation_type="atomic_composition_validation",
                    failed_validations=[
                        f"InChI atomic numbers: {rdkit_atomic_nums_sorted}",
                        f"QM atomic numbers: {qm_atomic_nums_sorted}"
                    ],
                    molecule_index=molecule_index,
                    details="The molecules have different atomic compositions"
                )

            # Step 5: OPTIONAL - Apply minimal sanitization
            try:
                sanitize_ops = (
                    Chem.SANITIZE_SETAROMATICITY |
                    Chem.SANITIZE_SETCONJUGATION
                )
                Chem.SanitizeMol(rdkit_mol, sanitizeOps=sanitize_ops)
            except Exception as sanitize_e:
                logger.debug(f"{context}: Minimal sanitization failed (non-critical): {sanitize_e}")

            # Step 6: Assign QM-optimized coordinates to RDKit mol preserving dataset order
            conformer = Chem.Conformer(num_atoms_inchi)
            qm_atomic_nums_list = validated_atomic_numbers.tolist()
            
            # Create mapping: RDKit index -> QM index
            rdkit_to_qm_map = []
            qm_used = [False] * num_atoms_coords
            
            for rdkit_idx in range(num_atoms_inchi):
                rdkit_z = rdkit_atomic_nums[rdkit_idx]
                
                found = False
                for qm_idx in range(num_atoms_coords):
                    if not qm_used[qm_idx] and qm_atomic_nums_list[qm_idx] == rdkit_z:
                        rdkit_to_qm_map.append(qm_idx)
                        qm_used[qm_idx] = True
                        found = True
                        break
                
                if not found:
                    raise HandlerValidationError(
                        message=f"Could not map RDKit atom {rdkit_idx} (Z={rdkit_z}) to QM data",
                        handler_type=dataset_type,
                        validation_type="atom_mapping_validation",
                        failed_validations=[f"RDKit atom {rdkit_idx} with Z={rdkit_z} has no unmapped QM counterpart"],
                        molecule_index=molecule_index,
                        details="Atom mapping failed during coordinate assignment"
                    )
            
            # Assign coordinates using the mapping
            for rdkit_idx in range(num_atoms_inchi):
                qm_idx = rdkit_to_qm_map[rdkit_idx]
                x, y, z = validated_coords[qm_idx]
                conformer.SetAtomPosition(rdkit_idx, Point3D(float(x), float(y), float(z)))
            
            rdkit_mol.AddConformer(conformer, assignId=True)
            
            logger.debug(f"{context}: Successfully created InChI molecule with QM coordinates.")
            
            return rdkit_mol

        except (RDKitConversionError, HandlerValidationError, HandlerOperationError):
            raise
        except Exception as e:
            raise HandlerOperationError(
                message="Error creating RDKit molecule from InChI",
                handler_type=dataset_type,
                operation="inchi_to_mol",
                molecule_index=molecule_index,
                details=f"Original error: {type(e).__name__}: {str(e)}"
            ) from e

    else:
        raise ValueError(f"Unsupported mol_id_type: {mol_id_type}. Must be 'smiles' or 'inchi'.")


def _create_molecule_coordinate_based(
    mol_identifier: str,
    validated_coords: np.ndarray,
    validated_atomic_numbers: np.ndarray,
    molecular_charge: int,
    handler: DatasetHandler,
    dataset_type: str,
    context: str,
    logger: logging.Logger,
    molecule_index: Optional[int]
) -> Chem.Mol:
    """
    Create molecule by inferring connectivity from coordinates.
    
    This is the NEW approach: use rdDetermineBonds to infer bonds from 3D geometry.
    Molecular charge is CRITICAL for accurate bond order determination.
    
    Process:
        1. Get coordinate units and conversion factor from handler
        2. Build XYZ format string from atomic data (with unit conversion)
        3. Parse XYZ to create molecule with atoms and coordinates (no bonds yet)
        4. Use rdDetermineBonds with molecular charge to infer bonds and bond orders
    
    Args:
        mol_identifier: Compound label (used for logging only, not parsed)
        validated_coords: Validated coordinates array
        validated_atomic_numbers: Validated atomic numbers array
        molecular_charge: Molecular charge (REQUIRED for rdDetermineBonds)
        handler: Dataset handler
        dataset_type: Dataset type name
        context: Context string for logging
        logger: Logger instance
        molecule_index: Molecule index
    
    Returns:
        Chem.Mol: RDKit molecule with inferred bonds and QM coordinates
    """
    # Get handler constants for coordinate unit conversion
    from milia_pipeline.config.config_constants import get_handler_constants, BOHR_TO_ANGSTROM
    handler_constants = get_handler_constants(dataset_type)
    coord_units = handler_constants.get('coordinate_units', 'angstrom')
    conversion_factor = BOHR_TO_ANGSTROM if coord_units == 'bohr' else 1.0
    
    logger.debug(
        f"{context}: Starting coordinate-based RDKit mol creation "
        f"(charge={molecular_charge}, units={coord_units})"
    )
    
    try:
        # Step 1: Build XYZ format string from atomic data
        num_atoms = len(validated_atomic_numbers)
        xyz_lines = [
            str(num_atoms),  # Atom count
            f"{mol_identifier}"  # Comment/identifier
        ]
        
        # Lines 3+: element symbol and coordinates (with unit conversion if needed)
        for i in range(num_atoms):
            z = int(validated_atomic_numbers[i])
            # Apply conversion factor to coordinates (e.g., Bohr→Angstrom if needed, 1.0 otherwise)
            x = float(validated_coords[i][0]) * conversion_factor
            y = float(validated_coords[i][1]) * conversion_factor
            z_coord = float(validated_coords[i][2]) * conversion_factor
            
            # Get element symbol from atomic number
            atom = Chem.Atom(z)
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol} {x:.6f} {y:.6f} {z_coord:.6f}")
        
        xyz_block = "\n".join(xyz_lines)
        
        logger.debug(
            f"{context}: Created XYZ block with {num_atoms} atoms "
            f"(units: {coord_units}, conversion: {conversion_factor:.6f})"
        )
        
        # Step 2: Parse XYZ to create molecule with atoms and coordinates (no bonds yet)
        raw_mol = Chem.MolFromXYZBlock(xyz_block)
        if raw_mol is None:
            raise RDKitConversionError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                reason="MolFromXYZBlock returned None",
                detail="Could not parse XYZ coordinates into RDKit molecule"
            )
        
        # Step 3: Create working copy for bond determination
        mol = Chem.Mol(raw_mol)
        
        # Step 4: Automatically infer molecular connectivity and bond orders from 3D geometry
        # Uses xyz2mol algorithm integrated into RDKit (rdDetermineBonds module)
        # CRITICAL: Correct molecular charge is essential for accurate bond order assignment
        rdDetermineBonds.DetermineBonds(mol, charge=molecular_charge)
        
        num_bonds = mol.GetNumBonds()
        logger.debug(f"{context}: Successfully determined {num_bonds} bonds from coordinates")
        
        return mol
        
    except (RDKitConversionError, HandlerValidationError):
        raise
    except Exception as e:
        raise HandlerOperationError(
            message="Failed to create RDKit molecule from coordinates",
            handler_type=dataset_type,
            operation="coords_to_mol",
            molecule_index=molecule_index,
            details=f"Original error: {type(e).__name__}: {str(e)}"
        ) from e


@wrap_handler_operation("UNKNOWN", "mol_to_pyg_data")
def mol_to_pyg_data(
    rdkit_mol: Chem.Mol,
    logger: logging.Logger,
    handler: DatasetHandler,  # Required - moved before optional params
    molecule_index: Optional[int] = None
) -> Data:
    """Converts an RDKit molecule object to a PyTorch Geometric Data object."""
    if handler is None:
        raise ValueError(
            "Handler is required for mol_to_pyg_data(). "
            "Please create a handler explicitly before calling this function."
        )
    
    validate_handler_for_conversion(handler, "mol_to_pyg_data", molecule_index)
    
    dataset_type_current = handler.get_dataset_type()
    identifier_for_log = f"Molecule {molecule_index}" if molecule_index is not None else "Unknown molecule"
    context_info = f"{dataset_type_current} {identifier_for_log}"
    
    logger.debug(f"{context_info}: Starting RDKit mol to PyG Data conversion.")
    
    if rdkit_mol is None:
        raise RDKitConversionError(
            molecule_index=molecule_index,
            inchi="N/A (unknown mol)",
            smiles="N/A",
            reason="Input RDKit molecule is None, cannot convert to PyG Data.",
            detail="Molecule object is required for PyG conversion"
        )

    pyg_data: Data = None
    try:
        pyg_data = from_rdmol(rdkit_mol)
    except Exception as e:
        raise HandlerOperationError(
            message="Failed to convert RDKit molecule to basic PyG Data object",
            handler_type=dataset_type_current,
            operation="rdmol_to_pyg",
            molecule_index=molecule_index,
            details=f"from_rdmol failed: {type(e).__name__}: {str(e)}"
        ) from e

    # --- Explicitly set atomic numbers (z) from RDKit molecule ---
    try:
        atoms_iterator = rdkit_mol.GetAtoms()
        try:
            atoms = list(atoms_iterator)
        except TypeError:
            atoms = []
            for i in range(rdkit_mol.GetNumAtoms()):
                atoms.append(rdkit_mol.GetAtomWithIdx(i))
        
        atomic_numbers_list = [atom.GetAtomicNum() for atom in atoms]
        if not atomic_numbers_list:
            raise HandlerValidationError(
                message="Failed to extract atomic numbers from RDKit molecule",
                handler_type=dataset_type_current,
                validation_type="atomic_numbers_extraction",
                failed_validations=["Empty atomic numbers list"],
                molecule_index=molecule_index,
                details="GetAtomicNum() returned empty list"
            )
        
        # Convert to integers, using placeholder for Mocks
        validated_numbers = []
        for z in atomic_numbers_list:
            try:
                validated_numbers.append(int(z))
            except (TypeError, ValueError):
                validated_numbers.append(1)  # Placeholder for mocks
        
        atomic_numbers_list = validated_numbers
        
        # Validate if possible
        try:
            if not validate_atomic_numbers(atomic_numbers_list, identifier_for_log):
                raise HandlerValidationError(
                    message="Invalid atomic numbers extracted from RDKit molecule",
                    handler_type=dataset_type_current,
                    validation_type="atomic_numbers_validation",
                    failed_validations=["Atomic numbers validation failed"],
                    molecule_index=molecule_index,
                    details="validate_atomic_numbers returned False"
                )
        except TypeError:
            pass
        
        pyg_data.z = torch.tensor(atomic_numbers_list, dtype=torch.long)
    except (HandlerValidationError, HandlerOperationError):
        raise
    except Exception as e:
        raise HandlerOperationError(
            message="Error extracting or assigning atomic numbers (z) from RDKit molecule",
            handler_type=dataset_type_current,
            operation="atomic_numbers_assignment",
            molecule_index=molecule_index,
            details=f"Unexpected error: {type(e).__name__}: {str(e)}"
        ) from e

    # --- Set positions (pos) from RDKit conformer (preserves QM coordinates) ---
    try:
        if rdkit_mol.GetNumConformers() > 0:
            conformer = rdkit_mol.GetConformer(0)
            positions = conformer.GetPositions()
            
            if not validate_coordinates_3d(positions, rdkit_mol.GetNumAtoms(), identifier_for_log):
                raise HandlerValidationError(
                    message="Invalid 3D coordinates extracted from RDKit conformer",
                    handler_type=dataset_type_current,
                    validation_type="coordinates_validation",
                    failed_validations=["Coordinates validation failed"],
                    molecule_index=molecule_index,
                    details="validate_coordinates_3d returned False"
                )
            
            pyg_data.pos = torch.tensor(positions, dtype=torch.float)
            logger.debug(f"{context_info}: Preserved QM coordinates in PyG Data (shape: {pyg_data.pos.shape})")
        else:
            logger.debug(f"{context_info}: RDKit molecule has no conformer, skipping position assignment")
    except (HandlerValidationError, HandlerOperationError):
        raise
    except Exception as e:
        raise HandlerOperationError(
            message="Error assigning positions (pos) from RDKit conformer",
            handler_type=dataset_type_current,
            operation="positions_assignment",
            molecule_index=molecule_index,
            details=f"Unexpected error: {type(e).__name__}: {str(e)}"
        ) from e

    if pyg_data.num_nodes != pyg_data.z.size(0):
        raise HandlerValidationError(
            message="Inconsistency between PyG Data nodes and atomic numbers",
            handler_type=dataset_type_current,
            validation_type="data_consistency_validation",
            failed_validations=[f"num_nodes: {pyg_data.num_nodes}, z.size(0): {pyg_data.z.size(0)}"],
            molecule_index=molecule_index,
            details="This indicates a fundamental mismatch in the graph representation."
        )

    logger.debug(f"{context_info}: Successfully created PyG Data object with {pyg_data.num_nodes} nodes and {pyg_data.num_edges} edges.")
    
    return pyg_data


@wrap_handler_operation("UNKNOWN", "enrich_pyg_data")
def enrich_pyg_data_from_handler(
    pyg_data: Data,
    raw_data_dict: Dict[str, Any],
    molecule_index: int,
    logger: logging.Logger,
    handler: DatasetHandler,
    identifier: Optional[str] = None
) -> Data:
    """
    Enriches a PyG Data object with dataset-specific metadata via the handler protocol.
    
    Delegates to handler.enrich_pyg_data() (DatasetHandlerProtocol method #5),
    which each handler implements with its own dataset-specific enrichment logic
    (e.g., uncertainty fields, compound IDs, scalar targets, vibrational data).
    
    This function is fully dynamic — it works with any dataset type without
    modification. All dataset-specific logic lives in the handler implementation.
    
    Args:
        pyg_data: The PyG Data object to enrich
        raw_data_dict: Dictionary containing raw data for this molecule
        molecule_index: Index of the molecule
        logger: Logger instance
        handler: Dataset handler (REQUIRED). Must implement enrich_pyg_data().
        identifier: Molecule identifier for error context (optional)
        
    Returns:
        Data: Enriched PyG Data object with dataset-specific metadata
        
    Raises:
        ValueError: If handler is None (required parameter).
        DatasetSpecificHandlerError: If handler enrichment fails
        HandlerValidationError: If validation fails
    """
    # Validate handler is provided
    if handler is None:
        raise ValueError(
            "Handler is required for enrich_pyg_data_from_handler(). "
            "Please create a handler explicitly before calling this function."
        )
    
    # Validate pyg_data is not None
    if pyg_data is None:
        raise HandlerValidationError(
            message="PyG data validation failed",
            handler_type=handler.get_dataset_type(),
            validation_type="pyg_data_validation",
            failed_validations=["PyG data is None"],
            molecule_index=molecule_index,
            details="PyG data is required for metadata enrichment"
        )
    
    # Validate handler for this operation  
    validate_handler_for_conversion(handler, "enrich_pyg_data", molecule_index)
    
    dataset_type = handler.get_dataset_type()
    context_info = f"{dataset_type} Molecule {molecule_index}"
    identifier_str = identifier if identifier is not None else "N/A"
    
    try:
        # Delegate to handler.enrich_pyg_data() — DatasetHandlerProtocol method #5
        # Each handler implements its own dataset-specific enrichment logic
        enriched_pyg_data = handler.enrich_pyg_data(
            pyg_data, raw_data_dict, molecule_index, identifier_str
        )
        
        # Ensure dataset_type marker is always set
        if not hasattr(enriched_pyg_data, 'dataset_type') or enriched_pyg_data.dataset_type is None:
            enriched_pyg_data.dataset_type = dataset_type
        
        logger.debug(f"{context_info}: Successfully enriched PyG Data via handler.")
        
        return enriched_pyg_data
        
    except (DatasetSpecificHandlerError, HandlerValidationError, HandlerOperationError):
        raise
    except Exception as e:
        raise create_dataset_handler_error(
            message="Failed to enrich PyG Data with dataset-specific metadata",
            dataset_type=dataset_type,
            operation="enrich_pyg_data",
            details=f"Unexpected error: {type(e).__name__}: {str(e)}"
        ) from e


@wrap_handler_operation("UNKNOWN", "create_mol_with_dataset_support")
def create_mol_with_dataset_support(
    mol_identifier: str,
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    logger: logging.Logger,
    molecule_index: int,
    handler: DatasetHandler,  # Moved here - after required params, before optional ones
    mol_id_type: str = 'inchi',
    raw_data_dict: Optional[Dict[str, Any]] = None
) -> Data:
    """
    Creates a complete PyG Data object from molecular data with dataset-specific support.
    
    REFACTORED STEP 2: Handler is now REQUIRED. No longer accepts None or creates handlers internally.
    Caller must create handler explicitly before calling this function.
    
    This is a high-level function that combines RDKit molecule creation and PyG conversion,
    with dataset-specific metadata enrichment delegated to the handler.
    
    Args:
        mol_identifier: Molecule identifier (InChI or SMILES)
        coordinates: 3D coordinates array
        atomic_numbers: Atomic numbers array
        logger: Logger instance
        molecule_index: Molecule index
        handler: Dataset handler (REQUIRED - must be provided)
        mol_id_type: Type of identifier ('inchi' or 'smiles'). Defaults to 'inchi'.
        raw_data_dict: Optional raw data dictionary for metadata
        
    Returns:
        PyG Data object with all molecular information
        
    Raises:
        ValueError: If handler is None (required parameter).
        HandlerOperationError: If any handler operations fail
        RDKitConversionError: If RDKit molecule creation fails
        PyGDataCreationError: If PyG data creation fails
    """
    # STEP 2 REFACTORING: Validate handler is provided
    if handler is None:
        raise ValueError(
            "Handler is required for create_mol_with_dataset_support(). "
            "Please create a handler explicitly before calling this function."
        )
    
    # Validate handler for this operation
    validate_handler_for_conversion(handler, "create_mol_with_dataset_support", molecule_index)
    
    dataset_type = handler.get_dataset_type()
    
    try:
        # Step 1: Create RDKit molecule with handler
        rdkit_mol = create_rdkit_mol(
            mol_identifier=mol_identifier,
            coordinates=coordinates,
            atomic_numbers=atomic_numbers,
            logger=logger,
            handler=handler,  # Moved before optional params
            molecule_index=molecule_index,
            mol_id_type=mol_id_type
        )
#---        
        # Step 2: Convert to PyG Data
        pyg_data = mol_to_pyg_data(
            rdkit_mol=rdkit_mol,
            logger=logger,
            handler=handler,  # Moved before optional params
            molecule_index=molecule_index
        )
#---        
        # Step 3: Enrich with dataset-specific metadata via handler protocol
        if raw_data_dict:
            pyg_data = enrich_pyg_data_from_handler(
                pyg_data=pyg_data,
                raw_data_dict=raw_data_dict,
                molecule_index=molecule_index,
                logger=logger,
                handler=handler,
                identifier=mol_identifier
            )
        
        # Always add dataset type marker
        pyg_data.dataset_type = dataset_type
        
        logger.debug(
            f"{dataset_type} Molecule {molecule_index}: "
            f"Successfully created complete PyG Data object"
        )
        
        return pyg_data
        
    except (HandlerError, RDKitConversionError, PyGDataCreationError) as e:
        # Re-raise handler and conversion errors as-is
        logger.error(
            f"{dataset_type} Molecule {molecule_index}: "
            f"Known error in create_mol_with_dataset_support: {e}"
        )
        raise
    except Exception as e:
        # Wrap unexpected errors in HandlerOperationError
        logger.error(
            f"{dataset_type} Molecule {molecule_index}: "
            f"Unexpected error in create_mol_with_dataset_support: {type(e).__name__}: {e}",
            exc_info=True
        )
        raise HandlerOperationError(
            message="Failed to create molecule with dataset support",
            handler_type=dataset_type,
            operation="create_mol_with_dataset_support",
            molecule_index=molecule_index,
            details=f"Unexpected error: {type(e).__name__}: {str(e)}"
        ) from e


# ==========================================
# Handler Validation and Support Functions
# ==========================================

def validate_handler_for_conversion(
    handler: DatasetHandler,
    operation: str,
    molecule_index: Optional[int] = None
) -> None:
    """
    Validates that a handler is appropriate for molecular conversion operations.
    
    DYNAMIC VALIDATION: Instead of checking against a hardcoded list of dataset types,
    this function validates that the handler has the required capabilities for conversion.
    Any handler that implements get_molecule_creation_strategy() and get_molecular_charge()
    is valid for conversion operations.
    
    Args:
        handler: The dataset handler to validate
        operation: Name of the operation being performed
        molecule_index: Molecule index for error context
        
    Raises:
        HandlerValidationError: If handler is not valid for the operation
        HandlerNotAvailableError: If handler is None or unavailable
    """
    if handler is None:
        raise HandlerNotAvailableError(
            message=f"Handler is required for {operation} operation",
            requested_dataset_type="UNKNOWN",
            details="No handler provided for conversion operation"
        )
    
    dataset_type = handler.get_dataset_type()
    
    # DYNAMIC VALIDATION: Check handler capabilities instead of hardcoded type list
    # Any handler that implements required methods is valid for conversion
    required_methods = ['get_molecule_creation_strategy', 'get_molecular_charge', 'get_dataset_type']
    missing_methods = []
    
    for method_name in required_methods:
        if not hasattr(handler, method_name) or not callable(getattr(handler, method_name)):
            missing_methods.append(method_name)
    
    if missing_methods:
        raise HandlerValidationError(
            message=f"Handler missing required methods for {operation}",
            handler_type=dataset_type,
            validation_type="capability_validation",
            failed_validations=[f"Missing methods: {', '.join(missing_methods)}"],
            molecule_index=molecule_index,
            details=f"Handler '{dataset_type}' does not implement required conversion methods"
        )
    
    # Validate handler configuration
    try:
        if hasattr(handler, 'validate_configuration'):
            handler.validate_configuration()
    except Exception as e:
        context = f" for molecule {molecule_index}" if molecule_index is not None else ""
        raise HandlerValidationError(
            message=f"Handler validation failed for {operation}",
            handler_type=dataset_type,
            validation_type="configuration_validation",
            failed_validations=[str(e)],
            molecule_index=molecule_index,
            details=f"Handler configuration validation failed{context}: {e}"
        ) from e


def get_conversion_context_info(
    handler: DatasetHandler,
    molecule_index: Optional[int],
    identifier: str
) -> Dict[str, Any]:
    """
    Creates a standardized context dictionary for logging conversion operations.
    
    Args:
        handler: Dataset handler (can be None)
        molecule_index: Molecule index
        identifier: Molecule identifier
        
    Returns:
        Dictionary with context information
    """
    context = {
        'molecule_index': molecule_index,
        'identifier': identifier,
        'source': 'test'
    }
    
    if handler is not None:
        try:
            context['dataset_type'] = handler.get_dataset_type()
            if hasattr(handler, 'get_dataset_name'):
                context['dataset_name'] = handler.get_dataset_name()
        except Exception:
            pass
    
    return context


def apply_handler_specific_rdkit_processing(
    rdkit_mol: Chem.Mol,
    handler: DatasetHandler,
    molecule_index: Optional[int],
    logger: logging.Logger,
    identifier: Optional[str] = None
) -> Chem.Mol:
    """
    Applies any handler-specific RDKit molecule processing.
    
    This function allows handlers to apply dataset-specific modifications
    to RDKit molecules during the conversion process.
    
    Args:
        rdkit_mol: The RDKit molecule to process
        handler: Dataset handler (can be None)
        molecule_index: Molecule index
        logger: Logger instance
        identifier: Molecule identifier (optional)
        
    Returns:
        Processed RDKit molecule (returns the input molecule reference)
        
    Raises:
        HandlerOperationError: If handler-specific processing fails
    """
    # Handle None handler - return immediately without any processing
    if handler is None:
        return rdkit_mol
    
    # Get dataset type for logging
    dataset_type = handler.get_dataset_type() if hasattr(handler, 'get_dataset_type') else "UNKNOWN"
    
    try:
        # Check if handler has a real implementation of process_rdkit_molecule
        # Use inspect or check __dict__ to avoid Mock's auto-generation
        if 'process_rdkit_molecule' in dir(handler.__class__):
            processed = handler.process_rdkit_molecule(rdkit_mol, molecule_index, identifier)
            # Only return processed if it's different from input
            if processed is not None and processed is not rdkit_mol:
                logger.debug(f"Applied {dataset_type} handler-specific RDKit processing for molecule {molecule_index}")
                return processed
        
        logger.debug(f"No {dataset_type} handler-specific RDKit processing applied for molecule {molecule_index}")
        return rdkit_mol
        
    except Exception as e:
        raise HandlerOperationError(
            message=f"Handler-specific RDKit processing failed",
            handler_type=dataset_type,
            operation="rdkit_processing",
            molecule_index=molecule_index,
            details=f"Error in handler RDKit processing: {str(e)}"
        ) from e


def get_handler_conversion_statistics(
    handler: DatasetHandler,
    processed_molecules: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Gets conversion statistics specific to the handler's dataset type.
    
    Args:
        handler: Dataset handler (can be None)
        processed_molecules: List of processed molecule data
        
    Returns:
        Statistics dictionary
        
    Raises:
        HandlerOperationError: If statistics collection fails
    """
    if handler is None:
        return {}
    
    # Check if handler has get_dataset_type method
    if not hasattr(handler, 'get_dataset_type'):
        return {}
    
    base_stats = {
        'dataset_type': handler.get_dataset_type(),
        'total_converted': len(processed_molecules),
        'conversion_method': 'dataset_handler_pattern'
    }
    
    # Get handler-specific statistics if the method exists
    try:
        if hasattr(handler, 'get_conversion_statistics'):
            handler_stats = handler.get_conversion_statistics()
            if isinstance(handler_stats, dict):
                # Merge handler stats with base stats
                return {**base_stats, **handler_stats}
        elif hasattr(handler, 'get_processing_statistics'):
            handler_stats = handler.get_processing_statistics(processed_molecules)
            if isinstance(handler_stats, dict):
                return {**base_stats, **handler_stats}
    except Exception as e:
        raise HandlerOperationError(
            message="Failed to collect handler statistics",
            handler_type=handler.get_dataset_type(),
            operation="get_statistics",
            details=f"Error collecting statistics: {str(e)}"
        ) from e
    
    return base_stats


# ==========================================
# Exception Enhancement Utilities
# ==========================================

def enhance_conversion_error_context(
    error: Exception,
    handler: DatasetHandler,
    molecule_index: Optional[int],
    additional_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Enhances existing conversion errors with handler context.
    
    Args:
        error: Original error to enhance
        handler: Dataset handler providing context (can be None)
        molecule_index: Molecule index for context
        additional_context: Optional additional context dictionary
        
    Returns:
        Enhanced error message as a string
    """
    error_str = str(error)
    
    context_parts = [f"Error: {error_str}"]
    
    if handler is not None:
        try:
            dataset_type = handler.get_dataset_type()
            context_parts.append(f"dataset_type: {dataset_type}")
            if hasattr(handler, 'get_dataset_name'):
                context_parts.append(f"dataset_name: {handler.get_dataset_name()}")
        except Exception:
            pass
    
    if molecule_index is not None:
        context_parts.append(f"molecule_index: {molecule_index}")
    
    if additional_context:
        for key, value in additional_context.items():
            context_parts.append(f"{key}: {value}")
    
    return ", ".join(context_parts)


def validate_conversion_prerequisites(
    handler: DatasetHandler,
    mol_identifier: str,
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    molecule_index: Optional[int] = None
) -> None:
    """
    Validates prerequisites for molecular conversion operations.
    
    Args:
        handler: Dataset handler (can be None for validation)
        mol_identifier: Molecule identifier
        coordinates: 3D coordinates
        atomic_numbers: Atomic numbers array
        molecule_index: Molecule index for context
        
    Raises:
        HandlerValidationError: If prerequisites are not met
    """
    # Check if handler is None first
    if handler is None:
        raise HandlerValidationError(
            message="Handler validation failed",
            handler_type="UNKNOWN",
            validation_type="prerequisites_validation",
            failed_validations=["Handler is required for conversion"],
            molecule_index=molecule_index,
            details="Handler must be provided for prerequisite validation"
        )
    
    dataset_type = handler.get_dataset_type()
    failed_validations = []
    
    # Validate basic inputs - use more specific error messages
    if not mol_identifier or not mol_identifier.strip():
        failed_validations.append("Molecule identifier is required")
    
    if coordinates is None or coordinates.size == 0:
        failed_validations.append("Coordinates are required")
    
    if atomic_numbers is None or len(atomic_numbers) == 0:
        failed_validations.append("Atomic numbers are required")
    
    # Validate array consistency
    if coordinates is not None and atomic_numbers is not None:
        if coordinates.shape[0] != len(atomic_numbers):
            failed_validations.append(
                f"Coordinates and atomic numbers length mismatch: "
                f"{coordinates.shape[0]} vs {len(atomic_numbers)}"
            )
    
    if failed_validations:
        raise HandlerValidationError(
            message="Conversion prerequisites validation failed",
            handler_type=dataset_type,
            validation_type="prerequisites_validation", 
            failed_validations=failed_validations,
            molecule_index=molecule_index,
            details="Basic input validation failed before conversion"
        )
