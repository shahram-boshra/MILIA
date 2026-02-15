# milia_pipeline/descriptors/descriptor_calculator.py

"""
Descriptor calculation engine with batch processing, caching, and error handling.

This module provides the core calculation functionality for molecular descriptors,
handling requirements (3D coordinates, charges), errors, and performance optimization
through caching.

Pydantic V2 Migration:
    - Migrated CalculationResult from @dataclass to Pydantic BaseModel (mutable)
    - Migrated BatchCalculationResult from @dataclass to Pydantic BaseModel (mutable)
    - Added to_dict() method wrapping model_dump() for backward compatibility
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: Milia Team
Version: 1.1.0
"""

import hashlib
import logging
from typing import Any

from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem

from milia_pipeline.descriptors.descriptor_registry import DescriptorRegistry
from milia_pipeline.descriptors.descriptor_validator import DescriptorValidator
from milia_pipeline.exceptions import DescriptorCalculationError

logger = logging.getLogger(__name__)


class CalculationResult(BaseModel):
    """
    Result of a descriptor calculation.

    Pydantic V2 Migration:
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Added to_dict() method wrapping model_dump() for backward compatibility
        - NON-BREAKING: Same constructor API and attribute access preserved

    Attributes:
        success: Whether calculation succeeded
        value: Calculated descriptor value (None if failed)
        descriptor_name: Name of the descriptor
        error_message: Error message if calculation failed
        computation_time: Time taken for calculation in seconds
    """

    success: bool
    value: float | None
    descriptor_name: str
    error_message: str | None = None
    computation_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().

        Returns:
            Dictionary with all 5 fields: success, value, descriptor_name,
            error_message, computation_time
        """
        return self.model_dump()


class BatchCalculationResult(BaseModel):
    """
    Results of batch descriptor calculations.

    Pydantic V2 Migration:
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Added to_dict() method wrapping model_dump() for backward compatibility
        - NON-BREAKING: Same constructor API and attribute access preserved

    Attributes:
        successful: Dictionary mapping descriptor_name -> calculated value
        failed: Dictionary mapping descriptor_name -> error message
        total_time: Total time for batch calculation in seconds
        molecules_processed: Number of molecules processed
    """

    successful: dict[str, float]
    failed: dict[str, str]
    total_time: float
    molecules_processed: int

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().

        Returns:
            Dictionary with all 4 fields: successful, failed, total_time,
            molecules_processed
        """
        return self.model_dump()


class DescriptorCalculator:
    """
    Batch calculation engine for molecular descriptors.

    Features:
    - Single molecule and batch processing
    - 3D conformer generation when needed
    - Charge computation on demand
    - Result caching for performance
    - Comprehensive error handling
    """

    def __init__(
        self,
        registry: DescriptorRegistry | None = None,
        validator: DescriptorValidator | None = None,
        enable_cache: bool = True,
        fallback_on_error: bool = True,
        generate_conformers: bool = True,
    ):
        """
        Initialize descriptor calculator.

        Args:
            registry: Descriptor registry (uses global if None)
            validator: Descriptor validator (creates if None)
            enable_cache: Enable result caching
            fallback_on_error: Continue on calculation errors
            generate_conformers: Auto-generate 3D conformers for geometric descriptors
        """
        self.registry = registry or DescriptorRegistry.get_instance()
        self.validator = validator or DescriptorValidator()
        self.enable_cache = enable_cache
        self.fallback_on_error = fallback_on_error
        self.generate_conformers = generate_conformers

        # Statistics
        self._stats = {
            "total_calculations": 0,
            "successful": 0,
            "failed": 0,
            "cache_hits": 0,
            "conformers_generated": 0,
        }

        # Cache for expensive calculations
        self._cache: dict[str, float] = {}

        logger.info(
            f"DescriptorCalculator initialized (cache={enable_cache}, "
            f"fallback={fallback_on_error}, conformers={generate_conformers})"
        )

    def calculate_single(
        self, mol: Chem.Mol, descriptor_name: str, mol_identifier: str = "unknown"
    ) -> CalculationResult:
        """
        Calculate a single descriptor for a molecule.

        Args:
            mol: RDKit molecule object
            descriptor_name: Name of descriptor to calculate
            mol_identifier: Identifier for error messages

        Returns:
            CalculationResult with success status and value/error
        """
        import time

        start_time = time.time()

        self._stats["total_calculations"] += 1

        # Check cache
        if self.enable_cache:
            cache_key = self._get_cache_key(mol, descriptor_name)
            if cache_key in self._cache:
                self._stats["cache_hits"] += 1
                return CalculationResult(
                    success=True,
                    value=self._cache[cache_key],
                    descriptor_name=descriptor_name,
                    computation_time=time.time() - start_time,
                )

        # Get descriptor function
        if not self.registry.has_descriptor(descriptor_name):
            return CalculationResult(
                success=False,
                value=None,
                descriptor_name=descriptor_name,
                error_message=f"Descriptor '{descriptor_name}' not found in registry",
            )

        descriptor_func = self.registry.get_descriptor(descriptor_name)

        # Check requirements
        is_valid, missing_reqs = self.validator.check_requirements(mol, descriptor_name)

        # Handle 3D requirements
        if not is_valid and "3D coordinates" in missing_reqs:
            if self.generate_conformers:
                mol = self._ensure_3d_conformer(mol, mol_identifier)
                is_valid, missing_reqs = self.validator.check_requirements(mol, descriptor_name)
            else:
                return CalculationResult(
                    success=False,
                    value=None,
                    descriptor_name=descriptor_name,
                    error_message=f"Requires 3D coordinates: {missing_reqs}",
                )

        # Handle charge requirements
        if not is_valid and "partial charges" in missing_reqs:
            mol = self._compute_charges(mol, mol_identifier)
            is_valid, missing_reqs = self.validator.check_requirements(mol, descriptor_name)

        # Final requirement check
        if not is_valid:
            return CalculationResult(
                success=False,
                value=None,
                descriptor_name=descriptor_name,
                error_message=f"Requirements not met: {missing_reqs}",
            )

        # Calculate descriptor
        try:
            value = descriptor_func(mol)

            # Validate result
            is_valid_value, validation_msg = self.validator.validate_value(descriptor_name, value)

            if not is_valid_value:
                self._stats["failed"] += 1
                return CalculationResult(
                    success=False,
                    value=None,
                    descriptor_name=descriptor_name,
                    error_message=f"Invalid value: {validation_msg}",
                )

            # Cache result
            if self.enable_cache:
                self._cache[cache_key] = value

            self._stats["successful"] += 1
            return CalculationResult(
                success=True,
                value=value,
                descriptor_name=descriptor_name,
                computation_time=time.time() - start_time,
            )

        except Exception as e:
            self._stats["failed"] += 1
            error_msg = f"Calculation failed: {type(e).__name__}: {str(e)}"
            logger.debug(f"Descriptor '{descriptor_name}' failed for {mol_identifier}: {error_msg}")

            return CalculationResult(
                success=False,
                value=None,
                descriptor_name=descriptor_name,
                error_message=error_msg,
                computation_time=time.time() - start_time,
            )

    def calculate_batch(
        self, mol: Chem.Mol, descriptor_names: list[str], mol_identifier: str = "unknown"
    ) -> BatchCalculationResult:
        """
        Calculate multiple descriptors for a single molecule.

        Args:
            mol: RDKit molecule object
            descriptor_names: List of descriptor names
            mol_identifier: Identifier for error messages

        Returns:
            BatchCalculationResult with successful and failed descriptors
        """
        import time

        start_time = time.time()

        successful: dict[str, float] = {}
        failed: dict[str, str] = {}

        for desc_name in descriptor_names:
            result = self.calculate_single(mol, desc_name, mol_identifier)

            if result.success:
                successful[desc_name] = result.value
            else:
                if self.fallback_on_error:
                    failed[desc_name] = result.error_message
                else:
                    raise DescriptorCalculationError(
                        f"Descriptor '{desc_name}' calculation failed: {result.error_message}",
                        descriptor_name=desc_name,
                        molecule_id=mol_identifier,
                    )

        total_time = time.time() - start_time

        return BatchCalculationResult(
            successful=successful, failed=failed, total_time=total_time, molecules_processed=1
        )

    def calculate_for_molecules(
        self, molecules: list[tuple[Chem.Mol, str]], descriptor_names: list[str]
    ) -> list[BatchCalculationResult]:
        """
        Calculate descriptors for multiple molecules.

        Args:
            molecules: List of (mol, identifier) tuples
            descriptor_names: List of descriptor names

        Returns:
            List of BatchCalculationResult, one per molecule
        """
        results = []

        for mol, mol_id in molecules:
            result = self.calculate_batch(mol, descriptor_names, mol_id)
            results.append(result)

        return results

    def _ensure_3d_conformer(self, mol: Chem.Mol, mol_identifier: str) -> Chem.Mol:
        """
        Generate 3D conformer if molecule doesn't have one.

        Args:
            mol: RDKit molecule
            mol_identifier: Identifier for logging

        Returns:
            Molecule with 3D conformer
        """
        try:
            # Check if already has 3D conformer
            if mol.GetNumConformers() > 0:
                return mol

            # Create copy to avoid modifying original
            mol_copy = Chem.Mol(mol)

            # Add hydrogens (needed for good conformer generation)
            mol_copy = Chem.AddHs(mol_copy)

            # Generate conformer
            AllChem.EmbedMolecule(mol_copy, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_copy)

            self._stats["conformers_generated"] += 1
            logger.debug(f"Generated 3D conformer for {mol_identifier}")

            return mol_copy

        except Exception as e:
            logger.warning(f"Failed to generate conformer for {mol_identifier}: {e}")
            return mol

    def _compute_charges(self, mol: Chem.Mol, mol_identifier: str) -> Chem.Mol:
        """
        Compute Gasteiger partial charges.

        Args:
            mol: RDKit molecule
            mol_identifier: Identifier for logging

        Returns:
            Molecule with computed charges
        """
        try:
            # Create copy to avoid modifying original
            mol_copy = Chem.Mol(mol)

            # Compute Gasteiger charges
            AllChem.ComputeGasteigerCharges(mol_copy)

            logger.debug(f"Computed charges for {mol_identifier}")
            return mol_copy

        except Exception as e:
            logger.warning(f"Failed to compute charges for {mol_identifier}: {e}")
            return mol

    def _get_cache_key(self, mol: Chem.Mol, descriptor_name: str) -> str:
        """
        Generate cache key for molecule-descriptor pair.

        Args:
            mol: RDKit molecule
            descriptor_name: Descriptor name

        Returns:
            Unique cache key
        """
        # Use SMILES + descriptor name as cache key
        smiles = Chem.MolToSmiles(mol)
        key = f"{smiles}:{descriptor_name}"
        return hashlib.md5(key.encode()).hexdigest()

    def get_statistics(self) -> dict[str, Any]:
        """
        Get calculation statistics.

        Returns:
            Dictionary with statistics
        """
        total = self._stats["total_calculations"]
        if total > 0:
            success_rate = (self._stats["successful"] / total) * 100
            cache_hit_rate = (self._stats["cache_hits"] / total) * 100
        else:
            success_rate = 0.0
            cache_hit_rate = 0.0

        return {
            **self._stats,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        """Clear calculation cache."""
        self._cache.clear()
        logger.info("Descriptor calculation cache cleared")

    def reset_statistics(self):
        """Reset calculation statistics."""
        self._stats = {
            "total_calculations": 0,
            "successful": 0,
            "failed": 0,
            "cache_hits": 0,
            "conformers_generated": 0,
        }
