#!/usr/bin/env python3
"""
Comprehensive test suite for DescriptorCalculator.

This test suite provides production-ready tests for the DescriptorCalculator class
from the descriptor_calculator module. It covers:

- Performance benchmarks for single and batch calculations
- Caching behavior and statistics tracking
- Error handling with fallback mechanisms
- Edge cases (invalid molecules, empty inputs, missing descriptors)
- Pydantic V2 model compatibility (to_dict, model_dump)
- 3D conformer generation and charge computation requirements
- DescriptorCalculationError exception behavior

Tests run in Docker with proper Python path configuration and without mock pollution.
All mocking is done at test-level via pytest fixtures to prevent sys.modules pollution.

Author: Milia Team
Version: 1.1.0
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST (before any project imports)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from rdkit import Chem
from rdkit.Chem import AllChem

# Now import project modules after path is configured
from milia_pipeline.descriptors.descriptor_calculator import (
    DescriptorCalculator,
    CalculationResult,
    BatchCalculationResult
)
from milia_pipeline.exceptions import DescriptorCalculationError


class TestDescriptorPerformance:
    """
    Test descriptor calculation performance, correctness, and edge cases.
    
    This test class covers:
    - Performance benchmarks (single/batch/multi-molecule calculations)
    - Caching behavior and cache statistics
    - Error handling with and without fallback
    - Edge cases (invalid inputs, missing descriptors, requirements)
    - Pydantic V2 model serialization (to_dict)
    - Statistics tracking and reset
    """
    
    @pytest.fixture
    def molecules(self):
        """Generate test molecules"""
        smiles_list = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(C)C(=O)O",  # Isobutyric acid
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ] * 25  # 100 molecules total
        
        return [Chem.MolFromSmiles(smi) for smi in smiles_list]
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock descriptor registry"""
        registry = Mock()
        
        # Mock descriptor functions
        def mock_molwt(mol):
            """Mock molecular weight calculation"""
            return sum(atom.GetMass() for atom in mol.GetAtoms())
        
        def mock_tpsa(mol):
            """Mock TPSA calculation"""
            return 20.0  # Simplified mock value
        
        def mock_num_rotatable_bonds(mol):
            """Mock rotatable bonds calculation"""
            return 3
        
        def mock_num_hdonors(mol):
            """Mock H-bond donors calculation"""
            return 1
        
        def mock_num_hacceptors(mol):
            """Mock H-bond acceptors calculation"""
            return 2
        
        # Map descriptor names to functions
        descriptor_map = {
            "MolWt": mock_molwt,
            "TPSA": mock_tpsa,
            "NumRotatableBonds": mock_num_rotatable_bonds,
            "NumHDonors": mock_num_hdonors,
            "NumHAcceptors": mock_num_hacceptors
        }
        
        registry.has_descriptor.side_effect = lambda name: name in descriptor_map
        registry.get_descriptor.side_effect = lambda name: descriptor_map.get(name)
        
        # Mock list_available_descriptors to return 50 descriptors
        available_descriptors = list(descriptor_map.keys()) + [
            f"MockDescriptor{i}" for i in range(45)
        ]
        registry.list_available_descriptors.return_value = available_descriptors
        
        # For additional mock descriptors, return a simple function
        def default_descriptor(mol):
            return 1.0
        
        original_get = registry.get_descriptor.side_effect
        def get_descriptor_with_default(name):
            result = original_get(name)
            return result if result is not None else default_descriptor
        
        registry.get_descriptor.side_effect = get_descriptor_with_default
        
        return registry
    
    @pytest.fixture
    def mock_validator(self):
        """Create a mock descriptor validator"""
        validator = Mock()
        validator.check_requirements.return_value = (True, [])
        validator.validate_value.return_value = (True, None)
        return validator
    
    @pytest.fixture
    def calculator(self, mock_registry, mock_validator):
        """Create DescriptorCalculator with mocked dependencies"""
        return DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False  # Disable to avoid 3D generation overhead
        )
    
    def test_single_descriptor_speed(self, calculator, molecules):
        """Test speed of single descriptor calculation"""
        start = time.time()
        for mol in molecules:
            result = calculator.calculate_single(mol, "MolWt", "test_mol")
            assert result.success
        elapsed = time.time() - start
        
        # Should calculate 100 molecules in < 1 second
        assert elapsed < 1.0, f"Single descriptor took {elapsed:.3f}s, expected < 1.0s"
        
        per_molecule = elapsed / len(molecules)
        print(f"\nSingle descriptor: {per_molecule*1000:.2f} ms per molecule")
    
    def test_multiple_descriptors_speed(self, calculator, molecules):
        """Test speed of multiple descriptor calculations via batch"""
        descriptors = ["MolWt", "TPSA", "NumRotatableBonds", "NumHDonors", "NumHAcceptors"]
        
        start = time.time()
        for mol in molecules:
            result = calculator.calculate_batch(mol, descriptors, "test_mol")
            assert len(result.successful) == len(descriptors)
        elapsed = time.time() - start
        
        # Should calculate 5 descriptors for 100 molecules in < 2 seconds
        assert elapsed < 2.0, f"Multiple descriptors took {elapsed:.3f}s, expected < 2.0s"
        
        per_molecule = elapsed / len(molecules)
        print(f"\nMultiple descriptors: {per_molecule*1000:.2f} ms per molecule")
    
    def test_batch_calculation_overhead(self, calculator, molecules, mock_registry):
        """Test overhead of batch descriptor calculation"""
        descriptor_names = mock_registry.list_available_descriptors()[:50]
        
        start = time.time()
        results = []
        for mol in molecules[:10]:  # Test with 10 molecules
            result = calculator.calculate_batch(mol, descriptor_names, "test_mol")
            results.append(result)
        elapsed = time.time() - start
        
        # Verify all calculations were attempted
        assert len(results) == 10
        
        print(f"\nBatch calculation (10 mol × 50 desc): {elapsed:.2f} seconds")
        print(f"Per molecule: {elapsed/10:.2f} seconds")
    
    def test_cache_performance(self, calculator, molecules):
        """Test that caching improves performance for repeated calculations"""
        mol = molecules[0]
        descriptor_name = "MolWt"
        
        # First calculation (no cache)
        start = time.time()
        result1 = calculator.calculate_single(mol, descriptor_name, "test_mol")
        first_time = time.time() - start
        
        assert result1.success
        
        # Second calculation (should use cache)
        start = time.time()
        result2 = calculator.calculate_single(mol, descriptor_name, "test_mol")
        cached_time = time.time() - start
        
        assert result2.success
        assert result1.value == result2.value
        
        # Cached calculation should be faster (or at least not slower)
        print(f"\nFirst calculation: {first_time*1000:.2f} ms")
        print(f"Cached calculation: {cached_time*1000:.2f} ms")
        
        # Check statistics
        stats = calculator.get_statistics()
        assert stats['cache_hits'] >= 1
        assert stats['total_calculations'] >= 2
    
    def test_batch_calculation_with_failures(self, mock_registry, mock_validator):
        """Test batch calculation handles failures gracefully"""
        mol = Chem.MolFromSmiles("CCO")
        
        # Create a fresh calculator instance for this test to avoid state pollution
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False
        )
        
        # Define descriptor functions including one that will fail
        def failing_descriptor(mol):
            raise ValueError("Intentional failure")
        
        def working_descriptor(mol):
            return 1.0
        
        # Create descriptor map with failure
        descriptor_map = {
            "WorkingDesc1": working_descriptor,
            "FailingDesc": failing_descriptor,
            "WorkingDesc2": working_descriptor
        }
        
        # Configure mock to use the descriptor map
        mock_registry.has_descriptor.side_effect = lambda name: name in descriptor_map
        mock_registry.get_descriptor.side_effect = lambda name: descriptor_map.get(name)
        
        descriptor_names = ["WorkingDesc1", "FailingDesc", "WorkingDesc2"]
        
        result = calculator.calculate_batch(mol, descriptor_names, "test_mol")
        
        # Verify: should have 2 successful and 1 failed
        assert len(result.successful) == 2
        assert "FailingDesc" in result.failed
        assert "WorkingDesc1" in result.successful
        assert "WorkingDesc2" in result.successful
    
    def test_calculate_for_molecules(self, calculator, molecules):
        """Test multi-molecule calculation performance"""
        # Prepare molecule-identifier tuples
        mol_tuples = [(mol, f"mol_{i}") for i, mol in enumerate(molecules[:10])]
        descriptor_names = ["MolWt", "TPSA", "NumRotatableBonds"]
        
        start = time.time()
        results = calculator.calculate_for_molecules(mol_tuples, descriptor_names)
        elapsed = time.time() - start
        
        assert len(results) == 10
        assert all(isinstance(r, BatchCalculationResult) for r in results)
        
        print(f"\nMulti-molecule calculation (10 molecules, 3 descriptors): {elapsed:.2f} seconds")
    
    def test_statistics_tracking(self, calculator, molecules):
        """Test that statistics are properly tracked"""
        mol = molecules[0]
        
        # Reset statistics
        calculator.reset_statistics()
        stats = calculator.get_statistics()
        assert stats['total_calculations'] == 0
        
        # Perform calculations
        calculator.calculate_single(mol, "MolWt", "test_mol")
        calculator.calculate_single(mol, "TPSA", "test_mol")
        calculator.calculate_single(mol, "MolWt", "test_mol")  # Should hit cache
        
        stats = calculator.get_statistics()
        assert stats['total_calculations'] == 3
        assert stats['successful'] >= 2
        assert stats['cache_hits'] >= 1
        
        print(f"\nStatistics: {stats}")
    
    def test_cache_clear(self, calculator, molecules):
        """Test cache clearing functionality"""
        mol = molecules[0]
        
        # Perform calculation to populate cache
        calculator.calculate_single(mol, "MolWt", "test_mol")
        
        stats_before = calculator.get_statistics()
        assert stats_before['cache_size'] > 0
        
        # Clear cache
        calculator.clear_cache()
        
        stats_after = calculator.get_statistics()
        assert stats_after['cache_size'] == 0
    
    def test_calculator_initialization_options(self, mock_registry, mock_validator):
        """Test different initialization options"""
        # Test with cache disabled
        calc_no_cache = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=False
        )
        assert calc_no_cache.enable_cache is False
        
        # Test with fallback disabled
        calc_no_fallback = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            fallback_on_error=False
        )
        assert calc_no_fallback.fallback_on_error is False
        
        # Test with conformer generation enabled
        calc_with_conformers = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            generate_conformers=True
        )
        assert calc_with_conformers.generate_conformers is True


class TestDescriptorCalculatorEdgeCases:
    """
    Test edge cases and error conditions for DescriptorCalculator.
    
    This class focuses on:
    - Unknown/missing descriptor handling
    - Requirement failures (3D coordinates, charges)
    - Validation failures
    - Exception raising when fallback_on_error=False
    - Empty input handling
    """
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock descriptor registry with configurable behavior"""
        registry = Mock()
        
        def mock_molwt(mol):
            return sum(atom.GetMass() for atom in mol.GetAtoms())
        
        def mock_tpsa(mol):
            return 20.0
        
        descriptor_map = {
            "MolWt": mock_molwt,
            "TPSA": mock_tpsa,
        }
        
        registry.has_descriptor.side_effect = lambda name: name in descriptor_map
        registry.get_descriptor.side_effect = lambda name: descriptor_map.get(name)
        registry.list_available_descriptors.return_value = list(descriptor_map.keys())
        
        return registry
    
    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator with configurable behavior"""
        validator = Mock()
        validator.check_requirements.return_value = (True, [])
        validator.validate_value.return_value = (True, None)
        return validator
    
    def test_calculate_single_unknown_descriptor(self, mock_registry, mock_validator):
        """Test calculate_single with unknown descriptor returns failure result"""
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False
        )
        
        mol = Chem.MolFromSmiles("CCO")
        result = calculator.calculate_single(mol, "UnknownDescriptor", "test_mol")
        
        assert result.success is False
        assert result.value is None
        assert result.descriptor_name == "UnknownDescriptor"
        assert "not found in registry" in result.error_message
    
    def test_calculate_single_3d_requirement_without_conformers(self, mock_registry, mock_validator):
        """Test failure when 3D coordinates required but generate_conformers=False"""
        # Configure validator to report missing 3D coordinates
        mock_validator.check_requirements.return_value = (False, ["3D coordinates"])
        
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False  # Critical: disabled
        )
        
        mol = Chem.MolFromSmiles("CCO")
        result = calculator.calculate_single(mol, "MolWt", "test_mol")
        
        assert result.success is False
        assert "Requires 3D coordinates" in result.error_message
    
    def test_calculate_single_charge_requirement(self, mock_registry, mock_validator):
        """Test handling of partial charge requirements"""
        # First call: missing charges, second call: requirements met after charge computation
        call_count = [0]
        
        def check_requirements_side_effect(mol, desc_name):
            call_count[0] += 1
            if call_count[0] == 1:
                return (False, ["partial charges"])
            return (True, [])
        
        mock_validator.check_requirements.side_effect = check_requirements_side_effect
        
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False
        )
        
        mol = Chem.MolFromSmiles("CCO")
        result = calculator.calculate_single(mol, "MolWt", "test_mol")
        
        # Should succeed after charge computation
        assert result.success is True
        assert call_count[0] == 2  # Checked twice: before and after charge computation
    
    def test_calculate_single_validation_failure(self, mock_registry, mock_validator):
        """Test handling when value validation fails"""
        mock_validator.validate_value.return_value = (False, "Value out of range")
        
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False
        )
        
        mol = Chem.MolFromSmiles("CCO")
        result = calculator.calculate_single(mol, "MolWt", "test_mol")
        
        assert result.success is False
        assert "Invalid value" in result.error_message
    
    def test_calculate_batch_fallback_disabled_raises_exception(self, mock_registry, mock_validator):
        """Test that calculate_batch raises DescriptorCalculationError when fallback_on_error=False"""
        def failing_descriptor(mol):
            raise ValueError("Intentional calculation failure")
        
        descriptor_map = {
            "WorkingDesc": lambda mol: 1.0,
            "FailingDesc": failing_descriptor,
        }
        
        mock_registry.has_descriptor.side_effect = lambda name: name in descriptor_map
        mock_registry.get_descriptor.side_effect = lambda name: descriptor_map.get(name)
        
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=False,  # Critical: exception should be raised
            generate_conformers=False
        )
        
        mol = Chem.MolFromSmiles("CCO")
        
        with pytest.raises(DescriptorCalculationError) as exc_info:
            calculator.calculate_batch(mol, ["WorkingDesc", "FailingDesc"], "test_mol")
        
        assert "FailingDesc" in str(exc_info.value)
        assert exc_info.value.descriptor_name == "FailingDesc"
    
    def test_calculate_batch_empty_descriptor_list(self, mock_registry, mock_validator):
        """Test calculate_batch with empty descriptor list"""
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False
        )
        
        mol = Chem.MolFromSmiles("CCO")
        result = calculator.calculate_batch(mol, [], "test_mol")
        
        assert len(result.successful) == 0
        assert len(result.failed) == 0
        assert result.molecules_processed == 1
        assert result.total_time >= 0
    
    def test_calculate_for_molecules_empty_list(self, mock_registry, mock_validator):
        """Test calculate_for_molecules with empty molecule list"""
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False
        )
        
        results = calculator.calculate_for_molecules([], ["MolWt"])
        
        assert len(results) == 0
    
    def test_get_statistics_with_zero_calculations(self, mock_registry, mock_validator):
        """Test statistics calculation when no calculations performed"""
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False
        )
        
        stats = calculator.get_statistics()
        
        assert stats['total_calculations'] == 0
        assert stats['successful'] == 0
        assert stats['failed'] == 0
        assert stats['cache_hits'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['cache_hit_rate'] == 0.0
        assert stats['cache_size'] == 0
    
    def test_cache_disabled_no_cache_hits(self, mock_registry, mock_validator):
        """Test that cache hits are zero when caching is disabled"""
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=False,  # Critical: cache disabled
            fallback_on_error=True,
            generate_conformers=False
        )
        
        mol = Chem.MolFromSmiles("CCO")
        
        # Calculate same descriptor twice
        calculator.calculate_single(mol, "MolWt", "test_mol")
        calculator.calculate_single(mol, "MolWt", "test_mol")
        
        stats = calculator.get_statistics()
        assert stats['cache_hits'] == 0
        assert stats['total_calculations'] == 2


class TestPydanticModelCompatibility:
    """
    Test Pydantic V2 model compatibility for result classes.
    
    Verifies:
    - to_dict() method returns correct dictionary representation
    - model_dump() compatibility (Pydantic V2)
    - Field types and default values
    """
    
    def test_calculation_result_to_dict(self):
        """Test CalculationResult.to_dict() returns all fields"""
        result = CalculationResult(
            success=True,
            value=123.45,
            descriptor_name="MolWt",
            error_message=None,
            computation_time=0.001
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert result_dict['value'] == 123.45
        assert result_dict['descriptor_name'] == "MolWt"
        assert result_dict['error_message'] is None
        assert result_dict['computation_time'] == 0.001
        # Verify all 5 fields are present
        assert set(result_dict.keys()) == {'success', 'value', 'descriptor_name', 'error_message', 'computation_time'}
    
    def test_calculation_result_to_dict_failure_case(self):
        """Test CalculationResult.to_dict() for failure case"""
        result = CalculationResult(
            success=False,
            value=None,
            descriptor_name="InvalidDesc",
            error_message="Descriptor not found",
            computation_time=0.0005
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['success'] is False
        assert result_dict['value'] is None
        assert result_dict['error_message'] == "Descriptor not found"
    
    def test_batch_calculation_result_to_dict(self):
        """Test BatchCalculationResult.to_dict() returns all fields"""
        result = BatchCalculationResult(
            successful={"MolWt": 180.0, "TPSA": 20.0},
            failed={"BadDesc": "Not found"},
            total_time=0.05,
            molecules_processed=1
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['successful'] == {"MolWt": 180.0, "TPSA": 20.0}
        assert result_dict['failed'] == {"BadDesc": "Not found"}
        assert result_dict['total_time'] == 0.05
        assert result_dict['molecules_processed'] == 1
        # Verify all 4 fields are present
        assert set(result_dict.keys()) == {'successful', 'failed', 'total_time', 'molecules_processed'}
    
    def test_batch_calculation_result_empty_dicts(self):
        """Test BatchCalculationResult with empty dictionaries"""
        result = BatchCalculationResult(
            successful={},
            failed={},
            total_time=0.0,
            molecules_processed=0
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['successful'] == {}
        assert result_dict['failed'] == {}
    
    def test_calculation_result_model_dump_equivalence(self):
        """Test that to_dict() is equivalent to model_dump()"""
        result = CalculationResult(
            success=True,
            value=99.9,
            descriptor_name="TestDesc",
            error_message=None,
            computation_time=0.002
        )
        
        # Both should produce identical output
        assert result.to_dict() == result.model_dump()
    
    def test_batch_calculation_result_model_dump_equivalence(self):
        """Test that to_dict() is equivalent to model_dump() for BatchCalculationResult"""
        result = BatchCalculationResult(
            successful={"A": 1.0, "B": 2.0},
            failed={"C": "error"},
            total_time=1.5,
            molecules_processed=10
        )
        
        # Both should produce identical output
        assert result.to_dict() == result.model_dump()


class TestCacheKeyDeterminism:
    """
    Test cache key generation for determinism and correctness.
    
    The cache key should be deterministic based on molecule SMILES and descriptor name.
    """
    
    @pytest.fixture
    def calculator(self):
        """Create calculator with mock dependencies for cache testing"""
        mock_registry = Mock()
        mock_validator = Mock()
        mock_validator.check_requirements.return_value = (True, [])
        mock_validator.validate_value.return_value = (True, None)
        mock_registry.has_descriptor.return_value = True
        mock_registry.get_descriptor.return_value = lambda mol: 1.0
        
        return DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=False
        )
    
    def test_cache_key_same_molecule_same_descriptor(self, calculator):
        """Test that same molecule + descriptor produces same cache key"""
        mol = Chem.MolFromSmiles("CCO")
        
        key1 = calculator._get_cache_key(mol, "MolWt")
        key2 = calculator._get_cache_key(mol, "MolWt")
        
        assert key1 == key2
    
    def test_cache_key_same_molecule_different_descriptor(self, calculator):
        """Test that same molecule with different descriptor produces different key"""
        mol = Chem.MolFromSmiles("CCO")
        
        key1 = calculator._get_cache_key(mol, "MolWt")
        key2 = calculator._get_cache_key(mol, "TPSA")
        
        assert key1 != key2
    
    def test_cache_key_different_molecule_same_descriptor(self, calculator):
        """Test that different molecules with same descriptor produces different key"""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("c1ccccc1")
        
        key1 = calculator._get_cache_key(mol1, "MolWt")
        key2 = calculator._get_cache_key(mol2, "MolWt")
        
        assert key1 != key2
    
    def test_cache_key_equivalent_smiles(self, calculator):
        """Test that equivalent SMILES (canonicalized) produce same cache key"""
        # These should canonicalize to the same SMILES
        mol1 = Chem.MolFromSmiles("C(C)O")  # Ethanol written differently
        mol2 = Chem.MolFromSmiles("CCO")    # Ethanol
        
        key1 = calculator._get_cache_key(mol1, "MolWt")
        key2 = calculator._get_cache_key(mol2, "MolWt")
        
        # RDKit MolToSmiles canonicalizes, so these should be the same
        assert key1 == key2


class TestConformerAndChargeGeneration:
    """
    Test 3D conformer generation and charge computation functionality.
    
    These tests verify the internal methods work correctly when requirements
    are not initially met.
    """
    
    @pytest.fixture
    def calculator_with_conformers(self):
        """Create calculator with conformer generation enabled"""
        mock_registry = Mock()
        mock_validator = Mock()
        
        def mock_descriptor(mol):
            return 1.0
        
        mock_registry.has_descriptor.return_value = True
        mock_registry.get_descriptor.return_value = mock_descriptor
        mock_validator.check_requirements.return_value = (True, [])
        mock_validator.validate_value.return_value = (True, None)
        
        return DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=True  # Enabled
        )
    
    def test_ensure_3d_conformer_adds_conformer(self, calculator_with_conformers):
        """Test that _ensure_3d_conformer generates a conformer for 2D molecule"""
        mol = Chem.MolFromSmiles("CCO")  # 2D molecule, no conformer
        
        assert mol.GetNumConformers() == 0
        
        mol_with_conf = calculator_with_conformers._ensure_3d_conformer(mol, "test_mol")
        
        # Should have conformer now
        assert mol_with_conf.GetNumConformers() > 0
        
        # Statistics should reflect conformer generation
        stats = calculator_with_conformers.get_statistics()
        assert stats['conformers_generated'] >= 1
    
    def test_ensure_3d_conformer_preserves_existing(self, calculator_with_conformers):
        """Test that _ensure_3d_conformer preserves molecule with existing conformer"""
        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)  # Add conformer
        
        initial_conf_count = mol.GetNumConformers()
        assert initial_conf_count > 0
        
        mol_result = calculator_with_conformers._ensure_3d_conformer(mol, "test_mol")
        
        # Should return same molecule (or equivalent) since it already has conformer
        assert mol_result.GetNumConformers() == initial_conf_count
    
    def test_compute_charges_adds_gasteiger_charges(self, calculator_with_conformers):
        """Test that _compute_charges computes Gasteiger charges"""
        mol = Chem.MolFromSmiles("CCO")
        
        # Verify no charges initially
        has_charges_before = any(
            atom.HasProp('_GasteigerCharge') 
            for atom in mol.GetAtoms()
        )
        assert has_charges_before is False
        
        mol_with_charges = calculator_with_conformers._compute_charges(mol, "test_mol")
        
        # Verify charges were added
        has_charges_after = any(
            atom.HasProp('_GasteigerCharge') 
            for atom in mol_with_charges.GetAtoms()
        )
        assert has_charges_after is True
    
    def test_3d_requirement_triggers_conformer_generation(self):
        """Test that 3D requirement triggers conformer generation when enabled"""
        mock_registry = Mock()
        mock_validator = Mock()
        
        def mock_3d_descriptor(mol):
            # This would be a 3D descriptor
            return 5.0
        
        mock_registry.has_descriptor.return_value = True
        mock_registry.get_descriptor.return_value = mock_3d_descriptor
        
        # First call: missing 3D, second call: OK after conformer generation
        call_count = [0]
        def check_requirements_effect(mol, desc):
            call_count[0] += 1
            if call_count[0] == 1:
                return (False, ["3D coordinates"])
            return (True, [])
        
        mock_validator.check_requirements.side_effect = check_requirements_effect
        mock_validator.validate_value.return_value = (True, None)
        
        calculator = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=True,
            fallback_on_error=True,
            generate_conformers=True  # Critical: enabled
        )
        
        mol = Chem.MolFromSmiles("CCO")
        result = calculator.calculate_single(mol, "RadiusOfGyration", "test_mol")
        
        # Should succeed after conformer generation
        assert result.success is True
        assert call_count[0] == 2
        
        stats = calculator.get_statistics()
        assert stats['conformers_generated'] >= 1


# No teardown_module needed since we're not polluting sys.modules
# All mocking is done at test-level via fixtures
