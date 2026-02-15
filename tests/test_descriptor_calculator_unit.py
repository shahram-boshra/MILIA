#!/usr/bin/env python3
"""
Complete Unit Test Suite for descriptor_calculator.py

Tests the DescriptorCalculator class, CalculationResult model, and
BatchCalculationResult model with comprehensive mocking of external
dependencies (DescriptorRegistry, DescriptorValidator, RDKit).

All external dependencies are mocked at the test level using @patch
decorators to prevent mock pollution of sys.modules.

Author: Milia Team
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from milia_pipeline.descriptors.descriptor_calculator import (
    BatchCalculationResult,
    CalculationResult,
    DescriptorCalculator,
)
from milia_pipeline.exceptions import (
    DescriptorCalculationError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_registry():
    """Create a mock DescriptorRegistry with standard behavior."""
    registry = MagicMock()
    registry.has_descriptor.return_value = True
    registry.get_descriptor.return_value = lambda mol: 42.0
    return registry


@pytest.fixture
def mock_validator():
    """Create a mock DescriptorValidator with 'all valid' default behavior."""
    validator = MagicMock()
    # Default: requirements met, value valid
    validator.check_requirements.return_value = (True, [])
    validator.validate_value.return_value = (True, "")
    return validator


@pytest.fixture
def mock_mol():
    """Create a mock RDKit Mol object with standard behavior."""
    mol = MagicMock()
    mol.GetNumConformers.return_value = 0
    return mol


@pytest.fixture
def calculator(mock_registry, mock_validator):
    """Create a DescriptorCalculator with mocked dependencies."""
    return DescriptorCalculator(
        registry=mock_registry,
        validator=mock_validator,
        enable_cache=True,
        fallback_on_error=True,
        generate_conformers=True,
    )


@pytest.fixture
def calculator_no_cache(mock_registry, mock_validator):
    """Create a DescriptorCalculator with caching disabled."""
    return DescriptorCalculator(
        registry=mock_registry,
        validator=mock_validator,
        enable_cache=False,
        fallback_on_error=True,
        generate_conformers=True,
    )


@pytest.fixture
def calculator_no_fallback(mock_registry, mock_validator):
    """Create a DescriptorCalculator with fallback_on_error disabled."""
    return DescriptorCalculator(
        registry=mock_registry,
        validator=mock_validator,
        enable_cache=True,
        fallback_on_error=False,
        generate_conformers=True,
    )


@pytest.fixture
def calculator_no_conformers(mock_registry, mock_validator):
    """Create a DescriptorCalculator with conformer generation disabled."""
    return DescriptorCalculator(
        registry=mock_registry,
        validator=mock_validator,
        enable_cache=True,
        fallback_on_error=True,
        generate_conformers=False,
    )


# ===========================================================================
# CalculationResult Pydantic Model Tests
# ===========================================================================


class TestCalculationResult:
    """Test CalculationResult Pydantic BaseModel."""

    def test_successful_result_construction(self):
        """Test constructing a successful CalculationResult with all fields."""
        result = CalculationResult(
            success=True,
            value=3.14,
            descriptor_name="MolWt",
            error_message=None,
            computation_time=0.001,
        )
        assert result.success is True
        assert result.value == 3.14
        assert result.descriptor_name == "MolWt"
        assert result.error_message is None
        assert result.computation_time == 0.001

    def test_failed_result_construction(self):
        """Test constructing a failed CalculationResult."""
        result = CalculationResult(
            success=False,
            value=None,
            descriptor_name="PMI1",
            error_message="Requires 3D coordinates",
        )
        assert result.success is False
        assert result.value is None
        assert result.descriptor_name == "PMI1"
        assert result.error_message == "Requires 3D coordinates"
        assert result.computation_time is None  # default

    def test_default_optional_fields(self):
        """Test that optional fields default to None."""
        result = CalculationResult(
            success=True,
            value=1.0,
            descriptor_name="test",
        )
        assert result.error_message is None
        assert result.computation_time is None

    def test_to_dict_method(self):
        """Test to_dict() backward-compatible wrapper around model_dump()."""
        result = CalculationResult(
            success=True,
            value=5.5,
            descriptor_name="LogP",
            error_message=None,
            computation_time=0.05,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["success"] is True
        assert d["value"] == 5.5
        assert d["descriptor_name"] == "LogP"
        assert d["error_message"] is None
        assert d["computation_time"] == 0.05

    def test_to_dict_contains_all_five_fields(self):
        """Verify to_dict() returns exactly 5 documented fields."""
        result = CalculationResult(
            success=False,
            value=None,
            descriptor_name="test",
        )
        d = result.to_dict()
        expected_keys = {"success", "value", "descriptor_name", "error_message", "computation_time"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_matches_model_dump(self):
        """Verify to_dict() is consistent with model_dump()."""
        result = CalculationResult(
            success=True,
            value=2.0,
            descriptor_name="TPSA",
            computation_time=0.01,
        )
        assert result.to_dict() == result.model_dump()

    def test_model_is_mutable(self):
        """Test that CalculationResult is a mutable Pydantic model."""
        result = CalculationResult(
            success=True,
            value=1.0,
            descriptor_name="test",
        )
        result.value = 2.0
        assert result.value == 2.0

    def test_value_can_be_zero(self):
        """Test that a descriptor value of 0.0 is valid."""
        result = CalculationResult(
            success=True,
            value=0.0,
            descriptor_name="NumAromaticRings",
        )
        assert result.value == 0.0
        assert result.success is True

    def test_value_can_be_negative(self):
        """Test that a negative descriptor value is valid."""
        result = CalculationResult(
            success=True,
            value=-3.5,
            descriptor_name="MaxPartialCharge",
        )
        assert result.value == -3.5


# ===========================================================================
# BatchCalculationResult Pydantic Model Tests
# ===========================================================================


class TestBatchCalculationResult:
    """Test BatchCalculationResult Pydantic BaseModel."""

    def test_all_successful_construction(self):
        """Test construction with all descriptors successful."""
        result = BatchCalculationResult(
            successful={"MolWt": 150.0, "LogP": 2.5},
            failed={},
            total_time=0.1,
            molecules_processed=1,
        )
        assert result.successful == {"MolWt": 150.0, "LogP": 2.5}
        assert result.failed == {}
        assert result.total_time == 0.1
        assert result.molecules_processed == 1

    def test_mixed_success_failure_construction(self):
        """Test construction with both successes and failures."""
        result = BatchCalculationResult(
            successful={"MolWt": 150.0},
            failed={"PMI1": "Requires 3D coordinates"},
            total_time=0.2,
            molecules_processed=1,
        )
        assert "MolWt" in result.successful
        assert "PMI1" in result.failed

    def test_all_failed_construction(self):
        """Test construction with all descriptors failed."""
        result = BatchCalculationResult(
            successful={},
            failed={"A": "err1", "B": "err2"},
            total_time=0.05,
            molecules_processed=1,
        )
        assert len(result.successful) == 0
        assert len(result.failed) == 2

    def test_to_dict_method(self):
        """Test to_dict() backward-compatible wrapper."""
        result = BatchCalculationResult(
            successful={"X": 1.0},
            failed={"Y": "error"},
            total_time=0.3,
            molecules_processed=1,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["successful"] == {"X": 1.0}
        assert d["failed"] == {"Y": "error"}
        assert d["total_time"] == 0.3
        assert d["molecules_processed"] == 1

    def test_to_dict_contains_all_four_fields(self):
        """Verify to_dict() returns exactly 4 documented fields."""
        result = BatchCalculationResult(
            successful={},
            failed={},
            total_time=0.0,
            molecules_processed=0,
        )
        d = result.to_dict()
        expected_keys = {"successful", "failed", "total_time", "molecules_processed"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_matches_model_dump(self):
        """Verify to_dict() is consistent with model_dump()."""
        result = BatchCalculationResult(
            successful={"A": 1.0},
            failed={},
            total_time=0.1,
            molecules_processed=1,
        )
        assert result.to_dict() == result.model_dump()

    def test_model_is_mutable(self):
        """Test that BatchCalculationResult is a mutable Pydantic model."""
        result = BatchCalculationResult(
            successful={},
            failed={},
            total_time=0.0,
            molecules_processed=0,
        )
        result.molecules_processed = 5
        assert result.molecules_processed == 5


# ===========================================================================
# DescriptorCalculator Initialization Tests
# ===========================================================================


class TestDescriptorCalculatorInit:
    """Test DescriptorCalculator initialization and configuration."""

    def test_init_with_explicit_dependencies(self, mock_registry, mock_validator):
        """Test initialization with explicitly provided registry and validator."""
        calc = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
        )
        assert calc.registry is mock_registry
        assert calc.validator is mock_validator

    def test_init_default_flags(self, mock_registry, mock_validator):
        """Test default values for configuration flags."""
        calc = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
        )
        assert calc.enable_cache is True
        assert calc.fallback_on_error is True
        assert calc.generate_conformers is True

    def test_init_custom_flags(self, mock_registry, mock_validator):
        """Test initialization with non-default flags."""
        calc = DescriptorCalculator(
            registry=mock_registry,
            validator=mock_validator,
            enable_cache=False,
            fallback_on_error=False,
            generate_conformers=False,
        )
        assert calc.enable_cache is False
        assert calc.fallback_on_error is False
        assert calc.generate_conformers is False

    def test_init_statistics_zeroed(self, calculator):
        """Test that statistics start at zero."""
        stats = calculator._stats
        assert stats["total_calculations"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["cache_hits"] == 0
        assert stats["conformers_generated"] == 0

    def test_init_cache_empty(self, calculator):
        """Test that cache starts empty."""
        assert len(calculator._cache) == 0

    @patch("milia_pipeline.descriptors.descriptor_calculator.DescriptorRegistry")
    @patch("milia_pipeline.descriptors.descriptor_calculator.DescriptorValidator")
    def test_init_defaults_to_global_registry(self, MockValidator, MockRegistry):
        """Test that None registry falls back to get_instance()."""
        mock_instance = MagicMock()
        MockRegistry.get_instance.return_value = mock_instance
        MockValidator.return_value = MagicMock()

        calc = DescriptorCalculator(registry=None, validator=None)
        MockRegistry.get_instance.assert_called_once()
        assert calc.registry is mock_instance

    @patch("milia_pipeline.descriptors.descriptor_calculator.DescriptorRegistry")
    @patch("milia_pipeline.descriptors.descriptor_calculator.DescriptorValidator")
    def test_init_defaults_to_new_validator(self, MockValidator, MockRegistry):
        """Test that None validator creates a new DescriptorValidator."""
        MockRegistry.get_instance.return_value = MagicMock()
        mock_val = MagicMock()
        MockValidator.return_value = mock_val

        calc = DescriptorCalculator(registry=None, validator=None)
        MockValidator.assert_called_once()
        assert calc.validator is mock_val


# ===========================================================================
# calculate_single Tests
# ===========================================================================


class TestCalculateSingle:
    """Test DescriptorCalculator.calculate_single()."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_successful_calculation(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test a basic successful descriptor calculation."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=42.0)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_single(mock_mol, "MolWt", "mol_001")

        assert result.success is True
        assert result.value == 42.0
        assert result.descriptor_name == "MolWt"
        assert result.error_message is None
        assert result.computation_time is not None
        assert result.computation_time >= 0

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_descriptor_not_in_registry(self, MockChem, calculator, mock_registry, mock_mol):
        """Test failure when descriptor is not registered."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        mock_registry.has_descriptor.return_value = False

        result = calculator.calculate_single(mock_mol, "NonExistent", "mol_001")

        assert result.success is False
        assert result.value is None
        assert "not found in registry" in result.error_message

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_descriptor_function_called_with_mol(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that the descriptor function is called with the molecule."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=10.0)
        mock_registry.get_descriptor.return_value = desc_func

        calculator.calculate_single(mock_mol, "TestDesc", "mol_001")

        desc_func.assert_called_once_with(mock_mol)

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_calculation_exception_returns_failure(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that exceptions during calculation are caught and returned as failure."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(side_effect=ValueError("division by zero"))
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_single(mock_mol, "BadDesc", "mol_001")

        assert result.success is False
        assert result.value is None
        assert "ValueError" in result.error_message
        assert "division by zero" in result.error_message

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_calculation_exception_type_in_error_message(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that the exception type name is included in the error message."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(side_effect=RuntimeError("some error"))
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_single(mock_mol, "ErrDesc", "mol_001")

        assert "RuntimeError" in result.error_message
        assert "some error" in result.error_message

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_statistics_incremented_on_success(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that statistics are correctly updated on success."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        calculator.calculate_single(mock_mol, "Desc", "mol_001")

        assert calculator._stats["total_calculations"] == 1
        assert calculator._stats["successful"] == 1
        assert calculator._stats["failed"] == 0

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_statistics_incremented_on_failure(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that statistics are correctly updated on failure."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(side_effect=Exception("boom"))
        mock_registry.get_descriptor.return_value = desc_func

        calculator.calculate_single(mock_mol, "FailDesc", "mol_001")

        assert calculator._stats["total_calculations"] == 1
        assert calculator._stats["successful"] == 0
        assert calculator._stats["failed"] == 1

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_default_mol_identifier(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that default mol_identifier is 'unknown'."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        # Call without mol_identifier — should not raise
        result = calculator.calculate_single(mock_mol, "Desc")
        assert result.success is True


# ===========================================================================
# Cache Behavior Tests
# ===========================================================================


class TestCacheBehavior:
    """Test caching logic in calculate_single."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_cache_hit_returns_cached_value(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that a second call for the same mol+descriptor returns from cache."""
        MockChem.MolToSmiles.return_value = "CCO"
        desc_func = MagicMock(return_value=46.07)
        mock_registry.get_descriptor.return_value = desc_func

        result1 = calculator.calculate_single(mock_mol, "MolWt", "mol_001")
        result2 = calculator.calculate_single(mock_mol, "MolWt", "mol_001")

        assert result1.success is True
        assert result2.success is True
        assert result2.value == 46.07
        # Function should only be called once (second hit from cache)
        desc_func.assert_called_once()
        assert calculator._stats["cache_hits"] == 1

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_cache_disabled_no_caching(
        self, MockChem, calculator_no_cache, mock_registry, mock_validator, mock_mol
    ):
        """Test that caching is skipped when enable_cache=False."""
        MockChem.MolToSmiles.return_value = "CCO"
        desc_func = MagicMock(return_value=46.07)
        mock_registry.get_descriptor.return_value = desc_func

        calculator_no_cache.calculate_single(mock_mol, "MolWt", "mol_001")
        calculator_no_cache.calculate_single(mock_mol, "MolWt", "mol_001")

        # Function called both times (no caching)
        assert desc_func.call_count == 2
        assert calculator_no_cache._stats["cache_hits"] == 0

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_different_descriptors_not_cached_together(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that different descriptor names produce different cache keys."""
        MockChem.MolToSmiles.return_value = "CCO"
        func_a = MagicMock(return_value=1.0)
        func_b = MagicMock(return_value=2.0)

        # First call: descriptor A
        mock_registry.get_descriptor.return_value = func_a
        calculator.calculate_single(mock_mol, "DescA", "mol_001")

        # Second call: descriptor B (should NOT be a cache hit)
        mock_registry.get_descriptor.return_value = func_b
        calculator.calculate_single(mock_mol, "DescB", "mol_001")

        func_a.assert_called_once()
        func_b.assert_called_once()
        assert calculator._stats["cache_hits"] == 0

    def test_clear_cache(self, calculator):
        """Test that clear_cache() empties the cache dictionary."""
        calculator._cache["some_key"] = 42.0
        assert len(calculator._cache) == 1

        calculator.clear_cache()
        assert len(calculator._cache) == 0


# ===========================================================================
# Cache Key Generation Tests
# ===========================================================================


class TestCacheKeyGeneration:
    """Test _get_cache_key() method."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_cache_key_is_md5_of_smiles_and_name(self, MockChem, calculator, mock_mol):
        """Test that cache key is MD5 of 'SMILES:descriptor_name'."""
        MockChem.MolToSmiles.return_value = "CCO"

        key = calculator._get_cache_key(mock_mol, "MolWt")

        expected = hashlib.md5(b"CCO:MolWt").hexdigest()
        assert key == expected

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_cache_key_deterministic(self, MockChem, calculator, mock_mol):
        """Test that the same inputs always produce the same cache key."""
        MockChem.MolToSmiles.return_value = "c1ccccc1"

        key1 = calculator._get_cache_key(mock_mol, "LogP")
        key2 = calculator._get_cache_key(mock_mol, "LogP")
        assert key1 == key2

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_cache_key_differs_for_different_molecules(self, MockChem, calculator):
        """Test that different SMILES produce different cache keys."""
        mol_a = MagicMock()
        mol_b = MagicMock()

        MockChem.MolToSmiles.side_effect = lambda m: "CCO" if m is mol_a else "CC(=O)O"

        key_a = calculator._get_cache_key(mol_a, "MolWt")
        key_b = calculator._get_cache_key(mol_b, "MolWt")
        assert key_a != key_b


# ===========================================================================
# Requirement Handling Tests
# ===========================================================================


class TestRequirementHandling:
    """Test requirement checking and auto-resolution in calculate_single."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_requirements_met_proceeds_to_calculation(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test normal flow when all requirements are met."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=5.0)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_single(mock_mol, "MolWt", "mol_001")

        assert result.success is True
        mock_validator.check_requirements.assert_called()

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    @patch("milia_pipeline.descriptors.descriptor_calculator.AllChem")
    def test_3d_requirement_triggers_conformer_generation(
        self, MockAllChem, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that missing 3D coordinates trigger conformer generation."""
        # First check: fails with 3D requirement
        # Second check (after conformer generation): passes
        mock_validator.check_requirements.side_effect = [
            (False, ["3D coordinates"]),
            (True, []),
        ]
        desc_func = MagicMock(return_value=1.5)
        mock_registry.get_descriptor.return_value = desc_func

        # Mock conformer generation chain
        mock_mol.GetNumConformers.return_value = 0
        mol_copy = MagicMock()
        MockChem.Mol.return_value = mol_copy
        MockChem.AddHs.return_value = mol_copy

        result = calculator.calculate_single(mock_mol, "PMI1", "mol_001")

        assert result.success is True
        MockAllChem.EmbedMolecule.assert_called_once()
        MockAllChem.MMFFOptimizeMolecule.assert_called_once()

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_3d_requirement_no_conformer_gen_returns_failure(
        self, MockChem, calculator_no_conformers, mock_registry, mock_validator, mock_mol
    ):
        """Test that 3D requirement fails when conformer generation is disabled."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        mock_validator.check_requirements.return_value = (False, ["3D coordinates"])

        result = calculator_no_conformers.calculate_single(mock_mol, "PMI1", "mol_001")

        assert result.success is False
        assert "3D coordinates" in result.error_message

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    @patch("milia_pipeline.descriptors.descriptor_calculator.AllChem")
    def test_charge_requirement_triggers_charge_computation(
        self, MockAllChem, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that missing charges trigger Gasteiger charge computation."""
        # First check: fails with charge requirement
        # Second check (after charge computation): passes
        mock_validator.check_requirements.side_effect = [
            (False, ["partial charges"]),
            (True, []),
        ]
        desc_func = MagicMock(return_value=0.3)
        mock_registry.get_descriptor.return_value = desc_func

        mol_copy = MagicMock()
        MockChem.Mol.return_value = mol_copy

        result = calculator.calculate_single(mock_mol, "MaxPartialCharge", "mol_001")

        assert result.success is True
        MockAllChem.ComputeGasteigerCharges.assert_called_once_with(mol_copy)

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_unresolvable_requirements_return_failure(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that unresolvable missing requirements return failure."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        mock_validator.check_requirements.return_value = (False, ["some_unknown_requirement"])

        result = calculator.calculate_single(mock_mol, "WeirdDesc", "mol_001")

        assert result.success is False
        assert "Requirements not met" in result.error_message


# ===========================================================================
# Value Validation Tests
# ===========================================================================


class TestValueValidation:
    """Test post-calculation value validation."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_valid_value_returns_success(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that valid value passes validation."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=100.0)
        mock_registry.get_descriptor.return_value = desc_func
        mock_validator.validate_value.return_value = (True, "")

        result = calculator.calculate_single(mock_mol, "MolWt", "mol_001")
        assert result.success is True
        assert result.value == 100.0

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_invalid_value_returns_failure(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that invalid value (per validator) returns failure."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=float("nan"))
        mock_registry.get_descriptor.return_value = desc_func
        mock_validator.validate_value.return_value = (False, "NaN is not allowed")

        result = calculator.calculate_single(mock_mol, "MolWt", "mol_001")

        assert result.success is False
        assert result.value is None
        assert "Invalid value" in result.error_message
        assert "NaN" in result.error_message

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_invalid_value_increments_failed_stat(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that validation failure increments failed counter."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=float("inf"))
        mock_registry.get_descriptor.return_value = desc_func
        mock_validator.validate_value.return_value = (False, "Inf value")

        calculator.calculate_single(mock_mol, "Desc", "mol_001")

        assert calculator._stats["failed"] == 1
        assert calculator._stats["successful"] == 0


# ===========================================================================
# 3D Conformer Generation Tests
# ===========================================================================


class TestEnsure3DConformer:
    """Test _ensure_3d_conformer() method."""

    def test_mol_already_has_conformer_returns_same(self, calculator, mock_mol):
        """Test that a molecule with conformers is returned unchanged."""
        mock_mol.GetNumConformers.return_value = 1

        result = calculator._ensure_3d_conformer(mock_mol, "mol_001")

        assert result is mock_mol
        assert calculator._stats["conformers_generated"] == 0

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    @patch("milia_pipeline.descriptors.descriptor_calculator.AllChem")
    def test_conformer_generation_workflow(self, MockAllChem, MockChem, calculator, mock_mol):
        """Test the full conformer generation workflow."""
        mock_mol.GetNumConformers.return_value = 0
        mol_copy = MagicMock()
        MockChem.Mol.return_value = mol_copy
        mol_with_h = MagicMock()
        MockChem.AddHs.return_value = mol_with_h

        result = calculator._ensure_3d_conformer(mock_mol, "mol_001")

        MockChem.Mol.assert_called_once_with(mock_mol)
        MockChem.AddHs.assert_called_once_with(mol_copy)
        MockAllChem.EmbedMolecule.assert_called_once_with(mol_with_h, randomSeed=42)
        MockAllChem.MMFFOptimizeMolecule.assert_called_once_with(mol_with_h)
        assert result is mol_with_h
        assert calculator._stats["conformers_generated"] == 1

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_conformer_generation_failure_returns_original(self, MockChem, calculator, mock_mol):
        """Test that conformer generation failure returns the original molecule."""
        mock_mol.GetNumConformers.return_value = 0
        MockChem.Mol.side_effect = Exception("Cannot copy molecule")

        result = calculator._ensure_3d_conformer(mock_mol, "mol_001")

        assert result is mock_mol
        assert calculator._stats["conformers_generated"] == 0


# ===========================================================================
# Charge Computation Tests
# ===========================================================================


class TestComputeCharges:
    """Test _compute_charges() method."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    @patch("milia_pipeline.descriptors.descriptor_calculator.AllChem")
    def test_charge_computation_workflow(self, MockAllChem, MockChem, calculator, mock_mol):
        """Test the Gasteiger charge computation workflow."""
        mol_copy = MagicMock()
        MockChem.Mol.return_value = mol_copy

        result = calculator._compute_charges(mock_mol, "mol_001")

        MockChem.Mol.assert_called_once_with(mock_mol)
        MockAllChem.ComputeGasteigerCharges.assert_called_once_with(mol_copy)
        assert result is mol_copy

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_charge_computation_failure_returns_original(self, MockChem, calculator, mock_mol):
        """Test that charge computation failure returns the original molecule."""
        MockChem.Mol.side_effect = Exception("Cannot compute charges")

        result = calculator._compute_charges(mock_mol, "mol_001")

        assert result is mock_mol


# ===========================================================================
# calculate_batch Tests
# ===========================================================================


class TestCalculateBatch:
    """Test DescriptorCalculator.calculate_batch()."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_batch_all_successful(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test batch calculation where all descriptors succeed."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        call_count = {"n": 0}
        values = [100.0, 2.5, 50.0]

        def side_effect(mol):
            val = values[call_count["n"]]
            call_count["n"] += 1
            return val

        desc_func = MagicMock(side_effect=side_effect)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_batch(mock_mol, ["MolWt", "LogP", "TPSA"], "mol_001")

        assert isinstance(result, BatchCalculationResult)
        assert len(result.successful) == 3
        assert len(result.failed) == 0
        assert result.molecules_processed == 1
        assert result.total_time >= 0

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_batch_with_failures_fallback_enabled(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test batch with failures when fallback_on_error=True."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        # First descriptor succeeds, second fails (not in registry)
        mock_registry.has_descriptor.side_effect = [True, False, True]
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_batch(mock_mol, ["Good", "Missing", "AlsoGood"], "mol_001")

        assert "Good" in result.successful
        assert "Missing" in result.failed
        assert "AlsoGood" in result.successful

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_batch_failure_raises_without_fallback(
        self, MockChem, calculator_no_fallback, mock_registry, mock_validator, mock_mol
    ):
        """Test that batch raises DescriptorCalculationError when fallback=False."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        mock_registry.has_descriptor.return_value = False

        with pytest.raises(DescriptorCalculationError) as exc_info:
            calculator_no_fallback.calculate_batch(mock_mol, ["NonExistent"], "mol_001")
        assert exc_info.value.descriptor_name == "NonExistent"

    def test_batch_empty_list(self, calculator, mock_mol):
        """Test batch calculation with empty descriptor list."""
        result = calculator.calculate_batch(mock_mol, [], "mol_001")

        assert len(result.successful) == 0
        assert len(result.failed) == 0
        assert result.molecules_processed == 1

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_batch_molecules_processed_always_one(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that molecules_processed is always 1 for single-molecule batch."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_batch(mock_mol, ["A", "B", "C"], "mol_001")
        assert result.molecules_processed == 1


# ===========================================================================
# calculate_for_molecules Tests
# ===========================================================================


class TestCalculateForMolecules:
    """Test DescriptorCalculator.calculate_for_molecules()."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_multiple_molecules(self, MockChem, calculator, mock_registry, mock_validator):
        """Test calculating descriptors for multiple molecules."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        mol_a = MagicMock()
        mol_b = MagicMock()
        mol_c = MagicMock()

        molecules = [(mol_a, "mol_a"), (mol_b, "mol_b"), (mol_c, "mol_c")]
        results = calculator.calculate_for_molecules(molecules, ["MolWt"])

        assert len(results) == 3
        assert all(isinstance(r, BatchCalculationResult) for r in results)

    def test_empty_molecules_list(self, calculator):
        """Test calculate_for_molecules with empty list."""
        results = calculator.calculate_for_molecules([], ["MolWt"])
        assert results == []

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_results_order_matches_input_order(
        self, MockChem, calculator, mock_registry, mock_validator
    ):
        """Test that results are returned in the same order as input molecules."""
        MockChem.MolToSmiles.side_effect = lambda m: f"mock_smiles_{id(m)}"
        call_order = []

        def tracking_func(mol):
            call_order.append(id(mol))
            return 1.0

        desc_func = MagicMock(side_effect=tracking_func)
        mock_registry.get_descriptor.return_value = desc_func

        mol_a = MagicMock()
        mol_b = MagicMock()

        molecules = [(mol_a, "mol_a"), (mol_b, "mol_b")]
        results = calculator.calculate_for_molecules(molecules, ["Desc"])

        assert len(results) == 2
        # Verify processing order
        assert call_order[0] == id(mol_a)
        assert call_order[1] == id(mol_b)


# ===========================================================================
# Statistics Tests
# ===========================================================================


class TestStatistics:
    """Test get_statistics(), reset_statistics()."""

    def test_get_statistics_initial(self, calculator):
        """Test initial statistics structure and values."""
        stats = calculator.get_statistics()

        assert stats["total_calculations"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["cache_hits"] == 0
        assert stats["conformers_generated"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["cache_hit_rate"] == 0.0
        assert stats["cache_size"] == 0

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_get_statistics_after_calculations(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test statistics after some successful calculations."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        calculator.calculate_single(mock_mol, "A", "m1")
        calculator.calculate_single(mock_mol, "B", "m1")

        stats = calculator.get_statistics()
        assert stats["total_calculations"] == 2
        assert stats["successful"] == 2
        assert stats["success_rate"] == 100.0

    def test_get_statistics_success_rate_calculation(self, calculator):
        """Test success_rate calculation with mixed results."""
        calculator._stats["total_calculations"] = 4
        calculator._stats["successful"] = 3
        calculator._stats["failed"] = 1

        stats = calculator.get_statistics()
        assert stats["success_rate"] == 75.0

    def test_get_statistics_cache_hit_rate_calculation(self, calculator):
        """Test cache_hit_rate calculation."""
        calculator._stats["total_calculations"] = 10
        calculator._stats["cache_hits"] = 3

        stats = calculator.get_statistics()
        assert stats["cache_hit_rate"] == 30.0

    def test_get_statistics_cache_size(self, calculator):
        """Test cache_size reflects actual cache contents."""
        calculator._cache["key1"] = 1.0
        calculator._cache["key2"] = 2.0

        stats = calculator.get_statistics()
        assert stats["cache_size"] == 2

    def test_get_statistics_zero_total_no_division_error(self, calculator):
        """Test that zero total_calculations doesn't cause ZeroDivisionError."""
        stats = calculator.get_statistics()
        assert stats["success_rate"] == 0.0
        assert stats["cache_hit_rate"] == 0.0

    def test_reset_statistics(self, calculator):
        """Test that reset_statistics() zeroes all counters."""
        calculator._stats["total_calculations"] = 100
        calculator._stats["successful"] = 80
        calculator._stats["failed"] = 20
        calculator._stats["cache_hits"] = 50
        calculator._stats["conformers_generated"] = 5

        calculator.reset_statistics()

        assert calculator._stats["total_calculations"] == 0
        assert calculator._stats["successful"] == 0
        assert calculator._stats["failed"] == 0
        assert calculator._stats["cache_hits"] == 0
        assert calculator._stats["conformers_generated"] == 0

    def test_reset_statistics_does_not_clear_cache(self, calculator):
        """Test that reset_statistics() does NOT clear the cache."""
        calculator._cache["key"] = 42.0
        calculator.reset_statistics()
        assert "key" in calculator._cache


# ===========================================================================
# Integration-Style Scenario Tests (all mocked)
# ===========================================================================


class TestEndToEndScenarios:
    """Test full workflows combining multiple methods, all dependencies mocked."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_cache_populated_then_hit(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test full flow: calculate → cache → recalculate → cache hit."""
        MockChem.MolToSmiles.return_value = "CCO"
        desc_func = MagicMock(return_value=46.07)
        mock_registry.get_descriptor.return_value = desc_func

        # First call: computes and caches
        r1 = calculator.calculate_single(mock_mol, "MolWt", "ethanol")
        assert r1.success is True

        # Second call: cache hit
        r2 = calculator.calculate_single(mock_mol, "MolWt", "ethanol")
        assert r2.success is True
        assert r2.value == 46.07

        stats = calculator.get_statistics()
        assert stats["total_calculations"] == 2
        assert stats["successful"] == 1  # only first counts as compute-success
        assert stats["cache_hits"] == 1

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_batch_then_statistics_consistency(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that batch calculation stats are consistent."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        calculator.calculate_batch(mock_mol, ["A", "B", "C"], "mol_001")

        stats = calculator.get_statistics()
        assert stats["total_calculations"] == 3
        assert stats["successful"] == 3

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_multi_molecule_batch_flow(self, MockChem, calculator, mock_registry, mock_validator):
        """Test calculate_for_molecules returns correct number of results."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        mols = [(MagicMock(), f"mol_{i}") for i in range(5)]
        results = calculator.calculate_for_molecules(mols, ["MolWt", "LogP"])

        assert len(results) == 5
        for r in results:
            assert isinstance(r, BatchCalculationResult)
            assert r.molecules_processed == 1

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    @patch("milia_pipeline.descriptors.descriptor_calculator.AllChem")
    def test_3d_and_charge_resolution_in_sequence(
        self, MockAllChem, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that 3D then charge resolution works in sequence for different descriptors."""
        mol_copy = MagicMock()
        MockChem.Mol.return_value = mol_copy
        MockChem.AddHs.return_value = mol_copy

        # Descriptor 1: needs 3D
        mock_validator.check_requirements.side_effect = [
            (False, ["3D coordinates"]),  # first check
            (True, []),  # after conformer
        ]
        desc_func_3d = MagicMock(return_value=1.5)
        mock_registry.get_descriptor.return_value = desc_func_3d
        mock_mol.GetNumConformers.return_value = 0

        r1 = calculator.calculate_single(mock_mol, "PMI1", "mol_001")
        assert r1.success is True

        # Descriptor 2: needs charges
        mock_validator.check_requirements.side_effect = [
            (False, ["partial charges"]),  # first check
            (True, []),  # after charges
        ]
        desc_func_charge = MagicMock(return_value=0.5)
        mock_registry.get_descriptor.return_value = desc_func_charge

        r2 = calculator.calculate_single(mock_mol, "MaxPartialCharge", "mol_001")
        assert r2.success is True

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_clear_cache_then_recalculate(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that clearing cache forces recalculation."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=42.0)
        mock_registry.get_descriptor.return_value = desc_func

        # First calculation
        calculator.calculate_single(mock_mol, "Desc", "mol_001")
        assert desc_func.call_count == 1

        # Clear and recalculate
        calculator.clear_cache()
        calculator.calculate_single(mock_mol, "Desc", "mol_001")
        assert desc_func.call_count == 2


# ===========================================================================
# DescriptorCalculationError Exception Tests
# ===========================================================================


class TestDescriptorCalculationErrorUsage:
    """Test that DescriptorCalculationError is raised correctly in batch mode."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_exception_has_descriptor_name_attribute(
        self, MockChem, calculator_no_fallback, mock_registry, mock_validator, mock_mol
    ):
        """Test that raised exception contains descriptor_name."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        mock_registry.has_descriptor.return_value = False

        with pytest.raises(DescriptorCalculationError) as exc_info:
            calculator_no_fallback.calculate_batch(mock_mol, ["BadDesc"], "mol_001")
        assert exc_info.value.descriptor_name == "BadDesc"

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_exception_has_molecule_id_attribute(
        self, MockChem, calculator_no_fallback, mock_registry, mock_validator, mock_mol
    ):
        """Test that raised exception contains molecule_id."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        mock_registry.has_descriptor.return_value = False

        with pytest.raises(DescriptorCalculationError) as exc_info:
            calculator_no_fallback.calculate_batch(mock_mol, ["BadDesc"], "test_mol_xyz")
        assert exc_info.value.extra_info.get("molecule_id") == "test_mol_xyz"

    def test_exception_inherits_from_descriptor_error(self):
        """Test DescriptorCalculationError inheritance chain."""
        from milia_pipeline.exceptions import DescriptorError

        exc = DescriptorCalculationError(
            "test",
            descriptor_name="desc",
            molecule_id="mol_001",
        )
        assert isinstance(exc, DescriptorError)


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_descriptor_returns_zero(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that 0.0 is a valid descriptor value."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=0.0)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_single(mock_mol, "NumRings", "mol_001")
        assert result.success is True
        assert result.value == 0.0

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_descriptor_returns_very_large_value(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test descriptor returning a very large float."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1e15)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_single(mock_mol, "HugeDesc", "mol_001")
        assert result.success is True
        assert result.value == 1e15

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_descriptor_returns_negative_value(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test descriptor returning a negative float."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=-42.5)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_single(mock_mol, "NegDesc", "mol_001")
        assert result.success is True
        assert result.value == -42.5

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_batch_single_descriptor(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test batch with a single descriptor."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_batch(mock_mol, ["OnlyOne"], "mol_001")
        assert len(result.successful) == 1

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_multiple_sequential_failures_tracked(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that multiple sequential failures are all tracked."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(side_effect=Exception("fail"))
        mock_registry.get_descriptor.return_value = desc_func

        for i in range(5):
            calculator.calculate_single(mock_mol, f"Desc_{i}", "mol_001")

        assert calculator._stats["failed"] == 5
        assert calculator._stats["successful"] == 0
        assert calculator._stats["total_calculations"] == 5

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_computation_time_is_non_negative(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that computation_time is always non-negative."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(return_value=1.0)
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_single(mock_mol, "Desc", "mol_001")
        assert result.computation_time >= 0

    @patch("milia_pipeline.descriptors.descriptor_calculator.Chem")
    def test_failed_calculation_also_has_computation_time(
        self, MockChem, calculator, mock_registry, mock_validator, mock_mol
    ):
        """Test that failed calculations (via exception) still record computation_time."""
        MockChem.MolToSmiles.return_value = "mock_smiles"
        desc_func = MagicMock(side_effect=RuntimeError("boom"))
        mock_registry.get_descriptor.return_value = desc_func

        result = calculator.calculate_single(mock_mol, "FailDesc", "mol_001")
        assert result.success is False
        assert result.computation_time is not None
        assert result.computation_time >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
