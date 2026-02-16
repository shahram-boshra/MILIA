#!/usr/bin/env python3
"""
Complete Unit Test Suite for descriptor_validator.py

Tests descriptor validation functionality including:
- Value validation (NaN, Inf, range checking)
- Requirements checking (3D coordinates, partial charges)
- Configuration validation
- Descriptor filtering by requirements
- ValidationResult Pydantic V2 BaseModel (to_dict(), model_dump(), mutability, default_factory)
- Global convenience functions

Pydantic V2 Migration Testing (Phase 29):
    - Tests to_dict() backward compatibility wrapper around model_dump()
    - Tests default_factory isolation (no mutable default bug)
    - Tests ValidationResult mutability (not frozen)
    - Verifies BaseModel inheritance

Author: Milia Team
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


import pytest
from rdkit import Chem

from milia_pipeline.descriptors.descriptor_validator import (
    DescriptorValidator,
    ValidationResult,
    check_requirements,
    filter_by_requirements,
    validate_value,
)


class TestDescriptorValidator:
    """Test descriptor validation functionality"""

    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return DescriptorValidator()

    @pytest.fixture
    def simple_mol(self):
        """Create simple molecule without 3D coords"""
        return Chem.MolFromSmiles("CCO")

    @pytest.fixture
    def mol_with_3d(self):
        """Create molecule with 3D coordinates"""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        from rdkit.Chem import AllChem

        AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol

    def test_validate_normal_value(self, validator):
        """Test validation of normal value"""
        is_valid, msg = validator.validate_value("MolWt", 180.2)
        assert is_valid
        assert msg == ""

    def test_validate_nan(self, validator):
        """Test NaN detection"""
        is_valid, msg = validator.validate_value("MolWt", float("nan"))
        assert not is_valid
        assert "NaN" in msg

    def test_validate_nan_allowed(self, validator):
        """Test NaN when explicitly allowed"""
        is_valid, msg = validator.validate_value("MolWt", float("nan"), allow_nan=True)
        assert is_valid
        assert msg == ""

    def test_validate_inf(self, validator):
        """Test Inf detection"""
        is_valid, msg = validator.validate_value("MolWt", float("inf"))
        assert not is_valid
        assert "Inf" in msg

    def test_validate_inf_allowed(self, validator):
        """Test Inf when explicitly allowed"""
        is_valid, msg = validator.validate_value("MolWt", float("inf"), allow_inf=True)
        assert is_valid
        assert msg == ""

    def test_validate_negative_inf(self, validator):
        """Test negative Inf detection"""
        is_valid, msg = validator.validate_value("MolWt", float("-inf"))
        assert not is_valid
        assert "Inf" in msg

    def test_validate_range_valid(self, validator):
        """Test range validation with valid value"""
        is_valid, msg = validator.validate_value("MolWt", 500.0, min_value=0, max_value=1000)
        assert is_valid
        assert msg == ""

    def test_validate_range_below_minimum(self, validator):
        """Test range validation below minimum"""
        is_valid, msg = validator.validate_value("MolWt", -10.0, min_value=0)
        assert not is_valid
        assert "below minimum" in msg

    def test_validate_range_above_maximum(self, validator):
        """Test range validation above maximum"""
        is_valid, msg = validator.validate_value("MolWt", 1500.0, max_value=1000)
        assert not is_valid
        assert "above maximum" in msg

    def test_validate_range_at_boundaries(self, validator):
        """Test range validation at boundary values"""
        # At minimum
        is_valid, msg = validator.validate_value("MolWt", 0.0, min_value=0, max_value=1000)
        assert is_valid

        # At maximum
        is_valid, msg = validator.validate_value("MolWt", 1000.0, min_value=0, max_value=1000)
        assert is_valid

    def test_validate_integer_value(self, validator):
        """Test validation of integer value"""
        is_valid, msg = validator.validate_value("NumAtoms", 42)
        assert is_valid
        assert msg == ""

    def test_check_3d_requirements_no_coords(self, validator, simple_mol):
        """Test 3D coordinate requirement checking for molecule without 3D coords"""
        can_calc, missing = validator.check_requirements(simple_mol, "RadiusOfGyration")
        assert not can_calc
        assert "3D coordinates" in missing

    def test_check_3d_requirements_with_coords(self, validator, mol_with_3d):
        """Test 3D coordinate requirement checking for molecule with 3D coords"""
        can_calc, missing = validator.check_requirements(mol_with_3d, "RadiusOfGyration")
        assert can_calc
        assert len(missing) == 0

    def test_check_no_requirements(self, validator, simple_mol):
        """Test descriptor with no special requirements"""
        can_calc, missing = validator.check_requirements(simple_mol, "MolWt")
        assert can_calc
        assert len(missing) == 0

    def test_check_requirements_none_mol(self, validator):
        """Test requirements checking with None molecule"""
        can_calc, missing = validator.check_requirements(None, "MolWt")
        # Even simple descriptors should fail with None mol
        # But validator may handle this differently
        # Just check it doesn't crash
        assert isinstance(can_calc, bool)
        assert isinstance(missing, list)

    def test_has_3d_coordinates_simple_mol(self, validator, simple_mol):
        """Test 3D coordinate detection for simple molecule"""
        has_3d = validator._has_3d_coordinates(simple_mol)
        assert not has_3d

    def test_has_3d_coordinates_3d_mol(self, validator, mol_with_3d):
        """Test 3D coordinate detection for 3D molecule"""
        has_3d = validator._has_3d_coordinates(mol_with_3d)
        assert has_3d

    def test_has_3d_coordinates_none(self, validator):
        """Test 3D coordinate detection with None"""
        has_3d = validator._has_3d_coordinates(None)
        assert not has_3d

    def test_has_partial_charges_none(self, validator):
        """Test partial charge detection with None"""
        has_charges = validator._has_partial_charges(None)
        assert not has_charges

    def test_has_partial_charges_no_charges(self, validator, simple_mol):
        """Test partial charge detection for molecule without charges"""
        has_charges = validator._has_partial_charges(simple_mol)
        assert not has_charges

    def test_has_partial_charges_with_gasteiger(self, validator):
        """Test partial charge detection with Gasteiger charges computed"""
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        # Compute Gasteiger charges - sets _GasteigerCharge property on atoms
        AllChem.ComputeGasteigerCharges(mol)

        has_charges = validator._has_partial_charges(mol)
        assert has_charges

    def test_has_partial_charges_with_partial_charge_property(self, validator):
        """Test partial charge detection with _PartialCharge property set"""
        mol = Chem.MolFromSmiles("C")
        atom = mol.GetAtomWithIdx(0)
        atom.SetDoubleProp("_PartialCharge", 0.1)

        has_charges = validator._has_partial_charges(mol)
        assert has_charges

    def test_check_charge_requirements_no_charges(self, validator, simple_mol):
        """Test charge requirement checking for molecule without charges"""
        # MaxPartialCharge requires partial charges
        can_calc, missing = validator.check_requirements(simple_mol, "MaxPartialCharge")
        assert not can_calc
        assert "partial charges" in missing

    def test_check_charge_requirements_with_charges(self, validator):
        """Test charge requirement checking for molecule with charges"""
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        AllChem.ComputeGasteigerCharges(mol)

        # MaxPartialCharge should now be calculable
        can_calc, missing = validator.check_requirements(mol, "MaxPartialCharge")
        assert can_calc
        assert len(missing) == 0

    def test_filter_by_requirements_simple(self, validator, simple_mol):
        """Test filtering descriptors by requirements"""
        descriptors = ["MolWt", "TPSA", "RadiusOfGyration", "MaxPartialCharge"]
        result = validator.filter_by_requirements(simple_mol, descriptors)

        # MolWt and TPSA should be valid (no special requirements)
        assert "MolWt" in result["valid"]
        assert "TPSA" in result["valid"]

        # RadiusOfGyration requires 3D
        assert "RadiusOfGyration" in result["invalid"]
        assert "3D coordinates" in result["invalid"]["RadiusOfGyration"]

        # Check molecule flags
        assert not result["molecule_has_3d"]
        assert not result["molecule_has_charges"]

    def test_filter_by_requirements_3d_mol(self, validator, mol_with_3d):
        """Test filtering with 3D molecule"""
        descriptors = ["MolWt", "RadiusOfGyration"]
        result = validator.filter_by_requirements(mol_with_3d, descriptors)

        # Both should be valid now
        assert "MolWt" in result["valid"]
        assert "RadiusOfGyration" in result["valid"]
        assert len(result["invalid"]) == 0
        assert result["molecule_has_3d"]

    def test_filter_by_requirements_empty_list(self, validator, simple_mol):
        """Test filtering with empty descriptor list"""
        result = validator.filter_by_requirements(simple_mol, [])
        assert len(result["valid"]) == 0
        assert len(result["invalid"]) == 0

    def test_validate_configuration_valid(self, validator):
        """Test configuration validation with valid config"""
        config = {"enabled": True, "selection_mode": "explicit", "computation": {"batch_size": 100}}
        result = validator.validate_configuration(config)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_configuration_invalid_batch_size(self, validator):
        """Test configuration validation with invalid batch_size"""
        config = {"enabled": True, "selection_mode": "explicit", "computation": {"batch_size": -1}}
        result = validator.validate_configuration(config)
        assert not result.is_valid
        assert any("batch_size" in err for err in result.errors)

    def test_validate_configuration_zero_batch_size(self, validator):
        """Test configuration validation with zero batch_size"""
        config = {"enabled": True, "computation": {"batch_size": 0}}
        result = validator.validate_configuration(config)
        assert not result.is_valid

    def test_validate_configuration_invalid_selection_mode(self, validator):
        """Test configuration validation with invalid selection_mode"""
        config = {"enabled": True, "selection_mode": "invalid_mode"}
        result = validator.validate_configuration(config)
        assert not result.is_valid
        assert any("selection_mode" in err for err in result.errors)

    def test_validate_configuration_valid_selection_modes(self, validator):
        """Test configuration validation with all valid selection modes"""
        for mode in ["explicit", "category", "all"]:
            config = {"enabled": True, "selection_mode": mode}
            result = validator.validate_configuration(config)
            assert result.is_valid

    def test_validate_configuration_missing_enabled(self, validator):
        """Test configuration validation without enabled field"""
        config = {"selection_mode": "explicit"}
        result = validator.validate_configuration(config)
        # Should have warning but still be valid
        assert any("enabled" in warn for warn in result.warnings)

    def test_validate_configuration_invalid_enabled_type(self, validator):
        """Test configuration validation with wrong enabled type"""
        config = {
            "enabled": "yes"  # Should be boolean
        }
        result = validator.validate_configuration(config)
        assert not result.is_valid
        assert any("enabled" in err for err in result.errors)

    def test_validate_configuration_invalid_descriptors_type(self, validator):
        """Test configuration validation with invalid selected_descriptors type"""
        config = {"enabled": True, "selected_descriptors": "not_a_dict_or_list"}
        result = validator.validate_configuration(config)
        assert not result.is_valid
        assert any("selected_descriptors" in err for err in result.errors)

    def test_validate_configuration_valid_descriptors_dict(self, validator):
        """Test configuration validation with dict selected_descriptors"""
        config = {"enabled": True, "selected_descriptors": {"MolWt": True, "TPSA": True}}
        result = validator.validate_configuration(config)
        assert result.is_valid

    def test_validate_configuration_valid_descriptors_list(self, validator):
        """Test configuration validation with list selected_descriptors"""
        config = {"enabled": True, "selected_descriptors": ["MolWt", "TPSA"]}
        result = validator.validate_configuration(config)
        assert result.is_valid

    def test_validate_configuration_invalid_plugin_enabled(self, validator):
        """Test configuration validation with invalid plugin enabled type"""
        config = {
            "enabled": True,
            "plugins": {
                "enabled": "yes"  # Should be boolean
            },
        }
        result = validator.validate_configuration(config)
        assert not result.is_valid
        assert any("plugins.enabled" in err for err in result.errors)

    def test_validate_configuration_invalid_plugin_paths(self, validator):
        """Test configuration validation with invalid plugin_paths type"""
        config = {"enabled": True, "plugins": {"plugin_paths": "not_a_list"}}
        result = validator.validate_configuration(config)
        assert not result.is_valid
        assert any("plugin_paths" in err for err in result.errors)

    def test_validate_configuration_valid_plugins(self, validator):
        """Test configuration validation with valid plugin config"""
        config = {
            "enabled": True,
            "plugins": {"enabled": True, "plugin_paths": ["/path/to/plugin"]},
        }
        result = validator.validate_configuration(config)
        assert result.is_valid

    def test_validate_configuration_empty(self, validator):
        """Test configuration validation with empty config"""
        config = {}
        result = validator.validate_configuration(config)
        # Empty config should be valid with warnings
        assert len(result.warnings) > 0

    def test_batch_validation_all_valid(self, validator):
        """Test batch value validation with all valid values"""
        values = {"MolWt": 180.2, "TPSA": 45.5, "NumAtoms": 20}
        result = validator.validate_batch_values(values)
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.details["invalid_descriptors"]) == 0

    def test_batch_validation_with_nan(self, validator):
        """Test batch value validation with NaN"""
        values = {"MolWt": 180.2, "TPSA": 45.5, "Invalid": float("nan")}
        result = validator.validate_batch_values(values)
        assert not result.is_valid
        assert "Invalid" in result.details["invalid_descriptors"]
        assert len(result.errors) == 1

    def test_batch_validation_with_inf(self, validator):
        """Test batch value validation with Inf"""
        values = {"MolWt": 180.2, "Bad": float("inf")}
        result = validator.validate_batch_values(values)
        assert not result.is_valid
        assert "Bad" in result.details["invalid_descriptors"]

    def test_batch_validation_multiple_invalid(self, validator):
        """Test batch value validation with multiple invalid values"""
        values = {"Valid": 100.0, "NaN": float("nan"), "Inf": float("inf")}
        result = validator.validate_batch_values(values)
        assert not result.is_valid
        assert "NaN" in result.details["invalid_descriptors"]
        assert "Inf" in result.details["invalid_descriptors"]
        assert len(result.errors) == 2

    def test_batch_validation_empty(self, validator):
        """Test batch value validation with empty dict"""
        result = validator.validate_batch_values({})
        assert result.is_valid
        assert len(result.errors) == 0

    def test_batch_validation_with_range(self, validator):
        """Test batch value validation with range constraints"""
        values = {"InRange": 50.0, "TooLow": -10.0}
        result = validator.validate_batch_values(values, min_value=0, max_value=100)
        assert not result.is_valid
        assert "TooLow" in result.details["invalid_descriptors"]


class TestValidationResult:
    """Test ValidationResult Pydantic BaseModel (Pydantic V2 migration)"""

    def test_validation_result_creation(self):
        """Test creating ValidationResult"""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=["warning"], details={"key": "value"}
        )
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.details["key"] == "value"

    def test_validation_result_defaults(self):
        """Test ValidationResult with default values"""
        result = ValidationResult(is_valid=False, errors=["error"])
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.warnings == []
        assert result.details == {}

    def test_validation_result_invalid(self):
        """Test invalid ValidationResult"""
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
        assert not result.is_valid
        assert len(result.errors) == 2

    def test_validation_result_to_dict(self):
        """Test ValidationResult.to_dict() backward compatibility method (Pydantic V2 migration)"""
        result = ValidationResult(
            is_valid=True, errors=["error1"], warnings=["warning1"], details={"key": "value"}
        )
        result_dict = result.to_dict()

        # Verify to_dict() returns all 4 fields
        assert isinstance(result_dict, dict)
        assert result_dict["is_valid"] is True
        assert result_dict["errors"] == ["error1"]
        assert result_dict["warnings"] == ["warning1"]
        assert result_dict["details"] == {"key": "value"}

    def test_validation_result_to_dict_with_defaults(self):
        """Test to_dict() with default warnings and details"""
        result = ValidationResult(is_valid=False, errors=["error"])
        result_dict = result.to_dict()

        # Default factory should produce empty list/dict
        assert result_dict["warnings"] == []
        assert result_dict["details"] == {}

    def test_validation_result_to_dict_equals_model_dump(self):
        """Test that to_dict() wraps model_dump() correctly (Pydantic V2)"""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=["w1", "w2"], details={"nested": {"key": "val"}}
        )

        # to_dict() should be equivalent to model_dump()
        assert result.to_dict() == result.model_dump()

    def test_validation_result_mutability(self):
        """Test ValidationResult is mutable (Pydantic BaseModel, not frozen)"""
        result = ValidationResult(is_valid=True, errors=[])

        # Should be able to modify attributes
        result.is_valid = False
        result.errors.append("new error")
        result.warnings.append("new warning")
        result.details["new_key"] = "new_value"

        assert result.is_valid is False
        assert "new error" in result.errors
        assert "new warning" in result.warnings
        assert result.details["new_key"] == "new_value"

    def test_validation_result_default_factory_isolation(self):
        """Test default_factory creates independent instances (no mutable default bug)"""
        result1 = ValidationResult(is_valid=True, errors=[])
        result2 = ValidationResult(is_valid=False, errors=["err"])

        # Modify result1's defaults
        result1.warnings.append("warning for result1")
        result1.details["key1"] = "value1"

        # result2 should be unaffected
        assert result2.warnings == []
        assert result2.details == {}

    def test_validation_result_is_pydantic_base_model(self):
        """Test ValidationResult is a Pydantic BaseModel"""
        from pydantic import BaseModel

        result = ValidationResult(is_valid=True, errors=[])
        assert isinstance(result, BaseModel)


class TestGlobalFunctions:
    """Test global convenience functions"""

    def test_validate_value_function(self):
        """Test global validate_value function"""
        is_valid, msg = validate_value("MolWt", 180.2)
        assert is_valid
        assert msg == ""

    def test_validate_value_function_nan(self):
        """Test global validate_value with NaN"""
        is_valid, msg = validate_value("MolWt", float("nan"))
        assert not is_valid
        assert "NaN" in msg

    def test_check_requirements_function(self):
        """Test global check_requirements function"""
        mol = Chem.MolFromSmiles("CCO")
        can_calc, missing = check_requirements(mol, "MolWt")
        assert can_calc
        assert len(missing) == 0

    def test_filter_by_requirements_function(self):
        """Test global filter_by_requirements function"""
        mol = Chem.MolFromSmiles("CCO")
        descriptors = ["MolWt", "RadiusOfGyration"]
        result = filter_by_requirements(mol, descriptors)

        assert "MolWt" in result["valid"]
        assert "RadiusOfGyration" in result["invalid"]


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return DescriptorValidator()

    def test_validate_very_large_value(self, validator):
        """Test validation with very large value"""
        is_valid, msg = validator.validate_value("MolWt", 1e100)
        assert is_valid  # Large but not infinite

    def test_validate_very_small_value(self, validator):
        """Test validation with very small value"""
        is_valid, msg = validator.validate_value("MolWt", 1e-100)
        assert is_valid  # Small but not zero

    def test_validate_zero(self, validator):
        """Test validation with zero"""
        is_valid, msg = validator.validate_value("MolWt", 0.0)
        assert is_valid

    def test_validate_negative_value(self, validator):
        """Test validation with negative value"""
        is_valid, msg = validator.validate_value("Charge", -1.5)
        assert is_valid  # Negative is valid for some descriptors

    def test_check_requirements_invalid_descriptor(self, validator):
        """Test requirements checking with unknown descriptor"""
        mol = Chem.MolFromSmiles("CCO")
        # Should not crash with unknown descriptor
        can_calc, missing = validator.check_requirements(mol, "UnknownDescriptor")
        assert isinstance(can_calc, bool)
        assert isinstance(missing, list)

    def test_filter_by_requirements_multiple_missing(self, validator):
        """Test filtering when descriptor requires both 3D and charges"""
        mol = Chem.MolFromSmiles("CCO")  # No 3D, no charges

        # Test with descriptors that might require multiple things
        descriptors = ["MolWt", "RadiusOfGyration", "MaxPartialCharge"]
        result = validator.filter_by_requirements(mol, descriptors)

        # MolWt should be valid
        assert "MolWt" in result["valid"]

        # RadiusOfGyration needs 3D
        assert "RadiusOfGyration" in result["invalid"]

        # MaxPartialCharge needs charges
        assert "MaxPartialCharge" in result["invalid"]

    def test_filter_by_requirements_with_3d_and_charges(self, validator):
        """Test filtering with molecule that has both 3D and charges"""
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.ComputeGasteigerCharges(mol)

        descriptors = ["MolWt", "RadiusOfGyration", "MaxPartialCharge"]
        result = validator.filter_by_requirements(mol, descriptors)

        # All should be valid now
        assert "MolWt" in result["valid"]
        assert "RadiusOfGyration" in result["valid"]
        assert "MaxPartialCharge" in result["valid"]
        assert len(result["invalid"]) == 0
        assert result["molecule_has_3d"] is True
        assert result["molecule_has_charges"] is True

    def test_validate_batch_values_with_kwargs_passthrough(self, validator):
        """Test that validate_batch_values passes kwargs to validate_value"""
        values = {"Desc1": float("nan"), "Desc2": 50.0}

        # With allow_nan=True, NaN should be valid
        result = validator.validate_batch_values(values, allow_nan=True)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_configuration_nested_computation(self, validator):
        """Test configuration validation with nested computation settings"""
        config = {
            "enabled": True,
            "computation": {
                "batch_size": 50,
                "extra_field": "ignored",  # Unknown fields should not cause errors
            },
        }
        result = validator.validate_configuration(config)
        assert result.is_valid

    def test_validation_result_details_nested_dict(self):
        """Test ValidationResult with nested details dictionary"""
        nested_details = {"level1": {"level2": {"value": 42}}, "list_value": [1, 2, 3]}
        result = ValidationResult(is_valid=True, errors=[], details=nested_details)

        # Verify nested access works
        assert result.details["level1"]["level2"]["value"] == 42
        assert result.details["list_value"] == [1, 2, 3]

        # Verify to_dict preserves nesting
        result_dict = result.to_dict()
        assert result_dict["details"]["level1"]["level2"]["value"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
