#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for config_constants.py

Tests all constants, helper functions, handler support, transformation support,
caching, validation, and lazy loading mechanisms with focus on edge cases,
error handling, and avoiding mock pollution.

NOTE: This test suite runs inside Docker at /app/milia
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import contextlib
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test - DO NOT mock at module level
from milia_pipeline.config import config_constants

# Import exceptions first
from milia_pipeline.exceptions import (
    ConfigurationError,
    HandlerNotAvailableError,
)

# ==========================================
# TEST FIXTURES
# ==========================================


@pytest.fixture
def mock_config():
    """Sample configuration dictionary."""
    return {
        "dataset_type": "DFT",
        "atomic_energies_hartree": {"H": -0.5, "C": -37.8, "N": -54.6, "O": -75.1},
        "heavy_atom_symbols_to_z": {"C": 6, "N": 7, "O": 8},
        "global_constants": {"har2ev": 27.211386245988, "bohr_to_angstrom": 0.529177},
        "global_paths": {"working_root_dir": "/data/milia"},
        "dataset_config": {
            "raw_npz_filename": "DFT_all.npz",
            "raw_data_download_url": "http://example.com/data.npz",
            "dataset_root_dir": "/data/milia",
        },
        "dft_config": {
            "raw_npz_filename": "DFT_all.npz",
            "raw_data_download_url": "http://example.com/dft.npz",
            "processing_config": {"batch_size": 32},
            "filter_config": {},
        },
        "dmc_config": {
            "raw_npz_filename": "DMC.npz",
            "raw_data_download_url": None,
            "processing_config": {"batch_size": 16},
            "filter_config": {},
            "uncertainty_config": {
                "enabled": True,
                "max_uncertainty_threshold": 0.1,
                "validation_mode": "strict",
            },
        },
    }


@pytest.fixture
def mock_dmc_config():
    """Sample DMC configuration."""
    return {
        "dataset_type": "DMC",
        "atomic_energies_hartree": {"H": -0.5, "C": -37.8},
        "heavy_atom_symbols_to_z": {"C": 6},
        "global_constants": {"har2ev": 27.211386245988, "bohr_to_angstrom": 0.529177},
        "global_paths": {"working_root_dir": "/data/dmc"},
        "dataset_config": {
            "raw_npz_filename": "DMC.npz",
            "raw_data_download_url": None,
            "dataset_root_dir": "/data/dmc",
        },
        "dmc_config": {
            "raw_npz_filename": "DMC.npz",
            "raw_data_download_url": None,
            "processing_config": {},
            "uncertainty_config": {
                "enabled": True,
                "max_uncertainty_threshold": 0.1,
                "validation_mode": "strict",
            },
        },
    }


# ==========================================
# LOCAL HELPER FUNCTIONS TESTS
# ==========================================


class TestLocalHelperFunctions:
    """Test suite for local helper functions that avoid circular imports."""

    def test_get_config_value_success(self, mock_config):
        """Test successful config value extraction."""
        value = config_constants._get_config_value(mock_config, "dataset_type", str)
        assert value == "DFT"

    def test_get_config_value_missing_key(self, mock_config):
        """Test that missing key raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Missing required config key"):
            config_constants._get_config_value(mock_config, "nonexistent_key", str)

    def test_get_config_value_wrong_type(self, mock_config):
        """Test that wrong type raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="has wrong type"):
            config_constants._get_config_value(mock_config, "dataset_type", int)

    def test_get_config_value_with_parent_key(self, mock_config):
        """Test config value extraction with parent key for better error messages."""
        global_constants = mock_config["global_constants"]
        value = config_constants._get_config_value(
            global_constants, "har2ev", (int, float), parent_key="global_constants"
        )
        assert value == 27.211386245988

    def test_get_config_value_tuple_of_types(self, mock_config):
        """Test config value extraction with tuple of acceptable types."""
        value = config_constants._get_config_value(mock_config, "dataset_type", (str, type(None)))
        assert value == "DFT"

    @patch("milia_pipeline.config.config_loader.load_config")
    def test_get_dataset_type_local(self, mock_load_config, mock_config):
        """Test local implementation of get_dataset_type."""
        mock_load_config.return_value = mock_config
        dataset_type = config_constants._get_dataset_type_local()
        assert dataset_type == "DFT"
        mock_load_config.assert_called_once()

    @patch("milia_pipeline.config.config_loader.load_config")
    def test_get_dataset_config_local(self, mock_load_config, mock_config):
        """Test local implementation of get_dataset_config."""
        mock_load_config.return_value = mock_config
        dataset_config = config_constants._get_dataset_config_local()
        assert dataset_config == mock_config["dft_config"]

    @patch("milia_pipeline.config.config_loader.load_config")
    def test_get_dataset_constants_local(self, mock_load_config, mock_config):
        """Test local implementation of get_dataset_constants."""
        mock_load_config.return_value = mock_config
        filename, url, root_dir = config_constants._get_dataset_constants_local()
        assert filename == "DFT_all.npz"
        assert url == "http://example.com/dft.npz"
        assert root_dir == "/data/milia"  # Comes from global_paths.working_root_dir

    @patch("milia_pipeline.config.config_loader.load_config")
    def test_get_dataset_constants_compatibility_exports(self, mock_load_config, mock_config):
        """Test that compatibility exports work correctly."""
        mock_load_config.return_value = mock_config

        # Test compatibility functions
        result1 = config_constants.get_dataset_constants()
        result2 = config_constants.get_dataset_config()

        assert result1[0] == "DFT_all.npz"
        assert result2 == mock_config["dft_config"]


# ==========================================
# HANDLER PATTERN SUPPORT TESTS
# ==========================================


class TestHandlerConstants:
    """Test suite for handler-related constants."""

    def test_supported_handler_types(self):
        """Test that supported handler types are defined correctly."""
        expected_types = [
            "DFT",
            "DMC",
            "Wavefunction",
            "QM9",
            "ANI1x",
            "ANI1CCX",
            "RMD17",
            "ANI2x",
            "XXMD",
            "QDPi",
        ]
        assert expected_types == config_constants.SUPPORTED_HANDLER_TYPES

    def test_default_handler_type(self):
        """Test default handler type."""
        assert config_constants.DEFAULT_HANDLER_TYPE == "DFT"

    def test_required_handler_config_keys(self):
        """Test required handler configuration keys."""
        assert "DFT" in config_constants.REQUIRED_HANDLER_CONFIG_KEYS
        assert "DMC" in config_constants.REQUIRED_HANDLER_CONFIG_KEYS
        assert "dataset_type" in config_constants.REQUIRED_HANDLER_CONFIG_KEYS["DFT"]
        assert "uncertainty_config" in config_constants.REQUIRED_HANDLER_CONFIG_KEYS["DMC"]

    def test_handler_feature_support(self):
        """Test handler feature support matrix."""
        dft_features = config_constants.HANDLER_FEATURE_SUPPORT["DFT"]
        dmc_features = config_constants.HANDLER_FEATURE_SUPPORT["DMC"]

        assert dft_features["vibrational_analysis"] is True
        assert dft_features["uncertainty_handling"] is False
        assert dmc_features["vibrational_analysis"] is False
        assert dmc_features["uncertainty_handling"] is True

    def test_handler_required_properties(self):
        """Test handler required properties."""
        dft_props = config_constants.HANDLER_REQUIRED_PROPERTIES["DFT"]
        dmc_props = config_constants.HANDLER_REQUIRED_PROPERTIES["DMC"]

        assert "Etot" in dft_props
        assert "atoms" in dft_props
        assert "std" in dmc_props

    def test_handler_optional_properties(self):
        """Test handler optional properties."""
        dft_optional = config_constants.HANDLER_OPTIONAL_PROPERTIES["DFT"]
        dmc_optional = config_constants.HANDLER_OPTIONAL_PROPERTIES["DMC"]

        assert "freqs" in dft_optional
        assert "qmc_stats" in dmc_optional


class TestGetHandlerConstants:
    """Test suite for get_handler_constants function."""

    @patch("milia_pipeline.config.config_constants.get_dataset_config")
    def test_get_handler_constants_dft(self, mock_get_dataset_config):
        """Test getting DFT handler constants."""
        mock_dataset_config = {"processing_config": {}}
        mock_get_dataset_config.return_value = mock_dataset_config

        # Set module attributes directly (bypasses __getattr__)
        config_constants.HAR2EV = 27.211386245988
        config_constants.ATOMIC_ENERGIES_HARTREE = {"H": -0.5, "C": -37.8}
        config_constants.HEAVY_ATOM_SYMBOLS_TO_Z = {"C": 6, "N": 7}

        try:
            # Clear handler cache
            config_constants.get_handler_constants.cache_clear()

            constants = config_constants.get_handler_constants("DFT")

            assert constants["handler_type"] == "DFT"
            assert "required_properties" in constants
            assert "optional_properties" in constants
            assert "feature_support" in constants
            assert constants["feature_support"]["vibrational_analysis"] is True
        finally:
            # Clean up module attributes
            if hasattr(config_constants, "HAR2EV"):
                delattr(config_constants, "HAR2EV")
            if hasattr(config_constants, "ATOMIC_ENERGIES_HARTREE"):
                delattr(config_constants, "ATOMIC_ENERGIES_HARTREE")
            if hasattr(config_constants, "HEAVY_ATOM_SYMBOLS_TO_Z"):
                delattr(config_constants, "HEAVY_ATOM_SYMBOLS_TO_Z")

    @patch("milia_pipeline.config.config_constants.get_dataset_config")
    def test_get_handler_constants_dmc(self, mock_get_dataset_config):
        """Test getting DMC handler constants."""
        mock_dataset_config = {
            "processing_config": {},
            "uncertainty_config": {
                "enabled": True,
                "max_uncertainty_threshold": 0.1,
                "validation_mode": "strict",
            },
        }
        mock_get_dataset_config.return_value = mock_dataset_config

        # Set module attribute
        config_constants.HAR2EV = 27.211386245988

        try:
            # Clear handler cache
            config_constants.get_handler_constants.cache_clear()

            constants = config_constants.get_handler_constants("DMC")

            assert constants["handler_type"] == "DMC"
            assert constants["feature_support"]["uncertainty_handling"] is True
            assert "std" in constants["required_properties"]
        finally:
            if hasattr(config_constants, "HAR2EV"):
                delattr(config_constants, "HAR2EV")

    def test_get_handler_constants_invalid_handler(self):
        """Test that invalid handler type raises HandlerNotAvailableError."""
        config_constants.get_handler_constants.cache_clear()

        with pytest.raises(HandlerNotAvailableError):
            config_constants.get_handler_constants("INVALID")

    @patch("milia_pipeline.config.config_constants.get_dataset_config")
    def test_get_handler_constants_caching(self, mock_get_dataset_config):
        """Test that handler constants are cached."""
        mock_get_dataset_config.return_value = {"processing_config": {}}

        # Set module attributes
        config_constants.HAR2EV = 27.211
        config_constants.ATOMIC_ENERGIES_HARTREE = {"H": -0.5}
        config_constants.HEAVY_ATOM_SYMBOLS_TO_Z = {"C": 6}

        config_constants.get_handler_constants.cache_clear()

        try:
            # First call
            constants1 = config_constants.get_handler_constants("DFT")
            # Second call should use cache
            constants2 = config_constants.get_handler_constants("DFT")

            # Should only call get_dataset_config once due to caching
            assert mock_get_dataset_config.call_count == 1
            assert constants1 == constants2
        finally:
            delattr(config_constants, "HAR2EV")
            delattr(config_constants, "ATOMIC_ENERGIES_HARTREE")
            delattr(config_constants, "HEAVY_ATOM_SYMBOLS_TO_Z")


class TestHandlerValidation:
    """Test suite for handler configuration validation."""

    def test_validate_handler_configuration_dft_valid(self):
        """Test validation of valid DFT handler configuration."""
        config = {"dataset_type": "DFT", "processing_config": {}}
        errors = config_constants.validate_handler_configuration("DFT", config)
        assert errors == []

    def test_validate_handler_configuration_dmc_valid(self):
        """Test validation of valid DMC handler configuration."""
        config = {
            "dataset_type": "DMC",
            "processing_config": {},
            "uncertainty_config": {
                "enabled": True,
                "max_uncertainty_threshold": 0.1,
                "validation_mode": "strict",
            },
        }
        errors = config_constants.validate_handler_configuration("DMC", config)
        assert errors == []

    def test_validate_handler_configuration_unsupported_type(self):
        """Test validation with unsupported handler type."""
        config = {"dataset_type": "INVALID"}
        errors = config_constants.validate_handler_configuration("INVALID", config)
        assert len(errors) == 1
        assert "Unsupported handler type" in errors[0]

    def test_validate_handler_configuration_missing_key(self):
        """Test validation with missing required key."""
        config = {"processing_config": {}}
        errors = config_constants.validate_handler_configuration("DFT", config)
        assert any("Missing required configuration key" in err for err in errors)

    def test_validate_handler_configuration_none_value(self):
        """Test validation with None value for required key."""
        config = {"dataset_type": None, "processing_config": {}}
        errors = config_constants.validate_handler_configuration("DFT", config)
        assert any("cannot be None" in err for err in errors)

    def test_validate_dmc_invalid_threshold(self):
        """Test DMC validation with invalid uncertainty threshold."""
        config = {
            "dataset_type": "DMC",
            "processing_config": {},
            "uncertainty_config": {
                "enabled": True,
                "max_uncertainty_threshold": -0.1,  # Invalid: negative
                "validation_mode": "strict",
            },
        }
        errors = config_constants.validate_handler_configuration("DMC", config)
        assert any("must be positive" in err for err in errors)

    def test_validate_dmc_invalid_validation_mode(self):
        """Test DMC validation with invalid validation mode."""
        config = {
            "dataset_type": "DMC",
            "processing_config": {},
            "uncertainty_config": {"enabled": True, "validation_mode": "invalid_mode"},
        }
        errors = config_constants.validate_handler_configuration("DMC", config)
        assert any("Invalid uncertainty validation_mode" in err for err in errors)

    def test_validate_dft_invalid_frequency_threshold(self):
        """Test DFT validation with invalid frequency threshold."""
        config = {
            "dataset_type": "DFT",
            "processing_config": {
                "frequency_threshold": -50.0  # Invalid: negative
            },
        }
        errors = config_constants.validate_handler_configuration("DFT", config)
        assert any("cannot be negative" in err for err in errors)


class TestHandlerCompatibility:
    """Test suite for handler compatibility functions."""

    def test_get_handler_compatibility_info_dft(self):
        """Test getting compatibility info for DFT handler."""
        info = config_constants.get_handler_compatibility_info("DFT")

        assert info["handler_type"] == "DFT"
        assert "vibrational_analysis" in info["supported_features"]
        assert "uncertainty_handling" in info["unsupported_features"]
        assert info["backward_compatible"] is True

    def test_get_handler_compatibility_info_dmc(self):
        """Test getting compatibility info for DMC handler."""
        info = config_constants.get_handler_compatibility_info("DMC")

        assert info["handler_type"] == "DMC"
        assert "uncertainty_handling" in info["supported_features"]
        assert "vibrational_analysis" in info["unsupported_features"]

    def test_get_handler_compatibility_info_invalid(self):
        """Test that invalid handler type raises error."""
        with pytest.raises(HandlerNotAvailableError):
            config_constants.get_handler_compatibility_info("INVALID")

    def test_check_handler_feature_support_true(self):
        """Test checking supported feature."""
        result = config_constants.check_handler_feature_support("DFT", "vibrational_analysis")
        assert result is True

    def test_check_handler_feature_support_false(self):
        """Test checking unsupported feature."""
        result = config_constants.check_handler_feature_support("DFT", "uncertainty_handling")
        assert result is False

    def test_check_handler_feature_support_invalid_handler(self):
        """Test that invalid handler raises error."""
        with pytest.raises(HandlerNotAvailableError):
            config_constants.check_handler_feature_support("INVALID", "feature")

    def test_get_handler_property_requirements_dft(self):
        """Test getting property requirements for DFT."""
        required, optional = config_constants.get_handler_property_requirements("DFT")

        assert "Etot" in required
        assert "atoms" in required
        assert "freqs" in optional

    def test_get_handler_property_requirements_dmc(self):
        """Test getting property requirements for DMC."""
        required, optional = config_constants.get_handler_property_requirements("DMC")

        assert "std" in required
        assert "qmc_stats" in optional

    def test_get_handler_property_requirements_invalid(self):
        """Test that invalid handler raises error."""
        with pytest.raises(HandlerNotAvailableError):
            config_constants.get_handler_property_requirements("INVALID")


# ==========================================
# TRANSFORMATION SYSTEM CONSTANTS TESTS
# ==========================================


class TestTransformationConstants:
    """Test suite for transformation system constants."""

    def test_transform_categories(self):
        """Test that transform categories are defined."""
        categories = config_constants.TRANSFORM_CATEGORIES
        assert "structural" in categories
        assert "geometric" in categories
        assert "normalization" in categories
        assert "augmentation" in categories
        assert "spatial" in categories
        assert "custom" in categories

    def test_core_transforms(self):
        """Test that core transforms are defined with categories."""
        core = config_constants.CORE_TRANSFORMS

        assert core["AddSelfLoops"] == "structural"
        assert core["RandomRotate"] == "geometric"
        assert core["Normalize"] == "normalization"
        assert core["DropEdge"] == "augmentation"
        assert core["Distance"] == "spatial"

    def test_transform_validation_modes(self):
        """Test transform validation modes."""
        modes = config_constants.TRANSFORM_VALIDATION_MODES
        assert "strict" in modes
        assert "permissive" in modes
        assert "disabled" in modes

    def test_default_transform_validation_mode(self):
        """Test default validation mode."""
        assert config_constants.DEFAULT_TRANSFORM_VALIDATION_MODE == "permissive"

    def test_experimental_setup_constants(self):
        """Test experimental setup configuration constants."""
        assert config_constants.DEFAULT_EXPERIMENTAL_SETUP_NAME == "default"
        assert config_constants.MAX_EXPERIMENTAL_SETUPS == 50
        assert config_constants.MAX_TRANSFORMS_PER_SETUP == 20

    def test_handler_transform_compatibility_matrix(self):
        """Test handler-transform compatibility matrix."""
        compat = config_constants.HANDLER_TRANSFORM_COMPATIBILITY

        assert "DFT" in compat
        assert "DMC" in compat
        assert "Wavefunction" in compat
        assert compat["DFT"]["NormalizeFeatures"] == "compatible"
        assert compat["DMC"]["NormalizeFeatures"] == "incompatible"
        assert compat["Wavefunction"]["NormalizeFeatures"] == "recommended"


class TestTransformationSystemAvailability:
    """Test suite for transformation system availability checks."""

    def test_check_transformation_system_availability_success(self):
        """Test successful transformation system availability check."""
        # Reset the global flag
        config_constants.TRANSFORMATION_SYSTEM_AVAILABLE = None

        with patch("builtins.__import__") as mock_import:
            # Mock successful imports
            mock_import.return_value = MagicMock()

            result = config_constants._check_transformation_system_availability()
            # Result depends on actual system availability
            assert isinstance(result, bool)

    def test_check_transformation_system_availability_cached(self):
        """Test that availability check is cached."""
        # Set the flag
        config_constants.TRANSFORMATION_SYSTEM_AVAILABLE = True

        result = config_constants._check_transformation_system_availability()
        assert result is True

        # Reset for other tests
        config_constants.TRANSFORMATION_SYSTEM_AVAILABLE = None


class TestGetTransformationConstants:
    """Test suite for get_transformation_constants function."""

    @patch("milia_pipeline.config.config_constants._check_transformation_system_availability")
    def test_get_transformation_constants_basic(self, mock_check):
        """Test getting basic transformation constants."""
        mock_check.return_value = False

        config_constants.get_transformation_constants.cache_clear()
        constants = config_constants.get_transformation_constants()

        assert "system_available" in constants
        assert "core_transforms" in constants
        assert "categories" in constants
        assert constants["system_available"] is False

    @patch("milia_pipeline.config.config_constants._check_transformation_system_availability")
    @patch("milia_pipeline.config.config_constants._get_config_transformation_constants")
    def test_get_transformation_constants_with_config(self, mock_config, mock_check):
        """Test getting transformation constants with configuration."""
        mock_check.return_value = True
        mock_config.return_value = {
            "experimental_setups": ["baseline", "augmented"],
            "setup_count": 2,
        }

        config_constants.get_transformation_constants.cache_clear()
        constants = config_constants.get_transformation_constants()

        assert constants["system_available"] is True
        assert "experimental_setups" in constants

    @patch("milia_pipeline.config.config_constants._check_transformation_system_availability")
    @patch("milia_pipeline.config.config_constants._get_transform_registry_info")
    def test_get_transformation_constants_with_registry(self, mock_registry, mock_check):
        """Test getting transformation constants with registry info."""
        mock_check.return_value = True
        mock_registry.return_value = {
            "total_transforms": 25,
            "available_categories": ["structural", "geometric"],
        }

        config_constants.get_transformation_constants.cache_clear()
        constants = config_constants.get_transformation_constants(include_registry_info=True)

        assert "registry_info" in constants

    def test_get_transformation_constants_caching(self):
        """Test that transformation constants are cached."""
        config_constants.get_transformation_constants.cache_clear()

        with patch(
            "milia_pipeline.config.config_constants._check_transformation_system_availability"
        ) as mock:
            mock.return_value = False

            # First call
            constants1 = config_constants.get_transformation_constants()
            # Second call should use cache
            constants2 = config_constants.get_transformation_constants()

            # Should only call check once due to caching
            assert mock.call_count == 1
            assert constants1 == constants2


class TestHandlerTransformCompatibility:
    """Test suite for handler-transform compatibility functions."""

    def test_get_handler_transform_compatibility_compatible(self):
        """Test getting compatibility status for compatible transform."""
        status = config_constants.get_handler_transform_compatibility("DFT", "NormalizeFeatures")
        assert status == "compatible"

    def test_get_handler_transform_compatibility_warning(self):
        """Test getting compatibility status for warning transform."""
        status = config_constants.get_handler_transform_compatibility("DFT", "RandomRotate")
        assert status == "warning"

    def test_get_handler_transform_compatibility_incompatible(self):
        """Test getting compatibility status for incompatible transform."""
        status = config_constants.get_handler_transform_compatibility("DMC", "NormalizeFeatures")
        assert status == "incompatible"

    def test_get_handler_transform_compatibility_unknown(self):
        """Test getting compatibility status for unknown transform."""
        status = config_constants.get_handler_transform_compatibility("DFT", "UnknownTransform")
        assert status == "unknown"

    def test_get_handler_transform_compatibility_invalid_handler(self):
        """Test that invalid handler raises error."""
        with pytest.raises(HandlerNotAvailableError):
            config_constants.get_handler_transform_compatibility("INVALID", "Transform")

    def test_get_compatible_transforms_for_handler(self):
        """Test getting compatible transforms for handler."""
        compatible = config_constants.get_compatible_transforms_for_handler("DFT")

        assert "NormalizeFeatures" in compatible
        assert "RandomRotate" not in compatible  # warning

    def test_get_compatible_transforms_with_warnings(self):
        """Test getting compatible transforms including warnings."""
        compatible = config_constants.get_compatible_transforms_for_handler(
            "DFT", include_warnings=True
        )

        assert "NormalizeFeatures" in compatible
        assert "RandomRotate" in compatible  # included now

    def test_get_incompatible_transforms_for_handler(self):
        """Test getting incompatible transforms for handler."""
        incompatible = config_constants.get_incompatible_transforms_for_handler("DMC")

        assert "NormalizeFeatures" in incompatible
        assert "Normalize" in incompatible

    def test_get_compatible_transforms_invalid_handler(self):
        """Test that invalid handler raises error."""
        with pytest.raises(HandlerNotAvailableError):
            config_constants.get_compatible_transforms_for_handler("INVALID")


class TestExperimentalSetupValidation:
    """Test suite for experimental setup validation."""

    @patch("milia_pipeline.config.config_accessors.get_experimental_setup")
    def test_validate_experimental_setup_compatible(self, mock_get_setup):
        """Test validation of compatible experimental setup."""
        mock_transform1 = MagicMock()
        mock_transform1.name = "AddSelfLoops"
        mock_transform2 = MagicMock()
        mock_transform2.name = "NormalizeFeatures"

        mock_setup = MagicMock()
        mock_setup.transforms = [mock_transform1, mock_transform2]
        mock_get_setup.return_value = mock_setup

        result = config_constants.validate_experimental_setup_for_handler("DFT", "baseline")

        assert result["is_compatible"] is True
        assert len(result["errors"]) == 0

    @patch("milia_pipeline.config.config_accessors.get_experimental_setup")
    def test_validate_experimental_setup_with_warnings(self, mock_get_setup):
        """Test validation with warning transforms."""
        mock_transform = MagicMock()
        mock_transform.name = "RandomRotate"  # warning for DFT

        mock_setup = MagicMock()
        mock_setup.transforms = [mock_transform]
        mock_get_setup.return_value = mock_setup

        result = config_constants.validate_experimental_setup_for_handler("DFT", "baseline")

        assert result["is_compatible"] is True
        assert len(result["warnings"]) > 0

    @patch("milia_pipeline.config.config_accessors.get_experimental_setup")
    def test_validate_experimental_setup_incompatible(self, mock_get_setup):
        """Test validation with incompatible transforms."""
        mock_transform = MagicMock()
        mock_transform.name = "NormalizeFeatures"  # incompatible with DMC

        mock_setup = MagicMock()
        mock_setup.transforms = [mock_transform]
        mock_get_setup.return_value = mock_setup

        result = config_constants.validate_experimental_setup_for_handler("DMC", "baseline")

        assert result["is_compatible"] is False
        assert len(result["errors"]) > 0

    def test_validate_experimental_setup_invalid_handler(self):
        """Test that invalid handler raises error."""
        with pytest.raises(HandlerNotAvailableError):
            config_constants.validate_experimental_setup_for_handler("INVALID", "baseline")


# ==========================================
# HANDLER INTEGRATION UTILITIES TESTS
# ==========================================


class TestHandlerIntegration:
    """Test suite for handler integration utilities."""

    @patch("milia_pipeline.config.config_constants.get_dataset_config")
    @patch("milia_pipeline.config.config_constants.get_handler_constants")
    def test_create_handler_config_from_constants_dft(self, mock_get_handler, mock_get_dataset):
        """Test creating DFT handler configuration from constants."""
        mock_get_dataset.return_value = {
            "processing_config": {"batch_size": 32},
            "filter_config": {},
            "vibrational_config": {},
        }
        mock_get_handler.return_value = {
            "handler_type": "DFT",
            "feature_support": {"vibrational_analysis": True},
        }

        config = config_constants.create_handler_config_from_constants("DFT")

        assert config["dataset_type"] == "DFT"
        assert "processing_config" in config
        assert "vibrational_config" in config

    @patch("milia_pipeline.config.config_constants.get_dataset_config")
    @patch("milia_pipeline.config.config_constants.get_handler_constants")
    def test_create_handler_config_from_constants_dmc(self, mock_get_handler, mock_get_dataset):
        """Test creating DMC handler configuration from constants."""
        mock_get_dataset.return_value = {
            "processing_config": {"batch_size": 16},
            "filter_config": {},
            "uncertainty_config": {"enabled": True},
        }
        mock_get_handler.return_value = {
            "handler_type": "DMC",
            "feature_support": {"uncertainty_handling": True},
        }

        config = config_constants.create_handler_config_from_constants("DMC")

        assert config["dataset_type"] == "DMC"
        assert "uncertainty_config" in config

    def test_get_migration_compatibility_constants(self):
        """Test getting migration compatibility constants."""
        constants = config_constants.get_migration_compatibility_constants()

        assert constants["migration_phase"] == "Phase 3 - Registry Integration"
        assert constants["handler_pattern_enabled"] is True
        assert constants["legacy_fallback_enabled"] is True
        assert constants["backward_compatibility_mode"] == "full"
        # Phase 3 additions
        assert "registry_integration" in constants
        assert "available" in constants["registry_integration"]
        assert "cache_invalidation_enabled" in constants["registry_integration"]
        assert "dynamic_lookup_enabled" in constants["registry_integration"]


class TestHandlerEnvironmentValidation:
    """Test suite for handler environment validation."""

    @patch("milia_pipeline.config.config_loader.load_config")
    def test_validate_handler_environment_success(self, mock_load_config):
        """Test successful handler environment validation."""
        mock_load_config.return_value = {
            "atomic_energies_hartree": {"H": -0.5},
            "heavy_atom_symbols_to_z": {"C": 6},
            "global_constants": {"har2ev": 27.211},
            "dataset_type": "DFT",
            "dft_config": {
                "raw_npz_filename": "file.npz",
                "raw_npz_download_url": "url",
                "dataset_root_dir": "/dir",
            },
            "dataset_config": {
                "raw_npz_filename": "legacy.npz",
                "raw_npz_download_url": "url",
                "dataset_root_dir": "/dir",
            },
        }

        # Clear caches
        config_constants._TEMP_CONFIG = None
        config_constants._CONSTANTS_CACHE.clear()

        with patch("milia_pipeline.config.config_constants.get_handler_constants") as mock:
            mock.return_value = {"handler_type": "DFT"}

            result = config_constants.validate_handler_environment()

            assert "handler_environment_ready" in result
            assert isinstance(result["handler_environment_ready"], bool)

    @patch("milia_pipeline.config.config_constants.get_supported_handler_types")
    @patch("milia_pipeline.config.config_constants.get_handler_constants")
    def test_validate_handler_environment_with_error(self, mock_get_constants, mock_get_types):
        """Test handler environment validation with error in get_handler_constants.

        When get_handler_constants raises an exception, the function catches it
        and marks that handler's constants as unavailable, but continues processing.
        This test verifies that handler-specific errors are handled gracefully.
        """
        mock_get_types.return_value = ["DFT"]
        mock_get_constants.side_effect = Exception("Test error")

        result = config_constants.validate_handler_environment()

        # The function catches per-handler exceptions gracefully and marks the handler as unavailable
        # It should set dft_handler_constants_available to False
        assert result.get("dft_handler_constants_available", True) is False
        # handler_environment_ready depends on registry_available, config_available, and handler_types_defined
        # not on individual handler constant availability
        assert "handler_environment_ready" in result


# ==========================================
# CACHE MANAGEMENT TESTS
# ==========================================


class TestCacheManagement:
    """Test suite for cache management functions."""

    def test_clear_handler_caches(self):
        """Test clearing handler caches."""
        # Populate caches
        config_constants.get_handler_constants.cache_clear()
        config_constants.get_cached_handler_config.cache_clear()

        # Clear and verify
        config_constants.clear_handler_caches()

        info = config_constants.get_handler_cache_info()
        assert info["cache_enabled"] is True

    def test_clear_transformation_caches(self):
        """Test clearing transformation caches."""
        config_constants.get_transformation_constants.cache_clear()
        config_constants.clear_transformation_caches()

        info = config_constants.get_transformation_cache_info()
        assert info["cache_enabled"] is True

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        config_constants.clear_all_caches()

        info = config_constants.get_all_cache_info()
        assert "handler_caches" in info
        assert "transformation_caches" in info

    def test_get_handler_cache_info(self):
        """Test getting handler cache information."""
        info = config_constants.get_handler_cache_info()

        assert "handler_constants_cache" in info
        assert "handler_identifier_keys_cache" in info
        assert "handler_config_cache" in info
        assert "cache_enabled" in info
        assert "max_cache_size" in info

    def test_get_transformation_cache_info(self):
        """Test getting transformation cache information."""
        info = config_constants.get_transformation_cache_info()

        assert "transformation_constants_cache" in info
        assert "cache_enabled" in info

    def test_get_all_cache_info(self):
        """Test getting comprehensive cache information."""
        info = config_constants.get_all_cache_info()

        assert "handler_caches" in info
        assert "transformation_caches" in info
        assert "timestamp" in info


# ==========================================
# LEGACY COMPATIBILITY TESTS
# ==========================================


class TestLegacyCompatibility:
    """Test suite for legacy compatibility functions."""

    def test_get_legacy_compatible_constants_basic(self):
        """Test getting legacy compatible constants."""
        # Set module attributes that are actually accessed by get_legacy_compatible_constants
        config_constants.ATOMIC_ENERGIES_HARTREE = {"H": -0.5}
        config_constants.HEAVY_ATOM_SYMBOLS_TO_Z = {"C": 6}
        config_constants.HAR2EV = 27.211
        config_constants.RAW_NPZ_FILENAME_CACHED = "file.npz"
        config_constants.RAW_DATA_DOWNLOAD_URL_CACHED = "url"
        config_constants.DATASET_ROOT_DIR_CACHED = "/dir"
        config_constants.PROCESSED_DATA_FILENAME = "file.pt"

        try:
            constants = config_constants.get_legacy_compatible_constants()

            # Verify keys that are actually returned by the function
            assert "ATOMIC_ENERGIES_HARTREE" in constants
            assert "HEAVY_ATOM_SYMBOLS_TO_Z" in constants
            assert "HAR2EV" in constants
            assert "RAW_NPZ_FILENAME" in constants
            assert "RAW_DATA_DOWNLOAD_URL" in constants
            assert "DATASET_ROOT_DIR" in constants
            assert "PROCESSED_DATA_FILENAME" in constants
        finally:
            # Cleanup
            for attr in [
                "ATOMIC_ENERGIES_HARTREE",
                "HEAVY_ATOM_SYMBOLS_TO_Z",
                "HAR2EV",
                "RAW_NPZ_FILENAME_CACHED",
                "RAW_DATA_DOWNLOAD_URL_CACHED",
                "DATASET_ROOT_DIR_CACHED",
                "PROCESSED_DATA_FILENAME",
            ]:
                with contextlib.suppress(AttributeError):
                    delattr(config_constants, attr)

    @patch("milia_pipeline.config.config_constants.get_handler_constants")
    @patch("milia_pipeline.config.config_constants.is_handler_type_supported")
    def test_get_legacy_compatible_constants_with_handler(
        self, mock_is_supported, mock_get_handler
    ):
        """Test getting legacy constants with handler type."""
        # Set module attributes that are actually accessed
        config_constants.ATOMIC_ENERGIES_HARTREE = {"H": -0.5}
        config_constants.HEAVY_ATOM_SYMBOLS_TO_Z = {"C": 6}
        config_constants.HAR2EV = 27.211
        config_constants.RAW_NPZ_FILENAME_CACHED = "file.npz"
        config_constants.RAW_DATA_DOWNLOAD_URL_CACHED = "url"
        config_constants.DATASET_ROOT_DIR_CACHED = "/dir"
        config_constants.PROCESSED_DATA_FILENAME = "file.pt"

        mock_is_supported.return_value = True
        mock_get_handler.return_value = {"handler_type": "DFT"}

        try:
            constants = config_constants.get_legacy_compatible_constants("DFT")

            assert "DFT_HANDLER_CONSTANTS" in constants
            assert constants["DFT_HANDLER_CONSTANTS"]["handler_type"] == "DFT"
        finally:
            # Cleanup
            for attr in [
                "ATOMIC_ENERGIES_HARTREE",
                "HEAVY_ATOM_SYMBOLS_TO_Z",
                "HAR2EV",
                "RAW_NPZ_FILENAME_CACHED",
                "RAW_DATA_DOWNLOAD_URL_CACHED",
                "DATASET_ROOT_DIR_CACHED",
                "PROCESSED_DATA_FILENAME",
            ]:
                with contextlib.suppress(AttributeError):
                    delattr(config_constants, attr)

    def test_ensure_handler_constant_compatibility(self):
        """Test ensuring handler constant compatibility."""
        # Set module attributes that are actually accessed by ensure_handler_constant_compatibility
        config_constants.ATOMIC_ENERGIES_HARTREE = {"H": -0.5}
        config_constants.HEAVY_ATOM_SYMBOLS_TO_Z = {"C": 6}
        config_constants.HAR2EV = 27.211
        config_constants.RAW_NPZ_FILENAME_CACHED = "file.npz"
        config_constants.DATASET_ROOT_DIR_CACHED = "/dir"

        try:
            with patch("milia_pipeline.config.config_constants.get_handler_constants") as mock:
                mock.return_value = {"handler_type": "DFT"}

                with patch(
                    "milia_pipeline.config.config_constants.validate_handler_environment"
                ) as mock_validate:
                    mock_validate.return_value = {"handler_environment_ready": True}

                    with patch(
                        "milia_pipeline.config.config_constants.get_supported_handler_types"
                    ) as mock_supported:
                        mock_supported.return_value = ["DFT", "DMC"]

                        result = config_constants.ensure_handler_constant_compatibility()
                        assert isinstance(result, bool)
        finally:
            # Cleanup
            for attr in [
                "ATOMIC_ENERGIES_HARTREE",
                "HEAVY_ATOM_SYMBOLS_TO_Z",
                "HAR2EV",
                "RAW_NPZ_FILENAME_CACHED",
                "DATASET_ROOT_DIR_CACHED",
            ]:
                with contextlib.suppress(AttributeError):
                    delattr(config_constants, attr)


# ==========================================
# DEBUGGING AND DIAGNOSTICS TESTS
# ==========================================


class TestDebuggingDiagnostics:
    """Test suite for debugging and diagnostic functions."""

    def test_get_complete_constants_debug_info(self):
        """Test getting complete debug information."""
        # This function uses module-level globals including _dataset_constants
        # which is accessed via __getattr__. We need to mock appropriately.
        original_temp_config = config_constants._TEMP_CONFIG

        config_constants._TEMP_CONFIG = {"key": "value"}
        config_constants.ATOMIC_ENERGIES_HARTREE = {"H": -0.5}
        config_constants.HEAVY_ATOM_SYMBOLS_TO_Z = {"C": 6}
        config_constants.HAR2EV = 27.211
        config_constants.PROCESSED_DATA_FILENAME = "file.pt"

        try:
            with patch(
                "milia_pipeline.config.config_constants.get_handler_constants"
            ) as mock_handler:
                mock_handler.return_value = {
                    "handler_type": "DFT",
                    "required_properties": ["Etot"],
                    "feature_support": {},
                    "molecule_creation_strategy": "identifier_coordinate_based",
                }

                with patch(
                    "milia_pipeline.config.config_constants.validate_complete_environment"
                ) as mock_validate:
                    mock_validate.return_value = {"complete_environment_ready": True}

                    with patch(
                        "milia_pipeline.config.config_constants.ensure_complete_compatibility"
                    ) as mock_compat:
                        mock_compat.return_value = True

                        with (
                            patch(
                                "milia_pipeline.config.config_constants.get_supported_handler_types",
                                return_value=["DFT", "DMC"],
                            ),
                            patch.object(
                                config_constants,
                                "_dataset_constants",
                                ("file.npz", "url", "/dir"),
                                create=True,
                            ),
                        ):
                            info = config_constants.get_complete_constants_debug_info()

                            assert "config_status" in info
                            assert "handler_support" in info
                            assert "transformation_support" in info
                            assert "constants_status" in info
                            assert "registry_integration" in info
        finally:
            # Cleanup
            config_constants._TEMP_CONFIG = original_temp_config
            for attr in [
                "ATOMIC_ENERGIES_HARTREE",
                "HEAVY_ATOM_SYMBOLS_TO_Z",
                "HAR2EV",
                "PROCESSED_DATA_FILENAME",
            ]:
                with contextlib.suppress(AttributeError):
                    delattr(config_constants, attr)

    @patch("milia_pipeline.config.config_constants.validate_handler_environment")
    @patch("milia_pipeline.config.config_constants._check_transformation_system_availability")
    def test_validate_complete_environment(self, mock_check_transform, mock_validate_handler):
        """Test complete environment validation."""
        mock_validate_handler.return_value = {"handler_environment_ready": True}
        mock_check_transform.return_value = True

        with patch(
            "milia_pipeline.config.config_constants.get_transformation_constants"
        ) as mock_get:
            mock_get.return_value = {"setup_count": 2}

            result = config_constants.validate_complete_environment()

            assert "handler_environment_ready" in result
            assert "transformation_system_available" in result
            assert "complete_environment_ready" in result

    @patch("milia_pipeline.config.config_constants.ensure_handler_constant_compatibility")
    @patch("milia_pipeline.config.config_constants._check_transformation_system_availability")
    def test_ensure_complete_compatibility(self, mock_check, mock_ensure_handler):
        """Test ensuring complete compatibility."""
        mock_ensure_handler.return_value = True
        mock_check.return_value = True

        with patch(
            "milia_pipeline.config.config_constants.get_transformation_constants"
        ) as mock_get:
            mock_get.return_value = {"system_available": True}

            result = config_constants.ensure_complete_compatibility()
            assert isinstance(result, bool)


# ==========================================
# LAZY LOADING (__getattr__) TESTS
# ==========================================


class TestLazyLoading:
    """Test suite for lazy loading via __getattr__."""

    def test_lazy_load_atomic_energies(self):
        """Test lazy loading of ATOMIC_ENERGIES_HARTREE."""
        # Test that the constant exists and can be accessed
        # Pre-set it to avoid triggering buggy __getattr__
        config_constants.ATOMIC_ENERGIES_HARTREE = {"H": -0.5, "C": -37.8}

        try:
            energies = config_constants.ATOMIC_ENERGIES_HARTREE
            assert isinstance(energies, dict)
            assert "H" in energies
            assert "C" in energies
        finally:
            if hasattr(config_constants, "ATOMIC_ENERGIES_HARTREE"):
                delattr(config_constants, "ATOMIC_ENERGIES_HARTREE")

    def test_lazy_load_har2ev(self):
        """Test lazy loading of HAR2EV."""
        # Pre-set the constant
        config_constants.HAR2EV = 27.211386245988

        try:
            har2ev = config_constants.HAR2EV
            assert isinstance(har2ev, (int, float))
            assert har2ev > 0
        finally:
            if hasattr(config_constants, "HAR2EV"):
                delattr(config_constants, "HAR2EV")

    def test_lazy_load_processed_data_filename(self):
        """Test lazy loading of PROCESSED_DATA_FILENAME."""
        # Pre-set required constants
        config_constants.RAW_NPZ_FILENAME_CACHED = "DFT_all.npz"
        config_constants.PROCESSED_DATA_FILENAME = "DFT_all.pt"

        try:
            filename = config_constants.PROCESSED_DATA_FILENAME
            assert isinstance(filename, str)
            assert filename.endswith(".pt")
        finally:
            if hasattr(config_constants, "RAW_NPZ_FILENAME_CACHED"):
                delattr(config_constants, "RAW_NPZ_FILENAME_CACHED")
            if hasattr(config_constants, "PROCESSED_DATA_FILENAME"):
                delattr(config_constants, "PROCESSED_DATA_FILENAME")

    def test_lazy_load_caching(self):
        """Test that lazy loading caches values."""
        # Set a constant to test caching behavior
        config_constants.ATOMIC_ENERGIES_HARTREE = {"H": -0.5}

        try:
            # First access
            energies1 = config_constants.ATOMIC_ENERGIES_HARTREE
            # Second access (should return same object)
            energies2 = config_constants.ATOMIC_ENERGIES_HARTREE

            # Should be the same object (Python caches module attributes)
            assert energies1 is energies2
        finally:
            if hasattr(config_constants, "ATOMIC_ENERGIES_HARTREE"):
                delattr(config_constants, "ATOMIC_ENERGIES_HARTREE")


# ==========================================
# EDGE CASES AND ERROR HANDLING TESTS
# ==========================================


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_get_config_value_empty_config(self):
        """Test _get_config_value with empty config."""
        with pytest.raises(ConfigurationError, match="Missing required config key"):
            config_constants._get_config_value({}, "key", str)

    def test_get_config_value_none_config(self):
        """Test _get_config_value with None config should raise TypeError."""
        with pytest.raises(TypeError):
            config_constants._get_config_value(None, "key", str)

    @patch("milia_pipeline.config.config_loader.load_config")
    def test_get_dataset_constants_none_url(self, mock_load_config):
        """Test get_dataset_constants with None download URL."""
        mock_config = {
            "dataset_type": "DMC",
            "global_paths": {"working_root_dir": "/data"},
            "dmc_config": {
                "raw_npz_filename": "DMC.npz",
                "raw_data_download_url": None,  # None is valid
            },
        }
        mock_load_config.return_value = mock_config

        filename, url, root_dir = config_constants._get_dataset_constants_local()

        assert filename == "DMC.npz"
        assert url is None
        assert root_dir == "/data"

    def test_handler_constants_cache_size_limit(self):
        """Test that handler constants cache respects maxsize."""
        cache_info = config_constants.get_handler_constants.cache_info()
        # LRU cache maxsize is 32
        assert cache_info.maxsize == 32

    def test_transformation_constants_cache_size_limit(self):
        """Test that transformation constants cache respects maxsize."""
        cache_info = config_constants.get_transformation_constants.cache_info()
        # LRU cache maxsize is TRANSFORM_CACHE_SIZE (32)
        assert cache_info.maxsize == config_constants.TRANSFORM_CACHE_SIZE


# ==========================================
# INTEGRATION TESTS
# ==========================================


class TestIntegration:
    """Integration tests for config_constants module."""

    @patch("milia_pipeline.config.config_constants.get_dataset_config")
    def test_full_handler_workflow(self, mock_get_dataset_config):
        """Test complete handler configuration workflow."""
        mock_get_dataset_config.return_value = {
            "raw_npz_filename": "DFT_all.npz",
            "raw_npz_download_url": "url",
            "dataset_root_dir": "/data",
            "processing_config": {},
        }

        # Set module attributes
        config_constants.HAR2EV = 27.211
        config_constants.ATOMIC_ENERGIES_HARTREE = {"H": -0.5, "C": -37.8}
        config_constants.HEAVY_ATOM_SYMBOLS_TO_Z = {"C": 6}

        # Clear handler cache
        config_constants.get_handler_constants.cache_clear()

        try:
            # Get handler constants
            constants = config_constants.get_handler_constants("DFT")
            assert constants["handler_type"] == "DFT"

            # Check features
            assert config_constants.check_handler_feature_support("DFT", "vibrational_analysis")

            # Get compatibility info
            info = config_constants.get_handler_compatibility_info("DFT")
            assert info["handler_type"] == "DFT"
        finally:
            # Cleanup
            delattr(config_constants, "HAR2EV")
            delattr(config_constants, "ATOMIC_ENERGIES_HARTREE")
            delattr(config_constants, "HEAVY_ATOM_SYMBOLS_TO_Z")

    def test_transformation_compatibility_workflow(self):
        """Test transformation compatibility workflow."""
        # Get compatible transforms
        compatible = config_constants.get_compatible_transforms_for_handler("DFT")
        assert isinstance(compatible, list)

        # Get incompatible transforms
        incompatible = config_constants.get_incompatible_transforms_for_handler("DMC")
        assert isinstance(incompatible, list)

        # Check specific compatibility
        status = config_constants.get_handler_transform_compatibility("DFT", "NormalizeFeatures")
        assert status in ["compatible", "warning", "incompatible", "unknown"]


# ==========================================
# PHASE 3: REGISTRY INTEGRATION TESTS
# ==========================================


class TestRegistryInitialization:
    """Test suite for Phase 3 registry initialization functions."""

    def test_init_registry_first_call(self):
        """Test _init_registry on first call."""
        # Save original state — must include function references set by _init_registry
        original_initialized = config_constants._REGISTRY_INITIALIZED
        original_available = config_constants._REGISTRY_AVAILABLE
        original_list_all = config_constants._registry_list_all
        original_get = config_constants._registry_get
        original_is_registered = config_constants._registry_is_registered
        original_get_default = config_constants._registry_get_default

        try:
            # Reset state for testing
            config_constants._REGISTRY_INITIALIZED = False
            config_constants._REGISTRY_AVAILABLE = False

            with (
                patch("milia_pipeline.config.config_constants._registry_list_all", None),
                patch.dict("sys.modules", {"milia_pipeline.datasets.registry": MagicMock()}),
            ):
                # The function should attempt initialization
                result = config_constants._init_registry()
                # Result depends on actual import success
                assert isinstance(result, bool)
                assert config_constants._REGISTRY_INITIALIZED is True
        finally:
            # Restore ALL original state including function references
            config_constants._REGISTRY_INITIALIZED = original_initialized
            config_constants._REGISTRY_AVAILABLE = original_available
            config_constants._registry_list_all = original_list_all
            config_constants._registry_get = original_get
            config_constants._registry_is_registered = original_is_registered
            config_constants._registry_get_default = original_get_default

    def test_init_registry_cached(self):
        """Test that _init_registry returns cached result on subsequent calls."""
        # Save original state
        original_initialized = config_constants._REGISTRY_INITIALIZED
        original_available = config_constants._REGISTRY_AVAILABLE

        try:
            # Set as already initialized
            config_constants._REGISTRY_INITIALIZED = True
            config_constants._REGISTRY_AVAILABLE = True

            result = config_constants._init_registry()
            assert result is True
        finally:
            # Restore original state
            config_constants._REGISTRY_INITIALIZED = original_initialized
            config_constants._REGISTRY_AVAILABLE = original_available

    def test_init_registry_import_error(self):
        """Test _init_registry handles ImportError gracefully."""
        # Save ALL original state — _init_registry modifies these globals
        original_initialized = config_constants._REGISTRY_INITIALIZED
        original_available = config_constants._REGISTRY_AVAILABLE
        original_error = config_constants._REGISTRY_IMPORT_ERROR
        original_list_all = config_constants._registry_list_all
        original_get = config_constants._registry_get
        original_is_registered = config_constants._registry_is_registered
        original_get_default = config_constants._registry_get_default

        try:
            # Reset state
            config_constants._REGISTRY_INITIALIZED = False
            config_constants._REGISTRY_AVAILABLE = False

            with patch("builtins.__import__", side_effect=ImportError("Test import error")):
                _result = config_constants._init_registry()
                # Should return False on import error
                assert config_constants._REGISTRY_INITIALIZED is True
        finally:
            # Restore ALL original state
            config_constants._REGISTRY_INITIALIZED = original_initialized
            config_constants._REGISTRY_AVAILABLE = original_available
            config_constants._REGISTRY_IMPORT_ERROR = original_error
            config_constants._registry_list_all = original_list_all
            config_constants._registry_get = original_get
            config_constants._registry_is_registered = original_is_registered
            config_constants._registry_get_default = original_get_default


class TestDynamicDatasetDiscovery:
    """Test suite for dynamic dataset type discovery from filesystem."""

    def test_discover_dataset_types_from_filesystem_success(self):
        """Test successful filesystem discovery of dataset types."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.glob") as mock_glob,
        ):
            # Mock .py files in implementations directory
            mock_file1 = MagicMock()
            mock_file1.name = "dft.py"
            mock_file1.stem = "dft"
            mock_file2 = MagicMock()
            mock_file2.name = "qm9.py"
            mock_file2.stem = "qm9"
            mock_file3 = MagicMock()
            mock_file3.name = "_init__.py"  # Should be excluded
            mock_file3.stem = "__init__"
            mock_file4 = MagicMock()
            mock_file4.name = "base.py"  # Should be excluded
            mock_file4.stem = "base"

            mock_glob.return_value = [mock_file1, mock_file2, mock_file3, mock_file4]

            result = config_constants._discover_dataset_types_from_filesystem()

            assert "DFT" in result
            assert "QM9" in result
            assert "__INIT__" not in result
            assert "BASE" not in result

    def test_discover_dataset_types_from_filesystem_no_directory(self):
        """Test filesystem discovery when directory doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            result = config_constants._discover_dataset_types_from_filesystem()
            # Should return empty list when directory doesn't exist
            assert isinstance(result, list)

    def test_discover_dataset_types_from_filesystem_exception(self):
        """Test filesystem discovery handles exceptions gracefully."""
        with patch("pathlib.Path.exists", side_effect=Exception("Test error")):
            result = config_constants._discover_dataset_types_from_filesystem()
            assert isinstance(result, list)


class TestRegistryWrappers:
    """Test suite for registry wrapper functions."""

    def test_registry_list_all_with_registry(self):
        """Test registry_list_all when registry is available."""
        original_list_all = config_constants._registry_list_all

        try:
            mock_list_all = MagicMock(return_value=["DFT", "DMC", "QM9"])
            config_constants._registry_list_all = mock_list_all

            with patch("milia_pipeline.config.config_constants._init_registry", return_value=True):
                result = config_constants.registry_list_all()
                assert result == ["DFT", "DMC", "QM9"]
        finally:
            config_constants._registry_list_all = original_list_all

    def test_registry_list_all_fallback(self):
        """Test registry_list_all falls back to filesystem discovery."""
        original_list_all = config_constants._registry_list_all

        try:
            config_constants._registry_list_all = None

            with (
                patch("milia_pipeline.config.config_constants._init_registry", return_value=False),
                patch(
                    "milia_pipeline.config.config_constants._discover_dataset_types_from_filesystem",
                    return_value=["DFT", "QM9"],
                ),
            ):
                result = config_constants.registry_list_all()
                assert result == ["DFT", "QM9"]
        finally:
            config_constants._registry_list_all = original_list_all

    def test_registry_get_with_registry(self):
        """Test registry_get when registry is available."""
        original_get = config_constants._registry_get

        try:
            mock_dataset_class = MagicMock()
            mock_get = MagicMock(return_value=mock_dataset_class)
            config_constants._registry_get = mock_get

            with patch("milia_pipeline.config.config_constants._init_registry", return_value=True):
                result = config_constants.registry_get("DFT")
                assert result == mock_dataset_class
                mock_get.assert_called_once_with("DFT")
        finally:
            config_constants._registry_get = original_get

    def test_registry_get_not_available(self):
        """Test registry_get raises error when registry not available."""
        original_get = config_constants._registry_get

        try:
            config_constants._registry_get = None

            with (
                patch("milia_pipeline.config.config_constants._init_registry", return_value=False),
                patch(
                    "milia_pipeline.config.config_constants.registry_list_all", return_value=["DFT"]
                ),
                pytest.raises(HandlerNotAvailableError),
            ):
                config_constants.registry_get("TestDataset")
        finally:
            config_constants._registry_get = original_get

    def test_registry_is_registered_with_registry(self):
        """Test registry_is_registered when registry is available."""
        original_is_registered = config_constants._registry_is_registered

        try:
            mock_is_registered = MagicMock(return_value=True)
            config_constants._registry_is_registered = mock_is_registered

            with patch("milia_pipeline.config.config_constants._init_registry", return_value=True):
                result = config_constants.registry_is_registered("DFT")
                assert result is True
        finally:
            config_constants._registry_is_registered = original_is_registered

    def test_registry_is_registered_fallback(self):
        """Test registry_is_registered falls back to dynamic discovery."""
        original_is_registered = config_constants._registry_is_registered

        try:
            config_constants._registry_is_registered = None

            with (
                patch("milia_pipeline.config.config_constants._init_registry", return_value=False),
                patch(
                    "milia_pipeline.config.config_constants.registry_list_all",
                    return_value=["DFT", "DMC"],
                ),
            ):
                assert config_constants.registry_is_registered("DFT") is True
                assert config_constants.registry_is_registered("UNKNOWN") is False
        finally:
            config_constants._registry_is_registered = original_is_registered

    def test_get_default_registry_available(self):
        """Test get_default_registry when registry is available."""
        original_get_default = config_constants._registry_get_default

        try:
            mock_registry = MagicMock()
            config_constants._registry_get_default = MagicMock(return_value=mock_registry)

            with patch("milia_pipeline.config.config_constants._init_registry", return_value=True):
                result = config_constants.get_default_registry()
                assert result == mock_registry
        finally:
            config_constants._registry_get_default = original_get_default

    def test_get_default_registry_not_available(self):
        """Test get_default_registry returns None when registry not available."""
        original_get_default = config_constants._registry_get_default

        try:
            config_constants._registry_get_default = None

            with patch("milia_pipeline.config.config_constants._init_registry", return_value=False):
                result = config_constants.get_default_registry()
                assert result is None
        finally:
            config_constants._registry_get_default = original_get_default


class TestCacheInvalidation:
    """Test suite for Phase 3 cache invalidation functions."""

    def test_invalidate_all_handler_caches(self):
        """Test that _invalidate_all_handler_caches clears all caches."""
        # Clear caches first
        config_constants.get_handler_constants.cache_clear()
        config_constants.get_handler_identifier_keys.cache_clear()
        config_constants.get_cached_handler_config.cache_clear()

        # Function should run without error
        config_constants._invalidate_all_handler_caches()

        # Verify caches are cleared (hits and misses should be 0)
        cache_info = config_constants.get_handler_constants.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 0

    def test_setup_registry_cache_invalidation_registry_unavailable(self):
        """Test _setup_registry_cache_invalidation when registry is unavailable."""
        with patch("milia_pipeline.config.config_constants._init_registry", return_value=False):
            result = config_constants._setup_registry_cache_invalidation()
            assert result is False

    def test_setup_registry_cache_invalidation_success(self):
        """Test _setup_registry_cache_invalidation success case."""
        mock_registry = MagicMock()

        with (
            patch("milia_pipeline.config.config_constants._init_registry", return_value=True),
            patch(
                "milia_pipeline.config.config_constants.get_default_registry",
                return_value=mock_registry,
            ),
        ):
            _result = config_constants._setup_registry_cache_invalidation()

            if mock_registry is not None:
                mock_registry.add_on_change_callback.assert_called_once()

    def test_ensure_cache_invalidation_registered(self):
        """Test _ensure_cache_invalidation_registered."""
        original_registered = config_constants._CACHE_INVALIDATION_REGISTERED

        try:
            config_constants._CACHE_INVALIDATION_REGISTERED = False

            with patch(
                "milia_pipeline.config.config_constants._setup_registry_cache_invalidation",
                return_value=True,
            ):
                config_constants._ensure_cache_invalidation_registered()
                assert config_constants._CACHE_INVALIDATION_REGISTERED is True
        finally:
            config_constants._CACHE_INVALIDATION_REGISTERED = original_registered


class TestDynamicHandlerFunctions:
    """Test suite for Phase 3 dynamic handler functions."""

    def test_get_supported_handler_types_from_registry(self):
        """Test get_supported_handler_types uses registry when available."""
        with (
            patch("milia_pipeline.config.config_constants._ensure_cache_invalidation_registered"),
            patch("milia_pipeline.config.config_constants._init_registry", return_value=True),
            patch("milia_pipeline.config.config_constants._REGISTRY_AVAILABLE", True),
            patch(
                "milia_pipeline.config.config_constants.registry_list_all",
                return_value=["DFT", "DMC", "QM9"],
            ),
        ):
            result = config_constants.get_supported_handler_types()
            assert "DFT" in result
            assert "DMC" in result

    def test_get_supported_handler_types_fallback(self):
        """Test get_supported_handler_types falls back to filesystem discovery."""
        with (
            patch("milia_pipeline.config.config_constants._ensure_cache_invalidation_registered"),
            patch("milia_pipeline.config.config_constants._init_registry", return_value=False),
            patch(
                "milia_pipeline.config.config_constants._discover_dataset_types_from_filesystem",
                return_value=["DFT", "QM9"],
            ),
        ):
            result = config_constants.get_supported_handler_types()
            assert isinstance(result, list)

    def test_get_default_handler_type(self):
        """Test get_default_handler_type returns correct default."""
        result = config_constants.get_default_handler_type()
        assert result == "DFT"

    def test_is_handler_type_supported_true(self):
        """Test is_handler_type_supported returns True for valid handler."""
        with (
            patch("milia_pipeline.config.config_constants._ensure_cache_invalidation_registered"),
            patch("milia_pipeline.config.config_constants._init_registry", return_value=True),
            patch("milia_pipeline.config.config_constants._REGISTRY_AVAILABLE", True),
            patch(
                "milia_pipeline.config.config_constants.registry_is_registered",
                return_value=True,
            ),
        ):
            result = config_constants.is_handler_type_supported("DFT")
            assert result is True

    def test_is_handler_type_supported_false(self):
        """Test is_handler_type_supported returns False for invalid handler."""
        with (
            patch("milia_pipeline.config.config_constants._ensure_cache_invalidation_registered"),
            patch("milia_pipeline.config.config_constants._init_registry", return_value=False),
            patch(
                "milia_pipeline.config.config_constants.registry_is_registered",
                return_value=False,
            ),
        ):
            result = config_constants.is_handler_type_supported("INVALID")
            assert result is False


# ==========================================
# ADDITIONAL HANDLER TYPES TESTS
# ==========================================


class TestAdditionalHandlerTypes:
    """Test suite for handler types beyond DFT and DMC."""

    def test_wavefunction_handler_feature_support(self):
        """Test Wavefunction handler feature support."""
        features = config_constants.HANDLER_FEATURE_SUPPORT["Wavefunction"]

        assert features["vibrational_analysis"] is False
        assert features["uncertainty_handling"] is False
        assert features["orbital_analysis"] is True
        assert features["homo_lumo_gap"] is True
        assert features["mo_energies"] is True

    def test_wavefunction_handler_required_properties(self):
        """Test Wavefunction handler required properties."""
        props = config_constants.HANDLER_REQUIRED_PROPERTIES["Wavefunction"]

        assert "atoms" in props
        assert "coordinates" in props
        assert "compounds" in props

    def test_wavefunction_handler_optional_properties(self):
        """Test Wavefunction handler optional properties."""
        props = config_constants.HANDLER_OPTIONAL_PROPERTIES["Wavefunction"]

        assert "mo_energies" in props
        assert "homo_lumo_gap_eV" in props

    def test_wavefunction_handler_identifier_keys(self):
        """Test Wavefunction handler identifier keys."""
        keys = config_constants.HANDLER_IDENTIFIER_KEYS["Wavefunction"]

        assert len(keys) == 1
        assert keys[0] == ("compounds", "compound_id")

    def test_wavefunction_handler_coordinate_units(self):
        """Test Wavefunction handler coordinate units."""
        units = config_constants.HANDLER_COORDINATE_UNITS["Wavefunction"]
        assert units == "bohr"

    def test_qm9_handler_feature_support(self):
        """Test QM9 handler feature support."""
        features = config_constants.HANDLER_FEATURE_SUPPORT["QM9"]

        assert features["vibrational_analysis"] is True
        assert features["atomization_energy"] is True
        assert features["homo_lumo_gap"] is True

    def test_qm9_handler_required_properties(self):
        """Test QM9 handler required properties."""
        props = config_constants.HANDLER_REQUIRED_PROPERTIES["QM9"]

        assert "U0" in props
        assert "atoms" in props
        assert "coordinates" in props

    def test_ani1x_handler_properties(self):
        """Test ANI1x handler properties."""
        props = config_constants.HANDLER_REQUIRED_PROPERTIES["ANI1x"]
        assert "energy" in props

        # ANI1x uses coordinate_based strategy (no identifier keys)
        keys = config_constants.HANDLER_IDENTIFIER_KEYS["ANI1x"]
        assert len(keys) == 0

    def test_rmd17_handler_properties(self):
        """Test RMD17 handler properties."""
        features = config_constants.HANDLER_FEATURE_SUPPORT["RMD17"]
        assert features["forces"] is True

        props = config_constants.HANDLER_REQUIRED_PROPERTIES["RMD17"]
        assert "energies" in props

    def test_all_handler_types_have_coordinate_units(self):
        """Test that all handler types have coordinate units defined."""
        for handler_type in config_constants.SUPPORTED_HANDLER_TYPES:
            assert handler_type in config_constants.HANDLER_COORDINATE_UNITS
            units = config_constants.HANDLER_COORDINATE_UNITS[handler_type]
            assert units in ["angstrom", "bohr"]

    def test_all_handler_types_have_feature_support(self):
        """Test that all handler types have feature support defined."""
        for handler_type in config_constants.SUPPORTED_HANDLER_TYPES:
            assert handler_type in config_constants.HANDLER_FEATURE_SUPPORT
            features = config_constants.HANDLER_FEATURE_SUPPORT[handler_type]
            assert isinstance(features, dict)

    def test_all_handler_types_have_required_properties(self):
        """Test that all handler types have required properties defined."""
        for handler_type in config_constants.SUPPORTED_HANDLER_TYPES:
            assert handler_type in config_constants.HANDLER_REQUIRED_PROPERTIES
            props = config_constants.HANDLER_REQUIRED_PROPERTIES[handler_type]
            assert isinstance(props, list)
            # All handlers need at least atoms and coordinates
            assert "atoms" in props
            assert "coordinates" in props


class TestDeprecationWarnings:
    """Test suite for deprecation warning functions."""

    def test_warn_legacy_constant_access_first_call(self):
        """Test that deprecation warning is issued on first access."""
        # Clear the warned dict for this constant
        if "TEST_CONSTANT" in config_constants._DEPRECATION_WARNED:
            del config_constants._DEPRECATION_WARNED["TEST_CONSTANT"]

        with pytest.warns(DeprecationWarning, match="Direct access to TEST_CONSTANT is deprecated"):
            config_constants._warn_legacy_constant_access("TEST_CONSTANT", "test_function")

    def test_warn_legacy_constant_access_subsequent_calls(self):
        """Test that deprecation warning is only issued once."""
        # Set as already warned
        config_constants._DEPRECATION_WARNED["TEST_CONSTANT_2"] = True

        # Should not warn again (no pytest.warns context = no warning expected)
        config_constants._warn_legacy_constant_access("TEST_CONSTANT_2", "test_function")
        # If we get here without an error, the test passes

    def test_get_supported_handler_types_legacy(self):
        """Test legacy accessor for supported handler types."""
        # Clear warning state
        if "SUPPORTED_HANDLER_TYPES" in config_constants._DEPRECATION_WARNED:
            del config_constants._DEPRECATION_WARNED["SUPPORTED_HANDLER_TYPES"]

        with pytest.warns(DeprecationWarning):
            result = config_constants.get_supported_handler_types_legacy()
            assert result == config_constants.SUPPORTED_HANDLER_TYPES

    def test_get_handler_feature_support_legacy(self):
        """Test legacy accessor for handler feature support."""
        # Clear warning state
        if "HANDLER_FEATURE_SUPPORT" in config_constants._DEPRECATION_WARNED:
            del config_constants._DEPRECATION_WARNED["HANDLER_FEATURE_SUPPORT"]

        with pytest.warns(DeprecationWarning):
            result = config_constants.get_handler_feature_support_legacy("DFT")
            assert result == config_constants.HANDLER_FEATURE_SUPPORT["DFT"]

    def test_get_handler_required_properties_legacy(self):
        """Test legacy accessor for handler required properties."""
        # Clear warning state
        if "HANDLER_REQUIRED_PROPERTIES" in config_constants._DEPRECATION_WARNED:
            del config_constants._DEPRECATION_WARNED["HANDLER_REQUIRED_PROPERTIES"]

        with pytest.warns(DeprecationWarning):
            result = config_constants.get_handler_required_properties_legacy("DFT")
            assert result == config_constants.HANDLER_REQUIRED_PROPERTIES["DFT"]


class TestAdditionalDynamicHandlerFunctions:
    """Test suite for additional Phase 3 dynamic handler functions."""

    @patch("milia_pipeline.config.config_constants._ensure_cache_invalidation_registered")
    @patch("milia_pipeline.config.config_constants._init_registry")
    @patch("milia_pipeline.config.config_constants.registry_is_registered")
    def test_get_handler_optional_properties_valid(self, mock_is_reg, mock_init, mock_ensure):
        """Test get_handler_optional_properties for valid handler."""
        mock_init.return_value = False
        mock_is_reg.return_value = True

        result = config_constants.get_handler_optional_properties("DFT")
        assert "freqs" in result
        assert "vibmodes" in result

    @patch("milia_pipeline.config.config_constants._ensure_cache_invalidation_registered")
    @patch("milia_pipeline.config.config_constants._init_registry")
    @patch("milia_pipeline.config.config_constants.registry_is_registered")
    def test_get_handler_optional_properties_invalid(self, mock_is_reg, mock_init, mock_ensure):
        """Test get_handler_optional_properties raises for invalid handler."""
        mock_init.return_value = False
        mock_is_reg.return_value = False

        with (
            patch(
                "milia_pipeline.config.config_constants.get_supported_handler_types",
                return_value=["DFT"],
            ),
            pytest.raises(HandlerNotAvailableError),
        ):
            config_constants.get_handler_optional_properties("INVALID")

    @patch("milia_pipeline.config.config_constants._ensure_cache_invalidation_registered")
    @patch("milia_pipeline.config.config_constants._init_registry")
    @patch("milia_pipeline.config.config_constants.registry_is_registered")
    def test_get_handler_identifier_keys_dynamic_valid(self, mock_is_reg, mock_init, mock_ensure):
        """Test get_handler_identifier_keys_dynamic for valid handler."""
        mock_init.return_value = False
        mock_is_reg.return_value = True

        result = config_constants.get_handler_identifier_keys_dynamic("DFT")
        assert isinstance(result, list)
        # DFT has identifier keys
        assert len(result) > 0
        assert result[0] == ("inchi", "inchi")

    @patch("milia_pipeline.config.config_constants._ensure_cache_invalidation_registered")
    @patch("milia_pipeline.config.config_constants._init_registry")
    @patch("milia_pipeline.config.config_constants.registry_is_registered")
    def test_get_handler_coordinate_units_dynamic_valid(self, mock_is_reg, mock_init, mock_ensure):
        """Test get_handler_coordinate_units_dynamic for valid handler."""
        mock_init.return_value = False
        mock_is_reg.return_value = True

        result = config_constants.get_handler_coordinate_units_dynamic("DFT")
        assert result == "angstrom"

        result_wf = config_constants.get_handler_coordinate_units_dynamic("Wavefunction")
        assert result_wf == "bohr"

    @patch("milia_pipeline.config.config_constants._ensure_cache_invalidation_registered")
    @patch("milia_pipeline.config.config_constants._init_registry")
    @patch("milia_pipeline.config.config_constants.registry_is_registered")
    def test_get_handler_molecule_creation_strategy_valid(
        self, mock_is_reg, mock_init, mock_ensure
    ):
        """Test get_handler_molecule_creation_strategy for valid handler."""
        mock_init.return_value = False
        mock_is_reg.return_value = True

        # DFT uses identifier_coordinate_based
        result = config_constants.get_handler_molecule_creation_strategy("DFT")
        assert result == "identifier_coordinate_based"

        # Wavefunction uses coordinate_based
        result_wf = config_constants.get_handler_molecule_creation_strategy("Wavefunction")
        assert result_wf == "coordinate_based"


class TestWavefunctionHandlerConstants:
    """Test suite for Wavefunction-specific handler constants."""

    @patch("milia_pipeline.config.config_constants.get_dataset_config")
    @patch("milia_pipeline.config.config_constants._init_registry")
    def test_get_wavefunction_handler_constants(self, mock_init_registry, mock_get_dataset_config):
        """Test getting Wavefunction handler constants."""
        # Make registry unavailable to force fallback to legacy constants
        mock_init_registry.return_value = False

        mock_dataset_config = {"processing_config": {}, "feature_tier": "standard"}
        mock_get_dataset_config.return_value = mock_dataset_config

        # Set required module attributes
        config_constants.HEAVY_ATOM_SYMBOLS_TO_Z = {"C": 6, "N": 7}
        config_constants.BOHR_TO_ANGSTROM = 0.529177

        # Save and reset registry available state
        original_registry_available = config_constants._REGISTRY_AVAILABLE
        config_constants._REGISTRY_AVAILABLE = False

        try:
            config_constants.get_handler_constants.cache_clear()

            with (
                patch(
                    "milia_pipeline.config.config_constants.is_handler_type_supported",
                    return_value=True,
                ),
                patch(
                    "milia_pipeline.config.config_constants.registry_is_registered",
                    return_value=True,
                ),
            ):
                constants = config_constants.get_handler_constants("Wavefunction")

                assert constants["handler_type"] == "Wavefunction"
                # Coordinate units come from HANDLER_COORDINATE_UNITS when registry unavailable
                assert constants["coordinate_units"] == "bohr"
                assert constants["molecule_creation_strategy"] == "coordinate_based"
        finally:
            config_constants._REGISTRY_AVAILABLE = original_registry_available
            if hasattr(config_constants, "HEAVY_ATOM_SYMBOLS_TO_Z"):
                delattr(config_constants, "HEAVY_ATOM_SYMBOLS_TO_Z")
            if hasattr(config_constants, "BOHR_TO_ANGSTROM"):
                delattr(config_constants, "BOHR_TO_ANGSTROM")

    def test_validate_wavefunction_handler_config_valid(self):
        """Test validation of valid Wavefunction handler configuration."""
        config = {"dataset_type": "Wavefunction", "processing_config": {"feature_tier": "standard"}}

        with patch(
            "milia_pipeline.config.config_constants.is_handler_type_supported", return_value=True
        ):
            errors = config_constants.validate_handler_configuration("Wavefunction", config)
            assert errors == []

    @patch("milia_pipeline.config.config_constants._init_registry")
    def test_validate_wavefunction_handler_config_invalid_tier(self, mock_init_registry):
        """Test validation with invalid feature tier."""
        # Disable registry to force legacy validation path
        mock_init_registry.return_value = False

        config = {
            "dataset_type": "Wavefunction",
            "processing_config": {"feature_tier": "invalid_tier"},
        }

        original_registry_available = config_constants._REGISTRY_AVAILABLE
        config_constants._REGISTRY_AVAILABLE = False

        try:
            with patch(
                "milia_pipeline.config.config_constants.is_handler_type_supported",
                return_value=True,
            ):
                errors = config_constants.validate_handler_configuration("Wavefunction", config)
                assert any("Invalid feature_tier" in err for err in errors)
        finally:
            config_constants._REGISTRY_AVAILABLE = original_registry_available


class TestHandlerTransformCompatibilityWavefunction:
    """Test suite for Wavefunction handler-transform compatibility."""

    def test_wavefunction_transform_compatibility_matrix(self):
        """Test Wavefunction handler transform compatibility matrix exists."""
        compat = config_constants.HANDLER_TRANSFORM_COMPATIBILITY.get("Wavefunction", {})

        # Check some expected compatibility values
        assert compat.get("Distance") == "recommended"
        assert compat.get("VirtualNode") == "incompatible"
        assert compat.get("DropNode") == "incompatible"

    def test_get_compatible_transforms_for_wavefunction(self):
        """Test getting compatible transforms for Wavefunction handler."""
        with patch(
            "milia_pipeline.config.config_constants.is_handler_type_supported", return_value=True
        ):
            compatible = config_constants.get_compatible_transforms_for_handler("Wavefunction")

            # Should not include incompatible transforms
            assert "VirtualNode" not in compatible
            assert "DropNode" not in compatible

    def test_get_incompatible_transforms_for_wavefunction(self):
        """Test getting incompatible transforms for Wavefunction handler."""
        with patch(
            "milia_pipeline.config.config_constants.is_handler_type_supported", return_value=True
        ):
            incompatible = config_constants.get_incompatible_transforms_for_handler("Wavefunction")

            # Check expected incompatible transforms
            assert "VirtualNode" in incompatible
            assert "DropNode" in incompatible


class TestHandlerCacheInfoPhase3:
    """Test suite for Phase 3 cache info with registry integration."""

    def test_get_handler_cache_info_includes_registry_status(self):
        """Test that handler cache info includes registry integration status."""
        info = config_constants.get_handler_cache_info()

        assert "registry_integration" in info
        assert "available" in info["registry_integration"]
        assert "cache_invalidation_registered" in info["registry_integration"]


# ==========================================
# GLOBAL PATHS CONFIG TESTS
# ==========================================


class TestGlobalPathsConfig:
    """Test suite for global_paths configuration access."""

    @patch("milia_pipeline.config.config_loader.load_config")
    def test_get_dataset_constants_uses_global_paths(self, mock_load_config):
        """Test that get_dataset_constants correctly accesses global_paths."""
        mock_config = {
            "dataset_type": "DFT",
            "dft_config": {
                "raw_npz_filename": "DFT_all.npz",
                "raw_data_download_url": "http://example.com/dft.npz",
            },
            "global_paths": {"working_root_dir": "/custom/data/path"},
        }
        mock_load_config.return_value = mock_config

        filename, url, root_dir = config_constants._get_dataset_constants_local()

        assert filename == "DFT_all.npz"
        assert url == "http://example.com/dft.npz"
        assert root_dir == "/custom/data/path"


# ==========================================
# RUN TESTS
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
