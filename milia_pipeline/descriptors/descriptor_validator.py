"""
Descriptor Validation System

Provides validation for:
- Descriptor values (NaN, Inf, range checking)
- Molecular requirements (3D coords, charges)
- Configuration validation
- Descriptor filtering by requirements

Pydantic V2 Migration (Phase 29):
    - Migrated ValidationResult from @dataclass to Pydantic BaseModel (mutable)
    - Uses Field(default_factory=list) and Field(default_factory=dict) for mutable defaults
    - Removed __post_init__ - default factory pattern handles None → empty conversion
    - Added to_dict() method wrapping model_dump() for backward compatibility
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: Milia Team
Version: 1.1.0
"""

import math
import logging
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field

from rdkit import Chem

from .descriptor_categories import (
    DescriptorCategory,
    get_descriptor_metadata,
    requires_3d_coordinates,
    requires_partial_charges,
)
from milia_pipeline.exceptions import DescriptorValidationError


logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT DATACLASS
# =============================================================================

class ValidationResult(BaseModel):
    """
    Result of a validation check.
    
    Pydantic V2 Migration (Phase 29):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=list) for warnings (replaces None default + __post_init__)
        - Uses Field(default_factory=dict) for details (replaces None default + __post_init__)
        - NON-BREAKING: Same constructor API and attribute access preserved
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages (empty if valid)
        warnings: List of warning messages
        details: Additional validation details
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Backward compatible method wrapping Pydantic V2's model_dump().
        
        Returns:
            Dictionary with all 4 fields: is_valid, errors, warnings, details
        """
        return self.model_dump()


# =============================================================================
# DESCRIPTOR VALIDATOR
# =============================================================================

class DescriptorValidator:
    """
    Validator for molecular descriptors.
    
    Provides validation for descriptor values, molecular requirements,
    and configuration parameters.
    
    Usage:
        >>> validator = DescriptorValidator()
        >>> is_valid, msg = validator.validate_value("MolWt", 180.2)
        >>> can_calc, missing = validator.check_requirements(mol, "RadiusOfGyration")
    """
    
    def __init__(self):
        """Initialize validator"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # =========================================================================
    # VALUE VALIDATION
    # =========================================================================
    
    def validate_value(
        self,
        descriptor_name: str,
        value: Union[float, int],
        allow_nan: bool = False,
        allow_inf: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate a descriptor value.
        
        Args:
            descriptor_name: Name of the descriptor
            value: Descriptor value to validate
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            min_value: Optional minimum value (inclusive)
            max_value: Optional maximum value (inclusive)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for NaN
        if math.isnan(value):
            if not allow_nan:
                return False, f"Descriptor {descriptor_name} returned NaN"
            return True, ""
        
        # Check for Inf
        if math.isinf(value):
            if not allow_inf:
                return False, f"Descriptor {descriptor_name} returned Inf"
            return True, ""
        
        # Range checking
        if min_value is not None and value < min_value:
            return False, f"Descriptor {descriptor_name} value {value} below minimum {min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"Descriptor {descriptor_name} value {value} above maximum {max_value}"
        
        return True, ""
    
    def validate_batch_values(
        self,
        descriptor_values: Dict[str, float],
        **validation_kwargs
    ) -> ValidationResult:
        """
        Validate a batch of descriptor values.
        
        Args:
            descriptor_values: Dictionary mapping descriptor_name -> value
            **validation_kwargs: Keyword arguments for validate_value()
        
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        invalid_descriptors = []
        
        for desc_name, value in descriptor_values.items():
            is_valid, error_msg = self.validate_value(
                desc_name, value, **validation_kwargs
            )
            
            if not is_valid:
                errors.append(error_msg)
                invalid_descriptors.append(desc_name)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details={"invalid_descriptors": invalid_descriptors}
        )
    
    # =========================================================================
    # REQUIREMENT CHECKING
    # =========================================================================
    
    def check_requirements(
        self,
        mol: Chem.Mol,
        descriptor_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if molecule meets requirements for descriptor calculation.
        
        Args:
            mol: RDKit molecule object
            descriptor_name: Name of descriptor to check
        
        Returns:
            Tuple of (can_calculate, missing_requirements)
        """
        missing = []
        
        # Check 3D coordinates requirement
        if requires_3d_coordinates(descriptor_name):
            if not self._has_3d_coordinates(mol):
                missing.append("3D coordinates")
        
        # Check partial charges requirement
        if requires_partial_charges(descriptor_name):
            if not self._has_partial_charges(mol):
                missing.append("partial charges")
        
        return len(missing) == 0, missing
    
    def _has_3d_coordinates(self, mol: Chem.Mol) -> bool:
        """Check if molecule has 3D coordinates"""
        if mol is None:
            return False
        
        try:
            conf = mol.GetConformer()
            return conf.Is3D() if hasattr(conf, 'Is3D') else True
        except (ValueError, RuntimeError):
            return False
    
    def _has_partial_charges(self, mol: Chem.Mol) -> bool:
        """Check if molecule has partial charges"""
        if mol is None:
            return False
        
        try:
            for atom in mol.GetAtoms():
                if atom.HasProp('_GasteigerCharge') or atom.HasProp('_PartialCharge'):
                    return True
            return False
        except Exception:
            return False
    
    def filter_by_requirements(
        self,
        mol: Chem.Mol,
        descriptor_names: List[str]
    ) -> Dict[str, Any]:
        """
        Filter descriptors based on molecular requirements.
        
        Args:
            mol: RDKit molecule object
            descriptor_names: List of descriptor names to check
        
        Returns:
            Dictionary with:
                - valid: List of descriptor names that can be calculated
                - invalid: Dictionary mapping descriptor -> missing requirements
        """
        has_3d = self._has_3d_coordinates(mol)
        has_charges = self._has_partial_charges(mol)
        
        valid = []
        invalid = {}
        
        for desc_name in descriptor_names:
            missing = []
            
            if requires_3d_coordinates(desc_name) and not has_3d:
                missing.append("3D coordinates")
            
            if requires_partial_charges(desc_name) and not has_charges:
                missing.append("partial charges")
            
            if missing:
                invalid[desc_name] = missing
            else:
                valid.append(desc_name)
        
        return {
            "valid": valid,
            "invalid": invalid,
            "molecule_has_3d": has_3d,
            "molecule_has_charges": has_charges
        }
    
    # =========================================================================
    # CONFIGURATION VALIDATION
    # =========================================================================
    
    def validate_configuration(
        self,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate descriptor configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        
        # Check enabled field
        if 'enabled' not in config:
            warnings.append("'enabled' field not found in config, assuming False")
        elif not isinstance(config['enabled'], bool):
            errors.append("'enabled' must be boolean")
        
        # Check selection_mode
        if 'selection_mode' in config:
            valid_modes = ['explicit', 'category', 'all']
            if config['selection_mode'] not in valid_modes:
                errors.append(f"Invalid selection_mode: {config['selection_mode']}, must be one of {valid_modes}")
        
        # Check selected_descriptors if present
        if 'selected_descriptors' in config:
            if not isinstance(config['selected_descriptors'], (dict, list)):
                errors.append("'selected_descriptors' must be dict or list")
        
        # Check computation settings
        if 'computation' in config:
            comp_config = config['computation']
            
            if 'batch_size' in comp_config:
                if not isinstance(comp_config['batch_size'], int) or comp_config['batch_size'] <= 0:
                    errors.append("'batch_size' must be positive integer")
        
        # Check plugin settings
        if 'plugins' in config:
            plugin_config = config['plugins']
            
            if 'enabled' in plugin_config and not isinstance(plugin_config['enabled'], bool):
                errors.append("'plugins.enabled' must be boolean")
            
            if 'plugin_paths' in plugin_config and not isinstance(plugin_config['plugin_paths'], list):
                errors.append("'plugins.plugin_paths' must be list")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


# =============================================================================
# GLOBAL VALIDATOR INSTANCE
# =============================================================================

# Global validator instance for convenience
validator = DescriptorValidator()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_value(descriptor_name: str, value: float, **kwargs) -> Tuple[bool, str]:
    """Validate descriptor value using global validator"""
    return validator.validate_value(descriptor_name, value, **kwargs)


def check_requirements(mol: Chem.Mol, descriptor_name: str) -> Tuple[bool, List[str]]:
    """Check requirements using global validator"""
    return validator.check_requirements(mol, descriptor_name)


def filter_by_requirements(mol: Chem.Mol, descriptor_names: List[str]) -> Dict[str, Any]:
    """Filter descriptors using global validator"""
    return validator.filter_by_requirements(mol, descriptor_names)
