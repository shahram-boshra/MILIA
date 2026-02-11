"""
Base classes for dataset definitions.

This module provides:
- DatasetMetadata: Immutable metadata for a dataset type
- DatasetSchema: Immutable schema definition
- DatasetFeatures: Immutable feature support flags
- BaseDataset: Abstract base class with __init_subclass__ validation

Evidence sources:
- Refactoring plan v2.1.0 (lines 491-731)
- config_containers.py frozen dataclass pattern (lines 32-77)
- dataset_handlers.py handler constructor (lines 178-227)
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Tuple, Optional, Type, Any
from pydantic.dataclasses import dataclass  # Pydantic V2 drop-in replacement with runtime validation
from pydantic import field_validator  # Available for future field-level validators
from dataclasses import field  # Keep for default_factory compatibility
import logging

from milia_pipeline.datasets.protocols import (
    DatasetHandlerProtocol,
    DatasetConverterProtocol,
    DatasetValidatorProtocol
)


@dataclass(frozen=True)
class DatasetMetadata:
    """
    Immutable metadata for a dataset type.
    
    Attributes:
        name: Unique identifier for the dataset type (e.g., 'DFT', 'QM9')
        version: Semantic version string
        description: Human-readable description
        author: Optional author or organization
        license: Optional license identifier
    """
    name: str
    version: str
    description: str
    author: Optional[str] = None
    license: Optional[str] = None
    
    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("DatasetMetadata.name must be a non-empty string")
        if not self.version or not isinstance(self.version, str):
            raise ValueError("DatasetMetadata.version must be a non-empty string")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("DatasetMetadata.description must be a non-empty string")


@dataclass(frozen=True)
class DatasetSchema:
    """
    Immutable schema definition for a dataset.
    
    Attributes:
        required_properties: Properties that must be present
        optional_properties: Properties that may be present
        identifier_keys: Mappings from NPZ key to identifier type
        coordinate_units: Unit of coordinates ('angstrom' or 'bohr')
        energy_units: Unit of energy values
    """
    required_properties: Tuple[str, ...]
    optional_properties: Tuple[str, ...] = ()
    identifier_keys: Tuple[Tuple[str, str], ...] = ()
    coordinate_units: str = 'angstrom'
    energy_units: str = 'hartree'
    
    def __post_init__(self):
        if not isinstance(self.required_properties, tuple):
            raise TypeError("required_properties must be a tuple")
        if not self.required_properties:
            raise ValueError("required_properties cannot be empty")
        
        valid_coord_units = ('angstrom', 'bohr')
        if self.coordinate_units not in valid_coord_units:
            raise ValueError(f"coordinate_units must be one of {valid_coord_units}")
        
        valid_energy_units = ('hartree', 'eV', 'kcal/mol', 'kJ/mol')
        if self.energy_units not in valid_energy_units:
            raise ValueError(f"energy_units must be one of {valid_energy_units}")


@dataclass(frozen=True)
class DatasetFeatures:
    """
    Immutable feature support flags.
    
    Attributes:
        vibrational_analysis: Support for frequency/vibration data
        uncertainty_handling: Support for uncertainty/std values
        atomization_energy: Support for atomization energy calculation
        rotational_constants: Support for rotational constants
        frequency_analysis: Support for frequency analysis
        orbital_analysis: Support for MO analysis (Wavefunction-specific)
        homo_lumo_gap: Support for HOMO-LUMO gap
        mo_energies: Support for MO energies
    """
    vibrational_analysis: bool = False
    uncertainty_handling: bool = False
    atomization_energy: bool = False
    rotational_constants: bool = False
    frequency_analysis: bool = False
    orbital_analysis: bool = False
    homo_lumo_gap: bool = False
    mo_energies: bool = False
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary for compatibility with existing code."""
        return {
            'vibrational_analysis': self.vibrational_analysis,
            'uncertainty_handling': self.uncertainty_handling,
            'atomization_energy': self.atomization_energy,
            'rotational_constants': self.rotational_constants,
            'frequency_analysis': self.frequency_analysis,
            'orbital_analysis': self.orbital_analysis,
            'homo_lumo_gap': self.homo_lumo_gap,
            'mo_energies': self.mo_energies,
        }
    
    def supports(self, feature_name: str) -> bool:
        """Check if a specific feature is supported."""
        return self.to_dict().get(feature_name, False)


class BaseDataset(ABC):
    """
    Abstract base class for all dataset types.
    
    Subclasses MUST:
    1. Define class attributes: metadata, schema, features, config_key
    2. Implement abstract methods: get_required_properties, get_feature_support, 
       get_molecule_creation_strategy
    
    The __init_subclass__ method provides compile-time validation that catches
    missing attributes immediately when the class is defined.
    """
    
    metadata: ClassVar[DatasetMetadata]
    schema: ClassVar[DatasetSchema]
    features: ClassVar[DatasetFeatures]
    config_key: ClassVar[str]
    
    handler_class: ClassVar[Optional[Type[DatasetHandlerProtocol]]] = None
    converter_class: ClassVar[Optional[Type[DatasetConverterProtocol]]] = None
    validator_class: ClassVar[Optional[Type[DatasetValidatorProtocol]]] = None
    
    @classmethod
    @abstractmethod
    def get_required_properties(cls) -> List[str]:
        """Return list of required properties for this dataset."""
        ...
    
    @classmethod
    @abstractmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        """Return feature support dictionary."""
        ...
    
    @classmethod
    @abstractmethod
    def get_molecule_creation_strategy(cls) -> str:
        """Return molecule creation strategy ('identifier_coordinate_based' or 'coordinate_based')."""
        ...
    
    @classmethod
    def get_optional_properties(cls) -> List[str]:
        """Return list of optional properties (can be overridden)."""
        return list(cls.schema.optional_properties)
    
    @classmethod
    def get_identifier_keys(cls) -> List[Tuple[str, str]]:
        """Return identifier key mappings."""
        return list(cls.schema.identifier_keys)
    
    @classmethod
    def get_coordinate_units(cls) -> str:
        """Return coordinate units."""
        return cls.schema.coordinate_units
    
    @classmethod
    def get_energy_units(cls) -> str:
        """Return energy units."""
        return cls.schema.energy_units
    
    @classmethod
    def create_handler(
        cls,
        dataset_config: Any,
        filter_config: Any,
        processing_config: Any,
        logger: logging.Logger,
        experimental_setup: Optional[str] = None
    ) -> DatasetHandlerProtocol:
        """
        Factory method to create handler instance.
        Signature matches dataset_handlers.py handler constructor (lines 178-227).
        """
        if cls.handler_class is not None:
            return cls.handler_class(
                dataset_config,
                filter_config,
                processing_config,
                logger,
                experimental_setup
            )
        
        raise NotImplementedError(
            f"Dataset '{cls.metadata.name}' does not have a handler_class defined. "
            f"Either set handler_class or override create_handler()."
        )
    
    @classmethod
    def get_config_schema(cls) -> Optional[Type]:
        """Return Pydantic model for config validation. Override for custom schema."""
        return None
    
    def __init_subclass__(cls, **kwargs):
        """
        Validate subclass implementation at class definition time.
        This catches missing attributes immediately when the class is defined.
        """
        super().__init_subclass__(**kwargs)
        
        if ABC in cls.__bases__:
            return
        
        if hasattr(cls, '__abstractmethods__') and cls.__abstractmethods__:
            return
        
        required_attrs = ['metadata', 'schema', 'features', 'config_key']
        missing = [attr for attr in required_attrs if not hasattr(cls, attr)]
        
        if missing:
            raise TypeError(
                f"Dataset class '{cls.__name__}' missing required class attributes: {missing}"
            )
        
        if not isinstance(cls.metadata, DatasetMetadata):
            raise TypeError(
                f"'{cls.__name__}.metadata' must be a DatasetMetadata instance, "
                f"got {type(cls.metadata).__name__}"
            )
        
        if not isinstance(cls.schema, DatasetSchema):
            raise TypeError(
                f"'{cls.__name__}.schema' must be a DatasetSchema instance, "
                f"got {type(cls.schema).__name__}"
            )
        
        if not isinstance(cls.features, DatasetFeatures):
            raise TypeError(
                f"'{cls.__name__}.features' must be a DatasetFeatures instance, "
                f"got {type(cls.features).__name__}"
            )
        
        if not isinstance(cls.config_key, str) or not cls.config_key:
            raise TypeError(
                f"'{cls.__name__}.config_key' must be a non-empty string"
            )
