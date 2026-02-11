"""
Protocol definitions for dataset handlers, converters, and validators.

These protocols define the contracts that all dataset-related classes must fulfill.
Based on evidence from dataset_handlers.py DatasetHandler ABC (lines 166-403).

The DatasetHandlerProtocol captures ALL 11 abstract methods from the actual
DatasetHandler abstract base class.
"""

from typing import Protocol, runtime_checkable, Dict, List, Any, Optional
import numpy as np
from torch_geometric.data import Data


@runtime_checkable
class DatasetHandlerProtocol(Protocol):
    """
    Contract that all dataset handlers must fulfill.
    
    This Protocol captures ALL 11 abstract methods from dataset_handlers.py
    DatasetHandler ABC (lines 229-403).
    
    The @runtime_checkable decorator enables isinstance() checks at runtime.
    """
    
    def get_dataset_type(self) -> str:
        """Return the dataset type this handler supports."""
        ...
    
    def validate_molecule_data(
        self,
        raw_properties_dict: Dict[str, Any],
        molecule_index: int,
        identifier: str = "N/A"
    ) -> None:
        """Validate dataset-specific molecular data with exception handling."""
        ...
    
    def get_required_properties(self) -> List[str]:
        """Get list of properties required for this dataset type."""
        ...
    
    def process_property_value(
        self,
        key: str,
        value: Any,
        molecule_index: int,
        identifier: str = "N/A"
    ) -> Any:
        """Process a property value according to dataset-specific requirements."""
        ...
    
    def enrich_pyg_data(
        self,
        pyg_data: Data,
        raw_properties_dict: Dict[str, Any],
        molecule_index: int,
        identifier: str = "N/A"
    ) -> Data:
        """Add dataset-specific enrichments to PyG Data object."""
        ...
    
    def get_processing_statistics(
        self,
        processed_molecules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate dataset-specific processing statistics."""
        ...
    
    def get_supported_structural_features(self) -> Dict[str, List[str]]:
        """Get structural features supported by this dataset type."""
        ...
    
    def get_molecular_charge(
        self,
        raw_properties_dict: Dict[str, Any],
        atomic_numbers: np.ndarray,
        mol_identifier: Optional[str] = None
    ) -> int:
        """Determine molecular charge from available data."""
        ...
    
    def get_molecule_creation_strategy(self) -> str:
        """Determine the molecule creation strategy for this dataset type."""
        ...
    
    def get_transform_recommendations(self) -> Dict[str, List[str]]:
        """Get transform recommendations without requiring validation."""
        ...
    
    def get_supported_descriptors(self) -> Dict[str, List[str]]:
        """Get molecular descriptors supported by this dataset type."""
        ...


@runtime_checkable
class DatasetConverterProtocol(Protocol):
    """Contract for data converters."""
    
    def convert(self, raw_data: Any) -> Data:
        """Convert raw data to PyG format."""
        ...
    
    def supports_format(self, format_type: str) -> bool:
        """Check if converter supports given format."""
        ...


@runtime_checkable
class DatasetValidatorProtocol(Protocol):
    """Contract for data validators."""
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset-specific data."""
        ...
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Return validation rules for inspection."""
        ...
