# custom_transforms.py
# Extensible Domain Transform Layer - Custom Transform Base Classes
# milia Pipeline - Extensible Domain-Specific Molecular Transformations

"""
Custom Transform Framework for milia Pipeline
=============================================

Extensible base classes for domain-specific molecular transformations that integrate
seamlessly with existing TransformRegistry and validation infrastructure from the Core Registry and Introspection layers.

Key Features:
- Integration with Core TransformRegistry infrastructure
- Advanced parameter introspection and validation support
- milia-specific quantum property handling
- Chemistry-aware validation
- Research-grade metadata tracking

Architecture:
    CustomTransformBase (abstract)
        └── MolecularTransformBase (chemistry validation)
            └── QuantumTransformBase (milia quantum properties)

Usage:
    >>> class MyTransform(QuantumTransformBase):
    ...     def __init__(self, threshold: float = 0.5):
    ...         super().__init__()
    ...         self.threshold = threshold
    ...
    ...     def transform(self, data: Data) -> Data:
    ...         # Custom logic
    ...         return data
    ...
    ...     @classmethod
    ...     def get_metadata(cls) -> TransformMetadata:
    ...         return TransformMetadata(
    ...             name="MyTransform",
    ...             version="1.0.0",
    ...             author="Researcher Name",
    ...             category="quantum",
    ...             description="Transform description"
    ...         )

Integration Points:
- TransformRegistry: Auto-discovery and registration
- Parameter introspection: Type hints and constraints
- Validation system: Multi-level validation
- Configuration system: YAML compatibility

Author: milia Team
Layer: Extensible Domain Transform Infrastructure (Custom Base Classes)
Dependencies: Core Registry Layer (graph_transforms), Introspection Layer (parameter introspection, validation, configuration)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# Core infrastructure imports (Registry and Introspection layers)
# Lazy import to avoid circular dependency with plugin_system
GRAPH_TRANSFORMS_AVAILABLE = False
TransformRegistry = None
ValidationLevel = None
ValidationScope = None
get_transform_info = None
validate_comprehensive = None


def _lazy_import_graph_transforms():
    """Lazy import to avoid circular dependency"""
    global GRAPH_TRANSFORMS_AVAILABLE, TransformRegistry, ValidationLevel, ValidationScope
    global get_transform_info, validate_comprehensive, _IMPORTING_GRAPH_TRANSFORMS

    if GRAPH_TRANSFORMS_AVAILABLE and TransformRegistry is not None:
        return True

    # Prevent re-entry
    if _IMPORTING_GRAPH_TRANSFORMS:
        return False

    _IMPORTING_GRAPH_TRANSFORMS = True
    try:
        from .graph_transforms import TransformRegistry as _TR
        from .graph_transforms import ValidationLevel as _VL
        from .graph_transforms import ValidationScope as _VS
        from .graph_transforms import get_transform_info as _GTI
        from .graph_transforms import validate_comprehensive as _VC

        TransformRegistry = _TR
        ValidationLevel = _VL
        ValidationScope = _VS
        get_transform_info = _GTI
        validate_comprehensive = _VC
        GRAPH_TRANSFORMS_AVAILABLE = True
        return True
    except ImportError as e:
        logging.getLogger(__name__).debug(f"Failed to import graph_transforms: {e}")
        return False
    finally:
        _IMPORTING_GRAPH_TRANSFORMS = False


# Define minimal fallbacks at MODULE level (not inside function)
if not GRAPH_TRANSFORMS_AVAILABLE:

    class ValidationLevel:
        STRICT = "strict"
        NORMAL = "normal"
        LENIENT = "lenient"

    class ValidationScope:
        FULL = "full"
        BASIC = "basic"
        MINIMAL = "minimal"


try:
    from ..config.config_containers import TransformSpec

    CONFIG_CONTAINERS_AVAILABLE = True
except ImportError:
    CONFIG_CONTAINERS_AVAILABLE = False
    TransformSpec = None

try:
    from ..exceptions import (
        TransformConfigurationError,
        TransformExecutionError,
        TransformValidationError,
    )

    EXCEPTIONS_AVAILABLE = True
except ImportError:
    # Fallback exception definitions
    EXCEPTIONS_AVAILABLE = False

    class TransformValidationError(Exception):
        """Validation error during transform configuration or execution."""

        def __init__(self, message: str, transform_name: str = None, **kwargs):
            super().__init__(message)
            self.transform_name = transform_name
            self.details = kwargs

    class TransformExecutionError(Exception):
        """Error during transform execution."""

        def __init__(
            self, message: str, transform_name: str = None, original_error: Exception = None
        ):
            super().__init__(message)
            self.transform_name = transform_name
            self.original_error = original_error

    class TransformConfigurationError(Exception):
        """Error in transform configuration."""

        def __init__(self, message: str, transform_name: str = None, **kwargs):
            super().__init__(message)
            self.transform_name = transform_name
            self.details = kwargs


# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# TRANSFORM METADATA
# =============================================================================


class TransformMetadata(BaseModel):
    """
    Metadata for custom transforms.

    Integrates with the parameter introspection and validation system and provides
    comprehensive information for transform discovery, validation, and documentation.

    Pydantic V2 Migration (Phase 16):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses model_dump() for to_dict() method (backward compatible)
        - NON-BREAKING: Same constructor API and attribute access
        - Follows established pattern from device_manager.py (Phase 7)

    Attributes:
        name: Transform name (must be unique in registry)
        version: Semantic version (e.g., "1.0.0")
        author: Author or organization name
        category: Transform category ('molecular', 'quantum', 'experimental', 'augmentation')
        description: Brief description of transform purpose
        paper_reference: Optional citation or paper reference
        github_url: Optional repository URL
        validated_datasets: List of datasets this transform has been validated on
        required_node_features: Required node attribute names (e.g., ['x', 'z'])
        required_edge_features: Required edge attribute names (e.g., ['edge_attr'])
        required_graph_attributes: Required graph-level attributes (e.g., ['energy', 'forces'])

    Example:
        >>> metadata = TransformMetadata(
        ...     name="NormalizeEnergy",
        ...     version="1.0.0",
        ...     author="milia Team",
        ...     category="quantum",
        ...     description="Normalize energy values for training stability",
        ...     validated_datasets=["milia_DFT", "milia_DMC"],
        ...     required_graph_attributes=["energy"]
        ... )
    """

    name: str
    version: str
    author: str
    category: str  # 'molecular', 'quantum', 'experimental', 'augmentation'
    description: str

    # Optional fields
    paper_reference: str | None = None
    github_url: str | None = None
    validated_datasets: list[str] = Field(default_factory=list)

    # Parameter constraint introspection integration
    required_node_features: list[str] = Field(default_factory=list)
    required_edge_features: list[str] = Field(default_factory=list)
    required_graph_attributes: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metadata to dictionary for serialization.

        Backward compatible method wrapping Pydantic V2's model_dump().

        Returns:
            Dictionary representation of metadata
        """
        return self.model_dump()

    def __str__(self) -> str:
        """String representation for logging and debugging."""
        return (
            f"TransformMetadata(name='{self.name}', version='{self.version}', "
            f"category='{self.category}')"
        )


# =============================================================================
# CUSTOM TRANSFORM BASE CLASS
# =============================================================================


class CustomTransformBase(BaseTransform, ABC):
    """
    Base class for all custom molecular transformations.

    Integrates with core infrastructure layers:
    - Auto-discovery via TransformRegistry
    - Parameter constraint introspection
    - Multi-level validation integration
    - YAML configuration compatibility

    Design Pattern:
    1. Inherit from this class
    2. Implement transform() method with your custom logic
    3. Define get_metadata() class method with transform information
    4. Optional: Override get_parameter_constraints() for validation
    5. Optional: Implement validate_input() and validate_output() for custom validation

    Features:
    - Automatic error handling and wrapping
    - Usage statistics tracking
    - Validation compatibility checking
    - Integration with multi-level validation system (strict/normal/lenient)

    Example:
        >>> class MyMolecularTransform(CustomTransformBase):
        ...     def __init__(self, threshold: float = 0.5):
        ...         super().__init__()
        ...         self.threshold = threshold
        ...
        ...     def transform(self, data: Data) -> Data:
        ...         # Apply transformation
        ...         if hasattr(data, 'x'):
        ...             data.x = data.x * self.threshold
        ...         return data
        ...
        ...     @classmethod
        ...     def get_metadata(cls) -> TransformMetadata:
        ...         return TransformMetadata(
        ...             name="MyMolecularTransform",
        ...             version="1.0.0",
        ...             author="Researcher Name",
        ...             category="molecular",
        ...             description="Scale node features by threshold"
        ...         )
    """

    def __init__(self):
        """Initialize transform with metadata and tracking."""
        super().__init__()
        self._metadata = self.get_metadata()
        self._validation_cache: dict[str, Any] = {}
        self._call_count = 0
        self._error_count = 0
        self._logger = logging.getLogger(f"{__name__}.{self._metadata.name}")

    @abstractmethod
    def transform(self, data: Data) -> Data:
        """
        Apply the transformation to a PyG Data object.

        This is the core method that must be implemented by all custom transforms.
        It should be pure - no side effects - and all modifications should be
        to the returned Data object.

        Args:
            data: Input molecular graph (PyG Data object)

        Returns:
            Transformed molecular graph (modified Data object)

        Raises:
            TransformExecutionError: If transformation fails

        Note:
            - This method should be deterministic for reproducibility
            - Avoid modifying the input data in-place when possible
            - Use clone() for tensors if you need to preserve the original

        Example:
            >>> def transform(self, data: Data) -> Data:
            ...     data = data.clone()
            ...     data.x = data.x * 2.0
            ...     return data
        """
        pass

    def __call__(self, data: Data) -> Data | None:
        """
        Wrapper with error handling and validation.

        Integrates with multi-level validation system and provides automatic
        error handling, logging, and statistics tracking.

        Args:
            data: Input molecular graph

        Returns:
            Transformed data or None if transform signals skip

        Raises:
            TransformExecutionError: If transformation fails unrecoverably
        """
        try:
            # Pre-validation (optional override)
            if hasattr(self, "validate_input"):
                try:
                    self.validate_input(data)
                except Exception as e:
                    self._logger.warning(f"Input validation warning: {e}")
                    # Continue anyway unless strict mode

            # Apply transformation
            result = self.transform(data)

            # Handle None returns (filter transforms)
            if result is None:
                self._logger.debug(f"Transform {self._metadata.name} filtered out sample")
                self._call_count += 1
                return None

            # Post-validation (optional override)
            if hasattr(self, "validate_output"):
                try:
                    self.validate_output(result)
                except Exception as e:
                    self._logger.warning(f"Output validation warning: {e}")

            # Track successful usage
            self._call_count += 1

            return result

        except TransformExecutionError:
            # Re-raise our own exceptions
            self._error_count += 1
            raise
        except Exception as e:
            # Wrap other exceptions with context
            self._error_count += 1
            error_msg = (
                f"Transform '{self._metadata.name}' failed on sample "
                f"with {data.num_nodes} nodes: {str(e)}"
            )
            self._logger.error(error_msg)
            raise TransformExecutionError(
                error_msg, transform_name=self._metadata.name, original_error=e
            ) from e

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> TransformMetadata:
        """
        Provide metadata for the transform.

        Used for:
        - Auto-discovery in TransformRegistry
        - Documentation generation
        - Validation and compatibility checking
        - Parameter constraint introspection

        Returns:
            TransformMetadata instance with complete information

        Example:
            >>> @classmethod
            >>> def get_metadata(cls) -> TransformMetadata:
            ...     return TransformMetadata(
            ...         name="MyTransform",
            ...         version="1.0.0",
            ...         author="Your Name",
            ...         category="molecular",
            ...         description="What this transform does",
            ...         required_node_features=["x"],
            ...         validated_datasets=["milia_DFT"]
            ...     )
        """
        pass

    @classmethod
    def get_parameter_constraints(cls) -> dict[str, Any]:
        """
        Define parameter constraints for validation.

        Integrates with parameter constraint introspection system. This method allows
        you to specify validation constraints that can't be expressed in type hints.

        Returns:
            Dictionary mapping parameter names to constraint specifications:
            {
                'parameter_name': {
                    'type': expected_type,
                    'range': (min_value, max_value),  # for numeric types
                    'min': min_value,  # alternative to range
                    'max': max_value,  # alternative to range
                    'choices': [valid_choice1, valid_choice2],  # for enums
                    'default': default_value,
                    'description': 'Human-readable description'
                }
            }

        Note:
            This supplements the automatic type hint introspection.
            Use this for constraints that can't be expressed in type hints alone.

        Example:
            >>> @classmethod
            >>> def get_parameter_constraints(cls) -> Dict[str, Any]:
            ...     return {
            ...         'threshold': {
            ...             'type': float,
            ...             'range': (0.0, 1.0),
            ...             'default': 0.5,
            ...             'description': 'Threshold for filtering'
            ...         },
            ...         'k_neighbors': {
            ...             'type': int,
            ...             'min': 1,
            ...             'max': 100,
            ...             'default': 10,
            ...             'description': 'Number of neighbors'
            ...         },
            ...         'method': {
            ...             'type': str,
            ...             'choices': ['average', 'max', 'sum'],
            ...             'default': 'average',
            ...             'description': 'Aggregation method'
            ...         }
            ...     }
        """
        return {}

    @classmethod
    def get_required_node_attributes(cls) -> set[str]:
        """
        Define required node attributes for validation.

        Returns:
            Set of required node attribute names (e.g., {'x', 'pos', 'z'})

        Example:
            >>> @classmethod
            >>> def get_required_node_attributes(cls) -> Set[str]:
            ...     return {'x', 'pos'}  # Requires node features and positions
        """
        return set()

    @classmethod
    def get_required_edge_attributes(cls) -> set[str]:
        """
        Define required edge attributes for validation.

        Returns:
            Set of required edge attribute names (e.g., {'edge_attr', 'edge_weight'})

        Example:
            >>> @classmethod
            >>> def get_required_edge_attributes(cls) -> Set[str]:
            ...     return {'edge_attr'}  # Requires edge features
        """
        return set()

    @classmethod
    def get_required_graph_attributes(cls) -> set[str]:
        """
        Define required graph-level attributes for validation.

        Returns:
            Set of required graph attribute names (e.g., {'energy', 'forces'})

        Example:
            >>> @classmethod
            >>> def get_required_graph_attributes(cls) -> Set[str]:
            ...     return {'energy', 'forces'}  # Requires QM properties
        """
        return set()

    def validate_compatibility(
        self,
        data: Data,
        validation_level: str = "normal",  # Changed from ValidationLevel enum to string
    ) -> tuple[bool, list[str]]:
        """
        Validate compatibility with input data.

        Integrates with multi-level validation system.

        Args:
            data: Input molecular graph
            validation_level: Strictness level ('strict', 'normal', 'lenient')

        Returns:
            Tuple of (is_compatible, list_of_warnings)

        Example:
            >>> transform = MyTransform()
            >>> data = Data(x=torch.ones(5, 3))
            >>> is_valid, warnings = transform.validate_compatibility(data)
            >>> if warnings:
            ...     print(f"Warnings: {warnings}")
        """
        warnings = []
        is_strict = validation_level == "strict"

        # Check required node attributes
        for attr in self.get_required_node_attributes():
            if attr == "x":
                if not hasattr(data, "x") or data.x is None:
                    if is_strict:
                        return False, ["Missing required node features (x)"]
                    warnings.append("Missing node features (x)")
            else:
                if not hasattr(data, attr):
                    if is_strict:
                        return False, [f"Missing required node attribute: {attr}"]
                    warnings.append(f"Missing node attribute: {attr}")

        # Check required edge attributes
        for attr in self.get_required_edge_attributes():
            if not hasattr(data, attr) or getattr(data, attr) is None:
                if is_strict:
                    return False, [f"Missing required edge attribute: {attr}"]
                warnings.append(f"Missing edge attribute: {attr}")

        # Check required graph attributes
        for attr in self.get_required_graph_attributes():
            if not hasattr(data, attr):
                if is_strict:
                    return False, [f"Missing required graph attribute: {attr}"]
                warnings.append(f"Missing graph attribute: {attr}")

        return True, warnings

    def get_usage_statistics(self) -> dict[str, Any]:
        """
        Get usage statistics for this transform instance.

        Returns:
            Dictionary with usage metrics:
            {
                'transform_name': str,
                'call_count': int,
                'error_count': int,
                'success_rate': float,
                'metadata': dict
            }

        Example:
            >>> transform = MyTransform()
            >>> # ... use transform ...
            >>> stats = transform.get_usage_statistics()
            >>> print(f"Success rate: {stats['success_rate']:.2%}")
        """
        total_calls = self._call_count + self._error_count
        success_rate = (self._call_count / total_calls) if total_calls > 0 else 0.0

        return {
            "transform_name": self._metadata.name,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "success_rate": success_rate,
            "metadata": self._metadata.to_dict(),
        }


# =============================================================================
# MOLECULAR TRANSFORM BASE CLASS
# =============================================================================


class MolecularTransformBase(CustomTransformBase):
    """
    Specialized base class for molecular graph transformations.

    Provides chemistry-specific utilities:
    - Atom type validation (atomic numbers 1-118)
    - Bond order checking (single, double, triple, aromatic)
    - Charge conservation validation
    - Coordinate manipulation helpers
    - Reasonable molecular size limits

    Use this for transforms that operate on molecular structures and need
    chemistry-aware validation.

    Chemistry Constants:
        VALID_ATOMIC_NUMBERS: Set of valid atomic numbers (1-118)
        VALID_BOND_TYPES: Set of valid bond orders (1-4)
        MIN_ATOMS: Minimum atoms per molecule (1)
        MAX_ATOMS: Maximum atoms per molecule (10000)

    Example:
        >>> class MyChemistryTransform(MolecularTransformBase):
        ...     def transform(self, data: Data) -> Data:
        ...         # Validate molecular structure first
        ...         is_valid, issues = self.validate_molecular_structure(data)
        ...         if not is_valid:
        ...             raise ValueError(f"Invalid molecule: {issues}")
        ...         # Apply transformation
        ...         return data
    """

    # Chemistry constants
    VALID_ATOMIC_NUMBERS = set(range(1, 119))  # H (1) to Og (118)
    VALID_BOND_TYPES = {1, 2, 3, 4}  # single, double, triple, aromatic

    # Reasonable limits for molecular systems
    MIN_ATOMS = 1
    MAX_ATOMS = 10000  # Reasonable upper limit for most molecules

    @classmethod
    def get_required_node_attributes(cls) -> set[str]:
        """Molecular graphs require at least node features (atomic numbers)."""
        return {"x"}

    def validate_molecular_structure(self, data: Data) -> tuple[bool, list[str]]:
        """
        Validate basic molecular structure constraints.

        Checks:
        - Valid atomic numbers (1-118)
        - Valid bond types (1-4) if edge_attr present
        - Reasonable atom count (1-10000)
        - Coordinate consistency if pos present
        - No NaN or infinite values

        Args:
            data: Molecular graph to validate

        Returns:
            Tuple of (is_valid, list_of_issues)

        Example:
            >>> data = Data(x=torch.tensor([[6], [8], [1]]))  # C, O, H
            >>> transform = MolecularTransformBase()
            >>> is_valid, issues = transform.validate_molecular_structure(data)
            >>> assert is_valid and len(issues) == 0
        """
        issues = []

        # Validate atomic numbers
        if hasattr(data, "x") and data.x is not None:
            # First column is atomic number for multi-feature tensors
            atomic_nums = data.x[:, 0] if data.x.dim() > 1 else data.x

            # Check for invalid atomic numbers
            invalid_atoms = [
                z.item() for z in atomic_nums if z.item() not in self.VALID_ATOMIC_NUMBERS
            ]
            if invalid_atoms:
                issues.append(
                    f"Invalid atomic numbers found: {invalid_atoms[:5]}"
                    + ("..." if len(invalid_atoms) > 5 else "")
                )

            # Check for NaN or infinite values
            if not torch.all(torch.isfinite(data.x)):
                issues.append("Non-finite values in node features")

        # Validate bond types
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            bond_types = data.edge_attr[:, 0] if data.edge_attr.dim() > 1 else data.edge_attr

            # Check for invalid bond types
            invalid_bonds = [b.item() for b in bond_types if b.item() not in self.VALID_BOND_TYPES]
            if invalid_bonds:
                issues.append(f"Invalid bond types found: {set(invalid_bonds)}")

            # Check for finite values
            if not torch.all(torch.isfinite(data.edge_attr)):
                issues.append("Non-finite values in edge features")

        # Validate atom count
        if data.num_nodes < self.MIN_ATOMS:
            issues.append(f"Too few atoms: {data.num_nodes} < {self.MIN_ATOMS}")
        if data.num_nodes > self.MAX_ATOMS:
            issues.append(f"Too many atoms: {data.num_nodes} > {self.MAX_ATOMS}")

        # Validate coordinates
        if hasattr(data, "pos") and data.pos is not None:
            if data.pos.shape[0] != data.num_nodes:
                issues.append(f"Coordinate count mismatch: {data.pos.shape[0]} != {data.num_nodes}")
            if data.pos.dim() != 2 or data.pos.shape[1] != 3:
                issues.append(f"Expected 3D coordinates [N, 3], got shape {data.pos.shape}")
            if not torch.all(torch.isfinite(data.pos)):
                issues.append("Non-finite coordinates detected")

        return len(issues) == 0, issues


# =============================================================================
# QUANTUM TRANSFORM BASE CLASS
# =============================================================================


class QuantumTransformBase(MolecularTransformBase):
    """
    Base class for quantum chemistry transformations.

    Specifically designed for milia quantum properties:
    - DFT energies (energy attribute)
    - DMC reference data (dmc_energy, dmc_uncertainty)
    - Vibrational modes (vibmodes)
    - Mulliken charges (charges)
    - Quantum forces (forces)

    Use this for transforms that manipulate quantum mechanical properties
    from DFT or DMC calculations.

    milia-Specific Attributes:
        energy: DFT total energy (kcal/mol)
        dmc_energy: DMC reference energy (kcal/mol)
        dmc_uncertainty: DMC statistical uncertainty
        vibmodes: Vibrational modes [n_modes, n_atoms, 3]
        charges: Mulliken atomic charges [n_atoms]
        forces: Atomic forces [n_atoms, 3]

    Example:
        >>> class MyQuantumTransform(QuantumTransformBase):
        ...     def transform(self, data: Data) -> Data:
        ...         # Validate quantum properties
        ...         is_valid, issues = self.validate_quantum_properties(data)
        ...         if not is_valid:
        ...             self._logger.warning(f"Quantum validation issues: {issues}")
        ...         # Apply transformation
        ...         return data
    """

    @classmethod
    def get_required_graph_attributes(cls) -> set[str]:
        """Quantum transforms typically need energy data."""
        return {"energy"}

    def validate_quantum_properties(self, data: Data) -> tuple[bool, list[str]]:
        """
        Validate quantum mechanical properties.

        milia-specific checks:
        - Energy values are finite
        - DMC uncertainty is positive (if present)
        - Charges sum correctly (if present)
        - Vibrational modes shape correct (if present)
        - Forces shape matches atoms (if present)

        Args:
            data: Molecular graph with quantum properties

        Returns:
            Tuple of (is_valid, list_of_issues)

        Example:
            >>> data = Data(
            ...     x=torch.ones(3, 1),
            ...     energy=torch.tensor(-100.5),
            ...     charges=torch.tensor([0.1, -0.05, -0.05])
            ... )
            >>> transform = QuantumTransformBase()
            >>> is_valid, issues = transform.validate_quantum_properties(data)
        """
        issues = []

        # Energy validation
        if hasattr(data, "energy") and data.energy is not None:
            if not torch.isfinite(data.energy):
                issues.append(f"Non-finite energy: {data.energy}")
        else:
            # Energy is typically required for quantum transforms
            issues.append("Missing required 'energy' attribute")

        # DMC uncertainty validation (milia-specific)
        if hasattr(data, "dmc_uncertainty"):
            if data.dmc_uncertainty < 0 and data.dmc_uncertainty is not None:
                issues.append(f"Negative DMC uncertainty: {data.dmc_uncertainty}")
            if not torch.isfinite(data.dmc_uncertainty):
                issues.append("Non-finite DMC uncertainty")

        # DMC energy validation
        if hasattr(data, "dmc_energy"):
            if not torch.isfinite(data.dmc_energy) and data.dmc_energy is not None:
                issues.append("Non-finite DMC energy")

        # Mulliken charges validation (milia-specific)
        if hasattr(data, "charges") and data.charges is not None:
            if data.charges.shape[0] != data.num_nodes:
                issues.append(f"Charge count mismatch: {data.charges.shape[0]} != {data.num_nodes}")
            if not torch.all(torch.isfinite(data.charges)):
                issues.append("Non-finite charges detected")

            # Check charge conservation (if total_charge available)

            if hasattr(data, "total_charge"):
                charge_sum = data.charges.sum()
                expected_charge = (
                    data.total_charge
                    if isinstance(data.total_charge, torch.Tensor)
                    else torch.tensor(data.total_charge)
                )

                # Check for non-finite values before comparison
                if not torch.isfinite(expected_charge):
                    issues.append("Non-finite total_charge value")
                elif not torch.isclose(charge_sum, expected_charge, atol=0.01):
                    issues.append(
                        f"Charge not conserved: sum={charge_sum:.3f}, expected={expected_charge:.3f}"
                    )

        # Vibrational modes validation (milia-specific)
        if hasattr(data, "vibmodes") and data.vibmodes is not None:
            # milia vibmodes: [n_modes, n_atoms, 3]
            if data.vibmodes.dim() == 3:
                if data.vibmodes.shape[1] != data.num_nodes:
                    issues.append(
                        f"Vibmode atom dimension mismatch: "
                        f"{data.vibmodes.shape[1]} != {data.num_nodes}"
                    )
                if data.vibmodes.shape[2] != 3:
                    issues.append(f"Expected 3D vibmodes, got shape {data.vibmodes.shape}")
                if not torch.all(torch.isfinite(data.vibmodes)):
                    issues.append("Non-finite vibrational modes")
            else:
                issues.append(f"Expected 3D vibmodes tensor, got {data.vibmodes.dim()}D")

        # Forces validation
        if hasattr(data, "forces") and data.forces is not None:
            if data.forces.shape[0] != data.num_nodes:
                issues.append(f"Forces count mismatch: {data.forces.shape[0]} != {data.num_nodes}")
            if data.forces.dim() != 2 or data.forces.shape[1] != 3:
                issues.append(f"Expected forces shape [N, 3], got {data.forces.shape}")
            if not torch.all(torch.isfinite(data.forces)):
                issues.append("Non-finite forces detected")

        return len(issues) == 0, issues


# =============================================================================
# EXAMPLE CUSTOM TRANSFORMS - milia SPECIFIC
# =============================================================================


class NormalizeVibrationalModes(QuantumTransformBase):
    """
    Normalize milia vibrational modes by magnitude.

    Research Use Cases:
    - Training stability improvement
    - Cross-molecule comparison
    - Ablation studies on vibrational mode importance
    - Feature scaling experiments

    milia-specific: Operates on 'vibmodes' attribute [n_modes, n_atoms, 3].

    Parameters:
        normalize_per_mode: If True, normalize each mode independently.
                          If False, normalize all modes together.
        epsilon: Small value to prevent division by zero.

    Example:
        >>> transform = NormalizeVibrationalModes(normalize_per_mode=True)
        >>> data = Data(
        ...     x=torch.ones(3, 1),
        ...     vibmodes=torch.randn(5, 3, 3),  # 5 modes, 3 atoms, 3D
        ...     num_nodes=3
        ... )
        >>> result = transform(data)
        >>> # Verify each mode has unit norm
        >>> for i in range(result.vibmodes.shape[0]):
        ...     mode_norm = torch.norm(result.vibmodes[i])
        ...     assert torch.isclose(mode_norm, torch.tensor(1.0), atol=1e-6)
    """

    def __init__(self, normalize_per_mode: bool = True, epsilon: float = 1e-8):
        """
        Initialize vibrational mode normalization transform.

        Args:
            normalize_per_mode: If True, normalize each mode independently.
                               If False, normalize all modes together.
            epsilon: Small value to prevent division by zero (default: 1e-8)
        """
        super().__init__()
        self.normalize_per_mode = normalize_per_mode
        self.epsilon = epsilon

    def transform(self, data: Data) -> Data:
        """Apply vibrational mode normalization."""

        if not hasattr(data, "vibmodes") or data.vibmodes is None:
            self._logger.warning("vibmodes lost during cloning, skipping normalization")
            return data

        # Check for NaN/Inf in input
        if not torch.all(torch.isfinite(data.vibmodes)):
            self._logger.warning("Non-finite values in input vibmodes, skipping normalization")
            return data

        # Validate tensor dimensionality
        if data.vibmodes.dim() != 3:
            self._logger.error(
                f"Expected 3D vibmodes [n_modes, n_atoms, 3], got {data.vibmodes.dim()}D"
            )
            raise TransformExecutionError(
                f"Invalid vibmodes dimensionality: {data.vibmodes.dim()}D",
                transform_name=self._metadata.name,
            )

        # Validate shape structure
        if data.vibmodes.shape[2] != 3:
            self._logger.error(
                f"Expected 3D coordinates in vibmodes, got shape {data.vibmodes.shape}"
            )
            raise TransformExecutionError(
                f"Invalid vibmodes shape: {data.vibmodes.shape}", transform_name=self._metadata.name
            )

        # CLONE DATA TO PREVENT IN-PLACE MODIFICATION
        data = data.clone()

        # Store original for debugging
        original_shape = data.vibmodes.shape

        if self.normalize_per_mode:
            norms = torch.norm(data.vibmodes.view(data.vibmodes.shape[0], -1), dim=1, keepdim=True)
            norms = norms.view(-1, 1, 1)

            # Validate shape compatibility before operations
            expected_norm_shape = (data.vibmodes.shape[0], 1, 1)
            if norms.shape != expected_norm_shape:
                self._logger.error(f"Norms shape mismatch: {norms.shape} != {expected_norm_shape}")
                raise TransformExecutionError(
                    "Shape mismatch in normalization", transform_name=self._metadata.name
                )

            # Check if any norms are too small (avoid division by near-zero)
            if torch.any(norms < self.epsilon):
                self._logger.warning(
                    f"Some mode norms below epsilon ({self.epsilon}), adding epsilon to prevent division issues"
                )

            norms = norms + self.epsilon
            data.vibmodes = data.vibmodes / norms  # Now safe - working on cloned data

            # Validate output shape unchanged
            if data.vibmodes.shape != original_shape:
                self._logger.error(
                    f"Vibmodes shape changed: {original_shape} -> {data.vibmodes.shape}"
                )
                raise TransformExecutionError(
                    "Shape changed during normalization", transform_name=self._metadata.name
                )

            # Verify no NaN/Inf produced
            if not torch.all(torch.isfinite(data.vibmodes)):
                self._logger.error("Non-finite values produced during per-mode normalization")
                raise TransformExecutionError(
                    f"Normalization produced non-finite values (shape={original_shape}, "
                    f"epsilon={self.epsilon}, normalize_per_mode={self.normalize_per_mode})",
                    transform_name=self._metadata.name,
                )
        else:
            # Global normalization
            global_norm = torch.norm(data.vibmodes)

            # Check if norm is too small
            if global_norm < self.epsilon:
                self._logger.warning(
                    f"Global norm ({global_norm:.2e}) below epsilon ({self.epsilon}), skipping normalization"
                )
                return data

            global_norm = global_norm + self.epsilon
            data.vibmodes = data.vibmodes / global_norm  # Working on cloned data

            # Verify no NaN/Inf produced
            if not torch.all(torch.isfinite(data.vibmodes)):
                self._logger.error("Non-finite values produced during global normalization")
                raise TransformExecutionError(
                    f"Normalization produced non-finite values (shape={original_shape}, "
                    f"global_norm={global_norm:.2e}, epsilon={self.epsilon})",
                    transform_name=self._metadata.name,
                )

        # Verify shape unchanged
        if hasattr(data, "vibmodes") and data.vibmodes is not None:
            assert data.vibmodes.shape == original_shape, "Shape changed during normalization"

        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        """Get transform metadata."""
        return TransformMetadata(
            name="NormalizeVibrationalModes",
            version="1.0.0",
            author="milia Team",
            category="quantum",
            description="Normalize milia vibrational modes for training stability",
            validated_datasets=["milia_DFT", "milia_DMC"],
            required_graph_attributes=["vibmodes"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict[str, Any]:
        """Define parameter constraints."""
        return {
            "normalize_per_mode": {
                "type": bool,
                "default": True,
                "description": "Normalize each mode independently (True) or globally (False)",
            },
            "epsilon": {
                "type": float,
                "range": (1e-12, 1e-6),
                "default": 1e-8,
                "description": "Small value to prevent division by zero",
            },
        }


class FilterByDMCUncertainty(QuantumTransformBase):
    """
    Filter molecules based on DMC uncertainty.

    Research Use Cases:
    - Remove high-uncertainty samples for cleaner training
    - Create difficulty-stratified datasets
    - Ablation studies on data quality impact
    - Uncertainty-aware dataset construction

    milia-specific: Operates on 'dmc_uncertainty' attribute.

    Parameters:
        max_uncertainty: Maximum allowed DMC uncertainty (kcal/mol)
        remove: If True, return None for filtered samples.
               If False, mark sample with flag for downstream filtering.

    Example:
        >>> # Single sample usage
        >>> transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)
        >>> data = Data(
        ...     x=torch.ones(3, 1),
        ...     dmc_uncertainty=torch.tensor(0.05),
        ...     num_nodes=3
        ... )
        >>> result = transform(data)
        >>> assert result.is_high_uncertainty == False
        >>>
        >>> # Research workflow: Filter dataset
        >>> from torch_geometric.loader import DataLoader
        >>> filter_transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=True)
        >>> clean_dataset = [filter_transform(d) for d in dataset if filter_transform(d) is not None]
        >>> # Use clean_dataset for training
    """

    def __init__(self, max_uncertainty: float = 0.1, remove: bool = False):
        """
        Initialize DMC uncertainty filter.

        Args:
            max_uncertainty: Maximum allowed DMC uncertainty (kcal/mol)
            remove: If True, return None for filtered samples.
                   If False, mark sample with flag for downstream filtering.
        """
        super().__init__()
        self.max_uncertainty = max_uncertainty
        self.remove = remove

    def transform(self, data: Data) -> Data | None:
        """Apply uncertainty-based filtering."""

        if not hasattr(data, "dmc_uncertainty"):
            self._logger.debug("No dmc_uncertainty attribute, passing through")
            return data

        # Validate it's a tensor
        if not isinstance(data.dmc_uncertainty, torch.Tensor):
            self._logger.error(f"dmc_uncertainty must be tensor, got {type(data.dmc_uncertainty)}")
            raise TransformExecutionError(
                "Invalid dmc_uncertainty type", transform_name=self._metadata.name
            )

        # Validate it's finite
        if not torch.isfinite(data.dmc_uncertainty):
            self._logger.warning("Non-finite dmc_uncertainty, skipping filtering")
            return data

        # Check uncertainty threshold
        is_high_uncertainty = data.dmc_uncertainty > self.max_uncertainty

        if self.remove and is_high_uncertainty:
            self._logger.debug(
                f"Filtered sample with uncertainty {data.dmc_uncertainty.item():.4f} > {self.max_uncertainty}"
            )
            return None

        # CLONE DATA TO PREVENT IN-PLACE MODIFICATION
        data = data.clone()

        # Mark sample with flag for downstream filtering
        data.is_high_uncertainty = is_high_uncertainty  # Working on cloned data

        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        """Get transform metadata."""
        return TransformMetadata(
            name="FilterByDMCUncertainty",
            version="1.0.0",
            author="milia Team",
            category="quantum",
            description="Filter molecules by DMC uncertainty threshold",
            validated_datasets=["milia_DMC"],
            required_graph_attributes=["dmc_uncertainty"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict[str, Any]:
        """Define parameter constraints."""
        return {
            "max_uncertainty": {
                "type": float,
                "range": (0.0, 10.0),
                "default": 0.1,
                "description": "Maximum allowed uncertainty in kcal/mol",
            },
            "remove": {
                "type": bool,
                "default": False,
                "description": "Remove samples (True) vs. mark for filtering (False)",
            },
        }


class ScaleMullikenCharges(QuantumTransformBase):
    """
    Scale Mulliken charges by a factor.

    Research Use Cases:
    - Charge magnitude ablation studies
    - Testing model sensitivity to charge features
    - Charge normalization experiments
    - Feature importance analysis

    milia-specific: Operates on 'charges' attribute.

    Parameters:
        scale_factor: Multiplication factor for charges
        center: If True, center charges to mean=0 before scaling

    Example:
        >>> transform = ScaleMullikenCharges(scale_factor=2.0, center=True)
        >>> data = Data(
        ...     x=torch.ones(3, 1),
        ...     charges=torch.tensor([0.5, -0.3, -0.2]),
        ...     num_nodes=3
        ... )
        >>> result = transform(data)
        >>> # Original: [0.5, -0.3, -0.2], mean=0.0
        >>> # After centering: [0.5, -0.3, -0.2] (already centered)
        >>> # After scaling: [1.0, -0.6, -0.4]
        >>> assert torch.allclose(result.charges, torch.tensor([1.0, -0.6, -0.4]))
    """

    def __init__(self, scale_factor: float = 1.0, center: bool = False):
        """
        Initialize charge scaling transform.

        Args:
            scale_factor: Multiplication factor for charges
            center: If True, center charges to mean=0 before scaling
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.center = center

    def transform(self, data: Data) -> Data:
        """Apply charge scaling."""

        if not hasattr(data, "charges") or data.charges is None:
            self._logger.debug("No charges attribute, skipping scaling")
            return data

        # Clone the entire data object to avoid modifying original
        data = data.clone()

        # Re-check after cloning (defensive)
        if not hasattr(data, "charges") or data.charges is None:
            self._logger.warning("charges lost during cloning, skipping scaling")
            return data

        # Validate charges is not empty
        if data.charges.numel() == 0:
            self._logger.warning("Empty charges tensor, skipping scaling")
            return data

        # Handle charges dimensionality flexibly
        if data.charges.dim() == 1:
            charges = data.charges.clone()
        elif data.charges.dim() == 2:
            if data.charges.size(1) == 1:
                charges = data.charges.squeeze(1).clone()
                self._logger.debug(f"Squeezed 2D charges {data.charges.shape} to 1D")
            else:
                charges = data.charges[:, 0].clone()
                self._logger.warning(f"Using first column from 2D charges {data.charges.shape}")
        else:
            self._logger.error(f"Expected 1D or 2D charges, got {data.charges.dim()}D")
            raise TransformExecutionError(
                f"Invalid charges dimensionality: {data.charges.dim()}D",
                transform_name=self._metadata.name,
            )

        charges = data.charges.clone()
        original_shape = charges.shape  # Store original shape

        # Check for non-finite values before operations
        if not torch.all(torch.isfinite(charges)):
            self._logger.warning("Non-finite values in charges, skipping scaling")
            return data

        if self.center:
            charge_mean = charges.mean()
            charges = charges - charge_mean
            self._logger.debug(f"Centered charges (mean was {charge_mean:.4f})")

            # Validate shape unchanged after centering
            if charges.shape != original_shape:
                self._logger.error(
                    f"Charges shape changed after centering: {original_shape} -> {charges.shape}"
                )
                raise TransformExecutionError(
                    "Shape mismatch after centering", transform_name=self._metadata.name
                )

        data.charges = charges * self.scale_factor

        # Validate shape unchanged after scaling
        if data.charges.shape != original_shape:
            self._logger.error(
                f"Charges shape changed after scaling: {original_shape} -> {data.charges.shape}"
            )
            raise TransformExecutionError(
                "Shape mismatch after scaling", transform_name=self._metadata.name
            )

        # Verify output is finite
        if not torch.all(torch.isfinite(data.charges)):
            self._logger.error("Non-finite values produced during scaling")
            raise TransformExecutionError("Scaling produced non-finite values")

        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        """Get transform metadata."""
        return TransformMetadata(
            name="ScaleMullikenCharges",
            version="1.0.0",
            author="milia Team",
            category="quantum",
            description="Scale Mulliken charges for ablation studies",
            validated_datasets=["milia_DFT", "milia_DMC"],
            required_graph_attributes=["charges"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict[str, Any]:
        """Define parameter constraints."""
        return {
            "scale_factor": {
                "type": float,
                "range": (0.0, 10.0),
                "default": 1.0,
                "description": "Multiplication factor for charges",
            },
            "center": {
                "type": bool,
                "default": False,
                "description": "Center charges to mean=0 before scaling",
            },
        }


# =============================================================================
# TARGET NORMALIZATION TRANSFORMS
# =============================================================================


class StandardizeTargets(QuantumTransformBase):
    """
    Standardize target values (y) using z-score normalization.

    Transforms targets to have mean=0 and std=1, which is critical for:
    - Training stability with large energy values (DFT energies in Hartrees)
    - Faster convergence during optimization
    - Meaningful loss values for monitoring

    Formula: y_standardized = (y - mean) / std

    The computed mean and std are stored in the Data object for inverse
    transformation during inference/evaluation.

    Attributes Stored:
        y_mean: Original mean of targets (for inverse transform)
        y_std: Original standard deviation of targets (for inverse transform)
        y_standardized: Flag indicating targets are standardized

    Example:
        >>> # In config.yaml standard_transforms:
        >>> # - name: "StandardizeTargets"
        >>> #   enabled: true
        >>> #   params:
        >>> #     attrs: ["y"]
        >>> #     eps: 1e-8
        >>>
        >>> transform = StandardizeTargets(attrs=["y"])
        >>> data = Data(x=torch.ones(3, 16), y=torch.tensor([-100.5, -98.2, -102.1]))
        >>> result = transform(data)
        >>> # result.y is now standardized (mean≈0, std≈1)
        >>> # result.y_mean and result.y_std stored for inverse transform

    Note:
        For DFT energies in Hartrees (typically -100 to -1000+ range),
        standardization brings values to a reasonable scale for neural networks.
    """

    def __init__(self, attrs: list[str] = None, eps: float = 1e-8):
        """
        Initialize standardization transform.

        Args:
            attrs: List of attribute names to standardize.
                   Default: ["y"] for target values.
                   Can include other attributes like ["y", "energy", "forces"].
            eps: Small constant to prevent division by zero when std is very small.
        """
        super().__init__()
        self.attrs = attrs if attrs is not None else ["y"]
        self.eps = eps

    def transform(self, data: Data) -> Data:
        """Apply z-score standardization to specified attributes."""
        # Clone to avoid modifying original
        data = data.clone()

        for attr in self.attrs:
            if not hasattr(data, attr) or getattr(data, attr) is None:
                self._logger.debug(f"Attribute '{attr}' not found, skipping standardization")
                continue

            values = getattr(data, attr)

            # Handle scalar values (single target per graph)
            if values.dim() == 0:
                values = values.unsqueeze(0)
                was_scalar = True
            else:
                was_scalar = False

            # Ensure float type for computation
            if not values.is_floating_point():
                values = values.float()

            # Check for valid values
            if not torch.all(torch.isfinite(values)):
                self._logger.warning(f"Non-finite values in '{attr}', skipping standardization")
                continue

            # Compute statistics
            mean = values.mean()
            # Use correction=0 (population std) for single-element tensors to avoid
            # "degrees of freedom <= 0" warning. Sample std (correction=1) is undefined
            # for n=1 since degrees of freedom would be 0. For single values, population
            # std is mathematically correct (std=0 for a single value).
            std = values.std(correction=0) if values.numel() == 1 else values.std()

            # Handle zero or near-zero std (constant values)
            if std < self.eps:
                # Only warn if multiple values have near-zero std (indicates constant data)
                # For single-element tensors, std=0 is mathematically expected and normal
                if values.numel() > 1:
                    self._logger.warning(
                        f"Standard deviation of '{attr}' is near zero ({std:.2e}), "
                        f"using eps={self.eps} to prevent division by zero"
                    )
                std = torch.tensor(self.eps, dtype=values.dtype, device=values.device)

            # Apply standardization
            standardized = (values - mean) / std

            # Restore scalar if needed
            if was_scalar:
                standardized = standardized.squeeze(0)

            # Store standardized values and statistics
            setattr(data, attr, standardized)
            setattr(data, f"{attr}_mean", mean)
            setattr(data, f"{attr}_std", std)

            self._logger.debug(
                f"Standardized '{attr}': mean={mean:.4f}, std={std:.4f} -> mean≈0, std≈1"
            )

        # Mark data as standardized
        data.targets_standardized = True

        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        """Get transform metadata."""
        return TransformMetadata(
            name="StandardizeTargets",
            version="1.0.0",
            author="milia Team",
            category="normalization",
            description="Z-score standardization of target values (mean=0, std=1) for training stability",
            validated_datasets=["milia_DFT", "milia_DMC", "QM9"],
            required_graph_attributes=["y"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict[str, Any]:
        """Define parameter constraints."""
        return {
            "attrs": {
                "type": list,
                "default": ["y"],
                "description": "List of attribute names to standardize",
            },
            "eps": {
                "type": float,
                "range": (1e-12, 1e-4),
                "default": 1e-8,
                "description": "Small constant to prevent division by zero",
            },
        }


class NormalizeTargets(QuantumTransformBase):
    """
    Normalize target values (y) to a specified range using min-max scaling.

    Transforms targets to a bounded range (default [0, 1]), which is useful for:
    - Targets with known bounds
    - Activation functions with bounded outputs (sigmoid, tanh)
    - Comparing models across different scales

    Formula: y_normalized = (y - min) / (max - min) * (range_max - range_min) + range_min

    The computed min and max are stored in the Data object for inverse
    transformation during inference/evaluation.

    Attributes Stored:
        y_min: Original minimum of targets (for inverse transform)
        y_max: Original maximum of targets (for inverse transform)
        y_normalized: Flag indicating targets are normalized

    Example:
        >>> # In config.yaml standard_transforms:
        >>> # - name: "NormalizeTargets"
        >>> #   enabled: true
        >>> #   params:
        >>> #     attrs: ["y"]
        >>> #     range_min: 0.0
        >>> #     range_max: 1.0
        >>>
        >>> transform = NormalizeTargets(attrs=["y"], range_min=0.0, range_max=1.0)
        >>> data = Data(x=torch.ones(3, 16), y=torch.tensor([-100.5, -98.2, -102.1]))
        >>> result = transform(data)
        >>> # result.y is now in [0, 1] range
        >>> # result.y_min and result.y_max stored for inverse transform

    Note:
        For DFT energies, standardization (StandardizeTargets) is typically
        preferred over min-max normalization because energy ranges can vary
        significantly across datasets and molecules.
    """

    def __init__(
        self,
        attrs: list[str] = None,
        range_min: float = 0.0,
        range_max: float = 1.0,
        eps: float = 1e-8,
    ):
        """
        Initialize normalization transform.

        Args:
            attrs: List of attribute names to normalize.
                   Default: ["y"] for target values.
            range_min: Minimum value of target range.
            range_max: Maximum value of target range.
            eps: Small constant to prevent division by zero.
        """
        super().__init__()
        self.attrs = attrs if attrs is not None else ["y"]
        self.range_min = range_min
        self.range_max = range_max
        self.eps = eps

        if range_min >= range_max:
            raise TransformConfigurationError(
                f"range_min ({range_min}) must be less than range_max ({range_max})",
                transform_name="NormalizeTargets",
            )

    def transform(self, data: Data) -> Data:
        """Apply min-max normalization to specified attributes."""
        # Clone to avoid modifying original
        data = data.clone()

        for attr in self.attrs:
            if not hasattr(data, attr) or getattr(data, attr) is None:
                self._logger.debug(f"Attribute '{attr}' not found, skipping normalization")
                continue

            values = getattr(data, attr)

            # Handle scalar values
            if values.dim() == 0:
                values = values.unsqueeze(0)
                was_scalar = True
            else:
                was_scalar = False

            # Ensure float type
            if not values.is_floating_point():
                values = values.float()

            # Check for valid values
            if not torch.all(torch.isfinite(values)):
                self._logger.warning(f"Non-finite values in '{attr}', skipping normalization")
                continue

            # Compute min and max
            val_min = values.min()
            val_max = values.max()
            val_range = val_max - val_min

            # Handle zero range (constant values)
            if val_range < self.eps:
                self._logger.warning(
                    f"Range of '{attr}' is near zero ({val_range:.2e}), "
                    f"setting to midpoint of target range"
                )
                midpoint = (self.range_min + self.range_max) / 2
                normalized = torch.full_like(values, midpoint)
            else:
                # Apply min-max normalization
                normalized = (values - val_min) / val_range
                # Scale to target range
                normalized = normalized * (self.range_max - self.range_min) + self.range_min

            # Restore scalar if needed
            if was_scalar:
                normalized = normalized.squeeze(0)

            # Store normalized values and statistics
            setattr(data, attr, normalized)
            setattr(data, f"{attr}_min", val_min)
            setattr(data, f"{attr}_max", val_max)

            self._logger.debug(
                f"Normalized '{attr}': [{val_min:.4f}, {val_max:.4f}] -> "
                f"[{self.range_min}, {self.range_max}]"
            )

        # Mark data as normalized
        data.targets_normalized = True

        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        """Get transform metadata."""
        return TransformMetadata(
            name="NormalizeTargets",
            version="1.0.0",
            author="milia Team",
            category="normalization",
            description="Min-max normalization of target values to specified range",
            validated_datasets=["milia_DFT", "milia_DMC", "QM9"],
            required_graph_attributes=["y"],
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict[str, Any]:
        """Define parameter constraints."""
        return {
            "attrs": {
                "type": list,
                "default": ["y"],
                "description": "List of attribute names to normalize",
            },
            "range_min": {
                "type": float,
                "default": 0.0,
                "description": "Minimum value of target range",
            },
            "range_max": {
                "type": float,
                "default": 1.0,
                "description": "Maximum value of target range",
            },
            "eps": {
                "type": float,
                "range": (1e-12, 1e-4),
                "default": 1e-8,
                "description": "Small constant to prevent division by zero",
            },
        }


# =============================================================================
# TARGET DISCRETIZATION TRANSFORM FOR CLASSIFICATION
# =============================================================================


class DiscretizeTargets(QuantumTransformBase):
    """
    Discretize continuous target values into class labels for classification tasks.

    Converts continuous regression targets (e.g., DFT energies, bond lengths) into
    discrete integer class indices suitable for classification with CrossEntropyLoss.

    Supported Classification Task Types (ALL software-supported classifications):
        - graph_classification: Graph-level targets (scalar or [1] or [num_targets])
        - node_classification: Node-level targets (shape [num_nodes] or [num_nodes, num_features])
        - link_prediction (multi-class): Edge-level targets (shape [num_edges] or [num_edges, num_features])
          Note: Standard link_prediction is binary (0/1). This enables multi-class edge classification
          when continuous edge values need discretization.

    This transform is essential when:
        - Using --task-type graph_classification with regression datasets
        - Using --task-type node_classification with continuous node labels
        - Using --task-type link_prediction with multi-class edge labels
        - Converting continuous quantum properties to categorical labels
        - Enabling classification-based analysis of continuous data

    Discretization Strategies:
        - 'quantile': Equal number of samples per bin (default, handles skewed data)
        - 'uniform': Equal-width bins across value range
        - 'kmeans': Bins based on k-means clustering centroids

    Target Level Detection:
        The target level can be specified explicitly via the target_level parameter,
        or detected automatically (target_level='auto', the default).

        Explicit target_level (recommended when task_type is known):
            - 'graph': Force graph-level treatment (for graph_classification)
            - 'node': Force node-level treatment (for node_classification)
            - 'edge': Force edge-level treatment (for link_prediction)

        Automatic detection (target_level='auto') uses this priority order:
            1. Attribute name: attr in ['edge_label', 'edge_y', 'edge_value', 'edge_target', 'edge_attr']
               → Always edge-level (highest priority, handles num_nodes == num_edges case)
            2. Dimension matching: first_dim == num_nodes → node-level
            3. Dimension matching: first_dim == num_edges → edge-level
            4. Default: graph-level (scalar, or dims don't match nodes/edges)

        WARNING: Automatic detection can fail when y.shape[0] == num_nodes for
        graph-level multi-target data. Use explicit target_level='graph' in such cases.

    Target Shape Handling:
        - Scalar (graph-level single target): Discretized to scalar class index
        - [num_targets] (graph-level multi-target): Select target_column, output [1]
        - [num_nodes] (node-level single target): Output shape [num_nodes]
        - [num_nodes, features] (node-level multi-target): Select target_column, output [num_nodes]
        - [num_edges] (edge-level single target): Output shape [num_edges]
        - [num_edges, features] (edge-level multi-target): Select target_column, output [num_edges]

    Attributes Stored:
        {attr}_original: Original continuous target values (for reference/inverse transform)
        {attr}_bin_edges: Computed bin edges used for discretization
        {attr}_num_classes: Number of discrete classes created
        {attr}_discretization_strategy: Strategy used for binning
        {attr}_is_node_level: Boolean indicating if targets are node-level
        {attr}_is_edge_level: Boolean indicating if targets are edge-level
        {attr}_target_level: String: 'graph', 'node', or 'edge'
        targets_discretized: Flag indicating transform was applied

    Example (graph_classification):
        >>> transform = DiscretizeTargets(n_bins=5, strategy="quantile")
        >>> data = Data(x=torch.ones(3, 16), y=torch.tensor([-100.5]))  # graph-level
        >>> result = transform(data)
        >>> # result.y is now integer class index (0, 1, 2, 3, or 4)
        >>> # result.y_num_classes = 5
        >>> # result.y_target_level = 'graph'

    Example (node_classification):
        >>> transform = DiscretizeTargets(n_bins=3, strategy="uniform")
        >>> data = Data(x=torch.ones(100, 16), y=torch.randn(100))  # 100 nodes
        >>> result = transform(data)
        >>> # result.y.shape = [100], dtype=torch.long
        >>> # result.y_target_level = 'node'

    Example (edge/link_prediction multi-class):
        >>> transform = DiscretizeTargets(n_bins=4, strategy="quantile", attrs=["edge_label"])
        >>> # Edge-level targets: one value per edge (e.g., bond strength)
        >>> data = Data(x=torch.ones(10, 16), edge_index=..., edge_label=torch.randn(50))
        >>> result = transform(data)
        >>> # result.edge_label.shape = [50], dtype=torch.long (classes 0,1,2,3)
        >>> # result.edge_label_target_level = 'edge'

    Config Usage:
        >>> # In config.yaml standard_transforms:
        >>> # - name: "DiscretizeTargets"
        >>> #   enabled: true
        >>> #   params:
        >>> #     n_bins: 5
        >>> #     strategy: "quantile"
        >>> #     target_column: 0
        >>> #     attrs: ["y"]  # or ["edge_label"] for edge classification

    Utility Methods:
        compute_class_weights(dataset, attr, method, device):
            Compute class weights for handling imbalanced classification.
            Methods: 'balanced' (sklearn-style), 'inverse', 'sqrt_inverse'
            Returns: torch.Tensor of shape [num_classes]

        get_num_classes(data_or_dataset, attr):
            Get number of classes from discretized data for model factory integration.
            Returns: int or None

    Class Weights Example:
        >>> # After applying transform to dataset:
        >>> weights = DiscretizeTargets.compute_class_weights(dataset, method='balanced')
        >>> loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    Model Integration Example:
        >>> # Get num_classes for model out_channels:
        >>> num_classes = DiscretizeTargets.get_num_classes(dataset[0])
        >>> model = factory.create_model(name='GCN', out_channels=num_classes)

    Note:
        For PyG classification, targets must be integer class indices starting from 0.
        The number of classes (n_bins) determines the model's output dimension.

    Integration:
        When used with classification task types:
        - Model factory should use data.{attr}_num_classes for out_channels
        - CrossEntropyLoss expects integer targets in range [0, num_classes-1]
        - Shape is preserved: [num_nodes] for node, [num_edges] for edge classification
        - Use compute_class_weights() for imbalanced data handling
    """

    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = "quantile",
        target_column: int = 0,
        attrs: list[str] | None = None,
        fitted_bin_edges: dict[str, torch.Tensor] | None = None,
        target_level: str = "auto",
    ):
        """
        Initialize discretization transform.

        Args:
            n_bins: Number of discrete bins/classes to create (must be >= 2).
                    This determines the model's output dimension for classification.
            strategy: Binning strategy. One of:
                - 'quantile': Equal number of samples per bin (recommended for skewed data)
                - 'uniform': Equal-width bins across value range
                - 'kmeans': Bins based on k-means clustering
            target_column: For multi-target data, which column to discretize.
                          Default: 0 (first target). Ignored for single-target data.
            attrs: List of attribute names to discretize.
                   Default: ["y"] for target values.
            fitted_bin_edges: Optional pre-computed bin edges dict {attr: edges_tensor}.
                             When provided, these edges are used instead of computing
                             new edges per sample. This ensures consistency across
                             train/val/test splits. Use fit() method to compute these.
            target_level: Explicit target level specification. One of:
                - 'auto': Infer from dimensions (default, backward compatible)
                - 'graph': Force graph-level treatment (for graph_classification)
                - 'node': Force node-level treatment (for node_classification)
                - 'edge': Force edge-level treatment (for link_prediction)
                When task_type is known, passing explicit target_level avoids
                ambiguity when y.shape[0] == num_nodes or num_edges == num_nodes.
        """
        super().__init__()

        if n_bins < 2:
            raise TransformConfigurationError(
                f"n_bins must be >= 2, got {n_bins}", transform_name="DiscretizeTargets"
            )

        valid_strategies = ["quantile", "uniform", "kmeans"]
        if strategy.lower() not in valid_strategies:
            raise TransformConfigurationError(
                f"strategy must be one of {valid_strategies}, got '{strategy}'",
                transform_name="DiscretizeTargets",
            )

        valid_target_levels = ["auto", "graph", "node", "edge"]
        if target_level.lower() not in valid_target_levels:
            raise TransformConfigurationError(
                f"target_level must be one of {valid_target_levels}, got '{target_level}'",
                transform_name="DiscretizeTargets",
            )

        self.n_bins = n_bins
        self.strategy = strategy.lower()
        self.target_column = target_column
        self.attrs = attrs if attrs is not None else ["y"]
        self.target_level = target_level.lower()

        # Store pre-fitted bin edges for consistent discretization
        self._fitted_bin_edges: dict[str, torch.Tensor] = fitted_bin_edges or {}
        self._is_fitted = bool(fitted_bin_edges)

    def fit(self, data_list, attrs: list[str] | None = None) -> "DiscretizeTargets":
        """
        Fit the discretizer by computing bin edges from a collection of data.

        This method collects all target values from the provided data and computes
        global bin edges that will be used consistently for all subsequent transform() calls.

        IMPORTANT: Call fit() on training data BEFORE applying transform() to ensure
        consistent discretization across train/val/test splits.

        Args:
            data_list: Iterable of Data objects (e.g., training dataset)
            attrs: Attributes to fit. Default: uses self.attrs

        Returns:
            self (fitted transform instance)

        Example:
            >>> transform = DiscretizeTargets(n_bins=10, strategy='quantile')
            >>> transform.fit(train_dataset)  # Compute bin edges from training data
            >>> train_data = [transform(d) for d in train_dataset]  # Apply with fitted edges
            >>> val_data = [transform(d) for d in val_dataset]  # Same edges for val
        """
        import torch

        fit_attrs = attrs if attrs is not None else self.attrs

        for attr in fit_attrs:
            # Collect all values for this attribute
            all_values = []
            for data in data_list:
                if hasattr(data, attr) and getattr(data, attr) is not None:
                    values = getattr(data, attr)

                    # Handle multi-target: select specific column
                    # ============================================================
                    # CRITICAL: Handle ALL multi-dimensional targets consistently
                    # to match transform() behavior. This prevents dimension
                    # inconsistency during fit/transform when some graphs have
                    # [N, F>1] and others have [N, 1].
                    # ============================================================

                    # Case 1: 2D+ targets (node/edge/graph level with features)
                    if values.dim() > 1:
                        last_dim_size = values.shape[-1]
                        if last_dim_size > 1:
                            # Multi-feature: select specified column
                            if self.target_column < last_dim_size:
                                values = values[..., self.target_column]
                                self._logger.debug(
                                    f"fit(): Multi-target (dim>1, features={last_dim_size}): "
                                    f"selected column {self.target_column}"
                                )
                        else:
                            # Single-feature [N, 1] or [1, 1]: squeeze for consistency
                            values = values.squeeze(-1)
                            self._logger.debug(
                                "fit(): Single-feature 2D target (shape[-1]=1): squeezed"
                            )

                    # Case 2: Graph-level 1D target (num_targets >= 1)
                    # Handles BOTH multi-target [T>1] AND single-target [1] cases
                    # For consistency with transform(), we extract a single value
                    elif values.dim() == 1 and values.size(0) >= 1:
                        first_dim = values.size(0)

                        # Determine target level: use explicit if provided, else infer
                        if self.target_level != "auto":
                            # Explicit target_level provided - use it directly
                            is_node_level = self.target_level == "node"
                            is_edge_level = self.target_level == "edge"
                        else:
                            # Auto-detect from dimensions (original heuristic)
                            # Get num_nodes to check if this matches node count
                            num_nodes = data.num_nodes if hasattr(data, "num_nodes") else None
                            if num_nodes is None and hasattr(data, "x") and data.x is not None:
                                num_nodes = data.x.size(0)

                            num_edges = data.num_edges if hasattr(data, "num_edges") else None
                            if (
                                num_edges is None
                                and hasattr(data, "edge_index")
                                and data.edge_index is not None
                            ):
                                num_edges = data.edge_index.size(1)

                            # Check if first dimension matches nodes or edges
                            is_node_level = num_nodes is not None and first_dim == num_nodes
                            is_edge_level = num_edges is not None and first_dim == num_edges

                        # If NOT node-level AND NOT edge-level, then it's graph-level target
                        if not is_node_level and not is_edge_level:
                            if first_dim > 1:
                                # Multi-target case: select specified column
                                if self.target_column < first_dim:
                                    values = values[self.target_column].unsqueeze(0)
                                    self._logger.debug(
                                        f"fit(): Graph-level 1D multi-target: selected target {self.target_column} of {first_dim}"
                                    )
                            else:
                                # Single-target case: y=[value], keep as-is (it will be flattened)
                                # No column selection needed since there's only one value
                                self._logger.debug(
                                    "fit(): Graph-level 1D single-target: using value as-is"
                                )

                    all_values.append(values.flatten())

            if not all_values:
                self._logger.warning(f"No values found for attribute '{attr}' during fit()")
                continue

            # Concatenate all values
            combined_values = torch.cat(all_values).float()

            # Skip if already integer
            if combined_values.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                self._logger.info(f"'{attr}' is already integer type, skipping fit")
                continue

            # Compute bin edges based on strategy
            if self.strategy == "quantile":
                bin_edges = self._compute_bin_edges_quantile(combined_values)
            elif self.strategy == "uniform":
                bin_edges = self._compute_bin_edges_uniform(combined_values)
            elif self.strategy == "kmeans":
                bin_edges = self._compute_bin_edges_kmeans(combined_values)
            else:
                bin_edges = self._compute_bin_edges_quantile(combined_values)

            # Store fitted bin edges
            self._fitted_bin_edges[attr] = bin_edges

            self._logger.info(
                f"Fitted bin edges for '{attr}' from {len(all_values)} samples "
                f"({len(combined_values)} total values) using '{self.strategy}' strategy"
            )

        self._is_fitted = True
        return self

    def is_fitted(self) -> bool:
        """Check if transform has been fitted with bin edges."""
        return self._is_fitted

    def get_fitted_bin_edges(self, attr: str = "y") -> torch.Tensor | None:
        """Get the fitted bin edges for an attribute."""
        return self._fitted_bin_edges.get(attr)

    def _compute_bin_edges_quantile(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute bin edges using quantile-based strategy.

        Equal number of samples in each bin (handles skewed distributions).
        """
        # Compute quantile boundaries
        quantiles = torch.linspace(0, 1, self.n_bins + 1, device=values.device)
        bin_edges = torch.quantile(values.float(), quantiles)

        # Handle duplicate edges (can occur with many identical values)
        # Ensure edges are strictly increasing
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i - 1]:
                bin_edges[i] = bin_edges[i - 1] + 1e-6

        return bin_edges

    def _compute_bin_edges_uniform(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute bin edges using uniform-width strategy.

        Equal-width bins across the value range.
        """
        val_min = values.min()
        val_max = values.max()

        # Handle constant values
        if val_max - val_min < 1e-8:
            # Create artificial bins around the constant value
            bin_edges = torch.linspace(
                val_min - 1, val_max + 1, self.n_bins + 1, device=values.device
            )
        else:
            bin_edges = torch.linspace(val_min, val_max, self.n_bins + 1, device=values.device)

        return bin_edges

    def _compute_bin_edges_kmeans(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute bin edges using k-means clustering.

        Bins based on k-means cluster boundaries.
        """
        # Simple 1D k-means implementation
        values_np = values.cpu().numpy().flatten()

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=self.n_bins, random_state=42, n_init=10)
            kmeans.fit(values_np.reshape(-1, 1))

            # Get sorted cluster centers
            centers = np.sort(kmeans.cluster_centers_.flatten())

            # Compute bin edges as midpoints between centers
            bin_edges = [values_np.min()]
            for i in range(len(centers) - 1):
                bin_edges.append((centers[i] + centers[i + 1]) / 2)
            bin_edges.append(values_np.max())

            return torch.tensor(bin_edges, dtype=values.dtype, device=values.device)

        except ImportError:
            self._logger.warning(
                "sklearn not available for kmeans strategy, falling back to quantile"
            )
            return self._compute_bin_edges_quantile(values)

    def _assign_bins(self, values: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
        """
        Assign values to bins based on bin edges.

        Returns integer class indices in range [0, n_bins-1].
        """
        # Use bucketize for efficient binning
        # right=False means bins are [edge[i], edge[i+1])
        bin_indices = torch.bucketize(values, bin_edges[1:-1], right=False)

        # Clamp to valid range [0, n_bins-1]
        bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)

        return bin_indices.long()

    def transform(self, data: Data) -> Data:
        """
        Apply discretization to convert continuous targets to class indices.

        Handles both:
        - Graph-level targets: scalar or shape [1] or [num_targets] -> output shape preserved
        - Node-level targets: shape [num_nodes] or [num_nodes, num_features] -> output shape [num_nodes]

        For multi-target data (last dim > 1), only the specified target_column is discretized.
        """
        # Clone to avoid modifying original
        data = data.clone()

        for attr in self.attrs:
            if not hasattr(data, attr) or getattr(data, attr) is None:
                self._logger.debug(f"Attribute '{attr}' not found, skipping discretization")
                continue

            values = getattr(data, attr)

            # Store original values
            setattr(data, f"{attr}_original", values.clone())

            # Track original shape for proper restoration
            original_shape = values.shape
            original_dim = values.dim()

            # Handle scalar values (single target per graph)
            if original_dim == 0:
                values = values.unsqueeze(0)
                was_scalar = True
            else:
                was_scalar = False

            # Determine target level: graph-level, node-level, or edge-level
            # Get num_nodes
            num_nodes = (
                data.num_nodes
                if hasattr(data, "num_nodes")
                else (data.x.size(0) if hasattr(data, "x") and data.x is not None else None)
            )

            # Get num_edges
            num_edges = (
                data.num_edges
                if hasattr(data, "num_edges")
                else (
                    data.edge_index.size(1)
                    if hasattr(data, "edge_index") and data.edge_index is not None
                    else None
                )
            )

            # Determine target level based on attribute name and dimensions
            # Priority order:
            #   1. Attribute name (edge_* attrs are always edge-level)
            #   2. Dimension matching (check both node and edge dimensions)
            #   3. Default to graph-level
            #
            # IMPORTANT: Attribute name check MUST come FIRST because when
            # num_nodes == num_edges, dimension checks alone cannot disambiguate.
            # Edge-related attribute names (edge_label, edge_y, etc.) definitively
            # indicate edge-level targets regardless of dimension matching.

            target_level = "graph"  # default
            if not was_scalar:
                first_dim = values.size(0)

                # Determine target level: use explicit if provided, else infer
                if self.target_level != "auto":
                    # Explicit target_level provided - use it directly
                    target_level = self.target_level
                else:
                    # Auto-detect using priority order:
                    # 1. FIRST: Check attribute name for edge-related attributes
                    #    This takes precedence because it's unambiguous semantic information
                    if attr in ["edge_label", "edge_y", "edge_value", "edge_target", "edge_attr"]:
                        target_level = "edge"
                    # 2. THEN: Check dimension matching
                    #    Node check comes before edge because 'y' is more commonly node-level
                    elif num_nodes is not None and first_dim == num_nodes:
                        # Could be node-level OR edge-level if num_nodes == num_edges
                        # But since we already checked edge attr names above,
                        # reaching here with 'y' means it's node-level
                        target_level = "node"
                    elif num_edges is not None and first_dim == num_edges:
                        target_level = "edge"
                    # 3. Default: graph-level (already set above)

            is_node_level = target_level == "node"
            is_edge_level = target_level == "edge"

            # Handle multi-target: select specific column
            # ================================================================
            # CRITICAL: Handle ALL multi-dimensional targets consistently to
            # prevent dimension mismatch during PyG batching.
            #
            # The issue: If some graphs have y.shape=[N, 3] (3 features) and
            # others have y.shape=[N, 1] (1 feature), after column selection:
            # - [N, 3] -> select column -> [N] (1D)
            # - [N, 1] was SKIPPED -> stayed [N, 1] (2D) -> dimension mismatch!
            #
            # Solution: Handle ALL 2D targets (including shape[-1] == 1) to
            # ensure consistent 1D output for node/edge level targets.
            # ================================================================
            multi_target_handled = False

            if values.dim() > 1:
                # 2D+ targets: [num_nodes/edges, num_features] or [1, num_targets]
                last_dim_size = values.shape[-1]

                if last_dim_size > 1:
                    # Multi-feature case: select specified column
                    if self.target_column >= last_dim_size:
                        raise TransformExecutionError(
                            f"target_column {self.target_column} out of range for "
                            f"target tensor with {last_dim_size} columns",
                            transform_name="DiscretizeTargets",
                        )
                    # Select the specified column, preserving other dimensions
                    values = values[..., self.target_column]
                    multi_target_handled = True
                    self._logger.debug(
                        f"Multi-target detected (dim>1, features={last_dim_size}): "
                        f"discretizing column {self.target_column}"
                    )
                else:
                    # Single-feature case: shape is [N, 1] or [1, 1]
                    # Squeeze out the last dimension for consistency
                    # This ensures [N, 1] -> [N] to match graphs that have [N, F>1] -> [N]
                    values = values.squeeze(-1)
                    multi_target_handled = True
                    self._logger.debug(
                        f"Single-feature 2D target detected (shape[-1]=1): "
                        f"squeezed to {list(values.shape)} for PyG batching consistency"
                    )
            elif target_level == "graph" and not was_scalar and values.dim() == 1:
                # Graph-level 1D target handling: shape [num_targets] where num_targets >= 1
                # This handles BOTH:
                # - Multi-target: y=[energy, gap, dipole, ...] with num_targets > 1
                # - Single-target: y=[energy] with num_targets == 1
                #
                # CRITICAL: For PyG batching consistency, ALL graph-level targets must output
                # as scalars. When PyG batches graphs with scalar y, it concatenates them into
                # [batch_size] which is exactly what CrossEntropyLoss expects: target shape [batch]
                #
                # This fixes the dimension mismatch error "Tensors must have same number of
                # dimensions: got 1 and 2" that occurs when some molecules have y=[] (scalar)
                # and others have y=[1] (1D tensor with single element).

                num_targets = values.size(0)

                if num_targets > 1:
                    # Multi-target case: select specified column
                    if self.target_column >= num_targets:
                        raise TransformExecutionError(
                            f"target_column {self.target_column} out of range for "
                            f"graph-level target tensor with {num_targets} targets",
                            transform_name="DiscretizeTargets",
                        )
                    values = values[
                        self.target_column : self.target_column + 1
                    ]  # [num_targets] -> [1]
                    self._logger.debug(
                        f"Graph-level multi-target detected (1D): discretizing target {self.target_column} "
                        f"of {num_targets} targets -> output shape: [1]"
                    )
                else:
                    # Single-target case: y=[value] -> keep as [1] for PyG batching
                    # PyG docs: "graph-level targets of shape [1, *]"
                    # Already shape [1], no extraction needed
                    self._logger.debug(
                        "Graph-level single-target detected (1D): keeping shape [1] for "
                        "PyG batching consistency"
                    )

                multi_target_handled = True

            # Store shape before flattening for restoration
            pre_flatten_shape = values.shape

            # Flatten for processing (handles both 1D and multi-D)
            values_flat = values.flatten()

            # Check for valid values
            if not torch.all(torch.isfinite(values_flat)):
                self._logger.warning(f"Non-finite values in '{attr}', skipping discretization")
                continue

            # Check if already integer (no discretization needed)
            if values_flat.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                self._logger.debug(
                    f"'{attr}' already has integer type, checking if valid class indices"
                )
                # Verify valid class indices
                if values_flat.min() >= 0 and values_flat.max() < self.n_bins:
                    setattr(data, f"{attr}_num_classes", self.n_bins)
                    setattr(data, f"{attr}_discretized", True)
                    setattr(data, f"{attr}_is_node_level", is_node_level)
                    setattr(data, f"{attr}_is_edge_level", is_edge_level)
                    setattr(data, f"{attr}_target_level", target_level)
                    continue

            # Ensure float type for binning computation
            values_float = values_flat.float()

            # Use pre-fitted bin edges if available (CRITICAL for train/val consistency)
            # Otherwise compute fresh edges (for single-sample or unfitted usage)
            if attr in self._fitted_bin_edges:
                bin_edges = self._fitted_bin_edges[attr]
                # Move to same device as values if needed
                if bin_edges.device != values_float.device:
                    bin_edges = bin_edges.to(values_float.device)
            else:
                # Compute bin edges based on strategy (per-sample, less ideal)
                if self.strategy == "quantile":
                    bin_edges = self._compute_bin_edges_quantile(values_float)
                elif self.strategy == "uniform":
                    bin_edges = self._compute_bin_edges_uniform(values_float)
                elif self.strategy == "kmeans":
                    bin_edges = self._compute_bin_edges_kmeans(values_float)
                else:
                    # Should not reach here due to __init__ validation
                    bin_edges = self._compute_bin_edges_quantile(values_float)

            # Assign values to bins
            class_indices = self._assign_bins(values_float, bin_edges)

            # CRITICAL: Restore proper shape for correct PyG batching
            # ================================================================
            # Shape restoration rules (based on PyG documentation):
            # - Original scalar: squeeze back to scalar [] (rare case)
            # - 1D graph-level target [T] (T>=1): output [1] for PyG batching
            #   PyG docs: "graph-level targets of shape [1, *]"
            #   PyG concatenates: [1] + [1] + ... = [batch_size]
            # - 2D graph-level multi-target [1, T]: output [1] (preserve first dim)
            # - Node-level [num_nodes]: reshape to [num_nodes]
            # - Edge-level [num_edges]: reshape to [num_edges]
            # - Node-level multi-target [num_nodes, T]: reshape to [num_nodes]
            # ================================================================

            # Determine if original was 1D graph-level target (needs [1] output for PyG batching)
            # This now covers BOTH multi-target [T>1] AND single-target [1] cases
            was_1d_graph_target = (
                len(original_shape) == 1
                and original_shape[0] >= 1
                and target_level == "graph"
                and multi_target_handled
            )

            if was_scalar:
                # Original was scalar, restore to scalar
                class_indices = (
                    class_indices.squeeze(0) if class_indices.dim() > 0 else class_indices
                )
            elif was_1d_graph_target:
                # 1D Graph-level target [T] -> [1]
                # CRITICAL: Output must be [1] (1D with 1 element), NOT scalar []
                # PyG's collation calls value.size(0) which fails on scalars:
                # "Dimension specified as 0 but tensor has no dimensions"
                if class_indices.dim() == 0:
                    class_indices = class_indices.unsqueeze(0)  # scalar -> [1]
                elif class_indices.numel() == 1 and class_indices.dim() > 1:
                    class_indices = class_indices.view(1)  # [1,1,...] -> [1]
                # else: already shape [1], keep as-is
            elif len(pre_flatten_shape) > 0:
                # All other cases: restore to pre-flatten shape
                # This handles:
                # - 2D graph-level [1, T] -> [1] after column select -> restore to [1]
                # - Node-level [N] -> [N]
                # - Node-level multi-target [N, T] -> [N] after column select -> restore to [N]
                # - Edge-level [E] -> [E]
                class_indices = class_indices.view(pre_flatten_shape)

            # Store discretized values and metadata
            setattr(data, attr, class_indices)
            setattr(data, f"{attr}_bin_edges", bin_edges)
            setattr(data, f"{attr}_num_classes", self.n_bins)
            setattr(data, f"{attr}_discretization_strategy", self.strategy)
            setattr(data, f"{attr}_is_node_level", is_node_level)
            setattr(data, f"{attr}_is_edge_level", is_edge_level)
            setattr(data, f"{attr}_target_level", target_level)

            # Log distribution info
            unique_classes, counts = torch.unique(class_indices, return_counts=True)
            class_dist = {int(c): int(cnt) for c, cnt in zip(unique_classes, counts, strict=False)}
            self._logger.debug(
                f"Discretized {target_level}-level '{attr}' (shape {list(class_indices.shape)}) "
                f"into {self.n_bins} classes using '{self.strategy}' strategy. "
                f"Class distribution: {class_dist}"
            )

        # Mark data as discretized
        data.targets_discretized = True

        return data

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        """Get transform metadata for registry integration."""
        return TransformMetadata(
            name="DiscretizeTargets",
            version="1.1.0",
            author="milia Team",
            category="classification",
            description=(
                "Discretize continuous target values into class labels for classification. "
                "Supports graph_classification, node_classification, and edge/link classification. "
                "Strategies: quantile, uniform, kmeans. Automatically detects target level."
            ),
            validated_datasets=["milia_DFT", "milia_DMC", "QM9"],
            required_graph_attributes=[],  # Flexible - works with y, edge_label, etc.
        )

    @classmethod
    def get_parameter_constraints(cls) -> dict[str, Any]:
        """Define parameter constraints for validation and introspection."""
        return {
            "n_bins": {
                "type": int,
                "range": (2, 1000),
                "default": 5,
                "description": "Number of discrete bins/classes to create",
            },
            "strategy": {
                "type": str,
                "choices": ["quantile", "uniform", "kmeans"],
                "default": "quantile",
                "description": "Binning strategy: quantile (equal samples), uniform (equal width), or kmeans",
            },
            "target_column": {
                "type": int,
                "range": (0, 100),
                "default": 0,
                "description": "For multi-target data, which column to discretize",
            },
            "attrs": {
                "type": list,
                "default": ["y"],
                "description": "List of attribute names to discretize",
            },
        }

    @staticmethod
    def compute_class_weights(
        dataset, attr: str = "y", method: str = "balanced", device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Compute class weights from discretized dataset for handling class imbalance.

        This is a utility method to generate class weights that can be passed to
        CrossEntropyLoss(weight=...) for handling imbalanced classification.

        Supported Methods:
            - 'balanced': Inverse frequency weighting (n_samples / (n_classes * class_count))
            - 'inverse': Simple inverse frequency (1 / class_count)
            - 'sqrt_inverse': Square root of inverse frequency (1 / sqrt(class_count))

        Args:
            dataset: PyG dataset or list of Data objects with discretized targets
            attr: Attribute name containing discretized class indices (default: 'y')
            method: Weighting method ('balanced', 'inverse', 'sqrt_inverse')
            device: Device to place weights tensor on (default: None = CPU)

        Returns:
            torch.Tensor of shape [num_classes] with class weights

        Example:
            >>> # After applying DiscretizeTargets transform:
            >>> weights = DiscretizeTargets.compute_class_weights(dataset, method='balanced')
            >>> loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        Note:
            Call this AFTER applying DiscretizeTargets transform to the dataset.
            The method automatically detects num_classes from data.{attr}_num_classes
            or computes from unique values.
        """
        # Collect all class indices
        all_labels = []
        num_classes = None

        for data in dataset:
            if hasattr(data, attr) and getattr(data, attr) is not None:
                labels = getattr(data, attr)
                if labels.dim() == 0:
                    all_labels.append(labels.unsqueeze(0))
                else:
                    all_labels.append(labels.flatten())

                # Get num_classes from metadata if available
                if num_classes is None and hasattr(data, f"{attr}_num_classes"):
                    num_classes = getattr(data, f"{attr}_num_classes")

        if not all_labels:
            raise ValueError(f"No '{attr}' attribute found in dataset")

        # Concatenate all labels
        all_labels = torch.cat(all_labels, dim=0)

        # Ensure integer type
        if all_labels.dtype not in [torch.int, torch.int32, torch.int64, torch.long]:
            all_labels = all_labels.long()

        # Determine num_classes if not from metadata
        if num_classes is None:
            num_classes = int(all_labels.max().item()) + 1

        # Count occurrences of each class
        class_counts = torch.bincount(all_labels, minlength=num_classes).float()

        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1.0)

        # Compute weights based on method
        n_samples = all_labels.numel()

        if method == "balanced":
            # sklearn-style balanced: n_samples / (n_classes * np.bincount(y))
            weights = n_samples / (num_classes * class_counts)
        elif method == "inverse":
            # Simple inverse frequency
            weights = 1.0 / class_counts
        elif method == "sqrt_inverse":
            # Square root of inverse (less aggressive than pure inverse)
            weights = 1.0 / torch.sqrt(class_counts)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: 'balanced', 'inverse', 'sqrt_inverse'"
            )

        # Normalize weights to have mean of 1.0 (optional, for stability)
        weights = weights / weights.mean() * 1.0

        if device is not None:
            weights = weights.to(device)

        return weights

    @staticmethod
    def get_num_classes(data_or_dataset, attr: str = "y") -> int | None:
        """
        Get the number of classes from discretized data for model factory integration.

        This utility retrieves num_classes to set model out_channels for classification.

        Args:
            data_or_dataset: Single Data object or dataset/list of Data objects
            attr: Attribute name (default: 'y')

        Returns:
            Number of classes (int) or None if not discretized

        Example:
            >>> # For model factory integration:
            >>> num_classes = DiscretizeTargets.get_num_classes(dataset[0])
            >>> if num_classes:
            >>>     model = create_model(out_channels=num_classes)
        """
        # Handle single Data object
        if hasattr(data_or_dataset, f"{attr}_num_classes"):
            return getattr(data_or_dataset, f"{attr}_num_classes")

        # Handle dataset/list - get from first element
        if hasattr(data_or_dataset, "__getitem__"):
            try:
                first = data_or_dataset[0]
                if hasattr(first, f"{attr}_num_classes"):
                    return getattr(first, f"{attr}_num_classes")
            except (IndexError, TypeError):
                pass

        return None


# =============================================================================
# MODULE-LEVEL EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "TransformMetadata",
    "CustomTransformBase",
    "MolecularTransformBase",
    "QuantumTransformBase",
    # Target transforms
    "StandardizeTargets",
    "NormalizeTargets",
    "DiscretizeTargets",
    # Example transforms
    "NormalizeVibrationalModes",
    "FilterByDMCUncertainty",
    "ScaleMullikenCharges",
    # Exceptions (if available)
    "TransformValidationError",
    "TransformExecutionError",
    "TransformConfigurationError",
]


# Module initialization
if __name__ == "__main__":
    # Simple test when run as script
    print("milia Custom Transforms Module v1.0")
    print(f"Graph transforms available: {GRAPH_TRANSFORMS_AVAILABLE}")
    print(f"Config containers available: {CONFIG_CONTAINERS_AVAILABLE}")
    print(f"Exceptions module available: {EXCEPTIONS_AVAILABLE}")
    print("\nAvailable base classes:")
    print("  - CustomTransformBase")
    print("  - MolecularTransformBase")
    print("  - QuantumTransformBase")
    print("\nExample transforms:")
    print("  - NormalizeVibrationalModes")
    print("  - FilterByDMCUncertainty")
    print("  - ScaleMullikenCharges")

if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data

    print("Testing custom transforms for in-place modification bugs...\n")

    # Test 1: NormalizeVibrationalModes
    print("Test 1: NormalizeVibrationalModes")
    original_vibmodes = torch.randn(5, 3, 3)
    data1 = Data(x=torch.ones(3, 1), vibmodes=original_vibmodes.clone(), num_nodes=3)
    original_copy = data1.vibmodes.clone()

    transform1 = NormalizeVibrationalModes(normalize_per_mode=True)
    result1 = transform1(data1)

    # Check original was NOT modified
    assert torch.allclose(data1.vibmodes, original_copy), "❌ FAILED: Original vibmodes modified!"
    print("PASSED: Original vibmodes preserved")

    # Test 2: FilterByDMCUncertainty
    print("\nTest 2: FilterByDMCUncertainty")
    data2 = Data(x=torch.ones(3, 1), dmc_uncertainty=torch.tensor(0.15), num_nodes=3)

    transform2 = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)
    result2 = transform2(data2)

    # Check original does NOT have new attributes
    assert not hasattr(data2, "is_high_uncertainty"), "❌ FAILED: Original data modified!"
    assert hasattr(result2, "is_high_uncertainty"), "❌ FAILED: Result missing attribute!"
    print("PASSED: Original data preserved, result has new attributes")

    # Test 3: ScaleMullikenCharges
    print("\nTest 3: ScaleMullikenCharges")
    original_charges = torch.tensor([0.5, -0.3, -0.2])
    data3 = Data(x=torch.ones(3, 1), charges=original_charges.clone(), num_nodes=3)
    original_charges_copy = data3.charges.clone()

    transform3 = ScaleMullikenCharges(scale_factor=2.0, center=True)
    result3 = transform3(data3)

    # Check original was NOT modified
    assert torch.allclose(data3.charges, original_charges_copy), (
        "❌ FAILED: Original charges modified!"
    )
    assert not torch.allclose(result3.charges, original_charges_copy), (
        "❌ FAILED: Result not transformed!"
    )
    print("PASSED: Original charges preserved, result transformed")

    print("\n🎉 All tests passed!")
