"""
Descriptor Registry System

Central registry for descriptor discovery, registration, and management.
Supports both built-in RDKit descriptors and custom plugin descriptors.

Features:
- Auto-discovery of RDKit descriptors via introspection
- Plugin descriptor registration
- Thread-safe singleton pattern
- Metadata management
- Search and filtering capabilities

Pydantic V2 Migration (Phase 28):
    - Migrated DescriptorRegistration from @dataclass to Pydantic BaseModel (mutable)
    - Uses ConfigDict(arbitrary_types_allowed=True) for Callable and DescriptorMetadata types
    - Added to_dict() method wrapping model_dump() for backward compatibility
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: milia Team
Version: 1.1.0
"""

import datetime
import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D

from milia_pipeline.exceptions import (
    DescriptorValidationError,
)

from .descriptor_categories import (
    DescriptorCategory,
    DescriptorMetadata,
    get_all_descriptor_names,
    get_descriptor_metadata,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DESCRIPTOR REGISTRATION DATACLASS
# =============================================================================


class DescriptorRegistration(BaseModel):
    """
    Container for descriptor registration information.

    Pydantic V2 Migration (Phase 28):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses ConfigDict(arbitrary_types_allowed=True) for Callable and DescriptorMetadata types
        - Added to_dict() method wrapping model_dump() for backward compatibility
        - NON-BREAKING: Same constructor API and attribute access preserved

    Attributes:
        name: Descriptor name
        function: Callable descriptor function
        metadata: DescriptorMetadata object
        is_builtin: True if RDKit descriptor, False if plugin
        plugin_name: Name of plugin (if from plugin)
        registered_at: Timestamp of registration
    """

    # Allow arbitrary types for Callable and DescriptorMetadata
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    function: Callable
    metadata: DescriptorMetadata
    is_builtin: bool = True
    plugin_name: str | None = None
    registered_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().

        Note:
            The 'function' field will be included but may not be JSON-serializable.
            Use exclude={'function'} if JSON serialization is needed.
        """
        return self.model_dump()


# =============================================================================
# DESCRIPTOR REGISTRY (SINGLETON)
# =============================================================================


class DescriptorRegistry:
    """
    Thread-safe singleton registry for molecular descriptors.

    Manages both built-in RDKit descriptors and custom plugin descriptors.
    Provides discovery, registration, and search functionality.

    Usage:
        >>> registry = DescriptorRegistry()
        >>> registry.auto_discover_rdkit_descriptors()
        >>> desc_func = registry.get_descriptor("MolWt")
        >>> all_names = registry.list_available_descriptors()
    """

    _instances: dict[type, "DescriptorRegistry"] = {}
    _class_lock: threading.Lock = threading.Lock()

    def __new__(cls):
        """Implement singleton pattern with thread safety"""
        if cls not in cls._instances:
            with cls._class_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    @classmethod
    def get_instance(cls) -> "DescriptorRegistry":
        """
        Get singleton instance of the registry.

        Thread-safe factory method that returns the singleton instance.
        This is the recommended way to access the registry.

        Returns:
            DescriptorRegistry singleton instance

        Example:
            >>> registry = DescriptorRegistry.get_instance()
            >>> registry.auto_discover_rdkit_descriptors()
        """
        return cls()

    def __init__(self):
        """Initialize registry (only once due to singleton)"""
        if not hasattr(self, "_initialized"):
            self._descriptors: dict[str, DescriptorRegistration] = {}
            self._by_category: dict[DescriptorCategory, set[str]] = defaultdict(set)
            self._plugin_descriptors: dict[str, str] = {}  # name -> plugin_name
            self._lock = threading.RLock()  # Use RLock for reentrant locking

            # Initialize tracking for discovery process
            self._failed_descriptors = []
            self._mol_method_candidates = []

            self._initialized = True
            logger.info("DescriptorRegistry initialized")

            # Auto-discover RDKit descriptors
            self.auto_discover_rdkit_descriptors()

            # Register Mol method descriptors
            self.register_mol_method_descriptors()

            # Log comprehensive availability summary
            self.log_availability_summary()

    # =========================================================================
    # AUTO-DISCOVERY
    # =========================================================================

    def auto_discover_rdkit_descriptors(self) -> int:
        """
        Auto-discover and register all available RDKit descriptors.

        Gracefully handles:
        - Missing descriptors in current RDKit version
        - Case mismatches (e.g., FractionCsp3 vs FractionCSP3)
        - Mol method descriptors (e.g., GetNumHeavyAtoms)
        - Module location differences

        Returns:
            Number of descriptors successfully registered
        """
        logger.info("Starting RDKit descriptor auto-discovery...")

        discovered_count = 0
        failed_descriptors = []
        mol_method_candidates = []
        case_mismatch_candidates = []

        # Get all known descriptor names from categories
        known_descriptor_names = get_all_descriptor_names()

        # Discover from Descriptors module (2D) and Descriptors3D
        for name in known_descriptor_names:
            metadata = get_descriptor_metadata(name)
            if not metadata:
                continue

            try:
                func = None
                found_location = None

                # Try to find descriptor in specified module
                if metadata.rdkit_module == "Descriptors3D":
                    if hasattr(Descriptors3D, name):
                        func = getattr(Descriptors3D, name)
                        found_location = "Descriptors3D"
                elif metadata.rdkit_module == "Mol":
                    # This is a Mol method - handle separately
                    mol_method_candidates.append(name)
                    continue
                else:
                    # Try Descriptors module
                    if hasattr(Descriptors, name):
                        func = getattr(Descriptors, name)
                        found_location = "Descriptors"

                # If not found, try case-insensitive search
                if func is None:
                    func, found_location = self._try_case_insensitive_search(
                        name, Descriptors, Descriptors3D
                    )
                    if func is not None:
                        case_mismatch_candidates.append((name, found_location))

                # If found, verify and register
                if func is not None:
                    if not callable(func):
                        logger.debug(f"{name} found in {found_location} but is not callable")
                        failed_descriptors.append((name, "not_callable"))
                        continue

                    # Register the descriptor
                    self._register_internal(
                        name=name, function=func, metadata=metadata, is_builtin=True
                    )
                    discovered_count += 1
                    logger.debug(f"✓ Registered: {name} from {found_location}")
                else:
                    failed_descriptors.append((name, "not_found"))

            except Exception as e:
                logger.debug(f"Failed to discover descriptor {name}: {e}")
                failed_descriptors.append((name, f"error: {str(e)}"))

        # Report results
        logger.info(f"Auto-discovery complete: {discovered_count} descriptors registered")

        if case_mismatch_candidates:
            logger.info(
                f"  → {len(case_mismatch_candidates)} descriptors found with case differences:"
            )
            for name, location in case_mismatch_candidates[:5]:
                logger.info(f"      {name} (found in {location})")
            if len(case_mismatch_candidates) > 5:
                logger.info(f"      ... and {len(case_mismatch_candidates) - 5} more")

        if mol_method_candidates:
            logger.info(f"  → {len(mol_method_candidates)} descriptors require Mol method wrappers")
            logger.debug(f"      Mol methods: {mol_method_candidates}")

        if failed_descriptors:
            logger.info(
                f"  → {len(failed_descriptors)} descriptors not available in current RDKit version"
            )
            logger.debug(
                f"      Missing descriptors: {[name for name, _ in failed_descriptors[:10]]}"
            )

        # Store failed descriptors for reporting
        self._failed_descriptors = failed_descriptors
        self._mol_method_candidates = mol_method_candidates

        return discovered_count

    def _try_case_insensitive_search(
        self, name: str, *modules
    ) -> tuple[Callable | None, str | None]:
        """
        Try to find descriptor with case-insensitive search.

        Args:
            name: Descriptor name to search for
            *modules: RDKit modules to search in

        Returns:
            Tuple of (function, module_name) if found, else (None, None)
        """
        name_lower = name.lower()

        for module in modules:
            for attr_name in dir(module):
                if attr_name.lower() == name_lower and not attr_name.startswith("_"):
                    obj = getattr(module, attr_name)
                    if callable(obj):
                        module_name = module.__name__.split(".")[-1]
                        logger.debug(
                            f"Found {name} as {attr_name} in {module_name} (case mismatch)"
                        )
                        return obj, attr_name

        return None, None

    def register_mol_method_descriptors(self) -> int:
        """
        Register descriptors that are accessed via Mol methods.

        Some molecular properties (e.g., NumHeavyAtoms, NumAtoms) are accessed
        via Mol object methods rather than Descriptors module functions.
        This method creates wrapper functions for them.

        Returns:
            Number of Mol method descriptors successfully registered
        """
        if not self._mol_method_candidates:
            logger.debug("No Mol method descriptors to register")
            return 0

        logger.info(f"Registering {len(self._mol_method_candidates)} Mol method descriptors...")

        registered_count = 0
        failed_count = 0

        for desc_name in self._mol_method_candidates:
            try:
                # Try standard naming convention: NumHeavyAtoms -> GetNumHeavyAtoms
                method_name = f"Get{desc_name}"

                # Test if method exists
                test_mol = Chem.MolFromSmiles("C")
                if not hasattr(test_mol, method_name):
                    logger.debug(f"Mol method {method_name} not found for {desc_name}")
                    failed_count += 1
                    continue

                # Create wrapper function
                def make_wrapper(method_name):
                    def wrapper(mol):
                        if mol is None:
                            return None
                        try:
                            return getattr(mol, method_name)()
                        except Exception as e:
                            logger.debug(f"Error calling {method_name}: {e}")
                            return None

                    return wrapper

                wrapper_func = make_wrapper(method_name)

                # Get metadata
                metadata = get_descriptor_metadata(desc_name)
                if metadata is None:
                    logger.debug(f"No metadata for {desc_name}, skipping")
                    failed_count += 1
                    continue

                # Register the wrapper
                self._register_internal(
                    name=desc_name, function=wrapper_func, metadata=metadata, is_builtin=True
                )
                registered_count += 1
                logger.debug(f"✓ Registered Mol method: {desc_name} -> {method_name}()")

            except Exception as e:
                logger.debug(f"Failed to register Mol method descriptor {desc_name}: {e}")
                failed_count += 1

        logger.info(
            f"Mol method registration complete: {registered_count} registered, {failed_count} failed"
        )
        return registered_count

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register_descriptor(
        self,
        name: str,
        function: Callable,
        metadata: DescriptorMetadata | None = None,
        is_builtin: bool = False,
        plugin_name: str | None = None,
    ) -> None:
        """
        Register a descriptor (typically from a plugin).

        Args:
            name: Descriptor name (must be unique)
            function: Callable descriptor function(mol) -> float
            metadata: Optional DescriptorMetadata object
            is_builtin: True if RDKit descriptor
            plugin_name: Name of plugin providing descriptor

        Raises:
            DescriptorValidationError: If descriptor invalid or name conflict
        """
        # Validate inputs
        if not name:
            raise DescriptorValidationError(
                "Descriptor name cannot be empty", validation_type="name_check"
            )

        if not callable(function):
            raise DescriptorValidationError(
                f"Descriptor function must be callable, got {type(function)}",
                descriptor_name=name,
                validation_type="function_check",
            )

        # Check for conflicts
        if self.has_descriptor(name):
            existing = self._descriptors[name]
            if existing.is_builtin and not is_builtin:
                raise DescriptorValidationError(
                    f"Cannot override built-in descriptor '{name}' with plugin descriptor",
                    descriptor_name=name,
                    validation_type="conflict_check",
                )
            logger.warning(f"Overriding existing descriptor: {name}")

        # Create metadata if not provided
        if metadata is None:
            metadata = DescriptorMetadata(
                name=name,
                category=DescriptorCategory.CONSTITUTIONAL,  # Default
                description=f"Custom descriptor from plugin: {plugin_name or 'unknown'}",
            )

        # Register the descriptor
        self._register_internal(
            name=name,
            function=function,
            metadata=metadata,
            is_builtin=is_builtin,
            plugin_name=plugin_name,
        )

        logger.info(f"Registered descriptor: {name} (plugin={plugin_name}, builtin={is_builtin})")

    def _register_internal(
        self,
        name: str,
        function: Callable,
        metadata: DescriptorMetadata,
        is_builtin: bool,
        plugin_name: str | None = None,
    ) -> None:
        """
        Internal registration method (no validation).

        Args:
            name: Descriptor name
            function: Callable descriptor function
            metadata: DescriptorMetadata object
            is_builtin: True if RDKit descriptor
            plugin_name: Name of plugin
        """
        with self._lock:
            registration = DescriptorRegistration(
                name=name,
                function=function,
                metadata=metadata,
                is_builtin=is_builtin,
                plugin_name=plugin_name,
                registered_at=datetime.datetime.now().isoformat(),
            )

            self._descriptors[name] = registration
            self._by_category[metadata.category].add(name)

            if plugin_name and not is_builtin:
                self._plugin_descriptors[name] = plugin_name

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def get_descriptor(self, name: str) -> Callable | None:
        """
        Get descriptor function by name.

        Args:
            name: Descriptor name

        Returns:
            Descriptor function or None if not found
        """
        with self._lock:
            registration = self._descriptors.get(name)
            return registration.function if registration else None

    def get_descriptor_registration(self, name: str) -> DescriptorRegistration | None:
        """
        Get full registration information for a descriptor.

        Args:
            name: Descriptor name

        Returns:
            DescriptorRegistration object or None
        """
        with self._lock:
            return self._descriptors.get(name)

    def has_descriptor(self, name: str) -> bool:
        """
        Check if descriptor is registered.

        Args:
            name: Descriptor name

        Returns:
            True if descriptor is registered
        """
        with self._lock:
            return name in self._descriptors

    # =========================================================================
    # LISTING AND FILTERING
    # =========================================================================

    def list_available_descriptors(
        self,
        category: DescriptorCategory | None = None,
        include_plugins: bool = True,
        include_builtins: bool = True,
    ) -> list[str]:
        """
        List available descriptor names.

        Args:
            category: Optional category filter
            include_plugins: Include plugin descriptors
            include_builtins: Include built-in RDKit descriptors

        Returns:
            List of descriptor names matching criteria
        """
        with self._lock:
            if category:
                names = list(self._by_category.get(category, set()))
            else:
                names = list(self._descriptors.keys())

            # Filter by source
            if not include_plugins or not include_builtins:
                names = [
                    name
                    for name in names
                    if (include_builtins and self._descriptors[name].is_builtin)
                    or (include_plugins and not self._descriptors[name].is_builtin)
                ]

            return sorted(names)

    def get_plugin_descriptors(self) -> dict[str, str]:
        """
        Get mapping of plugin descriptor names to plugin names.

        Returns:
            Dictionary mapping descriptor_name -> plugin_name
        """
        with self._lock:
            return self._plugin_descriptors.copy()

    def get_fragment_descriptors(self) -> list[str]:
        """
        Get all fragment descriptors (fr_* descriptors).

        Returns:
            List of fragment descriptor names
        """
        return self.list_available_descriptors(category=DescriptorCategory.FRAGMENTS)

    def get_drug_likeness_descriptors(self) -> list[str]:
        """
        Get all drug-likeness descriptors.

        Returns:
            List of drug-likeness descriptor names
        """
        return self.list_available_descriptors(category=DescriptorCategory.DRUG_LIKENESS)

    def get_3d_descriptors(self) -> list[str]:
        """
        Get all descriptors that require 3D coordinates.

        Returns:
            List of 3D descriptor names
        """
        with self._lock:
            return [name for name, reg in self._descriptors.items() if reg.metadata.requires_3d]

    def get_charge_dependent_descriptors(self) -> list[str]:
        """
        Get all descriptors that require partial charges.

        Returns:
            List of charge-dependent descriptor names
        """
        with self._lock:
            return [
                name for name, reg in self._descriptors.items() if reg.metadata.requires_charges
            ]

    def list_all_descriptors(self) -> list[str]:
        """
        List all registered descriptors.

        Returns:
            List of all descriptor names (sorted)
        """
        with self._lock:
            return sorted(self._descriptors.keys())

    def get_metadata(self, name: str) -> DescriptorMetadata | None:
        """
        Get metadata for a descriptor.

        Args:
            name: Descriptor name

        Returns:
            DescriptorMetadata object or None if not found
        """
        with self._lock:
            registration = self._descriptors.get(name)
            return registration.metadata if registration else None

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics about registered descriptors
        """
        with self._lock:
            total = len(self._descriptors)
            builtins = sum(1 for reg in self._descriptors.values() if reg.is_builtin)
            plugins = total - builtins

            by_category = {cat.value: len(names) for cat, names in self._by_category.items()}

            requires_3d = len(self.get_3d_descriptors())
            requires_charges = len(self.get_charge_dependent_descriptors())

            return {
                "total_descriptors": total,
                "builtin_descriptors": builtins,
                "plugin_descriptors": plugins,
                "by_category": by_category,
                "requires_3d": requires_3d,
                "requires_charges": requires_charges,
                "plugins": len(set(self._plugin_descriptors.values())),
            }

    def get_availability_report(self) -> dict[str, Any]:
        """
        Generate comprehensive report of descriptor availability.

        Returns:
            Dictionary containing:
            - total_registered: Number of successfully registered descriptors
            - failed_descriptors: List of descriptors that couldn't be registered
            - mol_method_descriptors: List of descriptors using Mol methods
            - rdkit_version: RDKit version information
            - by_category: Breakdown by category
        """
        from rdkit import __version__ as rdkit_version

        # Count by category
        by_category = {}
        for reg in self._descriptors.values():
            cat = reg.metadata.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        # Identify Mol method descriptors
        mol_methods = [
            name for name, reg in self._descriptors.items() if reg.metadata.rdkit_module == "Mol"
        ]

        total_requested = len(get_all_descriptor_names())

        report = {
            "rdkit_version": rdkit_version,
            "total_registered": len(self._descriptors),
            "total_requested": total_requested,
            "failed_descriptors": getattr(self, "_failed_descriptors", []),
            "mol_method_descriptors": mol_methods,
            "by_category": by_category,
            "success_rate": len(self._descriptors) / total_requested * 100
            if total_requested > 0
            else 0,
        }

        return report

    def log_availability_summary(self):
        """Log a human-readable summary of descriptor availability."""
        report = self.get_availability_report()

        logger.info("=" * 70)
        logger.info("DESCRIPTOR REGISTRY SUMMARY")
        logger.info("=" * 70)
        logger.info(f"RDKit Version: {report['rdkit_version']}")
        logger.info(
            f"Total Registered: {report['total_registered']} / {report['total_requested']} "
            f"({report['success_rate']:.1f}%)"
        )

        if report["by_category"]:
            logger.info("\nBy Category:")
            for category, count in sorted(report["by_category"].items()):
                logger.info(f"  {category:20s}: {count:3d} descriptors")

        if report["mol_method_descriptors"]:
            logger.info(f"\nMol Method Descriptors: {len(report['mol_method_descriptors'])}")
            logger.debug(f"  {report['mol_method_descriptors']}")

        if report["failed_descriptors"]:
            logger.info(
                f"\nUnavailable in Current RDKit Version: {len(report['failed_descriptors'])}"
            )
            failed_names = [name for name, _ in report["failed_descriptors"][:10]]
            logger.debug(f"  {failed_names}")
            if len(report["failed_descriptors"]) > 10:
                logger.debug(f"  ... and {len(report['failed_descriptors']) - 10} more")

        logger.info("=" * 70)

    # =========================================================================
    # RESET (For Testing)
    # =========================================================================

    def reset(self) -> None:
        """
        Reset the registry (useful for testing).

        Clears all registered descriptors.
        """
        with self._lock:
            self._descriptors.clear()
            self._by_category.clear()
            self._plugin_descriptors.clear()
            logger.info("Registry reset")


# =============================================================================
# GLOBAL REGISTRY INSTANCE
# =============================================================================

# Global registry instance
registry = DescriptorRegistry()
registry.auto_discover_rdkit_descriptors()

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_descriptor(name: str) -> Callable | None:
    """Get descriptor function from global registry"""
    return registry.get_descriptor(name)


def has_descriptor(name: str) -> bool:
    """Check if descriptor exists in global registry"""
    return registry.has_descriptor(name)


def list_descriptors(category: DescriptorCategory | None = None) -> list[str]:
    """List available descriptors from global registry"""
    return registry.list_available_descriptors(category=category)


def auto_discover_rdkit() -> int:
    """Auto-discover RDKit descriptors in global registry"""
    return registry.auto_discover_rdkit_descriptors()
