"""
Data Converter

DYNAMIC, PRODUCTION-READY, FUTURE-PROOF data conversion for inference.
Supports multiple molecular formats via registry pattern.

Dependency Injection Pattern:
- File-based converters (XYZ, SDF) accept optional working_root_dir parameter
- No hidden config loading (Service Locator anti-pattern removed)
- Follows CallbackFactory pattern from models/training/callbacks.py

Author: MILIA Team
Version: 2.0.0
"""

import contextlib
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import torch
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 21: 3D CONFORMER GENERATION FOR PREDICTION FROM SMILES/InChI
# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC: Automatically detects when 3D features are needed from config
# PRODUCTION-READY: Uses RDKit's validated ETKDGv3 algorithm with CSD torsions
# FUTURE-PROOF: Works with any future 3D bond features added to config
#
# Evidence (RDKit Documentation):
# - ETKDGv3 is the default since 2024.03 release
# - Combines distance geometry with Cambridge Structural Database torsion preferences
# - AllChem.EmbedMolecule() generates 3D conformer from SMILES/InChI
#
# This function is ONLY called for prediction (post_training) molecules.
# Dataset molecules use QM-optimized coordinates via molecule_converter_core.py
# which passes coordinates parameter to add_structural_features().
# ═══════════════════════════════════════════════════════════════════════════

# Features that require 3D coordinates (conformer)
_3D_BOND_FEATURES = {"bond_length", "bond_length_binned"}


def _requires_3d_conformer(structural_features_config: dict[str, Any] | None) -> bool:
    """
    Check if the structural features config requires 3D coordinates.

    DYNAMIC: Checks bond features list for any 3D-requiring features

    Args:
        structural_features_config: Featurization config from checkpoint

    Returns:
        True if 3D conformer is needed, False otherwise
    """
    if structural_features_config is None:
        return False

    bond_features = structural_features_config.get("bond", [])
    return bool(_3D_BOND_FEATURES.intersection(set(bond_features)))


def _ensure_3d_conformer_for_prediction(
    mol, structural_features_config: dict[str, Any] | None
) -> bool:
    """
    Generate 3D conformer for prediction molecules if needed.

    DYNAMIC: Only generates conformer when 3D features are in config
    PRODUCTION-READY: Uses RDKit's ETKDGv3 (default since 2024.03)
    FUTURE-PROOF: Graceful handling of embedding failures

    IMPORTANT: This function is ONLY for prediction from SMILES/InChI.
    Dataset molecules receive QM-optimized coordinates via the coordinates
    parameter in add_structural_features(), which is NOT passed here.

    Args:
        mol: RDKit molecule object (modified in place)
        structural_features_config: Featurization config from checkpoint

    Returns:
        True if conformer was generated or already exists, False on failure
    """
    # Skip if no 3D features needed
    if not _requires_3d_conformer(structural_features_config):
        return True

    # Skip if conformer already exists
    if mol.GetNumConformers() > 0:
        return True

    # Generate 3D conformer using RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # ================================================================
        # CRITICAL: Add explicit hydrogens before conformer generation
        # ================================================================
        # Evidence (RDKit Documentation): "Note that we add Hs to the molecule
        # before generating the conformer. This is essential to get good structures"
        # Evidence (RDKit Blog): "add hydrogens so that we get a reasonable conformer"
        # ================================================================
        mol_with_hs = Chem.AddHs(mol)

        # Use ETKDGv3 - the default and most accurate method since RDKit 2024.03
        # Evidence: RDKit documentation states "Since the 2024.03 release ETKDGv3 is the default"
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # Reproducibility for consistent predictions

        result = AllChem.EmbedMolecule(mol_with_hs, params)

        if result == -1:
            # Embedding failed - try with random coordinates as fallback
            # Evidence: RDKit blog "Looking at random-coordinate embedding" shows this helps
            # for difficult molecules
            logger.debug("ETKDGv3 embedding failed, trying with random coordinates")
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol_with_hs, params)

        if result == -1:
            logger.warning(
                "Failed to generate 3D conformer for molecule. "
                "3D bond features (bond_length, bond_length_binned) will use fallback values."
            )
            return False

        # Optional: Optimize geometry with force field for better bond lengths
        try:
            AllChem.MMFFOptimizeMolecule(mol_with_hs)
        except Exception:
            # MMFF optimization is optional - UFF fallback
            with contextlib.suppress(Exception):  # Use unoptimized conformer
                AllChem.UFFOptimizeMolecule(mol_with_hs)

        # ================================================================
        # Transfer conformer coordinates to original molecule (without Hs)
        # ================================================================
        # The original mol object is used for feature extraction, so we need
        # to transfer the heavy atom coordinates from mol_with_hs to mol
        # ================================================================
        conf_with_hs = mol_with_hs.GetConformer()
        conf = Chem.Conformer(mol.GetNumAtoms())

        # Map heavy atoms from mol_with_hs to mol
        # Heavy atoms in mol_with_hs have the same indices as in mol (Hs are added at end)
        for i in range(mol.GetNumAtoms()):
            pos = conf_with_hs.GetAtomPosition(i)
            conf.SetAtomPosition(i, pos)

        mol.AddConformer(conf, assignId=True)

        logger.debug("Generated 3D conformer for prediction molecule (with H optimization)")
        return True

    except ImportError:
        logger.warning("RDKit AllChem not available - cannot generate 3D conformer")
        return False
    except Exception as e:
        logger.warning(f"Error generating 3D conformer: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL: Defines what a converter must implement
# ═══════════════════════════════════════════════════════════════════════════


class DataConverterProtocol(Protocol):
    """Protocol for data converters - enables duck typing."""

    def convert(self, input_data: Any, **kwargs) -> Data:
        """Convert input to PyG Data."""
        ...

    def can_convert(self, input_data: Any) -> bool:
        """Check if this converter can handle the input."""
        ...

    @property
    def format_name(self) -> str:
        """Return the format name this converter handles."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if required dependencies are available."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY: Dynamic, Thread-Safe, Following MILIA's Registry Pattern
# ═══════════════════════════════════════════════════════════════════════════


class DataConverterRegistry:
    """
    Registry for data converters.

    DYNAMIC: Converters self-register via @register_converter decorator
    PRODUCTION-READY: Thread-safe, graceful dependency handling
    FUTURE-PROOF: New formats added without modifying this class

    Follows MILIA's existing registry patterns (ModelRegistry, DatasetRegistry).
    """

    _instance: Optional["DataConverterRegistry"] = None
    _lock = threading.RLock()

    def __new__(cls):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._converters: dict[str, type[BaseDataConverter]] = {}
                cls._instance._initialized = False
            return cls._instance

    def register(self, format_name: str, converter_class: type["BaseDataConverter"]) -> None:
        """
        Register a converter for a format.

        Args:
            format_name: Format identifier (e.g., "smiles", "xyz")
            converter_class: Converter class
        """
        with self._lock:
            self._converters[format_name.lower()] = converter_class
            logger.debug(f"Registered converter for format: {format_name}")

    def get(self, format_name: str) -> type["BaseDataConverter"]:
        """Get converter class for format."""
        with self._lock:
            format_lower = format_name.lower()
            if format_lower not in self._converters:
                available = self.list_available()
                raise ValueError(
                    f"No converter registered for format '{format_name}'. "
                    f"Available formats: {available}"
                )
            return self._converters[format_lower]

    def list_all(self) -> list[str]:
        """List all registered format names."""
        with self._lock:
            return list(self._converters.keys())

    def list_available(self) -> list[str]:
        """List formats with available dependencies."""
        with self._lock:
            available = []
            for name, cls in self._converters.items():
                try:
                    instance = cls()
                    if instance.is_available:
                        available.append(name)
                except Exception:
                    pass  # Dependency not available
            return available

    def is_registered(self, format_name: str) -> bool:
        """Check if format is registered."""
        with self._lock:
            return format_name.lower() in self._converters

    def auto_detect(self, input_data: Any) -> str | None:
        """
        Auto-detect input format.

        DYNAMIC: Iterates through all registered converters and asks each
                 if it can handle the input.

        Args:
            input_data: Input data to detect format of

        Returns:
            Format name or None if not detected
        """
        with self._lock:
            for name, cls in self._converters.items():
                try:
                    instance = cls()
                    if instance.is_available and instance.can_convert(input_data):
                        return name
                except Exception:
                    continue
            return None


# Global registry instance
_registry = DataConverterRegistry()


def register_converter(format_name: str):
    """
    Decorator to register a converter class.

    Following MILIA's @register pattern from datasets module.

    Usage:
        @register_converter("xyz")
        class XYZConverter(BaseDataConverter):
            ...
    """

    def decorator(cls: type["BaseDataConverter"]) -> type["BaseDataConverter"]:
        _registry.register(format_name, cls)
        return cls

    return decorator


def get_registry() -> DataConverterRegistry:
    """Get the global converter registry."""
    return _registry


# ═══════════════════════════════════════════════════════════════════════════
# BASE CLASS: Abstract base for all converters
# ═══════════════════════════════════════════════════════════════════════════


class BaseDataConverter(ABC):
    """
    Abstract base class for data converters.

    Subclasses implement conversion for specific formats.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name this converter handles."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if required dependencies are available."""
        pass

    @abstractmethod
    def can_convert(self, input_data: Any) -> bool:
        """Check if this converter can handle the input."""
        pass

    @abstractmethod
    def convert(self, input_data: Any, **kwargs) -> Data:
        """Convert input to PyG Data."""
        pass

    def convert_batch(self, inputs: list[Any], **kwargs) -> Batch:
        """Convert multiple inputs to a batch."""
        data_list = [self.convert(inp, **kwargs) for inp in inputs]
        return Batch.from_data_list(data_list)


# ═══════════════════════════════════════════════════════════════════════════
# CONVERTER IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════


@register_converter("pyg_data")
class PyGDataConverter(BaseDataConverter):
    """
    Passthrough converter for PyG Data objects.

    Validates and optionally normalizes existing PyG Data.
    """

    @property
    def format_name(self) -> str:
        return "pyg_data"

    @property
    def is_available(self) -> bool:
        return True  # No extra dependencies

    def can_convert(self, input_data: Any) -> bool:
        return isinstance(input_data, Data)

    def convert(self, input_data: Any, **kwargs) -> Data:
        if not isinstance(input_data, Data):
            raise TypeError(f"Expected PyG Data, got {type(input_data)}")
        return input_data


@register_converter("dict")
class DictConverter(BaseDataConverter):
    """
    Convert dict with tensors to PyG Data.

    Expected keys: x, edge_index (required), edge_attr, pos, z, y (optional)
    """

    @property
    def format_name(self) -> str:
        return "dict"

    @property
    def is_available(self) -> bool:
        return True  # No extra dependencies

    def can_convert(self, input_data: Any) -> bool:
        if not isinstance(input_data, dict):
            return False
        # Must have at least x or z, and edge_index
        has_features = "x" in input_data or "z" in input_data
        has_edges = "edge_index" in input_data
        return has_features and has_edges

    def convert(self, input_data: Any, **kwargs) -> Data:
        if not isinstance(input_data, dict):
            raise TypeError(f"Expected dict, got {type(input_data)}")

        data_dict = {}
        for key, value in input_data.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value
            elif isinstance(value, (list, tuple)):
                data_dict[key] = torch.tensor(value)
            else:
                data_dict[key] = value  # Keep as-is (e.g., strings)

        return Data(**data_dict)


@register_converter("smiles")
class SMILESConverter(BaseDataConverter):
    """
    Convert SMILES strings to PyG Data using RDKit.

    Requires: pip install rdkit
    """

    def __init__(
        self,
        atom_features: list[str] | None = None,
        bond_features: list[str] | None = None,
        add_hydrogens: bool = False,
        structural_features_config: dict[str, Any] | None = None,  # FIX 20: NEW PARAMETER
    ):
        # ====================================================================
        # FIX 20: ACCEPT STRUCTURAL_FEATURES_CONFIG FOR TRAINING-COMPATIBLE FEATURIZATION
        # ====================================================================
        # DYNAMIC: Uses whatever featurization config is passed from checkpoint
        # PRODUCTION-READY: Falls back to simple defaults if no config provided
        # FUTURE-PROOF: Works with any structural_features_config structure
        # ====================================================================
        self.structural_features_config = structural_features_config
        self._use_structural_features = False

        if structural_features_config and (
            structural_features_config.get("atom") or structural_features_config.get("bond")
        ):
            # Use structural features from checkpoint config (training-time featurization)
            self._use_structural_features = True
            # Note: Actual features will be applied via add_structural_features() in convert()
            # Store config for use in convert() method
            self.atom_features = structural_features_config.get("atom", [])
            self.bond_features = structural_features_config.get("bond", [])
            logger.info(
                f"SMILESConverter using checkpoint featurization: "
                f"atom={self.atom_features}, bond={self.bond_features}"
            )
        else:
            # Use simple default features (original behavior)
            self.atom_features = atom_features or [
                "atomic_num",
                "degree",
                "formal_charge",
                "num_hs",
                "hybridization",
                "is_aromatic",
            ]
            self.bond_features = bond_features or ["bond_type", "is_conjugated", "is_in_ring"]

        self.add_hydrogens = add_hydrogens
        self._rdkit = None

    @property
    def format_name(self) -> str:
        return "smiles"

    @property
    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("rdkit") is not None

    def can_convert(self, input_data: Any) -> bool:
        if not isinstance(input_data, str):
            return False
        # ====================================================================
        # EXCLUDE InChI STRINGS - They have their own converter
        # ====================================================================
        # InChI strings start with "InChI=" and should NOT be handled by SMILES
        if input_data.startswith("InChI="):
            return False
        # Check if it looks like SMILES (basic heuristic)
        # SMILES contain atoms: C, N, O, S, P, F, Cl, Br, I, etc.
        # Not a file path
        return any(
            c in input_data for c in ["C", "N", "O", "c", "n", "o"]
        ) and not input_data.endswith((".xyz", ".sdf", ".mol", ".cif"))

    def convert(self, input_data: Any, **kwargs) -> Data:
        from rdkit import Chem

        smiles = input_data
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        if self.add_hydrogens:
            mol = Chem.AddHs(mol)

        # ====================================================================
        # FIX 20: USE STRUCTURAL FEATURES FROM CHECKPOINT IF AVAILABLE
        # ====================================================================
        # DYNAMIC: Applies same featurization as training via add_structural_features
        # PRODUCTION-READY: Falls back to simple features if no config
        # FUTURE-PROOF: Uses existing mol_structural_features.py infrastructure
        # ====================================================================
        if self._use_structural_features and self.structural_features_config:
            # Use add_structural_features for training-compatible featurization
            from milia_pipeline.molecules.mol_structural_features import add_structural_features

            # ================================================================
            # FIX 21: GENERATE 3D CONFORMER FOR BOND LENGTH FEATURES
            # ================================================================
            # DYNAMIC: Only generates when 3D features are in config
            # PRODUCTION-READY: Uses ETKDGv3 algorithm
            # FUTURE-PROOF: Handles any future 3D features
            # ================================================================
            _ensure_3d_conformer_for_prediction(mol, self.structural_features_config)

            # Create edge_index first (needed for add_structural_features)
            edge_index, edge_attr = self._get_bond_features(mol)

            # Create base PyG data with edge structure
            base_data = Data(
                edge_index=edge_index,
                edge_attr=edge_attr if edge_attr.numel() > 0 else None,
                smiles=smiles,
            )

            # Apply structural features (same as training) - this sets data.x
            enhanced_data = add_structural_features(
                rdkit_mol=mol,
                pyg_data=base_data,
                feature_config=self.structural_features_config,
                logger=logger,
            )

            return enhanced_data
        else:
            # Use simple default featurization (original behavior)
            x = self._get_atom_features(mol)
            edge_index, edge_attr = self._get_bond_features(mol)

            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr if edge_attr.numel() > 0 else None,
                smiles=smiles,
            )

    def _get_atom_features(self, mol) -> torch.Tensor:
        """Extract atom features from RDKit mol."""
        features = []
        for atom in mol.GetAtoms():
            atom_feat = []
            if "atomic_num" in self.atom_features:
                atom_feat.append(atom.GetAtomicNum())
            if "degree" in self.atom_features:
                atom_feat.append(atom.GetDegree())
            if "formal_charge" in self.atom_features:
                atom_feat.append(atom.GetFormalCharge())
            if "num_hs" in self.atom_features:
                atom_feat.append(atom.GetTotalNumHs())
            if "hybridization" in self.atom_features:
                atom_feat.append(int(atom.GetHybridization()))
            if "is_aromatic" in self.atom_features:
                atom_feat.append(int(atom.GetIsAromatic()))
            features.append(atom_feat)
        return torch.tensor(features, dtype=torch.float)

    def _get_bond_features(self, mol) -> tuple:
        """Extract edge index and features."""
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])

            bond_feat = []
            if "bond_type" in self.bond_features:
                bond_feat.append(float(bond.GetBondTypeAsDouble()))
            if "is_conjugated" in self.bond_features:
                bond_feat.append(int(bond.GetIsConjugated()))
            if "is_in_ring" in self.bond_features:
                bond_feat.append(int(bond.IsInRing()))

            edge_features.extend([bond_feat, bond_feat])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(self.bond_features)), dtype=torch.float)

        return edge_index, edge_attr


@register_converter("inchi")
class InChIConverter(BaseDataConverter):
    """
    Convert InChI (IUPAC International Chemical Identifier) to PyG Data.

    InChI is commonly used in chemical databases like PubChem.

    Requires: pip install rdkit

    Evidence (RDKit Documentation): rdkit.Chem.inchi.MolFromInchi() constructs
    a molecule from an InChI string.
    """

    def __init__(
        self,
        atom_features: list[str] | None = None,
        bond_features: list[str] | None = None,
        add_hydrogens: bool = False,
    ):
        self.atom_features = atom_features or [
            "atomic_num",
            "degree",
            "formal_charge",
            "num_hs",
            "hybridization",
            "is_aromatic",
        ]
        self.bond_features = bond_features or ["bond_type", "is_conjugated", "is_in_ring"]
        self.add_hydrogens = add_hydrogens

    @property
    def format_name(self) -> str:
        return "inchi"

    @property
    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("rdkit.Chem.inchi") is not None

    def can_convert(self, input_data: Any) -> bool:
        if not isinstance(input_data, str):
            return False
        # InChI strings start with "InChI=" prefix
        return input_data.startswith("InChI=")

    def convert(self, input_data: Any, **kwargs) -> Data:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolFromInchi

        inchi = input_data
        mol = MolFromInchi(inchi)

        if mol is None:
            raise ValueError(f"Invalid InChI: {inchi}")

        if self.add_hydrogens:
            mol = Chem.AddHs(mol)

        # Reuse SMILES converter's featurization logic
        smiles_conv = SMILESConverter(
            atom_features=self.atom_features, bond_features=self.bond_features
        )

        x = smiles_conv._get_atom_features(mol)
        edge_index, edge_attr = smiles_conv._get_bond_features(mol)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr if edge_attr.numel() > 0 else None,
            inchi=inchi,
        )


@register_converter("xyz")
class XYZConverter(BaseDataConverter):
    """
    Convert XYZ format (3D coordinates) to PyG Data.

    For 3D models like SchNet, DimeNet that need atomic numbers (z) and positions (pos).

    Dependency Injection Pattern:
    - Accepts optional working_root_dir for path resolution
    - Absolute paths used as-is
    - Relative paths resolved against working_root_dir (or cwd if not provided)

    Requires: pip install ase
    """

    def __init__(self, cutoff: float = 5.0, working_root_dir: Path | None = None):
        """
        Args:
            cutoff: Distance cutoff for creating edges (Angstrom)
            working_root_dir: Base directory for resolving relative paths.
                              If None, uses current working directory.
        """
        self.cutoff = cutoff
        self._working_root_dir = (
            Path(working_root_dir).expanduser().resolve() if working_root_dir else Path.cwd()
        )

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve path against working_root_dir."""
        path = Path(path).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self._working_root_dir / path).resolve()

    @property
    def format_name(self) -> str:
        return "xyz"

    @property
    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("ase.io") is not None

    def can_convert(self, input_data: Any) -> bool:
        # Check if it's an XYZ file path
        if isinstance(input_data, (str, Path)):
            path = self._resolve_path(input_data)
            return path.suffix.lower() == ".xyz" and path.exists()
        return False

    def convert(self, input_data: Any, **kwargs) -> Data:
        import ase.io
        from ase.neighborlist import neighbor_list

        # Resolve path against working_root_dir
        resolved_path = self._resolve_path(input_data)

        # Load XYZ file
        atoms = ase.io.read(str(resolved_path))

        # Atomic numbers and positions
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float)

        # Create edges based on distance cutoff
        cutoff = kwargs.get("cutoff", self.cutoff)
        i, j = neighbor_list("ij", atoms, cutoff)
        # ASE's neighbor_list returns numpy ndarrays. Wrapping two ndarrays in a
        # Python list before passing to torch.tensor() is a documented PyTorch
        # anti-pattern (UserWarning at tensor_new.cpp; see pytorch/pytorch#13918)
        # that triggers an O(N) element-wise copy. np.stack joins them into a
        # single contiguous [2, num_edges] ndarray — the exact shape PyG requires
        # for edge_index in COO format — and torch.from_numpy enables the
        # zero-copy fast ingestion path before the dtype cast to long.
        edge_index = torch.from_numpy(np.stack([i, j])).to(dtype=torch.long)

        return Data(z=z, pos=pos, edge_index=edge_index, num_nodes=len(atoms))


@register_converter("ase_atoms")
class ASEAtomsConverter(BaseDataConverter):
    """
    Convert ASE Atoms object to PyG Data.

    For direct integration with ASE-based workflows.

    Requires: pip install ase
    """

    def __init__(self, cutoff: float = 5.0):
        self.cutoff = cutoff

    @property
    def format_name(self) -> str:
        return "ase_atoms"

    @property
    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("ase") is not None

    def can_convert(self, input_data: Any) -> bool:
        try:
            from ase import Atoms

            return isinstance(input_data, Atoms)
        except ImportError:
            return False

    def convert(self, input_data: Any, **kwargs) -> Data:
        from ase.neighborlist import neighbor_list

        atoms = input_data
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float)

        cutoff = kwargs.get("cutoff", self.cutoff)
        i, j = neighbor_list("ij", atoms, cutoff)
        # ASE's neighbor_list returns numpy ndarrays. Wrapping two ndarrays in a
        # Python list before passing to torch.tensor() is a documented PyTorch
        # anti-pattern (UserWarning at tensor_new.cpp; see pytorch/pytorch#13918)
        # that triggers an O(N) element-wise copy. np.stack joins them into a
        # single contiguous [2, num_edges] ndarray — the exact shape PyG requires
        # for edge_index in COO format — and torch.from_numpy enables the
        # zero-copy fast ingestion path before the dtype cast to long.
        edge_index = torch.from_numpy(np.stack([i, j])).to(dtype=torch.long)

        return Data(z=z, pos=pos, edge_index=edge_index, num_nodes=len(atoms))


@register_converter("sdf")
class SDFConverter(BaseDataConverter):
    """
    Convert SDF/MOL format to PyG Data.

    Dependency Injection Pattern:
    - Accepts optional working_root_dir for path resolution
    - Absolute paths used as-is
    - Relative paths resolved against working_root_dir (or cwd if not provided)

    Requires: pip install rdkit
    """

    def __init__(self, working_root_dir: Path | None = None):
        """
        Args:
            working_root_dir: Base directory for resolving relative paths.
                              If None, uses current working directory.
        """
        self._working_root_dir = (
            Path(working_root_dir).expanduser().resolve() if working_root_dir else Path.cwd()
        )

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve path against working_root_dir."""
        path = Path(path).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self._working_root_dir / path).resolve()

    @property
    def format_name(self) -> str:
        return "sdf"

    @property
    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("rdkit") is not None

    def can_convert(self, input_data: Any) -> bool:
        if isinstance(input_data, (str, Path)):
            path = self._resolve_path(input_data)
            return path.suffix.lower() in (".sdf", ".mol") and path.exists()
        return False

    def _mol_to_data(self, mol, Chem) -> Data:
        """
        Convert a single RDKit mol to PyG Data.

        DYNAMIC: Reuses SMILESConverter featurization logic
        PRODUCTION-READY: Handles 3D coordinates when available
        FUTURE-PROOF: Preserves SMILES for downstream processing

        Args:
            mol: RDKit molecule object
            Chem: RDKit Chem module (passed to avoid re-import)

        Returns:
            PyG Data object
        """
        # Use SMILES converter logic for features
        smiles_conv = SMILESConverter()

        # Get atom and bond features
        x = smiles_conv._get_atom_features(mol)
        edge_index, edge_attr = smiles_conv._get_bond_features(mol)

        # Also get 3D coordinates if available
        pos = None
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            pos = torch.tensor(
                [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=torch.float
            )

        # Preserve SMILES for dynamic post-processing featurization
        smiles = Chem.MolToSmiles(mol)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr if edge_attr.numel() > 0 else None,
            pos=pos,
            smiles=smiles,  # Preserved for dynamic featurization
        )

    def convert_all(self, input_data: Any, **kwargs) -> list[Data]:
        """
        Convert ALL molecules from a multi-molecule SDF file to PyG Data list.

        ====================================================================
        FIX 24: MULTI-MOLECULE SDF FILE SUPPORT
        ====================================================================
        DYNAMIC: Iterates through SDMolSupplier to get all molecules
        PRODUCTION-READY: Handles parsing errors gracefully per-molecule
        FUTURE-PROOF: Returns List[Data] compatible with Batch.from_data_list()
        ====================================================================

        Args:
            input_data: Path to SDF file (str or Path)
            **kwargs: Additional arguments (unused, for API compatibility)

        Returns:
            List of PyG Data objects, one per molecule in SDF file

        Raises:
            ValueError: If no valid molecules found in file
        """
        from rdkit import Chem

        # Resolve path against working_root_dir
        resolved_path = self._resolve_path(input_data)

        # Load SDF file with SDMolSupplier
        suppl = Chem.SDMolSupplier(str(resolved_path))

        data_list = []
        failed_count = 0

        for idx, mol in enumerate(suppl):
            if mol is None:
                logger.warning(f"Failed to parse molecule {idx} from {resolved_path}")
                failed_count += 1
                continue

            try:
                data = self._mol_to_data(mol, Chem)
                data_list.append(data)
            except Exception as e:
                logger.warning(f"Failed to convert molecule {idx} from {resolved_path}: {e}")
                failed_count += 1

        if not data_list:
            raise ValueError(
                f"No valid molecules found in {resolved_path}. "
                f"Failed to parse {failed_count} molecule(s)."
            )

        if failed_count > 0:
            logger.info(
                f"Loaded {len(data_list)} molecules from {resolved_path} "
                f"({failed_count} failed to parse)"
            )
        else:
            logger.debug(f"Loaded {len(data_list)} molecules from {resolved_path}")

        return data_list

    def convert(self, input_data: Any, **kwargs) -> Data:
        """
        Convert SDF file to PyG Data.

        For multi-molecule SDF files, returns the FIRST molecule only.
        Use convert_all() to get all molecules.

        Args:
            input_data: Path to SDF file (str or Path)
            **kwargs: Additional arguments

        Returns:
            PyG Data object for first molecule in file
        """
        from rdkit import Chem

        # Resolve path against working_root_dir
        resolved_path = self._resolve_path(input_data)

        # Load SDF file
        suppl = Chem.SDMolSupplier(str(resolved_path))
        mol = next(iter(suppl))

        if mol is None:
            raise ValueError(f"Failed to load molecule from {resolved_path}")

        return self._mol_to_data(mol, Chem)


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS (DYNAMIC API)
# ═══════════════════════════════════════════════════════════════════════════


def _apply_structural_features_if_available(
    data: Data, structural_features_config: dict[str, Any] | None
) -> Data:
    """
    Apply structural features to PyG Data if config is provided and mol can be reconstructed.

    This is the DYNAMIC, PRODUCTION-READY, FUTURE-PROOF solution for applying
    training-time featurization to ANY converter output that preserves SMILES or InChI.

    DYNAMIC: Works with any converter that stores smiles or inchi attribute
    PRODUCTION-READY: Graceful fallback if mol cannot be reconstructed
    FUTURE-PROOF: Future converters automatically benefit by storing smiles/inchi

    Args:
        data: PyG Data object (may have smiles or inchi attribute)
        structural_features_config: Featurization config from checkpoint

    Returns:
        Enhanced PyG Data with structural features, or original data if not applicable
    """
    if structural_features_config is None:
        return data

    # Check if config has any features to apply
    if not (structural_features_config.get("atom") or structural_features_config.get("bond")):
        return data

    # Try to reconstruct RDKit mol from stored representation
    mol = None
    mol_source = None

    try:
        from rdkit import Chem

        # Try SMILES first (most common)
        if hasattr(data, "smiles") and data.smiles:
            mol = Chem.MolFromSmiles(data.smiles)
            mol_source = "smiles"
        # Try InChI second
        elif hasattr(data, "inchi") and data.inchi:
            from rdkit.Chem.inchi import MolFromInchi

            mol = MolFromInchi(data.inchi)
            mol_source = "inchi"
    except ImportError:
        logger.debug("RDKit not available - structural features not applied")
        return data

    if mol is None:
        # Can't reconstruct mol (XYZ, ASE, dict, or invalid SMILES/InChI)
        if mol_source:
            logger.warning(
                f"Failed to reconstruct mol from {mol_source} - structural features not applied"
            )
        else:
            logger.debug(
                "No SMILES/InChI in data - structural features not applied "
                "(this is expected for XYZ, ASE, dict formats)"
            )
        return data

    # Apply structural features using existing infrastructure
    try:
        from milia_pipeline.molecules.mol_structural_features import add_structural_features

        # ====================================================================
        # FIX 21: GENERATE 3D CONFORMER FOR BOND LENGTH FEATURES
        # ====================================================================
        # DYNAMIC: Only generates when 3D features are in config
        # PRODUCTION-READY: Uses ETKDGv3 algorithm
        # FUTURE-PROOF: Handles any future 3D features
        # ====================================================================
        _ensure_3d_conformer_for_prediction(mol, structural_features_config)

        # Preserve existing edge structure
        enhanced_data = add_structural_features(
            rdkit_mol=mol,
            pyg_data=data,
            feature_config=structural_features_config,
            logger=logger,
        )

        logger.debug(
            f"Applied structural features from checkpoint config: "
            f"atom={structural_features_config.get('atom', [])}"
        )

        return enhanced_data

    except Exception as e:
        logger.warning(f"Failed to apply structural features: {e} - returning original data")
        return data


def convert_to_pyg(
    input_data: Any,
    format: str | None = None,
    structural_features_config: dict[str, Any] | None = None,
    **kwargs,
) -> Data:
    """
    Convert any supported input to PyG Data.

    DYNAMIC: Auto-detects format if not specified.
    PRODUCTION-READY: Supports structural_features_config for training-compatible featurization.
    FUTURE-PROOF: Works with any converter; structural features applied via post-processing.

    Args:
        input_data: Input data (SMILES, XYZ path, dict, PyG Data, etc.)
        format: Optional format hint (auto-detected if None)
        structural_features_config: Optional featurization config from checkpoint.
                                    If provided, applies same featurization as training
                                    to ensure dimension compatibility.
        **kwargs: Additional arguments passed to converter

    Returns:
        PyG Data object

    Example:
        >>> from milia_pipeline.models.post_training import convert_to_pyg
        >>>
        >>> # SMILES (auto-detected)
        >>> data = convert_to_pyg("CCO")
        >>>
        >>> # XYZ file (auto-detected)
        >>> data = convert_to_pyg("molecule.xyz")
        >>>
        >>> # With structural features from checkpoint (for prediction mode)
        >>> config = predictor.structural_features_config
        >>> data = convert_to_pyg("CCO", structural_features_config=config)
        >>>
        >>> # Dict (explicit format)
        >>> data = convert_to_pyg({"x": x, "edge_index": ei}, format="dict")
        >>>
        >>> # PyG Data (passthrough)
        >>> data = convert_to_pyg(existing_data)
    """
    registry = get_registry()

    # Determine format
    if format is None:
        format = registry.auto_detect(input_data)
        if format is None:
            # ================================================================
            # FIX 23: DETECT UNAVAILABLE FORMATS AND PROVIDE HELPFUL ERROR
            # ================================================================
            # PROBLEM: When a format is registered but its dependencies are
            #          not installed (e.g., XYZ requires ASE), auto_detect
            #          returns None and the error message only shows available
            #          formats, leaving users confused about why their file
            #          type isn't recognized.
            #
            # SOLUTION: Check if the input LOOKS like a known format that's
            #           registered but unavailable, and provide specific
            #           installation instructions.
            #
            # DYNAMIC: Checks file extension against all registered formats
            # PRODUCTION-READY: Provides actionable error messages
            # FUTURE-PROOF: Works with any file-based format
            # ================================================================

            # Check for file-path-based formats that might be unavailable
            unavailable_format = None
            install_hint = None

            if isinstance(input_data, (str, Path)):
                input_path = Path(input_data)
                suffix = input_path.suffix.lower()

                # Map file extensions to format names and install hints
                extension_to_format = {
                    ".xyz": ("xyz", "pip install ase"),
                    ".sdf": ("sdf", "pip install rdkit"),
                    ".mol": ("sdf", "pip install rdkit"),
                }

                if suffix in extension_to_format:
                    format_name, hint = extension_to_format[suffix]
                    if registry.is_registered(format_name):
                        try:
                            converter_class = registry.get(format_name)
                            instance = converter_class()
                            if not instance.is_available:
                                unavailable_format = format_name
                                install_hint = hint
                        except Exception:
                            pass

            if unavailable_format:
                raise ImportError(
                    f"Detected '{unavailable_format}' format but required dependencies are not installed. "
                    f"Install with: {install_hint}"
                )
            else:
                raise ValueError(
                    f"Cannot auto-detect format for input of type {type(input_data)}. "
                    f"Available formats: {registry.list_available()}. "
                    f"Please specify format explicitly."
                )

    # Get converter and convert
    # Note: structural_features_config is handled in post-processing, not passed to converter
    converter_class = registry.get(format)
    converter = converter_class(**kwargs) if kwargs else converter_class()

    if not converter.is_available:
        raise ImportError(
            f"Converter for '{format}' requires dependencies that are not installed. "
            f"Check the converter's docstring for installation instructions."
        )

    # Step 1: Basic conversion
    data = converter.convert(input_data, **kwargs)

    # ========================================================================
    # FIX 20: DYNAMIC POST-PROCESSING FOR STRUCTURAL FEATURES
    # ========================================================================
    # DYNAMIC: Applies to ANY converter output that has smiles or inchi
    # PRODUCTION-READY: Single point of change for all converters
    # FUTURE-PROOF: New converters automatically benefit by storing smiles/inchi
    # ========================================================================
    if structural_features_config:
        data = _apply_structural_features_if_available(data, structural_features_config)

    return data


def convert_batch_to_pyg(inputs: list[Any], format: str | None = None, **kwargs) -> Batch:
    """
    Convert list of inputs to PyG Batch.

    Args:
        inputs: List of inputs (all same format)
        format: Optional format hint
        **kwargs: Additional arguments

    Returns:
        PyG Batch object
    """
    data_list = [convert_to_pyg(inp, format=format, **kwargs) for inp in inputs]
    return Batch.from_data_list(data_list)


def convert_sdf_to_pyg_list(
    sdf_path: str | Path,
    structural_features_config: dict[str, Any] | None = None,
    working_root_dir: Path | None = None,
    **kwargs,
) -> list[Data]:
    """
    Convert ALL molecules from a multi-molecule SDF file to list of PyG Data.

    ========================================================================
    FIX 24: MULTI-MOLECULE SDF FILE SUPPORT FOR PREDICTION
    ========================================================================
    DYNAMIC: Uses SDFConverter.convert_all() to iterate through all molecules
    PRODUCTION-READY: Applies structural_features_config to each molecule
    FUTURE-PROOF: Returns List[Data] compatible with DataLoader/Batch
    ========================================================================

    Args:
        sdf_path: Path to SDF file (str or Path)
        structural_features_config: Optional featurization config from checkpoint.
                                    If provided, applies same featurization as training.
        working_root_dir: Base directory for resolving relative paths.
        **kwargs: Additional arguments passed to converter

    Returns:
        List of PyG Data objects, one per molecule in SDF file

    Example:
        >>> from milia_pipeline.models.post_training import convert_sdf_to_pyg_list
        >>>
        >>> # Load all molecules from SDF
        >>> data_list = convert_sdf_to_pyg_list("molecules.sdf")
        >>>
        >>> # With structural features from checkpoint
        >>> config = predictor.structural_features_config
        >>> data_list = convert_sdf_to_pyg_list("molecules.sdf", structural_features_config=config)
        >>>
        >>> # Create batch for prediction
        >>> from torch_geometric.data import Batch
        >>> batch = Batch.from_data_list(data_list)
    """
    registry = get_registry()

    # Get SDF converter
    converter_class = registry.get("sdf")
    converter = converter_class(working_root_dir=working_root_dir)

    if not converter.is_available:
        raise ImportError("SDFConverter requires RDKit. Install with: pip install rdkit")

    # Use convert_all to get all molecules
    data_list = converter.convert_all(sdf_path, **kwargs)

    # Apply structural features to each molecule if config provided
    if structural_features_config:
        data_list = [
            _apply_structural_features_if_available(data, structural_features_config)
            for data in data_list
        ]

    return data_list


def list_available_formats() -> list[str]:
    """List formats with available dependencies."""
    return get_registry().list_available()


def list_all_formats() -> list[str]:
    """List all registered formats (including unavailable)."""
    return get_registry().list_all()


# Legacy compatibility aliases
smiles_to_data = convert_to_pyg  # For backward compatibility
