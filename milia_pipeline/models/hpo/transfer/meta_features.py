"""
Meta-Feature Extraction for HPO Transfer Learning

This module provides the MetaFeatureExtractor class for extracting meta-features
from datasets to enable similarity computation in HPO transfer learning.

Meta-features capture dataset characteristics without training and enable:
- Cross-dataset HPO transfer
- Warm-starting optimization on related tasks
- Few-shot hyperparameter optimization
- Dataset similarity computation for study selection

Meta-Feature Categories:
1. Statistical: Dataset size, dimensionality, target statistics
2. Graph-Specific: Density, degree distribution, clustering coefficient
3. Molecular-Specific: Atom types, bond types, molecular weight distribution
4. Feature-Based: Node/edge feature statistics

The module is designed to be future-proof:
- Supports dynamic dataset types via registry-based feature detection
- Gracefully handles missing attributes
- Provides configurable extraction with category selection
- Normalizes features for consistent similarity computation

Author: Milia Team
Version: 1.1.0

Pydantic V2 Migration (Phase 12):
    - Migrated MetaFeatureConfig from @dataclass(frozen=True) to BaseModel with frozen=True
    - Uses @field_validator for individual field validation (max_samples)
    - Uses @model_validator(mode='after') for cross-field validation (categories)
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established patterns from transfer_manager.py (Phase 11)

Pattern References:
- Frozen dataclass pattern: hpo_config.py (lines 59-131)
- Protocol pattern: backends/base.py (lines 18-157)
- Feature extraction pattern: mol_structural_features.py (lines 104-186)
- Statistics computation: pyg_integration.py (lines 348-601)
"""

import logging
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


# =============================================================================
# LAZY IMPORTS FOR OPTIONAL DEPENDENCIES
# =============================================================================


def _lazy_import_torch():
    """Lazy import torch to avoid import errors if not available."""
    try:
        import torch

        return torch
    except ImportError:
        return None


def _lazy_import_torch_geometric():
    """Lazy import torch_geometric utilities."""
    try:
        from torch_geometric.utils import degree

        return degree
    except ImportError:
        return None


def _lazy_import_rdkit():
    """Lazy import RDKit for molecular features."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        return Chem, Descriptors, rdMolDescriptors
    except ImportError:
        return None, None, None


# =============================================================================
# META-FEATURE CATEGORY ENUM
# =============================================================================


class MetaFeatureCategory(Enum):
    """
    Categories of meta-features that can be extracted.

    Allows selective extraction of feature subsets for performance
    or when certain data types are unavailable.

    Attributes:
        STATISTICAL: Basic size and dimensionality features
        GRAPH: Graph structure features (density, degree, clustering)
        MOLECULAR: Chemistry-specific features (atom types, bonds)
        TARGET: Target variable statistics
        NODE_FEATURES: Node feature statistics
        EDGE_FEATURES: Edge feature statistics
        ALL: All available categories
    """

    STATISTICAL = "statistical"
    GRAPH = "graph"
    MOLECULAR = "molecular"
    TARGET = "target"
    NODE_FEATURES = "node_features"
    EDGE_FEATURES = "edge_features"
    ALL = "all"


# =============================================================================
# META-FEATURE CONFIGURATION
# =============================================================================


class MetaFeatureConfig(BaseModel, frozen=True):
    """
    Configuration for meta-feature extraction.

    Pattern: Follows frozen BaseModel pattern from transfer_manager.py (Pydantic V2)

    Controls which meta-feature categories to extract and
    provides options for normalization and sampling.

    Attributes:
        categories: Categories of meta-features to extract
        max_samples: Maximum samples to analyze (None for all)
        normalize: Whether to normalize features to [0, 1]
        include_molecular: Whether to attempt molecular feature extraction
        compute_expensive: Whether to compute expensive features (clustering)

    Examples:
        >>> # Default configuration (all features)
        >>> config = MetaFeatureConfig()

        >>> # Only statistical and graph features
        >>> config = MetaFeatureConfig(
        ...     categories=(MetaFeatureCategory.STATISTICAL, MetaFeatureCategory.GRAPH)
        ... )

        >>> # Fast extraction without expensive features
        >>> config = MetaFeatureConfig(compute_expensive=False, max_samples=100)
    """

    categories: tuple[MetaFeatureCategory, ...] = (MetaFeatureCategory.ALL,)
    max_samples: int | None = None
    normalize: bool = False
    include_molecular: bool = True
    compute_expensive: bool = True

    @field_validator("max_samples")
    @classmethod
    def validate_max_samples(cls, v: int | None) -> int | None:
        """Validate max_samples is at least 1 or None."""
        if v is not None and v < 1:
            raise ValueError("max_samples must be at least 1 or None")
        return v

    @model_validator(mode="after")
    def validate_categories_not_empty(self) -> Self:
        """Validate that categories is not empty."""
        if not self.categories:
            raise ValueError("categories cannot be empty")
        return self

    def should_extract(self, category: MetaFeatureCategory) -> bool:
        """Check if a category should be extracted."""
        if MetaFeatureCategory.ALL in self.categories:
            return True
        return category in self.categories

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()


# =============================================================================
# META-FEATURE EXTRACTOR CLASS
# =============================================================================


class MetaFeatureExtractor:
    """
    Extracts meta-features from datasets for similarity computation.

    Meta-features capture dataset characteristics without training:
    - Statistical: size, dimensionality, class balance
    - Graph-specific: density, degree distribution, clustering
    - Molecular: atom types, bond types, molecular weight distribution

    This class is designed to be future-proof and handle:
    - Any PyG dataset or list of Data objects
    - Missing attributes gracefully
    - Dynamic dataset types via registry
    - Configurable extraction via MetaFeatureConfig

    Usage:
        >>> # Static method for simple extraction
        >>> features = MetaFeatureExtractor.extract(dataset)

        >>> # Instance method for configured extraction
        >>> extractor = MetaFeatureExtractor(config)
        >>> features = extractor.extract_features(dataset)

        >>> # Compute similarity between datasets
        >>> sim = MetaFeatureExtractor.compute_similarity(features_a, features_b)
    """

    # Class-level constants for normalization
    _NORMALIZATION_BOUNDS = {
        "n_samples": (1, 1_000_000),
        "n_features": (1, 10_000),
        "mean_nodes": (1, 10_000),
        "mean_edges": (1, 100_000),
        "mean_density": (0, 1),
        "mean_degree": (0, 100),
        "clustering_coefficient": (0, 1),
    }

    def __init__(self, config: MetaFeatureConfig | None = None):
        """
        Initialize MetaFeatureExtractor with optional configuration.

        Args:
            config: Extraction configuration (uses defaults if None)
        """
        self.config = config or MetaFeatureConfig()
        self._torch = _lazy_import_torch()
        self._degree_fn = _lazy_import_torch_geometric()
        self._rdkit = _lazy_import_rdkit()

    # =========================================================================
    # PRIMARY STATIC METHOD (Matches Blueprint API)
    # =========================================================================

    @staticmethod
    def extract(dataset, config: MetaFeatureConfig | None = None) -> dict[str, float]:
        """
        Extract meta-features from PyG dataset.

        This is the primary API matching the blueprint specification
        (MILIA_HPO_Implementation_Blueprint.md:4083-4112).

        Args:
            dataset: PyG Dataset or list of Data objects
            config: Optional extraction configuration

        Returns:
            Dictionary of meta-feature name to float value

        Examples:
            >>> features = MetaFeatureExtractor.extract(dataset)
            >>> print(features['n_samples'])
            1000
            >>> print(features['mean_nodes'])
            25.5
        """
        extractor = MetaFeatureExtractor(config)
        return extractor.extract_features(dataset)

    # =========================================================================
    # INSTANCE EXTRACTION METHOD
    # =========================================================================

    def extract_features(self, dataset) -> dict[str, float]:
        """
        Extract meta-features from dataset using instance configuration.

        Args:
            dataset: PyG Dataset or list of Data objects

        Returns:
            Dictionary of meta-feature name to float value
        """
        features: dict[str, float] = {}

        # Validate dataset
        if dataset is None or len(dataset) == 0:
            logger.warning("Empty dataset provided for meta-feature extraction")
            return features

        # Determine number of samples to analyze
        n_samples = len(dataset)
        if self.config.max_samples is not None:
            n_samples = min(n_samples, self.config.max_samples)

        # Extract features by category
        if self.config.should_extract(MetaFeatureCategory.STATISTICAL):
            features.update(self._extract_statistical_features(dataset, n_samples))

        if self.config.should_extract(MetaFeatureCategory.GRAPH):
            features.update(self._extract_graph_features(dataset, n_samples))

        if self.config.should_extract(MetaFeatureCategory.TARGET):
            features.update(self._extract_target_features(dataset, n_samples))

        if self.config.should_extract(MetaFeatureCategory.NODE_FEATURES):
            features.update(self._extract_node_feature_statistics(dataset, n_samples))

        if self.config.should_extract(MetaFeatureCategory.EDGE_FEATURES):
            features.update(self._extract_edge_feature_statistics(dataset, n_samples))

        if (
            self.config.should_extract(MetaFeatureCategory.MOLECULAR)
            and self.config.include_molecular
        ):
            features.update(self._extract_molecular_features(dataset, n_samples))

        # Normalize if configured
        if self.config.normalize:
            features = self._normalize_features(features)

        return features

    # =========================================================================
    # STATISTICAL FEATURES
    # =========================================================================

    def _extract_statistical_features(self, dataset, n_samples: int) -> dict[str, float]:
        """
        Extract basic statistical meta-features.

        Features:
        - n_samples: Total number of samples in dataset
        - n_features: Number of node features (if available)
        - n_edge_features: Number of edge features (if available)

        Args:
            dataset: PyG dataset
            n_samples: Number of samples to analyze

        Returns:
            Dictionary of statistical features
        """
        features: dict[str, float] = {}

        # Dataset size
        features["n_samples"] = float(len(dataset))

        # Get first sample for dimension inference
        try:
            first_data = dataset[0]

            # Node features dimension
            if hasattr(first_data, "x") and first_data.x is not None:
                x = first_data.x
                if hasattr(x, "shape"):
                    features["n_features"] = float(x.shape[1] if len(x.shape) > 1 else 1)
                elif hasattr(x, "size"):
                    features["n_features"] = float(x.size(1) if x.dim() > 1 else 1)

            # Edge features dimension
            if hasattr(first_data, "edge_attr") and first_data.edge_attr is not None:
                edge_attr = first_data.edge_attr
                if hasattr(edge_attr, "shape"):
                    features["n_edge_features"] = float(
                        edge_attr.shape[1] if len(edge_attr.shape) > 1 else 1
                    )
                elif hasattr(edge_attr, "size"):
                    features["n_edge_features"] = float(
                        edge_attr.size(1) if edge_attr.dim() > 1 else 1
                    )

        except Exception as e:
            logger.debug(f"Error extracting statistical features: {e}")

        return features

    # =========================================================================
    # GRAPH FEATURES
    # =========================================================================

    def _extract_graph_features(self, dataset, n_samples: int) -> dict[str, float]:
        """
        Extract graph structure meta-features.

        Features:
        - mean_nodes: Average number of nodes per graph
        - std_nodes: Standard deviation of node counts
        - min_nodes / max_nodes: Node count range
        - mean_edges: Average number of edges per graph
        - std_edges: Standard deviation of edge counts
        - mean_density: Average graph density
        - mean_degree: Average node degree across all graphs
        - std_degree: Standard deviation of degree distribution
        - min_degree / max_degree: Degree range
        - clustering_coefficient: Average clustering (if compute_expensive=True)

        Args:
            dataset: PyG dataset
            n_samples: Number of samples to analyze

        Returns:
            Dictionary of graph features
        """
        features: dict[str, float] = {}

        node_counts: list[int] = []
        edge_counts: list[int] = []
        densities: list[float] = []
        all_degrees: list[float] = []

        for i in range(n_samples):
            try:
                data = dataset[i]

                # Node count
                n_nodes = self._get_num_nodes(data)
                if n_nodes is not None and n_nodes > 0:
                    node_counts.append(n_nodes)

                # Edge count
                n_edges = self._get_num_edges(data)
                if n_edges is not None:
                    edge_counts.append(n_edges)

                # Density
                if n_nodes is not None and n_nodes > 1 and n_edges is not None:
                    max_edges = n_nodes * (n_nodes - 1)
                    if max_edges > 0:
                        densities.append(n_edges / max_edges)

                # Degree statistics
                degrees = self._compute_degrees(data, n_nodes)
                if degrees is not None and len(degrees) > 0:
                    all_degrees.extend(degrees)

            except Exception as e:
                logger.debug(f"Error processing graph {i}: {e}")
                continue

        # Compute statistics
        if node_counts:
            features["mean_nodes"] = float(np.mean(node_counts))
            features["std_nodes"] = float(np.std(node_counts))
            features["min_nodes"] = float(np.min(node_counts))
            features["max_nodes"] = float(np.max(node_counts))

        if edge_counts:
            features["mean_edges"] = float(np.mean(edge_counts))
            features["std_edges"] = float(np.std(edge_counts))
            features["min_edges"] = float(np.min(edge_counts))
            features["max_edges"] = float(np.max(edge_counts))

        if densities:
            features["mean_density"] = float(np.mean(densities))
            features["std_density"] = float(np.std(densities))

        if all_degrees:
            features["mean_degree"] = float(np.mean(all_degrees))
            features["std_degree"] = float(np.std(all_degrees))
            features["min_degree"] = float(np.min(all_degrees))
            features["max_degree"] = float(np.max(all_degrees))

            # Degree distribution percentiles
            features["degree_25th"] = float(np.percentile(all_degrees, 25))
            features["degree_50th"] = float(np.percentile(all_degrees, 50))
            features["degree_75th"] = float(np.percentile(all_degrees, 75))

        # Expensive features
        if self.config.compute_expensive and node_counts:
            clustering = self._compute_clustering_coefficients(dataset, n_samples)
            if clustering is not None:
                features["clustering_coefficient"] = clustering

        return features

    def _get_num_nodes(self, data) -> int | None:
        """Get number of nodes from a PyG Data object."""
        if hasattr(data, "num_nodes") and data.num_nodes is not None:
            return int(data.num_nodes)
        if hasattr(data, "x") and data.x is not None:
            if hasattr(data.x, "shape"):
                return int(data.x.shape[0])
            elif hasattr(data.x, "size"):
                return int(data.x.size(0))
        return None

    def _get_num_edges(self, data) -> int | None:
        """Get number of edges from a PyG Data object."""
        if hasattr(data, "edge_index") and data.edge_index is not None:
            if hasattr(data.edge_index, "shape"):
                return int(data.edge_index.shape[1])
            elif hasattr(data.edge_index, "size"):
                return int(data.edge_index.size(1))
        return None

    def _compute_degrees(self, data, n_nodes: int | None) -> list[float] | None:
        """Compute node degrees for a graph."""
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return None

        if n_nodes is None or n_nodes == 0:
            return None

        try:
            edge_index = data.edge_index

            # Use torch_geometric degree function if available
            if self._degree_fn is not None and self._torch is not None:
                deg = self._degree_fn(edge_index[0], num_nodes=n_nodes)
                return deg.tolist()

            # Fallback: manual degree computation
            if hasattr(edge_index, "numpy"):
                edge_index_np = edge_index.numpy()
            else:
                edge_index_np = np.array(edge_index)

            degrees = np.zeros(n_nodes)
            np.add.at(degrees, edge_index_np[0], 1)
            return degrees.tolist()

        except Exception as e:
            logger.debug(f"Error computing degrees: {e}")
            return None

    def _compute_clustering_coefficients(self, dataset, n_samples: int) -> float | None:
        """
        Compute average local clustering coefficient.

        This is an expensive operation that computes the clustering
        coefficient for each node and averages across the dataset.
        """
        try:
            clustering_coeffs: list[float] = []

            for i in range(min(n_samples, 100)):  # Limit for performance
                data = dataset[i]
                n_nodes = self._get_num_nodes(data)

                if n_nodes is None or n_nodes < 3:
                    continue

                if not hasattr(data, "edge_index") or data.edge_index is None:
                    continue

                # Build adjacency for clustering computation
                edge_index = data.edge_index
                if hasattr(edge_index, "numpy"):
                    edges = edge_index.numpy()
                else:
                    edges = np.array(edge_index)

                # Build adjacency set for each node
                adj: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
                for src, dst in zip(edges[0], edges[1]):
                    adj[int(src)].add(int(dst))

                # Compute local clustering for each node
                local_clustering: list[float] = []
                for node in range(n_nodes):
                    neighbors = adj[node]
                    k = len(neighbors)
                    if k < 2:
                        continue

                    # Count edges between neighbors
                    triangles = 0
                    neighbors_list = list(neighbors)
                    for j, n1 in enumerate(neighbors_list):
                        for n2 in neighbors_list[j + 1 :]:
                            if n2 in adj[n1]:
                                triangles += 1

                    # Local clustering coefficient
                    max_triangles = k * (k - 1) / 2
                    if max_triangles > 0:
                        local_clustering.append(triangles / max_triangles)

                if local_clustering:
                    clustering_coeffs.append(float(np.mean(local_clustering)))

            if clustering_coeffs:
                return float(np.mean(clustering_coeffs))
            return None

        except Exception as e:
            logger.debug(f"Error computing clustering coefficient: {e}")
            return None

    # =========================================================================
    # TARGET FEATURES
    # =========================================================================

    def _extract_target_features(self, dataset, n_samples: int) -> dict[str, float]:
        """
        Extract target variable meta-features.

        Features:
        - target_mean: Mean of target values
        - target_std: Standard deviation of targets
        - target_min / target_max: Target range
        - target_range: Max - Min
        - target_skewness: Skewness of target distribution
        - target_dim: Dimensionality of target (1 for scalar)

        Args:
            dataset: PyG dataset
            n_samples: Number of samples to analyze

        Returns:
            Dictionary of target features
        """
        features: dict[str, float] = {}
        targets: list[float] = []
        target_dim: int | None = None

        for i in range(n_samples):
            try:
                data = dataset[i]

                if not hasattr(data, "y") or data.y is None:
                    continue

                y = data.y

                # Get target dimension
                if target_dim is None:
                    if hasattr(y, "numel"):
                        target_dim = y.numel()
                    elif hasattr(y, "size"):
                        target_dim = int(np.prod(y.size()))
                    else:
                        target_dim = 1

                # Extract scalar value for statistics
                if hasattr(y, "numel") and y.numel() == 1:
                    targets.append(float(y.item()))
                elif hasattr(y, "mean"):
                    targets.append(float(y.mean().item()))
                elif hasattr(y, "__float__"):
                    targets.append(float(y))

            except Exception as e:
                logger.debug(f"Error extracting target from sample {i}: {e}")
                continue

        if targets:
            features["target_mean"] = float(np.mean(targets))
            features["target_std"] = float(np.std(targets))
            features["target_min"] = float(np.min(targets))
            features["target_max"] = float(np.max(targets))
            features["target_range"] = features["target_max"] - features["target_min"]

            # Skewness
            if len(targets) > 2 and features["target_std"] > 0:
                centered = np.array(targets) - features["target_mean"]
                features["target_skewness"] = float(
                    np.mean(centered**3) / (features["target_std"] ** 3)
                )

        if target_dim is not None:
            features["target_dim"] = float(target_dim)

        return features

    # =========================================================================
    # NODE FEATURE STATISTICS
    # =========================================================================

    def _extract_node_feature_statistics(self, dataset, n_samples: int) -> dict[str, float]:
        """
        Extract statistics about node features.

        Features:
        - node_feat_mean: Mean of all node features
        - node_feat_std: Standard deviation
        - node_feat_min / node_feat_max: Range
        - node_feat_sparsity: Fraction of zero values

        Args:
            dataset: PyG dataset
            n_samples: Number of samples to analyze

        Returns:
            Dictionary of node feature statistics
        """
        features: dict[str, float] = {}
        all_values: list[float] = []
        zero_count = 0
        total_count = 0

        for i in range(min(n_samples, 100)):  # Limit for memory
            try:
                data = dataset[i]

                if not hasattr(data, "x") or data.x is None:
                    continue

                x = data.x

                # Flatten and collect values
                if hasattr(x, "flatten"):
                    flat = x.flatten()
                    if hasattr(flat, "tolist"):
                        values = flat.tolist()
                    elif hasattr(flat, "numpy"):
                        values = flat.numpy().tolist()
                    else:
                        values = list(flat)
                else:
                    values = list(np.array(x).flatten())

                # Sample for memory efficiency
                if len(values) > 1000:
                    indices = np.random.choice(len(values), 1000, replace=False)
                    values = [values[j] for j in indices]

                all_values.extend(values)
                zero_count += sum(1 for v in values if abs(v) < 1e-10)
                total_count += len(values)

            except Exception as e:
                logger.debug(f"Error extracting node features from sample {i}: {e}")
                continue

        if all_values:
            features["node_feat_mean"] = float(np.mean(all_values))
            features["node_feat_std"] = float(np.std(all_values))
            features["node_feat_min"] = float(np.min(all_values))
            features["node_feat_max"] = float(np.max(all_values))

            if total_count > 0:
                features["node_feat_sparsity"] = float(zero_count / total_count)

        return features

    # =========================================================================
    # EDGE FEATURE STATISTICS
    # =========================================================================

    def _extract_edge_feature_statistics(self, dataset, n_samples: int) -> dict[str, float]:
        """
        Extract statistics about edge features.

        Features:
        - edge_feat_mean: Mean of all edge features
        - edge_feat_std: Standard deviation
        - edge_feat_min / edge_feat_max: Range
        - has_edge_features: 1.0 if edge features present, 0.0 otherwise

        Args:
            dataset: PyG dataset
            n_samples: Number of samples to analyze

        Returns:
            Dictionary of edge feature statistics
        """
        features: dict[str, float] = {}
        all_values: list[float] = []
        has_edge_features = False

        for i in range(min(n_samples, 100)):
            try:
                data = dataset[i]

                if not hasattr(data, "edge_attr") or data.edge_attr is None:
                    continue

                has_edge_features = True
                edge_attr = data.edge_attr

                # Flatten and collect values
                if hasattr(edge_attr, "flatten"):
                    flat = edge_attr.flatten()
                    if hasattr(flat, "tolist"):
                        values = flat.tolist()
                    elif hasattr(flat, "numpy"):
                        values = flat.numpy().tolist()
                    else:
                        values = list(flat)
                else:
                    values = list(np.array(edge_attr).flatten())

                # Sample for memory efficiency
                if len(values) > 1000:
                    indices = np.random.choice(len(values), 1000, replace=False)
                    values = [values[j] for j in indices]

                all_values.extend(values)

            except Exception as e:
                logger.debug(f"Error extracting edge features from sample {i}: {e}")
                continue

        features["has_edge_features"] = 1.0 if has_edge_features else 0.0

        if all_values:
            features["edge_feat_mean"] = float(np.mean(all_values))
            features["edge_feat_std"] = float(np.std(all_values))
            features["edge_feat_min"] = float(np.min(all_values))
            features["edge_feat_max"] = float(np.max(all_values))

        return features

    # =========================================================================
    # MOLECULAR FEATURES
    # =========================================================================

    def _extract_molecular_features(self, dataset, n_samples: int) -> dict[str, float]:
        """
        Extract molecular-specific meta-features.

        These features are extracted from molecular data when available:
        - Atom type distribution (fraction of C, N, O, etc.)
        - Bond type distribution (single, double, triple, aromatic)
        - Molecular weight statistics
        - Ring count statistics
        - Charge statistics (if available)

        Args:
            dataset: PyG dataset
            n_samples: Number of samples to analyze

        Returns:
            Dictionary of molecular features
        """
        features: dict[str, float] = {}

        # Atom type counts
        atom_counts: dict[int, int] = {}
        total_atoms = 0

        # Bond type counts (if available)
        bond_type_counts: dict[int, int] = {}
        total_bonds = 0

        # Molecular properties
        mol_weights: list[float] = []
        ring_counts: list[int] = []

        for i in range(n_samples):
            try:
                data = dataset[i]

                # Extract atomic numbers from z attribute or node features
                atomic_numbers = self._get_atomic_numbers(data)
                if atomic_numbers is not None:
                    for z in atomic_numbers:
                        atom_counts[z] = atom_counts.get(z, 0) + 1
                        total_atoms += 1

                # Extract bond types if available
                bond_types = self._get_bond_types(data)
                if bond_types is not None:
                    for bt in bond_types:
                        bond_type_counts[bt] = bond_type_counts.get(bt, 0) + 1
                        total_bonds += 1

                # Molecular weight (if SMILES or mol available)
                mol_weight = self._get_molecular_weight(data)
                if mol_weight is not None:
                    mol_weights.append(mol_weight)

                # Ring count (if available)
                ring_count = self._get_ring_count(data)
                if ring_count is not None:
                    ring_counts.append(ring_count)

            except Exception as e:
                logger.debug(f"Error extracting molecular features from sample {i}: {e}")
                continue

        # Atom type fractions
        if total_atoms > 0:
            # Common atoms: H=1, C=6, N=7, O=8, F=9, S=16, Cl=17, Br=35
            common_atoms = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}
            for z, name in common_atoms.items():
                count = atom_counts.get(z, 0)
                features[f"atom_frac_{name}"] = float(count / total_atoms)

            # Heavy atom ratio (non-hydrogen)
            h_count = atom_counts.get(1, 0)
            features["heavy_atom_ratio"] = float((total_atoms - h_count) / total_atoms)

            # Number of unique atom types
            features["n_atom_types"] = float(len(atom_counts))

        # Bond type fractions
        if total_bonds > 0:
            # Bond types: 1=single, 2=double, 3=triple, 12=aromatic
            bond_names = {1: "single", 2: "double", 3: "triple", 12: "aromatic"}
            for bt, name in bond_names.items():
                count = bond_type_counts.get(bt, 0)
                features[f"bond_frac_{name}"] = float(count / total_bonds)

        # Molecular weight statistics
        if mol_weights:
            features["mol_weight_mean"] = float(np.mean(mol_weights))
            features["mol_weight_std"] = float(np.std(mol_weights))
            features["mol_weight_min"] = float(np.min(mol_weights))
            features["mol_weight_max"] = float(np.max(mol_weights))

        # Ring count statistics
        if ring_counts:
            features["ring_count_mean"] = float(np.mean(ring_counts))
            features["ring_count_std"] = float(np.std(ring_counts))

        return features

    def _get_atomic_numbers(self, data) -> list[int] | None:
        """Extract atomic numbers from data object."""
        # Try z attribute (common for molecular graphs)
        if hasattr(data, "z") and data.z is not None:
            z = data.z
            if hasattr(z, "tolist"):
                return z.tolist()
            elif hasattr(z, "numpy"):
                return z.numpy().tolist()
            return list(z)

        # Try atomic_numbers attribute
        if hasattr(data, "atomic_numbers") and data.atomic_numbers is not None:
            an = data.atomic_numbers
            if hasattr(an, "tolist"):
                return an.tolist()
            return list(an)

        return None

    def _get_bond_types(self, data) -> list[int] | None:
        """Extract bond types from data object."""
        # Try edge_attr (first column often is bond type)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_attr = data.edge_attr

            # Check if it looks like one-hot encoded bond types
            if hasattr(edge_attr, "shape") and len(edge_attr.shape) > 1:
                if edge_attr.shape[1] <= 5:  # Likely one-hot bond types
                    if hasattr(edge_attr, "argmax"):
                        return (edge_attr.argmax(dim=1) + 1).tolist()

            # First column might be bond type
            if hasattr(edge_attr, "shape") and len(edge_attr.shape) == 2:
                first_col = edge_attr[:, 0]
                if hasattr(first_col, "tolist"):
                    return [int(x) for x in first_col.tolist()]

        # Try bond_type attribute
        if hasattr(data, "bond_type") and data.bond_type is not None:
            bt = data.bond_type
            if hasattr(bt, "tolist"):
                return bt.tolist()
            return list(bt)

        return None

    def _get_molecular_weight(self, data) -> float | None:
        """Get molecular weight from data object or compute from atomic numbers."""
        # Try mol_weight attribute
        if hasattr(data, "mol_weight") and data.mol_weight is not None:
            return float(data.mol_weight)

        # Compute from atomic numbers
        atomic_numbers = self._get_atomic_numbers(data)
        if atomic_numbers:
            # Approximate atomic weights for common elements
            atomic_weights = {
                1: 1.008,
                6: 12.011,
                7: 14.007,
                8: 15.999,
                9: 18.998,
                15: 30.974,
                16: 32.065,
                17: 35.453,
                35: 79.904,
                53: 126.904,
            }
            weight = sum(atomic_weights.get(z, 0) for z in atomic_numbers)
            if weight > 0:
                return weight

        return None

    def _get_ring_count(self, data) -> int | None:
        """Get ring count from data object."""
        if hasattr(data, "num_rings") and data.num_rings is not None:
            return int(data.num_rings)
        if hasattr(data, "ring_count") and data.ring_count is not None:
            return int(data.ring_count)
        return None

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _normalize_features(self, features: dict[str, float]) -> dict[str, float]:
        """
        Normalize features to [0, 1] range for similarity computation.

        Uses min-max normalization with predefined bounds for known features
        and z-score normalization for unknown features.
        """
        normalized: dict[str, float] = {}

        for name, value in features.items():
            if name in self._NORMALIZATION_BOUNDS:
                low, high = self._NORMALIZATION_BOUNDS[name]
                if high > low:
                    normalized[name] = (value - low) / (high - low)
                    normalized[name] = max(0.0, min(1.0, normalized[name]))
                else:
                    normalized[name] = 0.5
            else:
                # Keep original value for unknown features
                normalized[name] = value

        return normalized

    # =========================================================================
    # SIMILARITY COMPUTATION
    # =========================================================================

    @staticmethod
    def compute_similarity(features_a: dict[str, float], features_b: dict[str, float]) -> float:
        """
        Compute cosine similarity between two meta-feature vectors.

        This method matches the similarity computation in HPOTransferManager
        (MILIA_HPO_Implementation_Blueprint.md:4051-4070).

        Args:
            features_a: Meta-features from first dataset
            features_b: Meta-features from second dataset

        Returns:
            Cosine similarity in range [0, 1] (0 if no common features)

        Examples:
            >>> sim = MetaFeatureExtractor.compute_similarity(
            ...     {'n_samples': 1000, 'mean_nodes': 25},
            ...     {'n_samples': 1200, 'mean_nodes': 28}
            ... )
            >>> print(f"Similarity: {sim:.3f}")
        """
        # Find common keys
        common_keys = set(features_a.keys()) & set(features_b.keys())

        if not common_keys:
            return 0.0

        # Build vectors
        vec_a = np.array([features_a[k] for k in common_keys])
        vec_b = np.array([features_b[k] for k in common_keys])

        # Compute cosine similarity
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

        # Ensure result is in valid range
        return max(0.0, min(1.0, similarity))

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def get_feature_names() -> dict[str, list[str]]:
        """
        Get names of all possible meta-features by category.

        Returns:
            Dictionary mapping category names to feature name lists
        """
        return {
            "statistical": ["n_samples", "n_features", "n_edge_features"],
            "graph": [
                "mean_nodes",
                "std_nodes",
                "min_nodes",
                "max_nodes",
                "mean_edges",
                "std_edges",
                "min_edges",
                "max_edges",
                "mean_density",
                "std_density",
                "mean_degree",
                "std_degree",
                "min_degree",
                "max_degree",
                "degree_25th",
                "degree_50th",
                "degree_75th",
                "clustering_coefficient",
            ],
            "target": [
                "target_mean",
                "target_std",
                "target_min",
                "target_max",
                "target_range",
                "target_skewness",
                "target_dim",
            ],
            "node_features": [
                "node_feat_mean",
                "node_feat_std",
                "node_feat_min",
                "node_feat_max",
                "node_feat_sparsity",
            ],
            "edge_features": [
                "has_edge_features",
                "edge_feat_mean",
                "edge_feat_std",
                "edge_feat_min",
                "edge_feat_max",
            ],
            "molecular": [
                "atom_frac_H",
                "atom_frac_C",
                "atom_frac_N",
                "atom_frac_O",
                "atom_frac_F",
                "atom_frac_S",
                "atom_frac_Cl",
                "atom_frac_Br",
                "heavy_atom_ratio",
                "n_atom_types",
                "bond_frac_single",
                "bond_frac_double",
                "bond_frac_triple",
                "bond_frac_aromatic",
                "mol_weight_mean",
                "mol_weight_std",
                "mol_weight_min",
                "mol_weight_max",
                "ring_count_mean",
                "ring_count_std",
            ],
        }

    @staticmethod
    def get_category_for_feature(feature_name: str) -> str | None:
        """
        Get the category for a given feature name.

        Args:
            feature_name: Name of the meta-feature

        Returns:
            Category name or None if not found
        """
        all_features = MetaFeatureExtractor.get_feature_names()
        for category, features in all_features.items():
            if feature_name in features:
                return category
        return None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "MetaFeatureCategory",
    "MetaFeatureConfig",
    "MetaFeatureExtractor",
]
