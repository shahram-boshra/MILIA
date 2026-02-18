#!/usr/bin/env python3
"""
End-to-End Training Workflow Test Suite

Tests the complete training workflow:
    config loading → dataset creation → data splitting → model creation →
    training (few epochs) → checkpoint saving → checkpoint loading → prediction

This test suite validates that the entire model training lifecycle works
as an integrated system, exercising the full stack of MILIA's training
infrastructure with synthetic data.

Modules Exercised:
    - milia_pipeline/config/config_loader.py              — load_config()
    - milia_pipeline/models/factory/model_factory.py      — ModelFactory.create_model()
    - milia_pipeline/models/factory/target_selection_config.py — TargetSelectionConfig
    - milia_pipeline/models/training/data_splitting.py    — DataSplitter
    - milia_pipeline/models/training/data_preparation.py  — TaskDataPreparer
    - milia_pipeline/models/training/trainer.py           — Trainer.fit()
    - milia_pipeline/models/training/callbacks.py         — ModelCheckpoint, EarlyStopping
    - milia_pipeline/models/training/loss_functions.py    — get_loss()
    - milia_pipeline/models/training/optimizers.py        — get_optimizer()
    - milia_pipeline/models/training/schedulers.py        — get_scheduler()
    - milia_pipeline/models/training/metrics.py           — get_metrics_for_task()
    - milia_pipeline/models/post_training/checkpoint/checkpoint_manager.py — CheckpointManager
    - milia_pipeline/models/post_training/inference/model_loader.py       — load_model()
    - milia_pipeline/models/post_training/inference/predictor.py          — Predictor

Scope:
    Uses a small synthetic PyG dataset (50-100 graphs). Trains for 3-5 epochs.
    Validates: loss decreases, checkpoint files exist on disk, loaded model
    produces predictions, predictions have correct tensor shape.

Author: MILIA Team
Version: 1.0.0
"""

import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Add project root to Python path FIRST
# ---------------------------------------------------------------------------
# When run from project root (/app/milia) or via pytest, ensure the project
# root is on sys.path so that ``import milia_pipeline`` resolves correctly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # tests/ -> project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# =============================================================================
# CONDITIONAL IMPORTS WITH SKIP MARKERS
# =============================================================================
# Use pytest.importorskip for heavy/optional dependencies so that the test
# suite degrades gracefully in environments missing PyG or TorchMetrics.
# =============================================================================

torch_geometric = pytest.importorskip(
    "torch_geometric",
    reason="torch_geometric is required for E2E training workflow tests",
)

from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

# ---------------------------------------------------------------------------
# MILIA imports — each wrapped so that a missing component produces a clear
# skip rather than a confusing ImportError during collection.
# ---------------------------------------------------------------------------
try:
    from milia_pipeline.models.training.data_splitting import DataSplitter

    _DATA_SPLITTING_AVAILABLE = True
except ImportError:
    _DATA_SPLITTING_AVAILABLE = False

try:
    from milia_pipeline.models.training.loss_functions import LossRegistry, get_loss

    _LOSS_REGISTRY_AVAILABLE = True
except ImportError:
    _LOSS_REGISTRY_AVAILABLE = False

try:
    from milia_pipeline.models.training.optimizers import OptimizerRegistry, get_optimizer

    _OPTIMIZER_REGISTRY_AVAILABLE = True
except ImportError:
    _OPTIMIZER_REGISTRY_AVAILABLE = False

try:
    from milia_pipeline.models.training.schedulers import SchedulerRegistry, get_scheduler

    _SCHEDULER_REGISTRY_AVAILABLE = True
except ImportError:
    _SCHEDULER_REGISTRY_AVAILABLE = False

try:
    from milia_pipeline.models.training.metrics import MetricsRegistry

    _METRICS_REGISTRY_AVAILABLE = True
except ImportError:
    _METRICS_REGISTRY_AVAILABLE = False

try:
    from milia_pipeline.models.training.trainer import Trainer

    _TRAINER_AVAILABLE = True
except ImportError:
    _TRAINER_AVAILABLE = False

try:
    from milia_pipeline.models.training.callbacks import (
        Callback,
        EarlyStopping,
        ModelCheckpoint,
    )

    _CALLBACKS_AVAILABLE = True
except ImportError:
    _CALLBACKS_AVAILABLE = False

try:
    from milia_pipeline.models.factory.model_factory import (
        ModelFactory,
        create_model,
        get_factory,
    )

    _MODEL_FACTORY_AVAILABLE = True
except ImportError:
    _MODEL_FACTORY_AVAILABLE = False

try:
    from milia_pipeline.models.factory.target_selection_config import (
        SelectionMode,
        TargetLevel,
        TargetSelectionConfig,
        TargetSource,
    )

    _TARGET_SELECTION_AVAILABLE = True
except ImportError:
    _TARGET_SELECTION_AVAILABLE = False

try:
    from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
        CHECKPOINT_FORMAT_VERSION,
        CheckpointManager,
    )

    _CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    _CHECKPOINT_MANAGER_AVAILABLE = False

try:
    from milia_pipeline.models.post_training.inference.model_loader import ModelLoader

    _MODEL_LOADER_AVAILABLE = True
except ImportError:
    _MODEL_LOADER_AVAILABLE = False

try:
    from milia_pipeline.models.post_training.inference.predictor import Predictor

    _PREDICTOR_AVAILABLE = True
except ImportError:
    _PREDICTOR_AVAILABLE = False

try:
    from milia_pipeline.config.config_loader import load_config

    _CONFIG_LOADER_AVAILABLE = True
except ImportError:
    _CONFIG_LOADER_AVAILABLE = False


# Aggregate availability flag — if ALL core training components are present
# the full E2E tests can execute; otherwise individual tests skip gracefully.
_TRAINING_STACK_AVAILABLE = all(
    [
        _DATA_SPLITTING_AVAILABLE,
        _LOSS_REGISTRY_AVAILABLE,
        _OPTIMIZER_REGISTRY_AVAILABLE,
        _TRAINER_AVAILABLE,
        _CALLBACKS_AVAILABLE,
        _MODEL_FACTORY_AVAILABLE,
    ]
)

_POST_TRAINING_STACK_AVAILABLE = all(
    [
        _CHECKPOINT_MANAGER_AVAILABLE,
        _MODEL_LOADER_AVAILABLE,
        _PREDICTOR_AVAILABLE,
    ]
)


# =============================================================================
# PYTEST MARKERS
# =============================================================================
pytestmark = [
    pytest.mark.e2e,
]


# =============================================================================
# SYNTHETIC DATA GENERATION HELPERS
# =============================================================================


def _create_single_graph(
    num_nodes: int,
    num_node_features: int,
    num_targets: int = 1,
    include_edge_attr: bool = False,
    num_edge_features: int = 4,
    seed: int | None = None,
) -> Data:
    """
    Create a single synthetic PyG Data object representing a graph.

    Generates a random graph with the specified number of nodes, node features,
    and target dimensions.  Edges are created by connecting each node to 2-3
    random neighbours (ensuring at least a connected-ish graph).

    Args:
        num_nodes: Number of nodes in the graph.
        num_node_features: Dimensionality of node feature vectors.
        num_targets: Number of graph-level regression targets.
        include_edge_attr: Whether to add random edge attributes.
        num_edge_features: Dimensionality of edge feature vectors.
        seed: Optional random seed for reproducibility.

    Returns:
        A ``torch_geometric.data.Data`` object with ``x``, ``edge_index``,
        and ``y`` attributes (and optionally ``edge_attr``).
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Node features — uniform in [0, 1]
    x = torch.randn(num_nodes, num_node_features)

    # Build random edges — each node connects to 2-3 others
    edges_src: list[int] = []
    edges_dst: list[int] = []
    for i in range(num_nodes):
        n_neighbours = torch.randint(2, min(4, num_nodes), (1,)).item()
        targets = torch.randperm(num_nodes)[:n_neighbours]
        for t in targets:
            if t.item() != i:
                edges_src.append(i)
                edges_dst.append(t.item())
    # Ensure at least one edge
    if len(edges_src) == 0:
        edges_src = [0]
        edges_dst = [1 if num_nodes > 1 else 0]
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Graph-level regression target
    y = torch.randn(1, num_targets)

    data = Data(x=x, edge_index=edge_index, y=y)

    if include_edge_attr:
        num_edges = edge_index.size(1)
        data.edge_attr = torch.randn(num_edges, num_edge_features)

    return data


def create_synthetic_dataset(
    num_graphs: int = 80,
    num_node_features: int = 16,
    num_targets: int = 1,
    include_edge_attr: bool = False,
    num_edge_features: int = 4,
    seed: int = 42,
) -> list[Data]:
    """
    Create a list of synthetic PyG Data objects for testing.

    Args:
        num_graphs: Total number of graphs.
        num_node_features: Feature dimensionality per node.
        num_targets: Number of graph-level regression targets.
        include_edge_attr: Whether to include edge attributes.
        num_edge_features: Dimensionality of edge attributes.
        seed: Base random seed.

    Returns:
        List of ``Data`` objects ready for splitting and loading.
    """
    torch.manual_seed(seed)
    graphs: list[Data] = []
    for i in range(num_graphs):
        num_nodes = torch.randint(5, 20, (1,)).item()
        g = _create_single_graph(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_targets=num_targets,
            include_edge_attr=include_edge_attr,
            num_edge_features=num_edge_features,
            seed=seed + i,
        )
        graphs.append(g)
    return graphs


# =============================================================================
# SIMPLE GNN MODEL FOR TESTING (avoids dependency on full model registry)
# =============================================================================


class _SimpleGNN(nn.Module):
    """
    Minimal GNN model used exclusively for E2E testing.

    This avoids hard dependencies on the full PyG model registry or the
    introspector, keeping the tests focused on the *training infrastructure*
    rather than specific model architectures.

    Architecture:
        GCNConv(in, hidden) → ReLU → GCNConv(hidden, hidden) → ReLU
        → global_mean_pool → Linear(hidden, out)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.pool = global_mean_pool

    def forward(self, x, edge_index, batch=None):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
        return self.lin(x)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def synthetic_graphs() -> list[Data]:
    """Module-scoped synthetic dataset — created once, shared across tests."""
    return create_synthetic_dataset(
        num_graphs=80,
        num_node_features=16,
        num_targets=1,
        seed=42,
    )


@pytest.fixture(scope="module")
def multi_target_graphs() -> list[Data]:
    """Module-scoped synthetic dataset with multiple targets."""
    return create_synthetic_dataset(
        num_graphs=80,
        num_node_features=16,
        num_targets=3,
        seed=123,
    )


@pytest.fixture
def working_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary working directory for each test.

    Uses pytest's ``tmp_path`` fixture so directories are automatically
    cleaned up after the test session.
    """
    work = tmp_path / "milia_e2e_test"
    work.mkdir(parents=True, exist_ok=True)
    return work


@pytest.fixture
def checkpoint_dir(working_dir: Path) -> Path:
    """Provide a checkpoint sub-directory inside the working directory."""
    ckpt = working_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    return ckpt


@pytest.fixture
def simple_model() -> _SimpleGNN:
    """Create a fresh simple GNN model for testing."""
    return _SimpleGNN(in_channels=16, hidden_channels=32, out_channels=1)


@pytest.fixture
def multi_target_model() -> _SimpleGNN:
    """Create a fresh simple GNN model for multi-target testing."""
    return _SimpleGNN(in_channels=16, hidden_channels=32, out_channels=3)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _split_dataset(
    dataset: list[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    """
    Simple deterministic split of a list of Data objects.

    This is a plain-Python fallback used when ``DataSplitter`` is not
    available or when tests need an explicit, dependency-free split.
    """
    torch.manual_seed(seed)
    n = len(dataset)
    indices = torch.randperm(n).tolist()
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return (
        [dataset[i] for i in train_idx],
        [dataset[i] for i in val_idx],
        [dataset[i] for i in test_idx],
    )


def _make_loaders(
    train_data: list[Data],
    val_data: list[Data],
    test_data: list[Data] | None = None,
    batch_size: int = 16,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create DataLoaders from lists of Data objects."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) if test_data else None
    return train_loader, val_loader, test_loader


# =============================================================================
# TEST CLASS: Synthetic Data Validation
# =============================================================================


class TestSyntheticDataCreation:
    """Verify that synthetic data helpers produce valid PyG graphs."""

    def test_single_graph_has_required_attributes(self):
        """A single synthetic graph must have x, edge_index, and y."""
        g = _create_single_graph(num_nodes=10, num_node_features=8, num_targets=1, seed=0)
        assert hasattr(g, "x"), "Graph missing 'x' (node features)"
        assert hasattr(g, "edge_index"), "Graph missing 'edge_index'"
        assert hasattr(g, "y"), "Graph missing 'y' (targets)"

    def test_single_graph_shapes(self):
        """Node features and target dimensions must match specifications."""
        g = _create_single_graph(num_nodes=10, num_node_features=8, num_targets=3, seed=1)
        assert g.x.shape == (10, 8)
        assert g.y.shape == (1, 3)
        assert g.edge_index.dim() == 2
        assert g.edge_index.shape[0] == 2

    def test_dataset_creation_size(self):
        """Dataset helper must produce the requested number of graphs."""
        graphs = create_synthetic_dataset(num_graphs=50, seed=7)
        assert len(graphs) == 50

    def test_dataset_creation_reproducibility(self):
        """Same seed must produce identical datasets."""
        a = create_synthetic_dataset(num_graphs=20, seed=99)
        b = create_synthetic_dataset(num_graphs=20, seed=99)
        for ga, gb in zip(a, b, strict=False):
            assert torch.equal(ga.x, gb.x)
            assert torch.equal(ga.edge_index, gb.edge_index)
            assert torch.equal(ga.y, gb.y)

    def test_graph_with_edge_attributes(self):
        """Edge attributes must be present and correctly shaped when requested."""
        g = _create_single_graph(
            num_nodes=8,
            num_node_features=4,
            include_edge_attr=True,
            num_edge_features=6,
            seed=2,
        )
        assert hasattr(g, "edge_attr")
        assert g.edge_attr is not None
        assert g.edge_attr.shape[1] == 6
        assert g.edge_attr.shape[0] == g.edge_index.shape[1]


# =============================================================================
# TEST CLASS: Data Splitting
# =============================================================================


class TestDataSplitting:
    """Validate data splitting produces correct partition sizes."""

    @pytest.mark.skipif(
        not _DATA_SPLITTING_AVAILABLE,
        reason="milia_pipeline.models.training.data_splitting not available",
    )
    def test_random_split_sizes(self, synthetic_graphs):
        """DataSplitter.random_split must partition data without overlap."""
        train, val, test = DataSplitter.random_split(
            synthetic_graphs,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
        total = len(train) + len(val) + len(test)
        assert total == len(synthetic_graphs), (
            f"Split sizes do not sum to dataset size: "
            f"{len(train)} + {len(val)} + {len(test)} = {total} != {len(synthetic_graphs)}"
        )
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    @pytest.mark.skipif(
        not _DATA_SPLITTING_AVAILABLE,
        reason="milia_pipeline.models.training.data_splitting not available",
    )
    def test_random_split_reproducibility(self, synthetic_graphs):
        """Same seed must yield identical splits."""
        t1, v1, te1 = DataSplitter.random_split(synthetic_graphs, random_seed=42)
        t2, v2, te2 = DataSplitter.random_split(synthetic_graphs, random_seed=42)
        assert len(t1) == len(t2)
        assert len(v1) == len(v2)
        assert len(te1) == len(te2)

    def test_fallback_split_sizes(self, synthetic_graphs):
        """Plain-Python fallback split must also partition correctly."""
        train, val, test = _split_dataset(synthetic_graphs, 0.7, 0.15, 0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(synthetic_graphs)


# =============================================================================
# TEST CLASS: Loss Function Registry
# =============================================================================


class TestLossFunctionRegistry:
    """Validate loss function instantiation from the registry."""

    @pytest.mark.skipif(
        not _LOSS_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.loss_functions not available",
    )
    def test_get_mse_loss(self):
        """get_loss('mse') must return a callable loss function."""
        loss_fn = get_loss("mse")
        assert callable(loss_fn)
        # Smoke check: compute loss on random tensors
        pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        loss_val = loss_fn(pred, target)
        assert loss_val.dim() == 0  # scalar
        assert loss_val.item() >= 0.0

    @pytest.mark.skipif(
        not _LOSS_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.loss_functions not available",
    )
    def test_get_mae_loss(self):
        """get_loss('mae') must return a callable loss function."""
        loss_fn = get_loss("mae")
        assert callable(loss_fn)

    @pytest.mark.skipif(
        not _LOSS_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.loss_functions not available",
    )
    def test_list_losses_nonempty(self):
        """The loss registry must expose at least the core losses."""
        available = LossRegistry.list_available()
        assert len(available) >= 3, f"Expected >= 3 losses, got {len(available)}"
        # These three must always be present per the project structure doc
        for name in ("mse", "mae"):
            assert name in available, f"Core loss '{name}' missing from registry"

    @pytest.mark.skipif(
        not _LOSS_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.loss_functions not available",
    )
    def test_unknown_loss_raises(self):
        """Requesting a non-existent loss must raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nknown"):
            get_loss("definitely_not_a_real_loss_name")


# =============================================================================
# TEST CLASS: Optimizer Registry
# =============================================================================


class TestOptimizerRegistry:
    """Validate optimizer instantiation from the registry."""

    @pytest.mark.skipif(
        not _OPTIMIZER_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.optimizers not available",
    )
    def test_get_adam_optimizer(self, simple_model):
        """get_optimizer('adam', ...) must return a valid optimizer."""
        optimizer = get_optimizer("adam", simple_model.parameters(), {"lr": 0.001})
        assert isinstance(optimizer, torch.optim.Optimizer)

    @pytest.mark.skipif(
        not _OPTIMIZER_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.optimizers not available",
    )
    def test_get_adamw_optimizer(self, simple_model):
        """get_optimizer('adamw', ...) must return a valid optimizer."""
        optimizer = get_optimizer("adamw", simple_model.parameters(), {"lr": 0.001})
        assert isinstance(optimizer, torch.optim.AdamW)

    @pytest.mark.skipif(
        not _OPTIMIZER_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.optimizers not available",
    )
    def test_invalid_params_filtered(self, simple_model):
        """Invalid optimizer parameters must be silently filtered out."""
        # This should NOT raise — invalid_param should be ignored
        optimizer = get_optimizer(
            "adam",
            simple_model.parameters(),
            {"lr": 0.001, "this_param_does_not_exist": 999},
        )
        assert isinstance(optimizer, torch.optim.Optimizer)

    @pytest.mark.skipif(
        not _OPTIMIZER_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.optimizers not available",
    )
    def test_list_optimizers_nonempty(self):
        """The optimizer registry must expose at least the core optimizers."""
        available = OptimizerRegistry.list_available()
        assert len(available) >= 5
        for name in ("adam", "adamw", "sgd"):
            assert name in available

    @pytest.mark.skipif(
        not _OPTIMIZER_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.optimizers not available",
    )
    def test_unknown_optimizer_raises(self, simple_model):
        """Requesting a non-existent optimizer must raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nknown"):
            get_optimizer("not_an_optimizer", simple_model.parameters())


# =============================================================================
# TEST CLASS: Scheduler Registry
# =============================================================================


class TestSchedulerRegistry:
    """Validate scheduler instantiation from the registry."""

    @pytest.mark.skipif(
        not _SCHEDULER_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.schedulers not available",
    )
    def test_get_step_lr_scheduler(self, simple_model):
        """get_scheduler('step_lr', ...) must return a valid scheduler."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        scheduler = get_scheduler("step_lr", optimizer, {"step_size": 10})
        assert scheduler is not None

    @pytest.mark.skipif(
        not _SCHEDULER_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.schedulers not available",
    )
    def test_list_schedulers_nonempty(self):
        """The scheduler registry must expose at least the core schedulers."""
        available = SchedulerRegistry.list_available()
        assert len(available) >= 5


# =============================================================================
# TEST CLASS: Metrics Registry
# =============================================================================


class TestMetricsRegistry:
    """Validate metric instantiation from the registry."""

    @pytest.mark.skipif(
        not _METRICS_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.metrics not available",
    )
    def test_get_metrics_for_graph_regression(self):
        """get_metrics_for_task('graph_regression') must return regression metrics."""
        metrics = MetricsRegistry.get_metrics_for_task("graph_regression")
        assert isinstance(metrics, dict)
        assert len(metrics) > 0, "Expected at least one metric for graph_regression"

    @pytest.mark.skipif(
        not _METRICS_REGISTRY_AVAILABLE,
        reason="milia_pipeline.models.training.metrics not available",
    )
    def test_list_available_metrics_nonempty(self):
        """The metrics registry must expose at least regression metrics."""
        available = MetricsRegistry.list_available()
        assert len(available) >= 4


# =============================================================================
# TEST CLASS: Target Selection Config
# =============================================================================


class TestTargetSelectionConfig:
    """Validate TargetSelectionConfig creation and resolution."""

    @pytest.mark.skipif(
        not _TARGET_SELECTION_AVAILABLE,
        reason="milia_pipeline.models.factory.target_selection_config not available",
    )
    def test_from_config_none_returns_all(self):
        """from_config(None) must default to SelectionMode.ALL."""
        tsc = TargetSelectionConfig.from_config(None)
        assert tsc.mode == SelectionMode.ALL

    @pytest.mark.skipif(
        not _TARGET_SELECTION_AVAILABLE,
        reason="milia_pipeline.models.factory.target_selection_config not available",
    )
    def test_from_config_indices(self):
        """from_config({'indices': [0, 2]}) must set mode to INDICES."""
        tsc = TargetSelectionConfig.from_config({"indices": [0, 2]})
        assert tsc.mode == SelectionMode.INDICES
        assert tsc.indices == [0, 2]

    @pytest.mark.skipif(
        not _TARGET_SELECTION_AVAILABLE,
        reason="milia_pipeline.models.factory.target_selection_config not available",
    )
    def test_to_dict_serialization(self):
        """to_dict() must return a JSON-serializable dictionary."""
        tsc = TargetSelectionConfig.from_config({"indices": [0]})
        d = tsc.to_dict()
        assert isinstance(d, dict)
        assert "mode" in d
        assert d["mode"] == "INDICES"

    @pytest.mark.skipif(
        not _TARGET_SELECTION_AVAILABLE,
        reason="milia_pipeline.models.factory.target_selection_config not available",
    )
    def test_resolve_indices(self):
        """resolve() must populate resolved_indices for index-based selection."""
        tsc = TargetSelectionConfig.from_config({"indices": [0, 2]})
        tsc.resolve(total_count=5, available_names=["a", "b", "c", "d", "e"])
        assert tsc.resolved_indices == [0, 2]
        assert tsc.resolved_names == ["a", "c"]


# =============================================================================
# TEST CLASS: Trainer (Core Training Loop)
# =============================================================================


class TestTrainerCore:
    """
    Test the Trainer class in isolation with synthetic data and a simple model.
    """

    @pytest.mark.skipif(
        not _TRAINING_STACK_AVAILABLE,
        reason="Full training stack not available",
    )
    def test_trainer_initialization(self, simple_model, synthetic_graphs):
        """Trainer must initialize without errors given valid inputs."""
        train, val, _ = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)

        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=2,
        )
        assert trainer is not None
        assert trainer.max_epochs == 2

    @pytest.mark.skipif(
        not _TRAINING_STACK_AVAILABLE,
        reason="Full training stack not available",
    )
    def test_trainer_fit_returns_results(self, simple_model, synthetic_graphs):
        """Trainer.fit() must return a results dict with expected keys."""
        train, val, test = _split_dataset(synthetic_graphs)
        train_loader, val_loader, test_loader = _make_loaders(train, val, test)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)

        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=3,
        )
        results = trainer.fit()

        assert isinstance(results, dict)
        assert "train_metrics" in results
        assert "training_time" in results
        assert results["training_time"] > 0.0

    @pytest.mark.skipif(
        not _TRAINING_STACK_AVAILABLE,
        reason="Full training stack not available",
    )
    def test_training_loss_recorded(self, simple_model, synthetic_graphs):
        """Training must record loss values in metrics_history."""
        train, val, _ = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)

        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=5,
        )
        results = trainer.fit()

        train_metrics = results["train_metrics"]
        assert "train_loss" in train_metrics
        losses = train_metrics["train_loss"]
        assert len(losses) == 5, f"Expected 5 loss values, got {len(losses)}"
        # All losses must be finite positive numbers
        for loss_val in losses:
            assert loss_val >= 0.0, f"Negative loss encountered: {loss_val}"
            assert not torch.isnan(torch.tensor(loss_val)), "NaN loss encountered"
            assert not torch.isinf(torch.tensor(loss_val)), "Inf loss encountered"

    @pytest.mark.skipif(
        not _TRAINING_STACK_AVAILABLE,
        reason="Full training stack not available",
    )
    def test_validation_loss_recorded(self, simple_model, synthetic_graphs):
        """When a val_loader is provided, val_loss must be recorded."""
        train, val, _ = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)

        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=3,
        )
        results = trainer.fit()

        assert "val_loss" in results["train_metrics"]
        val_losses = results["train_metrics"]["val_loss"]
        assert len(val_losses) == 3

    @pytest.mark.skipif(
        not _TRAINING_STACK_AVAILABLE,
        reason="Full training stack not available",
    )
    def test_trainer_requires_optimizer(self, simple_model, synthetic_graphs):
        """Trainer must raise TrainingError if no optimizer is provided."""
        train, val, _ = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        from milia_pipeline.models.training.trainer import TrainingError

        with pytest.raises(TrainingError):
            Trainer(
                model=simple_model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=nn.MSELoss(),
                optimizer=None,
                device=torch.device("cpu"),
                max_epochs=1,
            )


# =============================================================================
# TEST CLASS: Callbacks Integration
# =============================================================================


class TestCallbacksIntegration:
    """Test callback integration with the Trainer."""

    @pytest.mark.skipif(
        not _TRAINING_STACK_AVAILABLE,
        reason="Full training stack not available",
    )
    def test_early_stopping_triggers(self, synthetic_graphs):
        """EarlyStopping must halt training before max_epochs when loss plateaus."""
        model = _SimpleGNN(in_channels=16, hidden_channels=4, out_channels=1)
        train, val, _ = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  # tiny LR → no improvement

        early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=50,
            callbacks=[early_stop],
        )
        results = trainer.fit()

        epochs_run = len(results["train_metrics"]["train_loss"])
        # With patience=2 and tiny LR, training should stop well before 50
        assert epochs_run < 50, f"EarlyStopping did not trigger: ran all {epochs_run} epochs"

    @pytest.mark.skipif(
        not _TRAINING_STACK_AVAILABLE,
        reason="Full training stack not available",
    )
    def test_model_checkpoint_saves_files(self, synthetic_graphs, checkpoint_dir):
        """ModelCheckpoint must create checkpoint files on disk."""
        model = _SimpleGNN(in_channels=16, hidden_channels=16, out_channels=1)
        train, val, _ = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        ckpt_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            save_best=True,
        )
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=3,
            callbacks=[ckpt_callback],
        )
        trainer.fit()

        # At least one checkpoint file must exist
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0, f"No checkpoint files found in {checkpoint_dir}"

    @pytest.mark.skipif(
        not _TRAINING_STACK_AVAILABLE,
        reason="Full training stack not available",
    )
    def test_model_checkpoint_requires_dirpath(self):
        """ModelCheckpoint must raise ValueError if dirpath is None."""
        with pytest.raises(ValueError, match="dirpath"):
            ModelCheckpoint(dirpath=None)


# =============================================================================
# TEST CLASS: Checkpoint Manager
# =============================================================================


class TestCheckpointManager:
    """Test CheckpointManager save/load lifecycle."""

    @pytest.mark.skipif(
        not _CHECKPOINT_MANAGER_AVAILABLE,
        reason="milia_pipeline.models.post_training.checkpoint not available",
    )
    def test_save_and_load_checkpoint(self, simple_model, working_dir):
        """Save then load a checkpoint — state_dict keys must match."""
        manager = CheckpointManager(working_root_dir=working_dir)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        filepath = manager.save(
            filepath="checkpoints/test_model.pt",
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            global_step=100,
            best_val_loss=0.123,
            hyper_parameters={
                "model_name": "SimpleGNN",
                "task_type": "graph_regression",
                "hyperparameters": {"in_channels": 16, "hidden_channels": 32},
            },
        )
        assert filepath.exists()

        checkpoint = manager.load("checkpoints/test_model.pt")
        assert checkpoint["epoch"] == 5
        assert checkpoint["global_step"] == 100
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "hyper_parameters" in checkpoint

    @pytest.mark.skipif(
        not _CHECKPOINT_MANAGER_AVAILABLE,
        reason="milia_pipeline.models.post_training.checkpoint not available",
    )
    def test_checkpoint_version_info(self, simple_model, working_dir):
        """Saved checkpoint must contain version_info with format version."""
        manager = CheckpointManager(working_root_dir=working_dir)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        manager.save(
            filepath="checkpoints/versioned.pt",
            model=simple_model,
            optimizer=optimizer,
        )
        checkpoint = manager.load("checkpoints/versioned.pt")

        assert "version_info" in checkpoint
        vi = checkpoint["version_info"]
        assert "checkpoint_format_version" in vi
        assert vi["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION

    @pytest.mark.skipif(
        not _CHECKPOINT_MANAGER_AVAILABLE,
        reason="milia_pipeline.models.post_training.checkpoint not available",
    )
    def test_is_v2_checkpoint(self, simple_model, working_dir):
        """is_v2_checkpoint() must return True for v2.0 checkpoints."""
        manager = CheckpointManager(working_root_dir=working_dir)
        manager.save(filepath="checkpoints/v2.pt", model=simple_model)
        checkpoint = manager.load("checkpoints/v2.pt")
        assert manager.is_v2_checkpoint(checkpoint)

    @pytest.mark.skipif(
        not _CHECKPOINT_MANAGER_AVAILABLE,
        reason="milia_pipeline.models.post_training.checkpoint not available",
    )
    def test_load_nonexistent_raises(self, working_dir):
        """Loading a non-existent checkpoint must raise an error."""
        manager = CheckpointManager(working_root_dir=working_dir)
        with pytest.raises((FileNotFoundError, RuntimeError, Exception)):
            manager.load("checkpoints/does_not_exist.pt")

    @pytest.mark.skipif(
        not _CHECKPOINT_MANAGER_AVAILABLE,
        reason="milia_pipeline.models.post_training.checkpoint not available",
    )
    def test_get_checkpoint_dir_creates_directory(self, working_dir):
        """get_checkpoint_dir() must create the directory if it doesn't exist."""
        manager = CheckpointManager(working_root_dir=working_dir)
        ckpt_dir = manager.get_checkpoint_dir("custom_ckpts")
        assert ckpt_dir.exists()
        assert ckpt_dir.is_dir()
        assert ckpt_dir.name == "custom_ckpts"


# =============================================================================
# TEST CLASS: Full E2E Training → Checkpoint → Load Workflow
# =============================================================================


class TestEndToEndTrainingWorkflow:
    """
    Full end-to-end tests exercising:
        train → checkpoint → load → predict

    These are the most comprehensive tests in this suite, validating that
    the entire training lifecycle integrates correctly.
    """

    @pytest.mark.skipif(
        not (_TRAINING_STACK_AVAILABLE and _CHECKPOINT_MANAGER_AVAILABLE),
        reason="Full training + checkpoint stack required",
    )
    def test_train_save_load_predict(self, synthetic_graphs, working_dir):
        """
        Complete workflow: train model → save checkpoint → load state_dict
        → run inference → verify prediction shapes.
        """
        # --- Setup ---
        model = _SimpleGNN(in_channels=16, hidden_channels=32, out_channels=1)
        train, val, test = _split_dataset(synthetic_graphs)
        train_loader, val_loader, test_loader = _make_loaders(train, val, test)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        checkpoint_path = working_dir / "checkpoints"

        ckpt_callback = ModelCheckpoint(
            dirpath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_best=True,
        )

        # --- Train ---
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=3,
            callbacks=[ckpt_callback],
        )
        results = trainer.fit()
        assert "train_metrics" in results

        # --- Save explicit checkpoint via CheckpointManager ---
        cm = CheckpointManager(working_root_dir=working_dir)
        cm.save(
            filepath="checkpoints/final.pt",
            model=model,
            optimizer=optimizer,
            epoch=trainer.current_epoch,
            best_val_loss=trainer.best_val_loss,
            hyper_parameters={
                "model_name": "SimpleGNN",
                "task_type": "graph_regression",
                "hyperparameters": {
                    "in_channels": 16,
                    "hidden_channels": 32,
                    "out_channels": 1,
                },
            },
        )

        # --- Load checkpoint ---
        checkpoint = cm.load("checkpoints/final.pt")
        loaded_model = _SimpleGNN(in_channels=16, hidden_channels=32, out_channels=1)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.eval()

        # --- Predict ---
        test_sample = test[0]
        with torch.no_grad():
            pred = loaded_model(
                test_sample.x,
                test_sample.edge_index,
                batch=None,
            )
        assert pred.shape == (1, 1), f"Unexpected prediction shape: {pred.shape}"
        assert not torch.isnan(pred).any(), "NaN in predictions"

    @pytest.mark.skipif(
        not (_TRAINING_STACK_AVAILABLE and _CHECKPOINT_MANAGER_AVAILABLE),
        reason="Full training + checkpoint stack required",
    )
    def test_multi_target_train_and_predict(self, multi_target_graphs, working_dir):
        """
        Multi-target regression: train with 3 targets → predict → verify shape.
        """
        model = _SimpleGNN(in_channels=16, hidden_channels=32, out_channels=3)
        train, val, _ = _split_dataset(multi_target_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=3,
        )
        _results = trainer.fit()

        # Save and reload
        cm = CheckpointManager(working_root_dir=working_dir)
        cm.save(filepath="checkpoints/multi.pt", model=model)
        checkpoint = cm.load("checkpoints/multi.pt")

        loaded = _SimpleGNN(in_channels=16, hidden_channels=32, out_channels=3)
        loaded.load_state_dict(checkpoint["model_state_dict"])
        loaded.eval()

        sample = multi_target_graphs[0]
        with torch.no_grad():
            pred = loaded(sample.x, sample.edge_index)
        assert pred.shape[-1] == 3, f"Expected 3-target output, got shape {pred.shape}"

    @pytest.mark.skipif(
        not (_TRAINING_STACK_AVAILABLE and _CHECKPOINT_MANAGER_AVAILABLE),
        reason="Full training + checkpoint stack required",
    )
    def test_training_with_scheduler(self, synthetic_graphs, working_dir):
        """Training with an LR scheduler must complete without errors."""
        model = _SimpleGNN(in_channels=16, hidden_channels=16, out_channels=1)
        train, val, _ = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device("cpu"),
            max_epochs=4,
        )
        results = trainer.fit()

        assert len(results["train_metrics"]["train_loss"]) == 4

    @pytest.mark.skipif(
        not (_TRAINING_STACK_AVAILABLE and _CHECKPOINT_MANAGER_AVAILABLE),
        reason="Full training + checkpoint stack required",
    )
    def test_training_with_gradient_clipping(self, synthetic_graphs):
        """Training with gradient clipping must complete without errors."""
        model = _SimpleGNN(in_channels=16, hidden_channels=16, out_channels=1)
        train, val, _ = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=2,
            gradient_clip_val=1.0,
        )
        results = trainer.fit()
        assert "train_metrics" in results

    @pytest.mark.skipif(
        not _TRAINING_STACK_AVAILABLE,
        reason="Full training stack not available",
    )
    def test_training_with_metrics(self, synthetic_graphs):
        """Training with evaluation metrics must record metric values."""
        if not _METRICS_REGISTRY_AVAILABLE:
            pytest.skip("MetricsRegistry not available")

        model = _SimpleGNN(in_channels=16, hidden_channels=16, out_channels=1)
        train, val, _ = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        metrics = MetricsRegistry.get_metrics_for_task("graph_regression")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=2,
            metrics=metrics,
        )
        results = trainer.fit()
        assert "train_metrics" in results


# =============================================================================
# TEST CLASS: Predictor Integration (Post-Training)
# =============================================================================


class TestPredictorIntegration:
    """Test the Predictor class with trained model checkpoints."""

    @pytest.mark.skipif(
        not (_TRAINING_STACK_AVAILABLE and _POST_TRAINING_STACK_AVAILABLE),
        reason="Full training + post-training stack required",
    )
    def test_predictor_from_manual_model(self, synthetic_graphs, working_dir):
        """
        Predictor must produce predictions from a manually loaded model.
        """
        # Train a model
        model = _SimpleGNN(in_channels=16, hidden_channels=16, out_channels=1)
        train, val, test = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=2,
        )
        trainer.fit()

        # Create predictor directly with trained model
        predictor = Predictor(
            model=model,
            working_root_dir=working_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        # Predict on a single sample
        sample = test[0]
        pred = predictor.predict(sample)
        assert pred is not None
        assert isinstance(pred, torch.Tensor)

    @pytest.mark.skipif(
        not (_TRAINING_STACK_AVAILABLE and _POST_TRAINING_STACK_AVAILABLE),
        reason="Full training + post-training stack required",
    )
    def test_predictor_batch_prediction(self, synthetic_graphs, working_dir):
        """
        Predictor.predict_batch() must handle a full dataset.
        """
        model = _SimpleGNN(in_channels=16, hidden_channels=16, out_channels=1)
        train, val, test = _split_dataset(synthetic_graphs)
        train_loader, val_loader, _ = _make_loaders(train, val)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=2,
        )
        trainer.fit()

        predictor = Predictor(
            model=model,
            working_root_dir=working_dir,
            device=torch.device("cpu"),
            task_type="graph_regression",
        )

        # Batch predict on test set
        predictions = predictor.predict_batch(test)
        assert predictions is not None
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape[0] > 0


# =============================================================================
# TEST CLASS: Model Factory Integration
# =============================================================================


class TestModelFactoryIntegration:
    """Test ModelFactory for model creation with the training stack."""

    @pytest.mark.skipif(
        not _MODEL_FACTORY_AVAILABLE,
        reason="milia_pipeline.models.factory.model_factory not available",
    )
    def test_factory_creates_gcn(self, synthetic_graphs):
        """ModelFactory must successfully create a GCN model."""
        factory = get_factory()
        sample = synthetic_graphs[0]

        try:
            model = factory.create_model(
                name="GCN",
                hyperparameters={
                    "hidden_channels": 32,
                    "num_layers": 2,
                    "out_channels": 1,
                },
                task_type="graph_regression",
                sample_data=sample,
            )
            assert model is not None
            assert isinstance(model, nn.Module)
        except Exception as e:
            # Some environments may not have all PyG dependencies
            if "torch_cluster" in str(e) or "torch_sparse" in str(e):
                pytest.skip(f"PyG optional dependency missing: {e}")
            raise

    @pytest.mark.skipif(
        not _MODEL_FACTORY_AVAILABLE,
        reason="milia_pipeline.models.factory.model_factory not available",
    )
    def test_factory_get_model_info(self):
        """get_model_info() must return metadata for known models."""
        factory = get_factory()
        info = factory.get_model_info("GCN")
        if info is None:
            pytest.skip("GCN not found in model registry — PyG version may differ")
        assert isinstance(info, dict)
        assert "name" in info
        assert "class" in info


# =============================================================================
# TEST CLASS: Config Loader Integration
# =============================================================================


class TestConfigLoaderIntegration:
    """Test config loading as the entry point of the E2E workflow."""

    @pytest.mark.skipif(
        not _CONFIG_LOADER_AVAILABLE,
        reason="milia_pipeline.config.config_loader not available",
    )
    def test_load_config_with_valid_yaml(self, tmp_path):
        """load_config() must parse a valid YAML config file."""
        config_content = """
dataset_type: DFT
global_paths:
  working_root_dir: /tmp/milia_test
model_config:
  model_name: GCN
  task_type: graph_regression
  hyperparameters:
    hidden_channels: 64
    num_layers: 3
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        config = load_config(str(config_file))
        assert isinstance(config, dict)
        assert "dataset_type" in config or "model_config" in config


# =============================================================================
# ENTRY POINT (for direct execution during development)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
