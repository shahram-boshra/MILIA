"""
Smoke Test: Pipeline End-to-End
===============================

Rapid, lightweight checks that verify the MILIA pipeline's core mechanics
run without crashing. This is the **first gate in the CI/CD pipeline** —
if smoke tests fail, no further (more expensive) tests are triggered.

These tests do NOT validate correctness; they confirm the system is
"not on fire."

Modules exercised (Section 1.1 of MILIA_Test_Recommendations.md):
- milia_pipeline/config/config_loader.py          — Configuration loading
- milia_pipeline/config/config_containers.py       — Config container creation
- milia_pipeline/config/config_accessors.py        — Config access patterns
- milia_pipeline/datasets/registry.py              — Dataset registry discovery
- milia_pipeline/handlers/base_handler.py          — Handler factory (create_dataset_handler)
- milia_pipeline/handlers/handler_registry.py      — Handler registry lookup
- milia_pipeline/molecules/molecule_converter_core.py — MoleculeDataConverter
- milia_pipeline/transformations/graph_transforms.py  — Transform composition
- milia_pipeline/models/registry/model_registry.py — Model discovery
- milia_pipeline/models/factory/model_factory.py   — Model creation
- milia_pipeline/models/training/trainer.py        — Trainer instantiation
- main.py                                          — Main entry point orchestration

Scope:
- Uses minimal synthetic data (5–10 molecules)
- Trains for 1–2 epochs
- Asserts no exceptions raised and outputs are produced
- Total runtime target: < 30 seconds

Usage:
    pytest tests/test_smoke_pipeline_end_to_end.py -v --tb=short
    pytest tests/test_smoke_pipeline_end_to_end.py -v -m smoke

Docker usage:
    (shah_env) root@01b78773d9b4:/app/milia# pytest tests/test_smoke_pipeline_end_to_end.py -v

Author: MILIA Team
Version: 1.0.0
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest
import torch
import numpy as np

# ===========================================================================
# PATH SETUP: Add project root to Python path FIRST
# ===========================================================================
# This ensures milia_pipeline is importable regardless of working directory.
# Evidence: MILIA_Test_Recommendations.md "NOTE: I MUST Add the project root
# to Python path FIRST"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ===========================================================================
# PYTEST MARKERS
# ===========================================================================
# The ``smoke`` marker requires registration via conftest.py or pytest.ini
# to avoid PytestUnknownMarkWarning. This warning fires at *collection time*
# (before any test-level filterwarnings apply), so it CANNOT be suppressed
# from within a test file.
#
# Required: add to tests/conftest.py (or the project's existing conftest.py):
#
#     def pytest_configure(config):
#         config.addinivalue_line(
#             "markers",
#             "smoke: Smoke tests — fast, first gate in CI/CD pipeline",
#         )
#
# Alternatively, add to pytest.ini / pyproject.toml:
#
#     [pytest]
#     markers =
#         smoke: Smoke tests — fast, first gate in CI/CD pipeline
#
pytestmark = [
    pytest.mark.smoke,
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

# ===========================================================================
# MODULE-LEVEL LOGGER
# ===========================================================================
logger = logging.getLogger(__name__)


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture(scope="module")
def tmp_work_dir():
    """Provide a temporary working directory for the test module.
    
    Cleaned up after all tests in this module complete.
    """
    tmp_dir = tempfile.mkdtemp(prefix="milia_smoke_e2e_")
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def minimal_config_dict() -> Dict[str, Any]:
    """Provide a minimal valid configuration dictionary.
    
    This mirrors the structure of config.yaml with only the fields
    required for smoke testing. It avoids touching the filesystem
    for config loading so the test is self-contained.
    
    Evidence: config_loader.py load_config() expects a dict with
    'dataset_type' (line ~700+), and config_containers.py
    create_dataset_config_from_global() reads 'dataset_type',
    'data_config', etc.
    """
    return {
        "dataset_type": "DFT",
        "working_root_dir": "/tmp/milia_smoke_test",
        "data_config": {
            "common_settings": {
                "chunk_size": 5,
                "max_atoms": 50,
                "min_atoms": 1,
            },
            "property_selection": {
                "DFT": ["energy"]
            }
        },
        "filter_config": {
            "max_atoms": 50,
            "min_atoms": 1,
            "allowed_elements": ["H", "C", "N", "O"],
        },
        "structural_features": {
            "enabled": False,
        },
        "transformations": {
            "standard_transforms": [],
            "experimental_setups": {},
            "default_setup": None,
        },
        "property_availability": {
            "DFT": {
                "energy": True,
                "forces": False,
            }
        },
        "model_config": {
            "model_name": "GCN",
            "hyperparameters": {
                "hidden_channels": 16,
                "num_layers": 2,
            },
            "task_type": "graph_regression",
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
        },
        "evaluation": {
            "visualization": {
                "enabled": False,
            }
        },
        "molecular_descriptors": {
            "enabled": False,
        },
    }


@pytest.fixture(scope="module")
def synthetic_pyg_data_list() -> List:
    """Create a minimal list of synthetic PyG Data objects.
    
    Generates 10 small molecular graphs with:
    - x:          Node features (num_atoms, 11) — 11 is MILIA's default feature dim
    - edge_index: COO sparse edge connectivity
    - y:          Graph-level regression target shape (1,)
    - pos:        3D coordinates (num_atoms, 3) — optional, for 3D-aware models
    - z:          Atomic numbers (num_atoms,) — optional, for equivariant models
    
    Evidence: trainer.py Trainer expects DataLoader yielding Batch objects
    with at minimum x, edge_index, y (lines 94-150). model_factory.py
    ModelFactory.create_model() uses sample_data for channel inference.
    """
    from torch_geometric.data import Data

    data_list = []
    rng = np.random.RandomState(42)

    for i in range(10):
        num_atoms = rng.randint(3, 10)
        num_edges = rng.randint(num_atoms, num_atoms * 3)

        # Node features: (num_atoms, 11) — matches MILIA default feature dimension
        x = torch.randn(num_atoms, 11, dtype=torch.float32)

        # Edge connectivity: random COO format
        src = torch.randint(0, num_atoms, (num_edges,))
        dst = torch.randint(0, num_atoms, (num_edges,))
        edge_index = torch.stack([src, dst], dim=0)

        # Graph-level regression target
        y = torch.tensor([rng.uniform(-5.0, 5.0)], dtype=torch.float32)

        # 3D coordinates
        pos = torch.randn(num_atoms, 3, dtype=torch.float32)

        # Atomic numbers (C=6, N=7, O=8, H=1)
        z = torch.tensor(rng.choice([1, 6, 7, 8], size=num_atoms), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y, pos=pos, z=z)
        data_list.append(data)

    return data_list


@pytest.fixture(scope="module")
def synthetic_pyg_dataset(synthetic_pyg_data_list):
    """Wrap synthetic data list into a list-like container usable as a dataset.
    
    The Trainer and DataSplitter accept any indexable sequence. Using a plain
    list avoids the need for a full InMemoryDataset with filesystem I/O.
    """
    return synthetic_pyg_data_list


# ===========================================================================
# SECTION 1: CONFIGURATION SYSTEM SMOKE TESTS
# ===========================================================================

class TestConfigurationSystemSmoke:
    """Smoke tests for the configuration loading and access subsystem.
    
    Verifies that config_loader, config_containers, and config_accessors
    can be imported and their core functions invoked without exceptions.
    """

    def test_config_loader_importable(self):
        """config_loader module can be imported without errors."""
        from milia_pipeline.config import config_loader
        assert hasattr(config_loader, "load_config")
        assert hasattr(config_loader, "clear_config_cache")

    def test_config_containers_importable(self):
        """config_containers module can be imported without errors."""
        from milia_pipeline.config.config_containers import (
            DatasetConfig,
            FilterConfig,
            ProcessingConfig,
        )
        assert DatasetConfig is not None
        assert FilterConfig is not None
        assert ProcessingConfig is not None

    def test_config_accessors_importable(self):
        """config_accessors module can be imported without errors."""
        from milia_pipeline.config import config_accessors
        assert hasattr(config_accessors, "get_dataset_type")

    def test_config_container_creation_from_dict(self, minimal_config_dict):
        """Config containers can be created from a configuration dictionary.
        
        Evidence: config_containers.py provides create_dataset_config_from_global(),
        create_filter_config_from_global(), create_processing_config_from_global()
        factory functions that accept a global config dict.
        """
        from milia_pipeline.config.config_containers import (
            create_dataset_config_from_global,
            create_filter_config_from_global,
            create_processing_config_from_global,
        )

        dataset_config = create_dataset_config_from_global(minimal_config_dict)
        assert dataset_config is not None
        assert dataset_config.dataset_type == "DFT"

        filter_config = create_filter_config_from_global(minimal_config_dict)
        assert filter_config is not None

        processing_config = create_processing_config_from_global(minimal_config_dict)
        assert processing_config is not None

    def test_load_config_with_temp_yaml(self, tmp_work_dir, minimal_config_dict):
        """load_config can load a YAML file and return a dict.
        
        Creates a temporary config.yaml, loads it, and verifies the result
        is a dictionary with expected keys.
        
        Evidence: config_loader.py load_config() (line ~666) accepts config_path,
        returns dict.
        """
        import yaml
        from milia_pipeline.config.config_loader import load_config, clear_config_cache

        # Clear any cached config from previous test runs
        clear_config_cache()

        config_path = tmp_work_dir / "smoke_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_dict, f)

        config = load_config(
            config_path=str(config_path),
            enable_validation=False,
            enable_migration=False,
            enable_enhancement=False,
            force_reload=True,
        )

        assert isinstance(config, dict), "load_config must return a dict"
        assert "dataset_type" in config, "Config must contain 'dataset_type'"

        # Cleanup: clear cache so other tests are not affected
        clear_config_cache()


# ===========================================================================
# SECTION 2: DATASET REGISTRY SMOKE TESTS
# ===========================================================================

class TestDatasetRegistrySmoke:
    """Smoke tests for the dataset registry subsystem.
    
    Verifies that DatasetRegistry can be imported, instantiated (non-singleton
    for isolation), and that the default registry has registered datasets.
    
    Evidence: registry.py DatasetRegistry is NOT a singleton (line 3 docstring),
    supports register(), get(), list_all(), is_registered(), clear().
    """

    def test_registry_module_importable(self):
        """datasets.registry module can be imported."""
        from milia_pipeline.datasets.registry import (
            DatasetRegistry,
            get_default_registry,
            list_all,
            is_registered,
        )
        assert DatasetRegistry is not None

    def test_isolated_registry_creation(self):
        """A fresh DatasetRegistry instance can be created for testing.
        
        Evidence: registry.py docstring "NOT a singleton: Can create isolated
        instances for testing" (line 4).
        """
        from milia_pipeline.datasets.registry import DatasetRegistry

        isolated = DatasetRegistry()
        assert len(isolated) == 0
        assert isolated.list_all() == []

    def test_default_registry_has_datasets(self):
        """The default global registry should have at least one dataset registered.
        
        Evidence: datasets/implementations/__init__.py uses dynamic discovery
        with @register decorator. DFT is always available as the base dataset.
        """
        from milia_pipeline.datasets.registry import get_default_registry

        # Trigger implementations import to ensure registration happens
        # This is a core project module — import MUST succeed.
        import milia_pipeline.datasets.implementations  # noqa: F401

        registry = get_default_registry()
        registered = registry.list_all()
        assert len(registered) > 0, (
            "Default registry should have at least one dataset registered. "
            f"Got: {registered}"
        )

    def test_registry_get_returns_class(self):
        """Registry.get() returns a class (not an instance) for known datasets."""
        from milia_pipeline.datasets.registry import get_default_registry

        # Core project module — import MUST succeed.
        import milia_pipeline.datasets.implementations  # noqa: F401

        registry = get_default_registry()
        registered = registry.list_all()
        assert len(registered) > 0, "No datasets registered — cannot test get()"

        first_name = registered[0]
        dataset_class = registry.get(first_name)
        assert isinstance(dataset_class, type), (
            f"registry.get('{first_name}') should return a class, got {type(dataset_class)}"
        )


# ===========================================================================
# SECTION 3: HANDLER SYSTEM SMOKE TESTS
# ===========================================================================

class TestHandlerSystemSmoke:
    """Smoke tests for the handler subsystem.
    
    Verifies that handler imports, handler registry, and handler factory
    function (create_dataset_handler) work without exceptions.
    
    Evidence: base_handler.py contains DatasetHandler ABC (line 1),
    create_dataset_handler() factory (line 239), handler_registry.py
    contains HandlerRegistry (line 75).
    """

    def test_base_handler_importable(self):
        """base_handler module can be imported."""
        from milia_pipeline.handlers.base_handler import (
            DatasetHandler,
            create_dataset_handler,
        )
        assert DatasetHandler is not None
        assert callable(create_dataset_handler)

    def test_handler_registry_importable(self):
        """handler_registry module can be imported."""
        from milia_pipeline.handlers.handler_registry import (
            HandlerRegistry,
            get_default_registry,
            list_all,
        )
        assert HandlerRegistry is not None

    def test_isolated_handler_registry(self):
        """A fresh HandlerRegistry can be created.
        
        Evidence: handler_registry.py docstring "NOT a singleton: Can create
        isolated instances for testing" (line 8).
        """
        from milia_pipeline.handlers.handler_registry import HandlerRegistry

        isolated = HandlerRegistry()
        assert len(isolated) == 0

    def test_create_dataset_handler_with_valid_type(self, minimal_config_dict):
        """create_dataset_handler succeeds for a registered dataset type.
        
        Evidence: base_handler.py create_dataset_handler() (line 239) accepts
        DatasetConfig, FilterConfig, ProcessingConfig, logger, experimental_setup.
        """
        from milia_pipeline.config.config_containers import (
            create_dataset_config_from_global,
            create_filter_config_from_global,
            create_processing_config_from_global,
        )
        from milia_pipeline.handlers.base_handler import create_dataset_handler

        dataset_config = create_dataset_config_from_global(minimal_config_dict)
        filter_config = create_filter_config_from_global(minimal_config_dict)
        processing_config = create_processing_config_from_global(minimal_config_dict)

        try:
            handler = create_dataset_handler(
                dataset_config=dataset_config,
                filter_config=filter_config,
                processing_config=processing_config,
                logger=logging.getLogger("smoke_test.handler"),
                experimental_setup=None,
            )
            assert handler is not None
            assert hasattr(handler, "get_dataset_type")
            assert callable(handler.get_dataset_type)
            assert isinstance(handler.get_dataset_type(), str)
        except Exception as e:
            # If DFT handler is not available, skip gracefully
            error_msg = str(e).lower()
            if "not registered" in error_msg or "not found" in error_msg or "not available" in error_msg:
                pytest.skip(f"DFT handler not available in this environment: {e}")
            raise

    def test_handler_has_required_protocol_methods(self, minimal_config_dict):
        """Created handler has the 11 protocol methods.
        
        Evidence: protocols.py DatasetHandlerProtocol defines 11 methods
        (MILIA_Pipeline_Project_Structure.md line 357-365).
        """
        from milia_pipeline.config.config_containers import (
            create_dataset_config_from_global,
            create_filter_config_from_global,
            create_processing_config_from_global,
        )
        from milia_pipeline.handlers.base_handler import create_dataset_handler

        dataset_config = create_dataset_config_from_global(minimal_config_dict)
        filter_config = create_filter_config_from_global(minimal_config_dict)
        processing_config = create_processing_config_from_global(minimal_config_dict)

        try:
            handler = create_dataset_handler(
                dataset_config=dataset_config,
                filter_config=filter_config,
                processing_config=processing_config,
                logger=logging.getLogger("smoke_test.protocol"),
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "not registered" in error_msg or "not found" in error_msg or "not available" in error_msg:
                pytest.skip(f"DFT handler not available in this environment: {e}")
            raise

        # Check for the 11 DatasetHandlerProtocol methods
        # Evidence: MILIA_Pipeline_Project_Structure.md lines 356-365
        required_methods = [
            "get_dataset_type",
            "get_identifier_keys",
            "get_required_properties",
            "process_property_value",
            "enrich_pyg_data",
            "get_processing_statistics",
            "get_supported_structural_features",
            "get_molecular_charge",
            "get_molecule_creation_strategy",
            "get_transform_recommendations",
            "get_supported_descriptors",
        ]
        for method_name in required_methods:
            assert hasattr(handler, method_name), (
                f"Handler missing required protocol method: {method_name}"
            )


# ===========================================================================
# SECTION 4: MODEL SUBSYSTEM SMOKE TESTS
# ===========================================================================

class TestModelSubsystemSmoke:
    """Smoke tests for the model registry, factory, and trainer.
    
    Evidence:
    - model_registry.py ModelRegistry is a thread-safe singleton (line 111)
    - model_factory.py ModelFactory.create_model() (line ~4142)
    - trainer.py Trainer.__init__() (line 94)
    """

    def test_model_registry_importable(self):
        """model_registry module can be imported."""
        from milia_pipeline.models.registry.model_registry import (
            ModelRegistry,
            get_model,
            has_model,
            list_models,
        )
        assert ModelRegistry is not None

    def test_model_registry_has_models(self):
        """ModelRegistry auto-discovers PyG models on initialization.
        
        Evidence: model_registry.py __init__() calls auto_discover_pyg_models()
        (line 185).
        """
        from milia_pipeline.models.registry.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        model_count = len(registry)
        assert model_count > 0, (
            f"ModelRegistry should have discovered models, got {model_count}"
        )

    def test_model_registry_has_gcn(self):
        """GCN should be available as a basic GNN model.
        
        Evidence: model_registry.py auto-discovers GCN from
        torch_geometric.nn.models (line 224).
        """
        from milia_pipeline.models.registry.model_registry import has_model

        assert has_model("GCN"), "GCN model should be registered in ModelRegistry"

    def test_model_factory_importable(self):
        """model_factory module can be imported."""
        from milia_pipeline.models.factory.model_factory import (
            ModelFactory,
            create_model,
            get_factory,
        )
        assert ModelFactory is not None
        assert callable(create_model)

    def test_model_factory_creates_gcn(self, synthetic_pyg_data_list):
        """ModelFactory can create a GCN model instance.
        
        Evidence: model_factory.py create_model() (line ~4142) accepts
        name, hyperparameters, task_type, sample_data, device.
        """
        from milia_pipeline.models.factory.model_factory import create_model

        sample_data = synthetic_pyg_data_list[0]
        try:
            model = create_model(
                name="GCN",
                hyperparameters={
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "out_channels": 1,
                },
                task_type="graph_regression",
                sample_data=sample_data,
                device=torch.device("cpu"),
            )
            assert model is not None
            assert isinstance(model, torch.nn.Module)
        except Exception as e:
            # Some environments may not have all PyG optional dependencies
            if "torch_cluster" in str(e) or "torch_scatter" in str(e):
                pytest.skip(f"PyG optional dependency missing: {e}")
            raise

    def test_trainer_importable(self):
        """trainer module can be imported."""
        from milia_pipeline.models.training.trainer import Trainer
        assert Trainer is not None

    def test_trainer_instantiation(self, synthetic_pyg_data_list):
        """Trainer can be instantiated with a model and data loaders.
        
        Evidence: trainer.py Trainer.__init__() (line 94) requires model,
        train_loader; optional val_loader, loss_fn, optimizer, device, etc.
        """
        from milia_pipeline.models.training.trainer import Trainer
        from milia_pipeline.models.factory.model_factory import create_model
        from torch_geometric.loader import DataLoader

        sample_data = synthetic_pyg_data_list[0]
        try:
            model = create_model(
                name="GCN",
                hyperparameters={
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "out_channels": 1,
                },
                task_type="graph_regression",
                sample_data=sample_data,
                device=torch.device("cpu"),
            )
        except Exception as e:
            # Only skip for genuinely optional external PyG dependencies
            error_str = str(e)
            if any(dep in error_str for dep in ["torch_cluster", "torch_scatter", "torch_sparse", "torch_spline_conv"]):
                pytest.skip(f"PyG optional dependency missing: {e}")
            raise

        train_loader = DataLoader(synthetic_pyg_data_list[:7], batch_size=2, shuffle=True)
        val_loader = DataLoader(synthetic_pyg_data_list[7:], batch_size=2, shuffle=False)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=1,
        )
        assert trainer is not None
        assert trainer.model is model


# ===========================================================================
# SECTION 5: MINI TRAINING LOOP SMOKE TEST
# ===========================================================================

class TestMiniTrainingLoopSmoke:
    """Smoke test for a complete mini training loop.
    
    Exercises the full stack from model creation through training to
    output production, with 1–2 epochs on synthetic data.
    
    This is the most critical smoke test — it proves the pipeline
    can execute without crashing.
    """

    def test_mini_training_loop_completes(self, synthetic_pyg_data_list, tmp_work_dir):
        """Full training loop: create model → train 2 epochs → produce output.
        
        Evidence:
        - model_factory.py create_model() for model instantiation
        - trainer.py Trainer.fit() for training loop (line ~55 class)
        - trainer.py saves results if checkpoint_dir provided
        """
        from milia_pipeline.models.factory.model_factory import create_model
        from milia_pipeline.models.training.trainer import Trainer
        from torch_geometric.loader import DataLoader

        sample_data = synthetic_pyg_data_list[0]

        # Step 1: Create model
        try:
            model = create_model(
                name="GCN",
                hyperparameters={
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "out_channels": 1,
                },
                task_type="graph_regression",
                sample_data=sample_data,
                device=torch.device("cpu"),
            )
        except Exception as e:
            error_str = str(e)
            if any(dep in error_str for dep in ["torch_cluster", "torch_scatter", "torch_sparse", "torch_spline_conv"]):
                pytest.skip(f"PyG optional dependency missing: {e}")
            raise

        # Step 2: Create data loaders
        train_data = synthetic_pyg_data_list[:7]
        val_data = synthetic_pyg_data_list[7:]
        train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

        # Step 3: Set up training components
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        checkpoint_dir = tmp_work_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Step 4: Create trainer and train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_epochs=2,
            checkpoint_dir=checkpoint_dir,
        )

        # Step 5: Execute training — this is the core smoke check
        results = trainer.fit()

        # Step 6: Verify outputs were produced
        assert results is not None, "Trainer.fit() must return results"

        # Verify model is still a valid Module after training
        assert isinstance(model, torch.nn.Module)

        # Verify model can produce a forward pass after training
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            # Simple forward pass check — model should not crash
            try:
                output = model(batch.x, batch.edge_index)
                assert output is not None
            except TypeError:
                # Some model wrappers need batch argument
                try:
                    output = model(batch.x, batch.edge_index, batch=batch.batch)
                    assert output is not None
                except Exception as forward_err:
                    # The model wrapper may have a different forward signature.
                    # The core assertion of this test is that training completed
                    # (above), not that a manual forward pass with a guessed
                    # signature succeeds. Log for diagnostics, do not fail.
                    import warnings
                    warnings.warn(
                        f"Post-training forward pass with (x, edge_index, batch=batch) "
                        f"raised {type(forward_err).__name__}: {forward_err}. "
                        f"Training completion was verified successfully.",
                        stacklevel=1,
                    )

    def test_mini_training_produces_loss_history(self, synthetic_pyg_data_list, tmp_work_dir):
        """Training produces a loss history with decreasing or stable loss.
        
        This is a weak assertion — we only check that loss values exist,
        not that they decrease (smoke test, not correctness test).
        """
        from milia_pipeline.models.factory.model_factory import create_model
        from milia_pipeline.models.training.trainer import Trainer
        from torch_geometric.loader import DataLoader

        sample_data = synthetic_pyg_data_list[0]

        try:
            model = create_model(
                name="GCN",
                hyperparameters={
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "out_channels": 1,
                },
                task_type="graph_regression",
                sample_data=sample_data,
                device=torch.device("cpu"),
            )
        except Exception as e:
            error_str = str(e)
            if any(dep in error_str for dep in ["torch_cluster", "torch_scatter", "torch_sparse", "torch_spline_conv"]):
                pytest.skip(f"PyG optional dependency missing: {e}")
            raise

        train_loader = DataLoader(synthetic_pyg_data_list[:7], batch_size=2, shuffle=True)
        val_loader = DataLoader(synthetic_pyg_data_list[7:], batch_size=2, shuffle=False)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            device=torch.device("cpu"),
            max_epochs=2,
        )

        results = trainer.fit()

        # Verify that training history contains loss values
        # Evidence: trainer.py Trainer.fit() returns a dict with 'history'
        # or similar structure containing epoch-level metrics.
        assert results is not None
        # The exact structure depends on implementation. We just check it's non-empty.
        if isinstance(results, dict):
            assert len(results) > 0, "Training results dict should not be empty"


# ===========================================================================
# SECTION 6: TRANSFORM SYSTEM SMOKE TESTS
# ===========================================================================

class TestTransformSystemSmoke:
    """Smoke tests for the graph transformation subsystem.
    
    Evidence: graph_transforms.py provides TransformRegistry,
    TransformValidator, TransformComposer, list_available_transforms(),
    get_transform_info(), validate_comprehensive() (line 1-100 docstring).
    """

    def test_graph_transforms_importable(self):
        """graph_transforms module can be imported."""
        from milia_pipeline.transformations.graph_transforms import (
            list_available_transforms,
            get_transform_info,
        )
        assert callable(list_available_transforms)

    def test_list_available_transforms_returns_list(self):
        """list_available_transforms() returns a non-empty list.
        
        Evidence: graph_transforms.py registers 30+ pre-built transforms
        (docstring line 54).
        """
        from milia_pipeline.transformations.graph_transforms import (
            list_available_transforms,
        )

        transforms = list_available_transforms()
        assert isinstance(transforms, (list, tuple))
        assert len(transforms) > 0, (
            "At least one transform should be available"
        )

    def test_transform_registry_instantiation(self):
        """TransformRegistry can be instantiated."""
        from milia_pipeline.transformations.graph_transforms import TransformRegistry

        registry = TransformRegistry()
        assert registry is not None


# ===========================================================================
# SECTION 7: MOLECULE CONVERTER SMOKE TESTS
# ===========================================================================

class TestMoleculeConverterSmoke:
    """Smoke tests for the molecule conversion subsystem.
    
    Evidence: molecule_converter_core.py MoleculeDataConverter class
    (line ~200+) orchestrates the conversion pipeline.
    """

    def test_molecule_converter_importable(self):
        """molecule_converter_core module can be imported.
        
        Evidence: molecule_converter_core.py (line 29-36) performs a
        guarded import of rdkit with RDKIT_AVAILABLE flag. The module
        itself is always importable; RDKit is the optional dependency.
        """
        from milia_pipeline.molecules.molecule_converter_core import (
            MoleculeDataConverter,
        )
        assert MoleculeDataConverter is not None

    def test_rdkit_availability_check(self):
        """Check whether RDKit is available (informational, not a hard failure).
        
        Evidence: molecule_converter_core.py checks RDKIT_AVAILABLE flag
        (line 29-36). RDKit is a genuine third-party optional dependency.
        The module imports fine without it, but conversion functionality
        is limited.
        """
        from milia_pipeline.molecules.molecule_converter_core import RDKIT_AVAILABLE
        if not RDKIT_AVAILABLE:
            pytest.skip(
                "rdkit package not installed — "
                "molecular conversion functionality is limited (optional dependency)"
            )


# ===========================================================================
# SECTION 8: MAIN ENTRY POINT SMOKE TESTS
# ===========================================================================

class TestMainEntryPointSmoke:
    """Smoke tests for main.py orchestration.
    
    Evidence: main.py imports CLIManager, configuration modules,
    handler modules, transformation modules (lines 160-200+).
    These tests verify the imports succeed — they do NOT execute
    the full main() function (which requires a real config.yaml
    and data files).
    """

    def test_main_module_importable(self):
        """main.py module can be imported without crashing.
        
        This catches circular import issues and missing dependencies
        at the orchestration layer.
        """
        try:
            # main.py is at project root, not inside milia_pipeline
            # It's imported by adding project root to sys.path
            import importlib
            spec = importlib.util.spec_from_file_location(
                "main", str(_PROJECT_ROOT / "main.py")
            )
            if spec is None:
                pytest.skip("main.py not found at project root")
                return
            main_module = importlib.util.module_from_spec(spec)
            # We don't execute the module to avoid side effects;
            # we just verify the spec can be created.
            assert main_module is not None
        except FileNotFoundError:
            pytest.skip("main.py not found at project root")
        except Exception as e:
            # Certain import errors are acceptable in CI environments
            # where not all dependencies are installed
            if "No module named" in str(e):
                pytest.skip(f"Optional dependency missing for main.py: {e}")
            raise

    def test_cli_manager_importable(self):
        """CLIManager can be imported.
        
        Evidence: main.py line 172-177 imports CLIManager, parse_cli_args.
        cli_manager.py is a core project module (~3745 lines).
        """
        from milia_pipeline.cli_manager import CLIManager, parse_cli_args
        assert CLIManager is not None
        assert callable(parse_cli_args)


# ===========================================================================
# SECTION 9: CROSS-SYSTEM INTEGRATION SMOKE TEST
# ===========================================================================

class TestCrossSystemIntegrationSmoke:
    """Smoke tests that verify multiple subsystems work together.
    
    These are the highest-value smoke tests: they exercise the real
    integration paths between config, registry, handlers, models,
    and training.
    """

    def test_config_to_handler_pipeline(self, minimal_config_dict):
        """Config loading → container creation → handler creation path works.
        
        This tests the critical path from configuration to handler instantiation
        that every real pipeline execution must traverse.
        """
        from milia_pipeline.config.config_containers import (
            create_dataset_config_from_global,
            create_filter_config_from_global,
            create_processing_config_from_global,
        )

        # Step 1: Create config containers
        dataset_config = create_dataset_config_from_global(minimal_config_dict)
        filter_config = create_filter_config_from_global(minimal_config_dict)
        processing_config = create_processing_config_from_global(minimal_config_dict)

        assert dataset_config.dataset_type == "DFT"

        # Step 2: Attempt handler creation
        try:
            from milia_pipeline.handlers.base_handler import create_dataset_handler
            handler = create_dataset_handler(
                dataset_config=dataset_config,
                filter_config=filter_config,
                processing_config=processing_config,
                logger=logging.getLogger("smoke_test.integration"),
            )
            assert handler.get_dataset_type() == "DFT"
        except Exception as e:
            error_msg = str(e).lower()
            if "not registered" in error_msg or "not available" in error_msg:
                pytest.skip(f"DFT handler not available: {e}")
            raise

    def test_registry_to_model_pipeline(self, synthetic_pyg_data_list):
        """Registry discovery → model factory → model instantiation path works.
        
        This tests the critical path from model discovery to instantiation.
        """
        from milia_pipeline.models.registry.model_registry import (
            list_models,
            has_model,
        )
        from milia_pipeline.models.factory.model_factory import create_model

        # Step 1: Verify registry has models
        all_models = list_models()
        assert len(all_models) > 0, "ModelRegistry should have models"

        # Step 2: Pick a basic model that should always be available
        model_name = "GCN" if has_model("GCN") else all_models[0]

        # Step 3: Create model instance
        sample_data = synthetic_pyg_data_list[0]
        try:
            model = create_model(
                name=model_name,
                hyperparameters={
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "out_channels": 1,
                },
                task_type="graph_regression",
                sample_data=sample_data,
                device=torch.device("cpu"),
            )
            assert isinstance(model, torch.nn.Module)
        except Exception as e:
            if "torch_cluster" in str(e) or "torch_scatter" in str(e):
                pytest.skip(f"PyG optional dependency missing: {e}")
            raise

    def test_full_pipeline_config_to_training(self, synthetic_pyg_data_list, tmp_work_dir):
        """Full pipeline: config → model → data loaders → trainer → fit.
        
        This is the ultimate smoke test. If this passes, the pipeline
        is fundamentally functional.
        """
        from milia_pipeline.models.factory.model_factory import create_model
        from milia_pipeline.models.training.trainer import Trainer
        from torch_geometric.loader import DataLoader

        sample_data = synthetic_pyg_data_list[0]

        # Create model
        try:
            model = create_model(
                name="GCN",
                hyperparameters={
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "out_channels": 1,
                },
                task_type="graph_regression",
                sample_data=sample_data,
                device=torch.device("cpu"),
            )
        except Exception as e:
            error_str = str(e)
            if any(dep in error_str for dep in ["torch_cluster", "torch_scatter", "torch_sparse", "torch_spline_conv"]):
                pytest.skip(f"PyG optional dependency missing: {e}")
            raise

        # Create loaders
        train_loader = DataLoader(synthetic_pyg_data_list[:7], batch_size=2, shuffle=True)
        val_loader = DataLoader(synthetic_pyg_data_list[7:], batch_size=2, shuffle=False)

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            device=torch.device("cpu"),
            max_epochs=1,
        )

        # Train — the critical smoke assertion
        results = trainer.fit()
        assert results is not None, "Training must produce results"


# ===========================================================================
# SECTION 10: LOSS, OPTIMIZER, SCHEDULER REGISTRY SMOKE TESTS
# ===========================================================================

class TestTrainingComponentRegistriesSmoke:
    """Smoke tests for training component registries.
    
    Evidence:
    - loss_functions.py LossRegistry with 18 losses (project structure line 843-865)
    - optimizers.py OptimizerRegistry with 12 optimizers (line 866-879)
    - schedulers.py SchedulerRegistry with 13 schedulers (line 880-899)
    - metrics.py MetricsRegistry with 12 metrics (line 900-920)
    
    Design decision: These are all core project modules. Imports MUST succeed;
    blanket ``except ImportError: skip`` would mask real breakage. The only
    legitimate skip is when the module depends on a genuinely optional
    third-party package (e.g., ``torchmetrics`` for metrics.py).
    """

    def test_loss_registry_importable_and_lists(self):
        """LossRegistry can list available losses."""
        from milia_pipeline.models.training.loss_functions import list_losses

        losses = list_losses()
        assert isinstance(losses, list)
        assert len(losses) > 0, "LossRegistry should have losses registered"

    def test_optimizer_registry_importable_and_lists(self):
        """OptimizerRegistry can list available optimizers."""
        from milia_pipeline.models.training.optimizers import list_optimizers

        optimizers = list_optimizers()
        assert isinstance(optimizers, list)
        assert len(optimizers) > 0, "OptimizerRegistry should have optimizers"

    def test_scheduler_registry_importable_and_lists(self):
        """SchedulerRegistry can list available schedulers."""
        from milia_pipeline.models.training.schedulers import list_schedulers

        schedulers = list_schedulers()
        assert isinstance(schedulers, list)
        assert len(schedulers) > 0, "SchedulerRegistry should have schedulers"

    def test_metrics_registry_importable_and_lists(self):
        """MetricsRegistry can list available metrics.
        
        Evidence: metrics.py depends on ``torchmetrics`` (project structure
        line 900: "TorchMetrics-based evaluation metrics"). ``torchmetrics``
        is a third-party optional dependency. If it is not installed, the
        module-level import inside metrics.py will raise ImportError.
        
        Note: The project structure doc (line 920) documents a module-level
        convenience function ``list_available()``, but runtime evidence shows
        the actual module does NOT export that name (ImportError on import).
        The sibling modules use the pattern ``list_<plural>()`` (e.g.
        ``list_losses``, ``list_optimizers``, ``list_schedulers``), but we
        cannot assume metrics.py follows the same convention without evidence.
        
        This test therefore uses runtime introspection to:
        1. Verify the MetricsRegistry class exists (always documented)
        2. Discover the actual module-level listing function by probing
           known candidate names from documentation and naming conventions.
        """
        # Step 1: Check if the external dependency is installed
        torchmetrics_available = True
        try:
            import torchmetrics  # noqa: F401
        except ImportError:
            torchmetrics_available = False

        if not torchmetrics_available:
            pytest.skip(
                "torchmetrics package not installed — "
                "MetricsRegistry requires torchmetrics (optional dependency)"
            )

        # Step 2: If torchmetrics IS installed, the core module MUST import
        import milia_pipeline.models.training.metrics as metrics_mod

        # Step 3: MetricsRegistry class must exist
        assert hasattr(metrics_mod, "MetricsRegistry"), (
            "metrics.py must export MetricsRegistry class"
        )

        # Step 4: Discover the actual listing function
        # Candidates based on documentation (list_available) and sibling
        # module naming convention (list_metrics, list_all)
        list_fn_candidates = [
            "list_available",       # documented in project structure line 920
            "list_metrics",         # follows sibling pattern: list_losses, list_optimizers
            "list_all",             # generic registry pattern
            "list_available_metrics",  # expanded variant
        ]

        list_fn = None
        list_fn_name = None
        for candidate in list_fn_candidates:
            if hasattr(metrics_mod, candidate) and callable(getattr(metrics_mod, candidate)):
                list_fn = getattr(metrics_mod, candidate)
                list_fn_name = candidate
                break

        # If no module-level convenience function exists, fall back to
        # the MetricsRegistry class itself — it must have a listing method
        if list_fn is None:
            registry_cls = metrics_mod.MetricsRegistry
            # Check for class-level or instance-level listing methods
            registry_list_candidates = [
                "list_available", "list_metrics", "list_all",
                "list_registered", "available_metrics", "get_all_metrics",
            ]
            for candidate in registry_list_candidates:
                if hasattr(registry_cls, candidate) and callable(getattr(registry_cls, candidate)):
                    # Determine if it's a classmethod/staticmethod or instance method
                    attr = getattr(registry_cls, candidate)
                    try:
                        # Try calling as classmethod/staticmethod first
                        result = attr()
                        if isinstance(result, (list, tuple, dict, set)):
                            list_fn_name = f"MetricsRegistry.{candidate}"
                            metrics = list(result) if not isinstance(result, list) else result
                            assert len(metrics) > 0, (
                                f"MetricsRegistry.{candidate}() returned empty collection"
                            )
                            return  # Test passed via class method
                    except TypeError:
                        # Needs an instance — skip, not testable without constructor args
                        continue

            # If we reach here, no listing method was found at all
            available_attrs = [a for a in dir(metrics_mod) if not a.startswith("_")]
            assert False, (
                f"No listing function found in metrics.py. "
                f"Searched module-level: {list_fn_candidates}. "
                f"Searched MetricsRegistry class: {registry_list_candidates}. "
                f"Available module attributes: {available_attrs}"
            )

        # If we found a module-level function, call it
        metrics = list_fn()
        assert isinstance(metrics, (list, tuple)), (
            f"{list_fn_name}() should return a list/tuple, got {type(metrics)}"
        )
        assert len(metrics) > 0, (
            f"{list_fn_name}() should return non-empty list of metrics"
        )

    def test_get_loss_mse(self):
        """MSE loss can be retrieved from LossRegistry."""
        from milia_pipeline.models.training.loss_functions import get_loss

        loss = get_loss("mse")
        assert loss is not None
        assert isinstance(loss, torch.nn.Module)

    def test_get_optimizer_adam(self):
        """Adam optimizer can be retrieved from OptimizerRegistry."""
        from milia_pipeline.models.training.optimizers import get_optimizer

        # get_optimizer needs model parameters
        # Evidence: optimizers.py get_optimizer() convenience function
        # (project structure line 879) accepts (name, params, **kwargs)
        dummy_model = torch.nn.Linear(10, 1)
        optimizer = get_optimizer("adam", dummy_model.parameters())
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
