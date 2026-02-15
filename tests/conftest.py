"""Pytest configuration — shared fixtures, custom markers, and warning filters.

This ``conftest.py`` lives in the ``tests/`` directory and is auto-discovered
by pytest at collection time (before any test runs).

It provides three categories of functionality:

1. **Shared fixtures** (Section 7.1 of ``MILIA_Test_Recommendations.md``) —
   reusable pytest fixtures consumed by smoke, contract, E2E, regression,
   configuration, and thread-safety tests:

   - ``synthetic_pyg_dataset`` — 30-graph synthetic PyG dataset
   - ``minimal_config``        — valid minimal ``config.yaml`` dict (session)
   - ``mutable_config``        — deep-copied ``minimal_config`` (function)
   - ``mock_checkpoint``       — v2.0 checkpoint file on disk
   - ``isolated_dataset_registry`` — fresh ``DatasetRegistry()`` instance
   - ``tmp_working_dir``       — temporary ``working_root_dir`` directory
   - ``sample_mol_data``       — 5 synthetic molecule property dicts

2. **Marker registration** — registers project-wide custom markers
   (``smoke``, ``contract``, ``e2e``, ``regression``, ``thread_safety``,
   ``slow``) so that ``--strict-markers`` mode does not raise
   ``PytestUnknownMarkWarning``.

3. **Third-party warning suppression** — filters out ``DeprecationWarning``
   messages emitted internally by matplotlib <3.10.7, which uses deprecated
   camelCase pyparsing APIs (``oneOf``, ``parseString``, ``resetCache``,
   ``enablePackrat``).  These warnings originate in matplotlib's own code
   (``_fontconfig_pattern.py``, ``_mathtext.py``), not in MILIA source.

   Upstream issue : https://github.com/matplotlib/matplotlib/issues/30617
   Fixed in       : matplotlib >=3.10.7
   Action         : Remove the filterwarnings lines after upgrading matplotlib.

Design decisions (fixtures):
    - NO ``sys.modules`` mocking at module level (mock pollution prevention).
    - All heavy imports (``torch``, ``torch_geometric``, ``numpy``) are
      deferred inside fixtures to avoid collection-time failures.
    - Fixtures that produce mutable state use ``function`` scope (default) to
      ensure test isolation unless otherwise noted.
    - Session-scoped fixtures are used only for genuinely immutable / read-only
      data (e.g. ``minimal_config``).
    - ``tmp_path`` / ``tmp_path_factory`` from pytest is preferred over manual
      ``tempfile`` usage for automatic cleanup.

References
----------
- Marker registration via ``pytest_configure``:
  https://docs.pytest.org/en/stable/example/markers.html
- ``config.addinivalue_line("filterwarnings", ...)`` pattern:
  https://docs.pytest.org/en/stable/how-to/capture-warnings.html
- Section 7.1: ``MILIA_Test_Recommendations.md``

Author : MILIA Team
Version: 1.1.0
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
# When running from the project root (/app/milia) this makes
# ``import milia_pipeline`` succeed even without ``pip install -e .``.
# Layout: milia/tests/conftest.py → .parent = tests/ → .parent = milia/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging — helpful during fixture debugging
# ---------------------------------------------------------------------------
logger = logging.getLogger("conftest")


# ============================================================================
# FIXTURE 1: synthetic_pyg_dataset — Small synthetic PyG dataset
# ============================================================================
# Scope: function (each test gets a fresh copy — mutations are safe).
# Used by: smoke tests (§1.1, §1.4), E2E tests (§3.1, §3.3, §3.4),
#          regression tests (§4.2, §4.3).


@pytest.fixture()
def synthetic_pyg_dataset():
    """
    Create a small (30-graph) synthetic PyG dataset for training / prediction
    tests.

    Each graph has:
        - ``x``          : Node feature matrix  [num_nodes, 9]
        - ``edge_index`` : COO edge indices      [2, num_edges]
        - ``edge_attr``  : Edge feature matrix   [num_edges, 4]
        - ``y``          : Graph-level target     [1]  (regression)
        - ``pos``        : 3-D node positions     [num_nodes, 3]
        - ``z``          : Atomic numbers         [num_nodes]  (int64)
        - ``batch``      : (not set — added by DataLoader)

    The graphs are intentionally tiny (3-8 nodes) so that training smoke tests
    complete in < 1 s per epoch.

    Returns
    -------
    list[torch_geometric.data.Data]
        A Python list of 30 PyG ``Data`` objects.
    """
    import torch
    from torch_geometric.data import Data

    dataset: list[Data] = []
    generator = torch.Generator().manual_seed(42)

    for i in range(30):
        num_nodes = torch.randint(3, 9, (1,), generator=generator).item()

        # --- Node features (e.g. one-hot element + simple descriptors) ---
        x = torch.randn(num_nodes, 9, generator=generator)

        # --- Edges: random sparse graph (at least num_nodes edges) ---
        num_edges = torch.randint(
            num_nodes,
            num_nodes * 3 + 1,
            (1,),
            generator=generator,
        ).item()
        src = torch.randint(0, num_nodes, (num_edges,), generator=generator)
        dst = torch.randint(0, num_nodes, (num_edges,), generator=generator)
        edge_index = torch.stack([src, dst], dim=0)

        # --- Edge features ---
        edge_attr = torch.randn(num_edges, 4, generator=generator)

        # --- Graph-level regression target ---
        y = torch.randn(1, generator=generator)

        # --- 3-D positions (Å) ---
        pos = torch.randn(num_nodes, 3, generator=generator)

        # --- Atomic numbers (H=1, C=6, N=7, O=8 — simplified) ---
        z = torch.tensor(
            [1, 6, 7, 8][: min(num_nodes, 4)] * ((num_nodes // 4) + 1),
            dtype=torch.long,
        )[:num_nodes]

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            z=z,
        )
        dataset.append(data)

    logger.debug("synthetic_pyg_dataset: created %d graphs", len(dataset))
    return dataset


# ============================================================================
# FIXTURE 2: minimal_config — Valid minimal configuration dictionary
# ============================================================================
# Scope: session (immutable reference data — tests must deep-copy if mutating).
# Used by: smoke tests (§1.1), config tests (§5.1, §5.2),
#          E2E tests (§3.1, §3.2), contract tests (§2.3).


@pytest.fixture(scope="session")
def minimal_config() -> dict[str, Any]:
    """
    Return a minimal but structurally valid ``config.yaml``-equivalent dict
    that satisfies the MILIA configuration loader.

    The dict mirrors the schema documented in ``MILIA_Pipeline_Project_Structure.md``
    (§ Configuration System) and contains the minimum required keys for:
        - ``dataset_type``
        - ``global_paths.working_root_dir``
        - ``data_config.common_settings``
        - ``filter_config``
        - ``structural_features``
        - ``models``

    **Important**: this fixture is session-scoped.  If a test needs to mutate
    the config it **must** call ``copy.deepcopy(minimal_config)`` first, or
    use the ``mutable_config`` fixture instead.

    Returns
    -------
    dict
        A nested configuration dictionary.
    """
    return {
        "dataset_type": "DFT",
        "global_paths": {
            "working_root_dir": "/tmp/milia_test_workdir",
        },
        "data_config": {
            "common_settings": {
                "chunk_size": 10,
                "coordinate_units": "angstrom",
            },
            "property_selection": {
                "DFT": {
                    "energy": True,
                    "forces": True,
                },
            },
        },
        "filter_config": {
            "max_atoms": 100,
            "min_atoms": 1,
            "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl"],
        },
        "structural_features": {
            "atom_features": ["atomic_number", "degree"],
            "bond_features": ["bond_type"],
        },
        "transformations": {
            "experimental_setup": "default",
        },
        "models": {
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": {
                "hidden_channels": 32,
                "num_layers": 2,
                "out_channels": 1,
            },
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "scheduler": "reduce_on_plateau",
                "loss": "mse",
            },
            "evaluation": {
                "metrics": ["mae", "mse"],
                "visualization": {
                    "enabled": False,
                },
            },
        },
    }


# ============================================================================
# FIXTURE 3: mock_checkpoint — Pre-built v2.0 checkpoint file
# ============================================================================
# Scope: function (each test gets a fresh file in its own tmp directory).
# Used by: smoke tests (§1.4), E2E tests (§3.1, §3.4),
#          regression tests (§4.2, §4.3).


@pytest.fixture()
def mock_checkpoint(tmp_path):
    """
    Create a v2.0-format checkpoint file on disk using a small model.

    The checkpoint follows the format documented in
    ``MILIA_Pipeline_Project_Structure.md`` (§ Checkpoint Manager):
        - ``model_state_dict``
        - ``hyper_parameters``  (model name, hyperparams, task_type)
        - ``data_info``         (structural_features_config, edge features flags)
        - ``version_info``      (format_version = '2.0', python, torch versions)
        - ``epoch``, ``best_metric``

    The model is a genuine (tiny) ``torch.nn.Module`` so that
    ``state_dict`` / ``load_state_dict`` round-trips are valid.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory (auto-cleaned).

    Returns
    -------
    dict
        ``{"path": Path, "model": nn.Module, "checkpoint": dict}``
        where *path* is the ``.pt`` file, *model* is the original module,
        and *checkpoint* is the in-memory dict that was saved.
    """
    import torch
    import torch.nn as nn

    # --- Minimal GCN-like model (no PyG dependency for the mock) ---
    class _TinyModel(nn.Module):
        """Two-layer MLP standing in for a GCN in checkpoint tests."""

        def __init__(self, in_channels: int = 9, hidden_channels: int = 32, out_channels: int = 1):
            super().__init__()
            self.lin1 = nn.Linear(in_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, out_channels)

        def forward(self, x):
            return self.lin2(torch.relu(self.lin1(x)))

    model = _TinyModel(in_channels=9, hidden_channels=32, out_channels=1)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hyper_parameters": {
            "model_name": "GCN",
            "hidden_channels": 32,
            "num_layers": 2,
            "out_channels": 1,
            "in_channels": 9,
            "task_type": "graph_regression",
        },
        "data_info": {
            "requires_edge_features": False,
            "uses_edge_features": False,
            "structural_features_config": {
                "atom_features": ["atomic_number", "degree"],
                "bond_features": ["bond_type"],
            },
        },
        "version_info": {
            "format_version": "2.0",
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "torch_version": torch.__version__,
            "checkpoint_type": "best",
        },
        "epoch": 5,
        "best_metric": 0.042,
    }

    ckpt_path = tmp_path / "checkpoints" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, ckpt_path)

    logger.debug("mock_checkpoint: saved to %s", ckpt_path)
    return {
        "path": ckpt_path,
        "model": model,
        "checkpoint": checkpoint,
    }


# ============================================================================
# FIXTURE 4: isolated_dataset_registry — Fresh DatasetRegistry instance
# ============================================================================
# Scope: function (each test gets a pristine, empty registry).
# Used by: contract tests (§2.1, §2.2, §2.3), thread safety tests (§6.1).


@pytest.fixture()
def isolated_dataset_registry():
    """
    Provide a **fresh** ``DatasetRegistry`` instance that is completely
    isolated from the default global registry.

    This prevents cross-test pollution when registering / unregistering
    dataset classes.  The ``DatasetRegistry`` class is explicitly designed
    to be non-singleton for this purpose (see ``registry.py`` docstring).

    Returns
    -------
    milia_pipeline.datasets.registry.DatasetRegistry
        An empty registry instance.
    """
    from milia_pipeline.datasets.registry import DatasetRegistry

    registry = DatasetRegistry()
    logger.debug("isolated_dataset_registry: created empty DatasetRegistry")
    return registry


# ============================================================================
# FIXTURE 5: tmp_working_dir — Temporary working_root_dir for DI-pattern
# ============================================================================
# Scope: function (fresh directory per test).
# Used by: regression tests (§4.2, §4.3), E2E tests (§3.1, §3.4).


@pytest.fixture()
def tmp_working_dir(tmp_path):
    """
    Provide a temporary directory suitable as ``working_root_dir`` for
    MILIA's Dependency Injection-pattern post-training components
    (``CheckpointManager``, ``ModelLoader``, ``Predictor``, ``FineTuner``).

    Pre-creates the ``checkpoints/``, ``predictions/``, and ``logs/``
    sub-directories that several components expect to exist.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory (auto-cleaned).

    Returns
    -------
    pathlib.Path
        The working root directory.
    """
    work_dir = tmp_path / "milia_workdir"
    (work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (work_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger.debug("tmp_working_dir: %s", work_dir)
    return work_dir


# ============================================================================
# FIXTURE 6: sample_mol_data — Synthetic molecular data dicts
# ============================================================================
# Scope: function (mutable dicts — tests may alter them).
# Used by: E2E tests (§3.2), contract tests (§2.1).


@pytest.fixture()
def sample_mol_data() -> list[dict[str, Any]]:
    """
    Provide a list of 5 synthetic molecular data dictionaries that mimic
    the raw property dicts passed through the MILIA preprocessing pipeline.

    Each dict contains keys that a ``DatasetHandler`` would expect when
    calling ``validate_molecule_data`` / ``enrich_pyg_data``:
        - ``atoms``        : list of atomic numbers
        - ``coordinates``  : list of [x, y, z] positions (Å)
        - ``energy``       : total energy (Hartree)
        - ``forces``       : per-atom forces (Hartree/Å)
        - ``identifier``   : molecule identifier string
        - ``smiles``       : SMILES string (for identifier-based
                             molecule creation strategies)
        - ``inchi``        : InChI string

    Returns
    -------
    list[dict]
        Five synthetic molecule property dictionaries.
    """
    import numpy as np

    molecules: list[dict[str, Any]] = []

    # Molecule templates: (name, atoms, smiles, inchi)
    _templates = [
        ("water", [8, 1, 1], "O", "InChI=1S/H2O/h1H2"),
        ("methane", [6, 1, 1, 1, 1], "C", "InChI=1S/CH4/h1H4"),
        ("ammonia", [7, 1, 1, 1], "N", "InChI=1S/H3N/h1H3"),
        (
            "ethanol",
            [6, 6, 8, 1, 1, 1, 1, 1, 1],
            "CCO",
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        ),
        ("formaldehyde", [6, 8, 1, 1], "C=O", "InChI=1S/CH2O/c1-2/h1H2"),
    ]

    rng = np.random.default_rng(seed=12345)

    for idx, (name, atoms, smiles, inchi) in enumerate(_templates):
        n_atoms = len(atoms)
        coords = rng.standard_normal((n_atoms, 3)).tolist()
        forces = (rng.standard_normal((n_atoms, 3)) * 0.01).tolist()
        energy = float(rng.standard_normal() * 10.0 - 76.0)

        molecules.append(
            {
                "atoms": atoms,
                "coordinates": coords,
                "energy": energy,
                "forces": forces,
                "identifier": f"mol_{idx:04d}_{name}",
                "smiles": smiles,
                "inchi": inchi,
            }
        )

    logger.debug("sample_mol_data: created %d synthetic molecules", len(molecules))
    return molecules


# ============================================================================
# FIXTURE 7: mutable_config — Deep-copy convenience wrapper
# ============================================================================
# Thin wrapper that deep-copies the session-scoped ``minimal_config`` so that
# tests can freely mutate their copy.


@pytest.fixture()
def mutable_config(minimal_config) -> dict[str, Any]:
    """
    Return a deep copy of ``minimal_config`` that tests can safely mutate.

    This avoids the common footgun of accidentally modifying the
    session-scoped config and breaking subsequent tests.

    Returns
    -------
    dict
        A deep copy of the minimal configuration dictionary.
    """
    return copy.deepcopy(minimal_config)


# ============================================================================
# pytest_configure — Markers + Warning Filters
# ============================================================================


def pytest_configure(config):
    """Register custom markers and third-party warning filters.

    This hook runs at plugin registration time (before collection),
    which is the earliest point where both marker registration and
    ``filterwarnings`` injection take effect.

    ``config.addinivalue_line`` appends to the same ini-option lists
    that ``pytest.ini`` / ``pyproject.toml`` would populate, so the
    semantics are identical.

    References
    ----------
    - Markers : https://docs.pytest.org/en/stable/example/markers.html
    - Warnings: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
    """
    # -----------------------------------------------------------------
    # Custom markers
    # -----------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "smoke: Smoke tests — fast, first gate in CI/CD pipeline",
    )
    config.addinivalue_line(
        "markers",
        "contract: Interface contract validation tests",
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end workflow tests",
    )
    config.addinivalue_line(
        "markers",
        "regression: Regression protection tests",
    )
    config.addinivalue_line(
        "markers",
        "thread_safety: Concurrent access tests",
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests taking > 30 seconds",
    )

    # -----------------------------------------------------------------
    # Third-party warning filters
    # -----------------------------------------------------------------
    # matplotlib <3.10.7 internally uses deprecated pyparsing camelCase
    # APIs.  The warnings originate in matplotlib/_fontconfig_pattern.py
    # and matplotlib/_mathtext.py — not in MILIA source code.
    #
    # Tracked  : https://github.com/matplotlib/matplotlib/issues/30617
    # Fixed in : matplotlib >=3.10.7 (min pyparsing bumped to 3.0)
    # Action   : Remove these four lines after upgrading matplotlib.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:'oneOf' deprecated:DeprecationWarning:matplotlib",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:'parseString' deprecated:DeprecationWarning:matplotlib",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:'resetCache' deprecated:DeprecationWarning:matplotlib",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:'enablePackrat' deprecated:DeprecationWarning:matplotlib",
    )
