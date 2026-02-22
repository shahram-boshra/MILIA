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
# Dynamic collection filtering — skip test files with unimportable deps
# ---------------------------------------------------------------------------
# CI installs only pip dev extras (pytest, ruff, pytest-mock). Heavy scientific
# dependencies (torch, numpy, PyG, RDKit, pydantic, PyYAML) are conda-managed
# and NOT available in CI pip-only environments.
#
# pytest must *import* every test file during collection to discover markers,
# so ``-m smoke`` alone is insufficient — test files that import heavy deps at
# module level cause ``ModuleNotFoundError`` during collection.
#
# Solution: Two-phase import probing when core heavy deps are absent.
#
# **Phase 1 (fast)**: AST-based ``find_spec`` check for direct imports whose
# top-level package is entirely absent (e.g. ``import torch``).
#
# **Phase 2 (deep)**: For files that pass Phase 1 (their top-level packages
# are installed), attempt the actual import in a **subprocess** for every
# unique fully-qualified module path referenced in the file.  This catches
# **transitive** failures — e.g. ``from milia_pipeline.exceptions import X``
# where ``milia_pipeline`` is installed but its ``__init__.py`` eagerly
# imports ``yaml`` / ``pydantic`` / ``numpy`` which are NOT installed.
# Results are cached per module path so each subprocess runs at most once.
#
# A subprocess is required because ``__import__()`` in-process leaves
# partially-initialized modules in ``sys.modules`` (per Python docs:
# "any module that was successfully loaded as a side-effect must remain
# in the cache"), which corrupts subsequent import attempts.
#
# This approach is:
#   - **Non-breaking**: When all deps are installed, nothing is skipped.
#   - **Dynamic**: Automatically adapts to any new test file or dependency.
#   - **Future-proof**: No hard-coded file lists or module names.
#
# Official pytest docs on ``collect_ignore``:
#   https://docs.pytest.org/en/stable/example/pythoncollection.html
# ---------------------------------------------------------------------------
def _heavy_deps_available() -> bool:
    """Return True if core heavy dependencies are importable."""
    for mod_name in ("torch", "numpy", "yaml", "pydantic"):
        try:
            __import__(mod_name)
        except ImportError:
            return False
    return True


# Module-level cache: maps fully-qualified module path → bool (importable?)
_import_probe_cache: dict[str, bool] = {}


def _probe_import(module_path: str) -> bool:
    """Check if *module_path* is importable using a subprocess.

    A subprocess is used to avoid polluting ``sys.modules`` in the main
    pytest process with partially-initialized modules (Python import docs:
    side-effect modules remain in ``sys.modules`` even when the failing
    module is removed).

    Results are cached so each module is probed at most once per session.
    """
    if module_path in _import_probe_cache:
        return _import_probe_cache[module_path]

    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", f"import {module_path}"],
        capture_output=True,
        timeout=30,
    )
    _import_probe_cache[module_path] = result.returncode == 0
    return _import_probe_cache[module_path]


def _get_unimportable_test_files() -> list[str]:
    """Return list of test file paths whose imports cannot be resolved.

    Parses each ``test_*.py`` file's AST to find module-level ``import X``
    and ``from X import ...`` statements, then applies a two-phase probe:

    1. ``importlib.util.find_spec`` on the top-level package (fast reject).
    2. Subprocess ``import`` on the full module path (catches transitive
       failures without polluting the main process).

    Files whose imports are all resolvable are NOT included in the result.
    """
    import ast
    import glob as _glob
    import importlib.util

    tests_dir = Path(__file__).resolve().parent
    all_test_files = _glob.glob(str(tests_dir / "test_*.py"))
    unimportable = []

    for filepath in all_test_files:
        try:
            source = Path(filepath).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=filepath)
        except (SyntaxError, UnicodeDecodeError):
            # If we can't even parse the file, skip it during collection
            unimportable.append(filepath)
            continue

        skip = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_pkg = alias.name.split(".")[0]
                    # Phase 1: top-level package missing entirely
                    if importlib.util.find_spec(top_pkg) is None:
                        skip = True
                        break
                    # Phase 2: deep probe — full module path
                    if not _probe_import(alias.name):
                        skip = True
                        break
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                top_pkg = node.module.split(".")[0]
                # Phase 1: top-level package missing entirely
                if importlib.util.find_spec(top_pkg) is None or not _probe_import(node.module):
                    skip = True
            if skip:
                break

        if skip:
            unimportable.append(filepath)

    return unimportable


if not _heavy_deps_available():
    collect_ignore = _get_unimportable_test_files()

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

    for _i in range(30):
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
# PROTECTION: Prevent module reload/deletion AND registry clearing pollution
# ============================================================================
# ROOT CAUSE (Module pollution): Some test files use setup_module()/
# teardown_module() to inject mocks into sys.modules. During teardown they
# may delete or reload core modules. This causes:
#   - Category C: isinstance() fails (exception class identity mismatch)
#   - Category D: func.__globals__ points to stale module dict after reload
#   - KeyError: module removed from sys.modules entirely
#
# ROOT CAUSE (Registry pollution): Some test files/classes call .clear() on
# global registry singletons (dataset registry, descriptor registry) for test
# isolation but fail to restore the original contents, leaving subsequent
# tests with empty registries. This causes:
#   - Category B: list_all()/list_descriptors() returns [] → assertion failures
#
# FIX: After each test item's complete teardown phase:
#   1. Restore any deleted/reloaded core modules in sys.modules
#   2. Restore any cleared/mutated registry contents from snapshots


def _build_core_module_snapshot() -> dict:
    """Capture references to core modules that must not be reloaded/deleted.

    Only snapshots modules that are already loaded — does not force imports.
    """
    _CRITICAL_PREFIXES = (
        "milia_pipeline.exceptions",
        "milia_pipeline.datasets.registry",
        "milia_pipeline.datasets.base",
        "milia_pipeline.datasets.implementations",
        "milia_pipeline.descriptors",
        "milia_pipeline.models.registry",
        "milia_pipeline.transformations.plugin_system",
        "milia_pipeline.transformations.custom_transforms",
        "milia_pipeline.transformations.graph_transforms",
        "milia_pipeline.molecules.mol_conversion_utils",
    )
    snapshot = {}
    for key, mod in list(sys.modules.items()):
        if mod is not None and any(key == p or key.startswith(p + ".") for p in _CRITICAL_PREFIXES):
            snapshot[key] = mod
    return snapshot


def _build_registry_snapshot() -> dict:
    """Snapshot the contents of known global registry singletons.

    Returns a dict of {registry_name: snapshot_data} where snapshot_data
    contains enough information to detect and repair clearing.
    Only snapshots registries that are already loaded — does not force imports.
    """
    snapshots = {}

    # 1. Dataset registry: milia_pipeline.datasets.registry._default_registry
    try:
        reg_mod = sys.modules.get("milia_pipeline.datasets.registry")
        if reg_mod is not None:
            default_reg = getattr(reg_mod, "_default_registry", None)
            if default_reg is not None:
                datasets_dict = getattr(default_reg, "_datasets", None)
                if datasets_dict is not None and len(datasets_dict) > 0:
                    snapshots["dataset_registry"] = {
                        "obj": default_reg,
                        "attr": "_datasets",
                        "snapshot": dict(datasets_dict),
                    }
    except Exception:
        pass

    # 2. Descriptor registry: DescriptorRegistry singleton
    #    Module: milia_pipeline.descriptors.descriptor_registry
    #    Uses __new__-based singleton with _instances class dict.
    #    Internal storage: _descriptors (dict), _by_category (defaultdict)
    try:
        desc_mod = sys.modules.get("milia_pipeline.descriptors.descriptor_registry")
        if desc_mod is not None:
            desc_registry_cls = getattr(desc_mod, "DescriptorRegistry", None)
            if desc_registry_cls is not None:
                instances = getattr(desc_registry_cls, "_instances", None)
                if instances and desc_registry_cls in instances:
                    instance = instances[desc_registry_cls]
                    descriptors_dict = getattr(instance, "_descriptors", None)
                    by_category = getattr(instance, "_by_category", None)
                    if descriptors_dict is not None and len(descriptors_dict) > 0:
                        snapshots["descriptor_registry"] = {
                            "obj": instance,
                            "attr": "_descriptors",
                            "snapshot": dict(descriptors_dict),
                        }
                    if by_category is not None and len(by_category) > 0:
                        snapshots["descriptor_by_category"] = {
                            "obj": instance,
                            "attr": "_by_category",
                            "snapshot": {k: set(v) for k, v in by_category.items()},
                        }
    except Exception:
        pass

    # 3. Model registry: ModelRegistry singleton
    #    Likely same __new__-based singleton pattern with _instances.
    try:
        model_mod = sys.modules.get("milia_pipeline.models.registry.model_registry")
        if model_mod is not None:
            model_registry_cls = getattr(model_mod, "ModelRegistry", None)
            if model_registry_cls is not None:
                # Try _instances dict pattern (same as DescriptorRegistry)
                instances = getattr(model_registry_cls, "_instances", None)
                instance = None
                if instances and model_registry_cls in instances:
                    instance = instances[model_registry_cls]
                # Fallback: try _instance attribute
                if instance is None:
                    instance = getattr(model_registry_cls, "_instance", None)
                if instance is not None:
                    # Try common internal dict attribute names
                    for attr_name in ("_models", "_registry", "_registered"):
                        internal = getattr(instance, attr_name, None)
                        if isinstance(internal, dict) and len(internal) > 0:
                            snapshots["model_registry"] = {
                                "obj": instance,
                                "attr": attr_name,
                                "snapshot": dict(internal),
                            }
                            break
    except Exception:
        pass

    return snapshots


def _restore_registries(snapshots: dict) -> None:
    """Restore registry contents if they were cleared since the snapshot."""
    for _name, info in snapshots.items():
        obj = info["obj"]
        attr = info["attr"]
        original = info["snapshot"]
        current = getattr(obj, attr, None)
        if current is not None and len(current) == 0 and len(original) > 0:
            # Registry was cleared — restore the snapshot.
            # For defaultdict(set), copy the set values to avoid sharing refs.
            for k, v in original.items():
                current[k] = copy.copy(v) if isinstance(v, set) else v


# Populated after collection, before test execution
_CORE_MODULE_SNAPSHOT: dict = {}
_REGISTRY_SNAPSHOT: dict = {}


def pytest_collection_finish(session):
    """Hook: called after collection, before test execution.

    At this point all test modules have been collected, which triggers imports
    of the modules-under-test. We snapshot the core modules and registry
    contents NOW so they contain the real, fully-initialized state.
    """
    global _CORE_MODULE_SNAPSHOT, _REGISTRY_SNAPSHOT
    _CORE_MODULE_SNAPSHOT = _build_core_module_snapshot()
    _REGISTRY_SNAPSHOT = _build_registry_snapshot()


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Hook (wrapper): runs AFTER each test item's full teardown phase.

    By using hookwrapper=True, the ``yield`` waits for the entire teardown
    phase to complete — including ``teardown_module()`` for the last test
    in a module, ``tearDown()`` for unittest classes, and fixture finalizers.
    Only then does it check and repair pollution.
    """
    yield
    global _CORE_MODULE_SNAPSHOT, _REGISTRY_SNAPSHOT

    # Lazily build snapshots if not yet populated (modules may not have
    # been imported during collection but are now loaded).
    if not _CORE_MODULE_SNAPSHOT:
        _CORE_MODULE_SNAPSHOT = _build_core_module_snapshot()
    if not _REGISTRY_SNAPSHOT:
        _REGISTRY_SNAPSHOT = _build_registry_snapshot()

    # Post-teardown 1: restore any removed/reloaded core modules
    if _CORE_MODULE_SNAPSHOT:
        for key, original_mod in _CORE_MODULE_SNAPSHOT.items():
            current = sys.modules.get(key)
            if current is not original_mod:
                sys.modules[key] = original_mod
    # Post-teardown 2: restore any cleared registry contents
    if _REGISTRY_SNAPSHOT:
        _restore_registries(_REGISTRY_SNAPSHOT)


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
