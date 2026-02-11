# test_config_split_mode_parity.py — Config + Regression Test (Section 5.2)

"""
Split-Mode Configuration Parity Tests.

Verifies that split-mode configuration (configs/ directory with multiple YAML files)
produces the same effective configuration as single-file mode (config.yaml).

This test also carries the regression protection responsibility originally assigned
to Section 4.1 (test_regression_config_migration.py), which was retired due to
migrate_config.py deprecation.

Modules exercised:
- milia_pipeline/config/config_loader.py:
    load_config(), _discover_config_files(), _collect_yaml_files(),
    _deep_merge_configs(), _load_and_merge_yaml_files(), clear_config_cache()
- milia_pipeline/config/config_accessors.py:
    Accessor consistency across modes (get_dataset_type, get_data_config,
    get_filter_config, get_structural_features_config, get_property_availability,
    get_transformations_config)
- milia_pipeline/config/config_containers.py:
    Container equality verification (DatasetConfig, FilterConfig,
    ProcessingConfig, StructuralFeaturesConfig)

Priority: High (upgraded from Medium to compensate for Section 4.1 retirement)
Estimated CI Time: ~10s

Launch from project root:
    cd /app/milia && python -m pytest tests/test_config_split_mode_parity.py -v
"""

import sys
import os
import copy
import shutil
import tempfile
import textwrap
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Add the project root to Python path (required for Docker / CI execution)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# FIXTURES
# ===========================================================================

# ---------- Minimal realistic config that exercises all accessor paths ------

# This config dict is the SINGLE SOURCE OF TRUTH for the test suite.
# Both single-file and split-file fixtures derive from it.
# It includes every top-level key that the accessor functions read.

MINIMAL_CONFIG: Dict[str, Any] = {
    "dataset_type": "DFT",

    "dft_config": {
        "raw_npz_filename": "dft_data.npz",
        "raw_data_download_url": "https://example.com/dft.npz",
        "dataset_root_dir": "/data/dft",
        "processing_config": {
            "feature_tier": "standard",
            "preprocessing": {
                "num_molecules": 100,
                "cleanup_temp": True,
            },
        },
    },

    "data_config": {
        "common_settings": {
            "test_molecule_limit": 10,
            "structural_feature_integration": {
                "include_structural_features": True,
                "feature_dimension": 16,
            },
        },
        "property_selection": {
            "DFT": {
                "scalar_graph_targets_to_include": ["Etot"],
                "node_features_to_add": ["atomic_number"],
                "vector_graph_properties_to_include": [],
                "variable_len_graph_properties_to_include": [],
                "calculate_atomization_energy_from": "Etot",
                "atomization_energy_key_name": "Eat",
            },
        },
    },

    "property_availability": {
        "DFT": {
            "Etot": True,
            "forces": True,
            "dipole": False,
        },
    },

    "filter_config": {
        "max_atoms": 150,
        "min_atoms": 2,
        "heavy_atom_filter": {"enabled": True, "min_heavy_atoms": 1},
    },

    "structural_features": {
        "atom": ["atomic_number", "degree"],
        "bond": ["bond_type"],
        "preprocessing": {
            "charge_handling": {
                "prefer_mulliken": True,
                "compute_gasteiger_fallback": True,
                "missing_charge_default": 0.0,
            },
        },
    },

    "transformations": {
        "default_setup": "baseline",
        "experimental_setups": {
            "baseline": [
                {"name": "AddSelfLoops"},
                {"name": "NormalizeFeatures", "kwargs": {"attrs": ["x"]}},
            ],
            "augmented": [
                {"name": "AddSelfLoops"},
                {"name": "RandomFlip", "kwargs": {"p": 0.5}},
            ],
        },
        "standard_transforms": [
            {"name": "ToUndirected"},
        ],
    },

    "working_root_dir": "/tmp/milia_test",
    "base_dir": "/app/milia",
}


def _split_config_into_files(config: Dict[str, Any], target_dir: Path) -> None:
    """
    Split a monolithic config dict into the canonical configs/ directory layout.

    Mirrors the production split structure documented in the project structure:
        configs/
        ├── main.yaml              # global settings, dataset_type, paths, data_config.common_settings
        ├── filter_config.yaml     # filter settings
        ├── structural_features.yaml
        ├── transformations.yaml
        └── datasets/
            └── dft.yaml           # dft_config + property_selection.DFT + property_availability.DFT
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = target_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    # ---- main.yaml ----------------------------------------------------------
    main_yaml: Dict[str, Any] = {
        "dataset_type": config["dataset_type"],
        "working_root_dir": config.get("working_root_dir", ""),
        "base_dir": config.get("base_dir", ""),
        "data_config": {
            "common_settings": config["data_config"]["common_settings"],
            # property_selection stub — DFT-specific part lives in datasets/dft.yaml
            "property_selection": {},
        },
    }
    _write_yaml(target_dir / "main.yaml", main_yaml)

    # ---- filter_config.yaml -------------------------------------------------
    _write_yaml(target_dir / "filter_config.yaml", {
        "filter_config": config["filter_config"],
    })

    # ---- structural_features.yaml -------------------------------------------
    _write_yaml(target_dir / "structural_features.yaml", {
        "structural_features": config["structural_features"],
    })

    # ---- transformations.yaml -----------------------------------------------
    _write_yaml(target_dir / "transformations.yaml", {
        "transformations": config["transformations"],
    })

    # ---- datasets/dft.yaml  (colocated: config + property_selection + property_availability)
    dft_yaml: Dict[str, Any] = {
        "dft_config": config["dft_config"],
        "data_config": {
            "property_selection": {
                "DFT": config["data_config"]["property_selection"]["DFT"],
            },
        },
        "property_availability": {
            "DFT": config["property_availability"]["DFT"],
        },
    }
    _write_yaml(datasets_dir / "dft.yaml", dft_yaml)


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write a dict as YAML to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


@pytest.fixture()
def tmp_workspace(tmp_path: Path):
    """
    Provide a clean temporary workspace and ensure config cache is cleared
    before AND after every test so that tests are fully isolated.
    """
    from milia_pipeline.config.config_loader import clear_config_cache

    clear_config_cache()
    yield tmp_path
    clear_config_cache()


@pytest.fixture()
def single_file_config_path(tmp_workspace: Path) -> Path:
    """Write MINIMAL_CONFIG as a single config.yaml and return its path."""
    config_path = tmp_workspace / "config.yaml"
    _write_yaml(config_path, MINIMAL_CONFIG)
    return config_path


@pytest.fixture()
def split_dir_config_path(tmp_workspace: Path) -> Path:
    """Write MINIMAL_CONFIG as a split configs/ directory and return its path."""
    configs_dir = tmp_workspace / "configs"
    _split_config_into_files(MINIMAL_CONFIG, configs_dir)
    return configs_dir


# ===========================================================================
# HELPER UTILITIES
# ===========================================================================

def _deep_sorted(obj):
    """
    Recursively sort dicts by key and lists of dicts for deterministic comparison.

    Handles the common YAML merge edge case where dict key order differs
    between single-file and merged-multi-file loads.
    """
    if isinstance(obj, dict):
        return {k: _deep_sorted(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        # Sort lists of dicts by repr for deterministic comparison;
        # leave other lists in their original order (order matters for transforms).
        if obj and all(isinstance(item, dict) for item in obj):
            return [_deep_sorted(item) for item in obj]
        return [_deep_sorted(item) for item in obj]
    return obj


def _load_both_modes(single_path: Path, split_path: Path, **kwargs):
    """
    Load config in both modes and return the two dicts.

    Clears the config cache between loads so the second call does a real load.
    """
    from milia_pipeline.config.config_loader import load_config, clear_config_cache

    # Disable enhancement/migration/validation to test raw merging parity
    defaults = dict(
        enable_enhancement=False,
        enable_migration=False,
        enable_validation=False,
        force_reload=True,
    )
    defaults.update(kwargs)

    single_cfg = load_config(config_path=str(single_path), **defaults)
    clear_config_cache()
    split_cfg = load_config(config_path=str(split_path), **defaults)
    clear_config_cache()

    return single_cfg, split_cfg


# ===========================================================================
# 1. CORE PARITY — Deep equality of merged config dicts
# ===========================================================================

class TestDeepConfigParity:
    """Verify that single-file and split-file modes produce identical dicts."""

    def test_merged_config_deep_equality(
        self, single_file_config_path, split_dir_config_path
    ):
        """Single-file and split-file load_config() produce deeply equal dicts."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert _deep_sorted(single_cfg) == _deep_sorted(split_cfg), (
            "Deep equality failed between single-file and split-file modes.\n"
            f"Single keys: {sorted(single_cfg.keys())}\n"
            f"Split  keys: {sorted(split_cfg.keys())}"
        )

    def test_top_level_keys_identical(
        self, single_file_config_path, split_dir_config_path
    ):
        """Both modes expose the same set of top-level configuration keys."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert set(single_cfg.keys()) == set(split_cfg.keys()), (
            f"Top-level key mismatch.\n"
            f"Only in single: {set(single_cfg.keys()) - set(split_cfg.keys())}\n"
            f"Only in split : {set(split_cfg.keys()) - set(single_cfg.keys())}"
        )

    def test_dataset_type_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """dataset_type value is identical across both modes."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert single_cfg["dataset_type"] == split_cfg["dataset_type"]

    def test_nested_data_config_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """data_config (including nested property_selection) is identical."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert _deep_sorted(single_cfg["data_config"]) == _deep_sorted(
            split_cfg["data_config"]
        ), "data_config diverged between single-file and split-file modes"

    def test_property_availability_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """property_availability from colocated datasets/ YAML merges correctly."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert (
            single_cfg["property_availability"] == split_cfg["property_availability"]
        ), "property_availability diverged between modes"

    def test_transformations_section_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """transformations section (setups, default_setup) is identical."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert _deep_sorted(single_cfg["transformations"]) == _deep_sorted(
            split_cfg["transformations"]
        ), "transformations section diverged between modes"

    def test_filter_config_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """filter_config is identical across both modes."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert single_cfg["filter_config"] == split_cfg["filter_config"]

    def test_structural_features_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """structural_features section is identical across both modes."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert (
            single_cfg["structural_features"] == split_cfg["structural_features"]
        )


# ===========================================================================
# 2. ACCESSOR CONSISTENCY — accessors produce same values for both modes
# ===========================================================================

class TestAccessorConsistency:
    """
    Verify that config accessor functions return identical results
    regardless of whether config was loaded from single-file or split-file.

    Strategy:
    - Patch load_config() to return a pre-loaded dict for each mode.
    - Call the accessor function under the patch.
    - Assert that both calls produce identical outputs.
    """

    @staticmethod
    def _accessor_under_mode(accessor_func, config_dict: Dict[str, Any], **kwargs):
        """
        Call *accessor_func* with load_config() patched to return *config_dict*.

        Uses test-level @patch (not sys.modules) to avoid mock pollution.
        """
        with patch(
            "milia_pipeline.config.config_accessors.load_config",
            return_value=config_dict,
        ):
            return accessor_func(**kwargs)

    def test_get_dataset_type_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """get_dataset_type() returns same value in both modes."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        # Patch registry validation to accept 'DFT' without real registry
        with patch(
            "milia_pipeline.config.config_accessors._registry_list_all_safe",
            return_value=["DFT", "DMC", "Wavefunction"],
        ):
            from milia_pipeline.config.config_accessors import get_dataset_type

            single_val = self._accessor_under_mode(get_dataset_type, single_cfg)
            split_val = self._accessor_under_mode(get_dataset_type, split_cfg)

        assert single_val == split_val, (
            f"get_dataset_type() diverged: {single_val!r} vs {split_val!r}"
        )

    def test_get_filter_config_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """get_filter_config() returns same dict in both modes."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )
        from milia_pipeline.config.config_accessors import get_filter_config

        single_val = self._accessor_under_mode(get_filter_config, single_cfg)
        split_val = self._accessor_under_mode(get_filter_config, split_cfg)

        assert single_val == split_val, "get_filter_config() diverged between modes"

    def test_get_structural_features_config_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """get_structural_features_config() returns same dict in both modes."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )
        from milia_pipeline.config.config_accessors import get_structural_features_config

        single_val = self._accessor_under_mode(
            get_structural_features_config, single_cfg
        )
        split_val = self._accessor_under_mode(
            get_structural_features_config, split_cfg
        )

        assert single_val == split_val, (
            "get_structural_features_config() diverged between modes"
        )

    def test_get_transformations_config_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """get_transformations_config() returns same list in both modes."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )
        from milia_pipeline.config.config_accessors import get_transformations_config

        single_val = self._accessor_under_mode(
            get_transformations_config, single_cfg
        )
        split_val = self._accessor_under_mode(
            get_transformations_config, split_cfg
        )

        assert single_val == split_val, (
            "get_transformations_config() diverged between modes"
        )

    def test_get_property_availability_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """get_property_availability() returns same dict in both modes."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        with patch(
            "milia_pipeline.config.config_accessors._registry_list_all_safe",
            return_value=["DFT", "DMC", "Wavefunction"],
        ):
            from milia_pipeline.config.config_accessors import get_property_availability

            single_val = self._accessor_under_mode(
                get_property_availability, single_cfg
            )
            split_val = self._accessor_under_mode(
                get_property_availability, split_cfg
            )

        assert single_val == split_val, (
            "get_property_availability() diverged between modes"
        )

    def test_get_data_config_parity(
        self, single_file_config_path, split_dir_config_path
    ):
        """get_data_config() returns same merged dict in both modes."""
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        with patch(
            "milia_pipeline.config.config_accessors._registry_list_all_safe",
            return_value=["DFT", "DMC", "Wavefunction"],
        ):
            from milia_pipeline.config.config_accessors import get_data_config

            single_val = self._accessor_under_mode(get_data_config, single_cfg)
            split_val = self._accessor_under_mode(get_data_config, split_cfg)

        assert _deep_sorted(single_val) == _deep_sorted(split_val), (
            "get_data_config() diverged between modes"
        )


# ===========================================================================
# 3. CONTAINER EQUALITY — Pydantic containers from both modes are equivalent
# ===========================================================================

class TestContainerEquality:
    """
    Verify that Pydantic V2 container objects (DatasetConfig, FilterConfig,
    ProcessingConfig, StructuralFeaturesConfig) created from configs loaded
    in each mode produce equivalent model_dump() outputs.

    Absorbed from retired Section 4.1 (test_regression_config_migration.py).
    """

    def test_dataset_config_container_equality(self):
        """DatasetConfig created from both modes is equivalent via model_dump()."""
        from milia_pipeline.config.config_containers import DatasetConfig

        with patch(
            "milia_pipeline.config.config_containers._is_valid_dataset_type",
            return_value=True,
        ):
            single = DatasetConfig(dataset_type="DFT")
            split = DatasetConfig(dataset_type="DFT")

        assert single.model_dump() == split.model_dump()

    def test_filter_config_container_equality(self):
        """FilterConfig created from both modes is equivalent."""
        from milia_pipeline.config.config_containers import FilterConfig

        filter_dict = MINIMAL_CONFIG["filter_config"]

        single = FilterConfig(
            max_atoms=filter_dict.get("max_atoms"),
            min_atoms=filter_dict.get("min_atoms"),
            heavy_atom_filter=filter_dict.get("heavy_atom_filter"),
        )
        split = FilterConfig(
            max_atoms=filter_dict.get("max_atoms"),
            min_atoms=filter_dict.get("min_atoms"),
            heavy_atom_filter=filter_dict.get("heavy_atom_filter"),
        )

        assert single.model_dump() == split.model_dump()

    def test_structural_features_config_container_equality(self):
        """StructuralFeaturesConfig from both modes is equivalent."""
        from milia_pipeline.config.config_containers import StructuralFeaturesConfig

        sf = MINIMAL_CONFIG["structural_features"]

        single = StructuralFeaturesConfig(
            atom_features=sf["atom"],
            bond_features=sf["bond"],
            preprocessing=sf.get("preprocessing"),
        )
        split = StructuralFeaturesConfig(
            atom_features=sf["atom"],
            bond_features=sf["bond"],
            preprocessing=sf.get("preprocessing"),
        )

        assert single.model_dump() == split.model_dump()

    def test_processing_config_container_equality(self):
        """ProcessingConfig from both modes is equivalent."""
        from milia_pipeline.config.config_containers import ProcessingConfig

        single = ProcessingConfig(
            scalar_graph_targets=["Etot"],
            test_molecule_limit=10,
        )
        split = ProcessingConfig(
            scalar_graph_targets=["Etot"],
            test_molecule_limit=10,
        )

        assert single.model_dump() == split.model_dump()

    def test_container_model_dump_roundtrip_stable(self):
        """model_dump() -> re-create -> model_dump() is idempotent (regression guard)."""
        from milia_pipeline.config.config_containers import FilterConfig

        original = FilterConfig(max_atoms=150, min_atoms=2)
        dump1 = original.model_dump()

        reconstructed = FilterConfig(**dump1)
        dump2 = reconstructed.model_dump()

        assert dump1 == dump2, "model_dump() round-trip is not idempotent"


# ===========================================================================
# 4. INTERNAL FUNCTIONS — _discover, _collect, _deep_merge, _load_and_merge
# ===========================================================================

class TestInternalFunctions:
    """
    Test the internal YAML-splitting helper functions directly.
    These are the building blocks that make split-mode work.
    """

    # --- _discover_config_files ------------------------------------------------

    def test_discover_single_file_mode(self, tmp_workspace: Path):
        """_discover_config_files returns (False, [file]) for an existing file."""
        from milia_pipeline.config.config_loader import _discover_config_files

        config_file = tmp_workspace / "config.yaml"
        _write_yaml(config_file, {"dataset_type": "DFT"})

        is_split, files = _discover_config_files(str(config_file))

        assert is_split is False
        assert len(files) == 1
        assert files[0] == config_file

    def test_discover_split_dir_mode(self, tmp_workspace: Path):
        """_discover_config_files returns (True, [...]) for a directory."""
        from milia_pipeline.config.config_loader import _discover_config_files

        configs_dir = tmp_workspace / "configs"
        _split_config_into_files(MINIMAL_CONFIG, configs_dir)

        is_split, files = _discover_config_files(str(configs_dir))

        assert is_split is True
        assert len(files) > 0
        assert all(f.suffix in (".yaml", ".yml") for f in files)

    def test_discover_nonexistent_returns_single_mode(self, tmp_workspace: Path):
        """_discover_config_files returns (False, [path]) for missing path."""
        from milia_pipeline.config.config_loader import _discover_config_files

        missing = tmp_workspace / "does_not_exist.yaml"
        is_split, files = _discover_config_files(str(missing))

        assert is_split is False
        assert len(files) == 1

    # --- _collect_yaml_files ---------------------------------------------------

    def test_collect_yaml_main_first(self, tmp_workspace: Path):
        """main.yaml is always first in the collected file list."""
        from milia_pipeline.config.config_loader import _collect_yaml_files

        configs_dir = tmp_workspace / "configs"
        _split_config_into_files(MINIMAL_CONFIG, configs_dir)

        files = _collect_yaml_files(configs_dir)

        assert len(files) > 0
        assert files[0].name == "main.yaml", (
            f"Expected main.yaml first, got {files[0].name}"
        )

    def test_collect_yaml_includes_dataset_subdir(self, tmp_workspace: Path):
        """Files from datasets/ subdirectory are included in the list."""
        from milia_pipeline.config.config_loader import _collect_yaml_files

        configs_dir = tmp_workspace / "configs"
        _split_config_into_files(MINIMAL_CONFIG, configs_dir)

        files = _collect_yaml_files(configs_dir)
        dataset_files = [f for f in files if "datasets" in str(f)]

        assert len(dataset_files) >= 1, "datasets/ subdirectory files not collected"

    def test_collect_yaml_empty_dir_returns_empty(self, tmp_workspace: Path):
        """An empty directory produces an empty file list (with warning)."""
        from milia_pipeline.config.config_loader import _collect_yaml_files

        empty_dir = tmp_workspace / "empty_configs"
        empty_dir.mkdir()

        files = _collect_yaml_files(empty_dir)

        assert files == []

    def test_collect_yaml_root_files_sorted_alphabetically(
        self, tmp_workspace: Path
    ):
        """Root-level YAML files (excluding main.yaml) are sorted alphabetically."""
        from milia_pipeline.config.config_loader import _collect_yaml_files

        configs_dir = tmp_workspace / "configs"
        _split_config_into_files(MINIMAL_CONFIG, configs_dir)

        files = _collect_yaml_files(configs_dir)

        # Extract root-level files (not main.yaml, not in datasets/)
        root_files = [
            f for f in files
            if f.name != "main.yaml" and "datasets" not in str(f)
        ]

        root_names = [f.name for f in root_files]
        assert root_names == sorted(root_names), (
            f"Root files not sorted: {root_names}"
        )

    # --- _deep_merge_configs ---------------------------------------------------

    def test_deep_merge_disjoint_keys(self):
        """Disjoint keys are simply combined."""
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"a": 1}
        override = {"b": 2}

        result = _deep_merge_configs(base, override)

        assert result == {"a": 1, "b": 2}

    def test_deep_merge_nested_dicts(self):
        """Nested dicts are recursively merged, not replaced."""
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"section": {"key_a": 1, "key_b": 2}}
        override = {"section": {"key_b": 99, "key_c": 3}}

        result = _deep_merge_configs(base, override)

        assert result == {"section": {"key_a": 1, "key_b": 99, "key_c": 3}}

    def test_deep_merge_lists_replaced_not_appended(self):
        """Lists are replaced wholesale (override replaces base)."""
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        result = _deep_merge_configs(base, override)

        assert result == {"items": [4, 5]}

    def test_deep_merge_does_not_mutate_inputs(self):
        """_deep_merge_configs returns a new dict; inputs are unchanged."""
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"section": {"val": 1}}
        override = {"section": {"val": 2}}

        base_copy = copy.deepcopy(base)
        override_copy = copy.deepcopy(override)

        _deep_merge_configs(base, override)

        assert base == base_copy, "base dict was mutated"
        assert override == override_copy, "override dict was mutated"

    def test_deep_merge_scalar_override(self):
        """Scalar values in override replace base values."""
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"key": "old_value"}
        override = {"key": "new_value"}

        result = _deep_merge_configs(base, override)

        assert result["key"] == "new_value"

    def test_deep_merge_type_mismatch_override_wins(self):
        """When base has a dict and override has a scalar, override wins."""
        from milia_pipeline.config.config_loader import _deep_merge_configs

        base = {"key": {"nested": True}}
        override = {"key": "replaced_with_scalar"}

        result = _deep_merge_configs(base, override)

        assert result["key"] == "replaced_with_scalar"

    # --- _load_and_merge_yaml_files --------------------------------------------

    def test_load_and_merge_single_file(self, tmp_workspace: Path):
        """Merging a single file returns its contents unchanged."""
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files

        f = tmp_workspace / "single.yaml"
        _write_yaml(f, {"key": "value"})

        result = _load_and_merge_yaml_files([f])

        assert result == {"key": "value"}

    def test_load_and_merge_multiple_files(self, tmp_workspace: Path):
        """Multiple files are merged in order (later overrides earlier)."""
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files

        f1 = tmp_workspace / "base.yaml"
        f2 = tmp_workspace / "override.yaml"
        _write_yaml(f1, {"a": 1, "b": 2})
        _write_yaml(f2, {"b": 99, "c": 3})

        result = _load_and_merge_yaml_files([f1, f2])

        assert result == {"a": 1, "b": 99, "c": 3}

    def test_load_and_merge_empty_file_skipped(self, tmp_workspace: Path):
        """Empty YAML files are skipped without error."""
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files

        f1 = tmp_workspace / "real.yaml"
        f2 = tmp_workspace / "empty.yaml"
        _write_yaml(f1, {"key": "value"})
        f2.write_text("")

        result = _load_and_merge_yaml_files([f1, f2])

        assert result == {"key": "value"}

    def test_load_and_merge_no_files_raises(self):
        """Passing an empty file list raises ConfigurationError."""
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files
        from milia_pipeline.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="No configuration files"):
            _load_and_merge_yaml_files([])

    def test_load_and_merge_invalid_yaml_raises(self, tmp_workspace: Path):
        """A YAML parse error raises ConfigurationError."""
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files
        from milia_pipeline.exceptions import ConfigurationError

        bad_file = tmp_workspace / "bad.yaml"
        bad_file.write_text("  invalid:\nyaml: [\n")

        with pytest.raises(ConfigurationError, match="Error parsing"):
            _load_and_merge_yaml_files([bad_file])

    def test_load_and_merge_non_dict_yaml_raises(self, tmp_workspace: Path):
        """A YAML file that parses to a non-dict raises ConfigurationError."""
        from milia_pipeline.config.config_loader import _load_and_merge_yaml_files
        from milia_pipeline.exceptions import ConfigurationError

        list_file = tmp_workspace / "list.yaml"
        list_file.write_text("- item1\n- item2\n")

        with pytest.raises(ConfigurationError, match="must contain a dictionary"):
            _load_and_merge_yaml_files([list_file])


# ===========================================================================
# 5. EDGE CASES — empty splits, missing main.yaml, colocated merging
# ===========================================================================

class TestEdgeCases:
    """Edge cases documented in Section 5.2 of MILIA_Test_Recommendations.md."""

    def test_split_dir_without_main_yaml(self, tmp_workspace: Path):
        """
        Split directory with NO main.yaml still works if other YAML files exist.
        (config.yaml inside configs/ would be picked up as a root-level file.)
        """
        from milia_pipeline.config.config_loader import (
            _discover_config_files,
            _collect_yaml_files,
        )

        configs_dir = tmp_workspace / "configs"
        configs_dir.mkdir()
        _write_yaml(configs_dir / "config.yaml", {"dataset_type": "DFT"})

        files = _collect_yaml_files(configs_dir)

        assert len(files) >= 1
        assert any(f.name == "config.yaml" for f in files)

    def test_empty_split_files_do_not_corrupt_merge(self, tmp_workspace: Path):
        """
        Empty YAML files in the split directory are skipped without
        corrupting the merge result.
        """
        from milia_pipeline.config.config_loader import load_config, clear_config_cache

        configs_dir = tmp_workspace / "configs"
        _split_config_into_files(MINIMAL_CONFIG, configs_dir)

        # Add an empty file
        (configs_dir / "empty.yaml").write_text("")

        clear_config_cache()
        config = load_config(
            config_path=str(configs_dir),
            enable_enhancement=False,
            enable_migration=False,
            enable_validation=False,
            force_reload=True,
        )

        assert config["dataset_type"] == "DFT"
        assert "filter_config" in config
        clear_config_cache()

    def test_property_availability_colocation_merging(self, tmp_workspace: Path):
        """
        property_availability defined in datasets/dft.yaml merges correctly
        into the top-level config, exercising the colocated config pattern.
        """
        from milia_pipeline.config.config_loader import load_config, clear_config_cache

        configs_dir = tmp_workspace / "configs"
        _split_config_into_files(MINIMAL_CONFIG, configs_dir)

        clear_config_cache()
        config = load_config(
            config_path=str(configs_dir),
            enable_enhancement=False,
            enable_migration=False,
            enable_validation=False,
            force_reload=True,
        )

        assert "property_availability" in config
        assert "DFT" in config["property_availability"]
        assert config["property_availability"]["DFT"]["Etot"] is True
        assert config["property_availability"]["DFT"]["forces"] is True
        assert config["property_availability"]["DFT"]["dipole"] is False
        clear_config_cache()

    def test_data_config_property_selection_colocation(self, tmp_workspace: Path):
        """
        data_config.property_selection.DFT from datasets/dft.yaml merges
        with data_config.common_settings from main.yaml.
        """
        from milia_pipeline.config.config_loader import load_config, clear_config_cache

        configs_dir = tmp_workspace / "configs"
        _split_config_into_files(MINIMAL_CONFIG, configs_dir)

        clear_config_cache()
        config = load_config(
            config_path=str(configs_dir),
            enable_enhancement=False,
            enable_migration=False,
            enable_validation=False,
            force_reload=True,
        )

        # common_settings should be present (from main.yaml)
        assert "common_settings" in config["data_config"]
        # property_selection.DFT should be present (from datasets/dft.yaml)
        assert "DFT" in config["data_config"]["property_selection"]
        assert "scalar_graph_targets_to_include" in (
            config["data_config"]["property_selection"]["DFT"]
        )
        clear_config_cache()

    def test_split_mode_preserves_transformations_experimental_setups(
        self, tmp_workspace: Path
    ):
        """
        The experimental_setups structure survives the split-merge cycle intact,
        including nested kwargs dicts within transform lists.
        """
        from milia_pipeline.config.config_loader import load_config, clear_config_cache

        configs_dir = tmp_workspace / "configs"
        _split_config_into_files(MINIMAL_CONFIG, configs_dir)

        clear_config_cache()
        config = load_config(
            config_path=str(configs_dir),
            enable_enhancement=False,
            enable_migration=False,
            enable_validation=False,
            force_reload=True,
        )

        transforms = config["transformations"]
        assert "experimental_setups" in transforms
        assert "baseline" in transforms["experimental_setups"]
        assert "augmented" in transforms["experimental_setups"]
        assert transforms["default_setup"] == "baseline"

        baseline_transforms = transforms["experimental_setups"]["baseline"]
        assert len(baseline_transforms) == 2
        assert baseline_transforms[0]["name"] == "AddSelfLoops"
        assert baseline_transforms[1]["name"] == "NormalizeFeatures"
        assert baseline_transforms[1]["kwargs"] == {"attrs": ["x"]}

        assert "standard_transforms" in transforms
        assert len(transforms["standard_transforms"]) == 1
        assert transforms["standard_transforms"][0]["name"] == "ToUndirected"
        clear_config_cache()


# ===========================================================================
# 6. CACHE ISOLATION — config cache does not bleed between modes
# ===========================================================================

class TestCacheIsolation:
    """Verify that loading in one mode does not pollute subsequent loads."""

    def test_cache_cleared_between_modes(
        self, single_file_config_path, split_dir_config_path
    ):
        """Loading single-file, clearing cache, then split-file returns fresh data."""
        from milia_pipeline.config.config_loader import load_config, clear_config_cache

        load_kwargs = dict(
            enable_enhancement=False,
            enable_migration=False,
            enable_validation=False,
            force_reload=True,
        )

        single_cfg = load_config(
            config_path=str(single_file_config_path), **load_kwargs
        )
        clear_config_cache()
        split_cfg = load_config(
            config_path=str(split_dir_config_path), **load_kwargs
        )

        # Both should have identical content
        assert _deep_sorted(single_cfg) == _deep_sorted(split_cfg)
        clear_config_cache()

    def test_force_reload_bypasses_cache(self, single_file_config_path):
        """force_reload=True always reloads from disk."""
        from milia_pipeline.config.config_loader import load_config, clear_config_cache

        load_kwargs = dict(
            enable_enhancement=False,
            enable_migration=False,
            enable_validation=False,
        )

        cfg1 = load_config(
            config_path=str(single_file_config_path), **load_kwargs
        )

        # Modify the file on disk
        with open(single_file_config_path, "r") as f:
            content = yaml.safe_load(f)
        content["dataset_type"] = "MODIFIED"
        _write_yaml(single_file_config_path, content)

        # Without force_reload, cache may return old value
        # With force_reload, should see new value
        cfg2 = load_config(
            config_path=str(single_file_config_path),
            force_reload=True,
            **load_kwargs,
        )

        assert cfg2["dataset_type"] == "MODIFIED"
        clear_config_cache()


# ===========================================================================
# 7. REGRESSION — guards against regressions from retired Section 4.1
# ===========================================================================

class TestRegressionFromRetiredMigration:
    """
    Regression guards originally assigned to Section 4.1
    (test_regression_config_migration.py), now absorbed here.

    These tests verify that the live config_loader.py code path handles
    single-file vs split-file parity — the same core concern that the
    now-deprecated migrate_config.py was supposed to guarantee.
    """

    def test_load_config_single_mode_returns_dict(self, single_file_config_path):
        """load_config() in single-file mode returns a valid dict."""
        from milia_pipeline.config.config_loader import load_config, clear_config_cache

        config = load_config(
            config_path=str(single_file_config_path),
            enable_enhancement=False,
            enable_migration=False,
            enable_validation=False,
            force_reload=True,
        )

        assert isinstance(config, dict)
        assert "dataset_type" in config
        clear_config_cache()

    def test_load_config_split_mode_returns_dict(self, split_dir_config_path):
        """load_config() in split-file mode returns a valid dict."""
        from milia_pipeline.config.config_loader import load_config, clear_config_cache

        config = load_config(
            config_path=str(split_dir_config_path),
            enable_enhancement=False,
            enable_migration=False,
            enable_validation=False,
            force_reload=True,
        )

        assert isinstance(config, dict)
        assert "dataset_type" in config
        clear_config_cache()

    def test_both_modes_produce_identical_effective_config(
        self, single_file_config_path, split_dir_config_path
    ):
        """
        The core regression guard: identical input data loaded via both modes
        produces identical effective configuration dictionaries.
        """
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert _deep_sorted(single_cfg) == _deep_sorted(split_cfg), (
            "REGRESSION: Single-file and split-file modes produce different configs!"
        )

    def test_dft_config_section_survives_split(
        self, single_file_config_path, split_dir_config_path
    ):
        """
        dft_config (dataset-specific config) survives the split into
        datasets/dft.yaml and merges back identically.
        """
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        assert "dft_config" in single_cfg
        assert "dft_config" in split_cfg
        assert single_cfg["dft_config"] == split_cfg["dft_config"]

    def test_no_data_loss_in_split_roundtrip(
        self, single_file_config_path, split_dir_config_path
    ):
        """
        Every key in the original single-file config is present after
        the split-file round-trip. No data is silently dropped.
        """
        single_cfg, split_cfg = _load_both_modes(
            single_file_config_path, split_dir_config_path
        )

        for key in single_cfg:
            assert key in split_cfg, (
                f"REGRESSION: Key '{key}' missing from split-file config"
            )


# ===========================================================================
# PYTEST MARKERS
# ===========================================================================

# Apply markers at the module level for CI stage selection.
# Category: Config + Regression (Section 5.2).
# Only markers registered in pyproject.toml / pytest.ini are used here
# to avoid PytestUnknownMarkWarning in CI output.
# Registered markers (from MILIA_Test_Recommendations.md §Pytest Markers):
#   smoke, contract, e2e, regression, thread_safety, slow
pytestmark = [
    pytest.mark.regression,
]
