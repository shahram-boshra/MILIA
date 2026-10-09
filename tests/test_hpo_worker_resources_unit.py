#!/usr/bin/env python3
"""
Contract tests for worker sizing and ``gc_after_trial`` (P2-6, blueprint S3 / F21, F50).

The launcher sizes each worker of a shared study: one GPU (``CUDA_VISIBLE_DEVICES`` / Compose
``device_ids``) and a CPU thread budget (``OMP_NUM_THREADS``, which PyTorch reads at start). A worker
reports both, warns on oversubscription, and refuses a processed dataset larger than the memory
available to it (host memory and cgroup v2 limits). ``HPOConfig.gc_after_trial`` reaches
``Study.optimize``.
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import optuna
import pytest
from pydantic import ValidationError

from milia_pipeline.exceptions import HPOError
from milia_pipeline.models.hpo import worker_resources as wr
from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend
from milia_pipeline.models.hpo.hpo_config import HPOConfig, StorageConfig, StudyConfig

GIB = 2**30


def _resources(cpus=4, threads=1, cuda=0):
    return wr.WorkerResources(
        usable_cpus=cpus,
        torch_threads=threads,
        omp_num_threads=None,
        cuda_visible_devices=None,
        cuda_devices=cuda,
    )


@pytest.mark.contract
class TestWarnings:
    @pytest.mark.parametrize(
        ("cpus", "threads", "warned"),
        [(4, 4, True), (4, 8, True), (4, 2, False), (4, 1, False), (1, 1, False)],
    )
    def test_threads_using_every_cpu(self, cpus, threads, warned):
        found = _resources(cpus=cpus, threads=threads).warnings()
        assert any("OMP_NUM_THREADS=floor(" in w for w in found) is warned

    @pytest.mark.parametrize(("cuda", "warned"), [(0, False), (1, False), (2, True)])
    def test_more_than_one_gpu_visible(self, cuda, warned):
        found = _resources(cuda=cuda).warnings()
        assert any("CUDA devices are visible" in w for w in found) is warned

    def test_describe_reports_unset_variables(self):
        text = _resources().describe()
        assert "CUDA_VISIBLE_DEVICES=<unset>" in text and "OMP_NUM_THREADS=<unset>" in text


@pytest.mark.contract
def test_launcher_thread_budget_reaches_pytorch():
    """OMP_NUM_THREADS set by the launcher is the thread count PyTorch uses (no MILIA setting)."""
    code = (
        "from milia_pipeline.models.hpo.worker_resources import describe_worker_resources as d;"
        "r = d(); print(r.torch_threads, r.omp_num_threads)"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["1", "1"]


def _cgroup(tmp_path, entry, levels):
    """Fake /proc/self/cgroup and a cgroup v2 tree: levels maps a relative path to its files."""
    root = tmp_path / "cgroup"
    for relative, files in levels.items():
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in files.items():
            (directory / name).write_text(value, encoding="ascii")
    root.mkdir(exist_ok=True)
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(entry, encoding="ascii")
    return root, proc


def _limit(maximum, current, inactive_file=0):
    return {
        "memory.max": f"{maximum}\n",
        "memory.current": f"{current}\n",
        "memory.stat": f"anon 1\ninactive_file {inactive_file}\nactive_file 5\n",
    }


@pytest.mark.contract
class TestCgroupHeadroom:
    def test_container_limit_excludes_reclaimable_cache(self, tmp_path):
        root, proc = _cgroup(tmp_path, "0::/\n", {".": _limit(1000, 600, inactive_file=100)})
        assert wr.cgroup_memory_headroom(root, proc) == 500  # 1000 - (600 - 100)

    @pytest.mark.parametrize(
        "levels",
        [
            {"slice": _limit(10_000, 9_500), "slice/job": _limit(5_000, 1_000)},  # parent tighter
            {"slice": _limit(10_000, 1_000), "slice/job": _limit(5_000, 4_500)},  # leaf tighter
        ],
    )
    def test_tightest_level_wins(self, tmp_path, levels):
        root, proc = _cgroup(tmp_path, "0::/slice/job\n", levels)
        assert wr.cgroup_memory_headroom(root, proc) == 500

    def test_usage_above_limit_gives_zero(self, tmp_path):
        root, proc = _cgroup(tmp_path, "0::/\n", {".": _limit(1000, 1200)})
        assert wr.cgroup_memory_headroom(root, proc) == 0

    @pytest.mark.parametrize(
        ("entry", "levels"),
        [
            ("0::/\n", {".": {"memory.max": "max\n", "memory.current": "5\n"}}),  # no limit
            ("12:memory:/docker/x\n", {}),  # cgroup v1 only
            ("0::/../outside\n", {}),  # path outside the hierarchy
            ("0::/missing\n", {}),  # no interface files
        ],
    )
    def test_no_limit_gives_none(self, tmp_path, entry, levels):
        root, proc = _cgroup(tmp_path, entry, levels)
        assert wr.cgroup_memory_headroom(root, proc) is None

    def test_unreadable_proc_file_gives_none(self, tmp_path):
        assert wr.cgroup_memory_headroom(tmp_path, tmp_path / "absent") is None


@pytest.mark.contract
class TestMemoryCheck:
    @pytest.mark.parametrize(("host", "headroom", "expected"), [(8, None, 8), (8, 2, 2), (3, 6, 3)])
    def test_available_is_the_smaller_bound(self, host, headroom, expected):
        with (
            patch.object(wr.psutil, "virtual_memory", return_value=MagicMock(available=host)),
            patch.object(wr, "cgroup_memory_headroom", return_value=headroom),
        ):
            assert wr.available_memory_bytes() == expected

    def test_dataset_that_fits(self, tmp_path):
        dataset = tmp_path / "data.pt"
        dataset.write_bytes(b"x" * 100)
        with patch.object(wr, "available_memory_bytes", return_value=GIB):
            assert wr.check_worker_memory(dataset) == (100, GIB)

    def test_dataset_larger_than_available_memory_stops_the_worker(self, tmp_path):
        dataset = tmp_path / "data.pt"
        dataset.write_bytes(b"x" * 100)
        with (
            patch.object(wr, "available_memory_bytes", return_value=99),
            pytest.raises(HPOError, match="larger than the memory available"),
        ):
            wr.check_worker_memory(dataset)

    def test_real_host_reports_positive_memory(self):
        assert wr.available_memory_bytes() > 0


@pytest.mark.contract
class TestGcAfterTrial:
    def test_default_off(self):
        assert HPOConfig().gc_after_trial is False

    def test_from_dict(self):
        assert HPOConfig.from_dict({"gc_after_trial": True}).gc_after_trial is True

    @pytest.mark.parametrize("value", ["yes", 1, 0, None])
    def test_rejects_non_bool(self, value):
        with pytest.raises(ValidationError):
            HPOConfig.from_dict({"gc_after_trial": value})

    @pytest.mark.parametrize(("flag", "collections"), [(False, 0), (True, 3)])
    def test_backend_collects_after_every_trial(self, flag, collections):
        study = optuna.create_study()
        with patch("optuna.study._optimize.gc.collect") as collect:
            OptunaBackend().optimize(
                study,
                lambda t: t.suggest_float("x", 0.0, 1.0),
                n_trials=3,
                show_progress_bar=False,
                gc_after_trial=flag,
            )
        assert collect.call_count == collections

    @pytest.mark.parametrize("flag", [False, True])
    def test_manager_passes_config_value(self, flag, tmp_path):
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        backend = MagicMock()
        backend.get_best_params.return_value = {"lr": 0.01}
        backend.get_best_value.return_value = 0.1
        study = StudyConfig(
            storage_options=StorageConfig(kind="journal_file", journal_path=str(tmp_path / "j"))
        )
        config = HPOConfig(enabled=True, gc_after_trial=flag, study=study)
        with patch("milia_pipeline.models.hpo.hpo_manager.get_backend", return_value=backend):
            manager = HPOManager(config, worker_index=0)
        with (
            patch.object(manager, "_filter_search_space_for_model", return_value={}),
            patch.object(manager, "_create_objective", return_value=MagicMock()),
            patch("milia_pipeline.models.hpo.hpo_manager.get_factory", return_value=MagicMock()),
        ):
            manager.optimize(model_name="GCN", dataset=MagicMock())
        assert backend.optimize.call_args.kwargs["gc_after_trial"] is flag
