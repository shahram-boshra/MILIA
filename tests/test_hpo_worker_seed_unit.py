#!/usr/bin/env python3
"""
Contract tests for per-worker sampler seeds (P2-3b, blueprint S3 / F8).

Workers of one study that share a sampler seed repeat each other's trials (R5, R62). Each worker gets
``SeedSequence([worker_index, seed])`` — NumPy's recommended derivation; ``seed + worker_index`` is
documented as unsafe. ``worker_index=None`` keeps the configured seed (single-process behaviour).
"""

from unittest.mock import MagicMock, patch

import optuna
import pytest

from milia_pipeline.exceptions import HPOConfigurationError
from milia_pipeline.models.hpo.hpo_config import (
    HPOConfig,
    SamplerConfig,
    StorageConfig,
    StudyConfig,
)
from milia_pipeline.models.hpo.seeding import derive_worker_seed, validate_worker_index

# SeedSequence([w, 42]).generate_state(1, uint32)[0] — identical on numpy 1.26.4 (pinned) and 2.5.3 (R62)
GOLDEN_SEED_42 = [355231105, 3394225019, 2520480020, 1846748587]


@pytest.fixture(autouse=True)
def _quiet_optuna():
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    yield
    optuna.logging.set_verbosity(verbosity)


@pytest.mark.contract
class TestDeriveWorkerSeed:
    def test_single_process_keeps_seed(self):
        assert derive_worker_seed(42, None) == 42

    @pytest.mark.parametrize("worker_index", [None, 0, 3])
    def test_unseeded_stays_unseeded(self, worker_index):
        assert derive_worker_seed(None, worker_index) is None

    def test_golden_values(self):
        """Pins the derivation: a change would silently alter every seeded parallel study."""
        assert [derive_worker_seed(42, w) for w in range(4)] == GOLDEN_SEED_42

    def test_plain_int_in_randomstate_range(self):
        seeds = [derive_worker_seed(s, w) for s in (0, 42, 2**31) for w in range(8)]
        assert all(type(s) is int and 0 <= s <= 2**32 - 1 for s in seeds)

    def test_distinct_across_workers(self):
        seeds = [derive_worker_seed(42, w) for w in range(256)]
        assert len(set(seeds)) == 256

    def test_no_overlap_between_neighbouring_root_seeds(self):
        """The unsafe `seed + worker` scheme gives worker 1 of seed 42 == worker 0 of seed 43."""
        assert derive_worker_seed(42, 1) != derive_worker_seed(43, 0)

    def test_deterministic(self):
        assert derive_worker_seed(7, 5) == derive_worker_seed(7, 5)


@pytest.mark.contract
class TestValidateWorkerIndex:
    @pytest.mark.parametrize("value", [None, 0, 1, 63])
    def test_accepts(self, value):
        assert validate_worker_index(value) == value

    @pytest.mark.parametrize("value", [-1, True, False, 1.0, "1"])
    def test_rejects(self, value):
        with pytest.raises(HPOConfigurationError, match="worker_index"):
            validate_worker_index(value)


def _run_worker(storage, seed: int | None) -> None:
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    study = optuna.load_study(study_name="workers", storage=storage, sampler=sampler)
    study.optimize(lambda t: t.suggest_float("x", -10.0, 10.0) ** 2, n_trials=5)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("seeds", "unique"),
    [
        ([42, 42], 5),  # the defect (F8): the second worker repeats the first
        ([derive_worker_seed(42, 0), derive_worker_seed(42, 1)], 10),
    ],
)
def test_two_workers_on_one_study(tmp_path, seeds, unique):
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(tmp_path / "workers.log"))
    )
    optuna.create_study(study_name="workers", storage=storage)
    for seed in seeds:
        _run_worker(storage, seed)
    params = [
        t.params["x"] for t in optuna.load_study(study_name="workers", storage=storage).trials
    ]
    assert len(params) == 10
    assert len(set(params)) == unique


def _sampler_seed_passed(worker_index, tmp_path):
    from milia_pipeline.models.hpo.hpo_manager import HPOManager

    backend = MagicMock()
    backend.get_best_params.return_value = {"lr": 0.01}
    backend.get_best_value.return_value = 0.1  # formatted with :.6f by the manager
    # A worker needs a storage that can be shared (P2-3e): journal file in the test's directory
    study = StudyConfig(
        storage_options=StorageConfig(kind="journal_file", journal_path=str(tmp_path / "j.log"))
    )
    config = HPOConfig(enabled=True, sampler=SamplerConfig(seed=42), study=study)
    with patch("milia_pipeline.models.hpo.hpo_manager.get_backend", return_value=backend):
        manager = HPOManager(config, worker_index=worker_index)
    with (
        patch.object(manager, "_filter_search_space_for_model", return_value={}),
        patch.object(manager, "_create_objective", return_value=MagicMock()),
        patch("milia_pipeline.models.hpo.hpo_manager.get_factory", return_value=MagicMock()),
    ):
        manager.optimize(model_name="GCN", dataset=MagicMock())
    return backend.create_sampler.call_args.kwargs["seed"]


@pytest.mark.contract
class TestManagerSeed:
    def test_single_process_uses_configured_seed(self, tmp_path):
        assert _sampler_seed_passed(None, tmp_path) == 42

    def test_worker_uses_derived_seed(self, tmp_path):
        assert _sampler_seed_passed(2, tmp_path) == GOLDEN_SEED_42[2]

    def test_invalid_worker_index_rejected(self):
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        with (
            patch("milia_pipeline.models.hpo.hpo_manager.get_backend"),
            pytest.raises(HPOConfigurationError, match="worker_index"),
        ):
            HPOManager(HPOConfig(enabled=True), worker_index=-1)
