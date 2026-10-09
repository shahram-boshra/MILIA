#!/usr/bin/env python3
"""
Contract tests for shared-study workers (P2-3e, blueprint S1 / S3, F2 / F23).

``milia --hpo-init`` creates the study once (direction + metric recorded, no trial, no dataset);
``milia --train --hpo --hpo-worker INDEX`` joins it without creating it, as worker INDEX (derived
sampler seed, no progress bar). A shared study needs a shareable storage: persistent, not SQLite, an
RDB with a trial heartbeat; ``constant_liar: false`` is logged. GPU pinning (``CUDA_VISIBLE_DEVICES``)
belongs to the launcher, so MILIA only reports it.
"""

import argparse
import contextlib
import logging
from unittest.mock import MagicMock, Mock, patch

import optuna
import pytest

from milia_pipeline.exceptions import HPOConfigurationError, HPOError
from milia_pipeline.models.hpo.hpo_config import (
    HPOConfig,
    SamplerConfig,
    SamplerType,
    StorageConfig,
    StudyConfig,
)
from milia_pipeline.models.hpo.shared_study import validate_shared_study

SECRET = "s3cr3t-pw"  # pragma: allowlist secret
SERVER_URL = f"postgresql+psycopg://milia:{SECRET}@db.example:5432/hpo"
ENV = "MILIA_TEST_SHARED_URL"


def _config(*, storage=None, storage_options=None, **sampler):
    study = StudyConfig(storage=storage, storage_options=storage_options)
    return HPOConfig(enabled=True, study=study, sampler=SamplerConfig(**sampler))


def _rdb(heartbeat_interval=60):
    return StorageConfig(kind="rdb", url_env=ENV, heartbeat_interval=heartbeat_interval)


def _journal(tmp_path):
    return StorageConfig(kind="journal_file", journal_path=str(tmp_path / "study.log"))


# =============================================================================
# Storage rules (shared_study.validate_shared_study)
# =============================================================================


@pytest.mark.contract
class TestStorageRules:
    @pytest.mark.parametrize(
        ("kwargs", "environ", "match"),
        [
            ({}, {}, "persistent storage"),
            ({"storage": "sqlite:///hpo.db"}, {}, "SQLite"),
            ({"storage_options": _rdb()}, {ENV: "sqlite:///hpo.db"}, "SQLite"),
            ({"storage": SERVER_URL}, {}, "heartbeat"),
            ({"storage_options": _rdb(None)}, {ENV: SERVER_URL}, "heartbeat"),
            ({"storage": "not a url"}, {}, "not a valid SQLAlchemy URL"),
            ({"storage_options": _rdb()}, {}, ENV),
        ],
        ids=[
            "in-memory",
            "sqlite-url",
            "sqlite-from-env",
            "server-url-no-heartbeat",
            "rdb-no-heartbeat",
            "invalid-url",
            "unset-env",
        ],
    )
    def test_rejected(self, kwargs, environ, match):
        with pytest.raises(HPOConfigurationError, match=match) as excinfo:
            validate_shared_study(_config(**kwargs), environ)
        # the URL may hold a password: it never reaches the message or the exception (F10)
        assert SECRET not in str(excinfo.value)
        assert SECRET not in repr(vars(excinfo.value))

    def test_rdb_with_heartbeat_accepted(self):
        validate_shared_study(_config(storage_options=_rdb()), {ENV: SERVER_URL})

    def test_journal_file_accepted(self, tmp_path):
        validate_shared_study(_config(storage_options=_journal(tmp_path)), {})


@pytest.mark.contract
class TestConstantLiarWarning:
    @pytest.mark.parametrize(
        ("sampler", "warned"),
        [
            ({"constant_liar": False}, True),
            ({"constant_liar": True}, False),
            ({"type": SamplerType.RANDOM, "constant_liar": False}, False),
        ],
        ids=["tpe-off", "tpe-on", "random-off"],
    )
    def test_warning_only_for_tpe_without_constant_liar(self, tmp_path, caplog, sampler, warned):
        config = _config(storage_options=_journal(tmp_path), **sampler)
        with caplog.at_level(logging.WARNING, logger="milia_pipeline.models.hpo.shared_study"):
            validate_shared_study(config, {})
        assert ("constant_liar is false" in caplog.text) is warned


# =============================================================================
# HPOManager: worker validation at construction, init_study
# =============================================================================


@pytest.mark.contract
class TestManager:
    def test_worker_with_in_memory_storage_rejected_at_construction(self):
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        with (
            patch("milia_pipeline.models.hpo.hpo_manager.get_backend"),
            pytest.raises(HPOConfigurationError, match="persistent storage"),
        ):
            HPOManager(HPOConfig(enabled=True), worker_index=0)

    def test_single_process_keeps_in_memory_storage(self):
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        with patch("milia_pipeline.models.hpo.hpo_manager.get_backend"):
            assert HPOManager(HPOConfig(enabled=True)).worker_index is None

    def test_init_study_creates_study_with_direction_and_metric(self, tmp_path):
        from milia_pipeline.models.hpo.backends.optuna_backend import _METRIC_USER_ATTR
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = HPOConfig(
            enabled=True,
            study=StudyConfig(
                study_name="shared",
                direction="maximize",
                metric="val_accuracy",
                storage_options=_journal(tmp_path),
            ),
        )
        study = HPOManager(config).init_study()

        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(tmp_path / "study.log"))
        )
        stored = optuna.load_study(study_name="shared", storage=storage)
        assert stored.direction == optuna.study.StudyDirection.MAXIMIZE
        assert stored.user_attrs[_METRIC_USER_ATTR] == "val_accuracy"
        assert len(stored.trials) == 0
        assert study.study_name == "shared"

    def test_init_study_is_rerunnable_and_guards_metric(self, tmp_path):
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        def config(metric):
            study = StudyConfig(
                study_name="shared", metric=metric, storage_options=_journal(tmp_path)
            )
            return HPOConfig(enabled=True, study=study)

        HPOManager(config("val_loss")).init_study()
        HPOManager(config("val_loss")).init_study()  # same study again: loaded, not an error
        with pytest.raises(HPOConfigurationError, match="metric"):
            HPOManager(config("val_mae")).init_study()

    def test_init_study_rejects_sqlite(self):
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = HPOConfig(enabled=True, study=StudyConfig(storage="sqlite:///hpo.db"))
        with pytest.raises(HPOConfigurationError, match="SQLite"):
            HPOManager(config).init_study()


# =============================================================================
# CLI flags (cli_manager)
# =============================================================================


def _parse(argv):
    from milia_pipeline.cli_manager import CLIManager

    return CLIManager().parse_args(argv)


@pytest.mark.contract
class TestCli:
    def test_defaults(self):
        args = _parse([])
        assert args.hpo_init is False
        assert args.hpo_worker is None
        assert args.hpo_finalize is False

    def test_worker_index_parsed(self):
        args = _parse(["--train", "--hpo", "--hpo-worker", "2"])
        assert args.hpo_worker == 2

    def test_worker_with_resume_study(self):
        args = _parse(["--train", "--hpo", "--hpo-worker", "0", "--resume-study", "s"])
        assert (args.hpo_worker, args.resume_study) == (0, "s")

    def test_init_alone(self):
        assert _parse(["--hpo-init"]).hpo_init is True

    def test_finalize_parsed(self):
        args = _parse(["--train", "--hpo", "--hpo-finalize", "--resume-study", "s"])
        assert (args.hpo_finalize, args.resume_study, args.hpo_worker) == (True, "s", None)

    @pytest.mark.parametrize(
        "argv",
        [
            ["--hpo-init", "--hpo-worker", "0"],
            ["--train", "--hpo", "--hpo-worker", "x"],
            ["--train", "--hpo", "--hpo-finalize", "--hpo-worker", "0"],  # P2-3f
            ["--hpo-init", "--hpo-finalize"],  # P2-3f
        ],
        ids=["init-and-worker", "non-integer", "finalize-and-worker", "init-and-finalize"],
    )
    def test_argparse_rejects(self, argv):
        with pytest.raises(SystemExit) as excinfo:
            _parse(argv)
        assert excinfo.value.code == 2

    @pytest.mark.parametrize(
        ("argv", "match"),
        [
            (["--train", "--hpo", "--hpo-worker", "-1"], ">= 0"),
            (["--hpo-worker", "0"], "requires --train --hpo"),
            (["--train", "--hpo-worker", "0"], "requires --train --hpo"),
            (["--train", "--no-hpo", "--hpo-worker", "0"], "requires --train --hpo"),
            (["--hpo-init", "--train"], "--train"),
            (["--hpo-init", "--resume-study", "s"], "--resume-study"),
            # P2-3f
            (["--train", "--hpo", "--hpo-worker", "0", "--force-reload"], "--force-reload"),
            (["--hpo-finalize"], "requires --train --hpo"),
            (["--train", "--hpo-finalize"], "requires --train --hpo"),
        ],
        ids=[
            "negative",
            "no-train",
            "no-hpo",
            "no-hpo-explicit",
            "init-train",
            "init-resume",
            "worker-force-reload",
            "finalize-no-train",
            "finalize-no-hpo",
        ],
    )
    def test_validation_rejects(self, argv, match):
        from milia_pipeline.cli_manager import CLIValidationError

        with pytest.raises(CLIValidationError, match=match):
            _parse(argv)


# =============================================================================
# main.py: --hpo-init mode, worker routing
# =============================================================================


def _hpo_init_args():
    return argparse.Namespace(
        n_trials=None,
        hpo_timeout=None,
        cv_folds=None,
        hpo_backend=None,
        sampler=None,
        pruner=None,
        mode=None,
        model_name=None,
        task_type=None,
    )


@pytest.mark.contract
class TestMainInit:
    def test_init_mode_creates_study(self, tmp_path):
        import main

        journal = str(tmp_path / "study.log")
        config = {
            "models": {
                "hpo": {
                    "study": {
                        "study_name": "shared",
                        "storage_options": {"kind": "journal_file", "journal_path": journal},
                    }
                }
            }
        }
        assert main.handle_hpo_init_mode(_hpo_init_args(), Mock(), config) == 0
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(journal)
        )
        assert optuna.study.get_all_study_names(storage) == ["shared"]

    def test_init_mode_fails_on_in_memory_storage(self):
        import main

        assert main.handle_hpo_init_mode(_hpo_init_args(), Mock(), {"models": {"hpo": {}}}) == 1

    def test_main_runs_init_before_any_dataset_work(self):
        import main
        from milia_pipeline.cli_manager import CLIManager

        args = CLIManager().parse_args(["--hpo-init"])
        cli_manager = Mock()
        cli_manager.handle_plugin_operations.return_value = False
        cli_manager.handle_research_api_commands.return_value = False
        cli_manager.handle_descriptor_operations.return_value = False
        cli_manager.config = {"models": {"hpo": {}}}
        with (
            patch("main.parse_cli_args", return_value=(args, cli_manager)),
            patch("main.setup_logging", return_value=Mock()),
            patch("main._register_custom_transforms_on_startup"),
            patch("main._discover_and_register_plugins"),
            patch("main.handle_hpo_init_mode", return_value=0) as init_mode,
            patch("main.validate_configuration") as validate_dataset_config,
            patch("main.create_dataset_with_error_handling") as create_dataset,
        ):
            assert main.main() == 0
        init_mode.assert_called_once()
        assert init_mode.call_args.args[2] is cli_manager.config
        validate_dataset_config.assert_not_called()
        create_dataset.assert_not_called()


def _worker_run(worker_index, resume_study=None, finalize=False, return_saves=False):
    """Run main._run_hpo_training with a mocked manager; return (manager class mock, manager mock)
    and, with ``return_saves``, the mocks of the results writer and of the final-model writer."""
    import main

    args = Mock()
    args.model_name = "GCN"
    args.resume_study = resume_study
    args.hpo_worker = worker_index
    args.hpo_finalize = finalize
    manager = MagicMock()
    manager.resume_study.return_value = {"lr": 0.01}
    manager.optimize.return_value = {"lr": 0.01}
    manager.get_best_value.return_value = 0.1
    manager.train_final_model.return_value = (Mock(), Mock(), {"best_val_loss": 0.2})
    study = StudyConfig(study_name="shared", storage="postgresql://h/db")
    with (
        patch("main.HPOConfig") as hpo_config_cls,
        patch("main.HPOManager", return_value=manager) as manager_cls,
        patch("main._save_hpo_results") as save_results,
        patch("main._create_callbacks", return_value=[]),
        patch("main._save_training_results") as save_model,
    ):
        hpo_config_cls.from_dict.return_value = Mock(n_trials=4, study=study)
        assert main._run_hpo_training(args, Mock(), Mock(), {"models": {"hpo": {}}}) == 0
    if return_saves:
        return manager_cls, manager, save_results, save_model
    return manager_cls, manager


@pytest.mark.contract
class TestMainWorker:
    def test_worker_joins_configured_study(self):
        manager_cls, manager = _worker_run(3)
        assert manager_cls.call_args.kwargs["worker_index"] == 3
        manager.optimize.assert_not_called()
        assert manager.resume_study.call_args.args[:2] == ("shared", "postgresql://h/db")
        assert manager.resume_study.call_args.kwargs["additional_trials"] == 4

    def test_worker_joins_named_study(self):
        _, manager = _worker_run(0, resume_study="other")
        assert manager.resume_study.call_args.args[0] == "other"

    def test_single_process_unchanged(self):
        manager_cls, manager, save_results, save_model = _worker_run(None, return_saves=True)
        assert manager_cls.call_args.kwargs["worker_index"] is None
        manager.optimize.assert_called_once()
        manager.resume_study.assert_not_called()
        save_results.assert_called_once()
        manager.train_final_model.assert_called_once()
        save_model.assert_called_once()


@pytest.mark.contract
class TestWorkerLifecycle:
    """P2-3f (F49): workers only run trials; --hpo-finalize writes results and the model once."""

    def test_worker_writes_no_results_and_no_model(self):
        _, manager, save_results, save_model = _worker_run(1, return_saves=True)
        manager.resume_study.assert_called_once()
        save_results.assert_not_called()
        manager.train_final_model.assert_not_called()
        save_model.assert_not_called()

    def test_finalize_loads_study_without_trials_and_finalizes_once(self):
        manager_cls, manager, save_results, save_model = _worker_run(
            None, finalize=True, return_saves=True
        )
        assert manager_cls.call_args.kwargs["worker_index"] is None
        manager.optimize.assert_not_called()
        assert manager.resume_study.call_args.args[:2] == ("shared", "postgresql://h/db")
        assert manager.resume_study.call_args.kwargs["additional_trials"] == 0
        save_results.assert_called_once()
        manager.train_final_model.assert_called_once()
        save_model.assert_called_once()

    def test_finalize_uses_named_study(self):
        _, manager = _worker_run(None, resume_study="other", finalize=True)
        assert manager.resume_study.call_args.args[0] == "other"
        assert manager.resume_study.call_args.kwargs["additional_trials"] == 0

    def test_finalize_requires_completed_trials(self, tmp_path):
        """Real journal study with no finished trial: finalizing fails instead of retraining."""
        from milia_pipeline.models.hpo.backends import build_storage
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        study = StudyConfig(study_name="shared", storage_options=_journal(tmp_path))
        config = HPOConfig(enabled=True, study=study)
        HPOManager(config).init_study()
        with pytest.raises(HPOError, match="No completed trials"):
            HPOManager(config).resume_study("shared", build_storage(study), additional_trials=0)

    def _main_with_dataset(self, tmp_path, argv, processed):
        import main
        from milia_pipeline.cli_manager import CLIManager

        if processed:
            path = tmp_path / "processed" / main.PROCESSED_DATA_FILENAME
            path.parent.mkdir(parents=True)
            path.write_bytes(b"")
        args = CLIManager().parse_args([*argv, "--root-dir", str(tmp_path)])
        cli_manager = Mock()
        cli_manager.handle_plugin_operations.return_value = False
        cli_manager.handle_research_api_commands.return_value = False
        cli_manager.handle_descriptor_operations.return_value = False
        cli_manager.config = {"models": {"hpo": {}}}
        with (
            patch("main.parse_cli_args", return_value=(args, cli_manager)),
            patch("main.setup_logging", return_value=Mock()),
            patch("main._register_custom_transforms_on_startup"),
            patch("main._discover_and_register_plugins"),
            patch("main.validate_configuration", return_value=(Mock(), Mock(), Mock(), {})),
            patch("main.print_dataset_info"),
            # Reaching the dataset build is the observable; stop there (main() exits 1 on errors)
            patch(
                "main.create_dataset_with_error_handling", side_effect=RuntimeError("stop")
            ) as cd,
            pytest.raises(SystemExit) if processed else contextlib.nullcontext(),
        ):
            result = main.main()
        return (None if processed else result), cd

    def test_worker_refuses_unprocessed_dataset(self, tmp_path):
        result, create_dataset = self._main_with_dataset(
            tmp_path, ["--train", "--hpo", "--hpo-worker", "0"], processed=False
        )
        assert result == 1
        create_dataset.assert_not_called()

    def test_worker_proceeds_with_processed_dataset(self, tmp_path):
        _, create_dataset = self._main_with_dataset(
            tmp_path, ["--train", "--hpo", "--hpo-worker", "0"], processed=True
        )
        create_dataset.assert_called_once()
