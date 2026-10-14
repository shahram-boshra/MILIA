#!/usr/bin/env python3
"""
Contract tests for multi-objective studies (``study.objectives``).

A study with two or more objectives is created with one Optuna direction per objective; each trial
returns one value per objective, read at the epoch the trainer keeps, plus ``n_parameters`` (the
model's parameter count). Pruning is unavailable (Optuna raises for ``Trial.report`` in multi-objective
studies) and CMA-ES is single-objective, so both are rejected when the configuration is read. The
best trial is the Pareto-front trial best on the first objective, then the next, then the lowest
trial number.
"""

import logging
import multiprocessing
from unittest.mock import MagicMock, patch

import optuna
import pytest
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState, create_trial
from pydantic import ValidationError
from torch import nn

from milia_pipeline.exceptions import HPOConfigurationError, HPOError
from milia_pipeline.models.hpo.backends.optuna_backend import _METRIC_USER_ATTR, OptunaBackend
from milia_pipeline.models.hpo.hpo_config import (
    HPOConfig,
    ObjectiveConfig,
    OptimizationDirection,
    StorageConfig,
    StudyConfig,
)

TWO = [{"metric": "val_mae", "direction": "minimize"}, {"metric": "n_parameters"}]


def _config(study=None, **overrides):
    return HPOConfig.from_dict(
        {
            "enabled": True,
            "study": study or {"objectives": TWO},
            "pruner": {"type": "none"},
            **overrides,
        }
    )


@pytest.mark.contract
class TestConfig:
    def test_objectives_accessors(self):
        study = _config().study
        assert study.is_multi_objective
        assert study.objective_metrics == ("val_mae", "n_parameters")
        assert study.objective_directions == ("minimize", "minimize")
        assert study.objectives_label == "val_mae (minimize), n_parameters (minimize)"

    def test_python_api(self):
        study = StudyConfig(
            objectives=(
                ObjectiveConfig(metric="val_r2", direction=OptimizationDirection.MAXIMIZE),
                ObjectiveConfig(metric="n_parameters"),
            )
        )
        assert study.objective_directions == ("maximize", "minimize")

    def test_single_objective_accessors_unchanged(self):
        study = StudyConfig(metric="val_mae", direction=OptimizationDirection.MAXIMIZE)
        assert not study.is_multi_objective
        assert study.objective_metrics == ("val_mae",)
        assert study.objective_directions == ("maximize",)

    @pytest.mark.parametrize("key", ["metric", "direction"])
    def test_objectives_exclude_metric_and_direction(self, key):
        value = "val_loss" if key == "metric" else "minimize"
        with pytest.raises(ValidationError, match=f"either study.objectives or study.{key}"):
            _config({"objectives": TWO, key: value})

    @pytest.mark.parametrize(
        ("objectives", "match"),
        [
            ([{"metric": "val_mae"}], "at least 2"),
            ([{"metric": "val_mae"}, {"metric": "val_mae"}], "repeats metric"),
            ([{"metric": "val_mae"}, {"metric": " "}], "cannot be empty"),
            ([{"metric": "val_mae"}, {"metric": "n_parameters", "weight": 2}], "Extra inputs"),
        ],
    )
    def test_invalid_objectives(self, objectives, match):
        with pytest.raises(ValidationError, match=match):
            _config({"objectives": objectives})

    def test_unknown_direction_names_the_objective(self):
        bad = [{"metric": "a", "direction": "up"}, {"metric": "b"}]
        with pytest.raises(ValueError, match=r"study\.objectives\[0\]\.direction 'up'"):
            _config({"objectives": bad})

    def test_pruner_must_be_none(self):
        with pytest.raises(ValidationError, match="cannot prune"):
            HPOConfig.from_dict({"study": {"objectives": TWO}})  # default pruner: median

    def test_cmaes_rejected(self):
        with pytest.raises(ValidationError, match="'cmaes' supports one objective"):
            _config(sampler={"type": "cmaes"})

    @pytest.mark.parametrize("sampler", ["tpe", "nsgaii", "random", "qmc"])
    def test_multi_objective_samplers_accepted(self, sampler):
        assert _config(sampler={"type": sampler}).sampler.type.value == sampler

    def test_single_objective_keeps_its_pruner(self):
        assert HPOConfig.from_dict({}).pruner.type.value == "median"


def _study(directions, values):
    study = optuna.create_study(directions=directions)
    for number, vals in enumerate(values):
        study.add_trial(
            create_trial(
                params={"x": float(number)},
                distributions={"x": FloatDistribution(0.0, 100.0)},
                values=vals,
            )
            if vals is not None
            else create_trial(state=TrialState.FAIL)
        )
    return study


@pytest.mark.contract
class TestBackend:
    def test_create_study_with_directions_records_metrics(self):
        backend = OptunaBackend()
        study = backend.create_study(
            "mo",
            "minimize",
            directions=("minimize", "maximize"),
            metric=("val_mae", "val_r2"),
        )
        assert [d.name for d in study.directions] == ["MINIMIZE", "MAXIMIZE"]
        assert study.user_attrs[_METRIC_USER_ATTR] == ["val_mae", "val_r2"]

    def test_loaded_study_with_other_objectives_rejected(self, tmp_path):
        backend = OptunaBackend()
        storage = f"sqlite:///{tmp_path / 's.db'}"
        backend.create_study(
            "mo",
            "minimize",
            storage,
            directions=("minimize", "minimize"),
            metric=("val_mae", "n_parameters"),
        )
        with pytest.raises(HPOConfigurationError, match="direction"):
            backend.create_study(
                "mo",
                "minimize",
                storage,
                directions=("minimize", "maximize"),
                metric=("val_mae", "n_parameters"),
            )
        with pytest.raises(HPOConfigurationError, match="optimizes metric"):
            backend.create_study(
                "mo",
                "minimize",
                storage,
                directions=("minimize", "minimize"),
                metric=("val_mae", "training_time"),
            )
        with pytest.raises(HPOConfigurationError, match="direction"):
            backend.create_study("mo", "minimize", storage, metric="val_mae")  # single objective

    def test_best_trial_is_pareto_trial_best_on_first_objective(self):
        # minimize, maximize: front = trials 0, 1, 3 (trial 2 is dominated by 0); trial 3 has the
        # lowest first value
        study = _study(
            ["minimize", "maximize"],
            [[1.0, 5.0], [2.0, 9.0], [3.0, 1.0], [0.5, 0.0], None],
        )
        backend = OptunaBackend()
        assert {t.number for t in study.best_trials} == {0, 1, 3}
        assert backend.get_best_trial_number(study) == 3
        assert backend.get_best_params(study) == {"x": 3.0}
        assert backend.get_best_value(study) == 0.5
        assert backend.get_best_values(study) == [0.5, 0.0]

    def test_tie_on_first_objective_broken_by_second_then_number(self):
        # first objective equal; second maximized → trial 1 (7.0) beats trial 0 (3.0)
        study = _study(["minimize", "maximize"], [[1.0, 3.0], [1.0, 7.0], [1.0, 7.0]])
        assert OptunaBackend().get_best_trial_number(study) == 1

    def test_maximized_first_objective(self):
        study = _study(["maximize", "minimize"], [[0.9, 10.0], [0.95, 50.0], [0.7, 5.0]])
        assert OptunaBackend().get_best_trial_number(study) == 1

    def test_no_completed_trial_raises(self):
        with pytest.raises(HPOError, match="No completed trials"):
            OptunaBackend().get_best_params(_study(["minimize", "minimize"], [None]))

    def test_all_trials_report_values(self):
        multi = OptunaBackend().get_all_trials(_study(["minimize", "minimize"], [[1.0, 2.0]]))
        assert multi[0]["value"] is None and multi[0]["values"] == [1.0, 2.0]
        single = OptunaBackend().get_all_trials(_study(["minimize"], [[4.0]]))
        assert single[0]["value"] == 4.0 and single[0]["values"] == [4.0]

    def test_single_objective_selection_unchanged(self):
        study = _study(["maximize"], [[0.2], [0.9], [0.5]])
        backend = OptunaBackend()
        assert backend.get_best_value(study) == 0.9
        assert backend.get_best_trial_number(study) == 1


RESULTS = {
    "train_metrics": {"val_loss": [0.5, 0.2, 0.3], "val_mae": [0.9, 0.4, 0.1]},
    "test_metrics": {},
    "training_time": 3.0,
    "best_epoch": 1,
    "best_val_loss": 0.2,
}


def _objective(config, results=RESULTS, cv_folds=0):
    """Build the real trial objective with training mocked; return (objective, mocks)."""
    from milia_pipeline.models.hpo.hpo_manager import HPOManager

    module = "milia_pipeline.models.hpo.hpo_manager"
    trainer = MagicMock()
    trainer.fit.return_value = dict(results)
    trainer_cls = MagicMock(return_value=trainer)
    factory = MagicMock()
    factory.create_model_with_info.return_value = (nn.Linear(2, 1), {})  # 3 parameters
    callback_factory = MagicMock()
    patches = [
        patch(f"{module}.get_factory", return_value=factory),
        patch(f"{module}.infer_task_type", return_value="graph_regression"),
        patch(f"{module}._TARGET_SELECTION_AVAILABLE", False),
        patch(f"{module}.create_hpo_callback", callback_factory),
        patch(f"{module}.DataSplitter"),
        patch(f"{module}._prepare_data_for_task_hpo", return_value=([1], [1], None)),
        patch("milia_pipeline.models.training.loaders.make_loader", return_value=MagicMock()),
        patch(
            "milia_pipeline.models.acceleration.config_builder.build_acceleration",
            return_value=None,
        ),
        patch(f"{module}.Trainer", trainer_cls),
        patch.object(HPOManager, "_build_composite_hyperparameters", return_value={}),
    ]
    for p in patches:
        p.start()
    try:
        splitter = __import__(module, fromlist=["DataSplitter"]).DataSplitter
        splitter.random_split.return_value = ([1], [1], [])
        splitter.k_fold_split.return_value = [([1], [1])] * max(cv_folds, 1)
        manager = HPOManager(config)
        manager._model_factory = factory
        objective = manager._create_objective(
            model_name="GCN",
            dataset=[MagicMock()],
            base_hyperparameters={},
            trainer_kwargs={"max_epochs": 1},
            additional_callbacks=[],
        )
    except BaseException:
        for p in patches:
            p.stop()
        raise
    return objective, manager, trainer_cls, callback_factory, patches


@pytest.mark.contract
class TestObjective:
    def test_returns_one_value_per_objective_without_pruning_callback(self):
        objective, manager, trainer_cls, callback_factory, patches = _objective(_config())
        try:
            manager.backend.suggest_params = MagicMock(return_value={})
            assert objective(MagicMock(number=0)) == (0.4, 3.0)  # val_mae at kept epoch, 3 params
        finally:
            for p in patches:
                p.stop()
        callback_factory.assert_not_called()
        assert trainer_cls.call_args.kwargs["hpo_callback"] is None

    def test_single_objective_still_returns_a_float_with_callback(self):
        objective, manager, _, callback_factory, patches = _objective(
            HPOConfig.from_dict({"enabled": True, "study": {"metric": "val_mae"}})
        )
        try:
            manager.backend.suggest_params = MagicMock(return_value={})
            assert objective(MagicMock(number=0)) == 0.4
        finally:
            for p in patches:
                p.stop()
        callback_factory.assert_called_once()

    def test_n_parameters_usable_as_single_objective(self):
        objective, manager, _, _, patches = _objective(
            HPOConfig.from_dict({"enabled": True, "study": {"metric": "n_parameters"}})
        )
        try:
            manager.backend.suggest_params = MagicMock(return_value={})
            assert objective(MagicMock(number=0)) == 3.0
        finally:
            for p in patches:
                p.stop()

    def test_cross_validation_aggregates_each_objective_and_never_reports(self):
        objective, manager, _, _, patches = _objective(
            _config(cv_folds=2, cv_metric_aggregation="mean"), cv_folds=2
        )
        trial = MagicMock(number=0)
        try:
            manager.backend.suggest_params = MagicMock(return_value={})
            assert objective(trial) == (0.4, 3.0)
        finally:
            for p in patches:
                p.stop()
        trial.report.assert_not_called()
        trial.should_prune.assert_not_called()


@pytest.mark.contract
def test_cross_validation_returns_tuple_per_metric():
    from milia_pipeline.models.hpo.hpo_manager import _run_cross_validation

    module = "milia_pipeline.models.hpo.hpo_manager"
    fold_results = [
        {**RESULTS, "train_metrics": {"val_loss": [0.2], "val_mae": [0.2]}, "best_epoch": 0},
        {**RESULTS, "train_metrics": {"val_loss": [0.4], "val_mae": [0.6]}, "best_epoch": 0},
    ]
    trainer = MagicMock()
    trainer.fit.side_effect = fold_results
    factory = MagicMock()
    factory.create_model_with_info.return_value = (nn.Linear(3, 2), {})  # 8 parameters
    with (
        patch(f"{module}.DataSplitter") as splitter,
        patch(f"{module}.Trainer", return_value=trainer),
        patch("milia_pipeline.models.training.loaders.make_loader", return_value=MagicMock()),
    ):
        splitter.k_fold_split.return_value = [([MagicMock()], [MagicMock()])] * 2
        values = _run_cross_validation(
            model_name="GCN",
            dataset=MagicMock(),
            model_params={},
            optimizer_params={},
            scheduler_params={},
            loss_params={},
            trainer_kwargs={},
            callbacks=[],
            n_folds=2,
            metric="val_mae",
            aggregation="max",
            factory=factory,
            task_type="graph_regression",
            metrics=("val_mae", "n_parameters"),
        )
    assert values == (0.6, 8.0)


def _worker(path: str, index: int) -> None:
    from milia_pipeline.models.hpo.backends.storage_factory import build_storage
    from milia_pipeline.models.hpo.seeding import derive_worker_seed

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    storage = build_storage(
        StudyConfig(storage_options=StorageConfig(kind="journal_file", journal_path=path))
    )
    sampler = optuna.samplers.TPESampler(
        seed=derive_worker_seed(7, index), constant_liar=True, n_startup_trials=2
    )
    study = optuna.load_study(study_name="mo", storage=storage, sampler=sampler)
    study.optimize(
        lambda t: (t.suggest_float("x", 0.0, 1.0), 1.0 - t.suggest_float("y", 0.0, 1.0)),
        n_trials=6,
    )


@pytest.mark.contract
def test_shared_multi_objective_study_with_constant_liar(tmp_path):
    """--hpo-init creates a two-objective study; two spawn workers run seeded TPE with constant liar."""
    from milia_pipeline.models.hpo.hpo_manager import HPOManager

    path = str(tmp_path / "journal.log")
    config = HPOConfig.from_dict(
        {
            "enabled": True,
            "pruner": {"type": "none"},
            "study": {
                "study_name": "mo",
                "objectives": [
                    {"metric": "val_mae", "direction": "minimize"},
                    {"metric": "n_parameters", "direction": "minimize"},
                ],
                "storage_options": {"kind": "journal_file", "journal_path": path},
            },
        }
    )
    study = HPOManager(config).init_study()
    assert len(study.directions) == 2
    assert study.user_attrs[_METRIC_USER_ATTR] == ["val_mae", "n_parameters"]
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_worker, args=(path, i)) for i in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=300)
    assert [p.exitcode for p in processes] == [0, 0]
    logging.getLogger("optuna").setLevel(logging.ERROR)
    loaded = optuna.load_study(study_name="mo", storage=HPOManager(config).init_study()._storage)
    complete = [t for t in loaded.trials if t.state == TrialState.COMPLETE]
    assert len(complete) == 12
    assert all(len(t.values) == 2 for t in complete)
    assert len({(t.params["x"], t.params["y"]) for t in complete}) == 12


@pytest.mark.contract
def test_optimize_end_to_end_selects_pareto_trial():
    """HPOManager.optimize on a real Optuna study with two objectives: three trials, the best
    parameters are those of the Pareto trial best on the first objective."""
    runs = iter(
        [
            {**RESULTS, "train_metrics": {"val_loss": [0.3], "val_mae": [0.30]}, "best_epoch": 0},
            {**RESULTS, "train_metrics": {"val_loss": [0.1], "val_mae": [0.10]}, "best_epoch": 0},
            {**RESULTS, "train_metrics": {"val_loss": [0.2], "val_mae": [0.20]}, "best_epoch": 0},
        ]
    )
    config = _config(n_trials=3, sampler={"type": "random", "seed": 0})
    _, manager, trainer_cls, _, patches = _objective(config)
    trainer_cls.return_value.fit.side_effect = lambda: next(runs)
    try:
        with patch.object(type(manager), "_filter_search_space_for_model", return_value={}):
            manager.optimize(model_name="GCN", dataset=[MagicMock()])
    finally:
        for p in patches:
            p.stop()
    assert len(manager.study.directions) == 2
    assert [t.values for t in manager.study.trials] == [[0.3, 3.0], [0.1, 3.0], [0.2, 3.0]]
    assert manager.backend.get_best_trial_number(manager.study) == 1
    assert manager.get_best_trial()["number"] == 1
    assert manager.get_best_value() == 0.1


def _manager_with_study(study_dict, directions, values):
    from milia_pipeline.models.hpo.hpo_manager import HPOManager

    manager = HPOManager(
        _config(study_dict)
        if "objectives" in study_dict
        else HPOConfig.from_dict({"enabled": True, "study": study_dict})
    )
    manager.study = _study(directions, values)
    return manager


@pytest.mark.contract
class TestResults:
    def test_statistics_respect_a_maximized_direction(self):
        """F54: a maximized study's best value is its highest, not its lowest."""
        manager = _manager_with_study(
            {"metric": "val_r2", "direction": "maximize"}, ["maximize"], [[0.2], [0.9], [0.5]]
        )
        stats = manager.get_study_statistics()
        assert (stats["best_value"], stats["worst_value"]) == (0.9, 0.2)
        assert stats["objectives"] == [
            {
                "metric": "val_r2",
                "direction": "maximize",
                "best_value": 0.9,
                "worst_value": 0.2,
                "mean_value": pytest.approx(1.6 / 3),
            }
        ]
        assert "pareto_front_size" not in stats

    def test_statistics_per_objective_for_multi_objective(self):
        manager = _manager_with_study(
            {"objectives": [{"metric": "val_mae"}, {"metric": "val_r2", "direction": "maximize"}]},
            ["minimize", "maximize"],
            [[1.0, 5.0], [2.0, 9.0], [3.0, 1.0], [0.5, 0.0], None],
        )
        stats = manager.get_study_statistics()
        assert stats["best_value"] is None and stats["mean_value"] is None
        assert [(o["metric"], o["best_value"], o["worst_value"]) for o in stats["objectives"]] == [
            ("val_mae", 0.5, 3.0),
            ("val_r2", 9.0, 0.0),
        ]
        assert stats["best_trial_number"] == 3
        assert stats["pareto_front_size"] == 3
        assert stats["n_completed"] == 4 and stats["n_failed"] == 1

    def test_save_results_writes_pareto_front(self, tmp_path):
        import json

        manager = _manager_with_study(
            {"objectives": [{"metric": "val_mae"}, {"metric": "val_r2", "direction": "maximize"}]},
            ["minimize", "maximize"],
            [[2.0, 9.0], [1.0, 5.0], [3.0, 1.0]],
        )
        manager.best_params = manager.backend.get_best_params(manager.study)
        paths = manager.save_results(tmp_path)
        front = json.loads(paths["pareto_front_path"].read_text())
        assert front["selected_trial"] == 1
        assert front["objectives"] == [
            {"metric": "val_mae", "direction": "minimize"},
            {"metric": "val_r2", "direction": "maximize"},
        ]
        assert [t["number"] for t in front["trials"]] == [0, 1]
        assert front["trials"][1] == {
            "number": 1,
            "values": {"val_mae": 1.0, "val_r2": 5.0},
            "params": {"x": 1.0},
        }
        assert json.loads(paths["best_params_path"].read_text()) == {"x": 1.0}
        statistics = json.loads(paths["statistics_path"].read_text())
        assert statistics["best_trial_number"] == 1

    def test_single_objective_writes_no_pareto_file(self, tmp_path):
        manager = _manager_with_study({"metric": "val_loss"}, ["minimize"], [[0.3], [0.1]])
        manager.best_params = manager.backend.get_best_params(manager.study)
        paths = manager.save_results(tmp_path)
        assert "pareto_front_path" not in paths
        assert not (tmp_path / "pareto_front.json").exists()

    def test_backend_pareto_front_sorted_by_number(self):
        study = _study(["minimize", "minimize"], [[3.0, 1.0], [1.0, 3.0], [2.0, 2.0], [4.0, 4.0]])
        front = OptunaBackend().get_pareto_front(study)
        assert [t["number"] for t in front] == [0, 1, 2]
        assert front[0] == {"number": 0, "values": [3.0, 1.0], "params": {"x": 0.0}}


@pytest.mark.contract
def test_main_reports_the_selected_pareto_trial():
    """A single-process multi-objective run logs the selected trial, the front size and its values."""
    import main

    args = MagicMock(model_name="GCN", resume_study=None, hpo_worker=None, hpo_finalize=False)
    manager = MagicMock()
    manager.optimize.return_value = {"lr": 0.01}
    manager.get_study_statistics.return_value = {"best_trial_number": 3, "pareto_front_size": 2}
    manager.backend.get_best_values.return_value = [0.25, 1234.0]
    manager.train_final_model.return_value = (MagicMock(), MagicMock(), {})
    study = StudyConfig(
        objectives=(ObjectiveConfig(metric="val_mae"), ObjectiveConfig(metric="n_parameters"))
    )
    logger = MagicMock()
    with (
        patch("main.HPOConfig") as hpo_config_cls,
        patch("main.HPOManager", return_value=manager),
        patch("main._save_hpo_results"),
        patch("main._create_callbacks", return_value=[]),
        patch("main._save_training_results"),
    ):
        hpo_config_cls.from_dict.return_value = MagicMock(n_trials=4, study=study)
        assert main._run_hpo_training(args, logger, MagicMock(), {"models": {"hpo": {}}}) == 0
    infos = [str(c.args[0]) for c in logger.info.call_args_list]
    assert "Selected trial 3 of 2 on the Pareto front: val_mae=0.25, n_parameters=1234" in infos
    manager.get_best_value.assert_not_called()
