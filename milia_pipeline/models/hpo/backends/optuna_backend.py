# Location: milia_pipeline/models/hpo/backends/optuna_backend.py

"""
Optuna Backend Implementation

Primary HPO backend using Optuna's Tree-Parzen Estimators (TPE).
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any

try:
    import optuna
    from optuna.exceptions import ExperimentalWarning
    from optuna.trial import TrialState

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TrialState = None
    ExperimentalWarning = None

from milia_pipeline.exceptions import BackendError, HPOError

logger = logging.getLogger(__name__)


class OptunaBackend:
    """
    Optuna HPO backend implementation.

    Implements HPOBackendProtocol using Optuna library.

    Features:
        - TPE, CMA-ES, Random samplers
        - Median, Hyperband, Successive Halving pruners
        - SQLite/PostgreSQL storage for persistence
        - Parallel optimization support
        - Multi-objective optimization

    Usage:
        >>> backend = OptunaBackend()
        >>> study = backend.create_study("my_study", "minimize")
        >>> backend.optimize(study, objective_fn, n_trials=100)
        >>> best_params = backend.get_best_params(study)
    """

    def __init__(self):
        """Initialize Optuna backend."""
        if not OPTUNA_AVAILABLE:
            raise BackendError(
                "Optuna is not installed",
                backend_name="optuna",
                details="Install with: pip install optuna",
            )

        logger.info("OptunaBackend initialized")

    def create_study(
        self,
        study_name: str,
        direction: str,
        storage: str | None = None,
        load_if_exists: bool = True,
        sampler: Any | None = None,
        pruner: Any | None = None,
    ) -> "optuna.Study":
        """
        Create or load an Optuna study.

        Args:
            study_name: Name for the study
            direction: "minimize" or "maximize"
            storage: Storage URL (e.g., "sqlite:///optuna.db")
            load_if_exists: Whether to resume existing study
            sampler: Optuna sampler instance
            pruner: Optuna pruner instance

        Returns:
            Optuna Study object
        """
        try:
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                storage=storage,
                load_if_exists=load_if_exists,
                sampler=sampler,
                pruner=pruner,
            )

            n_existing = len(study.trials)
            if n_existing > 0:
                logger.info(f"Resumed study '{study_name}' with {n_existing} existing trials")
            else:
                logger.info(f"Created new study '{study_name}'")

            return study

        except optuna.exceptions.DuplicatedStudyError:
            # This shouldn't happen with load_if_exists=True, but handle it
            logger.warning(f"Study '{study_name}' exists, loading...")
            return optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
            )
        except Exception as e:
            raise BackendError(
                f"Failed to create study: {e}",
                backend_name="optuna",
                operation="create_study",
                details=str(e),
            ) from e

    def optimize(
        self,
        study: "optuna.Study",
        objective_fn: Callable[["optuna.Trial"], float],
        n_trials: int,
        timeout: int | None = None,
        n_jobs: int = 1,
        catch: tuple = (Exception,),
        callbacks: list[Callable] | None = None,
    ) -> None:
        """
        Run optimization on the study.

        Args:
            study: Optuna Study object
            objective_fn: Objective function taking trial, returning metric
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            n_jobs: Number of parallel jobs
            catch: Exceptions to catch and mark as failed
            callbacks: Optuna callbacks
        """
        logger.info(f"Starting optimization: {n_trials} trials, {n_jobs} jobs, timeout={timeout}s")

        try:
            study.optimize(
                objective_fn,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                catch=catch,
                callbacks=callbacks or [],
                show_progress_bar=True,
            )

            # Log summary
            completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
            pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
            failed = len([t for t in study.trials if t.state == TrialState.FAIL])

            logger.info(
                f"Optimization complete: {completed} completed, {pruned} pruned, {failed} failed"
            )

        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
            raise
        except Exception as e:
            raise BackendError(
                f"Optimization failed: {e}",
                backend_name="optuna",
                operation="optimize",
                details=str(e),
            ) from e

    def get_best_params(self, study: "optuna.Study") -> dict[str, Any]:
        """Get best hyperparameters from study."""
        try:
            return study.best_params
        except ValueError as e:
            # No completed trials
            raise HPOError(
                "No completed trials in study", study_name=study.study_name, details=str(e)
            ) from e

    def get_best_value(self, study: "optuna.Study") -> float:
        """Get best objective value from study."""
        try:
            return study.best_value
        except ValueError as e:
            raise HPOError(
                "No completed trials in study", study_name=study.study_name, details=str(e)
            ) from e

    def get_all_trials(self, study: "optuna.Study") -> list[dict[str, Any]]:
        """Get information about all trials."""
        trials_info = []

        for trial in study.trials:
            trial_info = {
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name,
                "duration": (
                    (trial.datetime_complete - trial.datetime_start).total_seconds()
                    if trial.datetime_complete and trial.datetime_start
                    else None
                ),
                "user_attrs": trial.user_attrs,
                "intermediate_values": trial.intermediate_values,
            }
            trials_info.append(trial_info)

        return trials_info

    def create_pruner(
        self, pruner_type: str, n_startup_trials: int = 5, n_warmup_steps: int = 10, **kwargs
    ) -> "optuna.pruners.BasePruner":
        """
        Create an Optuna pruner instance.

        Args:
            pruner_type: Type of pruner
            n_startup_trials: Trials before pruning begins
            n_warmup_steps: Steps before pruning within trial

        Returns:
            Optuna pruner instance
        """
        pruner_map = {
            "median": optuna.pruners.MedianPruner,
            "percentile": optuna.pruners.PercentilePruner,
            "hyperband": optuna.pruners.HyperbandPruner,
            "successive_halving": optuna.pruners.SuccessiveHalvingPruner,
            "threshold": optuna.pruners.ThresholdPruner,
            "patient": optuna.pruners.PatientPruner,
            "none": optuna.pruners.NopPruner,
        }

        if pruner_type not in pruner_map:
            available = ", ".join(pruner_map.keys())
            raise BackendError(
                f"Unknown pruner type: '{pruner_type}'",
                backend_name="optuna",
                operation="create_pruner",
                details=f"Available pruners: {available}",
            )

        pruner_cls = pruner_map[pruner_type]

        # Build pruner kwargs based on type
        pruner_kwargs = {}

        if pruner_type in ("median", "percentile"):
            pruner_kwargs = {
                "n_startup_trials": n_startup_trials,
                "n_warmup_steps": n_warmup_steps,
                "interval_steps": kwargs.get("interval_steps", 1),
            }
            if pruner_type == "percentile":
                pruner_kwargs["percentile"] = kwargs.get("percentile", 25.0)

        elif pruner_type == "hyperband":
            pruner_kwargs = {
                "min_resource": kwargs.get("min_resource", 1),
                "max_resource": kwargs.get("max_resource", n_warmup_steps + 100),
                "reduction_factor": kwargs.get("reduction_factor", 3),
            }

        elif pruner_type == "successive_halving":
            pruner_kwargs = {
                "min_resource": kwargs.get("min_resource", 1),
                "reduction_factor": kwargs.get("reduction_factor", 4),
                "min_early_stopping_rate": kwargs.get("min_early_stopping_rate", 0),
            }

        elif pruner_type == "threshold":
            pruner_kwargs = {
                "lower": kwargs.get("lower"),
                "upper": kwargs.get("upper"),
                "n_warmup_steps": n_warmup_steps,
            }

        elif pruner_type == "patient":
            # PatientPruner wraps another pruner
            wrapped_pruner = self.create_pruner(
                kwargs.get("wrapped_pruner_type", "median"),
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
            )
            pruner_kwargs = {
                "wrapped_pruner": wrapped_pruner,
                "patience": kwargs.get("patience", 5),
            }

        logger.debug(f"Creating {pruner_type} pruner with kwargs: {pruner_kwargs}")
        return pruner_cls(**pruner_kwargs)

    def _build_sampler_registry(self) -> dict[str, type]:
        """Build sampler registry dynamically based on Optuna version."""
        if hasattr(self, "_sampler_registry_cache"):
            return self._sampler_registry_cache

        potential_samplers = {
            "tpe": "TPESampler",
            "random": "RandomSampler",
            "cmaes": "CmaEsSampler",
            "grid": "GridSampler",
            "nsgaii": "NSGAIISampler",
            "motpe": "MOTPESampler",
            "qmc": "QMCSampler",
        }

        self._sampler_registry_cache = {
            key: getattr(optuna.samplers, cls_name)
            for key, cls_name in potential_samplers.items()
            if hasattr(optuna.samplers, cls_name)
        }

        return self._sampler_registry_cache

    def create_sampler(
        self, sampler_type: str, seed: int | None = None, n_startup_trials: int = 10, **kwargs
    ) -> "optuna.samplers.BaseSampler":
        """
        Create an Optuna sampler instance.

        Args:
            sampler_type: Type of sampler
            seed: Random seed
            n_startup_trials: Random trials before Bayesian optimization

        Returns:
            Optuna sampler instance
        """
        sampler_map = self._build_sampler_registry()

        if sampler_type not in sampler_map:
            available = ", ".join(sampler_map.keys())
            raise BackendError(
                f"Unknown sampler type: '{sampler_type}'",
                backend_name="optuna",
                operation="create_sampler",
                details=f"Available samplers: {available}",
            )

        sampler_cls = sampler_map[sampler_type]

        # Build sampler kwargs based on type
        sampler_kwargs = {"seed": seed}

        if sampler_type == "tpe":
            sampler_kwargs.update(
                {
                    "n_startup_trials": n_startup_trials,
                    "multivariate": kwargs.get("multivariate", True),
                    "constant_liar": kwargs.get("constant_liar", False),
                }
            )

        elif sampler_type == "cmaes":
            sampler_kwargs.update(
                {
                    "n_startup_trials": n_startup_trials,
                    "restart_strategy": kwargs.get("restart_strategy", "ipop"),
                }
            )

        elif sampler_type == "grid":
            # Grid sampler requires search_space
            if "search_space" not in kwargs:
                raise BackendError(
                    "GridSampler requires 'search_space' kwarg",
                    backend_name="optuna",
                    operation="create_sampler",
                )
            sampler_kwargs = {"search_space": kwargs["search_space"]}

        logger.debug(f"Creating {sampler_type} sampler with seed={seed}")

        # Optuna's TPESampler.__init__ unconditionally emits ExperimentalWarning
        # via warn_experimental_argument() when experimental features like
        # multivariate=True or constant_liar=True are passed. Since MILIA
        # deliberately opts into these features as architectural choices,
        # we suppress only ExperimentalWarning during sampler instantiation.
        with warnings.catch_warnings():
            if ExperimentalWarning is not None:
                warnings.filterwarnings(
                    "ignore",
                    category=ExperimentalWarning,
                )
            return sampler_cls(**sampler_kwargs)

    def suggest_params(
        self, trial: "optuna.Trial", search_space: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Suggest hyperparameters from search space.

        Utility method that maps search space config to Optuna suggest calls.

        Args:
            trial: Optuna Trial object
            search_space: Search space configuration

        Returns:
            Dict of suggested parameter values
        """
        params = {}

        def _get_config_value(config, key, default=None):
            """Extract value from config (dict or object), returning default if None."""
            if hasattr(config, key):
                value = getattr(config, key)
            elif isinstance(config, dict):
                value = config.get(key)
            else:
                value = None
            return value if value is not None else default

        def _get_required_value(config, key, full_name):
            """Extract required value, raising BackendError if missing."""
            value = _get_config_value(config, key)
            if value is None:
                raise BackendError(
                    f"Missing required parameter '{key}' for '{full_name}'",
                    backend_name="optuna",
                    operation="suggest_params",
                    details=f"Parameter '{full_name}' requires '{key}' to be specified",
                )
            return value

        for category, category_params in search_space.items():
            for param_name, config in category_params.items():
                full_name = f"{category}.{param_name}"

                param_type = _get_config_value(config, "type")
                if param_type is None:
                    raise BackendError(
                        f"Missing 'type' for parameter '{full_name}'",
                        backend_name="optuna",
                        operation="suggest_params",
                        details=f"Parameter '{full_name}' must specify a type",
                    )

                if isinstance(param_type, str):
                    param_type_str = param_type
                else:
                    param_type_str = (
                        param_type.value if hasattr(param_type, "value") else str(param_type)
                    )

                if param_type_str == "int":
                    low = _get_required_value(config, "low", full_name)
                    high = _get_required_value(config, "high", full_name)
                    step = _get_config_value(config, "step", default=1)
                    params[full_name] = trial.suggest_int(full_name, int(low), int(high), step=step)

                elif param_type_str == "float":
                    low = _get_required_value(config, "low", full_name)
                    high = _get_required_value(config, "high", full_name)
                    log = _get_config_value(config, "log", default=False)
                    params[full_name] = trial.suggest_float(full_name, low, high, log=log)

                elif param_type_str == "loguniform":
                    low = _get_required_value(config, "low", full_name)
                    high = _get_required_value(config, "high", full_name)
                    params[full_name] = trial.suggest_float(full_name, low, high, log=True)

                elif param_type_str == "categorical":
                    choices = _get_required_value(config, "choices", full_name)
                    params[full_name] = trial.suggest_categorical(full_name, choices)

                elif param_type_str == "uniform":
                    low = _get_required_value(config, "low", full_name)
                    high = _get_required_value(config, "high", full_name)
                    params[full_name] = trial.suggest_float(full_name, low, high)

                else:
                    raise BackendError(
                        f"Unknown parameter type: '{param_type_str}'",
                        backend_name="optuna",
                        operation="suggest_params",
                        details=f"Parameter: {full_name}",
                    )

        return params
