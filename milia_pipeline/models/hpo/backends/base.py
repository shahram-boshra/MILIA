"""
HPO Backend Protocol

Defines the abstract interface for HPO backends.
Enables swapping between Optuna and Ray Tune without code changes.
"""

from abc import abstractmethod
from collections.abc import Callable
import importlib.util
from typing import Any, Protocol, runtime_checkable

# Check optuna availability without importing it
try:
    OPTUNA_AVAILABLE = importlib.util.find_spec("optuna") is not None
except ValueError:
    # find_spec raises ValueError if the module is in sys.modules
    # but __spec__ is not set or is None (documented CPython behavior)
    OPTUNA_AVAILABLE = False


@runtime_checkable
class HPOBackendProtocol(Protocol):
    """
    Protocol defining the interface for HPO backends.

    Pattern: Follows DatasetHandlerProtocol (datasets/protocols.py)

    All HPO backends must implement these 6 methods:
    1. create_study() - Initialize optimization study
    2. optimize() - Run optimization loop
    3. get_best_params() - Retrieve best hyperparameters
    4. get_best_value() - Retrieve best objective value
    5. get_all_trials() - Retrieve all trial information
    6. create_pruner() - Create pruner instance
    7. create_sampler() - Create sampler instance
    """

    @abstractmethod
    def create_study(
        self,
        study_name: str,
        direction: str,
        storage: str | None = None,
        load_if_exists: bool = True,
        sampler: Any | None = None,
        pruner: Any | None = None,
    ) -> Any:
        """
        Create or load an HPO study.

        Args:
            study_name: Name for the study
            direction: "minimize" or "maximize"
            storage: Storage URL (None for in-memory)
            load_if_exists: Whether to resume existing study
            sampler: Sampler instance
            pruner: Pruner instance

        Returns:
            Study object (backend-specific type)
        """
        ...

    @abstractmethod
    def optimize(
        self,
        study: Any,
        objective_fn: Callable[[Any], float],
        n_trials: int,
        timeout: int | None = None,
        n_jobs: int = 1,
        catch: tuple = (),
        callbacks: list[Callable] | None = None,
    ) -> None:
        """
        Run optimization on the study.

        Args:
            study: Study object from create_study()
            objective_fn: Function that takes trial and returns metric
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            n_jobs: Number of parallel jobs
            catch: Exceptions to catch and mark as failed trials
            callbacks: Optuna-style callbacks
        """
        ...

    @abstractmethod
    def get_best_params(self, study: Any) -> dict[str, Any]:
        """
        Get best hyperparameters from completed study.

        Args:
            study: Completed study object

        Returns:
            Dict of parameter name to best value
        """
        ...

    @abstractmethod
    def get_best_value(self, study: Any) -> float:
        """
        Get best objective value from completed study.

        Args:
            study: Completed study object

        Returns:
            Best objective value
        """
        ...

    @abstractmethod
    def get_all_trials(self, study: Any) -> list[dict[str, Any]]:
        """
        Get information about all trials.

        Args:
            study: Study object

        Returns:
            List of trial info dicts with keys:
            - number: Trial number
            - params: Hyperparameters
            - value: Objective value (None if not completed)
            - state: Trial state (COMPLETE, PRUNED, FAIL, etc.)
            - duration: Trial duration in seconds
        """
        ...

    @abstractmethod
    def create_pruner(
        self, pruner_type: str, n_startup_trials: int = 5, n_warmup_steps: int = 10, **kwargs
    ) -> Any:
        """
        Create a pruner instance.

        Args:
            pruner_type: Type of pruner (median, hyperband, etc.)
            n_startup_trials: Trials before pruning begins
            n_warmup_steps: Steps before pruning within trial
            **kwargs: Additional pruner-specific arguments

        Returns:
            Pruner instance (backend-specific type)
        """
        ...

    @abstractmethod
    def create_sampler(
        self, sampler_type: str, seed: int | None = None, n_startup_trials: int = 10, **kwargs
    ) -> Any:
        """
        Create a sampler instance.

        Args:
            sampler_type: Type of sampler (tpe, random, cmaes, etc.)
            seed: Random seed for reproducibility
            n_startup_trials: Random trials before Bayesian optimization
            **kwargs: Additional sampler-specific arguments

        Returns:
            Sampler instance (backend-specific type)
        """
        ...


def get_backend(backend_name: str) -> HPOBackendProtocol:
    """
    Factory function to get HPO backend by name.

    Pattern: Follows create_dataset_handler() (handlers/__init__.py)

    Args:
        backend_name: Name of backend ("optuna" or "ray_tune")

    Returns:
        Backend instance implementing HPOBackendProtocol

    Raises:
        BackendError: If backend not found or not available
    """
    from milia_pipeline.exceptions import BackendError

    from .optuna_backend import OptunaBackend

    backends = {
        "optuna": OptunaBackend,
        # 'ray_tune': RayTuneBackend,  # Future
    }

    if backend_name not in backends:
        available = ", ".join(backends.keys())
        raise BackendError(
            f"Unknown HPO backend: '{backend_name}'",
            backend_name=backend_name,
            details=f"Available backends: {available}",
        )

    backend_cls = backends[backend_name]

    try:
        return backend_cls()
    except ImportError as e:
        raise BackendError(
            f"Backend '{backend_name}' dependencies not installed",
            backend_name=backend_name,
            details=str(e),
        ) from e
