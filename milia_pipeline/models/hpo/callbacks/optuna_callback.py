"""
Optuna Pruning Callback

Integrates Optuna's pruning mechanism with MILIA's callback system.
"""

import logging
from typing import Dict, Any, Optional

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

# Import from existing callback system
try:
    from milia_pipeline.models.training.callbacks import Callback
except ImportError:
    # Fallback for testing - signatures must match callbacks.py ABC
    class Callback:
        def set_trainer(self, trainer): pass
        def on_train_begin(self, trainer): pass
        def on_epoch_end(self, trainer, epoch, metrics): pass
        def on_train_end(self, trainer): pass

from milia_pipeline.exceptions import PruningError

logger = logging.getLogger(__name__)


class OptunaPruningCallback(Callback):
    """
    Callback for Optuna trial pruning during training.
    
    Extends the MILIA Callback ABC to integrate with Optuna's pruning system.
    Reports intermediate metrics to the trial and checks for pruning decisions.
    
    Pattern: Extends Callback ABC (callbacks.py:35-96)
    Inspired by: EarlyStopping (callbacks.py:103-227)
    
    Attributes:
        trial: Optuna Trial object
        monitor: Metric name to monitor for pruning
        report_every: Report metric every N epochs (default: 1)
    
    Usage:
        >>> trial = ...  # From Optuna objective function
        >>> callback = OptunaPruningCallback(
        ...     trial=trial,
        ...     monitor="val_loss",
        ...     report_every=1
        ... )
        >>> trainer.fit(callbacks=[callback])
    
    Integration Point:
        This callback is automatically created by HPOManager and injected
        into the Trainer's callback list during HPO trials.
    """
    
    def __init__(
        self,
        trial: 'optuna.Trial',
        monitor: str = "val_loss",
        report_every: int = 1,
    ):
        """
        Initialize OptunaPruningCallback.
        
        Args:
            trial: Optuna Trial object from objective function
            monitor: Metric name to report (must be in Trainer metrics)
            report_every: Frequency of metric reporting
        """
        super().__init__()
        
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for OptunaPruningCallback. "
                "Install with: pip install optuna"
            )
        
        self.trial = trial
        self.monitor = monitor
        self.report_every = report_every
        self._trainer = None
        self._last_reported_value: Optional[float] = None
        self._reported_steps: set = set()  # Track reported steps to prevent duplicates
        
        logger.debug(
            f"OptunaPruningCallback initialized: "
            f"monitor='{monitor}', report_every={report_every}"
        )
        
    
    def set_trainer(self, trainer) -> None:
        """
        Store reference to trainer.
        
        Pattern: Follows Callback.set_trainer() (callbacks.py:61-65)
        """
        self._trainer = trainer

    
    def on_train_begin(self, trainer) -> None:
        """
        Called at the beginning of training.
        
        Pattern: Follows Callback.on_train_begin() (callbacks.py:67-72)
        
        Args:
            trainer: Trainer instance
        """
        logger.debug(
            f"Training started for trial {self.trial.number}, "
            f"monitoring '{self.monitor}'"
        )

    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Report metric to Optuna and check for pruning.
        
        Pattern: Follows EarlyStopping.on_epoch_end() (callbacks.py:160-202)
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Dict of metric names to values
            
        Raises:
            optuna.TrialPruned: If trial should be pruned
            PruningError: If monitor metric not found in metrics
        """
        # Check if we should report this epoch
        if self.report_every > 1 and epoch % self.report_every != 0:
            return
        
        # Get monitored metric value
        value = metrics.get(self.monitor)
        
        if value is None:
            # Check alternative metric names
            alt_names = [
                f"val_{self.monitor}",
                f"validation_{self.monitor}",
                self.monitor.replace("val_", ""),
            ]
            for alt in alt_names:
                if alt in metrics:
                    value = metrics[alt]
                    break
        
        if value is None:
            available = ', '.join(sorted(metrics.keys()))
            logger.warning(
                f"Metric '{self.monitor}' not found in epoch {epoch} metrics. "
                f"Available: {available}"
            )
            # Don't raise - allow training to continue
            # This handles cases where metric is computed less frequently
            return
        
        # Store for reference
        self._last_reported_value = value
        
        # Report to Optuna only if this step hasn't been reported yet
        # This prevents the "step already reported" warning from Optuna
        if epoch not in self._reported_steps:
            self.trial.report(value, epoch)
            self._reported_steps.add(epoch)
            
            logger.debug(
                f"Trial {self.trial.number}, Epoch {epoch}: "
                f"{self.monitor}={value:.6f}"
            )
        else:
            logger.debug(
                f"Trial {self.trial.number}, Epoch {epoch}: "
                f"Step already reported, skipping duplicate report"
            )
        
        # Check for pruning
        if self.trial.should_prune():
            logger.info(
                f"Trial {self.trial.number} pruned at epoch {epoch} "
                f"({self.monitor}={value:.6f})"
            )
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} with {self.monitor}={value}"
            )
    
    def on_train_end(self, trainer) -> None:
        """
        Called at the end of training.
        
        Pattern: Follows Callback.on_train_end() (callbacks.py:90-96)
        
        Args:
            trainer: Trainer instance
        """
        # Retrieve final metric from trainer's metrics history or last reported value
        final_value = self._last_reported_value
        if hasattr(trainer, 'metrics_history') and trainer.metrics_history:
            metric_history = trainer.metrics_history.get(self.monitor, [])
            if metric_history:
                final_value = metric_history[-1]
        
        logger.debug(
            f"Training ended for trial {self.trial.number}, "
            f"final {self.monitor}={final_value}"
        )
    
    def should_stop(self) -> bool:
        """
        Check if training should stop due to pruning.
        
        This method allows integration with existing early stopping logic.
        Follows the same pattern as EarlyStopping.should_stop().
        
        Returns:
            True if trial is pruned, False otherwise
        """
        return self.trial.should_prune()


def create_hpo_callback(
    trial: 'optuna.Trial',
    monitor: str = "val_loss",
    report_every: int = 1,
    backend: str = "optuna"
) -> Callback:
    """
    Factory function to create HPO callback for the specified backend.
    
    Pattern: Follows create_dataset_handler() (handlers/__init__.py)
    
    Args:
        trial: Trial object from HPO backend
        monitor: Metric name to monitor
        report_every: Reporting frequency
        backend: Backend name ("optuna" or "ray_tune")
        
    Returns:
        Callback instance appropriate for the backend
    """
    if backend == "optuna":
        return OptunaPruningCallback(
            trial=trial,
            monitor=monitor,
            report_every=report_every
        )
    elif backend == "ray_tune":
        # Future: RayTuneReportCallback
        raise NotImplementedError(
            "Ray Tune callback not yet implemented. Use 'optuna' backend."
        )
    else:
        raise ValueError(f"Unknown backend: '{backend}'")
