"""
Model Monitoring

Comprehensive monitoring for deployed models in production.
Supports performance tracking, drift detection, health checks, and alerting.

Features:
- Performance metrics tracking (latency, throughput, errors)
- Data drift detection (input distribution changes)
- Model drift detection (performance degradation)
- Health checks and status monitoring
- Alert system for anomalies
- Automated retraining triggers
- Metrics logging and visualization
- A/B testing metrics

Pydantic V2 Migration (Phase 21):
    - Migrated MonitoringConfig from @dataclass to Pydantic BaseModel (mutable)
    - Migrated Alert from @dataclass to Pydantic BaseModel (mutable)
    - Uses model_dump() for MonitoringConfig.to_dict() (backward compatible)
    - Uses model_dump(mode='json') for Alert.to_dict() (enum/datetime serialization)
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established pattern from deployment_strategies.py (Phase 20)

Author: milia Team
Version: 1.1.0
"""

import json
import logging
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import AlertError, ModelError, MonitoringError
except ImportError:

    class ModelError(Exception):
        """Base exception for model-related errors."""

        pass

    class MonitoringError(ModelError):
        """Exception raised for monitoring-related errors."""

        pass

    class AlertError(MonitoringError):
        """Exception raised for alert-related errors."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# MONITORING TYPES
# =============================================================================


class MetricType(Enum):
    """Types of metrics to monitor."""

    LATENCY = "latency"  # Inference latency
    THROUGHPUT = "throughput"  # Requests per second
    ERROR_RATE = "error_rate"  # Error percentage
    ACCURACY = "accuracy"  # Model accuracy
    LOSS = "loss"  # Model loss
    MEMORY = "memory"  # Memory usage
    CPU = "cpu"  # CPU usage
    GPU = "gpu"  # GPU usage


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of drift to detect."""

    DATA_DRIFT = "data_drift"  # Input distribution change
    CONCEPT_DRIFT = "concept_drift"  # Target distribution change
    MODEL_DRIFT = "model_drift"  # Model performance degradation


# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================


class MonitoringConfig(BaseModel):
    """
    Configuration for model monitoring.

    Pydantic V2 Migration (Phase 21):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses model_dump() for to_dict() method (backward compatible)
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        enable_performance_tracking: Track performance metrics
        enable_drift_detection: Detect data/model drift
        enable_health_checks: Periodic health checks
        enable_alerting: Send alerts on anomalies
        drift_detection_method: Method for drift detection
        drift_threshold: Threshold for drift detection
        alert_threshold: Threshold for triggering alerts
        health_check_interval: Seconds between health checks
        metrics_window_size: Window size for rolling metrics
        log_predictions: Log individual predictions
        log_metrics_interval: Seconds between metrics logging
        retraining_trigger_threshold: Threshold for automatic retraining
    """

    enable_performance_tracking: bool = True
    enable_drift_detection: bool = True
    enable_health_checks: bool = True
    enable_alerting: bool = True
    drift_detection_method: str = "ks_test"  # ks_test, psi, wasserstein
    drift_threshold: float = 0.05
    alert_threshold: float = 0.1
    health_check_interval: int = 60
    metrics_window_size: int = 1000
    log_predictions: bool = False
    log_metrics_interval: int = 300
    retraining_trigger_threshold: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().
        """
        return self.model_dump()


class Alert(BaseModel):
    """
    Alert dataclass for monitoring notifications.

    Pydantic V2 Migration (Phase 21):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=datetime.now) for timestamp
        - Uses model_dump(mode='json') for automatic enum value and datetime ISO serialization
        - NON-BREAKING: Same constructor API, attribute access, and to_dict() output format

    Attributes:
        severity: Alert severity level (AlertSeverity enum)
        message: Alert message text
        metric_type: Type of metric that triggered the alert
        metric_value: Current value of the metric
        threshold: Threshold value that was exceeded
        timestamp: When the alert was created (defaults to now)
    """

    severity: AlertSeverity
    message: str
    metric_type: str
    metric_value: float
    threshold: float
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method using Pydantic V2's model_dump(mode='json')
        for automatic enum value extraction and datetime ISO serialization.

        Returns:
            Dictionary with:
            - 'severity': enum value string (e.g., 'warning', 'error')
            - 'timestamp': ISO format string (e.g., '2026-01-08T12:30:45.123456')
        """
        return self.model_dump(mode="json")


# =============================================================================
# MODEL MONITOR
# =============================================================================


class ModelMonitor:
    """
    Monitor for deployed models.

    Tracks performance, detects drift, manages health checks, and triggers alerts.

    Usage:
        >>> # Initialize monitor
        >>> monitor = ModelMonitor(config)
        >>>
        >>> # Log predictions
        >>> start_time = time.time()
        >>> output = model(input_data)
        >>> latency = time.time() - start_time
        >>>
        >>> monitor.log_prediction(
        ...     input_data=input_data,
        ...     output=output,
        ...     latency=latency,
        ...     ground_truth=labels
        ... )
        >>>
        >>> # Check for drift
        >>> drift_detected = monitor.detect_drift(new_data, reference_data)
        >>>
        >>> # Get metrics
        >>> metrics = monitor.get_metrics_summary()
    """

    def __init__(
        self,
        config: MonitoringConfig | None = None,
        model_name: str = "model",
        verbose: bool = True,
    ):
        """
        Initialize model monitor.

        Args:
            config: Monitoring configuration
            model_name: Name of model being monitored
            verbose: Whether to log information
        """
        self.config = config or MonitoringConfig()
        self.model_name = model_name
        self.verbose = verbose

        # Metrics storage
        self.metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.metrics_window_size)
        )

        # Alerts
        self.alerts: list[Alert] = []
        self.alert_callbacks: list[Callable] = []

        # Reference data for drift detection
        self.reference_data: torch.Tensor | None = None

        # Health status
        self.is_healthy = True
        self.last_health_check = datetime.now()

        # Statistics
        self.total_predictions = 0
        self.total_errors = 0
        self.start_time = datetime.now()

        if self.verbose:
            logger.info(
                f"ModelMonitor initialized for '{model_name}' - "
                f"Performance: {self.config.enable_performance_tracking}, "
                f"Drift: {self.config.enable_drift_detection}, "
                f"Alerts: {self.config.enable_alerting}"
            )

    # =========================================================================
    # PERFORMANCE TRACKING
    # =========================================================================

    def log_prediction(
        self,
        input_data: torch.Tensor,
        output: torch.Tensor,
        latency: float,
        ground_truth: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Log a prediction for monitoring.

        Args:
            input_data: Input tensor
            output: Model output
            latency: Inference latency in seconds
            ground_truth: Ground truth labels (optional)
            metadata: Additional metadata
        """
        self.total_predictions += 1

        if not self.config.enable_performance_tracking:
            return

        # Log latency
        self.metrics["latency"].append(latency)

        # Log accuracy if ground truth provided
        if ground_truth is not None:
            # Simple accuracy calculation (modify for your task)
            correct = (output.argmax(dim=-1) == ground_truth).float().mean().item()
            self.metrics["accuracy"].append(correct)

        # Log memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**2)  # MB
            self.metrics["memory"].append(memory_used)

        # Check for anomalies
        if self.config.enable_alerting:
            self._check_for_anomalies()

        # Drift detection
        if self.config.enable_drift_detection and self.reference_data is not None:
            if self.total_predictions % 100 == 0:  # Check every 100 predictions
                drift_score = self._calculate_drift(input_data)
                if drift_score > self.config.drift_threshold:
                    self._trigger_alert(
                        severity=AlertSeverity.WARNING,
                        message=f"Data drift detected (score: {drift_score:.4f})",
                        metric_type="drift",
                        metric_value=drift_score,
                        threshold=self.config.drift_threshold,
                    )

    def log_error(self, error: Exception, metadata: dict[str, Any] | None = None):
        """
        Log an error.

        Args:
            error: Exception that occurred
            metadata: Additional metadata
        """
        self.total_errors += 1

        error_rate = self.total_errors / max(self.total_predictions, 1)
        self.metrics["error_rate"].append(error_rate)

        if self.config.enable_alerting and error_rate > self.config.alert_threshold:
            self._trigger_alert(
                severity=AlertSeverity.ERROR,
                message=f"High error rate: {error_rate:.2%}",
                metric_type="error_rate",
                metric_value=error_rate,
                threshold=self.config.alert_threshold,
            )

        if self.verbose:
            logger.error(f"Prediction error: {error}")

    def _check_for_anomalies(self):
        """Check metrics for anomalies."""
        # Check latency
        if len(self.metrics["latency"]) > 10:
            recent_latency = np.mean(list(self.metrics["latency"])[-10:])
            avg_latency = np.mean(list(self.metrics["latency"]))

            # Alert if recent latency is 2x average
            if recent_latency > avg_latency * 2:
                self._trigger_alert(
                    severity=AlertSeverity.WARNING,
                    message=f"High latency detected: {recent_latency:.4f}s",
                    metric_type="latency",
                    metric_value=recent_latency,
                    threshold=avg_latency * 2,
                )

        # Check accuracy degradation
        if "accuracy" in self.metrics and len(self.metrics["accuracy"]) > 100:
            recent_acc = np.mean(list(self.metrics["accuracy"])[-100:])
            baseline_acc = np.mean(list(self.metrics["accuracy"])[:100])

            degradation = baseline_acc - recent_acc
            if degradation > self.config.retraining_trigger_threshold:
                self._trigger_alert(
                    severity=AlertSeverity.CRITICAL,
                    message=f"Model performance degraded: -{degradation:.2%}. Consider retraining.",
                    metric_type="accuracy_degradation",
                    metric_value=degradation,
                    threshold=self.config.retraining_trigger_threshold,
                )

    # =========================================================================
    # DRIFT DETECTION
    # =========================================================================

    def set_reference_data(self, reference_data: torch.Tensor):
        """
        Set reference data for drift detection.

        Args:
            reference_data: Reference dataset (e.g., training data sample)
        """
        self.reference_data = reference_data
        if self.verbose:
            logger.info(f"Set reference data for drift detection: {reference_data.shape}")

    def detect_drift(
        self, new_data: torch.Tensor, reference_data: torch.Tensor | None = None
    ) -> float:
        """
        Detect data drift between new and reference data.

        Args:
            new_data: New data to check
            reference_data: Reference data (uses stored if None)

        Returns:
            Drift score (0 = no drift, 1 = maximum drift)
        """
        if not self.config.enable_drift_detection:
            return 0.0

        reference_data = reference_data or self.reference_data

        if reference_data is None:
            warnings.warn("No reference data set for drift detection", stacklevel=2)
            return 0.0

        drift_score = self._calculate_drift(new_data, reference_data)

        if drift_score > self.config.drift_threshold and self.verbose:
            logger.warning(f"Drift detected: score = {drift_score:.4f}")

        return drift_score

    def _calculate_drift(
        self, new_data: torch.Tensor, reference_data: torch.Tensor | None = None
    ) -> float:
        """Calculate drift score."""
        reference_data = reference_data or self.reference_data

        if reference_data is None:
            return 0.0

        method = self.config.drift_detection_method

        try:
            if method == "ks_test":
                return self._ks_test_drift(new_data, reference_data)
            elif method == "psi":
                return self._psi_drift(new_data, reference_data)
            elif method == "wasserstein":
                return self._wasserstein_drift(new_data, reference_data)
            else:
                warnings.warn(f"Unknown drift detection method: {method}", stacklevel=2)
                return 0.0
        except Exception as e:
            logger.error(f"Drift calculation failed: {e}")
            return 0.0

    def _ks_test_drift(self, new_data: torch.Tensor, reference_data: torch.Tensor) -> float:
        """Kolmogorov-Smirnov test for drift."""
        from scipy.stats import ks_2samp

        # Flatten tensors
        new_flat = new_data.flatten().cpu().numpy()
        ref_flat = reference_data.flatten().cpu().numpy()

        # KS test
        statistic, p_value = ks_2samp(ref_flat, new_flat)

        # Return statistic as drift score
        return statistic

    def _psi_drift(self, new_data: torch.Tensor, reference_data: torch.Tensor) -> float:
        """Population Stability Index for drift."""
        # Flatten tensors
        new_flat = new_data.flatten().cpu().numpy()
        ref_flat = reference_data.flatten().cpu().numpy()

        # Create bins
        bins = np.percentile(ref_flat, np.linspace(0, 100, 11))

        # Calculate distributions
        ref_dist, _ = np.histogram(ref_flat, bins=bins)
        new_dist, _ = np.histogram(new_flat, bins=bins)

        # Normalize
        ref_dist = ref_dist / ref_dist.sum()
        new_dist = new_dist / new_dist.sum()

        # Calculate PSI
        epsilon = 1e-10
        psi = np.sum((new_dist - ref_dist) * np.log((new_dist + epsilon) / (ref_dist + epsilon)))

        return abs(psi)

    def _wasserstein_drift(self, new_data: torch.Tensor, reference_data: torch.Tensor) -> float:
        """Wasserstein distance for drift."""
        from scipy.stats import wasserstein_distance

        # Flatten tensors
        new_flat = new_data.flatten().cpu().numpy()
        ref_flat = reference_data.flatten().cpu().numpy()

        # Calculate Wasserstein distance
        distance = wasserstein_distance(ref_flat, new_flat)

        return distance

    # =========================================================================
    # HEALTH CHECKS
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check.

        Returns:
            Dictionary with health status
        """
        if not self.config.enable_health_checks:
            return {"status": "disabled"}

        self.last_health_check = datetime.now()

        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()

        # Calculate error rate
        error_rate = self.total_errors / max(self.total_predictions, 1)

        # Determine health status
        if error_rate > 0.1:
            self.is_healthy = False
            status = "unhealthy"
        elif error_rate > 0.05:
            self.is_healthy = True
            status = "degraded"
        else:
            self.is_healthy = True
            status = "healthy"

        health_info = {
            "status": status,
            "is_healthy": self.is_healthy,
            "uptime_seconds": uptime,
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "error_rate": error_rate,
            "last_check": self.last_health_check.isoformat(),
        }

        if self.verbose and status != "healthy":
            logger.warning(f"Health check: {status}")

        return health_info

    # =========================================================================
    # ALERTING
    # =========================================================================

    def _trigger_alert(
        self,
        severity: AlertSeverity,
        message: str,
        metric_type: str,
        metric_value: float,
        threshold: float,
    ):
        """Trigger an alert."""
        if not self.config.enable_alerting:
            return

        alert = Alert(
            severity=severity,
            message=message,
            metric_type=metric_type,
            metric_value=metric_value,
            threshold=threshold,
        )

        self.alerts.append(alert)

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        if self.verbose:
            logger.log(
                logging.CRITICAL if severity == AlertSeverity.CRITICAL else logging.WARNING,
                f"ALERT [{severity.value}]: {message}",
            )

    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Register callback for alerts.

        Args:
            callback: Function to call when alert triggered
        """
        self.alert_callbacks.append(callback)

    def get_alerts(
        self, severity: AlertSeverity | None = None, since: datetime | None = None
    ) -> list[Alert]:
        """
        Get alerts.

        Args:
            severity: Filter by severity
            since: Filter alerts since datetime

        Returns:
            List of alerts
        """
        alerts = self.alerts

        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]

        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]

        return alerts

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with metric statistics
        """
        summary = {}

        for metric_name, values in self.metrics.items():
            if len(values) == 0:
                continue

            values_array = np.array(list(values))

            summary[metric_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "p50": float(np.percentile(values_array, 50)),
                "p95": float(np.percentile(values_array, 95)),
                "p99": float(np.percentile(values_array, 99)),
                "count": len(values),
            }

        return summary

    def get_metric(self, metric_name: str) -> list[float] | None:
        """Get specific metric values."""
        if metric_name in self.metrics:
            return list(self.metrics[metric_name])
        return None

    def export_metrics(self, filepath: str | Path):
        """
        Export metrics to JSON file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "model_name": self.model_name,
            "export_time": datetime.now().isoformat(),
            "summary": self.get_metrics_summary(),
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "alerts": [a.to_dict() for a in self.alerts],
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        if self.verbose:
            logger.info(f"Exported metrics to {filepath}")

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.total_predictions = 0
        self.total_errors = 0
        self.start_time = datetime.now()

        if self.verbose:
            logger.info("Reset all metrics")

    def print_monitoring_summary(self):
        """Print formatted monitoring summary."""
        print("=" * 70)
        print(f"Monitoring Summary - {self.model_name}")
        print("=" * 70)
        print(f"Total Predictions: {self.total_predictions}")
        print(f"Total Errors: {self.total_errors}")
        print(f"Error Rate: {self.total_errors / max(self.total_predictions, 1):.2%}")
        print(f"Uptime: {(datetime.now() - self.start_time).total_seconds():.0f}s")
        print(f"Health Status: {'Healthy' if self.is_healthy else 'Unhealthy'}")
        print(f"Active Alerts: {len(self.alerts)}")

        print("\nMetrics:")
        summary = self.get_metrics_summary()
        for metric_name, stats in summary.items():
            print(f"  {metric_name}:")
            print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"    P50: {stats['p50']:.4f}, P95: {stats['p95']:.4f}, P99: {stats['p99']:.4f}")

        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_monitor(model_name: str = "model", enable_all: bool = True) -> ModelMonitor:
    """
    Create monitor with default settings.

    Args:
        model_name: Name of model
        enable_all: Enable all monitoring features

    Returns:
        ModelMonitor instance
    """
    if enable_all:
        config = MonitoringConfig(
            enable_performance_tracking=True,
            enable_drift_detection=True,
            enable_health_checks=True,
            enable_alerting=True,
        )
    else:
        config = MonitoringConfig()

    return ModelMonitor(config=config, model_name=model_name)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("monitoring module loaded")
