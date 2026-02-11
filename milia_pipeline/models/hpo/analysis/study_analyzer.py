"""
Study Analyzer Module

Provides comprehensive analysis utilities for HPO studies including parameter
importance analysis, convergence tracking, visualization data preparation,
statistical analysis, and export functionality.

This module is designed to work with completed HPO studies from the HPOManager
and provides both single-study and multi-study comparison capabilities.

Features:
    - Parameter importance analysis (fANOVA, MDI)
    - Trial history and convergence analysis
    - Optimization trajectory tracking
    - Multi-objective Pareto front analysis
    - Statistical distribution analysis
    - Visualization data preparation (for plots)
    - Export to multiple formats (DataFrame, JSON, CSV, dict)
    - Study comparison utilities

Pattern: Follows HPOManager structure (hpo_manager.py)

Author: Milia Team
Version: 1.1.0

Pydantic V2 Migration (Phase 15):
    - Migrated AnalysisConfig from @dataclass(frozen=True) to BaseModel with frozen=True
    - Uses @field_validator for individual field validation (convergence_window, n_importance_trials)
    - Uses @model_validator(mode='after') for cross-field validation (percentile_thresholds)
    - Added to_dict() method for backward compatibility (wraps model_dump())
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established patterns from meta_features.py (Phase 12)
"""

import logging
from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Tuple,
    Union,
    Callable,
    TYPE_CHECKING,
)
from enum import Enum
import json
from pathlib import Path

# Optional imports with graceful fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import optuna
    from optuna.trial import TrialState
    from optuna.importance import get_param_importances
    from optuna.importance import FanovaImportanceEvaluator
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TrialState = None
    get_param_importances = None
    FanovaImportanceEvaluator = None
    OPTUNA_AVAILABLE = False

# Import HPO-specific modules
from milia_pipeline.exceptions import (
    HPOError,
    HPOConfigurationError,
    StudyNotFoundError,
)

if TYPE_CHECKING:
    from .hpo_manager import HPOManager
    from .hpo_config import HPOConfig

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class ImportanceMethod(Enum):
    """
    Methods for computing parameter importance.
    
    Attributes:
        FANOVA: Functional ANOVA (default, most accurate)
        MDI: Mean Decrease Impurity (faster, less accurate)
    """
    FANOVA = "fanova"
    MDI = "mdi"


class ExportFormat(Enum):
    """
    Supported export formats for analysis results.
    
    Attributes:
        JSON: JSON file format
        CSV: CSV file format (requires pandas)
        DATAFRAME: Pandas DataFrame (requires pandas)
        DICT: Python dictionary
    """
    JSON = "json"
    CSV = "csv"
    DATAFRAME = "dataframe"
    DICT = "dict"


class AnalysisConfig(BaseModel, frozen=True):
    """
    Configuration for study analysis.
    
    Pattern: Follows frozen BaseModel pattern from meta_features.py (Pydantic V2)
    
    Attributes:
        importance_method: Method for parameter importance (fanova/mdi)
        n_importance_trials: Number of trials to use for importance calculation
        convergence_window: Window size for convergence analysis
        include_pruned: Whether to include pruned trials in analysis
        include_failed: Whether to include failed trials in analysis
        percentile_thresholds: Percentiles to compute for distributions
        
    Examples:
        >>> config = AnalysisConfig(importance_method=ImportanceMethod.FANOVA)
        >>> config = AnalysisConfig(convergence_window=20, include_pruned=True)
    """
    importance_method: ImportanceMethod = ImportanceMethod.FANOVA
    n_importance_trials: Optional[int] = None
    convergence_window: int = 10
    include_pruned: bool = False
    include_failed: bool = False
    percentile_thresholds: Tuple[float, ...] = (25.0, 50.0, 75.0, 90.0, 95.0)
    
    @field_validator('convergence_window')
    @classmethod
    def validate_convergence_window(cls, v: int) -> int:
        """Validate convergence_window is at least 1."""
        if v < 1:
            raise ValueError(
                f"convergence_window must be at least 1, got {v}"
            )
        return v
    
    @field_validator('n_importance_trials')
    @classmethod
    def validate_n_importance_trials(cls, v: Optional[int]) -> Optional[int]:
        """Validate n_importance_trials is at least 1 or None."""
        if v is not None and v < 1:
            raise ValueError(
                f"n_importance_trials must be at least 1 or None, got {v}"
            )
        return v
    
    @model_validator(mode='after')
    def validate_percentile_thresholds(self) -> Self:
        """Validate all percentile values are between 0 and 100."""
        for p in self.percentile_thresholds:
            if not 0 <= p <= 100:
                raise ValueError(
                    f"Percentile values must be between 0 and 100, got {p}"
                )
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()


# =============================================================================
# MAIN STUDY ANALYZER CLASS
# =============================================================================

class StudyAnalyzer:
    """
    Comprehensive analyzer for HPO studies.
    
    Provides analysis utilities for completed HPO studies including:
    - Parameter importance analysis
    - Convergence and trajectory tracking
    - Statistical distribution analysis
    - Multi-objective Pareto front analysis
    - Visualization data preparation
    - Export functionality
    
    Pattern: Follows HPOManager structure (hpo_manager.py)
    
    Attributes:
        study: Optuna Study object to analyze
        config: AnalysisConfig for analysis settings
        _trials_cache: Cached trial data for performance
        _importance_cache: Cached importance results
    
    Usage:
        >>> # From HPOManager
        >>> analyzer = StudyAnalyzer.from_manager(hpo_manager)
        >>> 
        >>> # From Optuna study directly
        >>> analyzer = StudyAnalyzer(study)
        >>> 
        >>> # Get parameter importance
        >>> importance = analyzer.get_parameter_importance()
        >>> 
        >>> # Get convergence data
        >>> convergence = analyzer.get_convergence_data()
        >>> 
        >>> # Export results
        >>> analyzer.export_results("results.json", format=ExportFormat.JSON)
    
    Examples:
        >>> # Full analysis workflow
        >>> analyzer = StudyAnalyzer.from_manager(manager)
        >>> 
        >>> # Parameter analysis
        >>> importance = analyzer.get_parameter_importance()
        >>> correlations = analyzer.get_parameter_correlations()
        >>> 
        >>> # Performance analysis
        >>> convergence = analyzer.get_convergence_data()
        >>> trajectory = analyzer.get_optimization_trajectory()
        >>> 
        >>> # Export all results
        >>> results = analyzer.get_comprehensive_analysis()
        >>> analyzer.export_results("analysis.json")
    """
    
    def __init__(
        self,
        study: 'optuna.Study',
        config: Optional[AnalysisConfig] = None,
    ):
        """
        Initialize StudyAnalyzer.
        
        Args:
            study: Optuna Study object to analyze
            config: Analysis configuration (optional)
            
        Raises:
            HPOError: If Optuna is not available or study is invalid
        """
        if not OPTUNA_AVAILABLE:
            raise HPOError(
                "Optuna is not installed",
                details="Install with: pip install optuna"
            )
        
        if study is None:
            raise HPOError(
                "Study cannot be None",
                details="Provide a valid Optuna Study object"
            )
        
        self.study = study
        self.config = config or AnalysisConfig()
        
        # Caches for performance
        self._trials_cache: Optional[List[Dict[str, Any]]] = None
        self._importance_cache: Optional[Dict[str, float]] = None
        self._completed_trials_cache: Optional[List[Dict[str, Any]]] = None
        
        logger.info(
            f"StudyAnalyzer initialized for study '{study.study_name}' "
            f"with {len(study.trials)} trials"
        )
    
    @classmethod
    def from_manager(
        cls,
        manager: 'HPOManager',
        config: Optional[AnalysisConfig] = None,
    ) -> 'StudyAnalyzer':
        """
        Create StudyAnalyzer from HPOManager.
        
        Args:
            manager: HPOManager instance with completed study
            config: Analysis configuration (optional)
            
        Returns:
            StudyAnalyzer instance
            
        Raises:
            HPOError: If manager has no study
            
        Examples:
            >>> manager.optimize(model_name="GCN", dataset=dataset)
            >>> analyzer = StudyAnalyzer.from_manager(manager)
        """
        if manager.study is None:
            raise HPOError(
                "HPOManager has no study",
                details="Run optimize() before creating analyzer"
            )
        
        return cls(manager.study, config)
    
    @classmethod
    def from_storage(
        cls,
        study_name: str,
        storage: str,
        config: Optional[AnalysisConfig] = None,
    ) -> 'StudyAnalyzer':
        """
        Create StudyAnalyzer by loading study from storage.
        
        Args:
            study_name: Name of the study to load
            storage: Storage URL (e.g., "sqlite:///optuna.db")
            config: Analysis configuration (optional)
            
        Returns:
            StudyAnalyzer instance
            
        Raises:
            StudyNotFoundError: If study not found in storage
            
        Examples:
            >>> analyzer = StudyAnalyzer.from_storage(
            ...     study_name="gcn_hpo",
            ...     storage="sqlite:///hpo_results.db"
            ... )
        """
        if not OPTUNA_AVAILABLE:
            raise HPOError(
                "Optuna is not installed",
                details="Install with: pip install optuna"
            )
        
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
            )
            return cls(study, config)
        except Exception as e:
            raise StudyNotFoundError(
                f"Failed to load study: {e}",
                study_name=study_name,
                storage_url=storage,
            )
    
    # =========================================================================
    # TRIAL DATA ACCESS
    # =========================================================================
    
    def get_trials(
        self,
        states: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get trial information with optional filtering by state.
        
        Args:
            states: List of states to include (COMPLETE, PRUNED, FAIL)
                   None means all trials based on config
            use_cache: Whether to use cached results
            
        Returns:
            List of trial info dicts with keys:
            - number: Trial number
            - params: Hyperparameters
            - value: Objective value (None if not completed)
            - state: Trial state
            - duration: Duration in seconds
            - user_attrs: User attributes
            - intermediate_values: Intermediate reported values
            
        Examples:
            >>> trials = analyzer.get_trials()
            >>> completed = analyzer.get_trials(states=['COMPLETE'])
        """
        if use_cache and self._trials_cache is not None and states is None:
            return self._trials_cache
        
        trials_info = []
        
        for trial in self.study.trials:
            state_name = trial.state.name
            
            # Filter by state if specified
            if states is not None:
                if state_name not in states:
                    continue
            else:
                # Use config settings
                if state_name == 'PRUNED' and not self.config.include_pruned:
                    continue
                if state_name == 'FAIL' and not self.config.include_failed:
                    continue
            
            trial_info = {
                'number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': state_name,
                'duration': (
                    (trial.datetime_complete - trial.datetime_start).total_seconds()
                    if trial.datetime_complete and trial.datetime_start
                    else None
                ),
                'user_attrs': trial.user_attrs,
                'intermediate_values': trial.intermediate_values,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete,
            }
            trials_info.append(trial_info)
        
        if states is None:
            self._trials_cache = trials_info
        
        return trials_info
    
    def get_completed_trials(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get only completed trials.
        
        Args:
            use_cache: Whether to use cached results
            
        Returns:
            List of completed trial info dicts
        """
        if use_cache and self._completed_trials_cache is not None:
            return self._completed_trials_cache
        
        completed = self.get_trials(states=['COMPLETE'], use_cache=False)
        self._completed_trials_cache = completed
        return completed
    
    def get_trial_count(self) -> Dict[str, int]:
        """
        Get count of trials by state.
        
        Returns:
            Dict with state counts:
            - total: Total number of trials
            - completed: Successfully completed trials
            - pruned: Pruned trials
            - failed: Failed trials
            - running: Currently running trials
        """
        counts = {
            'total': 0,
            'completed': 0,
            'pruned': 0,
            'failed': 0,
            'running': 0,
        }
        
        for trial in self.study.trials:
            counts['total'] += 1
            state = trial.state.name
            if state == 'COMPLETE':
                counts['completed'] += 1
            elif state == 'PRUNED':
                counts['pruned'] += 1
            elif state == 'FAIL':
                counts['failed'] += 1
            elif state == 'RUNNING':
                counts['running'] += 1
        
        return counts
    
    # =========================================================================
    # PARAMETER IMPORTANCE ANALYSIS
    # =========================================================================
    
    def get_parameter_importance(
        self,
        method: Optional[ImportanceMethod] = None,
        target: Optional[Callable[[optuna.trial.FrozenTrial], float]] = None,
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate parameter importance scores.
        
        Uses Optuna's importance module to calculate how much each
        hyperparameter contributes to the objective value.
        
        Args:
            method: Importance calculation method (fanova/mdi)
                   None uses config default
            target: Custom target function for importance calculation
            use_cache: Whether to use cached results
            
        Returns:
            Dict mapping parameter names to importance scores (0-1)
            Scores sum to 1.0
            
        Raises:
            HPOError: If not enough completed trials
            
        Examples:
            >>> importance = analyzer.get_parameter_importance()
            >>> print(f"Learning rate importance: {importance.get('lr', 0):.3f}")
            >>> 
            >>> # Get importance for specific target
            >>> importance = analyzer.get_parameter_importance(
            ...     target=lambda t: t.user_attrs.get('val_accuracy', 0)
            ... )
        """
        if use_cache and self._importance_cache is not None and method is None:
            return self._importance_cache
        
        completed_count = len(self.get_completed_trials())
        if completed_count < 2:
            raise HPOError(
                "Not enough completed trials for importance analysis",
                study_name=self.study.study_name,
                details=f"Need at least 2 completed trials, have {completed_count}"
            )
        
        method = method or self.config.importance_method
        
        try:
            if method == ImportanceMethod.FANOVA:
                evaluator = FanovaImportanceEvaluator()
            else:
                # MDI uses default evaluator
                evaluator = None
            
            importance = get_param_importances(
                self.study,
                evaluator=evaluator,
                target=target,
            )
            
            if method is None or method == self.config.importance_method:
                self._importance_cache = importance
            
            logger.debug(f"Calculated parameter importance: {importance}")
            return importance
            
        except Exception as e:
            raise HPOError(
                f"Failed to calculate parameter importance: {e}",
                study_name=self.study.study_name,
                details=str(e)
            )
    
    def get_parameter_importance_ranking(
        self,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get parameters ranked by importance.
        
        Args:
            top_k: Number of top parameters to return (None for all)
            
        Returns:
            List of (parameter_name, importance_score) tuples,
            sorted by importance descending
            
        Examples:
            >>> ranking = analyzer.get_parameter_importance_ranking(top_k=5)
            >>> for param, score in ranking:
            ...     print(f"{param}: {score:.3f}")
        """
        importance = self.get_parameter_importance()
        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked
    
    # =========================================================================
    # CONVERGENCE AND TRAJECTORY ANALYSIS
    # =========================================================================
    
    def get_convergence_data(self) -> Dict[str, Any]:
        """
        Analyze optimization convergence.
        
        Tracks how the best objective value improves over trials
        and calculates convergence metrics.
        
        Returns:
            Dict containing:
            - trial_numbers: List of trial numbers
            - best_values: Running best value at each trial
            - improvements: Improvement at each trial
            - convergence_rate: Rate of improvement
            - converged: Whether optimization appears converged
            - convergence_trial: Trial at which convergence was detected
            
        Examples:
            >>> convergence = analyzer.get_convergence_data()
            >>> if convergence['converged']:
            ...     print(f"Converged at trial {convergence['convergence_trial']}")
        """
        completed = self.get_completed_trials()
        
        if not completed:
            return {
                'trial_numbers': [],
                'best_values': [],
                'improvements': [],
                'convergence_rate': 0.0,
                'converged': False,
                'convergence_trial': None,
            }
        
        # Sort by trial number
        sorted_trials = sorted(completed, key=lambda t: t['number'])
        
        trial_numbers = []
        best_values = []
        improvements = []
        
        # Determine if we're minimizing or maximizing
        direction = self.study.direction.name
        is_minimize = direction == 'MINIMIZE'
        
        current_best = None
        
        for trial in sorted_trials:
            value = trial['value']
            if value is None:
                continue
            
            trial_numbers.append(trial['number'])
            
            if current_best is None:
                current_best = value
                improvement = 0.0
            else:
                if is_minimize:
                    if value < current_best:
                        improvement = current_best - value
                        current_best = value
                    else:
                        improvement = 0.0
                else:
                    if value > current_best:
                        improvement = value - current_best
                        current_best = value
                    else:
                        improvement = 0.0
            
            best_values.append(current_best)
            improvements.append(improvement)
        
        # Calculate convergence metrics
        window = min(self.config.convergence_window, len(improvements))
        if window > 0:
            recent_improvements = improvements[-window:]
            convergence_rate = sum(recent_improvements) / window
            
            # Consider converged if no improvement in window
            converged = all(imp == 0.0 for imp in recent_improvements)
            
            # Find convergence trial
            convergence_trial = None
            if converged and len(improvements) >= window:
                for i in range(len(improvements) - window, -1, -1):
                    if improvements[i] > 0:
                        convergence_trial = trial_numbers[i + 1] if i + 1 < len(trial_numbers) else None
                        break
        else:
            convergence_rate = 0.0
            converged = False
            convergence_trial = None
        
        return {
            'trial_numbers': trial_numbers,
            'best_values': best_values,
            'improvements': improvements,
            'convergence_rate': convergence_rate,
            'converged': converged,
            'convergence_trial': convergence_trial,
        }
    
    def get_optimization_trajectory(self) -> Dict[str, Any]:
        """
        Get the optimization trajectory over time.
        
        Tracks objective values, parameters, and timing for each trial.
        
        Returns:
            Dict containing:
            - trials: List of trial data with value, params, time
            - duration_total: Total optimization duration
            - duration_mean: Mean trial duration
            - values: List of objective values
            - best_value: Best objective value achieved
            - best_trial: Trial number of best result
            
        Examples:
            >>> trajectory = analyzer.get_optimization_trajectory()
            >>> print(f"Best value: {trajectory['best_value']}")
            >>> print(f"Total duration: {trajectory['duration_total']:.1f}s")
        """
        completed = self.get_completed_trials()
        
        if not completed:
            return {
                'trials': [],
                'duration_total': 0.0,
                'duration_mean': 0.0,
                'values': [],
                'best_value': None,
                'best_trial': None,
            }
        
        # Sort by trial number
        sorted_trials = sorted(completed, key=lambda t: t['number'])
        
        trials_data = []
        values = []
        durations = []
        
        for trial in sorted_trials:
            trial_data = {
                'number': trial['number'],
                'value': trial['value'],
                'params': trial['params'],
                'duration': trial['duration'],
            }
            trials_data.append(trial_data)
            
            if trial['value'] is not None:
                values.append(trial['value'])
            if trial['duration'] is not None:
                durations.append(trial['duration'])
        
        # Find best
        direction = self.study.direction.name
        is_minimize = direction == 'MINIMIZE'
        
        if values:
            if is_minimize:
                best_value = min(values)
            else:
                best_value = max(values)
            
            # Find best trial
            best_trial = None
            for trial in sorted_trials:
                if trial['value'] == best_value:
                    best_trial = trial['number']
                    break
        else:
            best_value = None
            best_trial = None
        
        return {
            'trials': trials_data,
            'duration_total': sum(durations) if durations else 0.0,
            'duration_mean': sum(durations) / len(durations) if durations else 0.0,
            'values': values,
            'best_value': best_value,
            'best_trial': best_trial,
        }
    
    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================
    
    def get_value_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for objective values.
        
        Returns:
            Dict containing:
            - count: Number of completed trials
            - mean: Mean objective value
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - median: Median value
            - percentiles: Dict of percentile values
            - range: Max - min
            
        Examples:
            >>> stats = analyzer.get_value_statistics()
            >>> print(f"Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        """
        completed = self.get_completed_trials()
        values = [t['value'] for t in completed if t['value'] is not None]
        
        if not values:
            return {
                'count': 0,
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'median': None,
                'percentiles': {},
                'range': None,
            }
        
        count = len(values)
        mean_val = sum(values) / count
        
        # Calculate std
        if count > 1:
            variance = sum((v - mean_val) ** 2 for v in values) / (count - 1)
            std_val = variance ** 0.5
        else:
            std_val = 0.0
        
        sorted_values = sorted(values)
        min_val = sorted_values[0]
        max_val = sorted_values[-1]
        
        # Median
        if count % 2 == 0:
            median_val = (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2
        else:
            median_val = sorted_values[count // 2]
        
        # Percentiles
        percentiles = {}
        for p in self.config.percentile_thresholds:
            idx = int((p / 100) * (count - 1))
            percentiles[p] = sorted_values[idx]
        
        return {
            'count': count,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'median': median_val,
            'percentiles': percentiles,
            'range': max_val - min_val,
        }
    
    def get_parameter_statistics(
        self,
        parameter_name: str,
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a specific parameter across trials.
        
        Args:
            parameter_name: Name of the parameter to analyze
            
        Returns:
            Dict containing statistics for the parameter values
            
        Raises:
            HPOError: If parameter not found
            
        Examples:
            >>> stats = analyzer.get_parameter_statistics('lr')
            >>> print(f"LR range: {stats['min']} - {stats['max']}")
        """
        completed = self.get_completed_trials()
        values = []
        
        for trial in completed:
            params = trial['params']
            # Handle both flat and prefixed parameter names
            if parameter_name in params:
                values.append(params[parameter_name])
            else:
                # Try to find with prefix
                for key, val in params.items():
                    if key.endswith(f'.{parameter_name}') or key == parameter_name:
                        values.append(val)
                        break
        
        if not values:
            raise HPOError(
                f"Parameter '{parameter_name}' not found in trials",
                study_name=self.study.study_name
            )
        
        # Check if categorical
        if all(isinstance(v, (str, bool)) for v in values):
            # Categorical analysis
            from collections import Counter
            counts = Counter(values)
            return {
                'type': 'categorical',
                'count': len(values),
                'unique_values': list(counts.keys()),
                'value_counts': dict(counts),
                'mode': counts.most_common(1)[0][0] if counts else None,
            }
        
        # Numeric analysis
        numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
        
        if not numeric_values:
            return {
                'type': 'mixed',
                'count': len(values),
                'values': values,
            }
        
        count = len(numeric_values)
        mean_val = sum(numeric_values) / count
        
        if count > 1:
            variance = sum((v - mean_val) ** 2 for v in numeric_values) / (count - 1)
            std_val = variance ** 0.5
        else:
            std_val = 0.0
        
        sorted_values = sorted(numeric_values)
        
        return {
            'type': 'numeric',
            'count': count,
            'mean': mean_val,
            'std': std_val,
            'min': sorted_values[0],
            'max': sorted_values[-1],
            'median': sorted_values[count // 2],
        }
    
    def get_parameter_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations between parameters and objective value.
        
        Returns:
            Dict mapping parameter names to correlation with objective
            and correlations between parameters
            
        Examples:
            >>> correlations = analyzer.get_parameter_correlations()
            >>> print(f"LR-objective correlation: {correlations['objective'].get('lr', 0):.3f}")
        """
        completed = self.get_completed_trials()
        
        if len(completed) < 3:
            logger.warning("Not enough trials for correlation analysis")
            return {'objective': {}, 'parameters': {}}
        
        # Extract numeric parameters
        param_names = set()
        for trial in completed:
            for key, val in trial['params'].items():
                if isinstance(val, (int, float)):
                    param_names.add(key)
        
        param_names = sorted(param_names)
        
        if not param_names:
            return {'objective': {}, 'parameters': {}}
        
        # Build data matrix
        objective_values = []
        param_values = {name: [] for name in param_names}
        
        for trial in completed:
            if trial['value'] is None:
                continue
            
            objective_values.append(trial['value'])
            for name in param_names:
                val = trial['params'].get(name, 0)
                param_values[name].append(float(val) if isinstance(val, (int, float)) else 0)
        
        # Calculate correlations
        def pearson_correlation(x: List[float], y: List[float]) -> float:
            """Calculate Pearson correlation coefficient."""
            n = len(x)
            if n < 2:
                return 0.0
            
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            
            numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            
            var_x = sum((xi - mean_x) ** 2 for xi in x)
            var_y = sum((yi - mean_y) ** 2 for yi in y)
            
            denominator = (var_x * var_y) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        
        # Objective correlations
        objective_corr = {}
        for name in param_names:
            objective_corr[name] = pearson_correlation(param_values[name], objective_values)
        
        # Parameter-parameter correlations
        param_corr = {}
        for name1 in param_names:
            param_corr[name1] = {}
            for name2 in param_names:
                if name1 != name2:
                    param_corr[name1][name2] = pearson_correlation(
                        param_values[name1],
                        param_values[name2]
                    )
        
        return {
            'objective': objective_corr,
            'parameters': param_corr,
        }
    
    # =========================================================================
    # MULTI-OBJECTIVE ANALYSIS
    # =========================================================================
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """
        Get Pareto-optimal trials for multi-objective studies.
        
        Returns:
            List of trial dicts on the Pareto front with:
            - number: Trial number
            - values: Objective values (tuple)
            - params: Hyperparameters
            
        Raises:
            HPOError: If study is not multi-objective
            
        Examples:
            >>> pareto = analyzer.get_pareto_front()
            >>> for trial in pareto:
            ...     print(f"Trial {trial['number']}: {trial['values']}")
        """
        # Check if multi-objective
        try:
            directions = self.study.directions
            if len(directions) < 2:
                raise HPOError(
                    "get_pareto_front() requires multi-objective study",
                    study_name=self.study.study_name,
                    details="Study has only one objective"
                )
        except AttributeError:
            raise HPOError(
                "get_pareto_front() requires multi-objective study",
                study_name=self.study.study_name
            )
        
        # Get Pareto front from Optuna
        best_trials = self.study.best_trials
        
        return [
            {
                'number': t.number,
                'values': t.values,
                'params': t.params,
            }
            for t in best_trials
        ]
    
    def get_hypervolume(
        self,
        reference_point: Optional[Tuple[float, ...]] = None,
    ) -> float:
        """
        Calculate hypervolume indicator for multi-objective study.
        
        The hypervolume measures the volume of objective space dominated
        by the Pareto front, relative to a reference point.
        
        Args:
            reference_point: Reference point for hypervolume calculation
                           (must be worse than all Pareto points)
            
        Returns:
            Hypervolume value (higher is better)
            
        Raises:
            HPOError: If study is not multi-objective or reference point invalid
            
        Examples:
            >>> hv = analyzer.get_hypervolume(reference_point=(1.0, 100.0))
            >>> print(f"Hypervolume: {hv:.4f}")
        """
        pareto = self.get_pareto_front()
        
        if not pareto:
            return 0.0
        
        if reference_point is None:
            raise HPOError(
                "reference_point required for hypervolume calculation",
                study_name=self.study.study_name
            )
        
        try:
            from optuna._hypervolume import WFG
            
            points = [t['values'] for t in pareto]
            wfg = WFG()
            return wfg.compute(points, reference_point)
            
        except ImportError:
            raise HPOError(
                "Hypervolume calculation requires Optuna >= 3.0",
                study_name=self.study.study_name
            )
    
    # =========================================================================
    # VISUALIZATION DATA PREPARATION
    # =========================================================================
    
    def get_optimization_history_data(self) -> Dict[str, Any]:
        """
        Prepare data for optimization history plot.
        
        Returns:
            Dict with data for plotting:
            - trial_numbers: X-axis values
            - values: Objective values for each trial
            - best_values: Running best at each trial
            - infeasible_trials: Trials that failed or were pruned
            
        Examples:
            >>> data = analyzer.get_optimization_history_data()
            >>> # Use with matplotlib:
            >>> plt.plot(data['trial_numbers'], data['best_values'])
        """
        all_trials = self.get_trials(states=['COMPLETE', 'PRUNED', 'FAIL'], use_cache=False)
        sorted_trials = sorted(all_trials, key=lambda t: t['number'])
        
        direction = self.study.direction.name
        is_minimize = direction == 'MINIMIZE'
        
        trial_numbers = []
        values = []
        best_values = []
        infeasible_trials = []
        
        current_best = None
        
        for trial in sorted_trials:
            trial_numbers.append(trial['number'])
            
            if trial['state'] != 'COMPLETE' or trial['value'] is None:
                values.append(None)
                infeasible_trials.append(trial['number'])
                best_values.append(current_best)
            else:
                value = trial['value']
                values.append(value)
                
                if current_best is None:
                    current_best = value
                elif is_minimize and value < current_best:
                    current_best = value
                elif not is_minimize and value > current_best:
                    current_best = value
                
                best_values.append(current_best)
        
        return {
            'trial_numbers': trial_numbers,
            'values': values,
            'best_values': best_values,
            'infeasible_trials': infeasible_trials,
            'direction': direction,
        }
    
    def get_parameter_importance_data(self) -> Dict[str, Any]:
        """
        Prepare data for parameter importance plot.
        
        Returns:
            Dict with data for plotting:
            - parameters: Parameter names
            - importances: Importance scores
            - sorted_indices: Indices for sorted display
            
        Examples:
            >>> data = analyzer.get_parameter_importance_data()
            >>> # Use with matplotlib:
            >>> plt.barh(data['parameters'], data['importances'])
        """
        try:
            importance = self.get_parameter_importance()
        except HPOError:
            return {
                'parameters': [],
                'importances': [],
                'sorted_indices': [],
            }
        
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        parameters = [item[0] for item in sorted_items]
        importances = [item[1] for item in sorted_items]
        sorted_indices = list(range(len(parameters)))
        
        return {
            'parameters': parameters,
            'importances': importances,
            'sorted_indices': sorted_indices,
        }
    
    def get_slice_plot_data(
        self,
        parameter_name: str,
    ) -> Dict[str, Any]:
        """
        Prepare data for parameter slice plot.
        
        Shows objective value vs parameter value for a single parameter.
        
        Args:
            parameter_name: Parameter to analyze
            
        Returns:
            Dict with data for plotting:
            - parameter_values: Parameter values
            - objective_values: Corresponding objective values
            - best_value: Best objective value
            - best_param: Parameter value at best objective
            
        Examples:
            >>> data = analyzer.get_slice_plot_data('lr')
            >>> plt.scatter(data['parameter_values'], data['objective_values'])
        """
        completed = self.get_completed_trials()
        
        param_values = []
        obj_values = []
        
        for trial in completed:
            if trial['value'] is None:
                continue
            
            # Get parameter value
            param_val = None
            for key, val in trial['params'].items():
                if key == parameter_name or key.endswith(f'.{parameter_name}'):
                    param_val = val
                    break
            
            if param_val is not None and isinstance(param_val, (int, float)):
                param_values.append(param_val)
                obj_values.append(trial['value'])
        
        # Find best
        direction = self.study.direction.name
        is_minimize = direction == 'MINIMIZE'
        
        if obj_values:
            if is_minimize:
                best_idx = obj_values.index(min(obj_values))
            else:
                best_idx = obj_values.index(max(obj_values))
            
            best_value = obj_values[best_idx]
            best_param = param_values[best_idx]
        else:
            best_value = None
            best_param = None
        
        return {
            'parameter_values': param_values,
            'objective_values': obj_values,
            'best_value': best_value,
            'best_param': best_param,
            'parameter_name': parameter_name,
        }
    
    def get_contour_plot_data(
        self,
        param_x: str,
        param_y: str,
    ) -> Dict[str, Any]:
        """
        Prepare data for 2D contour plot of two parameters.
        
        Args:
            param_x: Parameter for X-axis
            param_y: Parameter for Y-axis
            
        Returns:
            Dict with data for plotting:
            - x_values: X parameter values
            - y_values: Y parameter values
            - z_values: Objective values
            
        Examples:
            >>> data = analyzer.get_contour_plot_data('lr', 'hidden_channels')
        """
        completed = self.get_completed_trials()
        
        x_values = []
        y_values = []
        z_values = []
        
        for trial in completed:
            if trial['value'] is None:
                continue
            
            x_val = None
            y_val = None
            
            for key, val in trial['params'].items():
                if key == param_x or key.endswith(f'.{param_x}'):
                    x_val = val
                if key == param_y or key.endswith(f'.{param_y}'):
                    y_val = val
            
            if (x_val is not None and y_val is not None and
                isinstance(x_val, (int, float)) and isinstance(y_val, (int, float))):
                x_values.append(x_val)
                y_values.append(y_val)
                z_values.append(trial['value'])
        
        return {
            'x_values': x_values,
            'y_values': y_values,
            'z_values': z_values,
            'param_x': param_x,
            'param_y': param_y,
        }
    
    def get_parallel_coordinate_data(self) -> Dict[str, Any]:
        """
        Prepare data for parallel coordinate plot.
        
        Returns:
            Dict with data for plotting:
            - parameters: List of parameter names
            - trials: List of dicts with normalized parameter values
            - objective_values: Objective values for coloring
            
        Examples:
            >>> data = analyzer.get_parallel_coordinate_data()
        """
        completed = self.get_completed_trials()
        
        if not completed:
            return {
                'parameters': [],
                'trials': [],
                'objective_values': [],
            }
        
        # Get all numeric parameters
        param_names = set()
        for trial in completed:
            for key, val in trial['params'].items():
                if isinstance(val, (int, float)):
                    param_names.add(key)
        
        param_names = sorted(param_names)
        
        if not param_names:
            return {
                'parameters': [],
                'trials': [],
                'objective_values': [],
            }
        
        # Collect values and normalize
        param_ranges = {name: {'min': float('inf'), 'max': float('-inf')} 
                       for name in param_names}
        
        trials_data = []
        objective_values = []
        
        for trial in completed:
            if trial['value'] is None:
                continue
            
            trial_params = {}
            for name in param_names:
                val = trial['params'].get(name, 0)
                if isinstance(val, (int, float)):
                    trial_params[name] = val
                    param_ranges[name]['min'] = min(param_ranges[name]['min'], val)
                    param_ranges[name]['max'] = max(param_ranges[name]['max'], val)
                else:
                    trial_params[name] = 0
            
            trials_data.append(trial_params)
            objective_values.append(trial['value'])
        
        # Normalize to 0-1
        normalized_trials = []
        for trial_params in trials_data:
            normalized = {}
            for name in param_names:
                range_val = param_ranges[name]['max'] - param_ranges[name]['min']
                if range_val > 0:
                    normalized[name] = (trial_params[name] - param_ranges[name]['min']) / range_val
                else:
                    normalized[name] = 0.5
            normalized_trials.append(normalized)
        
        return {
            'parameters': param_names,
            'trials': normalized_trials,
            'objective_values': objective_values,
            'param_ranges': param_ranges,
        }
    
    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of the study.
        
        Combines all analysis methods into a single report.
        
        Returns:
            Dict containing:
            - study_info: Basic study information
            - trial_counts: Counts by state
            - value_statistics: Objective value statistics
            - convergence: Convergence analysis
            - trajectory: Optimization trajectory
            - importance: Parameter importance (if available)
            - correlations: Parameter correlations
            
        Examples:
            >>> analysis = analyzer.get_comprehensive_analysis()
            >>> print(json.dumps(analysis, indent=2, default=str))
        """
        analysis = {
            'study_info': {
                'study_name': self.study.study_name,
                'direction': self.study.direction.name,
                'n_trials': len(self.study.trials),
            },
            'trial_counts': self.get_trial_count(),
            'value_statistics': self.get_value_statistics(),
            'convergence': self.get_convergence_data(),
            'trajectory': self.get_optimization_trajectory(),
        }
        
        # Try to add importance (may fail with too few trials)
        try:
            analysis['importance'] = self.get_parameter_importance()
        except HPOError as e:
            logger.debug(f"Could not calculate importance: {e}")
            analysis['importance'] = None
        
        # Add correlations
        try:
            analysis['correlations'] = self.get_parameter_correlations()
        except Exception as e:
            logger.debug(f"Could not calculate correlations: {e}")
            analysis['correlations'] = None
        
        return analysis
    
    # =========================================================================
    # EXPORT FUNCTIONALITY
    # =========================================================================
    
    def export_results(
        self,
        path: Optional[Union[str, Path]] = None,
        format: ExportFormat = ExportFormat.JSON,
        include_trials: bool = True,
        include_analysis: bool = True,
    ) -> Union[Dict[str, Any], 'pd.DataFrame', None]:
        """
        Export analysis results to file or return as object.
        
        Args:
            path: Output file path (None to return object)
            format: Export format (json, csv, dataframe, dict)
            include_trials: Whether to include raw trial data
            include_analysis: Whether to include analysis results
            
        Returns:
            Exported data (if path is None) or None (if saved to file)
            
        Raises:
            HPOError: If format not supported or export fails
            
        Examples:
            >>> # Save to JSON file
            >>> analyzer.export_results("results.json", format=ExportFormat.JSON)
            >>> 
            >>> # Get as DataFrame
            >>> df = analyzer.export_results(format=ExportFormat.DATAFRAME)
            >>> 
            >>> # Get as dict
            >>> data = analyzer.export_results(format=ExportFormat.DICT)
        """
        # Build export data
        export_data = {
            'study_name': self.study.study_name,
            'direction': self.study.direction.name,
        }
        
        if include_trials:
            export_data['trials'] = self.get_trials(
                states=['COMPLETE', 'PRUNED', 'FAIL'],
                use_cache=False
            )
        
        if include_analysis:
            export_data['analysis'] = self.get_comprehensive_analysis()
        
        # Handle different formats
        if format == ExportFormat.DICT:
            if path is not None:
                raise HPOError(
                    "DICT format does not support file output",
                    details="Use JSON format for file output"
                )
            return export_data
        
        elif format == ExportFormat.JSON:
            # Make JSON serializable
            def json_serializer(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    return str(obj)
            
            json_str = json.dumps(export_data, indent=2, default=json_serializer)
            
            if path is not None:
                path = Path(path)
                path.write_text(json_str)
                logger.info(f"Exported results to {path}")
                return None
            else:
                return json.loads(json_str)
        
        elif format in (ExportFormat.CSV, ExportFormat.DATAFRAME):
            if not PANDAS_AVAILABLE:
                raise HPOError(
                    "Pandas required for CSV/DataFrame export",
                    details="Install with: pip install pandas"
                )
            
            # Build DataFrame from trials
            trials = export_data.get('trials', [])
            
            if not trials:
                df = pd.DataFrame()
            else:
                rows = []
                for trial in trials:
                    row = {
                        'trial_number': trial['number'],
                        'value': trial['value'],
                        'state': trial['state'],
                        'duration': trial['duration'],
                    }
                    # Flatten params
                    for key, val in trial['params'].items():
                        row[f'param_{key}'] = val
                    rows.append(row)
                
                df = pd.DataFrame(rows)
            
            if format == ExportFormat.DATAFRAME:
                return df
            else:
                # CSV
                if path is None:
                    raise HPOError(
                        "CSV format requires file path",
                        details="Provide path argument for CSV export"
                    )
                path = Path(path)
                df.to_csv(path, index=False)
                logger.info(f"Exported results to {path}")
                return None
        
        else:
            raise HPOError(
                f"Unsupported export format: {format}",
                details=f"Supported formats: {[f.value for f in ExportFormat]}"
            )
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert trial data to pandas DataFrame.
        
        Convenience method for quick DataFrame export.
        
        Returns:
            DataFrame with trial data
            
        Raises:
            HPOError: If pandas not available
            
        Examples:
            >>> df = analyzer.to_dataframe()
            >>> df.head()
        """
        return self.export_results(format=ExportFormat.DATAFRAME)
    
    # =========================================================================
    # STUDY COMPARISON
    # =========================================================================
    
    def compare_with(
        self,
        other: 'StudyAnalyzer',
    ) -> Dict[str, Any]:
        """
        Compare this study with another study.
        
        Args:
            other: Another StudyAnalyzer instance
            
        Returns:
            Dict containing comparison results:
            - studies: Names of compared studies
            - best_values: Best value from each study
            - winner: Study with better best value
            - trial_counts: Trial counts comparison
            - value_statistics: Statistics comparison
            
        Examples:
            >>> comparison = analyzer1.compare_with(analyzer2)
            >>> print(f"Winner: {comparison['winner']}")
        """
        self_stats = self.get_value_statistics()
        other_stats = other.get_value_statistics()
        
        self_trajectory = self.get_optimization_trajectory()
        other_trajectory = other.get_optimization_trajectory()
        
        # Determine winner based on direction
        direction = self.study.direction.name
        is_minimize = direction == 'MINIMIZE'
        
        self_best = self_trajectory['best_value']
        other_best = other_trajectory['best_value']
        
        if self_best is None and other_best is None:
            winner = None
        elif self_best is None:
            winner = other.study.study_name
        elif other_best is None:
            winner = self.study.study_name
        elif is_minimize:
            winner = self.study.study_name if self_best < other_best else other.study.study_name
        else:
            winner = self.study.study_name if self_best > other_best else other.study.study_name
        
        return {
            'studies': [self.study.study_name, other.study.study_name],
            'best_values': {
                self.study.study_name: self_best,
                other.study.study_name: other_best,
            },
            'winner': winner,
            'trial_counts': {
                self.study.study_name: self.get_trial_count(),
                other.study.study_name: other.get_trial_count(),
            },
            'value_statistics': {
                self.study.study_name: self_stats,
                other.study.study_name: other_stats,
            },
            'improvement': (
                (other_best - self_best) if self_best is not None and other_best is not None
                else None
            ),
        }
    
    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================
    
    def clear_cache(self) -> None:
        """
        Clear all cached data.
        
        Call this after modifying the study externally to ensure
        fresh data is loaded.
        """
        self._trials_cache = None
        self._importance_cache = None
        self._completed_trials_cache = None
        logger.debug("Cleared analyzer cache")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StudyAnalyzer(study_name='{self.study.study_name}', "
            f"n_trials={len(self.study.trials)})"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    'StudyAnalyzer',
    # Configuration
    'AnalysisConfig',
    # Enums
    'ImportanceMethod',
    'ExportFormat',
]
