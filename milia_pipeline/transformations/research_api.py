# research_api.py
"""
Experimental Research API for systematic experimentation.

This module provides high-level APIs for:
- Ablation studies (removing/adding transforms)
- Parameter sweeps (varying transform parameters)
- Comparative studies (comparing approaches)

Designed for ISI paper reproducibility and systematic research workflows.

Research-Grade Experimentation Framework
- Fluent builder APIs for experiment configuration
- Automatic experiment execution and result tracking
- Statistical analysis and report generation
- Full integration with milia transformation system
"""

from typing import List, Dict, Any, Optional, Callable, Tuple, Union, Set
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json
import itertools
import logging

# Third-party imports
import numpy as np
import pandas as pd
import yaml

from .graph_transforms import (
    TransformComposer,
    TransformRegistry,
    ValidationLevel,
    validate_comprehensive
)

try:
    from .custom_transforms import CustomTransformBase
    CUSTOM_TRANSFORMS_AVAILABLE = True
except ImportError:
    CUSTOM_TRANSFORMS_AVAILABLE = False
    CustomTransformBase = None

from ..config.config_containers import (
    TransformSpec,
    ExperimentalSetup,  
    TransformationConfig
)
from ..config.config_accessors import (
    get_transformation_config,
    get_experimental_setup,
    list_experimental_setups
)
from ..config.config_loader import load_config


from ..exceptions import (
    ConfigurationError,
    TransformConfigurationError,
    ValidationError
)

# Logging
try:
    from ..logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES FOR EXPERIMENT CONFIGURATION
# =============================================================================

class ExperimentConfiguration(BaseModel):
    """
    Configuration for a systematic experiment.
    
    Supports:
    - Ablation studies (removing/adding transforms)
    - Parameter sweeps (varying transform parameters)
    - Comparative studies (comparing approaches)
    
    Designed for reproducibility - can be serialized to YAML/JSON.
    
    Pydantic V2 Migration (Phase 27):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - Uses @model_validator(mode='after') for __post_init__ validation logic
        - Custom to_dict() PRESERVED (calls TransformSpec.to_dict() for each transform)
        - NON-BREAKING: Same constructor API and attribute access
    
    Attributes:
        name: Experiment name
        description: Human-readable description
        base_transforms: Base transform sequence as TransformSpec objects
        ablations: List of ablation variants
        parameter_sweeps: List of parameter sweep variants
        paper_reference: Optional paper section reference
        hypothesis: Research hypothesis
        expected_outcome: Expected experimental outcome
        num_runs: Number of runs per variant (for statistical significance)
        random_seed: Random seed for reproducibility
        results: Dictionary to store experimental results
        
    Example:
        >>> config = ExperimentConfiguration(
        ...     name="transform_ablation",
        ...     description="Study importance of each transform",
        ...     base_transforms=[
        ...         TransformSpec(name="AddSelfLoops", kwargs={}),
        ...         TransformSpec(name="GCNNorm", kwargs={})
        ...     ],
        ...     ablations=[
        ...         {'name': 'no_self_loops', 'transforms': ['GCNNorm']},
        ...         {'name': 'no_norm', 'transforms': ['AddSelfLoops']}
        ...     ],
        ...     hypothesis="Self-loops critical for message passing",
        ...     num_runs=5
        ... )
    """
    name: str
    description: str
    base_transforms: List[TransformSpec]
    
    # Experiment variants
    ablations: List[Dict[str, Any]] = Field(default_factory=list)
    parameter_sweeps: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata for publication
    paper_reference: Optional[str] = None
    hypothesis: Optional[str] = None
    expected_outcome: Optional[str] = None
    
    # Execution settings
    num_runs: int = 3  # For statistical significance
    random_seed: int = 42
    
    # Results storage
    results: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_experiment_configuration(self) -> 'ExperimentConfiguration':
        """
        Validate experiment configuration on creation.
        
        Replaces __post_init__ validation from dataclass implementation.
        """
        if not self.name:
            raise ConfigurationError(
                message="Experiment name is required",
                config_key="experiment.name"
            )
        
        if not isinstance(self.base_transforms, list):
            raise ConfigurationError(
                message="base_transforms must be a list",
                config_key="experiment.base_transforms",
                actual_value=type(self.base_transforms).__name__
            )
        
        if self.num_runs < 1:
            raise ConfigurationError(
                message=f"num_runs must be >= 1, got {self.num_runs}",
                config_key="experiment.num_runs",
                actual_value=self.num_runs
            )
        
        # Validate that we have at least one variant
        total_variants = len(self.ablations) + len(self.parameter_sweeps)
        if total_variants == 0:
            logger.warning(f"Experiment '{self.name}' has no variants configured")
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML/JSON."""
        return {
            'name': self.name,
            'description': self.description,
            'base_transforms': [t.to_dict() for t in self.base_transforms],
            'ablations': self.ablations,
            'parameter_sweeps': self.parameter_sweeps,
            'paper_reference': self.paper_reference,
            'hypothesis': self.hypothesis,
            'expected_outcome': self.expected_outcome,
            'num_runs': self.num_runs,
            'random_seed': self.random_seed,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfiguration':
        """Deserialize from dictionary."""
        # Convert transform dicts to TransformSpec objects
        base_transforms = []
        for t in data.get('base_transforms', []):
            if isinstance(t, dict):
                base_transforms.append(TransformSpec(**t))
            elif isinstance(t, TransformSpec):
                base_transforms.append(t)
            else:
                # Try to convert string to TransformSpec
                base_transforms.append(TransformSpec(
                    name=str(t),
                    kwargs={},
                    enabled=True
                ))
        
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            base_transforms=base_transforms,
            ablations=data.get('ablations', []),
            parameter_sweeps=data.get('parameter_sweeps', []),
            paper_reference=data.get('paper_reference'),
            hypothesis=data.get('hypothesis'),
            expected_outcome=data.get('expected_outcome'),
            num_runs=data.get('num_runs', 3),
            random_seed=data.get('random_seed', 42),
            metadata=data.get('metadata', {})
        )
    
    def save_to_yaml(self, path: Path) -> None:
        """Save experiment configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved experiment configuration to {path}")
    
    @classmethod
    def load_from_yaml(cls, path: Path) -> 'ExperimentConfiguration':
        """Load experiment configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        logger.info(f"Loaded experiment configuration from {path}")
        return cls.from_dict(data)
    
    def get_total_runs(self) -> int:
        """Calculate total number of runs across all variants."""
        total_variants = len(self.ablations) + len(self.parameter_sweeps)
        return total_variants * self.num_runs
    
    def get_variant_names(self) -> List[str]:
        """Get list of all variant names."""
        names = []
        names.extend([v['name'] for v in self.ablations])
        names.extend([v['name'] for v in self.parameter_sweeps])
        return names


# =============================================================================
# FLUENT BUILDER: ABLATION STUDIES
# =============================================================================

class AblationStudyBuilder:
    """
    Fluent builder for ablation study configurations.
    
    Provides a clean API for creating ablation studies that test the importance
    of individual transforms by systematically removing or replacing them.
    
    Example:
        >>> study = (AblationStudyBuilder("transform_ablation")
        ...     .with_baseline(["AddSelfLoops", "GCNNorm", "RandomRotate"])
        ...     .remove_transform("AddSelfLoops")
        ...     .remove_transform("RandomRotate")
        ...     .add_variant("with_dropout", ["DropEdge"], position=1)
        ...     .with_metadata(
        ...         hypothesis="Self-loops critical, augmentation helps",
        ...         expected_outcome="baseline > no_augmentation > no_self_loops"
        ...     )
        ...     .build())
        >>> print(f"Created study with {len(study.ablations)} variants")
    """
    
    def __init__(self, study_name: str):
        """
        Initialize ablation study builder.
        
        Args:
            study_name: Name for the ablation study
        """
        self.study_name = study_name
        self.baseline_transforms: List[str] = []
        self.variants: Dict[str, List[str]] = {}
        self.metadata: Dict[str, Any] = {}
        logger.debug(f"Created AblationStudyBuilder for '{study_name}'")
    
    def with_baseline(self, transforms: List[str]) -> 'AblationStudyBuilder':
        """
        Set baseline transform sequence.
        
        Args:
            transforms: List of transform names for baseline
            
        Returns:
            Self for method chaining
        """
        self.baseline_transforms = transforms.copy()
        self.variants['baseline'] = self.baseline_transforms
        logger.debug(f"Baseline set: {self.baseline_transforms}")
        return self
    
    def remove_transform(
        self,
        transform_name: str,
        variant_name: Optional[str] = None
    ) -> 'AblationStudyBuilder':
        """
        Create variant without specific transform.
        
        Args:
            transform_name: Transform to remove
            variant_name: Optional custom name (default: "without_{transform}")
            
        Returns:
            Self for method chaining
        """
        if not variant_name:
            variant_name = f"without_{transform_name}"
        
        self.variants[variant_name] = [
            t for t in self.baseline_transforms if t != transform_name
        ]
        logger.debug(f"Created variant '{variant_name}': {self.variants[variant_name]}")
        return self
    
    def keep_only(
        self,
        transform_names: List[str],
        variant_name: str = "minimal"
    ) -> 'AblationStudyBuilder':
        """
        Create variant with only specified transforms.
        
        Args:
            transform_names: Transforms to keep
            variant_name: Name for minimal variant
            
        Returns:
            Self for method chaining
        """
        self.variants[variant_name] = [
            t for t in self.baseline_transforms if t in transform_names
        ]
        logger.debug(f"Created variant '{variant_name}': {self.variants[variant_name]}")
        return self
    
    def add_variant(
        self,
        variant_name: str,
        additional_transforms: List[str],
        position: int = -1
    ) -> 'AblationStudyBuilder':
        """
        Create variant with additional transforms.
        
        Args:
            variant_name: Name for the new variant
            additional_transforms: Transforms to add
            position: Where to insert (-1 = append at end)
            
        Returns:
            Self for method chaining
        """
        variant = self.baseline_transforms.copy()
        
        if position == -1:
            variant.extend(additional_transforms)
        else:
            for i, t in enumerate(additional_transforms):
                variant.insert(position + i, t)
        
        self.variants[variant_name] = variant
        logger.debug(f"Created variant '{variant_name}': {variant}")
        return self
    
    def replace_transform(
        self,
        old_transform: str,
        new_transform: str,
        variant_name: Optional[str] = None
    ) -> 'AblationStudyBuilder':
        """
        Create variant replacing one transform with another.
        
        Args:
            old_transform: Transform to replace
            new_transform: Replacement transform
            variant_name: Optional custom name
            
        Returns:
            Self for method chaining
        """
        if not variant_name:
            variant_name = f"replace_{old_transform}_with_{new_transform}"
        
        self.variants[variant_name] = [
            new_transform if t == old_transform else t
            for t in self.baseline_transforms
        ]
        logger.debug(f"Created variant '{variant_name}': {self.variants[variant_name]}")
        return self
    
    def with_metadata(
        self,
        hypothesis: str,
        expected_outcome: str,
        paper_section: Optional[str] = None
    ) -> 'AblationStudyBuilder':
        """
        Add experiment metadata for publication.
        
        Args:
            hypothesis: Research hypothesis
            expected_outcome: Expected experimental outcome
            paper_section: Optional paper section reference
            
        Returns:
            Self for method chaining
        """
        self.metadata.update({
            'hypothesis': hypothesis,
            'expected_outcome': expected_outcome,
            'paper_section': paper_section
        })
        return self
    
    def build(self) -> ExperimentConfiguration:
        """
        Build the experiment configuration.
        
        Returns:
            Complete ExperimentConfiguration ready to run
            
        Raises:
            ConfigurationError: If baseline not set
        """
        if not self.baseline_transforms:
            raise ConfigurationError(
                message="Baseline transforms not set",
                config_key="ablation_study.baseline"
            )
        
        # Convert to ablation format
        ablations = [
            {
                'name': name,
                'transforms': transforms
            }
            for name, transforms in self.variants.items()
        ]
        
        base_specs = [
            TransformSpec(name=t, kwargs={}, enabled=True)
            for t in self.baseline_transforms
        ]
        
        # Separate direct ExperimentConfiguration fields from extra metadata
        direct_fields = {}
        extra_metadata = {}
        
        for key, value in self.metadata.items():
            if key in ('hypothesis', 'expected_outcome', 'paper_reference'):
                direct_fields[key] = value
            else:
                # Store in metadata dict for custom fields like 'paper_section'
                extra_metadata[key] = value
        
        config = ExperimentConfiguration(
            name=self.study_name,
            description=f"Ablation study: {self.study_name}",
            base_transforms=base_specs,
            ablations=ablations,
            metadata=extra_metadata,
            **direct_fields
        )
        
        logger.info(f"Built ablation study '{self.study_name}' with {len(ablations)} variants")
        return config


# =============================================================================
# FLUENT BUILDER: PARAMETER SWEEPS
# =============================================================================

class ParameterSweepBuilder:
    """
    Builder for parameter sweep experiments.
    
    Generates all combinations of parameter values to systematically explore
    the parameter space of a transform.
    
    Example:
        >>> sweep = (ParameterSweepBuilder("dropout_optimization")
        ...     .for_transform("DropEdge")
        ...     .sweep_parameter("p", [0.0, 0.1, 0.2, 0.3, 0.5])
        ...     .with_baseline_transforms(["AddSelfLoops", "GCNNorm"])
        ...     .with_metadata(
        ...         hypothesis="Moderate dropout improves generalization",
        ...         expected_outcome="Optimal p around 0.15"
        ...     )
        ...     .build())
        >>> print(f"Generated {len(sweep.parameter_sweeps)} parameter combinations")
    """
    
    def __init__(self, sweep_name: str):
        """
        Initialize parameter sweep builder.
        
        Args:
            sweep_name: Name for the parameter sweep
        """
        self.sweep_name = sweep_name
        self.target_transform: Optional[str] = None
        self.parameter_sweeps: Dict[str, List[Any]] = {}
        self.baseline_transforms: List[str] = []
        self.metadata: Dict[str, Any] = {}
        logger.debug(f"Created ParameterSweepBuilder for '{sweep_name}'")
    
    def for_transform(self, transform_name: str) -> 'ParameterSweepBuilder':
        """
        Specify transform to sweep parameters for.
        
        Args:
            transform_name: Name of target transform
            
        Returns:
            Self for method chaining
        """
        self.target_transform = transform_name
        logger.debug(f"Target transform: {transform_name}")
        return self
    
    def sweep_parameter(
        self,
        param_name: str,
        values: List[Any]
    ) -> 'ParameterSweepBuilder':
        """
        Define parameter values to sweep.
        
        Args:
            param_name: Name of parameter
            values: List of values to test
            
        Returns:
            Self for method chaining
        """
        self.parameter_sweeps[param_name] = values
        logger.debug(f"Sweeping {param_name}: {values}")
        return self
    
    def with_baseline_transforms(
        self,
        transforms: List[str]
    ) -> 'ParameterSweepBuilder':
        """
        Set baseline transforms (target transform added automatically).
        
        Args:
            transforms: Baseline transform sequence
            
        Returns:
            Self for method chaining
        """
        self.baseline_transforms = transforms
        return self
    
    def with_metadata(
        self,
        hypothesis: str,
        expected_outcome: str
    ) -> 'ParameterSweepBuilder':
        """
        Add experiment metadata.
        
        Args:
            hypothesis: Research hypothesis
            expected_outcome: Expected outcome
            
        Returns:
            Self for method chaining
        """
        self.metadata.update({
            'hypothesis': hypothesis,
            'expected_outcome': expected_outcome
        })
        return self
    
    def build(self) -> ExperimentConfiguration:
        """
        Build experiment configuration with all parameter combinations.
        
        Returns:
            Complete ExperimentConfiguration
            
        Raises:
            ConfigurationError: If target transform or parameters not specified
        """
        if not self.target_transform:
            raise ConfigurationError(
                message="Target transform not specified",
                config_key="parameter_sweep.target_transform"
            )
        
        if not self.parameter_sweeps:
            raise ConfigurationError(
                message="No parameters to sweep",
                config_key="parameter_sweep.parameters"
            )
        
        # Generate all parameter combinations
        param_combinations = self._generate_combinations(self.parameter_sweeps)
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        # Create variant for each combination
        variants = []
        for i, params in enumerate(param_combinations):
            variant_name = f"sweep_{i}_" + "_".join(
                f"{k}={v}" for k, v in params.items()
            )
            
            # Build transform sequence
            transforms = self.baseline_transforms.copy()
            transforms.append({
                'name': self.target_transform,
                **params
            })
            
            variants.append({
                'name': variant_name,
                'transforms': transforms,
                'parameters': params
            })
        
        base_specs = [
            TransformSpec(name=t, kwargs={}, enabled=True)
            for t in self.baseline_transforms
        ]
        
        # Separate direct ExperimentConfiguration fields from extra metadata
        direct_fields = {}
        extra_metadata = {}
        
        for key, value in self.metadata.items():
            if key in ('hypothesis', 'expected_outcome', 'paper_reference'):
                direct_fields[key] = value
            else:
                extra_metadata[key] = value
        
        config = ExperimentConfiguration(
            name=self.sweep_name,
            description=f"Parameter sweep: {self.target_transform}",
            base_transforms=base_specs,
            parameter_sweeps=variants,
            metadata=extra_metadata,
            **direct_fields
        )
        
        logger.info(f"Built parameter sweep '{self.sweep_name}' with {len(variants)} variants")
        return config
    
    @staticmethod
    def _generate_combinations(param_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations using Cartesian product.
        
        Args:
            param_dict: Dictionary of parameter_name -> list of values
            
        Returns:
            List of parameter dictionaries
        """
        keys = list(param_dict.keys())
        value_lists = [param_dict[k] for k in keys]
        
        combinations = []
        for values in itertools.product(*value_lists):
            combinations.append(dict(zip(keys, values)))
        
        return combinations


# =============================================================================
# FLUENT BUILDER: COMPARATIVE STUDIES
# =============================================================================

class ComparativeStudyBuilder:
    """
    Builder for comparative studies.
    
    Compares different approaches head-to-head to determine the best method
    for a particular task.
    
    Example:
        >>> study = (ComparativeStudyBuilder("normalization_comparison")
        ...     .add_approach("gcn_norm", ["AddSelfLoops", "GCNNorm"])
        ...     .add_approach("pair_norm", ["AddSelfLoops", "PairNorm"])
        ...     .add_approach("no_norm", ["AddSelfLoops"])
        ...     .with_evaluation_metric("validation_mae")
        ...     .with_metadata(
        ...         research_question="Which normalization for molecular graphs?",
        ...         expected_best="gcn_norm"
        ...     )
        ...     .build())
    """
    
    def __init__(self, study_name: str):
        """
        Initialize comparative study builder.
        
        Args:
            study_name: Name for the comparative study
        """
        self.study_name = study_name
        self.approaches: Dict[str, List[Union[str, Dict]]] = {}
        self.evaluation_metrics: List[str] = []
        self.metadata: Dict[str, Any] = {}
        logger.debug(f"Created ComparativeStudyBuilder for '{study_name}'")
    
    def add_approach(
        self,
        approach_name: str,
        transforms: List[Union[str, Dict]]
    ) -> 'ComparativeStudyBuilder':
        """
        Add an approach to compare.
        
        Args:
            approach_name: Name for this approach
            transforms: Transform sequence for this approach
            
        Returns:
            Self for method chaining
        """
        self.approaches[approach_name] = transforms
        logger.debug(f"Added approach '{approach_name}': {transforms}")
        return self
    
    def with_evaluation_metric(self, metric_name: str) -> 'ComparativeStudyBuilder':
        """
        Specify evaluation metric for comparison.
        
        Args:
            metric_name: Metric to use for comparison (e.g., "validation_mae")
            
        Returns:
            Self for method chaining
        """
        self.evaluation_metrics.append(metric_name)
        return self
    
    def with_metadata(
        self,
        research_question: str,
        expected_best: str
    ) -> 'ComparativeStudyBuilder':
        """
        Add study metadata.
        
        Args:
            research_question: The research question being answered
            expected_best: Expected best-performing approach
            
        Returns:
            Self for method chaining
        """
        self.metadata.update({
            'research_question': research_question,
            'expected_best': expected_best
        })
        return self
    
    def build(self) -> ExperimentConfiguration:
        """
        Build comparative study configuration.
        
        Returns:
            Complete ExperimentConfiguration
            
        Raises:
            ConfigurationError: If fewer than 2 approaches specified
        """
        if len(self.approaches) < 2:
            raise ConfigurationError(
                message="Comparative study needs at least 2 approaches",
                config_key="comparative_study.approaches",
                actual_value=len(self.approaches)
            )
        
        variants = [
            {
                'name': name,
                'transforms': transforms
            }
            for name, transforms in self.approaches.items()
        ]
        
        # Separate direct ExperimentConfiguration fields from extra metadata
        direct_fields = {}
        extra_metadata = {}
        
        for key, value in self.metadata.items():
            if key in ('hypothesis', 'expected_outcome', 'paper_reference'):
                direct_fields[key] = value
            else:
                # Store in metadata dict for custom fields like 'research_question', 'expected_best'
                extra_metadata[key] = value
        
        config = ExperimentConfiguration(
            name=self.study_name,
            description=f"Comparative study: {self.study_name}",
            base_transforms=[],  # No base for comparative studies
            ablations=variants,
            metadata=extra_metadata,
            **direct_fields
        )
        
        config.results['evaluation_metrics'] = self.evaluation_metrics
        
        logger.info(
            f"Built comparative study '{self.study_name}' "
            f"with {len(variants)} approaches"
        )
        return config


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Execute systematic experiments and track results.
    
    Handles:
    - Experiment execution across multiple runs
    - Result collection and aggregation
    - Statistical analysis
    - Report generation (JSON, CSV, Markdown)
    
    Example:
        >>> runner = ExperimentRunner(config, output_dir="./experiments")
        >>> results = runner.run_experiment(
        ...     dataset_loader=load_dataset,
        ...     model_trainer=train_model,
        ...     evaluator=evaluate_model,
        ...     num_runs=5
        ... )
    """
    
    def __init__(
        self,
        config: ExperimentConfiguration,
        output_dir: Union[str, Path],
        seed: int = 42
    ):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            output_dir: Directory for saving results
            seed: Random seed for reproducibility
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.results: Dict[str, List[Dict]] = defaultdict(list)
        
        logger.info(f"Initialized ExperimentRunner for '{config.name}'")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_experiment(
        self,
        dataset_loader: Callable[[List, int], Any],
        model_trainer: Callable[[Any], Tuple[Any, Dict]],
        evaluator: Callable[[Any, Any], Dict],
        num_runs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete experiment with multiple runs.
        
        Args:
            dataset_loader: Function(transforms, seed) -> dataset
            model_trainer: Function(dataset) -> (model, train_metrics)
            evaluator: Function(model, dataset) -> eval_metrics
            num_runs: Number of runs per variant (default from config)
            
        Returns:
            Comprehensive results dictionary with statistics
        """
        num_runs = num_runs or self.config.num_runs
        
        logger.info("="*60)
        logger.info(f"Starting Experiment: {self.config.name}")
        logger.info(f"Variants: {len(self.config.ablations + self.config.parameter_sweeps)}")
        logger.info(f"Runs per variant: {num_runs}")
        logger.info("="*60)
        
        all_variants = self.config.ablations + self.config.parameter_sweeps
        
        for variant_idx, variant in enumerate(all_variants, 1):
            variant_name = variant['name']
            
            logger.info(f"\n[{variant_idx}/{len(all_variants)}] Running variant: {variant_name}")
            
            variant_results = []
            
            for run_idx in range(num_runs):
                logger.info(f"  Run {run_idx + 1}/{num_runs}...")
                
                try:
                    # Load dataset with specific transforms
                    dataset = dataset_loader(
                        transforms=variant['transforms'],
                        seed=self.seed + run_idx
                    )
                    
                    # Train model
                    model, train_metrics = model_trainer(dataset)
                    
                    # Evaluate
                    eval_metrics = evaluator(model, dataset)
                    
                    # Store results
                    run_result = {
                        'run': run_idx,
                        'variant': variant_name,
                        'train_metrics': train_metrics,
                        'eval_metrics': eval_metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    variant_results.append(run_result)
                    
                    logger.info(
                        f"    ✓ Eval MAE: {eval_metrics.get('mae', 'N/A'):.4f}"
                    )
                
                except Exception as e:
                    logger.error(f"    ✗ Run {run_idx} failed: {e}")
                    variant_results.append({
                        'run': run_idx,
                        'variant': variant_name,
                        'error': str(e)
                    })
            
            # Store results for this variant
            self.results[variant_name] = variant_results
        
        # Statistical analysis
        logger.info("\nAnalyzing results...")
        summary = self._analyze_results()
        
        # Save results
        logger.info("Saving results...")
        self._save_results(summary)
        
        logger.info("="*60)
        logger.info("Experiment complete!")
        if summary.get('best_variant'):
            logger.info(f"Best variant: {summary['best_variant']['name']}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)
        
        return summary
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Perform statistical analysis on results."""
        summary = {
            'experiment': self.config.name,
            'timestamp': datetime.now().isoformat(),
            'variants': {},
            'hypothesis': self.config.hypothesis,
            'expected_outcome': self.config.expected_outcome
        }
        
        for variant_name, runs in self.results.items():
            # Filter successful runs
            successful_runs = [r for r in runs if 'error' not in r]
            
            if not successful_runs:
                summary['variants'][variant_name] = {
                    'num_runs': len(runs),
                    'status': 'all_failed'
                }
                continue
            
            # Extract evaluation metrics (assuming 'mae' as primary)
            eval_scores = [
                run['eval_metrics'].get('mae', float('inf'))
                for run in successful_runs
            ]
            
            variant_summary = {
                'num_runs': len(successful_runs),
                'failed_runs': len(runs) - len(successful_runs),
                'mean': float(np.mean(eval_scores)),
                'std': float(np.std(eval_scores)),
                'min': float(np.min(eval_scores)),
                'max': float(np.max(eval_scores)),
                'median': float(np.median(eval_scores)),
                'status': 'success'
            }
            
            summary['variants'][variant_name] = variant_summary
        
        # Find best variant (lowest MAE)
        valid_variants = {
            name: stats for name, stats in summary['variants'].items()
            if stats.get('status') == 'success'
        }
        
        if valid_variants:
            best_variant = min(
                valid_variants.items(),
                key=lambda x: x[1]['mean']
            )
            summary['best_variant'] = {
                'name': best_variant[0],
                'score': best_variant[1]['mean']
            }
        else:
            summary['best_variant'] = None
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any]) -> None:
        """Save results in multiple formats."""
        # Save JSON summary
        json_path = self.output_dir / f"{self.config.name}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  ✓ Saved JSON: {json_path}")
        
        # Save detailed results as JSON
        detailed_path = self.output_dir / f"{self.config.name}_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump(dict(self.results), f, indent=2)
        logger.info(f"  ✓ Saved detailed JSON: {detailed_path}")
        
        # Save CSV
        csv_path = self.output_dir / f"{self.config.name}_results.csv"
        self._save_results_csv(csv_path)
        logger.info(f"  ✓ Saved CSV: {csv_path}")
        
        # Generate markdown report
        md_path = self.output_dir / f"{self.config.name}_report.md"
        self._generate_markdown_report(md_path, summary)
        logger.info(f"  ✓ Saved report: {md_path}")
    
    def _save_results_csv(self, path: Path) -> None:
        """Save detailed results as CSV using pandas."""
        rows = []
        
        for variant_name, runs in self.results.items():
            for run in runs:
                if 'error' not in run:
                    row = {
                        'variant': variant_name,
                        'run': run['run'],
                        **{f"train_{k}": v for k, v in run['train_metrics'].items()},
                        **{f"eval_{k}": v for k, v in run['eval_metrics'].items()}
                    }
                    rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
        else:
            logger.warning("No successful runs to save to CSV")
    
    def _generate_markdown_report(self, path: Path, summary: Dict[str, Any]) -> None:
        """Generate human-readable markdown report."""
        with open(path, 'w') as f:
            f.write(f"# Experiment Report: {self.config.name}\n\n")
            f.write(f"**Generated**: {summary['timestamp']}\n\n")
            
            if self.config.description:
                f.write(f"## Description\n\n{self.config.description}\n\n")
            
            if self.config.hypothesis:
                f.write(f"## Hypothesis\n\n{self.config.hypothesis}\n\n")
            
            if self.config.expected_outcome:
                f.write(f"## Expected Outcome\n\n{self.config.expected_outcome}\n\n")
            
            f.write("## Results Summary\n\n")
            
            if summary.get('best_variant'):
                f.write(f"**Best Variant**: {summary['best_variant']['name']}\n")
                f.write(f"**Score**: {summary['best_variant']['score']:.4f}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| Variant | Runs | Mean | Std | Min | Max | Median |\n")
            f.write("|---------|------|------|-----|-----|-----|--------|\n")
            
            for variant_name, stats in summary['variants'].items():
                if stats.get('status') == 'success':
                    f.write(
                        f"| {variant_name} | {stats['num_runs']} | "
                        f"{stats['mean']:.4f} | {stats['std']:.4f} | "
                        f"{stats['min']:.4f} | {stats['max']:.4f} | "
                        f"{stats['median']:.4f} |\n"
                    )
            
            f.write("\n## Conclusion\n\n")
            f.write("*Add your interpretation and conclusions here.*\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_ablation_study(
    study_name: str,
    baseline_transforms: List[str],
    transforms_to_ablate: List[str],
    **metadata
) -> ExperimentConfiguration:
    """
    Quick ablation study creation.
    
    Args:
        study_name: Name of the study
        baseline_transforms: Full transform sequence
        transforms_to_ablate: Transforms to remove in ablations
        **metadata: Additional metadata (hypothesis, expected_outcome, etc.)
    
    Returns:
        ExperimentConfiguration ready to run
        
    Example:
        >>> config = create_ablation_study(
        ...     "importance_study",
        ...     ["AddSelfLoops", "GCNNorm", "RandomRotate"],
        ...     ["GCNNorm", "RandomRotate"],
        ...     hypothesis="Normalization and augmentation improve results"
        ... )
    """
    builder = AblationStudyBuilder(study_name).with_baseline(baseline_transforms)
    
    for transform in transforms_to_ablate:
        builder.remove_transform(transform)
    
    if metadata:
        builder.metadata.update(metadata)
    
    return builder.build()


def create_parameter_sweep(
    sweep_name: str,
    transform_name: str,
    parameter_ranges: Dict[str, List],
    baseline_transforms: List[str],
    **metadata
) -> ExperimentConfiguration:
    """
    Quick parameter sweep creation.
    
    Args:
        sweep_name: Name of the sweep
        transform_name: Transform to sweep parameters for
        parameter_ranges: Dict of parameter_name -> list of values
        baseline_transforms: Base transform sequence
        **metadata: Additional metadata
    
    Returns:
        ExperimentConfiguration ready to run
        
    Example:
        >>> config = create_parameter_sweep(
        ...     "dropout_optimization",
        ...     "DropEdge",
        ...     {"p": [0.1, 0.2, 0.3, 0.5]},
        ...     ["AddSelfLoops", "GCNNorm"],
        ...     hypothesis="Moderate dropout improves generalization"
        ... )
    """
    builder = (ParameterSweepBuilder(sweep_name)
               .for_transform(transform_name)
               .with_baseline_transforms(baseline_transforms))
    
    for param_name, values in parameter_ranges.items():
        builder.sweep_parameter(param_name, values)
    
    if metadata:
        builder.metadata.update(metadata)
    
    return builder.build()


def create_comparative_study(
    study_name: str,
    approaches: Dict[str, List],
    evaluation_metrics: List[str],
    **metadata
) -> ExperimentConfiguration:
    """
    Quick comparative study creation.
    
    Args:
        study_name: Name of the study
        approaches: Dict of approach_name -> transform sequence
        evaluation_metrics: Metrics to compare
        **metadata: Additional metadata
    
    Returns:
        ExperimentConfiguration ready to run
        
    Example:
        >>> config = create_comparative_study(
        ...     "norm_comparison",
        ...     {
        ...         "gcn": ["AddSelfLoops", "GCNNorm"],
        ...         "no_norm": ["AddSelfLoops"]
        ...     },
        ...     ["validation_mae", "test_mae"],
        ...     research_question="Best normalization method?"
        ... )
    """
    builder = ComparativeStudyBuilder(study_name)
    
    for name, transforms in approaches.items():
        builder.add_approach(name, transforms)
    
    for metric in evaluation_metrics:
        builder.with_evaluation_metric(metric)
    
    if metadata:
        builder.metadata.update(metadata)
    
    return builder.build()


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_experiments_from_config(
    config_path: Optional[Path] = None
) -> Dict[str, ExperimentConfiguration]:
    """
    Load experiments from YAML configuration file.
    
    Args:
        config_path: Path to experiments config (default: research_experiments.yaml)
        
    Returns:
        Dictionary of experiment_name -> ExperimentConfiguration
        
    Example:
        >>> experiments = load_experiments_from_config()
        >>> ablation_study = experiments['transform_ablation']
        >>> runner = ExperimentRunner(ablation_study, "./results")
    """
    if config_path is None:
        config_path = Path("research_experiments.yaml")
    
    if not config_path.exists():
        # Try loading from main config.yaml
        main_config = load_config()
        if 'experiments' in main_config:
            experiments_dict = main_config['experiments']
        else:
            logger.warning(f"No experiments found in {config_path} or config.yaml")
            return {}
    else:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        experiments_dict = data.get('experiments', {})
    
    # Convert to ExperimentConfiguration objects
    experiments = {}
    for name, config_data in experiments_dict.items():
        try:
            experiments[name] = ExperimentConfiguration.from_dict(config_data)
            logger.debug(f"Loaded experiment '{name}'")
        except Exception as e:
            logger.error(f"Failed to load experiment '{name}': {e}")
    
    logger.info(f"Loaded {len(experiments)} experiments from configuration")
    return experiments


def get_experiment(
    experiment_name: str,
    config_path: Optional[Path] = None
) -> ExperimentConfiguration:
    """
    Get specific experiment configuration by name.
    
    Args:
        experiment_name: Name of experiment to load
        config_path: Optional path to config file
        
    Returns:
        ExperimentConfiguration for the named experiment
        
    Raises:
        ConfigurationError: If experiment not found
        
    Example:
        >>> exp = get_experiment("transform_ablation")
        >>> runner = ExperimentRunner(exp, "./results")
    """
    experiments = load_experiments_from_config(config_path)
    
    if experiment_name not in experiments:
        available = list(experiments.keys())
        raise ConfigurationError(
            message=f"Experiment '{experiment_name}' not found",
            config_key="experiments",
            actual_value=experiment_name,
            details=f"Available experiments: {available}"
        )
    
    return experiments[experiment_name]


def list_available_experiments(
    config_path: Optional[Path] = None
) -> List[str]:
    """
    List all available experiment names.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        List of experiment names
        
    Example:
        >>> experiments = list_available_experiments()
        >>> print(f"Available: {experiments}")
    """
    experiments = load_experiments_from_config(config_path)
    return list(experiments.keys())


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'ExperimentConfiguration',
    'AblationStudyBuilder',
    'ParameterSweepBuilder',
    'ComparativeStudyBuilder',
    'ExperimentRunner',
    
    # Convenience functions
    'create_ablation_study',
    'create_parameter_sweep',
    'create_comparative_study',
    
    # Configuration loaders
    'load_experiments_from_config',
    'get_experiment',
    'list_available_experiments',
]
