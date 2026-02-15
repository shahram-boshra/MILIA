"""
Configuration Parser for Custom Architectures

Parses YAML/JSON configuration for custom architectures and ensembles.
Integrates with existing config system, templates, and validation.

Features:
- YAML/JSON configuration parsing
- Custom architecture specification
- Ensemble configuration
- Template integration
- Comprehensive validation
- Error reporting and suggestions
- Integration with ArchitectureBuilder, ModelComposer, and ArchitectureValidator

Author: milia Team
Version: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .architecture_builder import (
    ArchitectureBuilder,
)
from .model_composer import ModelComposer
from .templates import ArchitectureTemplates
from .validation import ArchitectureValidator

# Import exceptions
try:
    from milia_pipeline.exceptions import ConfigurationError
except ImportError:

    class ConfigurationError(Exception):
        """Configuration error."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG PARSER
# =============================================================================


class ArchitectureConfigParser:
    """
    Parser for architecture configurations.

    Parses YAML/JSON/dict configurations into ArchitectureBuilder or ModelComposer instances.
    Supports:
    - Custom architecture specifications
    - Ensemble compositions
    - Template-based architectures
    - Configuration validation
    - Error reporting with suggestions

    Configuration Format (Custom Architecture):
    ```yaml
    name: "CustomGNN"
    task_type: "graph_regression"
    in_channels: 16
    out_channels: 1
    layers:
      - type: "GCNConv"
        params:
          out_channels: 64
      - type: "ReLU"
      - type: "Dropout"
        params:
          p: 0.5
      - type: "GCNConv"
        params:
          out_channels: 32
      - type: "global_mean_pool"
      - type: "Linear"
        params:
          out_features: 1
    residual_connections:
      - start: 0
        end: 3
        type: "add"
    ```

    Configuration Format (Ensemble):
    ```yaml
    name: "CustomEnsemble"
    task_type: "graph_classification"
    composition:
      strategy: "parallel"
      fusion: "weighted"
    models:
      - name: "GCN"
        weight: 0.5
      - name: "GAT"
        weight: 0.5
    ```

    Configuration Format (Template-based):
    ```yaml
    template: "gcn_basic"
    task_type: "graph_regression"
    params:
      hidden_channels: 128
      num_layers: 4
      dropout: 0.3
    ```

    Usage:
        >>> parser = ArchitectureConfigParser()
        >>> # From dict
        >>> builder = parser.parse_custom_architecture(config_dict)
        >>> # From YAML file
        >>> builder = parser.parse_custom_architecture("config.yaml")
        >>> # From YAML string
        >>> yaml_str = "name: Test\\nlayers: [...]"
        >>> builder = parser.parse_custom_architecture(yaml_str)
    """

    def __init__(
        self,
        validator: ArchitectureValidator | None = None,
        templates: ArchitectureTemplates | None = None,
        strict_validation: bool = True,
    ):
        """
        Initialize parser.

        Args:
            validator: Custom validator (uses default if None)
            templates: Template provider (uses default if None)
            strict_validation: If True, raise errors on validation failures
        """
        self.validator = validator or ArchitectureValidator()
        self.templates = templates or ArchitectureTemplates()
        self.strict_validation = strict_validation

    # =========================================================================
    # CUSTOM ARCHITECTURE PARSING
    # =========================================================================

    def parse_custom_architecture(
        self,
        config: dict[str, Any] | Path | str,
        task_type: str | None = None,
        validate: bool = True,
    ) -> ArchitectureBuilder:
        """
        Parse custom architecture configuration.

        Supports:
        - Direct layer specification
        - Template-based with overrides
        - Mixed template + custom layers

        Args:
            config: Configuration dict, file path, or YAML/JSON string
            task_type: Override task type
            validate: Whether to validate architecture

        Returns:
            ArchitectureBuilder instance

        Raises:
            ConfigurationError: If configuration is invalid

        Example:
            >>> config = {
            ...     'name': 'MyGNN',
            ...     'task_type': 'graph_regression',
            ...     'in_channels': 16,
            ...     'out_channels': 1,
            ...     'layers': [
            ...         {'type': 'GCNConv', 'params': {'out_channels': 64}},
            ...         {'type': 'ReLU'},
            ...         {'type': 'global_mean_pool'},
            ...         {'type': 'Linear', 'params': {'out_features': 1}}
            ...     ]
            ... }
            >>> builder = parser.parse_custom_architecture(config)
            >>> model = builder.build()
        """
        # Load config if path or string
        if isinstance(config, (Path, str)):
            config = self._load_config(config)

        # Check if template-based
        if "template" in config:
            return self._parse_template_based(config, task_type, validate)

        # Validate configuration structure
        if validate:
            validation_result = self.validate_config(config, "custom_architecture")
            if not validation_result["valid"]:
                if self.strict_validation:
                    error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(
                        validation_result["errors"]
                    )
                    raise ConfigurationError(error_msg)
                else:
                    logger.warning(f"Configuration has warnings: {validation_result['warnings']}")

        # Validate required fields
        if "layers" not in config:
            raise ConfigurationError("Configuration missing 'layers' field")

        # Extract parameters
        name = config.get("name", "CustomArchitecture")
        task = task_type or config.get("task_type", "graph_regression")
        in_channels = config.get("in_channels", 16)
        out_channels = config.get("out_channels")
        if out_channels is None:
            # Default to 1, but ModelFactory will override with inferred value if sample_data provided
            logger.debug(
                f"'out_channels' not specified in config for '{name}'. "
                f"Defaulting to 1. ModelFactory may override with inferred value."
            )
            out_channels = 1

        # Create builder
        builder = ArchitectureBuilder(
            task_type=task, in_channels=in_channels, out_channels=out_channels, name=name
        )

        # Parse and add layers
        layers_config = config["layers"]
        for i, layer_spec in enumerate(layers_config):
            try:
                layer_type = layer_spec.get("type")
                if not layer_type:
                    raise ConfigurationError(
                        f"Layer {i} specification missing 'type': {layer_spec}"
                    )

                params = layer_spec.get("params", {})

                # Handle special parameters
                # out_channels can be at top level or in params
                if "out_channels" in layer_spec and "out_channels" not in params:
                    params["out_channels"] = layer_spec["out_channels"]

                builder.add_layer(layer_type, **params)

            except Exception as e:
                raise ConfigurationError(
                    f"Error parsing layer {i} ({layer_spec.get('type', 'unknown')}): {e}"
                )

        # Parse residual connections if present
        if "residual_connections" in config:
            for rc_spec in config["residual_connections"]:
                try:
                    start = rc_spec.get("start")
                    end = rc_spec.get("end")
                    connection_type = rc_spec.get("type", "add")

                    if start is None or end is None:
                        raise ConfigurationError(
                            f"Residual connection missing 'start' or 'end': {rc_spec}"
                        )

                    builder.add_residual_connection(start, end, connection_type)

                except Exception as e:
                    raise ConfigurationError(f"Error parsing residual connection {rc_spec}: {e}")

        logger.info(
            f"Parsed custom architecture '{name}' with {len(builder)} layers for task '{task}'"
        )

        # Validate architecture if requested
        if validate:
            validation_result = self.validator.validate(
                builder.layers, task, in_channels, out_channels
            )

            if not validation_result["valid"]:
                error_msg = "Architecture validation failed:\n  - " + "\n  - ".join(
                    validation_result["errors"]
                )
                if validation_result["suggestions"]:
                    error_msg += "\n\nSuggestions:\n  - " + "\n  - ".join(
                        validation_result["suggestions"]
                    )

                if self.strict_validation:
                    raise ConfigurationError(error_msg)
                else:
                    logger.warning(error_msg)

            if validation_result.get("warnings"):
                logger.warning(
                    "Architecture warnings:\n  - " + "\n  - ".join(validation_result["warnings"])
                )

        return builder

    def _parse_template_based(
        self, config: dict[str, Any], task_type: str | None, validate: bool
    ) -> ArchitectureBuilder:
        """
        Parse template-based architecture configuration.

        Args:
            config: Configuration with 'template' field
            task_type: Override task type
            validate: Whether to validate

        Returns:
            ArchitectureBuilder instance

        Raises:
            ConfigurationError: If template not found or invalid
        """
        template_name = config["template"]
        task = task_type or config.get("task_type", "graph_regression")

        # Get template parameters
        template_params = config.get("params", {})

        # Override with top-level parameters if present
        if "in_channels" in config:
            template_params["in_channels"] = config["in_channels"]
        if "out_channels" in config:
            template_params["out_channels"] = config["out_channels"]
        if "name" in config:
            template_params["name"] = config["name"]

        # Get template builder
        try:
            builder = self.templates.get_template(template_name, task_type=task, **template_params)
        except Exception as e:
            available = self.templates.list_templates()
            raise ConfigurationError(
                f"Failed to load template '{template_name}': {e}\nAvailable templates: {available}"
            )

        # Apply additional layers if specified
        if "additional_layers" in config:
            for layer_spec in config["additional_layers"]:
                layer_type = layer_spec.get("type")
                params = layer_spec.get("params", {})
                builder.add_layer(layer_type, **params)

        # Apply modifications if specified
        if "modifications" in config:
            mods = config["modifications"]

            # Insert layers
            if "insert" in mods:
                for insert_spec in mods["insert"]:
                    position = insert_spec.get("position")
                    layer_type = insert_spec.get("type")
                    params = insert_spec.get("params", {})
                    builder.insert_layer(position, layer_type, **params)

            # Remove layers
            if "remove" in mods:
                # Remove in reverse order to maintain indices
                for position in sorted(mods["remove"], reverse=True):
                    builder.remove_layer(position)

            # Replace layers
            if "replace" in mods:
                for replace_spec in mods["replace"]:
                    position = replace_spec.get("position")
                    layer_type = replace_spec.get("type")
                    params = replace_spec.get("params", {})
                    builder.replace_layer(position, layer_type, **params)

        logger.info(
            f"Parsed template-based architecture using '{template_name}' with {len(builder)} layers"
        )

        return builder

    # =========================================================================
    # ENSEMBLE PARSING
    # =========================================================================

    def parse_ensemble(
        self,
        config: dict[str, Any] | Path | str,
        task_type: str | None = None,
        validate: bool = True,
    ) -> ModelComposer:
        """
        Parse ensemble configuration.

        Note: This returns a ModelComposer with configuration but no models.
        Models must be added separately using add_model() or provided in config.

        Args:
            config: Configuration dict, file path, or YAML/JSON string
            task_type: Override task type
            validate: Whether to validate configuration

        Returns:
            ModelComposer instance (may or may not have models)

        Raises:
            ConfigurationError: If configuration is invalid

        Example:
            >>> config = {
            ...     'name': 'MyEnsemble',
            ...     'task_type': 'graph_classification',
            ...     'composition': {
            ...         'strategy': 'parallel',
            ...         'fusion': 'weighted'
            ...     }
            ... }
            >>> composer = parser.parse_ensemble(config)
            >>> composer.add_model(model1, weight=0.6)
            >>> composer.add_model(model2, weight=0.4)
            >>> ensemble = composer.build()
        """
        # Load config if path or string
        if isinstance(config, (Path, str)):
            config = self._load_config(config)

        # Validate configuration structure
        if validate:
            validation_result = self.validate_config(config, "ensemble")
            if not validation_result["valid"]:
                if self.strict_validation:
                    error_msg = "Ensemble configuration validation failed:\n  - " + "\n  - ".join(
                        validation_result["errors"]
                    )
                    raise ConfigurationError(error_msg)
                else:
                    logger.warning(
                        f"Ensemble configuration has warnings: {validation_result['warnings']}"
                    )

        # Extract parameters
        name = config.get("name", "Ensemble")
        task = task_type or config.get("task_type", "graph_regression")

        # Create composer
        composer = ModelComposer(task_type=task, name=name)

        # Set strategy and fusion
        # Support multiple config formats for flexibility:
        # Format 1 (config.yaml style): strategy at top level, fusion.method nested
        #   strategy: "parallel"
        #   fusion:
        #     method: "mean"
        # Format 2 (legacy/composition style): nested under 'composition'
        #   composition:
        #     strategy: "parallel"
        #     fusion: "mean"

        # Try top-level 'strategy' first (config.yaml format), then 'composition.strategy' (legacy)
        if "strategy" in config:
            strategy = config["strategy"]
        elif "composition" in config:
            strategy = config["composition"].get("strategy", "parallel")
        else:
            strategy = "parallel"

        # Try 'fusion.method' first (config.yaml format), then top-level 'fusion' (if string),
        # then 'composition.fusion' (legacy)
        fusion_config = config.get("fusion", {})
        if isinstance(fusion_config, dict):
            # config.yaml format: fusion.method
            fusion = fusion_config.get("method", "mean")
        elif isinstance(fusion_config, str):
            # Direct string format: fusion: "mean"
            fusion = fusion_config
        elif "composition" in config:
            # Legacy format: composition.fusion
            fusion = config["composition"].get("fusion", "mean")
        else:
            fusion = "mean"

        composer.set_strategy(strategy)
        composer.set_fusion(fusion)

        # Note: Model instances cannot be serialized in YAML/JSON
        # This is just for configuration structure
        if "models" in config:
            logger.info(
                f"Ensemble config specifies {len(config['models'])} models, "
                f"but models must be added programmatically via add_model()"
            )

        logger.info(
            f"Parsed ensemble configuration '{name}': "
            f"strategy={strategy}, fusion={fusion}, task={task}"
        )

        return composer

    # =========================================================================
    # CONFIGURATION LOADING
    # =========================================================================

    def _load_config(self, source: Path | str) -> dict[str, Any]:
        """
        Load configuration from file or string.

        Supports:
        - YAML files (.yaml, .yml)
        - JSON files (.json)
        - YAML strings
        - JSON strings

        Args:
            source: File path or YAML/JSON string

        Returns:
            Configuration dictionary

        Raises:
            ConfigurationError: If loading fails
        """
        try:
            # Check if it's a file path
            if isinstance(source, Path):
                path = source
            elif isinstance(source, str) and "\n" not in source and len(source) < 500:
                # Likely a file path (no newlines, reasonable length)
                path = Path(source)
            else:
                # It's a string content
                path = None

            if path is not None and path.exists():
                # Load from file
                with open(path) as f:
                    content = f.read()

                # Determine format from extension
                if path.suffix in [".yaml", ".yml"]:
                    config = yaml.safe_load(content)
                elif path.suffix == ".json":
                    config = json.loads(content)
                else:
                    # Try YAML first, then JSON
                    try:
                        config = yaml.safe_load(content)
                    except yaml.YAMLError:
                        config = json.loads(content)

                logger.debug(f"Loaded configuration from file: {path}")

            else:
                # It's a string content
                # Try YAML first (more permissive)
                try:
                    config = yaml.safe_load(source)
                except yaml.YAMLError:
                    # Try JSON
                    try:
                        config = json.loads(source)
                    except json.JSONDecodeError as e:
                        raise ConfigurationError(
                            f"Failed to parse configuration as YAML or JSON: {e}"
                        )

                logger.debug("Loaded configuration from string")

            # Validate it's a dictionary
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Invalid configuration format: expected dict, got {type(config).__name__}"
                )

            return config

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    # =========================================================================
    # CONFIGURATION VALIDATION
    # =========================================================================

    def validate_config(self, config: dict[str, Any], config_type: str) -> dict[str, Any]:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary
            config_type: "custom_architecture" or "ensemble"

        Returns:
            Validation result dictionary with:
            - valid: bool
            - errors: List[str]
            - warnings: List[str]
            - suggestions: List[str]

        Example:
            >>> result = parser.validate_config(config, 'custom_architecture')
            >>> if not result['valid']:
            ...     print(result['errors'])
        """
        errors = []
        warnings = []
        suggestions = []

        if config_type == "custom_architecture":
            # Check for template or layers
            if "template" not in config and "layers" not in config:
                errors.append("Configuration must have either 'template' or 'layers' field")
                suggestions.append("Add 'layers' array or specify 'template' name")

            # Validate layers if present
            if "layers" in config:
                if not isinstance(config["layers"], list):
                    errors.append("'layers' must be a list")
                elif len(config["layers"]) == 0:
                    errors.append("'layers' list is empty")
                    suggestions.append("Add at least one layer to the architecture")
                else:
                    # Validate each layer
                    for i, layer in enumerate(config["layers"]):
                        if not isinstance(layer, dict):
                            errors.append(f"Layer {i} is not a dictionary")
                        elif "type" not in layer:
                            errors.append(f"Layer {i} missing 'type' field")
                            suggestions.append(f"Add 'type' field to layer {i}")
                        else:
                            # Validate params if present
                            if "params" in layer and not isinstance(layer["params"], dict):
                                errors.append(f"Layer {i} 'params' must be a dictionary")

            # Validate template if present
            if "template" in config:
                template_name = config["template"]
                available_templates = self.templates.list_templates()
                if template_name not in available_templates:
                    errors.append(f"Unknown template: '{template_name}'")
                    suggestions.append(f"Available templates: {', '.join(available_templates)}")

            # Check optional fields
            if "task_type" not in config and "template" not in config:
                warnings.append("'task_type' not specified, will use default")
                suggestions.append("Specify 'task_type' for better validation")

            if "in_channels" not in config:
                warnings.append("'in_channels' not specified, will use default (16)")

            if "out_channels" not in config:
                warnings.append(
                    "'out_channels' not specified. Defaults to 1 here, but ModelFactory "
                    "will infer from sample_data if provided. For multi-target regression, "
                    "specify 'out_channels' explicitly or ensure sample_data is passed to ModelFactory."
                )

            # Validate residual connections if present
            if "residual_connections" in config:
                if not isinstance(config["residual_connections"], list):
                    errors.append("'residual_connections' must be a list")
                else:
                    for i, rc in enumerate(config["residual_connections"]):
                        if not isinstance(rc, dict):
                            errors.append(f"Residual connection {i} is not a dictionary")
                        else:
                            if "start" not in rc:
                                errors.append(f"Residual connection {i} missing 'start'")
                            if "end" not in rc:
                                errors.append(f"Residual connection {i} missing 'end'")
                            if "type" in rc and rc["type"] not in ["add", "concat"]:
                                warnings.append(
                                    f"Residual connection {i} has unusual type: {rc['type']}"
                                )

        elif config_type == "ensemble":
            # Check composition
            if "composition" in config:
                comp = config["composition"]
                if not isinstance(comp, dict):
                    errors.append("'composition' must be a dictionary")
                else:
                    # Validate strategy
                    if "strategy" in comp:
                        valid_strategies = ["parallel", "sequential", "hierarchical"]
                        if comp["strategy"] not in valid_strategies:
                            errors.append(
                                f"Invalid strategy: {comp['strategy']}. "
                                f"Must be one of: {valid_strategies}"
                            )

                    # Validate fusion
                    if "fusion" in comp:
                        valid_fusions = ["mean", "weighted", "attention", "voting"]
                        if comp["fusion"] not in valid_fusions:
                            errors.append(
                                f"Invalid fusion method: {comp['fusion']}. "
                                f"Must be one of: {valid_fusions}"
                            )
            else:
                warnings.append("'composition' not specified, will use defaults")

            # Check task type
            if "task_type" not in config:
                warnings.append("'task_type' not specified, will use default")

            # Note about models
            if "models" not in config:
                warnings.append(
                    "No models specified in config. "
                    "Models must be added programmatically via add_model()"
                )

        else:
            errors.append(f"Unknown configuration type: '{config_type}'")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
        }

    # =========================================================================
    # CONFIGURATION EXPORT
    # =========================================================================

    def export_builder_config(self, builder: ArchitectureBuilder, format: str = "yaml") -> str:
        """
        Export ArchitectureBuilder to configuration string.

        Args:
            builder: ArchitectureBuilder instance
            format: Output format ('yaml' or 'json')

        Returns:
            Configuration string

        Example:
            >>> builder = ArchitectureBuilder(...)
            >>> yaml_str = parser.export_builder_config(builder, 'yaml')
            >>> # Later, reload it:
            >>> builder2 = parser.parse_custom_architecture(yaml_str)
        """
        config = builder.get_config().to_dict()

        if format == "yaml":
            return yaml.dump(config, default_flow_style=False, sort_keys=False)
        elif format == "json":
            return json.dumps(config, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")

    def export_composer_config(self, composer: ModelComposer, format: str = "yaml") -> str:
        """
        Export ModelComposer to configuration string.

        Args:
            composer: ModelComposer instance
            format: Output format ('yaml' or 'json')

        Returns:
            Configuration string

        Note:
            Model instances cannot be serialized. Only structure is exported.
        """
        config = composer.get_config().to_dict()

        if format == "yaml":
            return yaml.dump(config, default_flow_style=False, sort_keys=False)
        elif format == "json":
            return json.dumps(config, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def save_config(
        self,
        config: ArchitectureBuilder | ModelComposer | dict[str, Any],
        path: Path | str,
        format: str | None = None,
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: ArchitectureBuilder, ModelComposer, or config dict
            path: Output file path
            format: Output format (inferred from extension if None)

        Example:
            >>> parser.save_config(builder, 'my_architecture.yaml')
        """
        path = Path(path)

        # Determine format
        if format is None:
            if path.suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif path.suffix == ".json":
                format = "json"
            else:
                # Default to YAML
                format = "yaml"

        # Convert to config dict if needed
        if isinstance(config, ArchitectureBuilder):
            config_str = self.export_builder_config(config, format)
        elif isinstance(config, ModelComposer):
            config_str = self.export_composer_config(config, format)
        elif isinstance(config, dict):
            if format == "yaml":
                config_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
            else:
                config_str = json.dumps(config, indent=2)
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        # Write to file
        with open(path, "w") as f:
            f.write(config_str)

        logger.info(f"Saved configuration to: {path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def parse_custom_architecture(
    config: dict[str, Any] | Path | str, task_type: str | None = None, validate: bool = True
) -> ArchitectureBuilder:
    """
    Convenience function to parse custom architecture.

    Args:
        config: Configuration dict, file path, or YAML/JSON string
        task_type: Override task type
        validate: Whether to validate architecture

    Returns:
        ArchitectureBuilder instance

    Example:
        >>> from milia_pipeline.models.builders import parse_custom_architecture
        >>> builder = parse_custom_architecture('my_arch.yaml')
        >>> model = builder.build()
    """
    parser = ArchitectureConfigParser()
    return parser.parse_custom_architecture(config, task_type, validate)


def parse_ensemble(
    config: dict[str, Any] | Path | str, task_type: str | None = None, validate: bool = True
) -> ModelComposer:
    """
    Convenience function to parse ensemble configuration.

    Args:
        config: Configuration dict, file path, or YAML/JSON string
        task_type: Override task type
        validate: Whether to validate configuration

    Returns:
        ModelComposer instance

    Example:
        >>> from milia_pipeline.models.builders import parse_ensemble
        >>> composer = parse_ensemble('my_ensemble.yaml')
        >>> composer.add_model(model1, weight=0.5)
        >>> composer.add_model(model2, weight=0.5)
        >>> ensemble = composer.build()
    """
    parser = ArchitectureConfigParser()
    return parser.parse_ensemble(config, task_type, validate)


def load_config(path: Path | str) -> dict[str, Any]:
    """
    Load configuration file.

    Args:
        path: Path to YAML or JSON file

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config('architecture.yaml')
        >>> builder = parse_custom_architecture(config)
    """
    parser = ArchitectureConfigParser()
    return parser._load_config(path)


def validate_config(
    config: dict[str, Any], config_type: str = "custom_architecture"
) -> dict[str, Any]:
    """
    Validate configuration structure.

    Args:
        config: Configuration dictionary
        config_type: Type of configuration ('custom_architecture' or 'ensemble')

    Returns:
        Validation result

    Example:
        >>> result = validate_config(config)
        >>> if not result['valid']:
        ...     print("Errors:", result['errors'])
        ...     print("Suggestions:", result['suggestions'])
    """
    parser = ArchitectureConfigParser()
    return parser.validate_config(config, config_type)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("config_parser module loaded")
