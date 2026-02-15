"""
Model Loader

Load trained models from checkpoints for inference.
DYNAMIC, PRODUCTION-READY, FUTURE-PROOF implementation.

Dependency Injection Pattern:
- All path resolution requires explicit `working_root_dir: Path` parameter
- No hidden config loading (Service Locator anti-pattern removed)
- Follows CallbackFactory pattern from models/training/callbacks.py

Author: MILIA Team
Version: 2.0.0
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ...factory.model_factory import get_factory
from ..checkpoint.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Load models from checkpoints for inference.

    DYNAMIC: Uses COMPLETE hyperparameters dict from checkpoint.
             NO hard-coded parameter handling - ANY model works.

    PRODUCTION-READY: Recreates model with ALL original settings
                      including wrappers (GraphLevelModelWrapper, etc.)

    FUTURE-PROOF: Works with ANY model in registry without code changes.
                  New models automatically supported via registry lookup.

    Dependency Injection Pattern:
    - Requires explicit working_root_dir: Path parameter
    - No hidden config loading
    - Follows CallbackFactory pattern from models/training/callbacks.py

    Usage:
        # Caller computes working_root_dir and passes it explicitly
        working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()

        # v2.0 checkpoint with relative path
        model, model_info = ModelLoader.load_from_checkpoint(
            "checkpoints/model.pt",
            working_root_dir=working_root_dir
        )
        model.eval()
        predictions = model(data)

        # v1.0 checkpoint (requires manual config)
        model, model_info = ModelLoader.load_from_checkpoint(
            "legacy_model.pt",
            working_root_dir=working_root_dir,
            model_name="GCN",
            hyperparameters={"hidden_channels": 64, ...},
            task_type="graph_regression"
        )
    """

    def __init__(self, working_root_dir: Path):
        """
        Initialize ModelLoader.

        Args:
            working_root_dir: Base directory for resolving relative paths.
                              Must be provided explicitly (Dependency Injection).
        """
        self._working_root_dir = Path(working_root_dir).expanduser().resolve()
        self.checkpoint_manager = CheckpointManager(working_root_dir=self._working_root_dir)
        self.model_factory = get_factory()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        working_root_dir: Path,
        device: torch.device | None = None,
        # Override parameters (for v1.0 checkpoints or customization)
        model_name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        task_type: str | None = None,
        strict: bool = True,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """
        Load model from checkpoint for inference.

        DYNAMIC: For v2.0 checkpoints, automatically recreates model using
                 COMPLETE hyperparameters dict - NO parameter assumptions.

        Dependency Injection: Requires explicit working_root_dir parameter.

        Path resolution search order:
        1. Absolute path (if exists)
        2. Relative path from cwd (if exists)
        3. Default checkpoint directory
        4. Relative to working_root_dir

        For v1.0 checkpoints: Requires model_name, hyperparameters, task_type.

        Args:
            checkpoint_path: Path to checkpoint file.
                             - Relative paths resolved against working_root_dir
                             - Searches default checkpoint directory
            working_root_dir: Base directory for resolving relative paths.
                              Must be provided explicitly (Dependency Injection).
            device: Target device (default: auto-detect)
            model_name: Override model name (required for v1.0 checkpoints)
            hyperparameters: Override hyperparameters (COMPLETE dict)
            task_type: Override task type (required for v1.0 checkpoints)
            strict: Whether to strictly enforce state_dict keys match

        Returns:
            Tuple of (model, model_info):
            - model: Model in eval mode, ready for inference
            - model_info: Complete model_info dict for intelligent usage

        Raises:
            ValueError: If v1.0 checkpoint and missing required parameters
            FileNotFoundError: If checkpoint not found after path resolution

        Example:
            >>> # Caller computes working_root_dir from config
            >>> working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
            >>>
            >>> # Load with explicit working_root_dir
            >>> model, info = ModelLoader.load_from_checkpoint(
            ...     "checkpoints/final_model.pt",
            ...     working_root_dir=working_root_dir
            ... )
            >>>
            >>> model.eval()
            >>> with torch.no_grad():
            ...     predictions = model(batch)
            >>> print(f"Uses edge features: {info['uses_edge_features']}")
        """
        loader = cls(working_root_dir=working_root_dir)
        return loader._load(
            checkpoint_path=checkpoint_path,
            device=device,
            model_name=model_name,
            hyperparameters=hyperparameters,
            task_type=task_type,
            strict=strict,
        )

    def _load(
        self,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
        model_name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        task_type: str | None = None,
        strict: bool = True,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Internal load implementation - DYNAMIC, PRODUCTION-READY, FUTURE-PROOF."""
        # Resolve checkpoint path via CheckpointManager
        resolved_checkpoint_path = self.checkpoint_manager._resolve_checkpoint_path(checkpoint_path)

        # Verify file exists
        if not resolved_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Searched locations:\n"
                f"  - {resolved_checkpoint_path}\n"
                f"  - Default checkpoint dir: {self.checkpoint_manager.get_checkpoint_dir()}\n"
                f"  - Working root: {self._working_root_dir}"
            )

        # Determine device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = self.checkpoint_manager.load(resolved_checkpoint_path, map_location=device)

        # ═══════════════════════════════════════════════════════════════
        # DYNAMIC: Extract COMPLETE configs from checkpoint
        # NO hard-coded parameter handling - use whatever is saved
        # ═══════════════════════════════════════════════════════════════
        hyper_params = checkpoint.get("hyper_parameters", {})

        # Get model identification (with override support)
        resolved_model_name = model_name or hyper_params.get("model_name")
        resolved_task_type = task_type or hyper_params.get("task_type")

        # ═══════════════════════════════════════════════════════════════
        # CRITICAL: Get COMPLETE hyperparameters dict (not individual fields)
        # ═══════════════════════════════════════════════════════════════
        # DYNAMIC: Check multiple locations for backward compatibility:
        # 1. Override parameter (highest priority)
        # 2. Direct 'hyperparameters' key (expected pattern - new checkpoints)
        # 3. Inside model_info.hyperparameters_values (trainer.py pattern)
        # ═══════════════════════════════════════════════════════════════
        if hyperparameters is not None:
            # Explicit override takes highest priority
            resolved_hyperparams = hyperparameters
            logger.info("Using hyperparameters from explicit override")
        elif hyper_params.get("hyperparameters"):
            # Direct 'hyperparameters' key (expected pattern)
            resolved_hyperparams = hyper_params.get("hyperparameters")
            logger.info(
                "Using hyperparameters from checkpoint['hyper_parameters']['hyperparameters']"
            )
        else:
            # Fallback: Inside model_info.hyperparameters_values (trainer.py pattern)
            saved_model_info = hyper_params.get("model_info", {})
            resolved_hyperparams = saved_model_info.get("hyperparameters_values", {})
            if resolved_hyperparams:
                logger.info(
                    "Using hyperparameters from checkpoint['hyper_parameters']['model_info']['hyperparameters_values']"
                )
            else:
                logger.warning(
                    "No hyperparameters found in checkpoint - model may not recreate correctly. "
                    "This checkpoint may have been created with an older version. "
                    "Consider re-training to create a checkpoint with full hyperparameters."
                )
                resolved_hyperparams = {}

        # ═══════════════════════════════════════════════════════════════
        # LOG CRITICAL HYPERPARAMETERS FOR DEBUGGING
        # ═══════════════════════════════════════════════════════════════
        key_params = ["in_channels", "hidden_channels", "out_channels", "num_layers"]
        found_params = {
            k: resolved_hyperparams.get(k) for k in key_params if k in resolved_hyperparams
        }
        missing_params = [k for k in key_params if k not in resolved_hyperparams]

        if found_params:
            param_str = ", ".join(f"{k}={v}" for k, v in found_params.items())
            logger.info(f"Checkpoint hyperparameters: {param_str}")

        if "in_channels" in missing_params:
            logger.warning(
                "CRITICAL: 'in_channels' not found in checkpoint hyperparameters. "
                "Model will be created with default in_channels which may cause "
                "dimension mismatch errors during inference. "
                "Checkpoint was likely created with an older version that didn't save in_channels."
            )

        # Get model_info for returning (PRODUCTION-READY)
        saved_model_info = hyper_params.get("model_info", {})

        # Get wrapper info (FUTURE-PROOF wrapper support)
        wrapper_info = hyper_params.get("wrapper_info", {})

        # Get target selection config if applicable
        target_selection_config = hyper_params.get("target_selection_config")

        # ═══════════════════════════════════════════════════════════════
        # VALIDATE REQUIRED PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        if not resolved_model_name:
            raise ValueError(
                "model_name is required but not found in checkpoint. "
                "This may be a v1.0 checkpoint. Please provide model_name parameter."
            )

        if not resolved_task_type:
            raise ValueError(
                "task_type is required but not found in checkpoint. "
                "This may be a v1.0 checkpoint. Please provide task_type parameter."
            )

        # ═══════════════════════════════════════════════════════════════
        # DYNAMIC MODEL RECREATION
        # Uses ModelFactory.create_model_with_info() with COMPLETE config
        # This ensures ALL runtime behaviors are recreated exactly
        # ═══════════════════════════════════════════════════════════════
        logger.info(
            f"Recreating {resolved_model_name} model for {resolved_task_type} "
            f"(DYNAMIC: using complete hyperparameters dict)"
        )

        # Use create_model_with_info for full model recreation
        model, model_info = self.model_factory.create_model_with_info(
            name=resolved_model_name,
            hyperparameters=resolved_hyperparams,  # COMPLETE dict
            task_type=resolved_task_type,
            sample_data=None,  # Not needed - we have complete config
            device=device,
            target_selection_config=target_selection_config,
        )

        # ═══════════════════════════════════════════════════════════════
        # LOAD STATE DICT
        # Handle both wrapped and unwrapped models with key prefix alignment
        # ═══════════════════════════════════════════════════════════════
        # DYNAMIC: Automatically detects and aligns key prefixes between
        #          saved state_dict and current model
        # PRODUCTION-READY: Handles all wrapper scenarios without code changes
        # FUTURE-PROOF: Works with ANY wrapper that uses 'model.' prefix pattern
        # ═══════════════════════════════════════════════════════════════
        saved_state_dict = checkpoint["model_state_dict"]

        # Detect if saved state_dict has 'model.' prefix (from wrapped model)
        saved_has_prefix = any(k.startswith("model.") for k in saved_state_dict.keys())

        # Detect if current model expects 'model.' prefix (is wrapped)
        model_state_dict = model.state_dict()
        model_expects_prefix = any(k.startswith("model.") for k in model_state_dict.keys())

        logger.debug(
            f"State dict alignment: saved_has_prefix={saved_has_prefix}, "
            f"model_expects_prefix={model_expects_prefix}"
        )

        # Align keys if there's a mismatch
        aligned_state_dict = saved_state_dict

        if saved_has_prefix and not model_expects_prefix:
            # Saved was wrapped, loading into unwrapped → strip 'model.' prefix
            logger.debug("Stripping 'model.' prefix from saved state_dict keys")
            aligned_state_dict = {
                k[6:] if k.startswith("model.") else k: v for k, v in saved_state_dict.items()
            }
        elif not saved_has_prefix and model_expects_prefix:
            # Saved was unwrapped, loading into wrapped → add 'model.' prefix
            logger.debug("Adding 'model.' prefix to saved state_dict keys")
            aligned_state_dict = {
                f"model.{k}" if not k.startswith("model.") else k: v
                for k, v in saved_state_dict.items()
            }

        # ═══════════════════════════════════════════════════════════════
        # FIX 22: PRE-INITIALIZE OUTPUT PROJECTION FOR GraphLevelModelWrapper
        # ═══════════════════════════════════════════════════════════════
        # PROBLEM: GraphLevelModelWrapper creates output_projection LAZILY
        #          during forward() when model output dim != out_channels.
        #          When loading checkpoint, output_projection is None but
        #          checkpoint contains output_projection.weight/.bias keys,
        #          causing "Unexpected key(s)" error in load_state_dict().
        #
        # SOLUTION: Detect output_projection keys in checkpoint and pre-
        #           initialize the layer before loading state_dict.
        #
        # DYNAMIC: Detects presence from checkpoint keys, not hard-coded
        # PRODUCTION-READY: Extracts dimensions from saved weight tensor
        # FUTURE-PROOF: Works for any wrapper with output_projection pattern
        # ═══════════════════════════════════════════════════════════════

        # Check if checkpoint has output_projection keys
        has_output_projection = any("output_projection" in k for k in aligned_state_dict.keys())

        if has_output_projection:
            # Find the weight tensor to extract dimensions
            weight_key = None
            for k in aligned_state_dict.keys():
                if "output_projection.weight" in k:
                    weight_key = k
                    break

            if weight_key is not None:
                # Extract dimensions from saved weight: [out_features, in_features]
                weight_tensor = aligned_state_dict[weight_key]
                out_features, in_features = weight_tensor.shape

                logger.info(
                    f"Pre-initializing output_projection: {in_features} -> {out_features} "
                    f"(from checkpoint)"
                )

                # Pre-initialize output_projection in the wrapper
                # Handle both direct wrapper and nested wrapper cases
                wrapper = None
                if hasattr(model, "output_projection"):
                    wrapper = model
                elif hasattr(model, "model") and hasattr(model.model, "output_projection"):
                    wrapper = model.model

                if wrapper is not None:
                    # Create the Linear layer with correct dimensions
                    wrapper.output_projection = torch.nn.Linear(in_features, out_features).to(
                        device
                    )
                    wrapper._model_out_dim = in_features
                    logger.debug(f"Initialized output_projection on {type(wrapper).__name__}")
                else:
                    logger.warning(
                        f"Checkpoint contains output_projection but model "
                        f"({type(model).__name__}) has no output_projection attribute. "
                        f"State dict loading may fail."
                    )

        # Load the aligned state dict
        try:
            model.load_state_dict(aligned_state_dict, strict=strict)
        except RuntimeError:
            # If still fails, try loading into inner model as fallback
            if hasattr(model, "model") and saved_has_prefix:
                logger.debug("Fallback: loading state_dict into inner model")
                # Strip prefix and load into inner model
                inner_state_dict = {
                    k[6:] if k.startswith("model.") else k: v for k, v in saved_state_dict.items()
                }
                model.model.load_state_dict(inner_state_dict, strict=strict)
            else:
                raise

        # Set to eval mode for inference
        model.eval()

        # ═══════════════════════════════════════════════════════════════
        # MERGE model_info: Prefer saved values, fill with recreated
        # This ensures we have COMPLETE info for downstream usage
        # ═══════════════════════════════════════════════════════════════
        final_model_info = {**model_info}  # Start with recreated
        final_model_info.update(saved_model_info)  # Override with saved

        # ═══════════════════════════════════════════════════════════════
        # FIX 18: INCLUDE data_info FOR FEATURIZATION CONFIG ACCESS
        # ═══════════════════════════════════════════════════════════════
        # DYNAMIC: Includes whatever is in checkpoint['data_info']
        # PRODUCTION-READY: Makes structural_features_config accessible
        # FUTURE-PROOF: Works with any data_info structure
        # ═══════════════════════════════════════════════════════════════
        data_info = checkpoint.get("data_info", {})
        if data_info:
            final_model_info["data_info"] = data_info
            if data_info.get("structural_features_config"):
                logger.info(
                    f"Featurization config loaded from checkpoint: "
                    f"atom={list(data_info['structural_features_config'].get('atom', []))}"
                )

        logger.info(f"Model loaded successfully from {resolved_checkpoint_path}")
        logger.debug(f"Model info: uses_edge_features={final_model_info.get('uses_edge_features')}")

        return model, final_model_info

    @classmethod
    def get_checkpoint_info(
        cls, checkpoint_path: str | Path, working_root_dir: Path
    ) -> dict[str, Any]:
        """
        Get information about a checkpoint without loading the model.

        DYNAMIC: Returns whatever metadata is in the checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
            working_root_dir: Base directory for resolving relative paths.

        Returns:
            Dictionary with checkpoint information
        """
        loader = cls(working_root_dir=working_root_dir)

        # Resolve path
        resolved_path = loader.checkpoint_manager._resolve_checkpoint_path(checkpoint_path)

        checkpoint = loader.checkpoint_manager.load(resolved_path)

        hyper_params = checkpoint.get("hyper_parameters", {})
        version_info = checkpoint.get("version_info", {})
        data_info = checkpoint.get("data_info", {})

        return {
            "format_version": version_info.get("checkpoint_format_version", "1.0"),
            "is_v2": loader.checkpoint_manager.is_v2_checkpoint(checkpoint),
            "model_name": hyper_params.get("model_name", "UNKNOWN"),
            "task_type": hyper_params.get("task_type", "UNKNOWN"),
            "epoch": checkpoint.get("epoch", 0),
            "best_val_loss": checkpoint.get("best_val_loss", None),
            "hyper_parameters": hyper_params,
            "data_info": data_info,
            "version_info": version_info,
            "checkpoint_path": str(resolved_path),  # Include resolved path
            # DYNAMIC: Include whatever else is in checkpoint
            "has_wrapper_info": "wrapper_info" in hyper_params,
            "has_target_selection": hyper_params.get("target_selection_config") is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS (Following Chemprop pattern)
# ═══════════════════════════════════════════════════════════════════════════


def load_model(
    checkpoint_path: str | Path,
    working_root_dir: Path,
    device: torch.device | None = None,
    **kwargs,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load model from checkpoint.

    DYNAMIC: Returns (model, model_info) tuple for full flexibility.

    Args:
        checkpoint_path: Path to checkpoint.
                         - Relative paths resolved against working_root_dir
                         - Searches default checkpoint directory
        working_root_dir: Base directory for resolving relative paths.
                          Must be provided explicitly (Dependency Injection).
        device: Target device
        **kwargs: Additional arguments passed to ModelLoader

    Returns:
        Tuple of (model, model_info)

    Example:
        >>> from milia_pipeline.models.post_training import load_model
        >>>
        >>> # Caller computes working_root_dir from config
        >>> working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
        >>>
        >>> # Load with explicit working_root_dir
        >>> model, info = load_model(
        ...     "checkpoints/final_model.pt",
        ...     working_root_dir=working_root_dir
        ... )
        >>>
        >>> print(f"Task type: {info['task_type']}")
    """
    return ModelLoader.load_from_checkpoint(
        checkpoint_path=checkpoint_path, working_root_dir=working_root_dir, device=device, **kwargs
    )


def load_model_only(
    checkpoint_path: str | Path,
    working_root_dir: Path,
    device: torch.device | None = None,
    **kwargs,
) -> nn.Module:
    """
    Load model from checkpoint (model only, discard info).

    For simple use cases where model_info is not needed.

    Args:
        checkpoint_path: Path to checkpoint
        working_root_dir: Base directory for resolving relative paths.
                          Must be provided explicitly (Dependency Injection).
        device: Target device
        **kwargs: Additional arguments

    Returns:
        Model in eval mode

    Example:
        >>> working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
        >>> model = load_model_only("final_model.pt", working_root_dir=working_root_dir)
        >>> predictions = model(data)
    """
    model, _ = load_model(
        checkpoint_path, working_root_dir=working_root_dir, device=device, **kwargs
    )
    return model
