"""
Fine Tuner

Transfer learning and fine-tuning for pre-trained models.
Follows standard transfer learning patterns.

Dependency Injection Pattern:
- All path resolution requires explicit `working_root_dir: Path` parameter
- No hidden config loading (Service Locator anti-pattern removed)
- Follows CallbackFactory pattern from models/training/callbacks.py

Author: MILIA Team
Version: 2.0.0
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum
import logging

import torch
import torch.nn as nn

from ..inference.model_loader import ModelLoader
from ..checkpoint.checkpoint_manager import CheckpointManager
from ...factory.model_factory import get_factory

logger = logging.getLogger(__name__)


class FreezeStrategy(Enum):
    """Strategies for freezing model components."""
    NONE = "none"  # Train all parameters
    ENCODER = "encoder"  # Freeze GNN encoder, train head
    ENCODER_PARTIAL = "encoder_partial"  # Freeze early layers
    ALL_BUT_LAST = "all_but_last"  # Freeze all but last layer


class FineTuner:
    """
    Fine-tune pre-trained models for transfer learning.
    
    Following Chemprop's approach:
    - Load pre-trained checkpoint
    - Optionally freeze encoder/layers
    - Train on new data
    
    Dependency Injection Pattern:
    - Requires explicit working_root_dir: Path parameter
    - No hidden config loading
    - Follows CallbackFactory pattern from models/training/callbacks.py
    
    Usage:
        # Caller computes working_root_dir from config
        working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
        
        # Basic fine-tuning with explicit working_root_dir
        fine_tuner = FineTuner.from_checkpoint(
            "checkpoints/pretrained.pt",
            working_root_dir=working_root_dir
        )
        model = fine_tuner.prepare_for_finetuning(
            new_out_channels=5,  # New task has 5 targets
            freeze_strategy=FreezeStrategy.ENCODER
        )
        
        # Then train with Trainer as normal
        trainer = Trainer(model=model, ...)
        trainer.fit()
    """
    
    def __init__(
        self,
        model: nn.Module,
        hyper_parameters: Dict[str, Any],
        working_root_dir: Path
    ):
        """
        Initialize fine tuner.
        
        Args:
            model: Pre-trained model
            hyper_parameters: Model hyper_parameters from checkpoint
            working_root_dir: Base directory for resolving relative paths.
                              Must be provided explicitly (Dependency Injection).
        """
        self.model = model
        self.hyper_parameters = hyper_parameters
        self.original_out_channels = hyper_parameters.get('out_channels', 1)
        self._working_root_dir = Path(working_root_dir).expanduser().resolve()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        working_root_dir: Path,
        device: Optional[torch.device] = None
    ) -> 'FineTuner':
        """
        Create FineTuner from checkpoint.
        
        Dependency Injection: Requires explicit working_root_dir parameter.
        
        Args:
            checkpoint_path: Path to pre-trained checkpoint.
                             - Relative paths resolved against working_root_dir
                             - Searches default checkpoint directory
            working_root_dir: Base directory for resolving relative paths.
                              Must be provided explicitly (Dependency Injection).
            device: Target device
            
        Returns:
            FineTuner instance
            
        Example:
            >>> # Caller computes working_root_dir from config
            >>> working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
            >>> 
            >>> # Using relative path
            >>> fine_tuner = FineTuner.from_checkpoint(
            ...     "checkpoints/pretrained.pt",
            ...     working_root_dir=working_root_dir
            ... )
        """
        # Create CheckpointManager for path resolution
        cm = CheckpointManager(working_root_dir=working_root_dir)
        resolved_path = cm._resolve_checkpoint_path(checkpoint_path)
        
        # Load model
        model, model_info = ModelLoader.load_from_checkpoint(
            checkpoint_path=resolved_path,
            working_root_dir=working_root_dir,
            device=device
        )
        
        # Get hyper_parameters
        checkpoint = cm.load(resolved_path)
        hyper_params = checkpoint.get('hyper_parameters', {})
        
        return cls(model=model, hyper_parameters=hyper_params, working_root_dir=working_root_dir)
    
    def prepare_for_finetuning(
        self,
        new_out_channels: Optional[int] = None,
        new_task_type: Optional[str] = None,
        freeze_strategy: FreezeStrategy = FreezeStrategy.ENCODER,
        freeze_layers: Optional[int] = None,
    ) -> nn.Module:
        """
        Prepare model for fine-tuning.
        
        Args:
            new_out_channels: New output dimension (e.g., different number of targets)
            new_task_type: New task type (e.g., switch from regression to classification)
            freeze_strategy: Strategy for freezing parameters
            freeze_layers: Number of layers to freeze (for ENCODER_PARTIAL)
            
        Returns:
            Model prepared for fine-tuning
        """
        model = self.model
        
        # Apply freeze strategy
        self._apply_freeze_strategy(model, freeze_strategy, freeze_layers)
        
        # Replace output head if needed
        if new_out_channels is not None:
            model = self._replace_output_head(model, new_out_channels)
        
        # Set to training mode
        model.train()
        
        # Log frozen parameters
        self._log_parameter_status(model)
        
        return model
    
    def _apply_freeze_strategy(
        self,
        model: nn.Module,
        strategy: FreezeStrategy,
        freeze_layers: Optional[int] = None
    ):
        """Apply parameter freezing strategy."""
        if strategy == FreezeStrategy.NONE:
            # Unfreeze all
            for param in model.parameters():
                param.requires_grad = True
            return
        
        if strategy == FreezeStrategy.ENCODER:
            # Freeze all conv/message passing layers
            self._freeze_encoder_layers(model)
            return
        
        if strategy == FreezeStrategy.ENCODER_PARTIAL:
            # Freeze first N layers
            n_layers = freeze_layers or 2
            self._freeze_first_n_layers(model, n_layers)
            return
        
        if strategy == FreezeStrategy.ALL_BUT_LAST:
            # Freeze all but the last linear layer
            self._freeze_all_but_last(model)
            return
    
    def _freeze_encoder_layers(self, model: nn.Module):
        """Freeze GNN encoder layers (conv layers)."""
        for name, param in model.named_parameters():
            # Freeze conv/message passing layers
            if any(x in name.lower() for x in ['conv', 'message', 'propagate', 'aggr']):
                param.requires_grad = False
                logger.debug(f"Frozen: {name}")
            else:
                param.requires_grad = True
    
    def _freeze_first_n_layers(self, model: nn.Module, n: int):
        """Freeze first N layers."""
        # Get layer names
        layer_names = []
        for name, _ in model.named_parameters():
            layer_base = name.split('.')[0]
            if layer_base not in layer_names:
                layer_names.append(layer_base)
        
        # Freeze first N
        layers_to_freeze = layer_names[:n]
        
        for name, param in model.named_parameters():
            layer_base = name.split('.')[0]
            if layer_base in layers_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def _freeze_all_but_last(self, model: nn.Module):
        """Freeze all parameters except the last layer."""
        param_names = list(dict(model.named_parameters()).keys())
        last_layer_prefix = param_names[-1].rsplit('.', 1)[0] if param_names else ''
        
        for name, param in model.named_parameters():
            if name.startswith(last_layer_prefix):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def _replace_output_head(
        self,
        model: nn.Module,
        new_out_channels: int
    ) -> nn.Module:
        """Replace output head for new task."""
        # Find the last linear layer
        last_linear = None
        last_linear_name = None
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
                last_linear_name = name
        
        if last_linear is None:
            logger.warning("No linear layer found to replace")
            return model
        
        # Create new linear layer
        new_linear = nn.Linear(
            last_linear.in_features,
            new_out_channels
        )
        
        # Replace in model
        parent_name = '.'.join(last_linear_name.split('.')[:-1])
        child_name = last_linear_name.split('.')[-1]
        
        if parent_name:
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, new_linear)
        else:
            setattr(model, child_name, new_linear)
        
        logger.info(
            f"Replaced output head: {last_linear.out_features} -> {new_out_channels}"
        )
        
        return model
    
    def _log_parameter_status(self, model: nn.Module):
        """Log trainable vs frozen parameter counts."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = trainable + frozen
        
        logger.info(
            f"Parameter status: "
            f"{trainable:,} trainable ({trainable/total*100:.1f}%), "
            f"{frozen:,} frozen ({frozen/total*100:.1f}%)"
        )
