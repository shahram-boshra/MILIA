"""
Checkpoint Manager

Enhanced checkpoint saving and loading with model recreation metadata.
DYNAMIC, PRODUCTION-READY, FUTURE-PROOF implementation.

Dependency Injection Pattern:
- All path resolution requires explicit `working_root_dir: Path` parameter
- No hidden config loading (Service Locator anti-pattern removed)
- Follows CallbackFactory pattern from models/training/callbacks.py

Author: MILIA Team
Version: 2.0.0
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import logging

import torch

logger = logging.getLogger(__name__)

# Checkpoint format version
CHECKPOINT_FORMAT_VERSION = "2.0"


class CheckpointManager:
    """
    Manager for enhanced checkpoint saving and loading.
    
    DYNAMIC: Saves COMPLETE hyperparameters dict (not hard-coded fields)
    PRODUCTION-READY: Saves complete model_info from create_model_with_info()
    FUTURE-PROOF: Version tracking, backward compatibility with v1.0
    
    Args:
        working_root_dir: Base directory for path resolution. All relative paths
                          are resolved against this directory. Required parameter
                          following Dependency Injection pattern.
    
    Usage:
        # Caller provides working_root_dir explicitly
        working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
        manager = CheckpointManager(working_root_dir=working_root_dir)
        
        # Save with relative path
        manager.save(
            filepath="checkpoints/model.pt",  # Resolved to working_root_dir/checkpoints/model.pt
            model=model,
            optimizer=optimizer,
            ...
        )
        
        # Load with automatic path resolution
        checkpoint = manager.load("checkpoints/model.pt")
    """
    
    def __init__(self, working_root_dir: Path):
        """
        Initialize CheckpointManager.
        
        Args:
            working_root_dir: Base directory for resolving relative paths.
                              Must be provided explicitly (Dependency Injection).
        """
        self._working_root_dir = Path(working_root_dir).expanduser().resolve()
    
    @staticmethod
    def create_version_info() -> Dict[str, Any]:
        """Create version_info dict for compatibility checking."""
        try:
            import torch_geometric
            pyg_version = torch_geometric.__version__
        except ImportError:
            pyg_version = "unknown"
        
        return {
            'milia_version': '1.0.0',  # TODO: Import from milia_pipeline.__version__
            'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
            'pytorch_version': str(torch.__version__),
            'pyg_version': str(pyg_version),
            'created_at': datetime.now().isoformat(),
        }
    
    def get_checkpoint_dir(self, subdir: str = "checkpoints") -> Path:
        """
        Get the default checkpoint directory.
        
        Args:
            subdir: Subdirectory name (default: "checkpoints")
            
        Returns:
            Path to checkpoint directory (created if needed)
        """
        checkpoint_dir = self._working_root_dir / subdir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def _resolve_path(self, path: Union[str, Path], create_parents: bool = False) -> Path:
        """
        Resolve path against working_root_dir.
        
        Args:
            path: Path to resolve (absolute paths returned as-is)
            create_parents: If True, create parent directories
            
        Returns:
            Resolved absolute path
        """
        path = Path(path).expanduser()
        if path.is_absolute():
            result = path.resolve()
        else:
            result = (self._working_root_dir / path).resolve()
        
        if create_parents:
            result.parent.mkdir(parents=True, exist_ok=True)
        
        return result
    
    def _resolve_checkpoint_path(self, checkpoint_path: Union[str, Path]) -> Path:
        """
        Resolve checkpoint path with intelligent search.
        
        Search order:
        1. If absolute and exists → return it
        2. If relative and exists from cwd → return it
        3. Check in default checkpoint directory (working_root_dir/checkpoints/)
        4. Resolve relative to working_root_dir
        
        Args:
            checkpoint_path: Checkpoint path to resolve
            
        Returns:
            Resolved path (not guaranteed to exist - caller should verify)
        """
        path = Path(checkpoint_path).expanduser()
        
        # 1. If absolute and exists, return it
        if path.is_absolute() and path.exists():
            return path.resolve()
        
        # 2. If relative and exists from cwd, return it
        if path.exists():
            return path.resolve()
        
        # 3. Check in default checkpoint directory
        checkpoint_dir = self.get_checkpoint_dir()
        candidate = checkpoint_dir / path.name
        if candidate.exists():
            logger.debug(f"Found checkpoint in default dir: {candidate}")
            return candidate
        
        # 4. Resolve relative to working_root_dir
        return self._resolve_path(path)
    
    def save(
        self,
        filepath: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics_history: Optional[Dict] = None,
        best_val_loss: float = float('inf'),
        hyper_parameters: Optional[Dict[str, Any]] = None,
        data_info: Optional[Dict[str, Any]] = None,
        **extra_data
    ) -> Path:
        """
        Save enhanced checkpoint with model recreation metadata.
        
        DYNAMIC: hyper_parameters should contain COMPLETE dicts, not individual fields
        PHASE 6: Filepath resolved via global_paths.working_root_dir
        
        Args:
            filepath: Path to save checkpoint.
                      - Relative paths resolved against global_paths.working_root_dir
                      - Absolute paths used as-is
            model: PyTorch model
            optimizer: Optional optimizer
            scheduler: Optional LR scheduler
            epoch: Current epoch
            global_step: Current global step
            metrics_history: Training metrics history
            best_val_loss: Best validation loss
            hyper_parameters: COMPLETE model recreation parameters including:
                - model_name: Registry name (e.g., "GCN")
                - task_type: Task type string
                - hyperparameters: COMPLETE hyperparameters dict
                - model_info: COMPLETE model_info from create_model_with_info()
                - wrapper_info: Wrapper type and parameters
                - target_selection_config: Target selection if applicable
            data_info: Data compatibility information
            **extra_data: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            # Training state
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'metrics_history': metrics_history or {},
            'best_val_loss': best_val_loss,
            
            # Model recreation metadata (DYNAMIC - stores COMPLETE dicts)
            'hyper_parameters': hyper_parameters or {},
            'data_info': data_info or {},
            'version_info': self.create_version_info(),
            
            # Extra data
            **extra_data
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Resolve path against working_root_dir
        resolved_filepath = self._resolve_path(filepath, create_parents=True)
        
        torch.save(checkpoint, resolved_filepath)
        
        logger.info(f"Saved enhanced checkpoint to {resolved_filepath}")
        logger.debug(f"Checkpoint format version: {CHECKPOINT_FORMAT_VERSION}")
        
        return resolved_filepath
    
    def load(
        self,
        filepath: Union[str, Path],
        map_location: Optional[torch.device] = None,
        weights_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint with backward compatibility.
        
        Path resolution search order:
        1. Checks if absolute path exists
        2. Checks if relative path exists from cwd
        3. Checks in default checkpoint directory
        4. Resolves relative to working_root_dir
        
        Args:
            filepath: Path to checkpoint.
                      - Supports relative paths resolved against working_root_dir
                      - Searches default checkpoint directory
            map_location: Device to load to
            weights_only: Restrict unpickling to tensors, primitive types, and
                          dictionaries (default: True). Matches PyTorch >= 2.6
                          default. Set to False only for third-party or legacy
                          checkpoints containing custom objects.
            
        Returns:
            Checkpoint dictionary
        """
        # Resolve path with intelligent search
        resolved_filepath = self._resolve_checkpoint_path(filepath)
        
        checkpoint = torch.load(
            resolved_filepath,
            map_location=map_location,
            weights_only=weights_only,
        )
        
        # Check format version
        version_info = checkpoint.get('version_info', {})
        format_version = version_info.get('checkpoint_format_version', '1.0')
        
        if format_version == '1.0':
            # Legacy checkpoint - add empty metadata
            logger.warning(
                f"Loading v1.0 checkpoint from {filepath}. "
                "Model recreation requires manual configuration."
            )
            checkpoint.setdefault('hyper_parameters', {})
            checkpoint.setdefault('data_info', {})
            checkpoint.setdefault('version_info', {'checkpoint_format_version': '1.0'})
        
        logger.info(f"Loaded checkpoint from {resolved_filepath} (format v{format_version})")
        
        return checkpoint
    
    def is_v2_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Check if checkpoint has v2.0 metadata."""
        version_info = checkpoint.get('version_info', {})
        format_version = version_info.get('checkpoint_format_version', '1.0')
        return format_version >= '2.0'
    
    def get_hyper_parameters(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hyper_parameters from checkpoint."""
        return checkpoint.get('hyper_parameters', {})
    
    def get_model_name(self, checkpoint: Dict[str, Any]) -> Optional[str]:
        """Extract model name from checkpoint."""
        hyper_params = self.get_hyper_parameters(checkpoint)
        return hyper_params.get('model_name')
