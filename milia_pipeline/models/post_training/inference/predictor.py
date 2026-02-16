"""
Predictor

PyG-compatible inference for molecular graphs.
Handles batching, device management, and output formatting.

Dependency Injection Pattern:
- All path resolution requires explicit `working_root_dir: Path` parameter
- No hidden config loading (Service Locator anti-pattern removed)
- Follows CallbackFactory pattern from models/training/callbacks.py

Author: MILIA Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

if TYPE_CHECKING:
    import numpy
    from torch_geometric.data import Dataset

logger = logging.getLogger(__name__)


class Predictor:
    """
    PyG-compatible predictor for molecular graphs.

    Unlike deployment_strategies.py predict() which expects torch.Tensor,
    this handles PyG Data objects correctly.

    Dependency Injection Pattern:
    - Requires explicit working_root_dir: Path parameter
    - No hidden config loading
    - Follows CallbackFactory pattern from models/training/callbacks.py

    Usage:
        # Caller computes working_root_dir from config
        working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()

        # From checkpoint with explicit working_root_dir
        predictor = Predictor.from_checkpoint(
            "checkpoints/model.pt",
            working_root_dir=working_root_dir
        )

        # Single prediction
        prediction = predictor.predict(data)

        # Batch prediction
        predictions = predictor.predict_batch(dataset)

        # Save predictions
        predictor.save_predictions(predictions, "results.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        working_root_dir: Path,
        device: torch.device | None = None,
        task_type: str | None = None,
        model_info: dict[str, Any] | None = None,  # FIX 19: NEW PARAMETER
    ):
        """
        Initialize predictor.

        Args:
            model: PyTorch model (should be in eval mode)
            working_root_dir: Base directory for resolving relative paths.
                              Must be provided explicitly (Dependency Injection).
            device: Target device
            task_type: Task type for output formatting
            model_info: Model metadata including featurization config from checkpoint
        """
        self.model = model
        self._working_root_dir = Path(working_root_dir).expanduser().resolve()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_type = task_type

        # ====================================================================
        # FIX 19: STORE model_info FOR FEATURIZATION CONFIG ACCESS
        # ====================================================================
        # DYNAMIC: Stores whatever model_info is passed from checkpoint
        # PRODUCTION-READY: Enables access to structural_features_config
        # FUTURE-PROOF: Works with any model_info structure
        # ====================================================================
        self.model_info = model_info or {}

        # Ensure model is on correct device and in eval mode
        self.model.to(self.device)
        self.model.eval()

    # ========================================================================
    # FIX 19: PROPERTY FOR FEATURIZATION CONFIG ACCESS
    # ========================================================================
    # DYNAMIC: Returns whatever structural_features_config is in model_info
    # PRODUCTION-READY: Provides clean API for accessing featurization config
    # FUTURE-PROOF: Works with any model_info/data_info structure
    # ========================================================================
    @property
    def structural_features_config(self) -> dict[str, Any] | None:
        """
        Get structural features config from checkpoint for featurization.

        Returns:
            Dict with 'atom' and 'bond' feature lists, or None if not available.

        Example:
            >>> predictor = Predictor.from_checkpoint("model.pt", working_root_dir=Path("."))
            >>> config = predictor.structural_features_config
            >>> if config:
            ...     print(f"Atom features: {config.get('atom', [])}")
        """
        if self.model_info:
            data_info = self.model_info.get("data_info", {})
            return data_info.get("structural_features_config")
        return None

    def _resolve_path(self, path: str | Path, create_parents: bool = False) -> Path:
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        working_root_dir: Path,
        device: torch.device | None = None,
        **loader_kwargs,
    ) -> Predictor:
        """
        Create predictor from checkpoint.

        Dependency Injection: Requires explicit working_root_dir parameter.

        Args:
            checkpoint_path: Path to model checkpoint.
                             - Relative paths resolved against working_root_dir
                             - Searches default checkpoint directory
            working_root_dir: Base directory for resolving relative paths.
                              Must be provided explicitly (Dependency Injection).
            device: Target device
            **loader_kwargs: Additional args for ModelLoader

        Returns:
            Predictor instance

        Example:
            >>> # Caller computes working_root_dir from config
            >>> working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
            >>>
            >>> # Using relative path
            >>> predictor = Predictor.from_checkpoint(
            ...     "checkpoints/best_model.pt",
            ...     working_root_dir=working_root_dir
            ... )
        """
        from ..checkpoint.checkpoint_manager import CheckpointManager
        from .model_loader import ModelLoader

        # Create CheckpointManager for path resolution
        cm = CheckpointManager(working_root_dir=working_root_dir)
        resolved_path = cm._resolve_checkpoint_path(checkpoint_path)

        # Load model
        model, model_info = ModelLoader.load_from_checkpoint(
            checkpoint_path=resolved_path,
            working_root_dir=working_root_dir,
            device=device,
            **loader_kwargs,
        )

        # Get task_type from checkpoint
        checkpoint = cm.load(resolved_path)
        hyper_params = checkpoint.get("hyper_parameters", {})
        task_type = hyper_params.get("task_type")

        # FIX 19: Pass model_info to __init__ (contains data_info with structural_features_config)
        return cls(
            model=model,
            working_root_dir=working_root_dir,
            device=device,
            task_type=task_type,
            model_info=model_info,  # FIX 19: Pass model_info for featurization config access
        )

    def predict(
        self, data: Data | Batch, return_numpy: bool = False
    ) -> torch.Tensor | numpy.ndarray:
        """
        Make prediction on PyG Data or Batch.

        Args:
            data: PyG Data object or Batch
            return_numpy: If True, return numpy array

        Returns:
            Predictions tensor or numpy array

        Example:
            >>> data = Data(x=node_features, edge_index=edge_index)
            >>> prediction = predictor.predict(data)
        """
        # Move data to device
        data = data.to(self.device)

        # Run inference
        with torch.no_grad():
            output = self._forward(data)

        # Post-process based on task type
        output = self._postprocess(output, data)

        if return_numpy:
            return output.cpu().numpy()
        return output

    def _forward(self, data: Data | Batch) -> torch.Tensor:
        """
        Forward pass handling PyG-specific attributes.

        =======================================================================
        FIX 23: DYNAMIC FORWARD SIGNATURE INTROSPECTION
        FIX 25: 3D MOLECULAR MODEL SUPPORT (SchNet, DimeNet, PaiNN, etc.)
        =======================================================================
        DYNAMIC: Uses Python's inspect module to introspect the model's forward
                 signature at runtime, detecting whether the model is a standard
                 GNN (x, edge_index, ...) or a 3D molecular model (z, pos, batch).
        PRODUCTION-READY: Handles all PyG model types including:
                          - Standard GNNs: GCN, GAT, GraphSAGE, GIN, etc.
                          - 3D Molecular Models: SchNet, DimeNet, PaiNN, EGNN, etc.
                          - Custom models and wrappers
        FUTURE-PROOF: Works with any model by introspecting its actual forward
                      signature, including future PyG models with new parameters.
        =======================================================================

        3D Molecular Models Detection:
        - These models use forward(z, pos, batch) instead of forward(x, edge_index, ...)
        - z: Atomic numbers tensor [num_atoms]
        - pos: 3D atomic positions tensor [num_atoms, 3]
        - batch: Batch assignment tensor [num_atoms]
        - Examples: SchNet, DimeNet, DimeNetPlusPlus, PaiNN, EGNN, ViSNet, etc.

        Evidence from PyG documentation:
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.SchNet.html
        - forward(z: Tensor, pos: Tensor, batch: Optional[Tensor] = None) → Tensor
        =======================================================================
        """
        import inspect

        # =================================================================
        # Get the actual model (unwrap if wrapped by GraphLevelModelWrapper)
        # =================================================================
        actual_model = self.model
        while hasattr(actual_model, "model") and actual_model.model is not actual_model:
            actual_model = actual_model.model

        # =================================================================
        # Introspect model's forward signature to detect model type
        # =================================================================
        try:
            sig = inspect.signature(actual_model.forward)
            params = list(sig.parameters.keys())
            valid_params = set(params)

            # =============================================================
            # FIX 25: DETECT 3D MOLECULAR MODELS
            # =============================================================
            # 3D molecular models have 'z' and 'pos' as their first parameters
            # instead of 'x' and 'edge_index'. We detect this by checking if:
            # 1. 'z' is in the first 2 parameters (atomic numbers)
            # 2. 'pos' is in the first 3 parameters (3D positions)
            # This is more robust than name-based detection (SchNet, DimeNet, etc.)
            # =============================================================
            first_params = params[:3] if len(params) >= 3 else params
            is_3d_molecular_model = "z" in first_params and "pos" in first_params

            if is_3d_molecular_model:
                # ---------------------------------------------------------
                # 3D MOLECULAR MODEL FORWARD PATH
                # ---------------------------------------------------------
                # These models expect: forward(z, pos, batch=batch)
                # - z: Atomic numbers from data.z
                # - pos: 3D coordinates from data.pos
                # - batch: Batch assignment from data.batch
                # ---------------------------------------------------------

                # Validate required data attributes
                if not hasattr(data, "z") or data.z is None:
                    raise ValueError(
                        "3D molecular model requires 'z' (atomic numbers) in data. "
                        "Ensure your input format provides atomic numbers (e.g., XYZ file)."
                    )
                if not hasattr(data, "pos") or data.pos is None:
                    raise ValueError(
                        "3D molecular model requires 'pos' (3D coordinates) in data. "
                        "Ensure your input format provides 3D positions (e.g., XYZ file)."
                    )

                z = data.z
                pos = data.pos
                batch = getattr(data, "batch", None)

                # Build kwargs for any additional parameters the model accepts
                kwargs_3d = {}
                if "batch" in valid_params and batch is not None:
                    kwargs_3d["batch"] = batch

                # Some 3D models may accept additional parameters
                if (
                    "edge_index" in valid_params
                    and hasattr(data, "edge_index")
                    and data.edge_index is not None
                ):
                    kwargs_3d["edge_index"] = data.edge_index

                logger.debug(
                    f"3D molecular model detected. Calling forward(z, pos, ...) "
                    f"with z.shape={list(z.shape)}, pos.shape={list(pos.shape)}"
                )

                return self.model(z, pos, **kwargs_3d)

            # =============================================================
            # STANDARD GNN FORWARD PATH
            # =============================================================
            # These models expect: forward(x, edge_index, ...)
            # =============================================================

            # Required arguments for standard GNNs
            x = data.x
            edge_index = data.edge_index

            # Build candidate kwargs from data attributes
            candidate_kwargs = {}

            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                candidate_kwargs["edge_attr"] = data.edge_attr

            if hasattr(data, "edge_weight") and data.edge_weight is not None:
                candidate_kwargs["edge_weight"] = data.edge_weight

            if hasattr(data, "batch") and data.batch is not None:
                candidate_kwargs["batch"] = data.batch

            if hasattr(data, "pos") and data.pos is not None:
                candidate_kwargs["pos"] = data.pos

            if hasattr(data, "batch_size") and data.batch_size is not None:
                candidate_kwargs["batch_size"] = data.batch_size

            # Check if forward accepts **kwargs (VAR_KEYWORD)
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            # Filter candidate kwargs to only those accepted by the model
            if accepts_var_keyword:
                # Model accepts **kwargs, pass all candidates
                kwargs = candidate_kwargs
            else:
                # Model has fixed signature, filter to valid params only
                kwargs = {k: v for k, v in candidate_kwargs.items() if k in valid_params}

            # Call model with filtered kwargs
            return self.model(x, edge_index, **kwargs)

        except (ValueError, TypeError) as e:
            # Fallback: If signature introspection fails, try both approaches
            logger.warning(f"Signature introspection failed: {e}. Using fallback logic.")

            # Try 3D model approach first if data has z and pos but no x
            if (
                hasattr(data, "z")
                and data.z is not None
                and hasattr(data, "pos")
                and data.pos is not None
            ):
                if not hasattr(data, "x") or data.x is None:
                    z = data.z
                    pos = data.pos
                    batch = getattr(data, "batch", None)
                    kwargs_3d = {"batch": batch} if batch is not None else {}
                    return self.model(z, pos, **kwargs_3d)

            # Fallback to standard GNN approach
            x = data.x
            edge_index = data.edge_index
            kwargs = {}
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                kwargs["edge_attr"] = data.edge_attr
            if hasattr(data, "batch") and data.batch is not None:
                kwargs["batch"] = data.batch
            return self.model(x, edge_index, **kwargs)

    def _postprocess(self, output: torch.Tensor, data: Data | Batch) -> torch.Tensor:
        """Post-process output based on task type."""
        if self.task_type and "classification" in self.task_type.lower():
            # For classification, return class predictions
            if output.dim() > 1 and output.size(-1) > 1:
                return output.argmax(dim=-1)
        return output

    def predict_batch(
        self,
        dataset: list[Data] | Dataset,
        batch_size: int = 32,
        num_workers: int = 0,
        return_numpy: bool = False,
    ) -> torch.Tensor | numpy.ndarray:
        """
        Make predictions on entire dataset.

        Args:
            dataset: List of Data objects or PyG Dataset
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes
            return_numpy: If True, return numpy array

        Returns:
            Concatenated predictions

        Example:
            >>> predictions = predictor.predict_batch(test_dataset)
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        all_predictions = []

        for batch in loader:
            preds = self.predict(batch, return_numpy=False)
            all_predictions.append(preds)

        # Concatenate all predictions
        output = torch.cat(all_predictions, dim=0)

        if return_numpy:
            return output.cpu().numpy()
        return output

    def save_predictions(
        self,
        predictions: torch.Tensor | numpy.ndarray,
        output_path: str | Path,
        format: str = "csv",
        include_inputs: bool = False,
        input_identifiers: list[str] | None = None,
    ) -> Path:
        """
        Save predictions to file.

        PHASE 6: Output path resolved via global_paths.working_root_dir.

        DYNAMIC: Supports multiple output formats (csv, json, npy, pt)
        PRODUCTION-READY: Auto-creates parent directories, handles all data types
        FUTURE-PROOF: New formats can be added without modifying other code

        Args:
            predictions: Predictions array/tensor
            output_path: Output file path (resolved via config)
            format: Output format ('csv', 'json', 'npy', 'pt')
            include_inputs: Include input identifiers in output
            input_identifiers: List of input identifiers (e.g., SMILES)

        Returns:
            Resolved output path

        Example:
            >>> predictor.save_predictions(preds, "results.csv", format='csv')
            >>> predictor.save_predictions(preds, "results.json", format='json',
            ...                            include_inputs=True, input_identifiers=smiles_list)
        """
        import numpy as np

        # Resolve output path against working_root_dir
        resolved_path = self._resolve_path(output_path, create_parents=True)

        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        # Save based on format
        if format == "csv":
            import pandas as pd

            # Handle multi-dimensional predictions
            if predictions.ndim == 1:
                df = pd.DataFrame({"prediction": predictions})
            else:
                # Multiple columns for multi-output
                df = pd.DataFrame(
                    predictions, columns=[f"prediction_{i}" for i in range(predictions.shape[1])]
                )

            if include_inputs and input_identifiers:
                df.insert(0, "input", input_identifiers)
            df.to_csv(resolved_path, index=False)

        elif format == "json":
            import json

            output_data = {"predictions": predictions.tolist()}
            if include_inputs and input_identifiers:
                output_data["inputs"] = input_identifiers
            with open(resolved_path, "w") as f:
                json.dump(output_data, f, indent=2)

        elif format == "npy":
            np.save(resolved_path, predictions)

        elif format == "pt":
            torch.save(torch.from_numpy(predictions), resolved_path)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', 'npy', or 'pt'.")

        logger.info(f"Predictions saved to: {resolved_path}")
        return resolved_path


# Convenience function
def predict(
    checkpoint_path: str | Path,
    data: Data | Batch,
    working_root_dir: Path,
    device: torch.device | None = None,
    return_numpy: bool = False,
) -> torch.Tensor | numpy.ndarray:
    """
    Quick prediction from checkpoint.

    Dependency Injection: Requires explicit working_root_dir parameter.

    Args:
        checkpoint_path: Path to model checkpoint.
                         - Relative paths resolved against working_root_dir
                         - Searches default checkpoint directory
        data: PyG Data object
        working_root_dir: Base directory for resolving relative paths.
                          Must be provided explicitly (Dependency Injection).
        device: Target device
        return_numpy: If True, return numpy array

    Returns:
        Predictions

    Example:
        >>> from milia_pipeline.models.post_training import predict
        >>>
        >>> # Caller computes working_root_dir from config
        >>> working_root_dir = Path(config['global_paths']['working_root_dir']).expanduser()
        >>>
        >>> # Predict with explicit working_root_dir
        >>> result = predict(
        ...     "checkpoints/model.pt",
        ...     my_data,
        ...     working_root_dir=working_root_dir
        ... )
    """
    predictor = Predictor.from_checkpoint(
        checkpoint_path, working_root_dir=working_root_dir, device=device
    )
    return predictor.predict(data, return_numpy=return_numpy)
