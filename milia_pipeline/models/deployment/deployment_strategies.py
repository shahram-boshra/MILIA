"""
Deployment Strategies

Comprehensive deployment strategies for production environments.
Supports cloud, edge, federated, and serverless deployment.

Features:
- Cloud deployment (AWS SageMaker, GCP AI Platform, Azure ML)
- Edge targets (mobile, IoT): deprecated, removed in 2.0.0; they write a deployment package
- Federated learning deployment
- Serverless deployment (Lambda, Cloud Functions)
- Container-based deployment (Docker, Kubernetes)
- Model serving strategies (REST API, gRPC, batch)
- A/B testing and canary deployment
- Blue-green deployment

Pydantic V2 Migration (Phase 20):
    - Migrated DeploymentConfig from @dataclass to Pydantic BaseModel (mutable)
    - Uses model_dump() for to_dict() method (backward compatible)
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established pattern from distributed_strategies.py (Phase 8)

Author: milia Team
Version: 1.1.0
"""

import importlib.util
import logging
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel

# Re-exported: callers import these names from this module (`X as X` marks a re-export).
from milia_pipeline.exceptions import ConfigurationError as ConfigurationError
from milia_pipeline.exceptions import DeploymentError as DeploymentError
from milia_pipeline.exceptions import ModelError as ModelError
from milia_pipeline.models.deployment.package import (
    is_package,
    load_package,
    package_checkpoint,
    verify_package,
)
from milia_pipeline.models.deployment.serving import write_serving_dockerfile

logger = logging.getLogger(__name__)


# =============================================================================
# DEPLOYMENT TYPES
# =============================================================================


class DeploymentTarget(Enum):
    """Deployment target environments."""

    CLOUD_AWS = "aws"  # AWS SageMaker
    CLOUD_GCP = "gcp"  # Google Cloud AI Platform
    CLOUD_AZURE = "azure"  # Azure Machine Learning
    EDGE_MOBILE = "mobile"  # Deprecated (removed in 2.0.0): writes a deployment package
    EDGE_IOT = "iot"  # Deprecated (removed in 2.0.0): writes a deployment package
    FEDERATED = "federated"  # Federated learning
    SERVERLESS = "serverless"  # Lambda/Cloud Functions
    CONTAINER = "container"  # Docker/Kubernetes
    LOCAL = "local"  # Local server


class ServingMode(Enum):
    """Model serving modes."""

    ONLINE = "online"  # Real-time inference
    BATCH = "batch"  # Batch processing
    STREAMING = "streaming"  # Streaming inference


# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================


class DeploymentConfig(BaseModel):
    """
    Configuration for model deployment.

    Pydantic V2 Migration (Phase 20):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses model_dump() for to_dict() method (backward compatible)
        - NON-BREAKING: Same constructor API and attribute access
        - Follows established pattern from distributed_strategies.py (Phase 8)

    Attributes:
        target: Deployment target environment
        serving_mode: Model serving mode
        instance_type: Instance/VM type for cloud deployment
        num_instances: Number of instances for scaling
        auto_scaling: Enable auto-scaling
        min_instances: Minimum instances for auto-scaling
        max_instances: Maximum instances for auto-scaling
        api_type: API type (rest, grpc, batch)
        enable_monitoring: Enable deployment monitoring
        enable_logging: Enable request/response logging
        enable_caching: Enable response caching
        timeout_seconds: Request timeout in seconds
        max_batch_size: Maximum batch size for inference
        base_image: Image the generated serving Dockerfile builds on (aws, gcp, azure,
            container); default: the published MILIA image of the running version
    """

    target: str = "local"
    serving_mode: str = "online"
    instance_type: str | None = None
    num_instances: int = 1
    auto_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 10
    api_type: str = "rest"
    enable_monitoring: bool = True
    enable_logging: bool = True
    enable_caching: bool = False
    timeout_seconds: int = 30
    max_batch_size: int = 32
    base_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().
        """
        return self.model_dump()


# =============================================================================
# BASE DEPLOYMENT STRATEGY
# =============================================================================


class DeploymentStrategy(ABC):
    """
    Abstract base class for deployment strategies.

    All deployment strategies must implement:
    - prepare_model: Prepare model for deployment
    - deploy: Deploy model to target
    - predict: Make predictions
    - teardown: Clean up deployment
    """

    def __init__(self, config: DeploymentConfig, verbose: bool = True):
        """
        Initialize deployment strategy.

        Args:
            config: Deployment configuration
            verbose: Whether to log information
        """
        self.config = config
        self.verbose = verbose
        self.is_deployed = False

    @abstractmethod
    def prepare_model(self, model: nn.Module, save_path: Path) -> Path:
        """
        Prepare model for deployment.

        Args:
            model: Model to deploy
            save_path: Path to save prepared model

        Returns:
            Path to prepared model
        """
        pass

    @abstractmethod
    def deploy(self, model_path: Path) -> dict[str, Any]:
        """
        Deploy model to target environment.

        Args:
            model_path: Path to prepared model

        Returns:
            Deployment information
        """
        pass

    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Any:
        """
        Make prediction using deployed model.

        Args:
            input_data: Input data for prediction
            **kwargs: Additional arguments

        Returns:
            Prediction results
        """
        pass

    @abstractmethod
    def teardown(self):
        """Clean up deployment resources."""
        pass

    def get_deployment_info(self) -> dict[str, Any]:
        """Get deployment information."""
        return {
            "target": self.config.target,
            "serving_mode": self.config.serving_mode,
            "is_deployed": self.is_deployed,
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLOUD DEPLOYMENT STRATEGIES
# =============================================================================


def _prepare_serving_package(
    strategy: DeploymentStrategy,
    model: nn.Module | str | Path,
    save_path: Path,
    run_as_root: bool = False,
) -> Path:
    """Package a MILIA checkpoint and write the Dockerfile that serves it (aws, gcp, azure,
    container). These targets run in another process, so they need a package that rebuilds
    the model; weights alone cannot."""
    if not isinstance(model, (str, Path)):
        raise DeploymentError(
            f"The {strategy.config.target} target serves a MILIA deployment package: pass the path "
            "of a MILIA checkpoint (for example '<working_root_dir>/checkpoints/best.pt'). Model "
            "weights alone cannot rebuild the model in the serving container."
        )
    save_path = Path(save_path)
    package_checkpoint(model, save_path)
    write_serving_dockerfile(
        save_path, base_image=strategy.config.base_image, run_as_root=run_as_root
    )
    if strategy.verbose:
        logger.info(f"Prepared {strategy.config.target} serving image context: {save_path}")
    return save_path


_SERVING_REQUEST = '{"instances": ["CCO", "c1ccccc1"]}'


def _serving_predict_error(strategy: DeploymentStrategy, how: str) -> DeploymentError:
    """Error for ``predict`` on a target whose ``deploy`` only prepares a serving image.

    MILIA creates no endpoint for these targets, so it has nothing to call; returning
    ``None`` would look like a prediction. ``how`` names the endpoint and route to call.
    """
    if not strategy.is_deployed:
        return DeploymentError("Model not deployed. Call deploy() first.")
    return DeploymentError(
        f"The {strategy.config.target} target prepares a serving image; MILIA does not create or "
        f"call its endpoint, so predict() has no result to return. {how} with the request body "
        f"{_SERVING_REQUEST} (one row of predictions per SMILES). For in-process predictions, "
        "use the 'local' target."
    )


class AWSDeploymentStrategy(DeploymentStrategy):
    """
    Deployment strategy for AWS SageMaker.

    Usage:
        >>> strategy = AWSDeploymentStrategy(config)
        >>> model_path = strategy.prepare_model(model, save_path)
        >>> deployment_info = strategy.deploy(model_path)  # status "prepared"

    Predictions come from the SageMaker endpoint created from the image, not from ``predict``.
    """

    def prepare_model(self, model: nn.Module | str | Path, save_path: Path) -> Path:
        """Package a MILIA checkpoint with a SageMaker-compatible serving Dockerfile.

        The image answers ``GET /ping`` and ``POST /invocations`` on port 8080 and accepts the
        ``serve`` argument SageMaker starts it with; it runs as root, as SageMaker expects.
        Push the built image to ECR; the package files may also be uploaded as ``model.tar.gz``
        (SageMaker extracts them to ``/opt/ml/model``).
        """
        return _prepare_serving_package(self, model, save_path, run_as_root=True)

    def deploy(self, model_path: Path) -> dict[str, Any]:
        """Deploy to AWS SageMaker."""
        try:
            _boto3_available = importlib.util.find_spec("boto3") is not None
        except ValueError:
            # find_spec raises ValueError if module is in sys.modules but
            # __spec__ is not set or is None (documented CPython behavior).
            # Being in sys.modules means it was imported → available.
            _boto3_available = True

        try:
            _sagemaker_available = importlib.util.find_spec("sagemaker") is not None
        except ValueError:
            _sagemaker_available = True

        if not (_boto3_available and _sagemaker_available):
            raise DeploymentError("AWS deployment requires: pip install boto3 sagemaker")

        # This is a placeholder - actual implementation requires AWS credentials
        logger.warning(
            "AWS deployment requires valid AWS credentials and SageMaker setup. "
            "This is a template implementation."
        )

        verify_package(model_path)
        deployment_info = {
            "target": "aws",
            "model_path": str(model_path),
            "status": "prepared",
            "message": "Serving image context prepared for SageMaker: build it, push it to ECR "
            "and create the endpoint with the AWS SDK.",
        }

        self.is_deployed = True
        return deployment_info

    def predict(self, input_data: Any, **kwargs) -> Any:
        """Not run by MILIA: call the SageMaker endpoint created from the image.

        Raises:
            DeploymentError: Always; names the SageMaker call and request body.
        """
        raise _serving_predict_error(
            self,
            "Invoke the SageMaker endpoint you create from the image (SageMaker Runtime "
            "InvokeEndpoint, ContentType application/json; the container serves POST /invocations)",
        )

    def teardown(self):
        """Delete SageMaker endpoint."""
        if self.is_deployed:
            logger.info("Tearing down AWS deployment")
            self.is_deployed = False


class GCPDeploymentStrategy(DeploymentStrategy):
    """Deployment strategy for Google Cloud AI Platform."""

    def prepare_model(self, model: nn.Module | str | Path, save_path: Path) -> Path:
        """Package a MILIA checkpoint with a Vertex AI custom-container serving Dockerfile.

        The server listens on ``AIP_HTTP_PORT`` and adds ``AIP_HEALTH_ROUTE`` and
        ``AIP_PREDICT_ROUTE`` when Vertex AI sets them.
        """
        return _prepare_serving_package(self, model, save_path)

    def deploy(self, model_path: Path) -> dict[str, Any]:
        """Deploy to GCP AI Platform."""
        logger.warning("GCP deployment requires GCP credentials and AI Platform setup.")

        verify_package(model_path)
        deployment_info = {"target": "gcp", "model_path": str(model_path), "status": "prepared"}

        self.is_deployed = True
        return deployment_info

    def predict(self, input_data: Any, **kwargs) -> Any:
        """Not run by MILIA: call the Vertex AI endpoint created from the image.

        Raises:
            DeploymentError: Always; names the Vertex AI call and request body.
        """
        raise _serving_predict_error(
            self,
            "Call predict on the Vertex AI endpoint you deploy the image to (the container serves "
            "AIP_PREDICT_ROUTE)",
        )

    def teardown(self):
        """Delete GCP deployment."""
        if self.is_deployed:
            logger.info("Tearing down GCP deployment")
            self.is_deployed = False


class AzureDeploymentStrategy(DeploymentStrategy):
    """Deployment strategy for Azure Machine Learning."""

    def prepare_model(self, model: nn.Module | str | Path, save_path: Path) -> Path:
        """Package a MILIA checkpoint with a serving Dockerfile for an Azure ML custom container
        (probes: ``GET /health``; scoring: ``POST /predict``, port 8080)."""
        return _prepare_serving_package(self, model, save_path)

    def deploy(self, model_path: Path) -> dict[str, Any]:
        """Deploy to Azure ML."""
        logger.warning("Azure deployment requires Azure credentials and ML workspace setup.")

        verify_package(model_path)
        deployment_info = {"target": "azure", "model_path": str(model_path), "status": "prepared"}

        self.is_deployed = True
        return deployment_info

    def predict(self, input_data: Any, **kwargs) -> Any:
        """Not run by MILIA: call the Azure ML online endpoint created from the image.

        Raises:
            DeploymentError: Always; names the Azure ML call and request body.
        """
        raise _serving_predict_error(
            self,
            "POST to the scoring URI of the Azure ML online endpoint you create from the image "
            "(the container serves POST /predict on port 8080)",
        )

    def teardown(self):
        """Delete Azure deployment."""
        if self.is_deployed:
            logger.info("Tearing down Azure deployment")
            self.is_deployed = False


# =============================================================================
# EDGE DEPLOYMENT STRATEGIES
# =============================================================================


EDGE_TARGET_DEPRECATION = (
    "The 'mobile' and 'iot' deployment targets (EdgeDeploymentStrategy) are deprecated and "
    "will be removed in MILIA 2.0.0. MILIA builds no on-device model: featurization needs RDKit "
    "and the models' graph pooling is data-dependent, so torch.export cannot export them for "
    "on-device runtimes. Use the 'container' target and "
    "serve the package over HTTP (python -m milia_pipeline.models.deployment.serving), or the "
    "'local' target in-process."
)


class EdgeDeploymentStrategy(DeploymentStrategy):
    """Deprecated target for edge devices (``mobile``, ``iot``); removed in MILIA 2.0.0.

    MILIA does not build an on-device model. Featurizing a molecule needs RDKit. PyTorch's
    TorchScript lite interpreter is deprecated in favour of ExecuTorch, which consumes
    ``torch.export`` programs, and MILIA's graph pooling is data-dependent, so its models do
    not export. ``prepare_model`` writes the same verified deployment package as the other
    targets; serve it with :mod:`.serving` (``container`` target) or load it with the ``local``
    target.
    """

    def __init__(self, config: DeploymentConfig, verbose: bool = True):
        """Initialize the deprecated edge strategy (emits ``DeprecationWarning``)."""
        warnings.warn(EDGE_TARGET_DEPRECATION, DeprecationWarning, stacklevel=2)
        logger.warning(EDGE_TARGET_DEPRECATION)
        super().__init__(config, verbose)

    def prepare_model(self, model: nn.Module | str | Path, save_path: Path) -> Path:
        """Write a deployment package for a MILIA checkpoint (no device-specific format)."""
        if not isinstance(model, (str, Path)):
            raise DeploymentError(
                f"The {self.config.target} target writes a MILIA deployment package: pass the path "
                "of a MILIA checkpoint (for example '<working_root_dir>/checkpoints/best.pt'). "
                "Model weights alone cannot rebuild the model."
            )
        save_path = Path(save_path)
        package_checkpoint(model, save_path)
        if self.verbose:
            logger.info(f"Prepared {self.config.target} deployment package: {save_path}")
        return save_path

    def deploy(self, model_path: Path) -> dict[str, Any]:
        """Verify the package; MILIA does not install it on a device."""
        verify_package(model_path)
        deployment_info = {
            "target": self.config.target,
            "model_path": str(model_path),
            "status": "prepared",
            "message": EDGE_TARGET_DEPRECATION,
        }

        self.is_deployed = True
        return deployment_info

    def predict(self, input_data: Any, **kwargs) -> Any:
        """Not supported: MILIA runs no model on the device."""
        if not self.is_deployed:
            raise DeploymentError("Model not deployed")
        raise DeploymentError(
            f"The {self.config.target} target does not run predictions. Serve the package "
            "(container target) or deploy it with the local target."
        )

    def teardown(self):
        """Clean up edge deployment."""
        if self.is_deployed:
            logger.info("Edge deployment teardown")
            self.is_deployed = False


# =============================================================================
# CONTAINER DEPLOYMENT
# =============================================================================


class ContainerDeploymentStrategy(DeploymentStrategy):
    """
    Deployment strategy for containers (Docker, Kubernetes).

    Creates Docker images and Kubernetes manifests.
    """

    def prepare_model(self, model: nn.Module | str | Path, save_path: Path) -> Path:
        """Package a MILIA checkpoint with a serving Dockerfile (Docker, Kubernetes)."""
        return _prepare_serving_package(self, model, save_path)

    def deploy(self, model_path: Path) -> dict[str, Any]:
        """Deploy container."""
        verify_package(model_path)
        deployment_info = {
            "target": "container",
            "model_path": str(model_path),
            "status": "ready_for_docker",
            "message": f"Build Docker image: docker build -t milia-serve {model_path}",
            "run_command": "docker run -p 8080:8080 milia-serve",
        }

        self.is_deployed = True

        if self.verbose:
            logger.info("Container deployment prepared")
            logger.info(f"Build: docker build -t milia-serve {model_path}")
            logger.info("Run: docker run -p 8080:8080 milia-serve")

        return deployment_info

    def predict(self, input_data: Any, **kwargs) -> Any:
        """Not run by MILIA: call the running container.

        Raises:
            DeploymentError: Always; names the container route and request body.
        """
        raise _serving_predict_error(
            self,
            "Build and run the image (docker build -t milia-serve <package>; docker run -p "
            "8080:8080 milia-serve), then POST to http://<host>:8080/predict",
        )

    def teardown(self):
        """Stop container."""
        if self.is_deployed:
            logger.info("Container deployment teardown")
            self.is_deployed = False


# =============================================================================
# LOCAL DEPLOYMENT
# =============================================================================


class LocalDeploymentStrategy(DeploymentStrategy):
    """Local deployment strategy for development/testing.

    Two sources are supported:

    * a MILIA checkpoint path: ``prepare_model`` writes a self-describing deployment package
      (``model.pt`` + ``deployment.json``, see :mod:`.package`) and ``deploy`` rebuilds the
      model from it with ``ModelLoader``, in this or any later process;
    * an ``nn.Module`` (for example one returned by ``optimize_model_for_deployment``):
      ``prepare_model`` saves its weights to ``model.pth`` and keeps the module, and ``deploy``
      loads those weights back into it. Weights alone cannot rebuild a model, so such a
      directory can be deployed only by the strategy that prepared it.
    """

    def __init__(self, config: DeploymentConfig, verbose: bool = True):
        """Initialize local deployment."""
        super().__init__(config, verbose)
        self.model = None
        self._module: nn.Module | None = None

    def prepare_model(self, model: nn.Module | str | Path, save_path: Path) -> Path:
        """Prepare a module or a MILIA checkpoint for local deployment."""
        save_path = Path(save_path)

        if isinstance(model, (str, Path)):
            package_checkpoint(model, save_path)
            if self.verbose:
                logger.info(f"Prepared deployment package from {model}: {save_path}")
            return save_path

        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "model.pth"
        torch.save(model.state_dict(), model_file)
        self._module = model

        if self.verbose:
            logger.info(f"Prepared model for local deployment: {model_file}")

        return save_path

    def deploy(self, model_path: Path) -> dict[str, Any]:
        """Deploy locally (load model into memory)."""
        model_path = Path(model_path)

        if is_package(model_path):
            self.model, _ = load_package(model_path)
        else:
            model_file = model_path / "model.pth"
            if not model_file.exists():
                raise DeploymentError(f"Model file not found: {model_file}")
            if self._module is None:
                raise DeploymentError(
                    f"{model_file} holds only weights, which cannot rebuild a model. Deploy it "
                    "with the strategy that prepared it, or prepare a package from a MILIA "
                    "checkpoint: prepare_model('<working_root_dir>/checkpoints/best.pt', path)."
                )
            self._module.load_state_dict(torch.load(model_file, weights_only=True))
            self.model = self._module.eval()

        deployment_info = {"target": "local", "model_path": str(model_path), "status": "deployed"}

        self.is_deployed = True

        if self.verbose:
            logger.info("Model deployed locally")

        return deployment_info

    def predict(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Make prediction using local model."""
        if not self.is_deployed or self.model is None:
            raise DeploymentError("Model not deployed. Call deploy() first.")

        with torch.no_grad():
            output = self.model(input_data)

        return output

    def teardown(self):
        """Clean up local deployment."""
        if self.is_deployed:
            self.model = None
            self.is_deployed = False
            if self.verbose:
                logger.info("Local deployment cleaned up")


# =============================================================================
# DEPLOYMENT MANAGER
# =============================================================================


class DeploymentManager:
    """
    Manager for model deployment strategies.

    Provides unified interface for deploying to various targets.

    Usage:
        >>> config = DeploymentConfig(target="local")
        >>> manager = DeploymentManager(config)
        >>>
        >>> # Prepare and deploy
        >>> model_path = manager.prepare_model(model, save_path)
        >>> deployment_info = manager.deploy(model_path)
        >>>
        >>> # Make predictions
        >>> result = manager.predict(input_data)
    """

    _strategies = {
        "aws": AWSDeploymentStrategy,
        "gcp": GCPDeploymentStrategy,
        "azure": AzureDeploymentStrategy,
        "mobile": EdgeDeploymentStrategy,
        "iot": EdgeDeploymentStrategy,
        "container": ContainerDeploymentStrategy,
        "local": LocalDeploymentStrategy,
    }

    def __init__(
        self,
        config: DeploymentConfig | None = None,
        target: str | None = None,
        verbose: bool = True,
    ):
        """
        Initialize deployment manager.

        Args:
            config: Deployment configuration
            target: Deployment target (alternative to config)
            verbose: Whether to log information
        """
        if config is None:
            config = DeploymentConfig(target=target or "local")

        self.config = config
        self.verbose = verbose

        # Create strategy
        strategy_class = self._strategies.get(config.target)
        if strategy_class is None:
            raise ConfigurationError(
                f"Unknown deployment target: {config.target}. "
                f"Available: {list(self._strategies.keys())}"
            )

        self.strategy = strategy_class(config, verbose)

        if self.verbose:
            logger.info(f"DeploymentManager initialized - Target: {config.target}")

    def prepare_model(self, model: nn.Module | str | Path, save_path: str | Path) -> Path:
        """
        Prepare model for deployment.

        Args:
            model: Model to deploy. The local target also takes the path of a MILIA checkpoint
                (for example ``{working_root_dir}/checkpoints/best.pt``) and writes a deployment
                package that rebuilds the model in any process.
            save_path: Path to save prepared model

        Returns:
            Path to prepared model
        """
        return self.strategy.prepare_model(model, Path(save_path))

    def deploy(self, model_path: str | Path) -> dict[str, Any]:
        """
        Deploy model to target environment.

        Args:
            model_path: Path to prepared model

        Returns:
            Deployment information
        """
        return self.strategy.deploy(Path(model_path))

    def predict(self, input_data: Any, **kwargs) -> Any:
        """
        Make prediction using deployed model.

        Args:
            input_data: Input data
            **kwargs: Additional arguments

        Returns:
            Prediction results
        """
        return self.strategy.predict(input_data, **kwargs)

    def teardown(self):
        """Clean up deployment resources."""
        self.strategy.teardown()

    def get_deployment_info(self) -> dict[str, Any]:
        """Get deployment information."""
        return self.strategy.get_deployment_info()

    @classmethod
    def list_available_targets(cls) -> list[str]:
        """List available deployment targets."""
        return list(cls._strategies.keys())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def deploy_locally(model: nn.Module | str | Path, save_path: str | Path) -> DeploymentManager:
    """
    Quick local deployment.

    Args:
        model: Model to deploy, or the path of a MILIA checkpoint to package
        save_path: Path to save model

    Returns:
        DeploymentManager instance
    """
    manager = DeploymentManager(target="local", verbose=True)
    model_path = manager.prepare_model(model, save_path)
    manager.deploy(model_path)
    return manager


def list_deployment_targets() -> list[str]:
    """List available deployment targets."""
    return DeploymentManager.list_available_targets()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("deployment_strategies module loaded")
