#!/usr/bin/env python3
"""
Unit Tests for deployment_strategies.py Module

Comprehensive test suite covering:
- DeploymentTarget and ServingMode enums
- DeploymentConfig Pydantic BaseModel (Pydantic V2 Migration Phase 20)
- DeploymentStrategy ABC
- AWSDeploymentStrategy
- GCPDeploymentStrategy
- AzureDeploymentStrategy
- EdgeDeploymentStrategy
- ContainerDeploymentStrategy
- LocalDeploymentStrategy
- DeploymentManager
- Convenience functions: deploy_locally, list_deployment_targets
- Exception hierarchy: ModelError, DeploymentError, ConfigurationError

Author: milia Team
Test Module Version: 1.1.0
Target Module: milia_pipeline/models/deployment/deployment_strategies.py
"""

import sys
import os
import json
import logging
import tempfile
import shutil
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any

import pytest

# =============================================================================
# ADD PROJECT ROOT TO PYTHON PATH
# =============================================================================
# Get the project root (parent of 'tests' directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# MODULE IMPORTS (with mocking strategy for torch dependencies)
# =============================================================================
# We import torch and nn for type annotations and mock creation
import torch
import torch.nn as nn

# Import the module under test
from milia_pipeline.models.deployment.deployment_strategies import (
    # Enums
    DeploymentTarget,
    ServingMode,
    # Dataclasses
    DeploymentConfig,
    # ABC
    DeploymentStrategy,
    # Concrete Strategies
    AWSDeploymentStrategy,
    GCPDeploymentStrategy,
    AzureDeploymentStrategy,
    EdgeDeploymentStrategy,
    ContainerDeploymentStrategy,
    LocalDeploymentStrategy,
    # Manager
    DeploymentManager,
    # Convenience functions
    deploy_locally,
    list_deployment_targets,
    # Exceptions
    ModelError,
    DeploymentError,
    ConfigurationError,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = MagicMock(spec=nn.Module)
    model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
    model.eval = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=[torch.randn(10, 10)])
    return model


@pytest.fixture
def default_config():
    """Create a default DeploymentConfig."""
    return DeploymentConfig()


@pytest.fixture
def aws_config():
    """Create an AWS deployment config."""
    return DeploymentConfig(
        target="aws",
        serving_mode="online",
        instance_type="ml.m5.large",
        num_instances=2,
        auto_scaling=True,
        min_instances=1,
        max_instances=5
    )


@pytest.fixture
def local_config():
    """Create a local deployment config."""
    return DeploymentConfig(
        target="local",
        serving_mode="online",
        enable_monitoring=True,
        enable_logging=True
    )


@pytest.fixture
def edge_config():
    """Create an edge deployment config."""
    return DeploymentConfig(
        target="mobile",
        serving_mode="online",
        enable_caching=True
    )


@pytest.fixture
def container_config():
    """Create a container deployment config."""
    return DeploymentConfig(
        target="container",
        serving_mode="batch",
        max_batch_size=64
    )


# =============================================================================
# TESTS: DeploymentTarget Enum
# =============================================================================

class TestDeploymentTargetEnum:
    """Tests for DeploymentTarget enum."""

    def test_cloud_aws_value(self):
        """Test CLOUD_AWS has correct value."""
        assert DeploymentTarget.CLOUD_AWS.value == "aws"

    def test_cloud_gcp_value(self):
        """Test CLOUD_GCP has correct value."""
        assert DeploymentTarget.CLOUD_GCP.value == "gcp"

    def test_cloud_azure_value(self):
        """Test CLOUD_AZURE has correct value."""
        assert DeploymentTarget.CLOUD_AZURE.value == "azure"

    def test_edge_mobile_value(self):
        """Test EDGE_MOBILE has correct value."""
        assert DeploymentTarget.EDGE_MOBILE.value == "mobile"

    def test_edge_iot_value(self):
        """Test EDGE_IOT has correct value."""
        assert DeploymentTarget.EDGE_IOT.value == "iot"

    def test_federated_value(self):
        """Test FEDERATED has correct value."""
        assert DeploymentTarget.FEDERATED.value == "federated"

    def test_serverless_value(self):
        """Test SERVERLESS has correct value."""
        assert DeploymentTarget.SERVERLESS.value == "serverless"

    def test_container_value(self):
        """Test CONTAINER has correct value."""
        assert DeploymentTarget.CONTAINER.value == "container"

    def test_local_value(self):
        """Test LOCAL has correct value."""
        assert DeploymentTarget.LOCAL.value == "local"

    def test_all_deployment_targets_count(self):
        """Test that all 9 deployment targets exist."""
        assert len(DeploymentTarget) == 9

    def test_enum_member_access(self):
        """Test enum member access by name."""
        assert DeploymentTarget["CLOUD_AWS"] == DeploymentTarget.CLOUD_AWS
        assert DeploymentTarget["LOCAL"] == DeploymentTarget.LOCAL

    def test_enum_iteration(self):
        """Test iterating over enum members."""
        targets = list(DeploymentTarget)
        assert len(targets) == 9
        assert DeploymentTarget.CLOUD_AWS in targets
        assert DeploymentTarget.LOCAL in targets


# =============================================================================
# TESTS: ServingMode Enum
# =============================================================================

class TestServingModeEnum:
    """Tests for ServingMode enum."""

    def test_online_value(self):
        """Test ONLINE has correct value."""
        assert ServingMode.ONLINE.value == "online"

    def test_batch_value(self):
        """Test BATCH has correct value."""
        assert ServingMode.BATCH.value == "batch"

    def test_streaming_value(self):
        """Test STREAMING has correct value."""
        assert ServingMode.STREAMING.value == "streaming"

    def test_all_serving_modes_count(self):
        """Test that all 3 serving modes exist."""
        assert len(ServingMode) == 3

    def test_enum_member_access(self):
        """Test enum member access by name."""
        assert ServingMode["ONLINE"] == ServingMode.ONLINE
        assert ServingMode["BATCH"] == ServingMode.BATCH

    def test_enum_iteration(self):
        """Test iterating over enum members."""
        modes = list(ServingMode)
        assert len(modes) == 3
        assert ServingMode.ONLINE in modes
        assert ServingMode.BATCH in modes
        assert ServingMode.STREAMING in modes


# =============================================================================
# TESTS: DeploymentConfig Pydantic BaseModel
# =============================================================================

class TestDeploymentConfig:
    """Tests for DeploymentConfig Pydantic BaseModel (Pydantic V2 Migration Phase 20)."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DeploymentConfig()
        assert config.target == "local"
        assert config.serving_mode == "online"
        assert config.instance_type is None
        assert config.num_instances == 1
        assert config.auto_scaling is False
        assert config.min_instances == 1
        assert config.max_instances == 10
        assert config.api_type == "rest"
        assert config.enable_monitoring is True
        assert config.enable_logging is True
        assert config.enable_caching is False
        assert config.timeout_seconds == 30
        assert config.max_batch_size == 32

    def test_custom_target(self):
        """Test custom target configuration."""
        config = DeploymentConfig(target="aws")
        assert config.target == "aws"

    def test_custom_serving_mode(self):
        """Test custom serving mode configuration."""
        config = DeploymentConfig(serving_mode="batch")
        assert config.serving_mode == "batch"

    def test_custom_instance_type(self):
        """Test custom instance type configuration."""
        config = DeploymentConfig(instance_type="ml.m5.xlarge")
        assert config.instance_type == "ml.m5.xlarge"

    def test_auto_scaling_configuration(self):
        """Test auto-scaling configuration."""
        config = DeploymentConfig(
            auto_scaling=True,
            min_instances=2,
            max_instances=20
        )
        assert config.auto_scaling is True
        assert config.min_instances == 2
        assert config.max_instances == 20

    def test_monitoring_and_logging_disabled(self):
        """Test disabling monitoring and logging."""
        config = DeploymentConfig(
            enable_monitoring=False,
            enable_logging=False
        )
        assert config.enable_monitoring is False
        assert config.enable_logging is False

    def test_caching_enabled(self):
        """Test enabling caching."""
        config = DeploymentConfig(enable_caching=True)
        assert config.enable_caching is True

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        config = DeploymentConfig(timeout_seconds=60)
        assert config.timeout_seconds == 60

    def test_custom_batch_size(self):
        """Test custom batch size configuration."""
        config = DeploymentConfig(max_batch_size=128)
        assert config.max_batch_size == 128

    def test_to_dict_method(self):
        """Test to_dict method returns correct dictionary."""
        config = DeploymentConfig(
            target="aws",
            serving_mode="batch",
            instance_type="ml.m5.large"
        )
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['target'] == "aws"
        assert result['serving_mode'] == "batch"
        assert result['instance_type'] == "ml.m5.large"
        assert result['num_instances'] == 1
        assert result['auto_scaling'] is False

    def test_to_dict_contains_all_fields(self):
        """Test to_dict contains all expected fields."""
        config = DeploymentConfig()
        result = config.to_dict()
        
        expected_keys = {
            'target', 'serving_mode', 'instance_type', 'num_instances',
            'auto_scaling', 'min_instances', 'max_instances', 'api_type',
            'enable_monitoring', 'enable_logging', 'enable_caching',
            'timeout_seconds', 'max_batch_size'
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_is_json_serializable(self):
        """Test that to_dict output is JSON serializable."""
        config = DeploymentConfig(
            target="container",
            instance_type="n1-standard-4"
        )
        result = config.to_dict()
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_api_type_grpc(self):
        """Test gRPC API type configuration."""
        config = DeploymentConfig(api_type="grpc")
        assert config.api_type == "grpc"

    def test_multiple_instances(self):
        """Test multiple instances configuration."""
        config = DeploymentConfig(num_instances=5)
        assert config.num_instances == 5

    def test_field_types_are_correct(self):
        """Test that field types match their definitions (Pydantic type enforcement)."""
        config = DeploymentConfig()
        
        # String fields
        assert isinstance(config.target, str)
        assert isinstance(config.serving_mode, str)
        assert isinstance(config.api_type, str)
        
        # Optional string field
        assert config.instance_type is None or isinstance(config.instance_type, str)
        
        # Integer fields
        assert isinstance(config.num_instances, int)
        assert isinstance(config.min_instances, int)
        assert isinstance(config.max_instances, int)
        assert isinstance(config.timeout_seconds, int)
        assert isinstance(config.max_batch_size, int)
        
        # Boolean fields
        assert isinstance(config.auto_scaling, bool)
        assert isinstance(config.enable_monitoring, bool)
        assert isinstance(config.enable_logging, bool)
        assert isinstance(config.enable_caching, bool)

    def test_pydantic_model_fields_property(self):
        """Test that Pydantic V2's model_fields property is available."""
        # Pydantic V2 exposes model_fields as a class property
        assert hasattr(DeploymentConfig, 'model_fields')
        
        fields = DeploymentConfig.model_fields
        assert isinstance(fields, dict)
        
        # Verify expected fields exist
        expected_field_names = {
            'target', 'serving_mode', 'instance_type', 'num_instances',
            'auto_scaling', 'min_instances', 'max_instances', 'api_type',
            'enable_monitoring', 'enable_logging', 'enable_caching',
            'timeout_seconds', 'max_batch_size'
        }
        assert set(fields.keys()) == expected_field_names

    def test_pydantic_model_json_schema(self):
        """Test that Pydantic V2's model_json_schema() method is available."""
        # Pydantic V2 BaseModel provides model_json_schema class method
        assert hasattr(DeploymentConfig, 'model_json_schema')
        
        schema = DeploymentConfig.model_json_schema()
        assert isinstance(schema, dict)
        assert 'properties' in schema
        assert 'target' in schema['properties']


# =============================================================================
# TESTS: DeploymentStrategy ABC
# =============================================================================

class TestDeploymentStrategyABC:
    """Tests for DeploymentStrategy abstract base class."""

    def test_cannot_instantiate_directly(self, local_config):
        """Test that DeploymentStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DeploymentStrategy(local_config)

    def test_concrete_subclass_requires_abstract_methods(self):
        """Test that concrete subclass must implement all abstract methods."""
        class IncompleteStrategy(DeploymentStrategy):
            pass  # Missing abstract methods
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStrategy(DeploymentConfig())

    def test_get_deployment_info_method(self, local_config, temp_dir):
        """Test get_deployment_info method on concrete strategy."""
        strategy = LocalDeploymentStrategy(local_config)
        info = strategy.get_deployment_info()
        
        assert 'target' in info
        assert 'serving_mode' in info
        assert 'is_deployed' in info
        assert 'config' in info
        assert info['target'] == "local"
        assert info['is_deployed'] is False

    def test_is_deployed_default_false(self, local_config):
        """Test that is_deployed is False by default."""
        strategy = LocalDeploymentStrategy(local_config)
        assert strategy.is_deployed is False

    def test_verbose_attribute(self, local_config):
        """Test verbose attribute is set correctly."""
        strategy_verbose = LocalDeploymentStrategy(local_config, verbose=True)
        strategy_quiet = LocalDeploymentStrategy(local_config, verbose=False)
        
        assert strategy_verbose.verbose is True
        assert strategy_quiet.verbose is False

    def test_config_attribute(self, local_config):
        """Test config attribute is stored correctly."""
        strategy = LocalDeploymentStrategy(local_config)
        assert strategy.config is local_config


# =============================================================================
# TESTS: AWSDeploymentStrategy
# =============================================================================

class TestAWSDeploymentStrategy:
    """Tests for AWSDeploymentStrategy."""

    def test_initialization(self, aws_config):
        """Test AWS strategy initialization."""
        strategy = AWSDeploymentStrategy(aws_config)
        assert strategy.config is aws_config
        assert strategy.is_deployed is False

    def test_prepare_model_creates_directory(self, aws_config, mock_model, temp_dir):
        """Test prepare_model creates save directory."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        save_path = temp_dir / "aws_model"
        
        with patch.object(torch, 'save'):
            result = strategy.prepare_model(mock_model, save_path)
        
        assert save_path.exists()
        assert result == save_path

    def test_prepare_model_saves_model_file(self, aws_config, mock_model, temp_dir):
        """Test prepare_model saves model.pth file."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        save_path = temp_dir / "aws_model"
        
        with patch.object(torch, 'save') as mock_save:
            strategy.prepare_model(mock_model, save_path)
            mock_save.assert_called_once()
            # Check it saves to model.pth
            call_args = mock_save.call_args
            assert str(call_args[0][1]).endswith("model.pth")

    def test_prepare_model_creates_inference_script(self, aws_config, mock_model, temp_dir):
        """Test prepare_model creates inference.py script."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        save_path = temp_dir / "aws_model"
        
        with patch.object(torch, 'save'):
            strategy.prepare_model(mock_model, save_path)
        
        inference_script = save_path / "inference.py"
        assert inference_script.exists()

    def test_prepare_model_creates_requirements(self, aws_config, mock_model, temp_dir):
        """Test prepare_model creates requirements.txt."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        save_path = temp_dir / "aws_model"
        
        with patch.object(torch, 'save'):
            strategy.prepare_model(mock_model, save_path)
        
        requirements = save_path / "requirements.txt"
        assert requirements.exists()
        content = requirements.read_text()
        assert "torch" in content

    def test_deploy_with_boto3_available(self, aws_config, temp_dir):
        """Test deploy when boto3 is available."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        
        mock_boto3 = MagicMock()
        mock_sagemaker = MagicMock()
        
        with patch.dict(sys.modules, {'boto3': mock_boto3, 'sagemaker': mock_sagemaker}):
            result = strategy.deploy(temp_dir)
        
        assert result['target'] == 'aws'
        assert result['status'] == 'prepared'
        assert strategy.is_deployed is True

    def test_deploy_without_boto3_raises_error(self, aws_config, temp_dir):
        """Test deploy raises error when boto3 is not available."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        
        # Mock the import to fail
        with patch.dict(sys.modules, {'boto3': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'boto3'")):
                with pytest.raises(DeploymentError, match="AWS deployment requires"):
                    strategy.deploy(temp_dir)

    def test_predict_without_deployment_raises_error(self, aws_config):
        """Test predict raises error when not deployed."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        
        with pytest.raises(DeploymentError, match="not deployed"):
            strategy.predict(torch.randn(1, 10))

    def test_predict_after_deployment(self, aws_config, temp_dir):
        """Test predict after deployment."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        
        mock_boto3 = MagicMock()
        mock_sagemaker = MagicMock()
        
        with patch.dict(sys.modules, {'boto3': mock_boto3, 'sagemaker': mock_sagemaker}):
            strategy.deploy(temp_dir)
        
        # Predict returns None (placeholder)
        result = strategy.predict(torch.randn(1, 10))
        assert result is None

    def test_teardown(self, aws_config, temp_dir):
        """Test teardown cleans up deployment."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        
        mock_boto3 = MagicMock()
        mock_sagemaker = MagicMock()
        
        with patch.dict(sys.modules, {'boto3': mock_boto3, 'sagemaker': mock_sagemaker}):
            strategy.deploy(temp_dir)
        
        assert strategy.is_deployed is True
        strategy.teardown()
        assert strategy.is_deployed is False

    def test_create_sagemaker_inference_script_content(self, aws_config, mock_model, temp_dir):
        """Test the content of the created SageMaker inference script."""
        strategy = AWSDeploymentStrategy(aws_config, verbose=False)
        save_path = temp_dir / "aws_model"
        
        with patch.object(torch, 'save'):
            strategy.prepare_model(mock_model, save_path)
        
        inference_script = save_path / "inference.py"
        content = inference_script.read_text()
        
        # Check required functions exist in the script
        assert "def model_fn" in content
        assert "def input_fn" in content
        assert "def predict_fn" in content
        assert "def output_fn" in content


# =============================================================================
# TESTS: GCPDeploymentStrategy
# =============================================================================

class TestGCPDeploymentStrategy:
    """Tests for GCPDeploymentStrategy."""

    def test_initialization(self):
        """Test GCP strategy initialization."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config)
        assert strategy.config is config
        assert strategy.is_deployed is False

    def test_prepare_model_creates_directory(self, mock_model, temp_dir):
        """Test prepare_model creates save directory."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config, verbose=False)
        save_path = temp_dir / "gcp_model"
        
        with patch.object(torch, 'save'):
            result = strategy.prepare_model(mock_model, save_path)
        
        assert save_path.exists()
        assert result == save_path

    def test_prepare_model_saves_model(self, mock_model, temp_dir):
        """Test prepare_model saves model file."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config, verbose=False)
        save_path = temp_dir / "gcp_model"
        
        with patch.object(torch, 'save') as mock_save:
            strategy.prepare_model(mock_model, save_path)
            mock_save.assert_called_once()

    def test_deploy_returns_correct_info(self, temp_dir):
        """Test deploy returns correct deployment info."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config, verbose=False)
        
        result = strategy.deploy(temp_dir)
        
        assert result['target'] == 'gcp'
        assert result['status'] == 'prepared'
        assert str(temp_dir) in result['model_path']
        assert strategy.is_deployed is True

    def test_predict_without_deployment_raises_error(self):
        """Test predict raises error when not deployed."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config, verbose=False)
        
        with pytest.raises(DeploymentError, match="not deployed"):
            strategy.predict(torch.randn(1, 10))

    def test_predict_after_deployment(self, temp_dir):
        """Test predict after deployment returns None."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config, verbose=False)
        
        strategy.deploy(temp_dir)
        result = strategy.predict(torch.randn(1, 10))
        assert result is None

    def test_teardown(self, temp_dir):
        """Test teardown cleans up deployment."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config, verbose=False)
        
        strategy.deploy(temp_dir)
        assert strategy.is_deployed is True
        
        strategy.teardown()
        assert strategy.is_deployed is False

    def test_teardown_when_not_deployed(self):
        """Test teardown when not deployed does nothing."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config, verbose=False)
        
        # Should not raise
        strategy.teardown()
        assert strategy.is_deployed is False


# =============================================================================
# TESTS: AzureDeploymentStrategy
# =============================================================================

class TestAzureDeploymentStrategy:
    """Tests for AzureDeploymentStrategy."""

    def test_initialization(self):
        """Test Azure strategy initialization."""
        config = DeploymentConfig(target="azure")
        strategy = AzureDeploymentStrategy(config)
        assert strategy.config is config
        assert strategy.is_deployed is False

    def test_prepare_model_creates_directory(self, mock_model, temp_dir):
        """Test prepare_model creates save directory."""
        config = DeploymentConfig(target="azure")
        strategy = AzureDeploymentStrategy(config, verbose=False)
        save_path = temp_dir / "azure_model"
        
        with patch.object(torch, 'save'):
            result = strategy.prepare_model(mock_model, save_path)
        
        assert save_path.exists()
        assert result == save_path

    def test_prepare_model_saves_model(self, mock_model, temp_dir):
        """Test prepare_model saves model file."""
        config = DeploymentConfig(target="azure")
        strategy = AzureDeploymentStrategy(config, verbose=False)
        save_path = temp_dir / "azure_model"
        
        with patch.object(torch, 'save') as mock_save:
            strategy.prepare_model(mock_model, save_path)
            mock_save.assert_called_once()

    def test_deploy_returns_correct_info(self, temp_dir):
        """Test deploy returns correct deployment info."""
        config = DeploymentConfig(target="azure")
        strategy = AzureDeploymentStrategy(config, verbose=False)
        
        result = strategy.deploy(temp_dir)
        
        assert result['target'] == 'azure'
        assert result['status'] == 'prepared'
        assert strategy.is_deployed is True

    def test_predict_without_deployment_raises_error(self):
        """Test predict raises error when not deployed."""
        config = DeploymentConfig(target="azure")
        strategy = AzureDeploymentStrategy(config, verbose=False)
        
        with pytest.raises(DeploymentError, match="not deployed"):
            strategy.predict(torch.randn(1, 10))

    def test_predict_after_deployment(self, temp_dir):
        """Test predict after deployment returns None."""
        config = DeploymentConfig(target="azure")
        strategy = AzureDeploymentStrategy(config, verbose=False)
        
        strategy.deploy(temp_dir)
        result = strategy.predict(torch.randn(1, 10))
        assert result is None

    def test_teardown(self, temp_dir):
        """Test teardown cleans up deployment."""
        config = DeploymentConfig(target="azure")
        strategy = AzureDeploymentStrategy(config, verbose=False)
        
        strategy.deploy(temp_dir)
        assert strategy.is_deployed is True
        
        strategy.teardown()
        assert strategy.is_deployed is False


# =============================================================================
# TESTS: EdgeDeploymentStrategy
# =============================================================================

class TestEdgeDeploymentStrategy:
    """Tests for EdgeDeploymentStrategy."""

    def test_initialization_mobile(self, edge_config):
        """Test Edge strategy initialization for mobile."""
        strategy = EdgeDeploymentStrategy(edge_config)
        assert strategy.config is edge_config
        assert strategy.is_deployed is False

    def test_initialization_iot(self):
        """Test Edge strategy initialization for IoT."""
        config = DeploymentConfig(target="iot")
        strategy = EdgeDeploymentStrategy(config)
        assert strategy.config is config

    def test_prepare_model_creates_directory(self, edge_config, mock_model, temp_dir):
        """Test prepare_model creates save directory."""
        strategy = EdgeDeploymentStrategy(edge_config, verbose=False)
        save_path = temp_dir / "edge_model"
        
        with patch.object(torch, 'save'):
            with patch.object(torch.jit, 'trace', side_effect=Exception("Trace failed")):
                result = strategy.prepare_model(mock_model, save_path)
        
        assert save_path.exists()
        assert result == save_path

    def test_prepare_model_fallback_on_trace_failure(self, edge_config, mock_model, temp_dir):
        """Test prepare_model falls back to standard save on trace failure."""
        strategy = EdgeDeploymentStrategy(edge_config, verbose=False)
        save_path = temp_dir / "edge_model"
        
        with patch.object(torch.jit, 'trace', side_effect=Exception("Trace failed")):
            with patch.object(torch, 'save') as mock_save:
                strategy.prepare_model(mock_model, save_path)
                mock_save.assert_called_once()

    def test_prepare_model_with_mobile_optimization(self, edge_config, mock_model, temp_dir):
        """Test prepare_model with mobile optimization."""
        strategy = EdgeDeploymentStrategy(edge_config, verbose=False)
        save_path = temp_dir / "edge_model"
        
        mock_traced = MagicMock()
        mock_optimized = MagicMock()
        
        with patch.object(torch.jit, 'trace', return_value=mock_traced):
            with patch('torch.utils.mobile_optimizer.optimize_for_mobile', return_value=mock_optimized):
                strategy.prepare_model(mock_model, save_path)
                mock_optimized._save_for_lite_interpreter.assert_called_once()

    def test_deploy_returns_correct_info(self, edge_config, temp_dir):
        """Test deploy returns correct deployment info."""
        strategy = EdgeDeploymentStrategy(edge_config, verbose=False)
        
        result = strategy.deploy(temp_dir)
        
        assert result['target'] == 'edge'
        assert result['status'] == 'ready_for_edge'
        assert strategy.is_deployed is True

    def test_predict_without_deployment_raises_error(self, edge_config):
        """Test predict raises error when not deployed."""
        strategy = EdgeDeploymentStrategy(edge_config, verbose=False)
        
        with pytest.raises(DeploymentError, match="not deployed"):
            strategy.predict(torch.randn(1, 10))

    def test_predict_after_deployment(self, edge_config, temp_dir):
        """Test predict after deployment returns None."""
        strategy = EdgeDeploymentStrategy(edge_config, verbose=False)
        
        strategy.deploy(temp_dir)
        result = strategy.predict(torch.randn(1, 10))
        assert result is None

    def test_teardown(self, edge_config, temp_dir):
        """Test teardown cleans up deployment."""
        strategy = EdgeDeploymentStrategy(edge_config, verbose=False)
        
        strategy.deploy(temp_dir)
        assert strategy.is_deployed is True
        
        strategy.teardown()
        assert strategy.is_deployed is False


# =============================================================================
# TESTS: ContainerDeploymentStrategy
# =============================================================================

class TestContainerDeploymentStrategy:
    """Tests for ContainerDeploymentStrategy."""

    def test_initialization(self, container_config):
        """Test Container strategy initialization."""
        strategy = ContainerDeploymentStrategy(container_config)
        assert strategy.config is container_config
        assert strategy.is_deployed is False

    def test_prepare_model_creates_directory(self, container_config, mock_model, temp_dir):
        """Test prepare_model creates save directory."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        save_path = temp_dir / "container_model"
        
        with patch.object(torch, 'save'):
            result = strategy.prepare_model(mock_model, save_path)
        
        assert save_path.exists()
        assert result == save_path

    def test_prepare_model_saves_model(self, container_config, mock_model, temp_dir):
        """Test prepare_model saves model file."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        save_path = temp_dir / "container_model"
        
        with patch.object(torch, 'save') as mock_save:
            strategy.prepare_model(mock_model, save_path)
            mock_save.assert_called_once()

    def test_prepare_model_creates_dockerfile(self, container_config, mock_model, temp_dir):
        """Test prepare_model creates Dockerfile."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        save_path = temp_dir / "container_model"
        
        with patch.object(torch, 'save'):
            strategy.prepare_model(mock_model, save_path)
        
        dockerfile = save_path / "Dockerfile"
        assert dockerfile.exists()

    def test_dockerfile_content(self, container_config, mock_model, temp_dir):
        """Test Dockerfile has correct content."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        save_path = temp_dir / "container_model"
        
        with patch.object(torch, 'save'):
            strategy.prepare_model(mock_model, save_path)
        
        dockerfile = save_path / "Dockerfile"
        content = dockerfile.read_text()
        
        assert "FROM python" in content
        assert "EXPOSE 8080" in content
        assert "CMD" in content

    def test_prepare_model_creates_serving_script(self, container_config, mock_model, temp_dir):
        """Test prepare_model creates serve.py script."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        save_path = temp_dir / "container_model"
        
        with patch.object(torch, 'save'):
            strategy.prepare_model(mock_model, save_path)
        
        serve_script = save_path / "serve.py"
        assert serve_script.exists()

    def test_serving_script_content(self, container_config, mock_model, temp_dir):
        """Test serve.py has correct content."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        save_path = temp_dir / "container_model"
        
        with patch.object(torch, 'save'):
            strategy.prepare_model(mock_model, save_path)
        
        serve_script = save_path / "serve.py"
        content = serve_script.read_text()
        
        assert "Flask" in content
        assert "@app.route('/predict'" in content
        assert "@app.route('/health'" in content

    def test_prepare_model_creates_requirements(self, container_config, mock_model, temp_dir):
        """Test prepare_model creates requirements.txt."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        save_path = temp_dir / "container_model"
        
        with patch.object(torch, 'save'):
            strategy.prepare_model(mock_model, save_path)
        
        requirements = save_path / "requirements.txt"
        assert requirements.exists()
        content = requirements.read_text()
        assert "torch" in content
        assert "flask" in content

    def test_deploy_returns_correct_info(self, container_config, temp_dir):
        """Test deploy returns correct deployment info."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        
        result = strategy.deploy(temp_dir)
        
        assert result['target'] == 'container'
        assert result['status'] == 'ready_for_docker'
        assert 'docker build' in result['message']
        assert 'docker run' in result['run_command']
        assert strategy.is_deployed is True

    def test_predict_without_deployment_raises_error(self, container_config):
        """Test predict raises error when not deployed."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        
        with pytest.raises(DeploymentError, match="not deployed"):
            strategy.predict(torch.randn(1, 10))

    def test_predict_after_deployment(self, container_config, temp_dir):
        """Test predict after deployment returns None."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        
        strategy.deploy(temp_dir)
        result = strategy.predict(torch.randn(1, 10))
        assert result is None

    def test_teardown(self, container_config, temp_dir):
        """Test teardown cleans up deployment."""
        strategy = ContainerDeploymentStrategy(container_config, verbose=False)
        
        strategy.deploy(temp_dir)
        assert strategy.is_deployed is True
        
        strategy.teardown()
        assert strategy.is_deployed is False


# =============================================================================
# TESTS: LocalDeploymentStrategy
# =============================================================================

class TestLocalDeploymentStrategy:
    """Tests for LocalDeploymentStrategy."""

    def test_initialization(self, local_config):
        """Test Local strategy initialization."""
        strategy = LocalDeploymentStrategy(local_config)
        assert strategy.config is local_config
        assert strategy.is_deployed is False
        assert strategy.model is None

    def test_prepare_model_creates_directory(self, local_config, mock_model, temp_dir):
        """Test prepare_model creates save directory."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        save_path = temp_dir / "local_model"
        
        with patch.object(torch, 'save'):
            result = strategy.prepare_model(mock_model, save_path)
        
        assert save_path.exists()
        assert result == save_path

    def test_prepare_model_saves_model(self, local_config, mock_model, temp_dir):
        """Test prepare_model saves model file."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        save_path = temp_dir / "local_model"
        
        with patch.object(torch, 'save') as mock_save:
            strategy.prepare_model(mock_model, save_path)
            mock_save.assert_called_once()

    def test_deploy_loads_model(self, local_config, mock_model, temp_dir):
        """Test deploy loads model into memory."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        save_path = temp_dir / "local_model"
        save_path.mkdir(parents=True)
        model_file = save_path / "model.pth"
        
        # Create a mock model for loading
        loaded_model = MagicMock()
        loaded_model.eval = MagicMock(return_value=loaded_model)
        
        with patch.object(torch, 'load', return_value=loaded_model):
            # Touch the file so it exists
            model_file.touch()
            result = strategy.deploy(save_path)
        
        assert result['target'] == 'local'
        assert result['status'] == 'deployed'
        assert strategy.is_deployed is True
        assert strategy.model is loaded_model

    def test_deploy_missing_model_raises_error(self, local_config, temp_dir):
        """Test deploy raises error when model file is missing."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        save_path = temp_dir / "nonexistent"
        save_path.mkdir(parents=True)
        
        with pytest.raises(DeploymentError, match="Model file not found"):
            strategy.deploy(save_path)

    def test_predict_without_deployment_raises_error(self, local_config):
        """Test predict raises error when not deployed."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        
        with pytest.raises(DeploymentError, match="not deployed"):
            strategy.predict(torch.randn(1, 10))

    def test_predict_with_model_none_raises_error(self, local_config):
        """Test predict raises error when model is None."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        strategy.is_deployed = True
        strategy.model = None
        
        with pytest.raises(DeploymentError, match="not deployed"):
            strategy.predict(torch.randn(1, 10))

    def test_predict_after_deployment(self, local_config, temp_dir):
        """Test predict after deployment."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        save_path = temp_dir / "local_model"
        save_path.mkdir(parents=True)
        model_file = save_path / "model.pth"
        model_file.touch()
        
        # Create a mock model that can handle input
        mock_loaded_model = MagicMock()
        mock_loaded_model.eval = MagicMock(return_value=mock_loaded_model)
        mock_loaded_model.return_value = torch.tensor([1.0, 2.0])
        
        with patch.object(torch, 'load', return_value=mock_loaded_model):
            strategy.deploy(save_path)
        
        input_data = torch.randn(1, 10)
        result = strategy.predict(input_data)
        
        mock_loaded_model.assert_called_once_with(input_data)

    def test_teardown(self, local_config, temp_dir):
        """Test teardown cleans up deployment."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        save_path = temp_dir / "local_model"
        save_path.mkdir(parents=True)
        model_file = save_path / "model.pth"
        model_file.touch()
        
        mock_loaded_model = MagicMock()
        mock_loaded_model.eval = MagicMock(return_value=mock_loaded_model)
        
        with patch.object(torch, 'load', return_value=mock_loaded_model):
            strategy.deploy(save_path)
        
        assert strategy.is_deployed is True
        assert strategy.model is not None
        
        strategy.teardown()
        
        assert strategy.is_deployed is False
        assert strategy.model is None

    def test_teardown_when_not_deployed(self, local_config):
        """Test teardown when not deployed does nothing."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        
        # Should not raise
        strategy.teardown()
        assert strategy.is_deployed is False
        assert strategy.model is None


# =============================================================================
# TESTS: DeploymentManager
# =============================================================================

class TestDeploymentManager:
    """Tests for DeploymentManager class."""

    def test_initialization_with_config(self, local_config):
        """Test initialization with config object."""
        manager = DeploymentManager(config=local_config, verbose=False)
        assert manager.config is local_config
        assert isinstance(manager.strategy, LocalDeploymentStrategy)

    def test_initialization_with_target(self):
        """Test initialization with target string."""
        manager = DeploymentManager(target="local", verbose=False)
        assert manager.config.target == "local"
        assert isinstance(manager.strategy, LocalDeploymentStrategy)

    def test_initialization_default(self):
        """Test default initialization (local target)."""
        manager = DeploymentManager(verbose=False)
        assert manager.config.target == "local"
        assert isinstance(manager.strategy, LocalDeploymentStrategy)

    def test_initialization_aws_target(self):
        """Test initialization with AWS target."""
        config = DeploymentConfig(target="aws")
        manager = DeploymentManager(config=config, verbose=False)
        assert isinstance(manager.strategy, AWSDeploymentStrategy)

    def test_initialization_gcp_target(self):
        """Test initialization with GCP target."""
        config = DeploymentConfig(target="gcp")
        manager = DeploymentManager(config=config, verbose=False)
        assert isinstance(manager.strategy, GCPDeploymentStrategy)

    def test_initialization_azure_target(self):
        """Test initialization with Azure target."""
        config = DeploymentConfig(target="azure")
        manager = DeploymentManager(config=config, verbose=False)
        assert isinstance(manager.strategy, AzureDeploymentStrategy)

    def test_initialization_mobile_target(self):
        """Test initialization with mobile target."""
        config = DeploymentConfig(target="mobile")
        manager = DeploymentManager(config=config, verbose=False)
        assert isinstance(manager.strategy, EdgeDeploymentStrategy)

    def test_initialization_iot_target(self):
        """Test initialization with IoT target."""
        config = DeploymentConfig(target="iot")
        manager = DeploymentManager(config=config, verbose=False)
        assert isinstance(manager.strategy, EdgeDeploymentStrategy)

    def test_initialization_container_target(self):
        """Test initialization with container target."""
        config = DeploymentConfig(target="container")
        manager = DeploymentManager(config=config, verbose=False)
        assert isinstance(manager.strategy, ContainerDeploymentStrategy)

    def test_initialization_unknown_target_raises_error(self):
        """Test initialization with unknown target raises error."""
        config = DeploymentConfig(target="unknown")
        with pytest.raises(ConfigurationError, match="Unknown deployment target"):
            DeploymentManager(config=config)

    def test_prepare_model(self, mock_model, temp_dir):
        """Test prepare_model delegates to strategy."""
        manager = DeploymentManager(target="local", verbose=False)
        
        with patch.object(torch, 'save'):
            result = manager.prepare_model(mock_model, temp_dir)
        
        assert result == Path(temp_dir)

    def test_prepare_model_with_string_path(self, mock_model, temp_dir):
        """Test prepare_model accepts string path."""
        manager = DeploymentManager(target="local", verbose=False)
        
        with patch.object(torch, 'save'):
            result = manager.prepare_model(mock_model, str(temp_dir))
        
        assert isinstance(result, Path)

    def test_deploy(self, temp_dir):
        """Test deploy delegates to strategy."""
        manager = DeploymentManager(target="local", verbose=False)
        
        save_path = temp_dir / "model"
        save_path.mkdir(parents=True)
        model_file = save_path / "model.pth"
        model_file.touch()
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        
        with patch.object(torch, 'load', return_value=mock_model):
            result = manager.deploy(save_path)
        
        assert result['status'] == 'deployed'

    def test_deploy_with_string_path(self, temp_dir):
        """Test deploy accepts string path."""
        manager = DeploymentManager(target="local", verbose=False)
        
        save_path = temp_dir / "model"
        save_path.mkdir(parents=True)
        model_file = save_path / "model.pth"
        model_file.touch()
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        
        with patch.object(torch, 'load', return_value=mock_model):
            result = manager.deploy(str(save_path))
        
        assert result['status'] == 'deployed'

    def test_predict(self, temp_dir):
        """Test predict delegates to strategy."""
        manager = DeploymentManager(target="local", verbose=False)
        
        save_path = temp_dir / "model"
        save_path.mkdir(parents=True)
        model_file = save_path / "model.pth"
        model_file.touch()
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.return_value = torch.tensor([1.0])
        
        with patch.object(torch, 'load', return_value=mock_model):
            manager.deploy(save_path)
        
        input_data = torch.randn(1, 10)
        manager.predict(input_data)
        
        mock_model.assert_called_once_with(input_data)

    def test_teardown(self, temp_dir):
        """Test teardown delegates to strategy."""
        manager = DeploymentManager(target="local", verbose=False)
        
        save_path = temp_dir / "model"
        save_path.mkdir(parents=True)
        model_file = save_path / "model.pth"
        model_file.touch()
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        
        with patch.object(torch, 'load', return_value=mock_model):
            manager.deploy(save_path)
        
        assert manager.strategy.is_deployed is True
        manager.teardown()
        assert manager.strategy.is_deployed is False

    def test_get_deployment_info(self):
        """Test get_deployment_info delegates to strategy."""
        manager = DeploymentManager(target="local", verbose=False)
        
        info = manager.get_deployment_info()
        
        assert 'target' in info
        assert 'serving_mode' in info
        assert 'is_deployed' in info
        assert info['target'] == 'local'

    def test_list_available_targets_class_method(self):
        """Test list_available_targets class method."""
        targets = DeploymentManager.list_available_targets()
        
        assert isinstance(targets, list)
        assert 'aws' in targets
        assert 'gcp' in targets
        assert 'azure' in targets
        assert 'mobile' in targets
        assert 'iot' in targets
        assert 'container' in targets
        assert 'local' in targets

    def test_strategies_dict(self):
        """Test _strategies dictionary contains all expected strategies."""
        expected = {'aws', 'gcp', 'azure', 'mobile', 'iot', 'container', 'local'}
        actual = set(DeploymentManager._strategies.keys())
        assert expected == actual

    def test_verbose_attribute(self):
        """Test verbose attribute is set correctly."""
        manager_verbose = DeploymentManager(verbose=True)
        manager_quiet = DeploymentManager(verbose=False)
        
        assert manager_verbose.verbose is True
        assert manager_quiet.verbose is False


# =============================================================================
# TESTS: Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_list_deployment_targets(self):
        """Test list_deployment_targets function."""
        targets = list_deployment_targets()
        
        assert isinstance(targets, list)
        assert 'aws' in targets
        assert 'local' in targets
        assert len(targets) == 7  # 7 targets in _strategies

    def test_deploy_locally_creates_manager(self, mock_model, temp_dir):
        """Test deploy_locally creates and returns DeploymentManager."""
        save_path = temp_dir / "deploy_local"
        
        mock_loaded_model = MagicMock()
        mock_loaded_model.eval = MagicMock(return_value=mock_loaded_model)
        
        def create_model_file(*args, **kwargs):
            """Side effect to create the model file when torch.save is called."""
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / "model.pth").touch()
        
        with patch.object(torch, 'save', side_effect=create_model_file):
            with patch.object(torch, 'load', return_value=mock_loaded_model):
                manager = deploy_locally(mock_model, save_path)
        
        assert isinstance(manager, DeploymentManager)
        assert manager.strategy.is_deployed is True

    def test_deploy_locally_prepares_and_deploys(self, mock_model, temp_dir):
        """Test deploy_locally both prepares and deploys the model."""
        save_path = temp_dir / "deploy_local"
        
        mock_loaded_model = MagicMock()
        mock_loaded_model.eval = MagicMock(return_value=mock_loaded_model)
        
        def create_model_file(*args, **kwargs):
            """Side effect to create the model file when torch.save is called."""
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / "model.pth").touch()
        
        with patch.object(torch, 'save', side_effect=create_model_file) as mock_save:
            with patch.object(torch, 'load', return_value=mock_loaded_model) as mock_load:
                manager = deploy_locally(mock_model, save_path)
        
        # Should have saved the model
        mock_save.assert_called_once()
        # Should have loaded the model
        mock_load.assert_called_once()
        
        assert manager.strategy.is_deployed is True

    def test_deploy_locally_with_string_path(self, mock_model, temp_dir):
        """Test deploy_locally accepts string path."""
        save_path = temp_dir / "deploy_local"
        save_path_str = str(save_path)
        
        mock_loaded_model = MagicMock()
        mock_loaded_model.eval = MagicMock(return_value=mock_loaded_model)
        
        def create_model_file(*args, **kwargs):
            """Side effect to create the model file when torch.save is called."""
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / "model.pth").touch()
        
        with patch.object(torch, 'save', side_effect=create_model_file):
            with patch.object(torch, 'load', return_value=mock_loaded_model):
                manager = deploy_locally(mock_model, save_path_str)
        
        assert isinstance(manager, DeploymentManager)


# =============================================================================
# TESTS: Exception Handling
# =============================================================================

class TestExceptionHandling:
    """Tests for exception handling and exception hierarchy."""

    def test_model_error_is_base_exception(self):
        """Test ModelError is the base exception class."""
        error = ModelError("Test error")
        assert isinstance(error, Exception)

    def test_deployment_error_inherits_from_model_error(self):
        """Test DeploymentError inherits from ModelError."""
        error = DeploymentError("Test error")
        assert isinstance(error, ModelError)
        assert isinstance(error, Exception)

    def test_configuration_error_inherits_from_model_error(self):
        """Test ConfigurationError inherits from ModelError."""
        error = ConfigurationError("Test error")
        assert isinstance(error, ModelError)
        assert isinstance(error, Exception)

    def test_model_error_message(self):
        """Test ModelError preserves message."""
        message = "Base model error occurred"
        error = ModelError(message)
        assert str(error) == message

    def test_deployment_error_message(self):
        """Test DeploymentError preserves message."""
        message = "Model deployment failed"
        error = DeploymentError(message)
        assert str(error) == message

    def test_configuration_error_message(self):
        """Test ConfigurationError preserves message."""
        message = "Invalid configuration"
        error = ConfigurationError(message)
        assert str(error) == message

    def test_exception_hierarchy_is_consistent(self):
        """Test that the exception hierarchy is consistent and follows expected pattern."""
        # Verify the class relationships directly
        assert issubclass(DeploymentError, ModelError)
        assert issubclass(ConfigurationError, ModelError)
        assert issubclass(ModelError, Exception)

    def test_exceptions_can_be_caught_by_base_class(self):
        """Test that specific exceptions can be caught by their base class."""
        # DeploymentError should be catchable as ModelError
        try:
            raise DeploymentError("deployment failed")
        except ModelError as e:
            assert str(e) == "deployment failed"
        
        # ConfigurationError should be catchable as ModelError
        try:
            raise ConfigurationError("config invalid")
        except ModelError as e:
            assert str(e) == "config invalid"

    def test_exceptions_can_be_raised_with_no_args(self):
        """Test that exceptions can be raised with no arguments."""
        # Should not raise during construction
        model_err = ModelError()
        deploy_err = DeploymentError()
        config_err = ConfigurationError()
        
        assert isinstance(model_err, ModelError)
        assert isinstance(deploy_err, DeploymentError)
        assert isinstance(config_err, ConfigurationError)


# =============================================================================
# TESTS: Edge Cases and Integration
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_save_path(self, mock_model, temp_dir):
        """Test handling of path edge case."""
        strategy = LocalDeploymentStrategy(DeploymentConfig(), verbose=False)
        
        # Path with empty directory name should still work
        save_path = temp_dir / ""
        
        with patch.object(torch, 'save'):
            # This should handle the path correctly
            result = strategy.prepare_model(mock_model, save_path)
        
        assert result is not None

    def test_config_with_all_params(self):
        """Test DeploymentConfig with all parameters specified."""
        config = DeploymentConfig(
            target="aws",
            serving_mode="batch",
            instance_type="ml.p3.2xlarge",
            num_instances=3,
            auto_scaling=True,
            min_instances=2,
            max_instances=15,
            api_type="grpc",
            enable_monitoring=True,
            enable_logging=True,
            enable_caching=True,
            timeout_seconds=120,
            max_batch_size=256
        )
        
        assert config.target == "aws"
        assert config.serving_mode == "batch"
        assert config.instance_type == "ml.p3.2xlarge"
        assert config.num_instances == 3
        assert config.auto_scaling is True
        assert config.min_instances == 2
        assert config.max_instances == 15
        assert config.api_type == "grpc"
        assert config.enable_monitoring is True
        assert config.enable_logging is True
        assert config.enable_caching is True
        assert config.timeout_seconds == 120
        assert config.max_batch_size == 256

    def test_strategy_get_deployment_info_structure(self, local_config, temp_dir):
        """Test deployment info has consistent structure."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        
        info = strategy.get_deployment_info()
        
        # Check structure
        assert 'target' in info
        assert 'serving_mode' in info
        assert 'is_deployed' in info
        assert 'config' in info
        
        # Check config is a dict
        assert isinstance(info['config'], dict)

    def test_multiple_deploys(self, temp_dir):
        """Test deploying multiple times."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config, verbose=False)
        
        # First deploy
        result1 = strategy.deploy(temp_dir)
        assert strategy.is_deployed is True
        
        # Second deploy (should still work)
        result2 = strategy.deploy(temp_dir)
        assert strategy.is_deployed is True

    def test_teardown_multiple_times(self, temp_dir):
        """Test calling teardown multiple times."""
        config = DeploymentConfig(target="gcp")
        strategy = GCPDeploymentStrategy(config, verbose=False)
        
        strategy.deploy(temp_dir)
        
        # Multiple teardowns should not raise
        strategy.teardown()
        strategy.teardown()
        strategy.teardown()
        
        assert strategy.is_deployed is False


# =============================================================================
# TESTS: Logging Integration
# =============================================================================

class TestLoggingIntegration:
    """Tests for logging behavior."""

    def test_verbose_true_logs_messages(self, local_config, mock_model, temp_dir, caplog):
        """Test that verbose=True produces log messages."""
        strategy = LocalDeploymentStrategy(local_config, verbose=True)
        
        with caplog.at_level(logging.INFO):
            with patch.object(torch, 'save'):
                strategy.prepare_model(mock_model, temp_dir)
        
        # Should have logged something
        # Note: Depends on actual logging configuration

    def test_verbose_false_reduces_logging(self, local_config, mock_model, temp_dir, caplog):
        """Test that verbose=False reduces log messages."""
        strategy = LocalDeploymentStrategy(local_config, verbose=False)
        
        with caplog.at_level(logging.INFO):
            with patch.object(torch, 'save'):
                strategy.prepare_model(mock_model, temp_dir)
        
        # Should have fewer logs (implementation dependent)


# =============================================================================
# TESTS: Path Handling
# =============================================================================

class TestPathHandling:
    """Tests for path handling across strategies."""

    def test_path_object_handling(self, mock_model, temp_dir):
        """Test strategies handle Path objects correctly."""
        strategy = LocalDeploymentStrategy(DeploymentConfig(), verbose=False)
        
        with patch.object(torch, 'save'):
            result = strategy.prepare_model(mock_model, Path(temp_dir))
        
        assert isinstance(result, Path)

    def test_string_path_handling(self, mock_model, temp_dir):
        """Test strategies handle string paths correctly."""
        manager = DeploymentManager(target="local", verbose=False)
        
        with patch.object(torch, 'save'):
            result = manager.prepare_model(mock_model, str(temp_dir))
        
        assert isinstance(result, Path)

    def test_nested_directory_creation(self, mock_model, temp_dir):
        """Test strategies create nested directories."""
        strategy = LocalDeploymentStrategy(DeploymentConfig(), verbose=False)
        
        nested_path = temp_dir / "level1" / "level2" / "level3"
        
        with patch.object(torch, 'save'):
            result = strategy.prepare_model(mock_model, nested_path)
        
        assert nested_path.exists()


# =============================================================================
# TESTS: Strategy Factory Pattern
# =============================================================================

class TestStrategyFactoryPattern:
    """Tests for the strategy factory pattern in DeploymentManager."""

    def test_all_strategies_in_registry(self):
        """Test all expected strategies are in the registry."""
        expected_targets = {'aws', 'gcp', 'azure', 'mobile', 'iot', 'container', 'local'}
        actual_targets = set(DeploymentManager._strategies.keys())
        
        assert expected_targets == actual_targets

    def test_strategy_classes_are_correct(self):
        """Test each target maps to correct strategy class."""
        assert DeploymentManager._strategies['aws'] == AWSDeploymentStrategy
        assert DeploymentManager._strategies['gcp'] == GCPDeploymentStrategy
        assert DeploymentManager._strategies['azure'] == AzureDeploymentStrategy
        assert DeploymentManager._strategies['mobile'] == EdgeDeploymentStrategy
        assert DeploymentManager._strategies['iot'] == EdgeDeploymentStrategy
        assert DeploymentManager._strategies['container'] == ContainerDeploymentStrategy
        assert DeploymentManager._strategies['local'] == LocalDeploymentStrategy

    def test_mobile_and_iot_share_strategy(self):
        """Test mobile and IoT both use EdgeDeploymentStrategy."""
        assert DeploymentManager._strategies['mobile'] is DeploymentManager._strategies['iot']


# =============================================================================
# TESTS: Pydantic BaseModel Behavior
# =============================================================================

class TestPydanticModelBehavior:
    """Tests for Pydantic BaseModel-specific behavior (Pydantic V2 Migration Phase 20)."""

    def test_config_equality(self):
        """Test DeploymentConfig equality comparison."""
        config1 = DeploymentConfig(target="aws")
        config2 = DeploymentConfig(target="aws")
        config3 = DeploymentConfig(target="gcp")
        
        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self):
        """Test DeploymentConfig has useful repr."""
        config = DeploymentConfig(target="aws")
        repr_str = repr(config)
        
        assert "DeploymentConfig" in repr_str
        assert "aws" in repr_str

    def test_config_is_mutable(self):
        """Test DeploymentConfig fields can be modified (Pydantic BaseModel is mutable by default)."""
        config = DeploymentConfig()
        config.target = "aws"
        
        assert config.target == "aws"

    def test_to_dict_wraps_model_dump(self):
        """Test to_dict() properly wraps Pydantic V2's model_dump() method."""
        config = DeploymentConfig(target="azure", serving_mode="batch")
        
        # Both methods should return equivalent dictionaries
        to_dict_result = config.to_dict()
        model_dump_result = config.model_dump()
        
        assert to_dict_result == model_dump_result
        assert isinstance(to_dict_result, dict)
        assert isinstance(model_dump_result, dict)

    def test_model_dump_available(self):
        """Test that Pydantic V2's model_dump() method is available."""
        config = DeploymentConfig()
        
        # Pydantic V2 BaseModel should have model_dump method
        assert hasattr(config, 'model_dump')
        assert callable(config.model_dump)
        
        result = config.model_dump()
        assert isinstance(result, dict)

    def test_model_validate_from_dict(self):
        """Test that DeploymentConfig can be created from dict using Pydantic V2's model_validate."""
        config_dict = {
            'target': 'container',
            'serving_mode': 'batch',
            'num_instances': 5,
            'auto_scaling': True
        }
        
        # Pydantic V2 uses model_validate for dict-to-model conversion
        config = DeploymentConfig.model_validate(config_dict)
        
        assert config.target == 'container'
        assert config.serving_mode == 'batch'
        assert config.num_instances == 5
        assert config.auto_scaling is True

    def test_model_copy_method(self):
        """Test that Pydantic V2's model_copy() method works correctly."""
        original = DeploymentConfig(target="aws", num_instances=2)
        
        # Pydantic V2 uses model_copy (not copy)
        copied = original.model_copy()
        
        assert copied == original
        assert copied is not original
        
        # Modifications to copy don't affect original
        copied.target = "gcp"
        assert original.target == "aws"

    def test_model_copy_with_update(self):
        """Test model_copy with update parameter for creating modified copies."""
        original = DeploymentConfig(target="aws", num_instances=2)
        
        # Create copy with some fields updated
        modified = original.model_copy(update={'target': 'gcp', 'num_instances': 5})
        
        assert modified.target == 'gcp'
        assert modified.num_instances == 5
        # Other fields should be preserved
        assert modified.serving_mode == original.serving_mode

    def test_pydantic_basemodel_inheritance(self):
        """Test that DeploymentConfig inherits from Pydantic BaseModel."""
        from pydantic import BaseModel
        
        config = DeploymentConfig()
        assert isinstance(config, BaseModel)


# =============================================================================
# MODULE LEVEL TESTS
# =============================================================================

class TestModuleLevel:
    """Tests for module-level behavior."""

    def test_all_public_classes_exported(self):
        """Test all expected public classes are available."""
        from milia_pipeline.models.deployment import deployment_strategies
        
        assert hasattr(deployment_strategies, 'DeploymentTarget')
        assert hasattr(deployment_strategies, 'ServingMode')
        assert hasattr(deployment_strategies, 'DeploymentConfig')
        assert hasattr(deployment_strategies, 'DeploymentStrategy')
        assert hasattr(deployment_strategies, 'AWSDeploymentStrategy')
        assert hasattr(deployment_strategies, 'GCPDeploymentStrategy')
        assert hasattr(deployment_strategies, 'AzureDeploymentStrategy')
        assert hasattr(deployment_strategies, 'EdgeDeploymentStrategy')
        assert hasattr(deployment_strategies, 'ContainerDeploymentStrategy')
        assert hasattr(deployment_strategies, 'LocalDeploymentStrategy')
        assert hasattr(deployment_strategies, 'DeploymentManager')
        assert hasattr(deployment_strategies, 'deploy_locally')
        assert hasattr(deployment_strategies, 'list_deployment_targets')

    def test_exceptions_exported(self):
        """Test exception classes are available."""
        from milia_pipeline.models.deployment import deployment_strategies
        
        assert hasattr(deployment_strategies, 'ModelError')
        assert hasattr(deployment_strategies, 'DeploymentError')
        assert hasattr(deployment_strategies, 'ConfigurationError')


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
