"""
Every MILIA exception is defined once, in ``milia_pipeline.exceptions`` (F61).

The acceleration and deployment modules used to import their exceptions inside
``try: ... except ImportError:`` and define look-alike classes in the ``except`` branch. Several
names (``DeploymentError``, ``HardwareError``, ``OptimizationError``, ...) did not exist in
``milia_pipeline.exceptions``, so every module silently used its own copy: an ``except``
clause written against one module's class did not catch the same-named error raised by another,
and none of them was a ``BaseProjectError``.
"""

import ast
from pathlib import Path

import pytest

import milia_pipeline.exceptions as exceptions
from milia_pipeline.models import acceleration, deployment
from milia_pipeline.models.acceleration import (
    computation_optimization,
    device_manager,
    distributed_strategies,
    memory_optimization,
)
from milia_pipeline.models.deployment import (
    DeploymentManager,
    deployment_strategies,
    model_optimization,
    monitoring,
)

PACKAGE_ROOT = Path(exceptions.__file__).parent

REEXPORTS = [
    (acceleration, "ModelError", "ModelError"),
    (acceleration, "HardwareError", "HardwareError"),
    (acceleration, "DeviceNotAvailableError", "DeviceNotAvailableError"),
    (acceleration, "VQMMemoryError", "AcceleratorMemoryError"),
    (acceleration, "OptimizationError", "OptimizationError"),
    (acceleration, "DistributedError", "DistributedError"),
    (computation_optimization, "HardwareError", "HardwareError"),
    (computation_optimization, "OptimizationError", "OptimizationError"),
    (device_manager, "DeviceNotAvailableError", "DeviceNotAvailableError"),
    (distributed_strategies, "DistributedError", "DistributedError"),
    (memory_optimization, "VQMMemoryError", "AcceleratorMemoryError"),
    (deployment, "DeploymentError", "DeploymentError"),
    (deployment, "OptimizationError", "OptimizationError"),
    (deployment, "ExportError", "ExportError"),
    (deployment, "MonitoringError", "MonitoringError"),
    (deployment, "AlertError", "AlertError"),
    (deployment, "ConfigurationError", "ConfigurationError"),
    (deployment_strategies, "DeploymentError", "DeploymentError"),
    (deployment_strategies, "ConfigurationError", "ConfigurationError"),
    (model_optimization, "OptimizationError", "OptimizationError"),
    (model_optimization, "ExportError", "ExportError"),
    (monitoring, "MonitoringError", "MonitoringError"),
    (monitoring, "AlertError", "AlertError"),
]


@pytest.mark.parametrize(
    ("module", "name", "canonical"),
    REEXPORTS,
    ids=[f"{m.__name__.rsplit('.', 1)[-1]}.{n}" for m, n, _ in REEXPORTS],
)
def test_reexported_exception_is_the_canonical_class(module, name, canonical):
    exported = getattr(module, name)
    assert exported is getattr(exceptions, canonical)
    assert issubclass(exported, exceptions.BaseProjectError)


def test_hierarchy_matches_the_documented_contracts():
    assert issubclass(exceptions.HardwareError, exceptions.ModelError)
    for child in (
        exceptions.DeviceNotAvailableError,
        exceptions.AcceleratorMemoryError,
        exceptions.OptimizationError,
        exceptions.DistributedError,
    ):
        assert issubclass(child, exceptions.HardwareError)
    for child in (exceptions.DeploymentError, exceptions.ExportError, exceptions.MonitoringError):
        assert issubclass(child, exceptions.ModelError)
    assert issubclass(exceptions.AlertError, exceptions.MonitoringError)
    assert str(exceptions.DeploymentError("plain message")) == "plain message"


def test_an_error_raised_by_a_strategy_is_caught_through_the_package_export(tmp_path):
    manager = DeploymentManager(target="local", verbose=False)
    with pytest.raises(deployment.DeploymentError):
        manager.deploy(tmp_path)  # no model.pth
    with pytest.raises(exceptions.ConfigurationError):
        DeploymentManager(target="no-such-target", verbose=False)


def _missing_exception_imports(path: Path) -> list[str]:
    """Names a module imports from milia_pipeline.exceptions that the module does not define."""
    return [
        alias.name
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.ImportFrom) and node.module == "milia_pipeline.exceptions"
        for alias in node.names
        if alias.name != "*" and not hasattr(exceptions, alias.name)
    ]


def test_every_name_imported_from_milia_exceptions_exists():
    """A missing name used to fall through to an ``except ImportError`` look-alike class."""
    offenders = {
        str(path.relative_to(PACKAGE_ROOT)): names
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if (names := _missing_exception_imports(path))
    }
    assert offenders == {}, f"Not defined in milia_pipeline.exceptions: {offenders}"
