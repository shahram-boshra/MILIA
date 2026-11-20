"""Custom transforms raise MILIA's central transform exceptions.

``custom_transforms`` imported ``TransformExecutionError`` from ``milia_pipeline.exceptions``,
which did not define it, so the import always failed and the module used local fallback classes
derived from ``Exception``: ``except BaseProjectError`` / ``except TransformError`` did not catch
transform failures, and errors carried no " (Transform: name)" context.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

import milia_pipeline.exceptions as central
import milia_pipeline.transformations as transformations
import milia_pipeline.transformations.custom_transforms as custom
from milia_pipeline.plugins.transformations.combinatorial_dual.transforms import QuotientGraph
from milia_pipeline.transformations.custom_transforms import (
    CustomTransformBase,
    DiscretizeTargets,
    NormalizeTargets,
    TransformMetadata,
)


@pytest.mark.parametrize(
    "name", ["TransformExecutionError", "TransformConfigurationError", "TransformValidationError"]
)
def test_custom_transforms_reexport_the_central_classes(name):
    assert getattr(custom, name) is getattr(central, name)
    assert getattr(transformations, name) is getattr(central, name)


def test_execution_and_precondition_errors_join_the_central_hierarchy():
    assert issubclass(central.TransformExecutionError, central.TransformError)
    assert issubclass(central.TransformExecutionError, central.BaseProjectError)
    assert issubclass(custom.TransformPreconditionError, central.TransformExecutionError)


def test_execution_error_keeps_the_original_error_and_names_the_transform():
    original = ValueError("boom")
    error = central.TransformExecutionError("failed", transform_name="T", original_error=original)
    assert error.original_error is original
    assert error.transform_name == "T"
    assert str(error) == "failed (Transform: T)"


class _Failing(CustomTransformBase):
    def transform(self, data: Data) -> Data:
        raise ValueError("Intentional failure")

    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name="Failing", version="1.0.0", author="Test", category="test", description="Test"
        )


def test_runtime_failure_is_caught_as_a_central_transform_error():
    with pytest.raises(central.TransformError) as error:
        _Failing()(Data(x=torch.ones(3, 5), num_nodes=3))
    assert isinstance(error.value, central.TransformExecutionError)
    assert isinstance(error.value.original_error, ValueError)
    assert str(error.value).endswith("(Transform: Failing)")


def test_plugin_precondition_failure_is_caught_as_a_central_transform_error():
    with pytest.raises(central.TransformExecutionError) as error:
        QuotientGraph()(
            Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]), num_nodes=3)
        )
    assert isinstance(error.value, custom.TransformPreconditionError)
    assert str(error.value).endswith("(Transform: QuotientGraph)")


@pytest.mark.parametrize(
    "build",
    [
        lambda: NormalizeTargets(range_min=1.0, range_max=0.0),
        lambda: DiscretizeTargets(n_bins=1),
    ],
    ids=["NormalizeTargets", "DiscretizeTargets"],
)
def test_configuration_failure_is_caught_as_a_central_configuration_error(build):
    with pytest.raises(central.ConfigurationError) as error:
        build()
    assert isinstance(error.value, central.TransformConfigurationError)
    assert isinstance(error.value, central.BaseProjectError)
