"""TorchScript in MILIA's Python API is deprecated (removed in 2.0.0).

PyTorch deprecated ``torch.jit`` (``torch.jit.script``/``trace``/``freeze`` warn since 2.10) in
favour of ``torch.compile`` and ``torch.export``. No configuration or CLI option uses MILIA's
TorchScript API, so it warns with ``DeprecationWarning`` at the caller (PEP 565).
"""

import warnings
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from milia_pipeline.models.acceleration.computation_optimization import (
    ComputationOptimizer,
    auto_optimize_model,
    get_optimal_settings,
)
from milia_pipeline.models.deployment import optimize_model_for_deployment
from milia_pipeline.models.deployment.model_optimization import ModelOptimizer

CPU = torch.device("cpu")


def _milia_deprecations(record) -> list[warnings.WarningMessage]:
    return [
        w
        for w in record
        if issubclass(w.category, DeprecationWarning) and "deprecated in MILIA" in str(w.message)
    ]


class TestComputationOptimizer:
    def test_jit_compile_warns_at_the_caller(self):
        with pytest.warns(DeprecationWarning, match=r"jit_compile=True\) uses TorchScript") as rec:
            ComputationOptimizer(jit_compile=True, verbose=False, device=CPU)
        assert [w.filename for w in _milia_deprecations(rec)] == [__file__]

    def test_default_does_not_warn(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ComputationOptimizer(verbose=False, device=CPU)
        assert _milia_deprecations(rec) == []

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("jit_script_model", (nn.Linear(2, 2),)),
            ("jit_freeze_model", (nn.Linear(2, 2),)),
            ("optimize_graph", (nn.Linear(2, 2),)),
            ("enable_fusion", ()),
            ("disable_fusion", ()),
        ],
    )
    def test_public_torchscript_methods_warn(self, method, args):
        optimizer = ComputationOptimizer(verbose=False, device=CPU)
        with (
            patch.object(torch._C, "_jit_set_texpr_fuser_enabled"),
            patch.object(torch._C, "_jit_set_nvfuser_enabled"),
            patch.object(torch._C, "_jit_set_profiling_executor"),
            pytest.warns(
                DeprecationWarning, match=rf"ComputationOptimizer\.{method} uses TorchScript"
            ) as rec,
        ):
            getattr(optimizer, method)(*args)
        assert [w.filename for w in _milia_deprecations(rec)] == [__file__]

    def test_optimize_model_warns_only_at_construction(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            optimizer = ComputationOptimizer(jit_compile=True, verbose=False, device=CPU)
            optimizer.optimize_model(nn.Linear(2, 2))
        assert len(_milia_deprecations(rec)) == 1


class TestInferencePreset:
    def test_inference_preset_no_longer_enables_torchscript(self):
        assert get_optimal_settings(CPU, task_type="inference")["jit_compile"] is False

    def test_auto_optimize_model_inference_does_not_warn(self):
        with (
            patch.object(ComputationOptimizer, "compile_model", side_effect=lambda m, *a, **k: m),
            warnings.catch_warnings(record=True) as rec,
        ):
            warnings.simplefilter("always")
            auto_optimize_model(nn.Linear(2, 2), task_type="inference", device=CPU)
        assert _milia_deprecations(rec) == []


class TestMobile:
    def test_model_optimizer_flag_warns(self):
        with pytest.warns(DeprecationWarning, match=r"optimize_for_mobile=True\) uses TorchScript"):
            ModelOptimizer(optimize_for_mobile=True, verbose=False)

    def test_optimize_for_mobile_warns(self, tmp_path):
        optimizer = ModelOptimizer(verbose=False)  # flag off: the method returns at once
        with pytest.warns(DeprecationWarning, match=r"optimize_for_mobile uses TorchScript"):
            optimizer.optimize_for_mobile(nn.Linear(2, 2), (torch.randn(1, 2),), tmp_path / "m.ptl")
        assert not (tmp_path / "m.ptl").exists()

    def test_optimize_model_for_deployment_flag_warns_once(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model = nn.Linear(2, 2)
            result = optimize_model_for_deployment(model, optimize_for_mobile=True, verbose=False)
        assert result is model
        milia = _milia_deprecations(rec)
        assert len(milia) == 1
        assert "optimize_model_for_deployment(optimize_for_mobile=True)" in str(milia[0].message)
        assert milia[0].filename == __file__
